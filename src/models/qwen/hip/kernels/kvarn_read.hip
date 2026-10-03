// KVarN decode read path (openspec add-kvarn-kv-cache-quantization, design D2,
// D5, D7, D8). One query reads the sink, the sealed tiles and the F16 tail ring
// through a single online softmax, choosing per key position whether the tile
// is quantized (e = 128 + 128t + 127 <= q - L - 1) or still F16. The
// dequantized tensor is never materialized: a lane decodes only the channels it
// consumes.
//
// The arithmetic is the folded form of design D5/D7. K folds the channel
// normalization scale and the RTN scale into the query once per tile, and the
// RTN zero point becomes the tile's constant term, so a key costs one
// multiply-add per channel:
//
//   partial = zero + sum_c (q[c] * channel_scale[c] * rtn_scale[c]) * code[c]
//   score   = lane_reduce(partial) * token_scale * 1/sqrt(head_dim)
//
// V accumulates the tile in the normalized space: the token scale and the RTN
// pair enter through the softmax weights, and the channel scale is applied once
// when the tile is merged into the output:
//
//   code_sum[v][k] += (weight * rtn_scale[token][v]) * code[token][c]
//   zero_sum[v]    += weight * rtn_zero[token][v]
//   out[v][k]      += channel_scale[c] * (code_sum[v][k] + zero_sum[v])
//
// A lane's vector `v` covers exactly the 128-channel group `v` (design D5), so
// a V zero-point term is one scalar per group per key.
//
// Every product and sum of these chains is an explicit `fmaf`, and a block of
// `kKeyBlock` keys shares one online-softmax maximum and one rescale of the
// accumulators (tasks 3.15(b)/(c)). The online, split-K and verification-row
// paths run the same helpers in the same order, so they stay bit-identical
// without disabling FP contraction for the whole translation unit: a file-wide
// `#pragma clang fp contract(off)` also split every F16 and Hadamard
// multiply-add of this file into two instructions.
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/kernels/kvarn_device.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/kvarn_reference.hpp"

namespace gufo::hip {
namespace {

constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kLanes = 32;
constexpr std::uint32_t kVectorWidth = kHeadDim / 4;
constexpr std::uint32_t kVectorsPerLane = kVectorWidth / kLanes;
constexpr std::uint32_t kTileTokens = 128;
constexpr std::uint32_t kKeyBlock = 8;
constexpr std::size_t kLowOffset = models::qwen::kKvarnPayloadMetadataBytes;
constexpr std::size_t kMidOffset =
    kLowOffset + models::qwen::kKvarnLowPlaneBytes;
constexpr std::size_t kHighOffset =
    kMidOffset + models::qwen::kKvarnBitPlaneBytes;

static_assert(kTileTokens == models::qwen::kKvarnTileTokens);
static_assert(kTileTokens == models::qwen::kKvarnSinkRows);
static_assert(kVectorsPerLane == 2);
static_assert(static_cast<std::uint32_t>(models::qwen::kKvarnValueGroups) ==
              kVectorsPerLane);
static_assert(static_cast<std::uint32_t>(models::qwen::kKvarnValueGroups) *
                  models::qwen::kKvarnValueGroupChannels ==
              kHeadDim);

__device__ __forceinline__ std::uint8_t KvarnUnpackCode(
    const std::uint8_t* payload, std::uint8_t bits, std::size_t index) {
  const std::uint8_t* const low = payload + kLowOffset;
  std::uint8_t code = static_cast<std::uint8_t>(
      (low[index / 2] >> (4U * (index % 2))) & 0x0FU);
  if (bits >= 5) {
    code = static_cast<std::uint8_t>(
        code |
        (((payload + kMidOffset)[index / 8] >> (index % 8)) & 1U) << 4);
  }
  if (bits >= 6) {
    code = static_cast<std::uint8_t>(
        code |
        (((payload + kHighOffset)[index / 8] >> (index % 8)) & 1U) << 5);
  }
  return code;
}

/// True when the view rule of design D2 reads tile `t` as quantized for a query
/// at absolute position `q`: e_t <= q - L - 1.
__device__ __forceinline__ bool KvarnTileQuantized(std::uint32_t q,
                                                   std::uint32_t tail_length,
                                                   std::uint32_t tile) {
  const std::uint32_t e_t = 128 + 128 * tile + 127;
  return e_t + tail_length + 1 <= q;
}

/// Payload of tile `t` for the K or V plane of one (layer, KV head).
__device__ __forceinline__ const std::uint8_t* KvarnTilePayload(
    const std::uint8_t* body, std::uint32_t layer, std::uint32_t kv_heads,
    std::uint32_t head, std::uint32_t tile, std::uint32_t tile_count,
    std::uint32_t k_tile_bytes, std::uint32_t v_tile_bytes, bool is_value) {
  const std::size_t offset = models::qwen::KvarnBodyTileOffset(
      layer, kv_heads, head, tile, tile_count, k_tile_bytes, v_tile_bytes);
  return body + offset + (is_value ? k_tile_bytes : 0);
}

/// F16 row of one (layer, KV head, position) in the sink or tail ring.
__device__ __forceinline__ const __half* KvarnF16Row(
    const __half* sink, const __half* tail, std::uint64_t sink_plane_elements,
    std::uint64_t tail_plane_elements, std::uint32_t layer,
    std::uint32_t kv_heads, std::uint32_t head, std::uint32_t position,
    std::uint32_t ring_rows, bool is_value) {
  if (position < models::qwen::kKvarnSinkRows) {
    const std::size_t row =
        models::qwen::KvarnSinkRowOffset(layer, kv_heads, head, position);
    return sink + (is_value ? sink_plane_elements : 0) + row;
  }
  const std::size_t row = models::qwen::KvarnTailRowOffset(
      layer, kv_heads, head, position, ring_rows);
  return tail + (is_value ? tail_plane_elements : 0) + row;
}

/// The four codes of one lane's `float4` group in a sealed tile. A lane owns
/// the channels `(lane + vector * 32) * 4 + k` (design D7), so the group's
/// four low nibbles sit in one aligned 16-bit word of the low plane and, for
/// 5- and 6-bit codes, its four extra bits in one nibble of a single byte of
/// the corresponding bit plane. The unpacked codes are identical to the
/// per-element `KvarnUnpackCode`; only the loads are wider.
__device__ __forceinline__ void KvarnUnpackLaneCodes(
    const std::uint8_t* payload, std::uint8_t bits, std::uint32_t token,
    std::uint32_t group, std::uint32_t (&codes)[4]) {
  const auto* const low =
      reinterpret_cast<const std::uint16_t*>(payload + kLowOffset);
  const std::uint32_t word = low[(token * (kHeadDim / 4)) + group];
  codes[0] = word & 0x000FU;
  codes[1] = (word >> 4) & 0x000FU;
  codes[2] = (word >> 8) & 0x000FU;
  codes[3] = (word >> 12) & 0x000FU;
  if (bits >= 5) {
    const std::uint32_t packed =
        (payload + kMidOffset)[(token * (kHeadDim / 8)) + (group >> 1)];
    const std::uint32_t shift = 4U * (group & 1U);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      codes[static_cast<std::uint32_t>(k)] |=
          ((packed >> (shift + static_cast<std::uint32_t>(k))) & 1U) << 4;
    }
  }
  if (bits >= 6) {
    const std::uint32_t packed =
        (payload + kHighOffset)[(token * (kHeadDim / 8)) + (group >> 1)];
    const std::uint32_t shift = 4U * (group & 1U);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      codes[static_cast<std::uint32_t>(k)] |=
          ((packed >> (shift + static_cast<std::uint32_t>(k))) & 1U) << 5;
    }
  }
}

/// Token normalization scale of one tile payload (design D5). K scores fold the
/// K tile's token scale and V folds the V tile's into the softmax weight, so
/// the two planes keep their own scales.
__device__ __forceinline__ float KvarnTileTokenScale(
    const std::uint8_t* payload, std::uint32_t token) {
  return __half2float(
      reinterpret_cast<const __half*>(payload + 4 * 256 + 2 * 256)[token]);
}

/// Per-lane folded K query of one sealed tile (design D5/D7).
struct KvarnKeyFold {
  /// `q[c] * channel_scale[c] * rtn_scale[c]` of the lane's channels.
  float query[kVectorsPerLane][4];
  /// `sum_c q[c] * channel_scale[c] * rtn_zero[c]`, the tile's constant term.
  float zero;
};

/// Folds one sealed K tile into the query (design D5/D7). Hoisted out of the
/// tile's 128 keys because the channel scale, the RTN scale and the RTN zero
/// point are per channel and do not depend on the token.
__device__ __forceinline__ KvarnKeyFold KvarnFoldKeyQuery(
    const std::uint8_t* payload, const float4 (&query)[kVectorsPerLane],
    std::uint32_t lane) {
  const auto* const channel_scale =
      reinterpret_cast<const __half*>(payload + 4 * kHeadDim);
  const auto* const rtn_scale = reinterpret_cast<const __half*>(payload);
  const auto* const rtn_zero = rtn_scale + models::qwen::kKvarnRtnParams;
  KvarnKeyFold fold;
  fold.zero = 0.0F;
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
    const auto* const query_vector =
        reinterpret_cast<const float*>(&query[vector]);
    const std::uint32_t base = (lane + (vector * kLanes)) * 4;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const std::uint32_t channel = base + static_cast<std::uint32_t>(k);
      const float scaled =
          query_vector[k] * __half2float(channel_scale[channel]);
      fold.query[vector][k] = scaled * __half2float(rtn_scale[channel]);
      fold.zero = fmaf(scaled, __half2float(rtn_zero[channel]), fold.zero);
    }
  }
  return fold;
}

/// One lane's K contribution of the `vector`-th group of one sealed key: the
/// folded query continues the running partial sum, in channel order (design
/// D5). Four codes are live at a time so the per-key block stays in registers.
__device__ __forceinline__ float KvarnKeyFoldVector(
    const KvarnKeyFold& fold, std::uint32_t vector,
    const std::uint32_t (&codes)[4], float score) {
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    score = fmaf(fold.query[vector][static_cast<std::uint32_t>(k)],
                 static_cast<float>(codes[static_cast<std::uint32_t>(k)]),
                 score);
  }
  return score;
}

/// F16 key score of one lane: the stored rotated row contributes as read.
__device__ __forceinline__ float KvarnF16KeyScore(
    const __half* row, const float4 (&query)[kVectorsPerLane],
    std::uint32_t lane) {
  float score = 0.0F;
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
    const auto* const query_vector =
        reinterpret_cast<const float*>(&query[vector]);
    const std::uint32_t base = (lane + (vector * kLanes)) * 4;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      score = fmaf(query_vector[k],
                   __half2float(row[base + static_cast<std::uint32_t>(k)]),
                   score);
    }
  }
  return score;
}

/// Lane reduction of one key score, shared by every read path so the tree and
/// the trailing scale keep one operand order (design D7, T-05).
__device__ __forceinline__ float KvarnReduceKeyScore(float score, float scale) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    score += __shfl_down(score, offset);
  }
  return __shfl(score, 0) * scale;
}

/// Per-lane V accumulator of one sealed tile, in the normalized space (design
/// D5/D7). The token scale and the RTN pair enter through the softmax weights;
/// the channel scale stays in the payload and is applied once, when the tile is
/// merged into the output.
struct KvarnValueFold {
  float code_sum[kVectorsPerLane][4];
  float zero_sum[kVectorsPerLane];
};

/// Starts a sealed V tile: clears the two accumulators.
__device__ __forceinline__ void KvarnValueFoldLoad(KvarnValueFold* fold) {
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      fold->code_sum[vector][k] = 0.0F;
    }
    fold->zero_sum[vector] = 0.0F;
  }
}

/// Rescales the tile accumulator by the online-softmax factor of one key block
/// (design D7). The output is rescaled separately because it also holds the
/// keys of earlier regions.
__device__ __forceinline__ void KvarnValueFoldRescale(KvarnValueFold* fold,
                                                      float scale) {
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      fold->code_sum[vector][k] *= scale;
    }
    fold->zero_sum[vector] *= scale;
  }
}

/// One token's V RTN pair of a sealed tile, for vector `vector` (design D5: one
/// pair per (token, 128-channel group)).
__device__ __forceinline__ void KvarnValueRtnPair(const std::uint8_t* payload,
                                                  std::uint32_t token,
                                                  std::uint32_t vector,
                                                  float* rtn_scale,
                                                  float* rtn_zero) {
  const auto* const scale = reinterpret_cast<const __half*>(payload);
  const auto* const zero = scale + models::qwen::kKvarnRtnParams;
  const std::uint32_t parameter =
      token * static_cast<std::uint32_t>(models::qwen::kKvarnValueGroups) +
      vector;
  *rtn_scale = __half2float(scale[parameter]);
  *rtn_zero = __half2float(zero[parameter]);
}

/// Accumulates one sealed key's `vector`-th group into its tile. `weight`
/// already carries the tile's token scale; the RTN pair is per (token,
/// 128-channel group) and the lane's vector `vector` covers exactly group
/// `vector` (design D5). Four codes are live at a time.
__device__ __forceinline__ void KvarnValueFoldVector(
    KvarnValueFold* fold, std::uint32_t vector,
    const std::uint32_t (&codes)[4], float weight, float rtn_scale,
    float rtn_zero) {
  const float weighted_scale = weight * rtn_scale;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    fold->code_sum[vector][static_cast<std::uint32_t>(k)] =
        fmaf(weighted_scale,
             static_cast<float>(codes[static_cast<std::uint32_t>(k)]),
             fold->code_sum[vector][static_cast<std::uint32_t>(k)]);
  }
  fold->zero_sum[vector] = fmaf(weight, rtn_zero, fold->zero_sum[vector]);
}

/// Merges a sealed V tile into the output: the tile's channel scale is applied
/// once, to the tile's normalized sum (design D5/D7).
__device__ __forceinline__ void KvarnValueFoldMerge(
    const KvarnValueFold& fold, const std::uint8_t* payload_v,
    std::uint32_t lane, float (&output)[kVectorsPerLane][4]) {
  const auto* const channel_scale =
      reinterpret_cast<const __half*>(payload_v + 4 * kHeadDim);
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
    const std::uint32_t base = (lane + (vector * kLanes)) * 4;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const std::uint32_t channel = base + static_cast<std::uint32_t>(k);
      output[vector][static_cast<std::uint32_t>(k)] =
          fmaf(__half2float(channel_scale[channel]),
               fold.code_sum[vector][static_cast<std::uint32_t>(k)] +
                   fold.zero_sum[vector],
               output[vector][static_cast<std::uint32_t>(k)]);
    }
  }
}

/// Merges the tile a verification row has open, re-deriving its payload from
/// the tile index so the channel scale can be read at merge time (design D5).
__device__ __forceinline__ void KvarnMergeOpenTile(
    const KvarnReadParams& kvarn, std::uint32_t layer_idx,
    std::uint32_t kv_head, std::uint32_t tile, std::uint32_t lane,
    KvarnValueFold* fold, float (&output)[kVectorsPerLane][4]) {
  const std::uint8_t* const payload_v = KvarnTilePayload(
      kvarn.body, layer_idx, kvarn.kv_heads, kv_head, tile, kvarn.tile_count,
      kvarn.k_tile_bytes, kvarn.v_tile_bytes, true);
  KvarnValueFoldMerge(*fold, payload_v, lane, output);
}

/// Per-lane online-softmax state of one query row (design D7).
struct KvarnOnlineState {
  float running_max;
  float running_sum;
  float output[kVectorsPerLane][4];
};

__device__ __forceinline__ void KvarnStateRescale(KvarnOnlineState* state,
                                                  float scale) {
  // `fmaf(x, s, 0)` is the explicit form of `x * s` (one rounding, exact for
  // this non-negative sum): the multiply is an intrinsic, so the machine
  // combiner cannot fuse the per-key `running_sum += weight` of a later block
  // step into it differently in the online, split-K and verification kernels.
  state->running_sum = fmaf(state->running_sum, scale, 0.0F);
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      state->output[vector][k] *= scale;
    }
  }
}

/// Opens a key block: folds the block's maximum into the online maximum and
/// rescales the running sum, the output and the open tile. Every key of the
/// block then uses the new maximum, so one rescale and one exponential of the
/// running maximum are shared by `kKeyBlock` keys (tasks 3.15(c)).
__device__ __forceinline__ void KvarnBlockOpen(
    const float (&scores)[kKeyBlock], std::uint32_t count,
    KvarnOnlineState* state, KvarnValueFold* tile, bool tile_active) {
  float block_max = scores[0];
#pragma unroll
  for (std::uint32_t i = 1; i < kKeyBlock; ++i) {
    if (i < count) {
      block_max = fmaxf(block_max, scores[i]);
    }
  }
  const float new_max = fmaxf(state->running_max, block_max);
  const float previous_scale = expf(state->running_max - new_max);
  state->running_max = new_max;
  KvarnStateRescale(state, previous_scale);
  if (tile_active) {
    KvarnValueFoldRescale(tile, previous_scale);
  }
}

/// Weight of one key of an open block. The raw exponential goes into the
/// running sum and the returned weight carries the token scale (design D5).
__device__ __forceinline__ float KvarnBlockWeight(float score,
                                                  KvarnOnlineState* state,
                                                  float token_scale) {
  const float weight = expf(score - state->running_max);
  state->running_sum += weight;
  return weight * token_scale;
}

/// Walks one query row's tile-aligned key partition `[begin, end)` and folds it
/// into `state` (design D2/D7/D8). The online and split-K decode paths call
/// this with the same partition structure, so their keys, weights and products
/// keep one operand order. A 128-key chunk is either one sealed tile (folded K,
/// normalized-space V) or F16 rows (sink, unsealed tile, tail ring).
__device__ __forceinline__ void KvarnWalkPartition(
    const KvarnReadParams& kvarn, std::uint32_t layer_idx,
    std::uint32_t kv_head, const float4 (&query)[kVectorsPerLane],
    std::uint32_t lane, float scale, std::uint32_t pos, std::uint32_t begin,
    std::uint32_t end, KvarnOnlineState* state) {
  for (std::uint32_t chunk = begin; chunk < end; chunk += kTileTokens) {
    const std::uint32_t chunk_end = min(chunk + kTileTokens, end);
    const bool sealed =
        chunk >= kTileTokens &&
        KvarnTileQuantized(pos, kvarn.tail_length,
                           (chunk - kTileTokens) / kTileTokens);
    if (!sealed) {
      for (std::uint32_t block = chunk; block < chunk_end;
           block += kKeyBlock) {
        const std::uint32_t count = min(kKeyBlock, chunk_end - block);
        float scores[kKeyBlock];
#pragma unroll
        for (std::uint32_t i = 0; i < kKeyBlock; ++i) {
          if (i < count) {
            const __half* const row = KvarnF16Row(
                kvarn.sink, kvarn.tail, kvarn.sink_plane_elements,
                kvarn.tail_plane_elements, layer_idx, kvarn.kv_heads, kv_head,
                block + i, kvarn.ring_rows, false);
            scores[i] = KvarnReduceKeyScore(
                KvarnF16KeyScore(row, query, lane), scale);
          }
        }
        KvarnBlockOpen(scores, count, state, nullptr, false);
#pragma unroll
        for (std::uint32_t i = 0; i < kKeyBlock; ++i) {
          if (i < count) {
            const float weight = KvarnBlockWeight(scores[i], state, 1.0F);
            const __half* const row = KvarnF16Row(
                kvarn.sink, kvarn.tail, kvarn.sink_plane_elements,
                kvarn.tail_plane_elements, layer_idx, kvarn.kv_heads, kv_head,
                block + i, kvarn.ring_rows, true);
#pragma unroll
            for (std::uint32_t vector = 0; vector < kVectorsPerLane;
                 ++vector) {
              const std::uint32_t base = (lane + (vector * kLanes)) * 4;
#pragma unroll
              for (int k = 0; k < 4; ++k) {
                state->output[vector][k] =
                    fmaf(weight,
                         __half2float(
                             row[base + static_cast<std::uint32_t>(k)]),
                         state->output[vector][k]);
              }
            }
          }
        }
      }
      continue;
    }
    const std::uint32_t tile = (chunk - kTileTokens) / kTileTokens;
    const std::uint8_t* const payload_k = KvarnTilePayload(
        kvarn.body, layer_idx, kvarn.kv_heads, kv_head, tile, kvarn.tile_count,
        kvarn.k_tile_bytes, kvarn.v_tile_bytes, false);
    const std::uint8_t* const payload_v = KvarnTilePayload(
        kvarn.body, layer_idx, kvarn.kv_heads, kv_head, tile, kvarn.tile_count,
        kvarn.k_tile_bytes, kvarn.v_tile_bytes, true);
    const KvarnKeyFold key_fold = KvarnFoldKeyQuery(payload_k, query, lane);
    KvarnValueFold value_fold;
    KvarnValueFoldLoad(&value_fold);
    const std::uint32_t tokens = chunk_end - chunk;
    for (std::uint32_t token = 0; token < tokens; token += kKeyBlock) {
      const std::uint32_t count = min(kKeyBlock, tokens - token);
      float scores[kKeyBlock];
#pragma unroll
      for (std::uint32_t i = 0; i < kKeyBlock; ++i) {
        if (i < count) {
          float score = key_fold.zero;
#pragma unroll
          for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
            std::uint32_t codes[4];
            KvarnUnpackLaneCodes(payload_k, kvarn.k_bits, token + i,
                                 (vector * kLanes) + lane, codes);
            score = KvarnKeyFoldVector(key_fold, vector, codes, score);
          }
          scores[i] = KvarnReduceKeyScore(
              score * KvarnTileTokenScale(payload_k, token + i), scale);
        }
      }
      KvarnBlockOpen(scores, count, state, &value_fold, true);
#pragma unroll
      for (std::uint32_t i = 0; i < kKeyBlock; ++i) {
        if (i < count) {
          const float weight = KvarnBlockWeight(
              scores[i], state, KvarnTileTokenScale(payload_v, token + i));
#pragma unroll
          for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
            std::uint32_t codes[4];
            KvarnUnpackLaneCodes(payload_v, kvarn.v_bits, token + i,
                                 (vector * kLanes) + lane, codes);
            float rtn_scale = 0.0F;
            float rtn_zero = 0.0F;
            KvarnValueRtnPair(payload_v, token + i, vector, &rtn_scale,
                              &rtn_zero);
            KvarnValueFoldVector(&value_fold, vector, codes, weight, rtn_scale,
                                 rtn_zero);
          }
        }
      }
    }
    KvarnValueFoldMerge(value_fold, payload_v, lane, state->output);
  }
}

__launch_bounds__(kLanes) __global__ void QwenKvarnDecodeOnlineKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    float* __restrict__ out_context, std::uint32_t layer_idx,
    std::uint32_t start_pos, const std::uint32_t* d_pos,
    std::uint32_t num_heads, std::uint32_t num_kv_heads,
    KvarnReadParams kvarn) {
  const std::size_t row_offset =
      static_cast<std::size_t>(blockIdx.y) * num_heads * kHeadDim;
  q += row_offset;
  if (gate != nullptr) {
    gate += row_offset;
  }
  out_context += row_offset;
  // Graph capture bakes no host position: the row's absolute position comes
  // from device memory when the caller supplies it (design D4/D7).
  const std::uint32_t pos =
      (d_pos != nullptr ? *d_pos : start_pos) + blockIdx.y;

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || lane >= kLanes) {
    return;
  }

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * kHeadDim));
  const float scale = rsqrtf(static_cast<float>(kHeadDim));

  float4 query_values[kVectorsPerLane];
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
    query_values[vector] = query[lane + (vector * kLanes)];
  }

  KvarnOnlineState state;
  state.running_max = -INFINITY;
  state.running_sum = 0.0F;
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      state.output[vector][k] = 0.0F;
    }
  }

  KvarnWalkPartition(kvarn, layer_idx, kv_head, query_values, lane, scale, pos,
                     0, pos + 1, &state);

  auto* output = reinterpret_cast<float4*>(
      out_context + (static_cast<std::size_t>(head) * kHeadDim));
  const auto* gate_values =
      gate == nullptr ? nullptr
                      : reinterpret_cast<const float4*>(
                            gate + (static_cast<std::size_t>(head) * kHeadDim));
  const float inverse_sum = 1.0F / state.running_sum;
  // Inverse Hadamard rotation of the merged context, once, before the SiLU
  // gate (design D3).
  __shared__ float s_context[kHeadDim];
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
    float4 result;
    result.x = state.output[vector][0] * inverse_sum;
    result.y = state.output[vector][1] * inverse_sum;
    result.z = state.output[vector][2] * inverse_sum;
    result.w = state.output[vector][3] * inverse_sum;
    *reinterpret_cast<float4*>(
        &s_context[(lane + (vector * kLanes)) * 4]) = result;
  }
  __syncwarp();
  KvarnFwhtRowWarp(s_context, static_cast<int>(lane));
  __syncwarp();
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
    float4 result =
        *reinterpret_cast<const float4*>(&s_context[(lane + (vector * kLanes)) * 4]);
    if (gate_values != nullptr) {
      const float4 gate_value = gate_values[lane + (vector * kLanes)];
      result.x *= 1.0F / (1.0F + expf(-gate_value.x));
      result.y *= 1.0F / (1.0F + expf(-gate_value.y));
      result.z *= 1.0F / (1.0F + expf(-gate_value.z));
      result.w *= 1.0F / (1.0F + expf(-gate_value.w));
    }
    output[lane + (vector * kLanes)] = result;
  }
}

/// One split of the KVarN decode. The split's key range is snapped to whole
/// 128-token tiles, so a split never straddles a sealed tile and the last split
/// absorbs the F16 tail rows (design D8). Partial max/sum and partial outputs
/// use the same scratch layout as the FP16 split-K decode, so the split count
/// (<= 32) and `DecodeAttentionScratchElements` stay valid. The walk is the one
/// `KvarnWalkPartition`, shared with the online kernel (design D7/T-05).
__launch_bounds__(kLanes) __global__ void QwenKvarnSplitKPartialsKernel(
    const float* __restrict__ q, float* __restrict__ scratch,
    std::uint32_t layer_idx, std::uint32_t start_pos,
    const std::uint32_t* d_pos, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t split_count,
    KvarnReadParams kvarn) {
  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;
  const std::size_t row = blockIdx.y;
  q += row * num_heads * kHeadDim;
  scratch += row * num_heads * max_splits * (kHeadDim + 2);
  const std::uint32_t pos =
      (d_pos != nullptr ? *d_pos : start_pos) + static_cast<std::uint32_t>(row);

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t split = blockIdx.z;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || split >= split_count || lane >= kLanes) {
    return;
  }

  const std::size_t partition_index =
      static_cast<std::size_t>(head) * max_splits + split;
  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
  auto* partial_outputs = reinterpret_cast<float4*>(scratch + stats_elements);
  auto* partial_output = partial_outputs + partition_index * kVectorWidth;

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * kHeadDim));
  const float scale = rsqrtf(static_cast<float>(kHeadDim));

  float4 query_values[kVectorsPerLane];
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
    query_values[vector] = query[lane + (vector * kLanes)];
  }

  const std::uint32_t sequence_length = pos + 1;
  // Whole-tile split: boundaries land on 128-token tiles and the last split
  // absorbs the remainder (design D8).
  const std::uint32_t blocks =
      (sequence_length + kTileTokens - 1) / kTileTokens;
  const std::uint32_t begin_block = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(split) * blocks) / split_count);
  const std::uint32_t end_block = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(split + 1) * blocks) / split_count);
  const std::uint32_t partition_begin = begin_block * kTileTokens;
  const std::uint32_t partition_end =
      min(end_block * kTileTokens, sequence_length);

  KvarnOnlineState state;
  state.running_max = -INFINITY;
  state.running_sum = 0.0F;
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      state.output[vector][k] = 0.0F;
    }
  }

  KvarnWalkPartition(kvarn, layer_idx, kv_head, query_values, lane, scale, pos,
                     partition_begin, partition_end, &state);

  if (lane == 0) {
    scratch[partition_index * 2] = state.running_max;
    scratch[partition_index * 2 + 1] = state.running_sum;
  }
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
    float4 result;
    result.x = state.output[vector][0];
    result.y = state.output[vector][1];
    result.z = state.output[vector][2];
    result.w = state.output[vector][3];
    partial_output[lane + (vector * kLanes)] = result;
  }
}

/// Merges the KVarN split-K partials into the context. Whole-tile rounding can
/// leave some of the 32 partitions without keys; those hold a sum of 0 and are
/// skipped, so the merge never depends on an empty partition's scale.
__launch_bounds__(kLanes) __global__ void QwenKvarnSplitKReduceKernel(
    const float* __restrict__ gate, const float* __restrict__ scratch,
    float* __restrict__ out_context, std::uint32_t num_heads,
    std::uint32_t split_count) {
  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;
  const std::size_t row = blockIdx.y;
  if (gate != nullptr) {
    gate += row * num_heads * kHeadDim;
  }
  out_context += row * num_heads * kHeadDim;
  scratch += row * num_heads * max_splits * (kHeadDim + 2);

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || lane >= kLanes) {
    return;
  }

  const std::size_t head_partition_offset =
      static_cast<std::size_t>(head) * max_splits;
  float global_max = -INFINITY;
  for (std::uint32_t split = 0; split < split_count; ++split) {
    global_max = fmaxf(
        global_max, scratch[((head_partition_offset + split) * 2) + 0]);
  }

  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
  const auto* partial_outputs =
      reinterpret_cast<const float4*>(scratch + stats_elements);
  auto* output = reinterpret_cast<float4*>(
      out_context + (static_cast<std::size_t>(head) * kHeadDim));
  const auto* gate_values =
      gate == nullptr ? nullptr
                      : reinterpret_cast<const float4*>(
                            gate + (static_cast<std::size_t>(head) * kHeadDim));

  float global_sum = 0.0F;
  float4 results[kVectorsPerLane]{};
  for (std::uint32_t split = 0; split < split_count; ++split) {
    const std::size_t partition_index = head_partition_offset + split;
    const float partition_sum = scratch[(partition_index * 2) + 1];
    if (partition_sum <= 0.0F) {
      continue;
    }
    const float partition_scale =
        expf(scratch[partition_index * 2] - global_max);
    global_sum = fmaf(partition_sum, partition_scale, global_sum);
#pragma unroll
    for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
      const float4 partial = partial_outputs[(partition_index * kVectorWidth) +
                                             lane + (vector * kLanes)];
      auto& result = results[vector];
      result.x = fmaf(partial.x, partition_scale, result.x);
      result.y = fmaf(partial.y, partition_scale, result.y);
      result.z = fmaf(partial.z, partition_scale, result.z);
      result.w = fmaf(partial.w, partition_scale, result.w);
    }
  }
  const float inverse_sum = 1.0F / global_sum;
  // The inverse rotation is applied once after the partial merge, before the
  // SiLU gate (design D3).
  __shared__ float s_context[kHeadDim];
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
    auto result = results[vector];
    result.x *= inverse_sum;
    result.y *= inverse_sum;
    result.z *= inverse_sum;
    result.w *= inverse_sum;
    *reinterpret_cast<float4*>(
        &s_context[(lane + (vector * kLanes)) * 4]) = result;
  }
  __syncwarp();
  KvarnFwhtRowWarp(s_context, static_cast<int>(lane));
  __syncwarp();
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
    float4 result =
        *reinterpret_cast<const float4*>(&s_context[(lane + (vector * kLanes)) * 4]);
    if (gate_values != nullptr) {
      const float4 gate_value = gate_values[lane + (vector * kLanes)];
      result.x *= 1.0F / (1.0F + expf(-gate_value.x));
      result.y *= 1.0F / (1.0F + expf(-gate_value.y));
      result.z *= 1.0F / (1.0F + expf(-gate_value.z));
      result.w *= 1.0F / (1.0F + expf(-gate_value.w));
    }
    output[lane + (vector * kLanes)] = result;
  }
}

// T-05 verification rows (design D2/D7/D8). One block covers one query head and
// one tile-aligned split across up to `kCausalMaxRows` rows, one warp per row.
// A chunk of key positions has its raw codes staged once in LDS and consumed by
// every row that reads it quantized; each row then runs the same folded
// arithmetic, in the same order, as the single-row decoders, so its logits stay
// bit-identical to the AR decode at the same position.
constexpr std::uint32_t kCausalMaxRows = 8;
constexpr std::uint32_t kCausalChunk = kKeyBlock;
constexpr std::uint32_t kCausalThreads = kCausalMaxRows * kLanes;

__launch_bounds__(kCausalThreads) __global__ void QwenKvarnCausalDecodeKernel(
    const float* __restrict__ q, float* __restrict__ scratch,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::uint32_t rows,
    std::uint32_t num_heads, std::uint32_t num_kv_heads,
    std::uint32_t split_count, KvarnReadParams kvarn) {
  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;
  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t split = blockIdx.y;
  const std::uint32_t warp = threadIdx.x / kLanes;
  const std::uint32_t lane = threadIdx.x % kLanes;

  __shared__ std::uint32_t s_begin[kCausalMaxRows];
  __shared__ std::uint32_t s_end[kCausalMaxRows];
  __shared__ std::uint32_t s_pos[kCausalMaxRows];
  // Raw codes of the chunk, one byte per (token, channel), staged once per
  // block. Staging the codes instead of decoded FP32 values keeps the LDS
  // footprint at 4 KiB (the FP32 staging cost occupancy; see EXPERIMENTS.md)
  // while the rows still run the identical folded arithmetic.
  __shared__ std::uint32_t s_codes_k[kCausalChunk * kHeadDim / 4];
  __shared__ std::uint32_t s_codes_v[kCausalChunk * kHeadDim / 4];

  // Per-row partition: rows below the split-K threshold use one partition
  // [0, sequence) exactly like the single-row online decoder; longer rows use
  // the tile-aligned split of the single-row split-K decoder (design D8).
  const bool row_active = warp < rows;
  const std::uint32_t pos = start_pos + warp;
  std::uint32_t begin = 0;
  std::uint32_t end = 0;
  if (row_active) {
    const std::uint32_t sequence = pos + 1;
    if (sequence < detail::kSplitKDecodeAttentionMinContext) {
      if (split == 0) {
        end = sequence;
      } else {
        begin = sequence;
        end = sequence;
      }
    } else {
      const std::uint32_t blocks = (sequence + kTileTokens - 1) / kTileTokens;
      const std::uint32_t begin_block = (split * blocks) / split_count;
      const std::uint32_t end_block = ((split + 1) * blocks) / split_count;
      begin = begin_block * kTileTokens;
      end = min(end_block * kTileTokens, sequence);
    }
  }
  if (lane == 0) {
    if (warp < rows) {
      s_begin[warp] = begin;
      s_end[warp] = end;
      s_pos[warp] = pos;
    }
  }
  __syncthreads();
  const bool active = row_active && s_begin[warp] < s_end[warp];

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  float* const row_scratch =
      scratch + static_cast<std::size_t>(warp) * num_heads * max_splits *
                    (kHeadDim + 2);

  float4 query_values[kVectorsPerLane];
  if (active) {
    const auto* const query = reinterpret_cast<const float4*>(
        q + (static_cast<std::size_t>(warp) * num_heads + head) * kHeadDim);
#pragma unroll
    for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
      query_values[vector] = query[lane + (vector * kLanes)];
    }
  }

  KvarnOnlineState state;
  state.running_max = -INFINITY;
  state.running_sum = 0.0F;
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      state.output[vector][k] = 0.0F;
    }
  }
  KvarnValueFold value_fold{};
  bool tile_open = false;
  std::uint32_t open_tile = 0;

  std::uint32_t union_begin = 0xFFFFFFFFU;
  std::uint32_t union_end = 0;
  for (std::uint32_t r = 0; r < rows; ++r) {
    if (s_begin[r] < s_end[r]) {
      union_begin = min(union_begin, s_begin[r]);
      union_end = max(union_end, s_end[r]);
    }
  }

  const float scale = rsqrtf(static_cast<float>(kHeadDim));
  for (std::uint32_t chunk = union_begin; chunk < union_end;
       chunk += kCausalChunk) {
    const std::uint32_t count = min(kCausalChunk, union_end - chunk);
    // Stage the raw codes of this chunk once: only positions at or beyond the
    // sink that at least one row reads quantized are decoded.
    for (std::uint32_t i = threadIdx.x; i < count * kHeadDim;
         i += kCausalThreads) {
      const std::uint32_t token = i / kHeadDim;
      const std::uint32_t channel = i % kHeadDim;
      const std::uint32_t position = chunk + token;
      std::uint8_t code_k = 0;
      std::uint8_t code_v = 0;
      if (position >= kTileTokens) {
        const std::uint32_t tile = (position - kTileTokens) / kTileTokens;
        const std::uint32_t e_t = kTileTokens + kTileTokens * tile + 127;
        bool staged = false;
        for (std::uint32_t r = 0; r < rows; ++r) {
          if (position >= s_begin[r] && position < s_end[r] &&
              e_t + kvarn.tail_length + 1 <= s_pos[r]) {
            staged = true;
            break;
          }
        }
        if (staged) {
          const std::uint32_t token_in_tile = position - kTileTokens -
                                              kTileTokens * tile;
          const std::uint8_t* const payload_k = KvarnTilePayload(
              kvarn.body, layer_idx, kvarn.kv_heads, kv_head, tile,
              kvarn.tile_count, kvarn.k_tile_bytes, kvarn.v_tile_bytes, false);
          const std::uint8_t* const payload_v = KvarnTilePayload(
              kvarn.body, layer_idx, kvarn.kv_heads, kv_head, tile,
              kvarn.tile_count, kvarn.k_tile_bytes, kvarn.v_tile_bytes, true);
          const std::size_t index =
              static_cast<std::size_t>(token_in_tile) * kHeadDim + channel;
          code_k = KvarnUnpackCode(payload_k, kvarn.k_bits, index);
          code_v = KvarnUnpackCode(payload_v, kvarn.v_bits, index);
        }
      }
      reinterpret_cast<std::uint8_t*>(s_codes_k)[i] = code_k;
      reinterpret_cast<std::uint8_t*>(s_codes_v)[i] = code_v;
    }
    __syncthreads();

    if (active) {
      const std::uint32_t lo = max(chunk, s_begin[warp]);
      const std::uint32_t hi = min(chunk + count, s_end[warp]);
      if (lo < hi) {
        // A chunk never straddles a tile, so the row's whole block belongs to
        // one region: a sealed tile or F16 rows (design D2).
        const bool quantized =
            lo >= kTileTokens &&
            KvarnTileQuantized(s_pos[warp], kvarn.tail_length,
                               (lo - kTileTokens) / kTileTokens);
        const std::uint32_t row_count = hi - lo;
        if (quantized) {
          const std::uint32_t tile = (lo - kTileTokens) / kTileTokens;
          if (tile_open && tile != open_tile) {
            KvarnMergeOpenTile(kvarn, layer_idx, kv_head, open_tile, lane,
                               &value_fold, state.output);
            tile_open = false;
          }
          const std::uint8_t* const payload_k = KvarnTilePayload(
              kvarn.body, layer_idx, kvarn.kv_heads, kv_head, tile,
              kvarn.tile_count, kvarn.k_tile_bytes, kvarn.v_tile_bytes, false);
          const std::uint8_t* const payload_v = KvarnTilePayload(
              kvarn.body, layer_idx, kvarn.kv_heads, kv_head, tile,
              kvarn.tile_count, kvarn.k_tile_bytes, kvarn.v_tile_bytes, true);
          if (!tile_open) {
            KvarnValueFoldLoad(&value_fold);
            tile_open = true;
            open_tile = tile;
          }
          const KvarnKeyFold key_fold =
              KvarnFoldKeyQuery(payload_k, query_values, lane);
          float scores[kKeyBlock];
#pragma unroll
          for (std::uint32_t i = 0; i < kKeyBlock; ++i) {
            if (i < row_count) {
              const std::uint32_t token =
                  lo + i - kTileTokens - kTileTokens * tile;
              float score = key_fold.zero;
#pragma unroll
              for (std::uint32_t vector = 0; vector < kVectorsPerLane;
                   ++vector) {
                // The staged plane is indexed by the token inside this chunk
                // (`i`, since a row's block starts at the chunk edge).
                const std::uint32_t word =
                    s_codes_k[i * (kHeadDim / 4) + lane + (vector * kLanes)];
                const std::uint32_t codes[4] = {
                    word & 0xFFU, (word >> 8) & 0xFFU, (word >> 16) & 0xFFU,
                    (word >> 24) & 0xFFU};
                score = KvarnKeyFoldVector(key_fold, vector, codes, score);
              }
              scores[i] = KvarnReduceKeyScore(
                  score * KvarnTileTokenScale(payload_k, token), scale);
            }
          }
          KvarnBlockOpen(scores, row_count, &state, &value_fold, true);
#pragma unroll
          for (std::uint32_t i = 0; i < kKeyBlock; ++i) {
            if (i < row_count) {
              const std::uint32_t token =
                  lo + i - kTileTokens - kTileTokens * tile;
              const float weight = KvarnBlockWeight(
                  scores[i], &state, KvarnTileTokenScale(payload_v, token));
#pragma unroll
              for (std::uint32_t vector = 0; vector < kVectorsPerLane;
                   ++vector) {
                const std::uint32_t word =
                    s_codes_v[i * (kHeadDim / 4) + lane + (vector * kLanes)];
                const std::uint32_t codes[4] = {
                    word & 0xFFU, (word >> 8) & 0xFFU, (word >> 16) & 0xFFU,
                    (word >> 24) & 0xFFU};
                float rtn_scale = 0.0F;
                float rtn_zero = 0.0F;
                KvarnValueRtnPair(payload_v, token, vector, &rtn_scale,
                                  &rtn_zero);
                KvarnValueFoldVector(&value_fold, vector, codes, weight,
                                     rtn_scale, rtn_zero);
              }
            }
          }
        } else {
          if (tile_open) {
            KvarnMergeOpenTile(kvarn, layer_idx, kv_head, open_tile, lane,
                               &value_fold, state.output);
            tile_open = false;
          }
          float scores[kKeyBlock];
#pragma unroll
          for (std::uint32_t i = 0; i < kKeyBlock; ++i) {
            if (i < row_count) {
              const __half* const row = KvarnF16Row(
                  kvarn.sink, kvarn.tail, kvarn.sink_plane_elements,
                  kvarn.tail_plane_elements, layer_idx, kvarn.kv_heads, kv_head,
                  lo + i, kvarn.ring_rows, false);
              scores[i] = KvarnReduceKeyScore(
                  KvarnF16KeyScore(row, query_values, lane), scale);
            }
          }
          KvarnBlockOpen(scores, row_count, &state, nullptr, false);
#pragma unroll
          for (std::uint32_t i = 0; i < kKeyBlock; ++i) {
            if (i < row_count) {
              const float weight = KvarnBlockWeight(scores[i], &state, 1.0F);
              const __half* const row = KvarnF16Row(
                  kvarn.sink, kvarn.tail, kvarn.sink_plane_elements,
                  kvarn.tail_plane_elements, layer_idx, kvarn.kv_heads, kv_head,
                  lo + i, kvarn.ring_rows, true);
#pragma unroll
              for (std::uint32_t vector = 0; vector < kVectorsPerLane;
                   ++vector) {
                const std::uint32_t base = (lane + (vector * kLanes)) * 4;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                  state.output[vector][k] = fmaf(
                      weight,
                      __half2float(row[base + static_cast<std::uint32_t>(k)]),
                      state.output[vector][k]);
                }
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }

  if (active && tile_open) {
    KvarnMergeOpenTile(kvarn, layer_idx, kv_head, open_tile, lane, &value_fold,
                       state.output);
    tile_open = false;
  }

  // Empty partitions keep a zero sum so the reduce step skips them, exactly as
  // the single-row decoder's empty partitions are skipped. Idle warps (beyond
  // the group's row count) own no scratch slice.
  if (warp >= rows) {
    return;
  }
  if (lane == 0) {
    row_scratch[(static_cast<std::size_t>(head) * max_splits + split) * 2] =
        active ? state.running_max : -INFINITY;
    row_scratch[(static_cast<std::size_t>(head) * max_splits + split) * 2 + 1] =
        active ? state.running_sum : 0.0F;
  }
  auto* const partial_outputs =
      reinterpret_cast<float4*>(row_scratch + stats_elements);
  float4* const partial_output =
      partial_outputs +
      (static_cast<std::size_t>(head) * max_splits + split) * kVectorWidth;
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
    float4 result;
    result.x = state.output[vector][0];
    result.y = state.output[vector][1];
    result.z = state.output[vector][2];
    result.w = state.output[vector][3];
    partial_output[lane + (vector * kLanes)] = result;
  }
}

}  // namespace

void LaunchKvarnCausalDecodeAttention(
    const float* q, const float* gate, float* out_context,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t num_heads, std::uint32_t num_kv_heads,
    const KvarnReadParams& kvarn, float* split_k_scratch, hipStream_t stream) {
  if (split_k_scratch == nullptr || batch_size == 0 || num_kv_heads == 0) {
    return;
  }
  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;
  const std::size_t row_stride =
      static_cast<std::size_t>(num_heads) * max_splits * (kHeadDim + 2);
  const std::size_t width = static_cast<std::size_t>(num_heads) * kHeadDim;
  for (std::size_t row = 0; row < batch_size; row += kCausalMaxRows) {
    const std::uint32_t rows = static_cast<std::uint32_t>(
        std::min<std::size_t>(kCausalMaxRows, batch_size - row));
    const std::uint32_t group_start = start_pos + static_cast<std::uint32_t>(row);
    const std::uint32_t splits =
        (group_start + rows >= detail::kSplitKDecodeAttentionMinContext)
            ? detail::kSplitKDecodeAttentionMaxSplits
            : 1;
    QwenKvarnCausalDecodeKernel<<<dim3(num_heads, splits), kCausalThreads, 0,
                                  stream>>>(
        q + row * width, split_k_scratch + row * row_stride, layer_idx,
        group_start, rows, num_heads, num_kv_heads, splits, kvarn);
    QwenKvarnSplitKReduceKernel<<<dim3(num_heads, rows), kLanes, 0, stream>>>(
        gate == nullptr ? nullptr : gate + row * width,
        split_k_scratch + row * row_stride, out_context + row * width,
        num_heads, splits);
  }
}

void LaunchKvarnDecodeAttention(const float* q, const float* gate,
                                float* out_context, std::uint32_t layer_idx,
                                std::uint32_t start_pos,
                                const std::uint32_t* d_pos,
                                std::size_t batch_size,
                                std::uint32_t num_heads,
                                std::uint32_t num_kv_heads,
                                const KvarnReadParams& kvarn,
                                float* split_k_scratch, hipStream_t stream) {
  const std::uint32_t sequence_length =
      start_pos + static_cast<std::uint32_t>(batch_size);
  if (split_k_scratch != nullptr && detail::IsSplitKDecodeAttentionSupported(
          sequence_length, num_heads, num_kv_heads, kHeadDim)) {
    const std::uint32_t splits =
        detail::SelectDecodeAttentionSplitCount(sequence_length);
    const dim3 partial_grid(num_heads, static_cast<unsigned>(batch_size),
                            splits);
    QwenKvarnSplitKPartialsKernel<<<partial_grid, kLanes, 0, stream>>>(
        q, split_k_scratch, layer_idx, start_pos, d_pos, num_heads,
        num_kv_heads, splits, kvarn);
    const dim3 reduce_grid(num_heads, static_cast<unsigned>(batch_size));
    QwenKvarnSplitKReduceKernel<<<reduce_grid, kLanes, 0, stream>>>(
        gate, split_k_scratch, out_context, num_heads, splits);
    return;
  }
  const dim3 grid(num_heads, static_cast<unsigned>(batch_size));
  QwenKvarnDecodeOnlineKernel<<<grid, kLanes, 0, stream>>>(
      q, gate, out_context, layer_idx, start_pos, d_pos, num_heads,
      num_kv_heads, kvarn);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
