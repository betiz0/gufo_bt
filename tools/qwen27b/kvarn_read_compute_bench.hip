// KVarN decode-read compute-only microbenchmark (openspec change
// add-kvarn-kv-cache-quantization, design D18 follow-up).
//
// The production depth cost is dominated by KV memory traffic (the live KV does
// not fit in cache), so it cannot separate compute from bandwidth. Here one
// full-attention layer's KV is small enough to stay resident in the SoC cache
// (2 MiB FP16 for the unset tail, ~10 MiB for tail 2048), its tiles are sealed
// with the CPU reference encoder, and the read kernel is timed over it. The
// per-(query head, key) unit cost is scaled to the 128K-token, 16-layer decode
// to give the compute-only floor of one generated token, next to the bandwidth
// floor B(depth)/BW. That decides whether 1.00x FP16 at 128K is reachable.
//
// Build (system hipcc):
//   hipcc -O3 --offload-arch=gfx1151 -std=c++20 -I. -c \
//     tools/qwen27b/kvarn_read_compute_bench.hip -o /tmp/kvrcb_main.o
//   hipcc -O3 -std=c++20 -I. -c src/models/qwen/kvarn_reference.cpp \
//     -o /tmp/kvrcb_ref.o
//   hipcc -O3 -std=c++20 -I. -c src/core/quant/ggml_dequant.cpp \
//     -o /tmp/kvrcb_quant.o
//   hipcc --offload-arch=gfx1151 /tmp/kvrcb_main.o /tmp/kvrcb_ref.o \
//     /tmp/kvrcb_quant.o -o /tmp/kvarn_read_compute_bench
//
// `tools/bench/build.sh tools/qwen27b/kvarn_read_compute_bench.hip` links the
// same two native units. The default mode times the read; `--check` seals a
// small fixture with the CPU encoder and compares the production read and the
// tile-unit prototype against a CPU reference; `--stream` times the production
// read, the prototype and the dense FP16 read over one 128K-token layer whose
// KV does not fit the SoC cache.
#define ENGINE_ENABLE_HIP 1
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/kernels/attention_decode.hip"
#include "src/models/qwen/hip/kernels/kvarn_read.hip"
#include "src/models/qwen/kvarn_reference.hpp"

// Tile-unit prototype of the KVarN decode read (openspec change
// improve-kvarn-k6v5-quality-and-speed, design D4/D5). One workgroup per
// (KV head, split) handles the KV head's six GQA query heads together: warps
// 0..5 each own one query head, and one 128-key tile (or an unsealed F16
// chunk) is expanded once per 32-key stage into LDS as f16 codes, shared by
// every head. The split partition and the partial layout are the production
// split-K ones, so `QwenKvarnSplitKReduceKernel` merges the partials unchanged.
namespace gufo::hip {
namespace {

constexpr std::uint32_t kPrototypeThreads = 256;
constexpr std::uint32_t kPrototypeHeadsPerGroup = 6;
constexpr std::uint32_t kPrototypeStageTokens = 32;
constexpr std::uint32_t kPrototypeStagesPerTile =
    kTileTokens / kPrototypeStageTokens;
constexpr std::uint32_t kPrototypeStageElements =
    kPrototypeStageTokens * kHeadDim;

/// One stored code as f16 through the FP16 magic number 0x6400 (= 1024.0):
/// the ULP at 1024 is 1, so `0x6400 + code` decodes to exactly `1024 + code`
/// for the at most 6-bit code and one half subtraction recovers the code.
__device__ __forceinline__ __half KvarnCodeHalf(std::uint32_t code) {
  return __hsub(__ushort_as_half(static_cast<unsigned short>(0x6400U + code)),
                __ushort_as_half(0x6400U));
}

/// Maximum of `count` scores held one per lane stride across the warp.
__device__ __forceinline__ float PrototypeTileMax(const float* scores,
                                                  std::uint32_t count,
                                                  std::uint32_t lane) {
  float local = -INFINITY;
  for (std::uint32_t t = lane; t < count; t += kLanes) {
    local = fmaxf(local, scores[t]);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    local = fmaxf(local, __shfl_down(local, offset));
  }
  return __shfl(local, 0);
}

/// Opens one chunk: folds the chunk's tile-wide maximum into the running one,
/// rescales the running sum and output, and writes the per-key weights.
__device__ __forceinline__ void PrototypeOpenChunk(
    float tile_max, std::uint32_t count, const float* scores, float* weights,
    std::uint32_t lane, float* running_max, float* running_sum,
    float (&output)[kVectorsPerLane][4]) {
  const float new_max = fmaxf(*running_max, tile_max);
  const float rescale = expf(*running_max - new_max);
  *running_sum *= rescale;
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      output[vector][k] *= rescale;
    }
  }
  float partial = 0.0F;
  for (std::uint32_t t = lane; t < count; t += kLanes) {
    const float weight = expf(scores[t] - new_max);
    weights[t] = weight;
    partial += weight;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    partial += __shfl_down(partial, offset);
  }
  *running_sum += __shfl(partial, 0);
  *running_max = new_max;
}

__launch_bounds__(kPrototypeThreads) __global__
    void QwenKvarnTilePrototypeKernel(
        const float* __restrict__ q, float* __restrict__ scratch,
        std::uint32_t layer_idx, std::uint32_t start_pos,
        std::uint32_t num_heads, std::uint32_t num_kv_heads,
        std::uint32_t split_count, KvarnReadParams kvarn) {
  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;
  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t split = blockIdx.y;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t warp = tid / kLanes;
  const std::uint32_t lane = tid % kLanes;
  if (kv_head >= num_kv_heads || split >= split_count) {
    return;
  }

  const std::uint32_t pos = start_pos;
  const std::uint32_t group_size = num_heads / num_kv_heads;
  const bool head_active = warp < group_size;
  const std::uint32_t head = kv_head * group_size + warp;

  // Whole-tile split boundaries, identical to QwenKvarnSplitKPartialsKernel.
  const std::uint32_t sequence_length = pos + 1;
  const std::uint32_t blocks =
      (sequence_length + kTileTokens - 1) / kTileTokens;
  const std::uint32_t begin_block = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(split) * blocks) / split_count);
  const std::uint32_t end_block = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(split + 1) * blocks) / split_count);
  const std::uint32_t begin = begin_block * kTileTokens;
  const std::uint32_t end = min(end_block * kTileTokens, sequence_length);

  __shared__ __half s_stage[kPrototypeStageElements];
  __shared__ float s_scores[kPrototypeHeadsPerGroup][kTileTokens];
  __shared__ float s_weights[kPrototypeHeadsPerGroup][kTileTokens];

  const float scale = rsqrtf(static_cast<float>(kHeadDim));
  float4 query_values[kVectorsPerLane]{};
  if (head_active) {
    const auto* const query = reinterpret_cast<const float4*>(
        q + static_cast<std::size_t>(head) * kHeadDim);
#pragma unroll
    for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
      query_values[vector] = query[lane + (vector * kLanes)];
    }
  }

  float running_max = -INFINITY;
  float running_sum = 0.0F;
  float output[kVectorsPerLane][4];
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      output[vector][k] = 0.0F;
    }
  }

  for (std::uint32_t chunk = begin; chunk < end; chunk += kTileTokens) {
    const std::uint32_t chunk_end = min(chunk + kTileTokens, end);
    const std::uint32_t count = chunk_end - chunk;
    const bool sealed = chunk >= kTileTokens &&
                        KvarnTileQuantized(pos, kvarn.tail_length,
                                           (chunk - kTileTokens) / kTileTokens);
    if (!sealed) {
      // Unsealed chunk: the F16 rows are already h16 in the sink/ring, so no
      // LDS staging is needed; every head scores the shared rows itself.
      if (head_active) {
        for (std::uint32_t t = 0; t < count; ++t) {
          const __half* const row =
              KvarnF16Row(kvarn.sink, kvarn.tail, kvarn.sink_plane_elements,
                          kvarn.tail_plane_elements, layer_idx, kvarn.kv_heads,
                          kv_head, chunk + t, kvarn.ring_rows, false);
          const float score = KvarnReduceKeyScore(
              KvarnF16KeyScore(row, query_values, lane), scale);
          if (lane == 0) {
            s_scores[warp][t] = score;
          }
        }
        __syncwarp();
        PrototypeOpenChunk(PrototypeTileMax(s_scores[warp], count, lane), count,
                           s_scores[warp], s_weights[warp], lane, &running_max,
                           &running_sum, output);
        __syncwarp();
        for (std::uint32_t t = 0; t < count; ++t) {
          const float weight = s_weights[warp][t];
          const __half* const row =
              KvarnF16Row(kvarn.sink, kvarn.tail, kvarn.sink_plane_elements,
                          kvarn.tail_plane_elements, layer_idx, kvarn.kv_heads,
                          kv_head, chunk + t, kvarn.ring_rows, true);
#pragma unroll
          for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
            const std::uint32_t base = (lane + (vector * kLanes)) * 4;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              output[vector][k] =
                  fmaf(weight,
                       __half2float(row[base + static_cast<std::uint32_t>(k)]),
                       output[vector][k]);
            }
          }
        }
      }
      continue;
    }

    // Sealed tile: expand the K payload once per stage and share it.
    const std::uint32_t tile = (chunk - kTileTokens) / kTileTokens;
    const std::uint8_t* const payload_k = KvarnTilePayload(
        kvarn.body, layer_idx, kvarn.kv_heads, kv_head, tile, kvarn.tile_count,
        kvarn.k_tile_bytes, kvarn.v_tile_bytes, false);
    const std::uint8_t* const payload_v = KvarnTilePayload(
        kvarn.body, layer_idx, kvarn.kv_heads, kv_head, tile, kvarn.tile_count,
        kvarn.k_tile_bytes, kvarn.v_tile_bytes, true);
    KvarnKeyFold fold{};
    if (head_active) {
      fold = KvarnFoldKeyQuery(payload_k, query_values, lane);
    }
    for (std::uint32_t stage = 0; stage < kPrototypeStagesPerTile; ++stage) {
      const std::uint32_t token0 = stage * kPrototypeStageTokens;
      for (std::uint32_t i = tid; i < kPrototypeStageElements;
           i += kPrototypeThreads) {
        const std::uint32_t token = i / kHeadDim;
        const std::uint32_t channel = i % kHeadDim;
        s_stage[i] = KvarnCodeHalf(KvarnUnpackCode(
            payload_k, kvarn.k_bits,
            static_cast<std::size_t>(token0 + token) * kHeadDim + channel));
      }
      __syncthreads();
      if (head_active) {
        const std::uint32_t stage_end =
            min(token0 + kPrototypeStageTokens, count);
        for (std::uint32_t token = token0; token < stage_end; ++token) {
          float score = fold.zero;
#pragma unroll
          for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
            const std::uint32_t base = (lane + (vector * kLanes)) * 4;
            std::uint32_t codes[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              codes[k] = static_cast<std::uint32_t>(
                  __half2float(s_stage[(token - token0) * kHeadDim + base +
                                       static_cast<std::uint32_t>(k)]));
            }
            score = KvarnKeyFoldVector(fold, vector, codes, score);
          }
          const float reduced = KvarnReduceKeyScore(
              score * KvarnTileTokenScale(payload_k, token), scale);
          if (lane == 0) {
            s_scores[warp][token] = reduced;
          }
        }
      }
      __syncthreads();
    }
    if (head_active) {
      PrototypeOpenChunk(PrototypeTileMax(s_scores[warp], count, lane), count,
                         s_scores[warp], s_weights[warp], lane, &running_max,
                         &running_sum, output);
      __syncwarp();
    }

    // V: expand the same tile once per stage, shared across the six heads.
    for (std::uint32_t stage = 0; stage < kPrototypeStagesPerTile; ++stage) {
      const std::uint32_t token0 = stage * kPrototypeStageTokens;
      for (std::uint32_t i = tid; i < kPrototypeStageElements;
           i += kPrototypeThreads) {
        const std::uint32_t token = i / kHeadDim;
        const std::uint32_t channel = i % kHeadDim;
        s_stage[i] = KvarnCodeHalf(KvarnUnpackCode(
            payload_v, kvarn.v_bits,
            static_cast<std::size_t>(token0 + token) * kHeadDim + channel));
      }
      __syncthreads();
      if (head_active) {
        const std::uint32_t stage_end =
            min(token0 + kPrototypeStageTokens, count);
        for (std::uint32_t token = token0; token < stage_end; ++token) {
          const float weighted =
              s_weights[warp][token] * KvarnTileTokenScale(payload_v, token);
#pragma unroll
          for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
            float rtn_scale = 0.0F;
            float rtn_zero = 0.0F;
            KvarnValueRtnPair(payload_v, token, vector, &rtn_scale, &rtn_zero);
            const std::uint32_t base = (lane + (vector * kLanes)) * 4;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              const std::uint32_t channel =
                  base + static_cast<std::uint32_t>(k);
              const float code =
                  __half2float(s_stage[(token - token0) * kHeadDim + channel]);
              const float channel_scale =
                  __half2float(reinterpret_cast<const __half*>(
                      payload_v + 4 * kHeadDim)[channel]);
              output[vector][k] = fmaf(
                  weighted, fmaf(code, rtn_scale, rtn_zero) * channel_scale,
                  output[vector][k]);
            }
          }
        }
      }
      __syncthreads();
    }
  }

  if (head_active) {
    const std::size_t stats_elements =
        static_cast<std::size_t>(num_heads) * max_splits * 2;
    const std::size_t partition_index =
        static_cast<std::size_t>(head) * max_splits + split;
    if (lane == 0) {
      scratch[partition_index * 2] = running_max;
      scratch[partition_index * 2 + 1] = running_sum;
    }
    auto* partial_outputs = reinterpret_cast<float4*>(scratch + stats_elements);
    float4* partial_output = partial_outputs + partition_index * kVectorWidth;
#pragma unroll
    for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
      float4 result;
      result.x = output[vector][0];
      result.y = output[vector][1];
      result.z = output[vector][2];
      result.w = output[vector][3];
      partial_output[lane + (vector * kLanes)] = result;
    }
  }
}

}  // namespace
}  // namespace gufo::hip

namespace {

namespace ref = gufo::models::qwen;

constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kLayers = 1;
constexpr std::uint32_t kKeys128K = 131072;
constexpr std::uint32_t kLayers128K = 16;
constexpr std::uint32_t kIterations = 400;
constexpr std::uint32_t kWarmup = 20;
// Measured FP16 KV read bandwidth of the same measurement series (design D18).
constexpr double kBandwidthGBs = 241.0;

float KeyValue(std::uint32_t position, std::uint32_t channel, bool is_value,
               std::uint32_t kv_head) {
  const float p = static_cast<float>(position);
  const float c = static_cast<float>(channel);
  const float h = static_cast<float>(kv_head);
  return is_value ? 0.25F * std::cos(p * 0.05F + c * 0.017F + h * 0.11F)
                  : 0.3F * std::sin(p * 0.07F + c * 0.013F + h * 0.07F);
}

float QueryValue(std::uint32_t head, std::uint32_t channel) {
  return 0.2F * std::sin(static_cast<float>(head) * 0.3F +
                         static_cast<float>(channel) * 0.021F);
}

std::size_t TilePayloadBytes(std::uint8_t bits) {
  return ref::KvarnTilePayloadBytes(bits);
}

/// One cache-resident full-attention layer of KVarN storage, sealed with the
/// CPU reference encoder so the read kernel exercises the quantized path.
struct KvarnFixture {
  std::uint32_t context{0};
  gufo::hip::QwenKvarnPreset preset{};
  std::uint32_t tail_rows{0};
  std::uint32_t ring_rows{0};
  std::uint32_t tile_count{0};
  std::size_t sink_plane{0};
  std::size_t tail_plane{0};
  std::size_t k_tile_bytes{0};
  std::size_t v_tile_bytes{0};
  void* sink{nullptr};
  void* tail{nullptr};
  std::uint8_t* body{nullptr};
  gufo::hip::KvarnReadParams read{};

  KvarnFixture(std::uint32_t context_tokens, std::uint8_t k_bits,
               std::uint8_t v_bits, std::uint32_t tail_tokens)
      : context(context_tokens) {
    preset.k_bits = k_bits;
    preset.v_bits = v_bits;
    preset.tail_length = tail_tokens;
    tail_rows = gufo::hip::QwenKvarnTailRows(preset);
    ring_rows = gufo::hip::QwenKvarnTailRingRows(preset);
    tile_count = (context - ref::kKvarnSinkRows + ref::kKvarnTileTokens - 1) /
                 ref::kKvarnTileTokens;
    k_tile_bytes = TilePayloadBytes(k_bits);
    v_tile_bytes = TilePayloadBytes(v_bits);
    sink_plane = static_cast<std::size_t>(kLayers) * kKvHeads *
                 ref::kKvarnSinkRows * kHeadDim;
    tail_plane =
        static_cast<std::size_t>(kLayers) * kKvHeads * ring_rows * kHeadDim;
    HIP_CHECK(hipMalloc(&sink, 2 * sink_plane * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&tail, 2 * tail_plane * sizeof(std::uint16_t)));
    void* body_raw = nullptr;
    HIP_CHECK(hipMalloc(&body_raw, static_cast<std::size_t>(kLayers) *
                                       kKvHeads * tile_count *
                                       (k_tile_bytes + v_tile_bytes)));
    body = static_cast<std::uint8_t*>(body_raw);

    // Tiles the view rule reads quantized for the query at `context - 1`.
    const std::uint32_t query = context - 1;
    const std::uint32_t quantized_tiles =
        query >= ref::kKvarnSinkRows + tail_rows + 1
            ? (query - ref::kKvarnSinkRows - tail_rows) / ref::kKvarnTileTokens
            : 0;
    const std::uint32_t first_f16 =
        ref::kKvarnSinkRows + ref::kKvarnTileTokens * quantized_tiles;

    // F16 sink and the unquantized tail rows; everything the query reads as
    // F16.
    std::vector<std::uint16_t> sink_host(2 * sink_plane, 0);
    std::vector<std::uint16_t> tail_host(2 * tail_plane, 0);
    const auto write_f16 = [&](std::uint32_t position) {
      for (std::uint32_t head = 0; head < kKvHeads; ++head) {
        for (const bool is_value : {false, true}) {
          for (std::uint32_t c = 0; c < kHeadDim; ++c) {
            const std::uint16_t encoded = __half_as_ushort(
                __float2half_rn(KeyValue(position, c, is_value, head)));
            std::size_t offset = 0;
            std::size_t plane = 0;
            if (position < ref::kKvarnSinkRows) {
              offset = ref::KvarnSinkRowOffset(0, kKvHeads, head, position);
              plane = is_value ? sink_plane : 0;
              sink_host[plane + offset + c] = encoded;
            } else {
              offset = ref::KvarnTailRowOffset(0, kKvHeads, head, position,
                                               ring_rows);
              plane = is_value ? tail_plane : 0;
              tail_host[plane + offset + c] = encoded;
            }
          }
        }
      }
    };
    for (std::uint32_t position = 0; position < ref::kKvarnSinkRows;
         ++position) {
      write_f16(position);
    }
    for (std::uint32_t position = first_f16; position < context; ++position) {
      write_f16(position);
    }
    HIP_CHECK(hipMemcpy(sink, sink_host.data(),
                        2 * sink_plane * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(tail, tail_host.data(),
                        2 * tail_plane * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));

    // Seal the quantized tiles with the CPU reference encoder.
    std::vector<float> tile(ref::kKvarnTileElements);
    std::vector<float> normalized(ref::kKvarnTileElements);
    for (std::uint32_t head = 0; head < kKvHeads; ++head) {
      for (const bool is_value : {false, true}) {
        for (std::uint32_t t = 0; t < quantized_tiles; ++t) {
          for (std::uint32_t token = 0; token < ref::kKvarnTileTokens;
               ++token) {
            const std::uint32_t position =
                ref::kKvarnSinkRows + ref::kKvarnTileTokens * t + token;
            float* row = tile.data() + token * ref::kKvarnTileChannels;
            for (std::uint32_t c = 0; c < ref::kKvarnTileChannels; ++c) {
              row[c] = KeyValue(position, c, is_value, head);
            }
            ref::KvarnFwhtHeadInPlace(
                std::span<float>(row, ref::kKvarnTileChannels));
          }
          ref::KvarnTileQuantResult result{};
          ref::KvarnQuantDiagnostics diag{};
          const auto status =
              ref::KvarnNormalizeTile(tile, normalized, &result, &diag);
          if (status != ref::KvarnQuantStatus::kOk) {
            throw std::runtime_error("reference tile normalization failed");
          }
          ref::KvarnQuantizeAsymmetricRtn(
              normalized,
              is_value ? ref::KvarnRtnAxis::kTokenGroup
                       : ref::KvarnRtnAxis::kChannel,
              is_value ? v_bits : k_bits, &result, &diag);
          ref::KvarnTilePayload payload{};
          ref::KvarnPackTilePayload(result, &payload);
          const std::size_t offset =
              ref::KvarnBodyTileOffset(0, kKvHeads, head, t, tile_count,
                                       k_tile_bytes, v_tile_bytes) +
              (is_value ? k_tile_bytes : 0);
          HIP_CHECK(hipMemcpy(body + offset, &payload,
                              is_value ? v_tile_bytes : k_tile_bytes,
                              hipMemcpyHostToDevice));
        }
      }
    }

    read.sink = static_cast<const __half*>(sink);
    read.tail = static_cast<const __half*>(tail);
    read.body = body;
    read.sink_plane_elements = sink_plane;
    read.tail_plane_elements = tail_plane;
    read.ring_rows = ring_rows;
    read.tile_count = tile_count;
    read.k_tile_bytes = static_cast<std::uint32_t>(k_tile_bytes);
    read.v_tile_bytes = static_cast<std::uint32_t>(v_tile_bytes);
    read.kv_heads = kKvHeads;
    read.tail_length = tail_rows;
    read.k_bits = k_bits;
    read.v_bits = v_bits;
  }

  ~KvarnFixture() {
    (void)hipFree(sink);
    (void)hipFree(tail);
    (void)hipFree(body);
  }
};

double TimeMs(const std::function<void()>& launch,
              std::uint32_t warmup = kWarmup,
              std::uint32_t iterations = kIterations) {
  for (std::uint32_t i = 0; i < warmup; ++i) {
    launch();
  }
  hipEvent_t begin{};
  hipEvent_t end{};
  HIP_CHECK(hipEventCreate(&begin));
  HIP_CHECK(hipEventCreate(&end));
  HIP_CHECK(hipEventRecord(begin));
  for (std::uint32_t i = 0; i < iterations; ++i) {
    launch();
  }
  HIP_CHECK(hipEventRecord(end));
  HIP_CHECK(hipEventSynchronize(end));
  float ms = 0.0F;
  HIP_CHECK(hipEventElapsedTime(&ms, begin, end));
  HIP_CHECK(hipEventDestroy(begin));
  HIP_CHECK(hipEventDestroy(end));
  return static_cast<double>(ms) / iterations;
}

/// Bytes of the whole 27B KVarN state at `capacity` tokens (design D5).
double KvarnBytes(std::uint8_t k_bits, std::uint8_t v_bits,
                  std::uint32_t tail_tokens, std::uint32_t capacity) {
  const std::uint32_t tail_rows =
      tail_tokens == 0 ? gufo::hip::kQwenKvarnW : tail_tokens;
  const std::uint32_t ring_rows =
      tail_rows + ref::kKvarnSinkRows + gufo::hip::kQwenKvarnW;
  const std::size_t tiles =
      (capacity - ref::kKvarnSinkRows + ref::kKvarnTileTokens - 1) /
      ref::kKvarnTileTokens;
  const std::size_t heads = static_cast<std::size_t>(kLayers128K) * kKvHeads;
  const std::size_t sink = heads * ref::kKvarnSinkRows * kHeadDim * 2 * 2;
  const std::size_t tail = heads * ring_rows * kHeadDim * 2 * 2;
  const std::size_t body =
      heads * tiles * (TilePayloadBytes(k_bits) + TilePayloadBytes(v_bits));
  return static_cast<double>(sink + tail + body);
}

int Run() {
  const std::uint32_t attention_width = kHeads * kHeadDim;
  std::vector<float> q(attention_width), gate(attention_width),
      out(attention_width);
  for (std::uint32_t i = 0; i < attention_width; ++i) {
    q[i] = 0.2F * std::sin(static_cast<float>(i) * 0.021F);
    gate[i] = 0.5F + 0.1F * std::cos(static_cast<float>(i) * 0.017F);
  }
  float* d_q = nullptr;
  float* d_gate = nullptr;
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_q, q.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, gate.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, out.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_q, q.data(), q.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, gate.data(), gate.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  const std::size_t row_scratch =
      gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kHeadDim);
  float* d_split = nullptr;
  HIP_CHECK(hipMalloc(&d_split, row_scratch * sizeof(float)));

  // FP16 reference for each context, laid out like the production layer.
  const auto time_fp16 = [&](std::uint32_t context) {
    const std::size_t total_plane =
        static_cast<std::size_t>(kLayers) * kKvHeads * context * kHeadDim;
    std::vector<std::uint16_t> host(2 * total_plane);
    for (std::uint32_t position = 0; position < context; ++position) {
      for (std::uint32_t head = 0; head < kKvHeads; ++head) {
        for (const bool is_value : {false, true}) {
          for (std::uint32_t c = 0; c < kHeadDim; ++c) {
            const std::size_t off =
                (is_value ? total_plane : 0) +
                (static_cast<std::size_t>(head) * context + position) *
                    kHeadDim +
                c;
            host[off] = __half_as_ushort(
                __float2half_rn(KeyValue(position, c, is_value, head)));
          }
        }
      }
    }
    __half* d_cache = nullptr;
    HIP_CHECK(hipMalloc(&d_cache, host.size() * sizeof(std::uint16_t)));
    HIP_CHECK(hipMemcpy(d_cache, host.data(),
                        host.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));
    const std::uint32_t position = context - 1;
    const double ms = TimeMs([&] {
      gufo::hip::LaunchAttention(d_q, nullptr, nullptr, d_gate, nullptr,
                                 nullptr, d_cache, d_cache + total_plane, d_out,
                                 0, position, context, kHeads, kKvHeads,
                                 kHeadDim, nullptr, d_split,
                                 /*skip_kv_write=*/true);
    });
    HIP_CHECK(hipFree(d_cache));
    return ms;
  };

  // A single small context is launch-overhead dominated, so sweep several
  // cache-resident contexts and least-squares fit time against the
  // (query head x key) work to get the per-unit compute cost.
  using gufo::hip::QwenKvarnSplitKReduceKernel;
  using gufo::hip::QwenKvarnTilePrototypeKernel;
  const auto fit = [](const std::vector<double>& units,
                      const std::vector<double>& ms) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    const double n = static_cast<double>(units.size());
    for (std::size_t i = 0; i < units.size(); ++i) {
      sx += units[i];
      sy += ms[i];
      sxx += units[i] * units[i];
      sxy += units[i] * ms[i];
    }
    const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const double intercept = (sy - slope * sx) / n;
    return std::pair<double, double>{slope, intercept};
  };

  const auto report = [&](const char* label, std::uint32_t base_context,
                          std::uint8_t k_bits, std::uint8_t v_bits,
                          std::uint32_t tail_tokens) {
    std::vector<double> units, fp16_ms, kvarn_ms, proto_ms;
    for (std::uint32_t step = 1; step <= 4; ++step) {
      const std::uint32_t context = base_context * step;
      const double work = static_cast<double>(kHeads) * context;
      fp16_ms.push_back(time_fp16(context));
      KvarnFixture fixture(context, k_bits, v_bits, tail_tokens);
      kvarn_ms.push_back(TimeMs([&] {
        gufo::hip::LaunchKvarnDecodeAttention(
            d_q, d_gate, d_out, 0, context - 1, nullptr, 1, kHeads, kKvHeads,
            fixture.read, d_split, nullptr);
      }));
      // Tile-unit prototype plus the production reduce, launched exactly as in
      // `--check` (design D4): the same split count the production read uses.
      const std::uint32_t splits =
          gufo::hip::detail::SelectDecodeAttentionSplitCount(context);
      proto_ms.push_back(TimeMs([&] {
        QwenKvarnTilePrototypeKernel<<<dim3(kKvHeads, splits),
                                       gufo::hip::kPrototypeThreads>>>(
            d_q, d_split, 0, context - 1, kHeads, kKvHeads, splits,
            fixture.read);
        QwenKvarnSplitKReduceKernel<<<dim3(kHeads, 1), gufo::hip::kLanes>>>(
            nullptr, d_split, d_out, kHeads, splits);
      }));
      units.push_back(work);
    }
    const auto [kvarn_slope_ms, kvarn_b] = fit(units, kvarn_ms);
    const auto [fp16_slope_ms, fp16_b] = fit(units, fp16_ms);
    const auto [proto_slope_ms, proto_b] = fit(units, proto_ms);
    const double total_units =
        static_cast<double>(kHeads) * kKeys128K * kLayers128K;
    const double kvarn_ns = kvarn_slope_ms * 1e6;
    const double fp16_ns = fp16_slope_ms * 1e6;
    const double proto_ns = proto_slope_ms * 1e6;
    const double compute_128k = kvarn_ns * total_units / 1e6;
    const double fp16_compute_128k = fp16_ns * total_units / 1e6;
    const double proto_compute_128k = proto_ns * total_units / 1e6;
    const double cap = kKeys128K + 64;
    const double kvarn_bytes = KvarnBytes(k_bits, v_bits, tail_tokens, cap);
    printf(
        "%s contexts=[%u..%u] kvarn=%.4f ns/unit (b=%.1f us) "
        "fp16=%.4f ns/unit (b=%.1f us)\n"
        "  compute_128k_ms=%.2f  fp16_compute_128k_ms=%.2f  "
        "bw_floor_ms=%.2f\n",
        label, base_context, base_context * 4, kvarn_ns, kvarn_b * 1e3, fp16_ns,
        fp16_b * 1e3, compute_128k, fp16_compute_128k,
        kvarn_bytes / kBandwidthGBs / 1e6);
    for (std::size_t i = 0; i < units.size(); ++i) {
      printf("    context=%u kvarn_ms=%.6f fp16_ms=%.6f\n",
             base_context * static_cast<std::uint32_t>(i + 1), kvarn_ms[i],
             fp16_ms[i]);
    }
    printf("  proto=%.4f ns/unit (b=%.1f us) compute_128k_ms=%.2f\n", proto_ns,
           proto_b * 1e3, proto_compute_128k);
  };

  // Base contexts: 128 sink + 128 token + 2048 fixed rows, scaled by the
  // sweep, so both presets span the same (4x) context range while the KVarN
  // body stays cache-resident.
  report("kvarn-k4v4-none", ref::kKvarnSinkRows + ref::kKvarnTileTokens + 2048,
         4, 4, 0);
  report("kvarn-k6v5-tail2048",
         ref::kKvarnSinkRows + ref::kKvarnTileTokens + 2048, 6, 5, 2048);

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_split));
  return 0;
}

/// FP64 reference for one query position of the check fixture: the same view
/// rule and decoded values as `Run`, with exact softmax arithmetic (modelled on
/// tests/models/qwen/hip/attention/kvarn_read_ops_test.cpp `Reference`).
std::vector<double> CheckReference(
    std::uint32_t pos, std::uint32_t head, std::uint32_t tail_rows,
    std::uint32_t tile_count, std::uint8_t k_bits, std::uint8_t v_bits,
    const std::vector<ref::KvarnTilePayload>& payload_k,
    const std::vector<ref::KvarnTilePayload>& payload_v) {
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  const std::uint32_t kv_head = head / (kHeads / kKvHeads);
  const auto quantized = [&](std::uint32_t position) {
    const double e_t =
        static_cast<double>(128 + 128 * ((position - 128) / 128) + 127);
    return position >= 128 && e_t + static_cast<double>(tail_rows) + 1.0 <=
                                  static_cast<double>(pos);
  };
  std::vector<double> scores(pos + 1);
  double running_max = -std::numeric_limits<double>::infinity();
  for (std::uint32_t position = 0; position <= pos; ++position) {
    double score = 0.0;
    if (!quantized(position)) {
      for (std::uint32_t c = 0; c < kHeadDim; ++c) {
        score += static_cast<double>(QueryValue(head, c)) *
                 static_cast<double>(__half2float(
                     __float2half_rn(KeyValue(position, c, false, kv_head))));
      }
    } else {
      const std::uint32_t tile = (position - 128) / 128;
      const std::uint32_t token = position - 128 - 128 * tile;
      const auto& payload = payload_k[kv_head * tile_count + tile];
      for (std::uint32_t c = 0; c < kHeadDim; ++c) {
        const double decoded = static_cast<double>(ref::KvarnDecodeTileElement(
            payload, ref::KvarnRtnAxis::kChannel, k_bits, token, c));
        const double channel_scale =
            __half2float(__ushort_as_half(payload.norm_channel_scale[c]));
        const double token_scale =
            __half2float(__ushort_as_half(payload.norm_token_scale[token]));
        score += static_cast<double>(QueryValue(head, c)) * decoded *
                 channel_scale * token_scale;
      }
    }
    scores[position] = score * static_cast<double>(scale);
    running_max = std::max(running_max, scores[position]);
  }
  double sum = 0.0;
  for (std::uint32_t position = 0; position <= pos; ++position) {
    scores[position] = std::exp(scores[position] - running_max);
    sum += scores[position];
  }
  std::vector<double> out(kHeadDim, 0.0);
  for (std::uint32_t position = 0; position <= pos; ++position) {
    const double weight = scores[position] / sum;
    if (!quantized(position)) {
      for (std::uint32_t c = 0; c < kHeadDim; ++c) {
        out[c] += weight * static_cast<double>(__half2float(__float2half_rn(
                               KeyValue(position, c, true, kv_head))));
      }
    } else {
      const std::uint32_t tile = (position - 128) / 128;
      const std::uint32_t token = position - 128 - 128 * tile;
      const auto& payload = payload_v[kv_head * tile_count + tile];
      for (std::uint32_t c = 0; c < kHeadDim; ++c) {
        const double decoded = static_cast<double>(ref::KvarnDecodeTileElement(
            payload, ref::KvarnRtnAxis::kTokenGroup, v_bits, token, c));
        const double channel_scale =
            __half2float(__ushort_as_half(payload.norm_channel_scale[c]));
        const double token_scale =
            __half2float(__ushort_as_half(payload.norm_token_scale[token]));
        out[c] += weight * decoded * channel_scale * token_scale;
      }
    }
  }
  std::vector<float> rotated(kHeadDim);
  for (std::uint32_t c = 0; c < kHeadDim; ++c) {
    rotated[c] = static_cast<float>(out[c]);
  }
  ref::KvarnFwhtHeadInPlace(rotated);
  for (std::uint32_t c = 0; c < kHeadDim; ++c) {
    out[c] = static_cast<double>(rotated[c]);
  }
  return out;
}

/// Runs the production read and the tile-unit prototype on one sealed fixture
/// and compares both against the FP64 reference. The declared limit is the one
/// the read-path ops test uses for the same FP32-kernel/FP64-reference gap.
int RunCheck() {
  constexpr std::uint32_t check_context = 384;
  constexpr std::uint8_t check_k_bits = 6;
  constexpr std::uint8_t check_v_bits = 5;
  constexpr double check_limit = 2e-4;
  const std::uint32_t pos = check_context - 1;

  KvarnFixture fixture(check_context, check_k_bits, check_v_bits, 0);
  const std::uint32_t tail = fixture.read.tail_length;
  const std::uint32_t tile_count = fixture.tile_count;
  const std::uint32_t quantized_tiles =
      pos >= ref::kKvarnSinkRows + tail + 1
          ? (pos - ref::kKvarnSinkRows - tail) / ref::kKvarnTileTokens
          : 0;

  std::vector<ref::KvarnTilePayload> payload_k(kKvHeads * tile_count);
  std::vector<ref::KvarnTilePayload> payload_v(kKvHeads * tile_count);
  for (std::uint32_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
    for (std::uint32_t tile = 0; tile < quantized_tiles; ++tile) {
      const std::size_t base = ref::KvarnBodyTileOffset(
          0, kKvHeads, kv_head, tile, tile_count, fixture.read.k_tile_bytes,
          fixture.read.v_tile_bytes);
      const std::size_t slot = kv_head * tile_count + tile;
      HIP_CHECK(hipMemcpy(&payload_k[slot], fixture.body + base,
                          fixture.read.k_tile_bytes, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(&payload_v[slot],
                          fixture.body + base + fixture.read.k_tile_bytes,
                          fixture.read.v_tile_bytes, hipMemcpyDeviceToHost));
    }
  }

  std::vector<float> q(static_cast<std::size_t>(kHeads) * kHeadDim);
  for (std::uint32_t head = 0; head < kHeads; ++head) {
    for (std::uint32_t c = 0; c < kHeadDim; ++c) {
      q[head * kHeadDim + c] = QueryValue(head, c);
    }
  }
  float* d_q = nullptr;
  float* d_prod = nullptr;
  float* d_proto = nullptr;
  HIP_CHECK(hipMalloc(&d_q, q.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_prod, q.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_proto, q.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_q, q.data(), q.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  const std::size_t scratch_elements =
      gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kHeadDim);
  float* d_split_prod = nullptr;
  float* d_split_proto = nullptr;
  HIP_CHECK(hipMalloc(&d_split_prod, scratch_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_split_proto, scratch_elements * sizeof(float)));

  const std::uint32_t splits =
      gufo::hip::detail::SelectDecodeAttentionSplitCount(check_context);
  gufo::hip::LaunchKvarnDecodeAttention(d_q, nullptr, d_prod, 0, pos, nullptr,
                                        1, kHeads, kKvHeads, fixture.read,
                                        d_split_prod, nullptr);
  gufo::hip::QwenKvarnTilePrototypeKernel<<<dim3(kKvHeads, splits),
                                            gufo::hip::kPrototypeThreads>>>(
      d_q, d_split_proto, 0, pos, kHeads, kKvHeads, splits, fixture.read);
  gufo::hip::
      QwenKvarnSplitKReduceKernel<<<dim3(kHeads, 1), gufo::hip::kLanes>>>(
          nullptr, d_split_proto, d_proto, kHeads, splits);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> prod(q.size());
  std::vector<float> proto(q.size());
  HIP_CHECK(hipMemcpy(prod.data(), d_prod, prod.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(proto.data(), d_proto, proto.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  double worst_prod = 0.0;
  double worst_proto = 0.0;
  bool pass = true;
  printf("check context=%u tail=%u quantized_tiles=%u splits=%u limit=%.1e\n",
         check_context, tail, quantized_tiles, splits, check_limit);
  for (std::uint32_t head = 0; head < kHeads; ++head) {
    const std::vector<double> expected =
        CheckReference(pos, head, tail, tile_count, check_k_bits, check_v_bits,
                       payload_k, payload_v);
    double head_prod = 0.0;
    double head_proto = 0.0;
    for (std::uint32_t c = 0; c < kHeadDim; ++c) {
      head_prod = std::max(
          head_prod, std::abs(static_cast<double>(prod[head * kHeadDim + c]) -
                              expected[c]));
      head_proto = std::max(
          head_proto, std::abs(static_cast<double>(proto[head * kHeadDim + c]) -
                               expected[c]));
    }
    worst_prod = std::max(worst_prod, head_prod);
    worst_proto = std::max(worst_proto, head_proto);
    pass = pass && head_prod <= check_limit && head_proto <= check_limit;
    printf("  head=%2u prod=%.3e proto=%.3e\n", head, head_prod, head_proto);
  }
  printf("%s max_prod=%.3e max_proto=%.3e\n", pass ? "PASS" : "FAIL",
         worst_prod, worst_proto);

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_prod));
  HIP_CHECK(hipFree(d_proto));
  HIP_CHECK(hipFree(d_split_prod));
  HIP_CHECK(hipFree(d_split_proto));
  return pass ? 0 : 1;
}

/// Streaming measurement (design D4/D5, task 5.3). One full-attention layer at
/// the 128K context: the KVarN body (~190 MiB) and the dense FP16 K/V planes
/// (~512 MiB) both exceed the SoC cache, so a decode step is dominated by KV
/// traffic and the prototype/current ratio is the number the go/no-go uses. The
/// two fixtures are built and freed in turn, so only one layer is ever
/// resident.
int RunStream() {
  constexpr std::uint32_t context = kKeys128K;
  constexpr std::uint8_t k_bits = 6;
  constexpr std::uint8_t v_bits = 5;
  constexpr std::uint32_t tail_tokens = 2048;
  constexpr std::uint32_t warmup = 5;
  constexpr std::uint32_t iterations = 20;
  const std::uint32_t pos = context - 1;

  const std::size_t attention_width =
      static_cast<std::size_t>(kHeads) * kHeadDim;
  std::vector<float> q(attention_width);
  for (std::uint32_t head = 0; head < kHeads; ++head) {
    for (std::uint32_t c = 0; c < kHeadDim; ++c) {
      q[head * kHeadDim + c] = QueryValue(head, c);
    }
  }
  float* d_q = nullptr;
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_q, q.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, q.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_q, q.data(), q.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  const std::size_t scratch_elements =
      gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kHeadDim);
  float* d_split = nullptr;
  HIP_CHECK(hipMalloc(&d_split, scratch_elements * sizeof(float)));
  const std::uint32_t splits =
      gufo::hip::detail::SelectDecodeAttentionSplitCount(context);

  double kvarn_ms = 0.0;
  double proto_ms = 0.0;
  {
    KvarnFixture fixture(context, k_bits, v_bits, tail_tokens);
    kvarn_ms = TimeMs(
        [&] {
          gufo::hip::LaunchKvarnDecodeAttention(d_q, nullptr, d_out, 0, pos,
                                                nullptr, 1, kHeads, kKvHeads,
                                                fixture.read, d_split, nullptr);
        },
        warmup, iterations);
    proto_ms = TimeMs(
        [&] {
          gufo::hip::QwenKvarnTilePrototypeKernel<<<
              dim3(kKvHeads, splits), gufo::hip::kPrototypeThreads>>>(
              d_q, d_split, 0, pos, kHeads, kKvHeads, splits, fixture.read);
          gufo::hip::QwenKvarnSplitKReduceKernel<<<dim3(kHeads, 1),
                                                   gufo::hip::kLanes>>>(
              nullptr, d_split, d_out, kHeads, splits);
        },
        warmup, iterations);
  }

  double fp16_ms = 0.0;
  const std::size_t total_plane =
      static_cast<std::size_t>(kLayers) * kKvHeads * context * kHeadDim;
  {
    std::vector<std::uint16_t> host(2 * total_plane);
    for (std::uint32_t position = 0; position < context; ++position) {
      for (std::uint32_t head = 0; head < kKvHeads; ++head) {
        for (const bool is_value : {false, true}) {
          for (std::uint32_t c = 0; c < kHeadDim; ++c) {
            const std::size_t off =
                (is_value ? total_plane : 0) +
                (static_cast<std::size_t>(head) * context + position) *
                    kHeadDim +
                c;
            host[off] = __half_as_ushort(
                __float2half_rn(KeyValue(position, c, is_value, head)));
          }
        }
      }
    }
    __half* d_cache = nullptr;
    HIP_CHECK(hipMalloc(&d_cache, host.size() * sizeof(std::uint16_t)));
    HIP_CHECK(hipMemcpy(d_cache, host.data(),
                        host.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));
    fp16_ms = TimeMs(
        [&] {
          gufo::hip::LaunchAttention(d_q, nullptr, nullptr, nullptr, nullptr,
                                     nullptr, d_cache, d_cache + total_plane,
                                     d_out, 0, pos, context, kHeads, kKvHeads,
                                     kHeadDim, nullptr, d_split,
                                     /*skip_kv_write=*/true);
        },
        warmup, iterations);
    HIP_CHECK(hipFree(d_cache));
  }

  const std::size_t kvarn_bytes =
      ref::ComputeKvarnRegions(k_bits, v_bits, tail_tokens,
                               gufo::hip::kQwenKvarnW, context, kLayers,
                               kKvHeads)
          .total_bytes;
  const std::size_t fp16_bytes = 2 * total_plane * sizeof(std::uint16_t);
  printf("stream context=%u tail=%u splits=%u warmup=%u iters=%u\n", context,
         tail_tokens, splits, warmup, iterations);
  printf("  kvarn_current   %8.4f ms  %7.1f MiB\n", kvarn_ms,
         static_cast<double>(kvarn_bytes) / 1048576.0);
  printf("  kvarn_prototype %8.4f ms  %7.1f MiB\n", proto_ms,
         static_cast<double>(kvarn_bytes) / 1048576.0);
  printf("  fp16            %8.4f ms  %7.1f MiB\n", fp16_ms,
         static_cast<double>(fp16_bytes) / 1048576.0);
  printf("  prototype/current=%.4f\n", proto_ms / kvarn_ms);

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_split));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "--check") == 0) {
    return RunCheck();
  }
  if (argc > 1 && std::strcmp(argv[1], "--stream") == 0) {
    return RunStream();
  }
  return Run();
}
