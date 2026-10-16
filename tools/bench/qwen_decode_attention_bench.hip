// Qwen3.8 27B decode/verification attention Phase 0 bench (openspec change
// update-qwen-decode-attention-blocks, tasks 1.1-1.4, design D1/D3/D5/D6/
// D10/D11/D15).
//
// Runs the same inputs through
//   * the control: a verbatim copy of the base revision's FP16 split-K
//     partials kernel and its launcher, and
//   * the candidate: one block per (KV head, row, partition), six GQA query
//     heads in the WMMA M dimension, each K/V tile staged into LDS once and
//     shared by all six heads, with the two partition methods (A/B, D11) and
//     with or without the Q/P high+residual split (D2),
// and reports time, FP64 max-absolute and RMS error, row independence and the
// resident sweep.
//
// Build and run (from the repository root):
//   nix develop -c tools/bench/build.sh tools/bench/qwen_decode_attention_bench.hip
//   /tmp/qwen_decode_attention_bench --mode cold --start-pos 131071 --rows 1
//   /tmp/qwen_decode_attention_bench --mode cold --start-pos 131071 --rows 8
//   /tmp/qwen_decode_attention_bench --mode accuracy
//   /tmp/qwen_decode_attention_bench --mode r1-check
//   /tmp/qwen_decode_attention_bench --mode resident
//
// Measurement procedure (design D6/D15/G0F):
//   * baseline: the revision this change branches off on
//     kvarn-kv-cache-quantization (`kBaseRevision` below); the control is the
//     verbatim copy of its FP16 split-K partials kernel,
//   * modes: `cold` times the given depth; `resident` sweeps 2304..9216
//     tokens (1 layer's KV stays in the SoC cache) and least-squares converts
//     the per-(query head, key) cost to 128K; `accuracy` compares against the
//     FP64 reference at series lengths 128/129/4096/4097/32768/131072 with the
//     non-saturating generation formula and with a 2^-8 scaling whose high +
//     residual split lands the residual in the f16 subnormal range;
//     `r1-check` compares every row of an R-row launch against the same row
//     launched alone,
//   * every timing is the median of `--iters` (default 5) launches after one
//     warm-up; accuracy is invariant across runs,
//   * the FP64 reference consumes exactly the FP16 values stored in the cache
//     and applies the gate in double,
//   * never overlap GPU jobs: run one configuration at a time, and not next to
//     the other model's measurements (design D7).
//
// The control copy is byte-identical to
//   git show 801abd3:src/models/qwen/hip/kernels/attention_decode.hip |
//     sed -n '624,794p;868,894p'
// and the two identifiers are renamed at inclusion so the production kernels
// link into the same binary; see tools/bench/detail/ for the file and the
// command that regenerates it.
//
// ---------------------------------------------------------------------- KVarN
// Phase 2 (design D14) adds the KVarN decode read at `--preset kvarn-k5v4`
// (K 5 bit, V 4 bit) and `--tail 2048` - the main configuration of the G0K
// gate. The control is a second verbatim copy, of the base revision's KVarN
// read kernels (`tools/bench/detail/qwen_kvarn_read_base_801abd3.inc`), and the
// candidate is one block per (KV head, partition) that expands the sealed tile
// once into LDS for all rows and all six GQA heads of the KV head, with a warp
// per row and the six query heads in the WMMA M dimension.
//
//   /tmp/qwen_decode_attention_bench --mode kvarn-cold
//   /tmp/qwen_decode_attention_bench --mode kvarn-accuracy --series 32768,131072
//   /tmp/qwen_decode_attention_bench --mode kvarn-r1-check --start-pos 131071
//
//   * sealing: one full-attention layer (4 KV heads) is sealed with the CPU
//     reference encoder of src/models/qwen/kvarn_reference.hpp. The F16 sink
//     holds positions [0, 128); every tile the *highest* row of the launch
//     reads quantized is sealed; the F16 tail ring holds the positions from the
//     *lowest* row's own view rule onward. A sub-launch that stays inside that
//     row group therefore reads exactly the bytes its own view rule selects.
//   * FP64 reference: the same view rule and the same sealed bytes, decoded
//     with the CPU reference (RTN pair, channel scale and token scale), an
//     exact double softmax, the weighted sum of the stored (rotated) V, then
//     the inverse Hadamard rotation and the SiLU gate.
//   * `kvarn-cold` reports R=1 and R=8 at every `--series` depth (default
//     32768 and 131072), for the control's decode and causal launchers and for
//     the candidate; every time is the median of `--iters` launches.
//   * `kvarn-accuracy` reports the max-absolute and RMS error of each row
//     against the FP64 reference at every `--series` depth.
//   * `kvarn-r1-check` compares every row of an R-row launch (R in 2, 3, 8)
//     against the same row launched alone, for both the control and the
//     candidate, at the shallow positions of
//     tests/models/qwen/hip/attention/kv_storage_ops_test.cpp and at
//     `--start-pos`; it exits non-zero when any row differs.
//   * VGPRs, occupancy and LDS bytes per kernel are in
//     /tmp/qwen_decode_attention_bench.res.txt (written by
//     tools/bench/build.sh -Rpass-analysis=kernel-resource-usage).
#define ENGINE_ENABLE_HIP 1
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/kernels/attention_decode.hip"
// `LaunchAttention` dispatches to the KVarN read kernels, so the unit has to
// define them too (the same include the KVarN compute bench uses).
#include "src/models/qwen/hip/kernels/kvarn_read.hip"

namespace gufo::hip {

// The base revision this bench's control was copied from (design D6: the
// comparison baseline is the revision the change branches off on
// kvarn-kv-cache-quantization; main and upstream/main do not carry KVarN).
constexpr char kBaseRevision[] = "801abd3397232321b5a1f6b0c074460b60255988";

// Verbatim control. Included with the two identifiers renamed so the copy keeps
// the base file's text unchanged.
#define QwenDecodeSplitKAttentionHalfPartialsKernel FrozenBaseHalfPartialsKernel
#define LaunchSplitKAttentionHalfPartials LaunchFrozenBaseHalfPartials
#include "tools/bench/detail/qwen_decode_attention_base_801abd3.inc"
#undef QwenDecodeSplitKAttentionHalfPartialsKernel
#undef LaunchSplitKAttentionHalfPartials

// Verbatim KVarN read control. The copy renames its own eight top-level
// identifiers to the same `Base*` names, so it is included as is; the
// production kernels `kvarn_read.hip` defines above stay untouched.
#include "tools/bench/detail/qwen_kvarn_read_base_801abd3.inc"

namespace {

// ---------------------------------------------------------------- candidate
constexpr std::uint32_t kCandHeadDim = 256;
constexpr std::uint32_t kCandGqa = 6;
constexpr std::uint32_t kCandStageTokens = 16;
constexpr std::uint32_t kCandWarps = 8;
constexpr std::uint32_t kCandThreads = 32 * kCandWarps;
constexpr std::uint32_t kCandTileTokens = 128;
constexpr std::uint32_t kCandSplits = detail::kSplitKDecodeAttentionMaxSplits;
constexpr std::uint32_t kCandDimChunks = kCandHeadDim / 16;
constexpr std::uint32_t kCandChunksPerWarp = kCandDimChunks / kCandWarps;
constexpr std::uint32_t kCandKStride = kCandHeadDim + 8;
constexpr std::uint32_t kCandVStride = kCandStageTokens + 8;
constexpr std::uint32_t kCandPStride = kCandStageTokens + 8;
constexpr std::uint32_t kCandMaxContext = 262144;
/// The high+residual split carries the residual at 2^11 so it stays a normal
/// f16 for any input above the subnormal range (design D2/Risks: an unscaled
/// residual falls into the f16 subnormal range, where WMMA inputs may be
/// flushed and the split buys nothing).
constexpr float kResidualScale = 2048.0F;
constexpr float kResidualScaleInverse = 1.0F / 2048.0F;
/// Second residual level (2^22) for the three-term P split probe.
constexpr bool kExplicitAccumulate = true;
constexpr float kResidualScale2 = 4194304.0F;
constexpr float kResidualScale2Inverse = 1.0F / 4194304.0F;

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f CandWmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ v16h CandZeroFrag() {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = make_uint4(0u, 0u, 0u, 0u);
  cvt.u[1] = make_uint4(0u, 0u, 0u, 0u);
  return cvt.f;
}

__device__ __forceinline__ v16h CandLoadFrag(const __half* p) {
  union {
    v16h f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

/// A-side fragment rows are 2i + lane/16, so this lane's eight accumulator
/// elements carry the rescale of query head 2i + lane/16; padding rows (>= the
/// six GQA heads) stay at 1.0 because their P rows are zero.
__device__ __forceinline__ v8f CandRescaleVector(const float* row_rescale,
                                                 std::uint32_t lane) {
  v8f scale;
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    const std::uint32_t row = (2 * i) + (lane / 16);
    scale[i] =
        row < kCandGqa ? row_rescale[row] : 1.0F;
  }
  return scale;
}

/// One block per (KV head, row, partition). `partition_length != 0` selects
/// method A (partitions of the launch's aligned length L), 0 selects method B
/// (128-token tiles spread over `kCandSplits` partitions, decided per row).
template<bool kSplitQ, int kPSplitTerms, bool kNormalizedAcc>
__launch_bounds__(kCandThreads) __global__
    void QwenDecodeBlockAttentionPartialsKernel(
        const float* __restrict__ q, const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache, float* __restrict__ scratch,
        std::uint32_t layer_idx, std::uint32_t start_pos,
        std::uint32_t row_base, std::uint32_t max_context,
        std::uint32_t num_heads, std::uint32_t num_kv_heads,
        std::uint32_t split_count, std::uint32_t partition_length) {
  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t row = row_base + blockIdx.y;
  const std::uint32_t split = blockIdx.z;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid % 32;
  const std::uint32_t warp = tid / 32;
  if (kv_head >= num_kv_heads || split >= split_count) {
    return;
  }

  const std::uint32_t sequence_length = start_pos + row + 1;
  std::uint32_t partition_begin = 0;
  std::uint32_t partition_end = 0;
  if (partition_length != 0) {
    partition_begin = split * partition_length;
    partition_end = partition_begin + partition_length;
  } else {
    const std::uint32_t blocks =
        (sequence_length + kCandTileTokens - 1) / kCandTileTokens;
    partition_begin =
        (split * blocks / kCandSplits) * kCandTileTokens;
    partition_end =
        ((split + 1) * blocks / kCandSplits) * kCandTileTokens;
  }
  const bool empty =
      partition_begin >= sequence_length || partition_begin >= partition_end;
  partition_end = empty ? partition_begin
                        : min(partition_end, sequence_length);

  float* const row_scratch =
      scratch + (static_cast<std::size_t>(row) * num_heads *
                 detail::kSplitKDecodeAttentionMaxSplits *
                 (kCandHeadDim + 2));
  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * kCandSplits * 2;
  auto* const partial_outputs =
      reinterpret_cast<float*>(row_scratch + stats_elements);

  __shared__ __align__(16) __half k_lds[kCandStageTokens * kCandKStride];
  __shared__ __align__(16) __half v_lds[kCandHeadDim * kCandVStride];
  __shared__ __align__(16) __half p_hi[kCandGqa * kCandPStride];
  __shared__ __align__(16) __half p_mid[kCandGqa * kCandPStride];
  __shared__ __align__(16) __half p_lo[kCandGqa * kCandPStride];
  __shared__ float s_lds[kCandWarps][kCandGqa][kCandStageTokens];
  __shared__ float row_max[kCandGqa];
  __shared__ float row_sum[kCandGqa];
  __shared__ float row_rescale[kCandGqa];

  if (empty) {
    const std::size_t head_base =
        static_cast<std::size_t>(kv_head) * kCandGqa;
    if (tid < kCandGqa) {
      const std::size_t index =
          ((head_base + tid) * kCandSplits) + split;
      row_scratch[index * 2] = -INFINITY;
      row_scratch[index * 2 + 1] = 0.0F;
    }
    for (std::size_t i = tid; i < kCandGqa * kCandHeadDim;
         i += kCandThreads) {
      const std::size_t head = i / kCandHeadDim;
      const std::size_t dim = i % kCandHeadDim;
      const std::size_t index = ((head_base + head) * kCandSplits) + split;
      partial_outputs[(index * kCandHeadDim) + dim] = 0.0F;
    }
    return;
  }

  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * kCandHeadDim;
  const std::size_t head_offset =
      (static_cast<std::size_t>(layer_idx) * max_context * kv_width) +
      (static_cast<std::size_t>(kv_head) * kCandHeadDim);
  const __half* const k_base = k_cache + head_offset;
  const __half* const v_base = v_cache + head_offset;
  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t head_begin = kv_head * group_size;
  const float scale = rsqrtf(static_cast<float>(kCandHeadDim));

  // Q high + residual fragments straight from global memory into registers.
  // The A-side fragment row for a lane is GQA query head (lane % 16), so only
  // the six real heads load and the padded rows stay zero.
  v16h q_frag_hi[kCandChunksPerWarp];
  v16h q_frag_lo[kCandChunksPerWarp];
  {
    const std::uint32_t head_row = lane & 15u;
    const float* const base =
        q + (static_cast<std::size_t>(row) * num_heads * kCandHeadDim) +
        (static_cast<std::size_t>(head_begin + head_row) * kCandHeadDim);
#pragma unroll
    for (std::uint32_t c = 0; c < kCandChunksPerWarp; ++c) {
      const std::uint32_t chunk = (warp * kCandChunksPerWarp) + c;
      if (head_row < kCandGqa) {
        union {
          v16h f;
          uint4 u[2];
        } high;
        union {
          v16h f;
          uint4 u[2];
        } low;
        const auto* const source =
            reinterpret_cast<const float*>(base + (chunk * 16));
#pragma unroll
        for (std::uint32_t half = 0; half < 2; ++half) {
          uint4 packed_high;
          uint4 packed_low;
          auto* const high_words = reinterpret_cast<__half*>(&packed_high);
          auto* const low_words = reinterpret_cast<__half*>(&packed_low);
          // One uint4 holds eight halves, so a 16-half fragment packs as two
          // uint4 words of eight heads' elements each.
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i) {
            const float value = source[(half * 8) + i] * scale;
            const __half high_value = __float2half_rn(value);
            high_words[i] = high_value;
            low_words[i] = kSplitQ ? __float2half_rn(
                                         (value - __half2float(high_value)) *
                                         kResidualScale)
                                   : __float2half_rn(0.0F);
          }
          high.u[half] = packed_high;
          low.u[half] = packed_low;
        }
        q_frag_hi[c] = high.f;
        q_frag_lo[c] = low.f;
      } else {
        q_frag_hi[c] = CandZeroFrag();
        q_frag_lo[c] = CandZeroFrag();
      }
    }
  }
  if (tid < kCandGqa) {
    row_max[tid] = -INFINITY;
    row_sum[tid] = 0.0F;
    row_rescale[tid] = 0.0F;
  }
  __syncthreads();


  v8f output_acc[kCandChunksPerWarp];
#pragma unroll
  for (std::uint32_t c = 0; c < kCandChunksPerWarp; ++c) {
    output_acc[c] = v8f{};
  }

  // Software pipeline: a stage's K/V sits in registers while the previous
  // stage is computed, so the global loads' latency is hidden behind the
  // S/softmax/PV work instead of being exposed in front of a barrier.
  uint4 k_regs[2];
  std::uint32_t v_regs[8];
  const auto load_stage = [&](std::uint32_t begin, std::uint32_t count,
                              uint4 (&k_out)[2],
                              std::uint32_t (&v_out)[8]) {
    {
      // One 16-half chunk of one token per thread: 256 chunks of 32 bytes.
      const std::uint32_t token = tid / 16;
      const std::uint32_t dim = (tid % 16) * 16;
      if (token < count) {
        const auto* const source = reinterpret_cast<const uint4*>(
            k_base + ((begin + token) * kv_width) + dim);
        k_out[0] = source[0];
        k_out[1] = source[1];
      } else {
        k_out[0] = make_uint4(0u, 0u, 0u, 0u);
        k_out[1] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    {
      // Two dimensions by eight tokens per thread, covering all 16 tokens of
      // a dimension pair with two threads.
      const std::uint32_t dim = 2 * (tid / 2);
      const std::uint32_t token_begin = (tid % 2) * 8;
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t token = token_begin + i;
        v_out[i] =
            token < count
                ? *reinterpret_cast<const std::uint32_t*>(
                      v_base + ((begin + token) * kv_width) + dim)
                : 0U;
      }
    }
  };
  const auto store_stage = [&](uint4 (&k_in)[2],
                               std::uint32_t (&v_in)[8]) {
    {
      const std::uint32_t token = tid / 16;
      const std::uint32_t dim = (tid % 16) * 16;
      auto* const target =
          reinterpret_cast<uint4*>(&k_lds[(token * kCandKStride) + dim]);
      target[0] = k_in[0];
      target[1] = k_in[1];
    }
    {
      const std::uint32_t dim = 2 * (tid / 2);
      const std::uint32_t token_begin = (tid % 2) * 8;
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        v_lds[((dim + 0) * kCandVStride) + token_begin + i] =
            __ushort_as_half(static_cast<unsigned short>(v_in[i] & 0xFFFFU));
        v_lds[((dim + 1) * kCandVStride) + token_begin + i] =
            __ushort_as_half(static_cast<unsigned short>(v_in[i] >> 16));
      }
    }
  };
  load_stage(partition_begin,
             min(kCandStageTokens, partition_end - partition_begin), k_regs,
             v_regs);
  for (std::uint32_t stage_begin = partition_begin;
       stage_begin < partition_end; stage_begin += kCandStageTokens) {
    const std::uint32_t stage_count =
        min(kCandStageTokens, partition_end - stage_begin);
    __syncthreads();
    // Every element is read from global memory once and written to LDS once,
    // then shared by the six query heads.
    store_stage(k_regs, v_regs);
    __syncthreads();
    const std::uint32_t next_begin = stage_begin + kCandStageTokens;
    if (next_begin < partition_end) {
      load_stage(next_begin, min(kCandStageTokens, partition_end - next_begin),
                 k_regs, v_regs);
    }

    // S = Q K^T over this warp's 16-dim chunks.
    v8f scores = v8f{};
#pragma unroll
    for (std::uint32_t c = 0; c < kCandChunksPerWarp; ++c) {
      const std::uint32_t chunk = (warp * kCandChunksPerWarp) + c;
      const v16h key_frag =
          CandLoadFrag(&k_lds[((lane & 15u) * kCandKStride) + (chunk * 16)]);
      scores = CandWmma(q_frag_hi[c], key_frag, scores);
      if constexpr (kSplitQ) {
        const v8f residual = CandWmma(q_frag_lo[c], key_frag, v8f{});
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          scores[i] = fmaf(residual[i], kResidualScaleInverse, scores[i]);
        }
      }
    }
    {
      const std::uint32_t head_row = lane / 16;
      const std::uint32_t token = lane & 15u;
#pragma unroll
      for (std::uint32_t i = 0; i < kCandGqa / 2; ++i) {
        s_lds[warp][(2 * i) + head_row][token] = scores[i];
      }
    }
    __syncthreads();

    // Online softmax: heads 2w and 2w+1 share a warp, and the 16 lanes of a
    // half wave reduce the 16 staged tokens. Every accumulation outside the
    // WMMA is an explicit fmaf.
    if (warp < kCandGqa / 2) {
      const std::uint32_t head_row = (2 * warp) + (lane / 16);
      const std::uint32_t token = lane & 15u;
      const float masked_weight =
          (token < stage_count) ? 0.0F : -INFINITY;
      float score = masked_weight;
#pragma unroll
      for (std::uint32_t slot = 0; slot < kCandWarps; ++slot) {
        score += s_lds[slot][head_row][token];
      }
      float tile_max = score;
#pragma unroll
      for (int offset = 8; offset > 0; offset /= 2) {
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, offset));
      }
      const float previous_max = row_max[head_row];
      const float stage_max = fmaxf(previous_max, tile_max);
      const float rescale =
          isfinite(previous_max) ? expf(previous_max - stage_max) : 0.0F;
      float weight = 0.0F;
      if (token < stage_count) {
        weight = expf(score - stage_max);
      }
      float part_sum = weight;
#pragma unroll
      for (int offset = 8; offset > 0; offset /= 2) {
        part_sum = fmaf(1.0F, part_sum, __shfl_xor(part_sum, offset));
      }
      // Normalized accumulation (probe): the PV accumulator tracks the
      // partition average instead of the unnormalized numerator, so its
      // roundings happen at the output's magnitude; the partial written for
      // the unchanged reduce is scaled back by the final sum.
      const float previous_sum = row_sum[head_row];
      const float carried = previous_sum * rescale;
      const float sum_new = carried + part_sum;
      const float inverse_sum = (kNormalizedAcc && sum_new > 0.0F)
                                    ? 1.0F / sum_new
                                    : 1.0F;
      const float scaled_weight = kNormalizedAcc ? weight * inverse_sum
                                                 : weight;
      const __half weight_high = __float2half_rn(scaled_weight);
      p_hi[(head_row * kCandPStride) + token] = weight_high;
      const float residual =
          scaled_weight - __half2float(weight_high);
      const __half mid_high = __float2half_rn(residual * kResidualScale);
      p_mid[(head_row * kCandPStride) + token] =
          kPSplitTerms >= 3 ? mid_high : __float2half_rn(0.0F);
      p_lo[(head_row * kCandPStride) + token] =
          kPSplitTerms >= 3
              ? __float2half_rn((residual - (__half2float(mid_high) *
                                             kResidualScaleInverse)) *
                                kResidualScale2)
              : kPSplitTerms >= 2
                    ? __float2half_rn(residual * kResidualScale)
                    : __float2half_rn(0.0F);
      if ((lane & 15u) == 0) {
        row_rescale[head_row] =
            kNormalizedAcc ? carried * inverse_sum : rescale;
        row_max[head_row] = stage_max;
        row_sum[head_row] = sum_new;
      }
    }
    __syncthreads();

    // O += P V for this warp's output chunks. The probability fragments do not
    // depend on the output dimension, so they are loaded once per stage. The
    // WMMA's own accumulate path rounds (and truncates) the incoming
    // accumulator, so each product is computed into a zero accumulator and
    // folded with one round-to-nearest fmaf that also applies the running
    // softmax rescale.
    const std::uint32_t probability_row = lane & 15u;
    v16h probabilities = CandZeroFrag();
    v16h residual_probabilities = CandZeroFrag();
    v16h middle_probabilities = CandZeroFrag();
    if (probability_row < kCandGqa) {
      probabilities = CandLoadFrag(&p_hi[(probability_row * kCandPStride)]);
      if constexpr (kPSplitTerms >= 2) {
        residual_probabilities =
            CandLoadFrag(&p_lo[(probability_row * kCandPStride)]);
      }
      if constexpr (kPSplitTerms >= 3) {
        middle_probabilities =
            CandLoadFrag(&p_mid[(probability_row * kCandPStride)]);
      }
    }
    const v8f rescale_vector = CandRescaleVector(row_rescale, lane);
#pragma unroll
    for (std::uint32_t c = 0; c < kCandChunksPerWarp; ++c) {
      const std::uint32_t chunk = (warp * kCandChunksPerWarp) + c;
      const v16h value_frag =
          CandLoadFrag(&v_lds[((chunk * 16) + probability_row) * kCandVStride]);
      if (kExplicitAccumulate) {
        const v8f product = CandWmma(probabilities, value_frag, v8f{});
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          output_acc[c][i] =
              fmaf(output_acc[c][i], rescale_vector[i], product[i]);
        }
      } else {
        v8f rescaled;
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          rescaled[i] = fmaf(output_acc[c][i], rescale_vector[i], 0.0F);
        }
        output_acc[c] = CandWmma(probabilities, value_frag, rescaled);
      }
      if constexpr (kPSplitTerms >= 2) {
        const v8f scaled = CandWmma(residual_probabilities, value_frag, v8f{});
        const float scale = kPSplitTerms >= 3 ? kResidualScale2Inverse
                                              : kResidualScaleInverse;
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          output_acc[c][i] = fmaf(scaled[i], scale, output_acc[c][i]);
        }
      }
      if constexpr (kPSplitTerms >= 3) {
        const v8f scaled = CandWmma(middle_probabilities, value_frag, v8f{});
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          output_acc[c][i] =
              fmaf(scaled[i], kResidualScaleInverse, output_acc[c][i]);
        }
      }
    }
  }

  __syncthreads();
  if ((warp < kCandGqa / 2) && ((lane & 15u) == 0)) {
    const std::uint32_t head = (2 * warp) + (lane / 16);
    const std::size_t index =
        (static_cast<std::size_t>(head_begin + head) * kCandSplits) + split;
    row_scratch[index * 2] = row_max[head];
    row_scratch[index * 2 + 1] = row_sum[head];
  }
#pragma unroll
  for (std::uint32_t c = 0; c < kCandChunksPerWarp; ++c) {
    const std::uint32_t chunk = (warp * kCandChunksPerWarp) + c;
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      const std::uint32_t head = (2 * i) + (lane / 16);
      if (head < kCandGqa) {
        const std::size_t index =
            (static_cast<std::size_t>(head_begin + head) * kCandSplits) +
            split;
        const std::uint32_t dim = (chunk * 16) + (lane & 15u);
        const float normalized =
            kNormalizedAcc ? output_acc[c][i] * row_sum[head] : 0.0F;
        partial_outputs[(index * kCandHeadDim) + dim] =
            kNormalizedAcc ? normalized : output_acc[c][i];
      }
    }
  }
}

/// Method A partition length: round_up(ceil(n / 32), 128), from n alone
/// (design D11).
std::uint32_t MethodPartitionLength(std::uint32_t sequence_length) {
  const std::uint32_t per_split =
      (sequence_length + kCandSplits - 1) / kCandSplits;
  return std::max<std::uint32_t>(((per_split + kCandTileTokens - 1) /
                                  kCandTileTokens) *
                                     kCandTileTokens,
                                 kCandTileTokens);
}

/// Candidate launcher: groups rows by method A partition length (method A) or
/// runs every row in one launch (method B), then reduces with the production
/// `QwenDecodeSplitKAttentionReduceKernel`.
template<bool kSplitQ, int kPSplitTerms, bool kNormalizedAcc = false>
void LaunchCandidate(const float* q, const float* gate, const __half* k_cache,
                     const __half* v_cache, float* out_context,
                     std::uint32_t layer_idx, std::uint32_t start_pos,
                     std::uint32_t rows, std::uint32_t max_context,
                     std::uint32_t num_heads, std::uint32_t num_kv_heads,
                     bool method_b, float* scratch, hipStream_t stream) {
  if (method_b) {
    QwenDecodeBlockAttentionPartialsKernel<kSplitQ, kPSplitTerms,
                                           kNormalizedAcc>
        <<<dim3(num_kv_heads, rows, kCandSplits), kCandThreads, 0, stream>>>(
            q, k_cache, v_cache, scratch, layer_idx, start_pos, 0, max_context,
            num_heads, num_kv_heads, kCandSplits, 0);
    QwenDecodeSplitKAttentionReduceKernel<<<dim3(num_heads, rows), 32, 0,
                                            stream>>>(
        gate, scratch, out_context, num_heads, kCandSplits);
    return;
  }
  std::uint32_t row = 0;
  while (row < rows) {
    const std::uint32_t length =
        MethodPartitionLength(start_pos + row + 1);
    std::uint32_t end = row + 1;
    while (end < rows && MethodPartitionLength(start_pos + end + 1) == length) {
      ++end;
    }
    // All 32 partitions are launched so every scratch slot is written; the
    // partitions past a row's count mark themselves empty.
    QwenDecodeBlockAttentionPartialsKernel<kSplitQ, kPSplitTerms,
                                           kNormalizedAcc>
        <<<dim3(num_kv_heads, end - row, kCandSplits), kCandThreads, 0,
           stream>>>(q, k_cache, v_cache, scratch, layer_idx, start_pos, row,
                     max_context, num_heads, num_kv_heads, kCandSplits,
                     length);
    row = end;
  }
  QwenDecodeSplitKAttentionReduceKernel<<<dim3(num_heads, rows), 32, 0,
                                          stream>>>(
      gate, scratch, out_context, num_heads, kCandSplits);
}

/// Probe split modes: bit 0 splits Q, bit 1 splits P.
void LaunchCandidateMode(int mode, const float* q, const float* gate,
                         const __half* k_cache, const __half* v_cache,
                         float* out_context, std::uint32_t layer_idx,
                         std::uint32_t start_pos, std::uint32_t rows,
                         std::uint32_t max_context, std::uint32_t num_heads,
                         std::uint32_t num_kv_heads, bool method_b,
                         float* scratch, hipStream_t stream) {
  switch (mode) {
    case 0:
      LaunchCandidate<false, 0>(q, gate, k_cache, v_cache, out_context,
                                layer_idx, start_pos, rows, max_context,
                                num_heads, num_kv_heads, method_b, scratch,
                                stream);
      break;
    case 1:
      LaunchCandidate<true, 0>(q, gate, k_cache, v_cache, out_context,
                               layer_idx, start_pos, rows, max_context,
                               num_heads, num_kv_heads, method_b, scratch,
                               stream);
      break;
    case 2:
      LaunchCandidate<false, 2>(q, gate, k_cache, v_cache, out_context,
                                layer_idx, start_pos, rows, max_context,
                                num_heads, num_kv_heads, method_b, scratch,
                                stream);
      break;
    case 5:
      LaunchCandidate<true, 2, true>(q, gate, k_cache, v_cache, out_context,
                                     layer_idx, start_pos, rows, max_context,
                                     num_heads, num_kv_heads, method_b, scratch,
                                     stream);
      break;
    case 7:
      // Production kernel (design D1/D11): the FP16 block kernel from
      // kernels/attention_decode.hip plus the unchanged production reduce.
      gufo::hip::QwenDecodeBlockAttentionPartialsKernel<<<
          dim3(num_kv_heads, rows, kCandSplits), 256, 0, stream>>>(
          q, k_cache, v_cache, scratch, layer_idx, start_pos, max_context,
          num_heads, num_kv_heads);
      QwenDecodeSplitKAttentionReduceKernel<<<dim3(num_heads, rows), 32, 0,
                                              stream>>>(
          gate, scratch, out_context, num_heads, kCandSplits);
      break;
    case 6:
      LaunchCandidate<false, 2, true>(q, gate, k_cache, v_cache, out_context,
                                      layer_idx, start_pos, rows, max_context,
                                      num_heads, num_kv_heads, method_b,
                                      scratch, stream);
      break;
    case 4:
      LaunchCandidate<true, 3>(q, gate, k_cache, v_cache, out_context,
                               layer_idx, start_pos, rows, max_context,
                               num_heads, num_kv_heads, method_b, scratch,
                               stream);
      break;
    default:
      LaunchCandidate<true, 2>(q, gate, k_cache, v_cache, out_context,
                               layer_idx, start_pos, rows, max_context,
                               num_heads, num_kv_heads, method_b, scratch,
                               stream);
      break;
  }
}

/// The frozen control path (design D6/D12): the base revision's split-K
/// partials kernel and its launcher, plus the unchanged production reduce.
/// The public launch functions cannot be used for the control any more once
/// the candidate replaces the FP16 decode path.
void LaunchFrozenControl(const float* q, const float* gate,
                         const __half* k_cache, const __half* v_cache,
                         float* out_context, std::uint32_t layer_idx,
                         std::uint32_t start_pos, std::uint32_t rows,
                         std::uint32_t max_context, std::uint32_t num_heads,
                         std::uint32_t num_kv_heads, float* scratch,
                         hipStream_t stream) {
  const std::uint32_t splits =
      detail::SelectDecodeAttentionSplitCount(start_pos + 1);
  LaunchFrozenBaseHalfPartials(q, k_cache, v_cache, scratch, layer_idx,
                               start_pos, rows, max_context, num_heads,
                               num_kv_heads, splits, stream);
  QwenDecodeSplitKAttentionReduceKernel<<<dim3(num_heads, rows), 32, 0,
                                          stream>>>(
      gate, scratch, out_context, num_heads, splits);
}

// ------------------------------------------------------------------ harness
struct Options {
  std::string mode{"cold"};
  std::uint32_t rows{1};
  std::uint32_t start_pos{131071};
  bool method_b{false};
  int split_mode{3};
  bool control{true};
  bool candidate{true};
  int iters{5};
  int input{0};
  int input_filter{-1};
  std::vector<std::uint32_t> series_filter;
  std::uint8_t kvarn_k_bits{5};
  std::uint8_t kvarn_v_bits{4};
  std::uint32_t kvarn_tail{2048};
  std::uint32_t sweep_lo{2304};
  std::uint32_t sweep_hi{9216};
  std::uint32_t sweep_step{1152};
};

struct Timing {
  double median_ms{0.0};
  std::vector<double> samples;
};

[[nodiscard]] double MedianOf(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

template<typename Launch>
Timing TimeLaunch(Launch&& launch, int iters) {
  launch();
  HIP_CHECK(hipDeviceSynchronize());
  Timing timing;
  for (int i = 0; i < iters; ++i) {
    hipEvent_t begin = nullptr;
    hipEvent_t end = nullptr;
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&end));
    HIP_CHECK(hipEventRecord(begin));
    launch();
    HIP_CHECK(hipEventRecord(end));
    HIP_CHECK(hipEventSynchronize(end));
    float elapsed = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&elapsed, begin, end));
    HIP_CHECK(hipEventDestroy(begin));
    HIP_CHECK(hipEventDestroy(end));
    timing.samples.push_back(elapsed);
  }
  timing.median_ms = MedianOf(timing.samples);
  return timing;
}

/// FP64 oracle over the FP16 cache values, mirroring
/// tests/models/qwen/hip/attention/long_context_ops_test.cpp.
void ReferenceForRow(const std::vector<float>& query,
                     const std::vector<float>& gate_values,
                     const std::vector<__half>& cache,
                     std::uint32_t layer, std::uint32_t position,
                     std::uint32_t max_context, std::uint32_t num_heads,
                     std::uint32_t num_kv_heads,
                     std::vector<double>& reference) {
  const std::uint32_t dim = kCandHeadDim;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * dim;
  const std::size_t tokens = static_cast<std::size_t>(position) + 1;
  const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
  reference.assign(static_cast<std::size_t>(num_heads) * dim, 0.0);
  std::vector<double> scores(tokens);
  for (std::uint32_t head = 0; head < num_heads; ++head) {
    const std::uint32_t kv_head = head / (num_heads / num_kv_heads);
    const __half* const cache_head =
        cache.data() +
        (static_cast<std::size_t>(layer) * max_context * kv_width) +
        (static_cast<std::size_t>(kv_head) * dim);
    const float* const query_head =
        query.data() + (static_cast<std::size_t>(head) * dim);
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t token = 0; token < tokens; ++token) {
      double dot = 0.0;
      const __half* const key = cache_head + (token * kv_width);
      for (std::uint32_t d = 0; d < dim; ++d) {
        dot += static_cast<double>(query_head[d]) *
               static_cast<double>(__half2float(key[d]));
      }
      scores[token] = dot * scale;
      maximum = std::max(maximum, scores[token]);
    }
    double denominator = 0.0;
    for (std::size_t token = 0; token < tokens; ++token) {
      scores[token] = std::exp(scores[token] - maximum);
      denominator += scores[token];
    }
    for (std::uint32_t d = 0; d < dim; ++d) {
      double weighted = 0.0;
      for (std::size_t token = 0; token < tokens; ++token) {
        weighted += scores[token] * static_cast<double>(__half2float(
                                        cache_head[(token * kv_width) + d]));
      }
      const double gated =
          1.0 / (1.0 + std::exp(-static_cast<double>(
                                   gate_values[(static_cast<std::size_t>(head) *
                                                 dim) +
                                                d])));
      reference[(static_cast<std::size_t>(head) * dim) + d] =
          (weighted / denominator) * gated;
    }
  }
}

struct ErrorStats {
  double max_abs{0.0};
  double rms{0.0};
};

ErrorStats CompareWithReference(const std::vector<float>& output,
                                const std::vector<double>& reference) {
  ErrorStats stats;
  double square_sum = 0.0;
  for (std::size_t i = 0; i < output.size(); ++i) {
    const double difference =
        static_cast<double>(output[i]) - reference[i];
    stats.max_abs = std::max(stats.max_abs, std::abs(difference));
    square_sum += difference * difference;
  }
  stats.rms =
      std::sqrt(square_sum / static_cast<double>(output.size()));
  return stats;
}

/// Deterministic host inputs. `input` 0 is the long_context_ops_test formula
/// (non-saturating); 1 scales it so that the Q/P high+residual split lands the
/// residual in the f16 subnormal range (design D5/Risks).
void BuildInputs(int input, std::uint32_t num_heads, std::uint32_t num_kv_heads,
                 std::uint32_t max_context, std::vector<float>& query,
                 std::vector<float>& gate_values, std::vector<__half>& cache) {
  const std::size_t attention_width =
      static_cast<std::size_t>(num_heads) * kCandHeadDim;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * kCandHeadDim;
  const std::size_t cache_elements =
      static_cast<std::size_t>(max_context) * kv_width;
  query.resize(attention_width);
  gate_values.resize(attention_width);
  std::vector<float> cache_values(cache_elements);
  for (std::size_t i = 0; i < attention_width; ++i) {
    query[i] = 0.1F * std::sin(static_cast<float>((i % 257) + 1) * 0.013F);
    gate_values[i] =
        0.4F * std::cos(static_cast<float>((i % 193) + 1) * 0.017F);
  }
  for (std::size_t i = 0; i < cache_elements; ++i) {
    cache_values[i] =
        0.03F * std::sin(static_cast<float>((i % 509) + 1) * 0.011F);
  }
  if (input == 3) {
    for (float& value : query) {
      value = 0.0F;
    }
  }
  if (input == 2) {
    // Diagnostic: zero queries make every weight exactly 1, so the expected
    // output is the plain mean of V times the gate.
    for (float& value : query) {
      value = 0.0F;
    }
  }
  if (input == 1) {
    // "small" input: 2^-8 keeps every staged operand a normal f16 (the Q/K/V
    // magnitudes stay above the 2^-14 normal minimum) while the high+residual
    // split's residual is at most 2^-12 of the value, i.e. in the f16
    // subnormal range (2^-24..2^-14) for every element.
    const float small = 0.00390625F;
    for (float& value : query) {
      value *= small;
    }
    for (float& value : cache_values) {
      value *= small;
    }
  }
  if (input == 3 || input == 4) {
    // Beyond the required inputs, recorded as a limitation probe: 2^-16 puts
    // every value in the f16 subnormal range, where the WMMA's own operand
    // conversion loses precision (see the bench header note).
    const float extreme = 0.0000152587890625F;
    for (float& value : query) {
      value *= extreme;
    }
    for (float& value : cache_values) {
      value *= extreme;
    }
  }
  cache.resize(cache_elements);
  for (std::size_t i = 0; i < cache_elements; ++i) {
    cache[i] = __float2half_rn(cache_values[i]);
  }
}

struct DeviceState {
  float* q{nullptr};
  float* gate{nullptr};
  void* k_cache{nullptr};
  void* v_cache{nullptr};
  float* out{nullptr};
  float* scratch{nullptr};
  float* reference_out{nullptr};
  float* reference_scratch{nullptr};
};

struct HostInputs {
  std::vector<float> query;
  std::vector<float> gate;
  std::vector<__half> cache;
  std::uint32_t max_context{0};
  std::uint32_t rows{0};
};

std::vector<float> CopyOutput(DeviceState& state, std::uint32_t rows) {
  const std::size_t width =
      static_cast<std::size_t>(rows) * 24 * kCandHeadDim;
  std::vector<float> output(width);
  HIP_CHECK(hipMemcpy(output.data(), state.out, width * sizeof(float),
                      hipMemcpyDeviceToHost));
  return output;
}

std::vector<float> CopyRow(const float* source) {
  const std::size_t width = 24 * kCandHeadDim;
  std::vector<float> output(width);
  HIP_CHECK(hipMemcpy(output.data(), source, width * sizeof(float),
                      hipMemcpyDeviceToHost));
  return output;
}

void RunCold(const Options& options, HostInputs& inputs, DeviceState& state) {
  const std::uint32_t rows = options.rows;
  const std::size_t row_scratch =
      detail::DecodeAttentionScratchElements(24, kCandHeadDim);
  const std::uint32_t n = options.start_pos + rows;
  printf("cold n=%u rows=%u method=%s split-mode=%d max-context=%u\n", n, rows,
         options.method_b ? "b" : "a", options.split_mode,
         inputs.max_context);
  if (options.candidate) {
    const auto launch = [&] {
      LaunchCandidateMode(options.split_mode, state.q, state.gate,
                          static_cast<const __half*>(state.k_cache),
                          static_cast<const __half*>(state.v_cache), state.out,
                          0, options.start_pos, rows, inputs.max_context, 24, 4,
                          options.method_b, state.scratch, nullptr);
      HIP_CHECK(hipGetLastError());
    };
    const Timing timing = TimeLaunch(launch, options.iters);
    printf("candidate mode %d median %.4f ms  samples", options.split_mode,
           timing.median_ms);
    for (double sample : timing.samples) {
      printf(" %.4f", sample);
    }
    printf("\n");
  }
  if (options.control) {
    const auto launch = [&] {
      LaunchFrozenControl(state.q, state.gate,
                          static_cast<const __half*>(state.k_cache),
                          static_cast<const __half*>(state.v_cache), state.out,
                          0, options.start_pos, rows, inputs.max_context, 24, 4,
                          state.scratch, nullptr);
      HIP_CHECK(hipGetLastError());
    };
    const Timing timing = TimeLaunch(launch, options.iters);
    printf("control   median %.4f ms  samples", timing.median_ms);
    for (double sample : timing.samples) {
      printf(" %.4f", sample);
    }
    printf("\n");
  }
}

/// Byte comparison of every row of an R-row launch against the same row
/// computed alone (design D3/G0F row independence, task 1.2).
int RunRowIndependence(const Options& options, HostInputs& inputs,
                       DeviceState& state) {
  const std::size_t width = 24 * kCandHeadDim;
  const std::uint32_t positions[] = {0,      17,    120,   124,   127,
                                     128,    504,   508,   511,   512,
                                     4093,   4095,  4096,  4097,  8192,
                                     32768,  65536, 131064};
  int failures = 0;
  for (const std::uint32_t start : positions) {
    for (std::uint32_t rows : {2U, 3U, 8U}) {
      if (start + rows > inputs.max_context) {
        continue;
      }
      auto run = [&](std::uint32_t position, std::uint32_t row_base,
                     std::uint32_t row_count, float* destination) {
        LaunchCandidateMode(
            options.split_mode,
            state.q + (static_cast<std::size_t>(row_base) * width), state.gate,
            static_cast<const __half*>(state.k_cache),
            static_cast<const __half*>(state.v_cache), destination, 0, position,
            row_count, inputs.max_context, 24, 4, options.method_b,
            state.scratch, nullptr);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
      };
      run(start, 0, rows, state.out);
      const std::vector<float> batched = CopyOutput(state, rows);
      std::size_t differences = 0;
      printf("start %6u rows %u:", start, rows);
      for (std::uint32_t row = 0; row < rows; ++row) {
        run(start + row, 0, 1, state.reference_out);
        const std::vector<float> single = CopyRow(state.reference_out);
        std::size_t row_differences = 0;
        for (std::size_t i = 0; i < width; ++i) {
          const float a = batched[(row * width) + i];
          const float b = single[i];
          if (std::memcmp(&a, &b, sizeof(float)) != 0) {
            ++row_differences;
          }
        }
        printf(" r%u:%zu", row, row_differences);
        differences += row_differences;
      }
      printf("\n");
      failures += differences != 0 ? 1 : 0;
    }
  }
  printf("%s\n", failures == 0 ? "row independence OK" : "row independence FAILED");
  return failures == 0 ? 0 : 1;
}

/// Prints the worst elements and, for the record, the head/dim they sit in.
void ReportWorst(const std::vector<float>& output,
                 const std::vector<double>& reference,
                 const std::vector<float>& control) {
  std::vector<std::size_t> order(output.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  std::partial_sort(order.begin(), order.begin() + 3, order.end(),
                    [&](std::size_t a, std::size_t b) {
                      return std::abs(static_cast<double>(output[a]) -
                                      reference[a]) >
                             std::abs(static_cast<double>(output[b]) -
                                      reference[b]);
                    });
  for (std::size_t i = 0; i < 3 && i < order.size(); ++i) {
    const std::size_t index = order[i];
    printf("  worst element head %2zu dim %3zu: out %.9e reference %.9e "
           "control %.9e\n",
           index / kCandHeadDim, index % kCandHeadDim,
           static_cast<double>(output[index]), reference[index],
           control.empty() ? 0.0 : static_cast<double>(control[index]));
  }
}

int RunAccuracy(const Options& options, HostInputs& inputs,
                DeviceState& state) {
  const std::uint32_t default_series[] = {128, 129, 4096, 4097, 32768, 131072};
  std::vector<std::uint32_t> series(default_series,
                                   default_series + 6);
  if (!options.series_filter.empty()) {
    series = options.series_filter;
  }
  const std::size_t width = 24 * kCandHeadDim;
  // Required accuracy inputs (0 formula, 1 small) then the beyond-required
  // all-subnormal limitation probe (4).
  for (int input : {0, 1, 4}) {
    if (options.input_filter >= 0 && options.input_filter != input) {
      continue;
    }
    BuildInputs(input, 24, 4, inputs.max_context, inputs.query, inputs.gate,
                inputs.cache);
    HIP_CHECK(hipMemcpy(state.q, inputs.query.data(), width * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(state.gate, inputs.gate.data(), width * sizeof(float),
                        hipMemcpyHostToDevice));
    for (std::uint32_t n : series) {
      const std::size_t cache_bytes = static_cast<std::size_t>(n) * 4 *
                                      kCandHeadDim * sizeof(__half);
      HIP_CHECK(hipMemcpy(state.k_cache, inputs.cache.data(), cache_bytes,
                          hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(state.v_cache, inputs.cache.data(), cache_bytes,
                          hipMemcpyHostToDevice));
      std::vector<double> reference;
      ReferenceForRow(inputs.query, inputs.gate, inputs.cache, 0, n - 1,
                      inputs.max_context, 24, 4, reference);
      printf("--- input %s n=%u\n",
             input == 0   ? "formula"
             : input == 1 ? "small"
             : input == 2 ? "zero-q"
             : input == 3 ? "small+zero-q"
                          : "all-subnormal",
             n);
      std::vector<float> control_output;
      if (options.control) {
        LaunchFrozenControl(state.q, state.gate,
                            static_cast<const __half*>(state.k_cache),
                            static_cast<const __half*>(state.v_cache), state.out,
                            0, n - 1, 1, inputs.max_context, 24, 4,
                            state.scratch, nullptr);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        control_output = CopyOutput(state, 1);
        const auto stats = CompareWithReference(control_output, reference);
        printf("control          max-abs %.6e rms %.6e\n", stats.max_abs,
               stats.rms);
      }
      const char* const mode_names[] = {"plain (none)   ", "split Q only   ",
                                        "split P only   ", "split Q and P  ",
                                        "Q + P 3 terms  ", "normalized acc ",
                                        "normalized P   ", "production     "};
      for (int mode : {7, 3, 4, 5, 6, 1, 2, 0}) {
        const auto launch = [&] {
          LaunchCandidateMode(mode, state.q, state.gate,
                              static_cast<const __half*>(state.k_cache),
                              static_cast<const __half*>(state.v_cache),
                              state.out, 0, n - 1, 1, inputs.max_context, 24,
                              4, options.method_b, state.scratch, nullptr);
          HIP_CHECK(hipGetLastError());
          HIP_CHECK(hipDeviceSynchronize());
        };
        launch();
        const std::vector<float> output = CopyOutput(state, 1);
        const auto stats = CompareWithReference(output, reference);
        printf("candidate %s max-abs %.6e rms %.6e\n", mode_names[mode],
               stats.max_abs, stats.rms);
        if (mode == 3) {
          ReportWorst(output, reference, control_output);
        }
      }
    }
  }
  return 0;
}

/// Resident sweep: one layer's KV fits in the SoC cache at these contexts, and
/// the least-squares fit of the time against the token count converts to the
/// 128K compute floor (design D6/D15).
int RunResident(const Options& options, HostInputs& inputs,
                DeviceState& state) {
  const std::size_t row_scratch =
      detail::DecodeAttentionScratchElements(24, kCandHeadDim);
  std::vector<double> xs;
  std::vector<double> candidate_ms;
  std::vector<double> control_ms;
  for (std::uint32_t n = options.sweep_lo; n <= options.sweep_hi;
       n += options.sweep_step) {
    if (n + 1 > inputs.max_context) {
      break;
    }
    const std::size_t cache_bytes =
        static_cast<std::size_t>(n) * 4 * kCandHeadDim * sizeof(__half);
    HIP_CHECK(hipMemcpy(state.k_cache, inputs.cache.data(), cache_bytes,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(state.v_cache, inputs.cache.data(), cache_bytes,
                        hipMemcpyHostToDevice));
    double candidate = 0.0;
    if (options.candidate) {
      const auto launch = [&] {
        LaunchCandidateMode(options.split_mode, state.q, state.gate,
                            static_cast<const __half*>(state.k_cache),
                            static_cast<const __half*>(state.v_cache), state.out,
                            0, n - 1, 1, inputs.max_context, 24, 4,
                            options.method_b, state.scratch, nullptr);
        HIP_CHECK(hipGetLastError());
      };
      candidate = TimeLaunch(launch, options.iters).median_ms;
    }
    double control = 0.0;
    if (options.control) {
      const auto launch = [&] {
        LaunchFrozenControl(state.q, state.gate,
                            static_cast<const __half*>(state.k_cache),
                            static_cast<const __half*>(state.v_cache), state.out,
                            0, n - 1, 1, inputs.max_context, 24, 4,
                            state.scratch, nullptr);
        HIP_CHECK(hipGetLastError());
      };
      control = TimeLaunch(launch, options.iters).median_ms;
    }
    printf("resident n=%6u candidate %.4f ms control %.4f ms\n", n, candidate,
           control);
    xs.push_back(static_cast<double>(n));
    candidate_ms.push_back(candidate);
    control_ms.push_back(control);
  }
  const auto fit = [](const std::vector<double>& x,
                      const std::vector<double>& y, const char* label) {
    if (x.size() < 2) {
      return;
    }
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xx = 0.0;
    double sum_xy = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      sum_x += x[i];
      sum_y += y[i];
      sum_xx += x[i] * x[i];
      sum_xy += x[i] * y[i];
    }
    const double count = static_cast<double>(x.size());
    const double slope =
        ((count * sum_xy) - (sum_x * sum_y)) / ((count * sum_xx) - sum_x * sum_x);
    const double intercept = (sum_y - (slope * sum_x)) / count;
    const double converted = intercept + (slope * 131072.0);
    const double unit_cost_ms = slope / 24.0;
    printf("%s: slope %.6e ms/token intercept %.4f ms unit-cost %.6e ms per "
           "(query head, key) 128K-> %.4f ms/layer\n",
           label, slope, intercept, unit_cost_ms, converted);
  };
  if (options.candidate) {
    fit(xs, candidate_ms, "candidate");
  }
  if (options.control) {
    fit(xs, control_ms, "control");
  }
  return 0;
}

/// Compares the candidate's per-partition (max, sum, numerator) against the
/// same quantities recomputed in double from the FP64 scores, to localize an
/// accuracy difference between the weight path and the value path.
int RunStatsCheck(const Options& options, HostInputs& inputs,
                  DeviceState& state) {
  const std::uint32_t n = options.start_pos + 1;
  const std::size_t cache_bytes =
      static_cast<std::size_t>(n) * 4 * kCandHeadDim * sizeof(__half);
  HIP_CHECK(hipMemcpy(state.k_cache, inputs.cache.data(), cache_bytes,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(state.v_cache, inputs.cache.data(), cache_bytes,
                      hipMemcpyHostToDevice));
  LaunchCandidateMode(options.split_mode, state.q, state.gate,
                      static_cast<const __half*>(state.k_cache),
                      static_cast<const __half*>(state.v_cache), state.out, 0,
                      n - 1, 1, inputs.max_context, 24, 4, options.method_b,
                      state.scratch, nullptr);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  const std::size_t row_scratch =
      detail::DecodeAttentionScratchElements(24, kCandHeadDim);
  std::vector<float> host(row_scratch);
  HIP_CHECK(hipMemcpy(host.data(), state.scratch, row_scratch * sizeof(float),
                      hipMemcpyDeviceToHost));
  const std::size_t stats_elements = static_cast<std::size_t>(24) *
                                     kCandSplits * 2;
  const float* const partials = host.data() + stats_elements;
  const std::uint32_t length = MethodPartitionLength(n);
  const double scale = 1.0 / std::sqrt(static_cast<double>(kCandHeadDim));
  std::vector<double> scores(n);
  printf("stats for split mode %d\n", options.split_mode);
  for (std::uint32_t head : {0U}) {
    const std::uint32_t kv_head = head / kCandGqa;
    const float* const query = inputs.query.data() + head * kCandHeadDim;
    for (std::uint32_t t = 0; t < n; ++t) {
      double dot = 0.0;
      const __half* const key =
          inputs.cache.data() +
          (static_cast<std::size_t>(t) * 4 * kCandHeadDim) +
          (kv_head * kCandHeadDim);
      for (std::uint32_t d = 0; d < kCandHeadDim; ++d) {
        dot += static_cast<double>(query[d]) *
               static_cast<double>(__half2float(key[d]));
      }
      scores[t] = dot * scale;
    }
    for (std::uint32_t split = 0; split < kCandSplits; ++split) {
      const std::uint32_t begin = split * length;
      if (begin >= n) {
        continue;
      }
      const std::uint32_t end = std::min(begin + length, n);
      const std::size_t index =
          (static_cast<std::size_t>(head) * kCandSplits) + split;
      const float kernel_max = host[index * 2];
      const float kernel_sum = host[index * 2 + 1];
      double sum = 0.0;
      double numerator = 0.0;
      double flat_numerator = 0.0;
      for (std::uint32_t t = begin; t < end; ++t) {
        const double w = std::exp(scores[t] - static_cast<double>(kernel_max));
        sum += w;
        const double value = static_cast<double>(
            __half2float(inputs.cache[(static_cast<std::size_t>(t) * 4 *
                                       kCandHeadDim) +
                                      (kv_head * kCandHeadDim)]));
        numerator += w * value;
        flat_numerator += value;
      }
      const double kernel_numerator =
          static_cast<double>(partials[(index * kCandHeadDim)]);
      if (split % 8 == 0) {
        printf("head %2u split %2u max %.9e sum %.6e numerator %.6e (weighted "
               "%.6e rel %.3e; unweighted %.6e rel %.3e)\n",
               head, split, static_cast<double>(kernel_max),
               static_cast<double>(kernel_sum), kernel_numerator, numerator,
               (kernel_numerator - numerator) / numerator, flat_numerator,
               (kernel_numerator - flat_numerator) / flat_numerator);
      }
    }
  }
  return 0;
}

// --------------------------------------------------------------- KVarN read
// Phase 2 of the change: the KVarN decode read at tail 2048, measured against
// the verbatim baseline copy of the read kernels (design D6/D14).
//
// Sealing and inputs (design D5):
//   * `--preset kvarn-k5v4` (K 5 bit, V 4 bit) and `--tail 2048` are the main
//     configuration of the G0K gate; the fixture is one full-attention layer of
//     kKvarnLayers x kKvarnKvHeads KV heads,
//   * the sink (128 rows) and the F16 tail ring (L + 128 + W rows) hold
//     `KvarnBenchKeyValue` as f16, and every tile the view rule of design D2
//     reads quantized for the largest query position is sealed with the CPU
//     reference encoder (`src/models/qwen/kvarn_reference.hpp`), exactly as
//     tools/qwen27b/kvarn_read_compute_bench.hip and
//     tools/bench/qwen_decode_attention_dump.hip do,
//   * all rows of one launch share one query row (`KvarnBenchQueryValue`) and
//     one SiLU gate; row r's causal endpoint is its own position start + r, so
//     a row never reads the rows after it.
//
// FP64 reference: the same view rule over the same sealed bytes, decoded with
// the CPU reference (`KvarnDecodeTileElement` plus the tile's channel and token
// scales), an exact double softmax, the weighted sum of the stored (rotated) V,
// then the inverse Hadamard rotation and the SiLU gate - the shape of
// tests/models/qwen/hip/attention/long_context_ops_test.cpp `Reference`.
//
// Modes (sealing happens once per depth, so one invocation is the unit):
//   kvarn-cold       R=1 and R=8 at every `--series` depth (default 32768 and
//                    131072), for the decode launcher and for the causal
//                    (verification-row) launcher; median of `--iters`
//   kvarn-accuracy   max-absolute and RMS error of every row against the FP64
//                    reference at every `--series` depth (R = `--rows`)
//   kvarn-r1-check   every row of an R-row launch (R in 2, 3, 8) compared byte
//                    for byte against the same row launched alone, at the
//                    shallow test positions and at `--start-pos`; exits non-zero
//                    when any row differs
// Run:
//   /tmp/qwen_decode_attention_bench --mode kvarn-cold
//   /tmp/qwen_decode_attention_bench --mode kvarn-accuracy --series 32768,131072
//   /tmp/qwen_decode_attention_bench --mode kvarn-r1-check
namespace kvarn_ref = gufo::models::qwen;

constexpr std::uint32_t kKvarnHeads = 24;
constexpr std::uint32_t kKvarnKvHeads = 4;
constexpr std::uint32_t kKvarnLayers = 1;
constexpr std::uint32_t kKvarnWidth = kKvarnHeads * kCandHeadDim;
/// One launch covers a whole verification group (design D8/W).
constexpr std::uint32_t kKvarnMaxRows = 8;

[[nodiscard]] float KvarnBenchKeyValue(std::uint32_t position,
                                       std::uint32_t channel, bool is_value,
                                       std::uint32_t kv_head) {
  const float p = static_cast<float>(position);
  const float c = static_cast<float>(channel);
  const float h = static_cast<float>(kv_head);
  return is_value ? 0.25F * std::cos(p * 0.05F + c * 0.017F + h * 0.11F)
                  : 0.3F * std::sin(p * 0.07F + c * 0.013F + h * 0.07F);
}

[[nodiscard]] float KvarnBenchQueryValue(std::uint32_t head,
                                         std::uint32_t channel) {
  return 0.2F * std::sin(static_cast<float>(head) * 0.3F +
                         static_cast<float>(channel) * 0.021F);
}

[[nodiscard]] float KvarnBenchGate(std::size_t index) {
  return std::cos(static_cast<float>(index) * 0.013F);
}

/// Host form of the read view rule (design D2): tile `tile` is read quantized
/// for a query at absolute position `pos` when `e_t + L + 1 <= pos`. Kept
/// separate from the device helper because the device one is `__device__` only.
[[nodiscard]] bool KvarnBenchTileQuantized(std::uint32_t pos,
                                           std::uint32_t tail_rows,
                                           std::uint32_t tile) {
  return 128U + (128U * tile) + 127U + tail_rows + 1U <= pos;
}

/// One full-attention layer of KVarN storage sealed with the CPU reference
/// encoder. The host copy of the sealed body stays alive so the FP64 reference
/// can decode exactly the bytes the kernels read.
///
/// `base` is the lowest query position of the launches the fixture serves and
/// `rows` their widest row group, so the fixture covers positions
/// `[0, base + rows)`. The F16 sink holds `[0, 128)`, every tile the *highest*
/// row (`base + rows - 1`) reads quantized is sealed, and the F16 tail ring
/// holds `[first_f16(base), base + rows)`: a later row's own F16 window is a
/// subset of that range and its sealed reads are a subset of the sealed tiles,
/// so any sub-launch with `start >= base` and `start + count <= base + rows`
/// reads exactly the bytes its own view rule selects. The range fits the ring
/// capacity `L + 128 + W` because the window is `(base - 128 - L) % 128 + L +
/// rows <= 127 + L + rows`.
struct KvarnFixture {
  std::uint32_t context{0};
  std::uint32_t tail_rows{0};
  std::uint32_t ring_rows{0};
  std::uint32_t tile_count{0};
  std::uint32_t quantized_tiles{0};
  std::size_t k_tile_bytes{0};
  std::size_t v_tile_bytes{0};
  std::size_t sink_plane{0};
  std::size_t tail_plane{0};
  std::vector<std::uint8_t> host_body;
  __half* sink{nullptr};
  __half* tail{nullptr};
  std::uint8_t* body{nullptr};
  KvarnReadParams read{};

  KvarnFixture(std::uint32_t base, std::uint32_t rows, std::uint8_t k_bits,
               std::uint8_t v_bits, std::uint32_t tail_tokens)
      : context(base + rows) {
    QwenKvarnPreset preset{};
    preset.k_bits = k_bits;
    preset.v_bits = v_bits;
    preset.tail_length = tail_tokens;
    tail_rows = QwenKvarnTailRows(preset);
    ring_rows = QwenKvarnTailRingRows(preset);
    // Contexts shorter than the sink hold no sealed tiles at all.
    tile_count =
        context > kvarn_ref::kKvarnSinkRows
            ? static_cast<std::uint32_t>(
                  (context - kvarn_ref::kKvarnSinkRows +
                   kvarn_ref::kKvarnTileTokens - 1) /
                  kvarn_ref::kKvarnTileTokens)
            : 0;
    k_tile_bytes = kvarn_ref::KvarnTilePayloadBytes(k_bits);
    v_tile_bytes = kvarn_ref::KvarnTilePayloadBytes(v_bits);
    sink_plane = static_cast<std::size_t>(kKvarnLayers) * kKvarnKvHeads *
                 kvarn_ref::kKvarnSinkRows * kCandHeadDim;
    tail_plane = static_cast<std::size_t>(kKvarnLayers) * kKvarnKvHeads *
                 ring_rows * kCandHeadDim;
    HIP_CHECK(hipMalloc(&sink, 2 * sink_plane * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&tail, 2 * tail_plane * sizeof(std::uint16_t)));
    host_body.assign(static_cast<std::size_t>(kKvarnLayers) * kKvarnKvHeads *
                         tile_count * (k_tile_bytes + v_tile_bytes),
                     0);
    if (!host_body.empty()) {
      void* body_raw = nullptr;
      HIP_CHECK(hipMalloc(&body_raw, host_body.size()));
      body = static_cast<std::uint8_t*>(body_raw);
    }

    // Quantized tiles for the highest row, the F16 window of the lowest row.
    const std::uint32_t highest = context - 1;
    quantized_tiles =
        highest >= kvarn_ref::kKvarnSinkRows + tail_rows + 1
            ? static_cast<std::uint32_t>(
                  (highest - kvarn_ref::kKvarnSinkRows - tail_rows) /
                  kvarn_ref::kKvarnTileTokens)
            : 0;
    const std::uint32_t lowest_tiles =
        base >= kvarn_ref::kKvarnSinkRows + tail_rows + 1
            ? static_cast<std::uint32_t>(
                  (base - kvarn_ref::kKvarnSinkRows - tail_rows) /
                  kvarn_ref::kKvarnTileTokens)
            : 0;
    const std::uint32_t first_f16 =
        kvarn_ref::kKvarnSinkRows + kvarn_ref::kKvarnTileTokens * lowest_tiles;

    std::vector<std::uint16_t> sink_host(2 * sink_plane, 0);
    std::vector<std::uint16_t> tail_host(2 * tail_plane, 0);
    const auto write_f16 = [&](std::uint32_t position) {
      for (std::uint32_t head = 0; head < kKvarnKvHeads; ++head) {
        for (const bool is_value : {false, true}) {
          for (std::uint32_t c = 0; c < kCandHeadDim; ++c) {
            const std::uint16_t encoded = __half_as_ushort(__float2half_rn(
                KvarnBenchKeyValue(position, c, is_value, head)));
            std::size_t offset = 0;
            std::size_t plane = 0;
            if (position < kvarn_ref::kKvarnSinkRows) {
              offset = kvarn_ref::KvarnSinkRowOffset(0, kKvarnKvHeads, head,
                                                     position);
              plane = is_value ? sink_plane : 0;
              sink_host[plane + offset + c] = encoded;
            } else {
              offset = kvarn_ref::KvarnTailRowOffset(0, kKvarnKvHeads, head,
                                                     position, ring_rows);
              plane = is_value ? tail_plane : 0;
              tail_host[plane + offset + c] = encoded;
            }
          }
        }
      }
    };
    for (std::uint32_t position = 0; position < kvarn_ref::kKvarnSinkRows;
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
    std::vector<float> tile(kvarn_ref::kKvarnTileElements);
    std::vector<float> normalized(kvarn_ref::kKvarnTileElements);
    for (std::uint32_t head = 0; head < kKvarnKvHeads; ++head) {
      for (const bool is_value : {false, true}) {
        for (std::uint32_t t = 0; t < quantized_tiles; ++t) {
          for (std::uint32_t token = 0; token < kvarn_ref::kKvarnTileTokens;
               ++token) {
            const std::uint32_t position =
                kvarn_ref::kKvarnSinkRows +
                kvarn_ref::kKvarnTileTokens * t + token;
            float* row = tile.data() + token * kvarn_ref::kKvarnTileChannels;
            for (std::uint32_t c = 0; c < kvarn_ref::kKvarnTileChannels; ++c) {
              row[c] = KvarnBenchKeyValue(position, c, is_value, head);
            }
            kvarn_ref::KvarnFwhtHeadInPlace(
                std::span<float>(row, kvarn_ref::kKvarnTileChannels));
          }
          kvarn_ref::KvarnTileQuantResult result{};
          kvarn_ref::KvarnQuantDiagnostics diag{};
          const auto status = kvarn_ref::KvarnNormalizeTile(tile, normalized,
                                                           &result, &diag);
          if (status != kvarn_ref::KvarnQuantStatus::kOk) {
            fprintf(stderr, "reference tile normalization failed\n");
            std::exit(1);
          }
          kvarn_ref::KvarnQuantizeAsymmetricRtn(
              normalized,
              is_value ? kvarn_ref::KvarnRtnAxis::kTokenGroup
                       : kvarn_ref::KvarnRtnAxis::kChannel,
              is_value ? v_bits : k_bits, &result, &diag);
          kvarn_ref::KvarnTilePayload payload{};
          kvarn_ref::KvarnPackTilePayload(result, &payload);
          const std::size_t offset = kvarn_ref::KvarnBodyTileOffset(
                                         0, kKvarnKvHeads, head, t, tile_count,
                                         k_tile_bytes, v_tile_bytes) +
                                     (is_value ? k_tile_bytes : 0);
          std::memcpy(host_body.data() + offset, &payload,
                      is_value ? v_tile_bytes : k_tile_bytes);
        }
      }
    }
    if (body != nullptr) {
      HIP_CHECK(hipMemcpy(body, host_body.data(), host_body.size(),
                          hipMemcpyHostToDevice));
    }

    read.sink = sink;
    read.tail = tail;
    read.body = body;
    read.sink_plane_elements = sink_plane;
    read.tail_plane_elements = tail_plane;
    read.ring_rows = ring_rows;
    read.tile_count = tile_count;
    read.k_tile_bytes = static_cast<std::uint32_t>(k_tile_bytes);
    read.v_tile_bytes = static_cast<std::uint32_t>(v_tile_bytes);
    read.kv_heads = kKvarnKvHeads;
    read.tail_length = tail_rows;
    read.k_bits = k_bits;
    read.v_bits = v_bits;
  }

  ~KvarnFixture() {
    (void)hipFree(sink);
    (void)hipFree(tail);
    (void)hipFree(body);
  }

  KvarnFixture(const KvarnFixture&) = delete;
  KvarnFixture& operator=(const KvarnFixture&) = delete;

  /// Sealed payload of one (KV head, tile) plane, from the host copy.
  [[nodiscard]] const kvarn_ref::KvarnTilePayload* Payload(
      std::uint32_t kv_head, std::uint32_t tile, bool is_value) const {
    const std::size_t offset = kvarn_ref::KvarnBodyTileOffset(
                                   0, kKvarnKvHeads, kv_head, tile, tile_count,
                                   k_tile_bytes, v_tile_bytes) +
                               (is_value ? k_tile_bytes : 0);
    return reinterpret_cast<const kvarn_ref::KvarnTilePayload*>(
        host_body.data() + offset);
  }
};

/// FP64 reference of one query row, every head, over the sealed fixture.
[[nodiscard]] std::vector<double> KvarnReferenceRow(
    const KvarnFixture& fixture, std::uint32_t pos,
    const std::vector<float>& query, const std::vector<float>& gate) {
  constexpr std::uint32_t dim = kCandHeadDim;
  const std::uint32_t group_size = kKvarnHeads / kKvarnKvHeads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
  std::vector<double> reference(static_cast<std::size_t>(kKvarnHeads) * dim,
                                0.0);
  std::vector<double> scores(static_cast<std::size_t>(pos) + 1);
  for (std::uint32_t head = 0; head < kKvarnHeads; ++head) {
    const std::uint32_t kv_head = head / group_size;
    const float* const query_head =
        query.data() + static_cast<std::size_t>(head) * dim;
    const auto quantized = [&](std::uint32_t position) {
      if (position < kvarn_ref::kKvarnSinkRows) {
        return false;
      }
      const std::uint32_t tile =
          (position - kvarn_ref::kKvarnSinkRows) / kvarn_ref::kKvarnTileTokens;
      return KvarnBenchTileQuantized(pos, fixture.tail_rows, tile);
    };
    const auto tile_of = [](std::uint32_t position) {
      return (position - kvarn_ref::kKvarnSinkRows) /
             kvarn_ref::kKvarnTileTokens;
    };
    const auto token_of = [](std::uint32_t position) {
      const std::uint32_t tile =
          (position - kvarn_ref::kKvarnSinkRows) / kvarn_ref::kKvarnTileTokens;
      return position - kvarn_ref::kKvarnSinkRows -
             kvarn_ref::kKvarnTileTokens * tile;
    };
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::uint32_t position = 0; position <= pos; ++position) {
      double dot = 0.0;
      if (!quantized(position)) {
        for (std::uint32_t c = 0; c < dim; ++c) {
          dot += static_cast<double>(query_head[c]) *
                 static_cast<double>(__half2float(__float2half_rn(
                     KvarnBenchKeyValue(position, c, false, kv_head))));
        }
      } else {
        const std::uint32_t token = token_of(position);
        const auto* const payload =
            fixture.Payload(kv_head, tile_of(position), false);
        const double token_scale = static_cast<double>(__half2float(
            __ushort_as_half(payload->norm_token_scale[token])));
        for (std::uint32_t c = 0; c < dim; ++c) {
          const double decoded = static_cast<double>(
              kvarn_ref::KvarnDecodeTileElement(
                  *payload, kvarn_ref::KvarnRtnAxis::kChannel,
                  fixture.read.k_bits, token, c));
          const double channel_scale = static_cast<double>(__half2float(
              __ushort_as_half(payload->norm_channel_scale[c])));
          dot += static_cast<double>(query_head[c]) * decoded * channel_scale *
                 token_scale;
        }
      }
      scores[position] = dot * scale;
      maximum = std::max(maximum, scores[position]);
    }
    double denominator = 0.0;
    for (std::uint32_t position = 0; position <= pos; ++position) {
      scores[position] = std::exp(scores[position] - maximum);
      denominator += scores[position];
    }
    std::vector<double> merged(dim, 0.0);
    for (std::uint32_t position = 0; position <= pos; ++position) {
      const double weight = scores[position] / denominator;
      if (!quantized(position)) {
        for (std::uint32_t c = 0; c < dim; ++c) {
          merged[c] +=
              weight *
              static_cast<double>(__half2float(__float2half_rn(
                  KvarnBenchKeyValue(position, c, true, kv_head))));
        }
      } else {
        const std::uint32_t tile = tile_of(position);
        const std::uint32_t token = token_of(position);
        const auto* const payload = fixture.Payload(kv_head, tile, true);
        const double token_scale = static_cast<double>(__half2float(
            __ushort_as_half(payload->norm_token_scale[token])));
        for (std::uint32_t c = 0; c < dim; ++c) {
          const double decoded = static_cast<double>(
              kvarn_ref::KvarnDecodeTileElement(
                  *payload, kvarn_ref::KvarnRtnAxis::kTokenGroup,
                  fixture.read.v_bits, token, c));
          const double channel_scale = static_cast<double>(__half2float(
              __ushort_as_half(payload->norm_channel_scale[c])));
          merged[c] += weight * decoded * channel_scale * token_scale;
        }
      }
    }
    // The merged context is rotated back once, before the SiLU gate.
    std::vector<float> rotated(dim);
    for (std::uint32_t c = 0; c < dim; ++c) {
      rotated[c] = static_cast<float>(merged[c]);
    }
    kvarn_ref::KvarnFwhtHeadInPlace(std::span<float>(rotated));
    for (std::uint32_t c = 0; c < dim; ++c) {
      const double gated = 1.0 / (1.0 + std::exp(-static_cast<double>(
                                             gate[static_cast<std::size_t>(head) *
                                                      dim + c])));
      reference[static_cast<std::size_t>(head) * dim + c] =
          static_cast<double>(rotated[c]) * gated;
    }
  }
  return reference;
}

/// Device buffers shared by the KVarN modes: one query row and one gate row per
/// verification row, plus the split-K scratch of a full row group.
struct KvarnDevice {
  std::vector<float> query;
  std::vector<float> gate;
  float* d_query{nullptr};
  float* d_gate{nullptr};
  float* d_out{nullptr};
  float* d_scratch{nullptr};

  KvarnDevice() {
    query.assign(static_cast<std::size_t>(kKvarnMaxRows) * kKvarnWidth, 0.0F);
    gate.assign(query.size(), 0.0F);
    for (std::uint32_t row = 0; row < kKvarnMaxRows; ++row) {
      for (std::uint32_t head = 0; head < kKvarnHeads; ++head) {
        for (std::uint32_t c = 0; c < kCandHeadDim; ++c) {
          const std::size_t index =
              ((static_cast<std::size_t>(row) * kKvarnHeads) + head) *
                  kCandHeadDim +
              c;
          query[index] = KvarnBenchQueryValue(head, c);
          gate[index] = KvarnBenchGate(index);
        }
      }
    }
    HIP_CHECK(hipMalloc(&d_query, query.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gate, gate.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, query.size() * sizeof(float)));
    const std::size_t row_scratch =
        detail::DecodeAttentionScratchElements(kKvarnHeads, kCandHeadDim);
    HIP_CHECK(hipMalloc(&d_scratch,
                        kKvarnMaxRows * row_scratch * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_query, query.data(), query.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_gate, gate.data(), gate.size() * sizeof(float),
                        hipMemcpyHostToDevice));
  }

  ~KvarnDevice() {
    (void)hipFree(d_query);
    (void)hipFree(d_gate);
    (void)hipFree(d_out);
    (void)hipFree(d_scratch);
  }

  KvarnDevice(const KvarnDevice&) = delete;
  KvarnDevice& operator=(const KvarnDevice&) = delete;

  [[nodiscard]] std::vector<float> ReadOutput(std::uint32_t rows) const {
    std::vector<float> host(static_cast<std::size_t>(rows) * kKvarnWidth);
    HIP_CHECK(hipMemcpy(host.data(), d_out, host.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
    return host;
  }
};

// --------------------------------------------------------------- candidate
// Design D14: one block per (KV head, partition) processes every row of the
// launch, the sealed tile is expanded once into LDS and shared by the six GQA
// heads of the KV head and by every row, and a warp owns one row and runs the
// WMMA with the six query heads in the M dimension.
//
// Operand layout (`v_wmma_f32_16x16x16_f16`, wave32 - the convention the Phase 1
// block candidate uses and its accuracy gate verified):
//   A element j of lane L = A[m = L % 16][k = base + j],
//   B element j of lane L = B[n = L % 16][k = base + j],
//   C element i of lane L = C[m = 2i + L / 16][n = L % 16],   i = 0..7
// so the score MMA has the query head as A's row and the key as B's row, and the
// PV MMA has the weight as A's row (the head) and the channel as B's row.
//
// Arithmetic (design D2/D5/D14). The B operand is exact in f16 - it is either a
// sealed tile's 4-6 bit code or a stored F16 key/value row - so only the A side
// carries a high + 2^11-scaled residual (design D2: the B side is f16-exact).
// The KVarN fold is the baseline's own folded form, with the channel scale, the
// RTN scale and the RTN zero point gathered on the A side and the token scale
// and the V channel scale in the FP32 post-processing:
//
//   score[h][t] = scale * tok_k[t] * ( z[h] + sum_c A_k[h][c] * code_k[t][c] )
//   A_k[h][c]   = q[h][c] * ch_scale_k[c] * rtn_scale_k[c]      (A-side fold)
//   z[h]        = sum_c q[h][c] * ch_scale_k[c] * rtn_zero_k[c]
//   out[h][c]  += ch_scale_v[c] * ( sum_t A_v[h][t] * code_v[t][c]
//                                   + sum_t weight[h][t] * rtn_zero_v[t][g] )
//   A_v[h][t]   = weight[h][t] * tok_v[t] * rtn_scale_v[t][g]
// with `g` the 128-channel group of the output channel `c` (design D5). The
// dequantized tensor never reaches global memory: it lives in LDS for one 16-key
// chunk and every row and head of the block shares it. Outside the WMMA every
// accumulation is an explicit `fmaf`, and a row's key sequence depends on its own
// causal endpoint alone (the chunk order, the 16-channel-chunk WMMA order, the
// reduction trees and the end mask never involve `rows`), so `kvarn-r1-check` can
// demand byte equality across R.
constexpr std::uint32_t kKvarnStage = 16;
constexpr std::uint32_t kKvarnCandWarps = 8;
constexpr std::uint32_t kKvarnCandThreads = kKvarnCandWarps * kLanes;
constexpr std::uint32_t kKvarnChunks = kHeadDim / 16;
constexpr std::uint32_t kKvarnGqa = 6;
/// 128-channel V groups per head (design D5).
constexpr std::uint32_t kKvarnGroups = 2;
static_assert(kKvarnGroups == models::qwen::kKvarnValueGroups);
static_assert(kKvarnGroups * models::qwen::kKvarnValueGroupChannels ==
              kHeadDim);
constexpr float kKvarnResidualFactor = 2048.0F;
constexpr float kKvarnResidualFactorInverse = 1.0F / 2048.0F;

/// f16 fragment of one operand row. The two uint4 words are the k values
/// `base + 0..7` and `base + 8..15`, packed exactly the way the Phase 1
/// candidate packs its fragments.
union KvarnFragment {
  v16h packed;
  uint4 words[2];
};

/// Builds one A-side operand pair with `value_of(k)` for the k values
/// `0..15`: the f16 high part and the 2^11-scaled residual, packed into the
/// fragment's two uint4 words exactly the way the Phase 1 candidate packs its
/// fragments (one uint4 assignment per eight elements, never an element-wise
/// write into a vector member).
template<typename ValueOf>
__device__ __forceinline__ void KvarnBuildFold(KvarnFragment* high_fragment,
                                               KvarnFragment* low_fragment,
                                               ValueOf value_of) {
#pragma unroll
  for (std::uint32_t half = 0; half < 2; ++half) {
    uint4 packed_high;
    uint4 packed_low;
    auto* const high_words = reinterpret_cast<__half*>(&packed_high);
    auto* const low_words = reinterpret_cast<__half*>(&packed_low);
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      const float value = value_of((half * 8) + i);
      const __half top = __float2half_rn(value);
      high_words[i] = top;
      low_words[i] = __float2half_rn(
          fmaf(value - __half2float(top), kKvarnResidualFactor, 0.0F));
    }
    high_fragment->words[half] = packed_high;
    low_fragment->words[half] = packed_low;
  }
}

/// Channel normalization scale of a sealed tile (design D5).
__device__ __forceinline__ float KvarnPayloadChannelScale(
    const std::uint8_t* payload, std::uint32_t channel) {
  return __half2float(
      reinterpret_cast<const __half*>(payload + 4 * kHeadDim)[channel]);
}

/// Token normalization scale of a sealed tile (design D5). K scores fold the K
/// tile's token scale and V weights fold the V tile's, so the two planes keep
/// their own scales.
__device__ __forceinline__ float KvarnPayloadTokenScale(
    const std::uint8_t* payload, std::uint32_t token) {
  return __half2float(
      reinterpret_cast<const __half*>(payload + 4 * kHeadDim + 2 * kHeadDim)
          [token]);
}

/// One K tile's RTN pair: one pair per channel (design D5).
__device__ __forceinline__ void KvarnTileKeyRtn(const std::uint8_t* payload,
                                                std::uint32_t channel,
                                                float* scale, float* zero) {
  const auto* const rtn_scale = reinterpret_cast<const __half*>(payload);
  const auto* const rtn_zero = rtn_scale + models::qwen::kKvarnRtnParams;
  *scale = __half2float(rtn_scale[channel]);
  *zero = __half2float(rtn_zero[channel]);
}

/// One V tile's RTN pair: one pair per (token, 128-channel group) (design D5).
__device__ __forceinline__ void KvarnTileValueRtn(const std::uint8_t* payload,
                                                  std::uint32_t token,
                                                  std::uint32_t group,
                                                  float* scale, float* zero) {
  const auto* const rtn_scale = reinterpret_cast<const __half*>(payload);
  const auto* const rtn_zero = rtn_scale + models::qwen::kKvarnRtnParams;
  const std::uint32_t parameter =
      (token * static_cast<std::uint32_t>(kKvarnGroups)) +
      group;
  *scale = __half2float(rtn_scale[parameter]);
  *zero = __half2float(rtn_zero[parameter]);
}

/// One stored code of a sealed tile. The 4-6 bit codes are exact in f16, which
/// is why the B operand needs no residual split (design D2).
__device__ __forceinline__ float KvarnTileCode(const std::uint8_t* payload,
                                               std::uint8_t bits,
                                               std::uint32_t token,
                                               std::uint32_t channel) {
  return static_cast<float>(KvarnUnpackCode(
      payload, bits,
      (static_cast<std::size_t>(token) * kHeadDim) + channel));
}

__launch_bounds__(kKvarnCandThreads) __global__
    void QwenKvarnBlockDecodePartialsKernel(
        const float* __restrict__ q, float* __restrict__ scratch,
        std::uint32_t layer_idx, std::uint32_t start_pos, std::uint32_t rows,
        std::uint32_t num_heads, std::uint32_t num_kv_heads,
        std::uint32_t split_count, KvarnReadParams kvarn) {
  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;
  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t split = blockIdx.y;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t warp = tid / kLanes;
  const std::uint32_t lane = tid % kLanes;
  const bool row_active = warp < rows;
  const std::uint32_t pos = start_pos + warp;
  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t head_begin = kv_head * group_size;
  const float scale = rsqrtf(static_cast<float>(kHeadDim));

  __shared__ std::uint32_t s_begin[kKvarnCandWarps];
  __shared__ std::uint32_t s_end[kKvarnCandWarps];
  __shared__ std::uint32_t s_pos[kKvarnCandWarps];
  /// The expanded tile operand of one 16-key chunk: the K codes or stored F16
  /// keys in `[token][channel]`, the V codes or stored F16 values in
  /// `[channel][token]`. Both are exact in f16 and shared by every row and
  /// every head of the block (design D14).
  __shared__ __align__(16) __half s_k[kKvarnStage * kHeadDim];
  __shared__ __align__(16) __half s_v[kHeadDim * kKvarnStage];
  /// The softmax weights of one warp, one row per query head of the KV head's
  /// group (the rows past the group stay zero).
  __shared__ float s_p[kKvarnCandWarps * 16 * kKvarnStage];

  // The row's tile-aligned partition: the same rule the verification-row
  // control and the single-row split-K decoder use (design D8).
  if (row_active && lane == 0) {
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
    const std::uint32_t sequence = pos + 1;
    if (sequence < detail::kSplitKDecodeAttentionMinContext) {
      if (split == 0) {
        end = sequence;
      } else {
        begin = sequence;
      }
    } else {
      const std::uint32_t blocks = (sequence + kTileTokens - 1) / kTileTokens;
      begin = ((split * blocks) / split_count) * kTileTokens;
      end = min((((split + 1) * blocks) / split_count) * kTileTokens,
                sequence);
    }
    s_begin[warp] = begin;
    s_end[warp] = end;
    s_pos[warp] = pos;
  }
  for (std::uint32_t i = tid; i < kKvarnCandWarps * 16 * kKvarnStage;
       i += kKvarnCandThreads) {
    s_p[i] = 0.0F;
  }
  __syncthreads();

  std::uint32_t union_begin = 0xFFFFFFFFU;
  std::uint32_t union_end = 0;
  for (std::uint32_t r = 0; r < rows; ++r) {
    if (s_begin[r] < s_end[r]) {
      union_begin = min(union_begin, s_begin[r]);
      union_end = max(union_end, s_end[r]);
    }
  }

  float running_max[3] = {-INFINITY, -INFINITY, -INFINITY};
  float running_sum[3] = {0.0F, 0.0F, 0.0F};
  float out_acc[kKvarnChunks][3];
#pragma unroll
  for (std::uint32_t cc = 0; cc < kKvarnChunks; ++cc) {
#pragma unroll
    for (std::uint32_t i = 0; i < 3; ++i) {
      out_acc[cc][i] = 0.0F;
    }
  }
  const std::uint32_t head_row = lane & 15U;
  const std::uint32_t key_in_chunk = lane & 15U;
  const std::uint32_t half_wave = lane / 16U;
  /// The row's A-side tile fold, recomputed when the row's tile changes: the
  /// constant RTN zero-point term of this lane's query head (its element `i`
  /// reaches the head `2i + half_wave`, so the score shuffles it across).
  std::uint32_t folded_tile = 0xFFFFFFFFU;
  float folded_zero = 0.0F;

  // The chunk loop and its barriers are block-uniform (the union of the rows'
  // partitions); a row that owns no keys of a chunk skips only its compute.
  for (std::uint32_t chunk = union_begin; chunk < union_end;
       chunk += kKvarnStage) {
    const std::uint32_t count = min(kKvarnStage, union_end - chunk);
    // True when some row reads a position of this chunk as a sealed tile: then
    // the stage carries the codes, and a row that still reads the chunk as F16
    // takes its operand from global memory. A chunk never straddles a tile, so
    // the decision is chunk-uniform for every row.
    bool chunk_decoded = false;
    for (std::uint32_t r = 0; r < rows; ++r) {
      const std::uint32_t lo_r = max(chunk, s_begin[r]);
      const std::uint32_t hi_r = min(chunk + count, s_end[r]);
      if (lo_r < hi_r && lo_r >= kTileTokens &&
          KvarnTileQuantized(s_pos[r], kvarn.tail_length,
                             (lo_r - kTileTokens) / kTileTokens)) {
        chunk_decoded = true;
        break;
      }
    }
    // ---- expand the tile operand once for the whole block (design D14) ----
    for (std::uint32_t i = tid; i < kKvarnStage * kHeadDim;
         i += kKvarnCandThreads) {
      const std::uint32_t token = i / kHeadDim;
      const std::uint32_t channel = i % kHeadDim;
      const std::uint32_t position = chunk + token;
      bool decoded = false;
      std::uint32_t tile = 0;
      std::uint32_t token_in_tile = 0;
      if (position >= kTileTokens) {
        tile = (position - kTileTokens) / kTileTokens;
        token_in_tile = position - kTileTokens - (kTileTokens * tile);
        for (std::uint32_t r = 0; r < rows; ++r) {
          if (position >= s_begin[r] && position < s_end[r] &&
              KvarnTileQuantized(s_pos[r], kvarn.tail_length, tile)) {
            decoded = true;
            break;
          }
        }
      }
      float k_operand = 0.0F;
      float v_operand = 0.0F;
      if (decoded) {
        const std::uint8_t* const payload_k = KvarnTilePayload(
            kvarn.body, layer_idx, kvarn.kv_heads, kv_head, tile,
            kvarn.tile_count, kvarn.k_tile_bytes, kvarn.v_tile_bytes, false);
        const std::uint8_t* const payload_v = KvarnTilePayload(
            kvarn.body, layer_idx, kvarn.kv_heads, kv_head, tile,
            kvarn.tile_count, kvarn.k_tile_bytes, kvarn.v_tile_bytes, true);
        k_operand = KvarnTileCode(payload_k, kvarn.k_bits, token_in_tile,
                                  channel);
        v_operand = KvarnTileCode(payload_v, kvarn.v_bits, token_in_tile,
                                  channel);
      } else {
        k_operand = __half2float(
            KvarnF16Row(kvarn.sink, kvarn.tail, kvarn.sink_plane_elements,
                        kvarn.tail_plane_elements, layer_idx, kvarn.kv_heads,
                        kv_head, position, kvarn.ring_rows, false)[channel]);
        v_operand = __half2float(
            KvarnF16Row(kvarn.sink, kvarn.tail, kvarn.sink_plane_elements,
                        kvarn.tail_plane_elements, layer_idx, kvarn.kv_heads,
                        kv_head, position, kvarn.ring_rows, true)[channel]);
      }
      s_k[i] = __float2half_rn(k_operand);
      s_v[(channel * kKvarnStage) + token] = __float2half_rn(v_operand);
    }
    __syncthreads();

    if (row_active && s_begin[warp] < s_end[warp]) {
      const std::uint32_t lo = max(chunk, s_begin[warp]);
      const std::uint32_t hi = min(chunk + count, s_end[warp]);
      if (lo < hi) {
        const std::uint32_t tile = (lo - kTileTokens) / kTileTokens;
        const bool sealed =
            lo >= kTileTokens &&
            KvarnTileQuantized(pos, kvarn.tail_length, tile);
        const std::uint8_t* payload_k = nullptr;
        const std::uint8_t* payload_v = nullptr;
        if (sealed) {
          payload_k = KvarnTilePayload(kvarn.body, layer_idx, kvarn.kv_heads,
                                       kv_head, tile, kvarn.tile_count,
                                       kvarn.k_tile_bytes, kvarn.v_tile_bytes,
                                       false);
          payload_v = KvarnTilePayload(kvarn.body, layer_idx, kvarn.kv_heads,
                                       kv_head, tile, kvarn.tile_count,
                                       kvarn.k_tile_bytes, kvarn.v_tile_bytes,
                                       true);
        }
        const bool use_lds = sealed || !chunk_decoded;
        const float* const q_row =
            q + (((static_cast<std::size_t>(warp) * num_heads) + head_begin +
                  head_row) *
                 kHeadDim);

        // ---- the tile's RTN zero-point term, once per row and tile ----
        if (sealed && tile != folded_tile) {
          float zero = 0.0F;
          if (head_row < kKvarnGqa) {
            for (std::uint32_t c = 0; c < kHeadDim; ++c) {
              float rtn_scale = 0.0F;
              float rtn_zero = 0.0F;
              KvarnTileKeyRtn(payload_k, c, &rtn_scale, &rtn_zero);
              const float scaled =
                  fmaf(q_row[c], KvarnPayloadChannelScale(payload_k, c), 0.0F);
              zero = fmaf(scaled, rtn_zero, zero);
            }
          }
          folded_zero = zero;
          folded_tile = tile;
        }

        // ---- S = A K^T over the 16 channel chunks of the 16-key chunk ----
        v8f scores_hi = v8f{};
        v8f scores_lo = v8f{};
        for (std::uint32_t cc = 0; cc < kKvarnChunks; ++cc) {
          KvarnFragment a_hi;
          KvarnFragment a_lo;
          KvarnBuildFold(&a_hi, &a_lo, [&](std::uint32_t j) {
            const std::uint32_t channel = (cc * 16) + j;
            if (head_row >= kKvarnGqa) {
              return 0.0F;
            }
            const float query = q_row[channel];
            if (!sealed) {
              return query;
            }
            float rtn_scale = 0.0F;
            float rtn_zero = 0.0F;
            KvarnTileKeyRtn(payload_k, channel, &rtn_scale, &rtn_zero);
            const float scaled =
                fmaf(query, KvarnPayloadChannelScale(payload_k, channel), 0.0F);
            return fmaf(scaled, rtn_scale, 0.0F);
          });
          v16h b;
          if (use_lds) {
            b = CandLoadFrag(&s_k[(key_in_chunk * kHeadDim) + (cc * 16)]);
          } else {
            const __half* const row = KvarnF16Row(
                kvarn.sink, kvarn.tail, kvarn.sink_plane_elements,
                kvarn.tail_plane_elements, layer_idx, kvarn.kv_heads, kv_head,
                chunk + key_in_chunk, kvarn.ring_rows, false);
            b = CandLoadFrag(row + (cc * 16));
          }
          // The WMMA truncates (does not round) its accumulate input, so every
          // product lands in a zero accumulator and folds with one
          // round-to-nearest `fmaf` (design D3; the Phase 0 note).
          const v8f product_hi = CandWmma(a_hi.packed, b, v8f{});
          const v8f product_lo = CandWmma(a_lo.packed, b, v8f{});
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i) {
            scores_hi[i] = fmaf(1.0F, product_hi[i], scores_hi[i]);
            scores_lo[i] = fmaf(1.0F, product_lo[i], scores_lo[i]);
          }
        }

        // ---- online softmax over the row's keys of this chunk ----
#pragma unroll
        for (std::uint32_t i = 0; i < 3; ++i) {
          const std::uint32_t position = chunk + key_in_chunk;
          const bool valid = position >= lo && position < hi;
          // The constant term of the head this lane's element `i` carries, read
          // from the lane that folded it (every lane must reach the shuffle).
          const float zero_term = __shfl(folded_zero, (2 * i) + half_wave);
          float score = -INFINITY;
          if (valid) {
            float raw =
                fmaf(kKvarnResidualFactorInverse, scores_lo[i], scores_hi[i]);
            if (sealed) {
              raw = fmaf(1.0F, zero_term, raw);
            }
            const float token_scale =
                sealed ? KvarnPayloadTokenScale(
                             payload_k,
                             position - kTileTokens - (kTileTokens * tile))
                       : 1.0F;
            score = fmaf(fmaf(raw, token_scale, 0.0F), scale, 0.0F);
          }
          float chunk_max = score;
#pragma unroll
          for (int offset = 8; offset > 0; offset /= 2) {
            chunk_max = fmaxf(chunk_max, __shfl_xor(chunk_max, offset));
          }
          const float new_max = fmaxf(running_max[i], chunk_max);
          const float previous = expf(fmaf(running_max[i], 1.0F, -new_max));
          running_max[i] = new_max;
          running_sum[i] = fmaf(running_sum[i], previous, 0.0F);
#pragma unroll
          for (std::uint32_t cc = 0; cc < kKvarnChunks; ++cc) {
            out_acc[cc][i] = fmaf(out_acc[cc][i], previous, 0.0F);
          }
          const float exponential = valid ? expf(score - new_max) : 0.0F;
          float partial = exponential;
#pragma unroll
          for (int offset = 8; offset > 0; offset /= 2) {
            partial = fmaf(1.0F, partial, __shfl_xor(partial, offset));
          }
          running_sum[i] = fmaf(1.0F, partial, running_sum[i]);
          const float value_scale =
              sealed ? KvarnPayloadTokenScale(
                           payload_v,
                           position - kTileTokens - (kTileTokens * tile))
                     : 1.0F;
          s_p[(((warp * 16) + (2 * i + half_wave)) * 16) + key_in_chunk] =
              fmaf(exponential, value_scale, 0.0F);
        }
        __syncwarp();

        // ---- O += channel_scale_v * (P V + zero point) ----
        // A side: the weights with the V tile's token scale (already folded
        // above) and the RTN scale of the output channel's 128-channel group.
        KvarnFragment a_hi[kKvarnGroups];
        KvarnFragment a_lo[kKvarnGroups];
        float zero_sum[3][kKvarnGroups];
#pragma unroll
        for (std::uint32_t i = 0; i < 3; ++i) {
#pragma unroll
          for (std::uint32_t g = 0; g < kKvarnGroups; ++g) {
            zero_sum[i][g] = 0.0F;
          }
        }
#pragma unroll
        for (std::uint32_t g = 0; g < kKvarnGroups; ++g) {
          KvarnBuildFold(&a_hi[g], &a_lo[g], [&](std::uint32_t j) {
            float value = s_p[((warp * 16) + head_row) * 16 + j];
            if (sealed) {
              float rtn_scale = 0.0F;
              float rtn_zero = 0.0F;
              KvarnTileValueRtn(
                  payload_v,
                  (chunk + j) - kTileTokens - (kTileTokens * tile), g,
                  &rtn_scale, &rtn_zero);
              value = fmaf(value, rtn_scale, 0.0F);
            }
            return value;
          });
        }
        if (sealed) {
#pragma unroll
          for (std::uint32_t i = 0; i < 3; ++i) {
            const float w =
                s_p[((warp * 16) + (2 * i + half_wave)) * 16 + key_in_chunk];
#pragma unroll
            for (std::uint32_t g = 0;
                 g < kKvarnGroups; ++g) {
              float rtn_scale = 0.0F;
              float rtn_zero = 0.0F;
              KvarnTileValueRtn(
                  payload_v,
                  (chunk + key_in_chunk) - kTileTokens - (kTileTokens * tile),
                  g, &rtn_scale, &rtn_zero);
              zero_sum[i][g] = fmaf(w, rtn_zero, 0.0F);
            }
          }
#pragma unroll
          for (std::uint32_t i = 0; i < 3; ++i) {
#pragma unroll
            for (std::uint32_t g = 0;
                 g < kKvarnGroups; ++g) {
              float total = zero_sum[i][g];
#pragma unroll
              for (int offset = 8; offset > 0; offset /= 2) {
                total = fmaf(1.0F, total, __shfl_xor(total, offset));
              }
              zero_sum[i][g] = total;
            }
          }
        }
        // The loop keeps the weight fragments in registers (the compiler
        // resolves the group index per iteration); forcing a full unroll costs
        // 55 spilled VGPRs.
        for (std::uint32_t cc = 0; cc < kKvarnChunks; ++cc) {
          const std::uint32_t group =
              cc / (kKvarnChunks / kKvarnGroups);
          v16h b;
          if (use_lds) {
            b = CandLoadFrag(&s_v[(((cc * 16) + key_in_chunk) * 16)]);
          } else {
            // The row still reads this chunk as F16 while another row reads it
            // as a sealed tile: take the stored row of every key of the chunk
            // straight from global memory (the same values the stage holds
            // when no row needs the codes).
            KvarnFragment global_v;
#pragma unroll
            for (std::uint32_t half = 0; half < 2; ++half) {
              uint4 packed;
              auto* const words = reinterpret_cast<__half*>(&packed);
#pragma unroll
              for (std::uint32_t i = 0; i < 8; ++i) {
                words[i] = KvarnF16Row(
                    kvarn.sink, kvarn.tail, kvarn.sink_plane_elements,
                    kvarn.tail_plane_elements, layer_idx, kvarn.kv_heads,
                    kv_head, chunk + (half * 8) + i,
                    kvarn.ring_rows, true)[(cc * 16) + key_in_chunk];
              }
              global_v.words[half] = packed;
            }
            b = global_v.packed;
          }
          const v8f product_hi = CandWmma(a_hi[group].packed, b, v8f{});
          const v8f product_lo = CandWmma(a_lo[group].packed, b, v8f{});
          const float channel_scale =
              sealed ? KvarnPayloadChannelScale(payload_v,
                                                (cc * 16) + key_in_chunk)
                     : 1.0F;
#pragma unroll
          for (std::uint32_t i = 0; i < 3; ++i) {
            float merged =
                fmaf(kKvarnResidualFactorInverse, product_lo[i], product_hi[i]);
            if (sealed) {
              merged = fmaf(1.0F, zero_sum[i][group], merged);
            }
            out_acc[cc][i] = fmaf(channel_scale, merged, out_acc[cc][i]);
          }
        }
      }
    }
    __syncthreads();
  }

  // ---- partials: the split-K layout the production reduce merges ----
  if (!row_active) {
    return;
  }
  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
  float* const my_scratch =
      scratch + (static_cast<std::size_t>(warp) * num_heads * max_splits *
                 (kHeadDim + 2));
  auto* const partial_outputs = my_scratch + stats_elements;
#pragma unroll
  for (std::uint32_t i = 0; i < 3; ++i) {
    const std::uint32_t head = (2 * i) + half_wave;
    const std::size_t index =
        (static_cast<std::size_t>(head) * max_splits) + split;
    if (key_in_chunk == 0) {
      my_scratch[(index * 2) + 0] = running_max[i];
      my_scratch[(index * 2) + 1] = running_sum[i];
    }
#pragma unroll
    for (std::uint32_t cc = 0; cc < kKvarnChunks; ++cc) {
      partial_outputs[(index * kHeadDim) + (cc * 16) + key_in_chunk] =
          out_acc[cc][i];
    }
  }
}

/// Candidate decode: one block per (KV head, partition) over a whole row group,
/// then the unchanged production reduce (design D14). `forced_splits` (0 = the
/// group-derived count) pins the partition structure so a single-row launch and
/// an R-row launch can be compared directly.
void LaunchKvarnCandidate(const float* q, const float* gate,
                          float* out_context, std::uint32_t layer_idx,
                          std::uint32_t start_pos, std::uint32_t rows,
                          std::uint32_t num_heads, std::uint32_t num_kv_heads,
                          const KvarnReadParams& kvarn,
                          float* split_k_scratch, std::uint32_t forced_splits,
                          hipStream_t stream) {
  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;
  if (split_k_scratch == nullptr || rows == 0 || num_kv_heads == 0) {
    return;
  }
  const std::size_t row_stride =
      static_cast<std::size_t>(num_heads) * max_splits * (kHeadDim + 2);
  const std::size_t width = static_cast<std::size_t>(num_heads) * kHeadDim;
  for (std::uint32_t row = 0; row < rows; row += kKvarnCandWarps) {
    const std::uint32_t group =
        std::min<std::uint32_t>(kKvarnCandWarps, rows - row);
    const std::uint32_t group_start = start_pos + row;
    const std::uint32_t splits =
        forced_splits != 0
            ? forced_splits
            : ((group_start + group >= detail::kSplitKDecodeAttentionMinContext)
                   ? detail::kSplitKDecodeAttentionMaxSplits
                   : 1);
    QwenKvarnBlockDecodePartialsKernel<<<dim3(num_kv_heads, splits),
                                         kKvarnCandThreads, 0, stream>>>(
        q + (row * width), split_k_scratch + (row * row_stride), layer_idx,
        group_start, group, num_heads, num_kv_heads, splits, kvarn);
    QwenKvarnSplitKReduceKernel<<<dim3(num_heads, group), kLanes, 0, stream>>>(
        gate == nullptr ? nullptr : gate + (row * width),
        split_k_scratch + (row * row_stride), out_context + (row * width),
        num_heads, splits);
  }
}

/// The control's read with a pinned split count: the production launcher
/// derives the count from the group's *last* row (`start_pos + batch_size`), so
/// a single-row launch of an earlier row would otherwise run a different
/// partition structure and disagree in the last bits.
void LaunchKvarnControlExplicit(const float* q, const float* gate,
                                float* out_context, std::uint32_t layer_idx,
                                std::uint32_t start_pos, std::uint32_t rows,
                                std::uint32_t num_heads,
                                std::uint32_t num_kv_heads,
                                const KvarnReadParams& kvarn,
                                float* split_k_scratch, std::uint32_t splits,
                                hipStream_t stream) {
  if (split_k_scratch == nullptr || rows == 0 || splits == 0) {
    return;
  }
  BaseQwenKvarnSplitKPartialsKernel<<<dim3(num_heads, rows, splits), kLanes, 0,
                                      stream>>>(
      q, split_k_scratch, layer_idx, start_pos, nullptr, num_heads,
      num_kv_heads, splits, kvarn);
  BaseQwenKvarnSplitKReduceKernel<<<dim3(num_heads, rows), kLanes, 0,
                                    stream>>>(gate, split_k_scratch,
                                              out_context, num_heads, splits);
}

/// The production launcher's group-level split count for one row group.
[[nodiscard]] std::uint32_t KvarnGroupSplits(std::uint32_t start_pos,
                                             std::uint32_t rows) {
  return detail::IsSplitKDecodeAttentionSupported(
             start_pos + rows, kKvarnHeads, kKvarnKvHeads, kCandHeadDim)
             ? detail::SelectDecodeAttentionSplitCount(start_pos + rows)
             : 1;
}

/// One KVarN decode read: the control (the verbatim baseline read) or the
/// candidate block structure (design D14). Both write the same split-K partial
/// layout and are merged by the unchanged production reduce.
void LaunchKvarnRead(bool candidate, const float* q, const float* gate,
                     float* out_context, std::uint32_t layer_idx,
                     std::uint32_t start_pos, std::uint32_t rows,
                     std::uint32_t num_heads, std::uint32_t num_kv_heads,
                     const KvarnReadParams& kvarn, float* split_k_scratch,
                     hipStream_t stream) {
  if (candidate) {
    LaunchKvarnCandidate(q, gate, out_context, layer_idx, start_pos, rows,
                         num_heads, num_kv_heads, kvarn, split_k_scratch, 0,
                         stream);
  } else {
    BaseLaunchKvarnDecodeAttention(q, gate, out_context, layer_idx, start_pos,
                                   nullptr, rows, num_heads, num_kv_heads,
                                   kvarn, split_k_scratch, stream);
  }
}

/// The same read with the split count pinned, so the row-independence check
/// compares runs whose partition structure is identical.
void LaunchKvarnReadPinned(bool candidate, const float* q, const float* gate,
                           float* out_context, std::uint32_t layer_idx,
                           std::uint32_t start_pos, std::uint32_t rows,
                           std::uint32_t num_heads, std::uint32_t num_kv_heads,
                           const KvarnReadParams& kvarn, float* split_k_scratch,
                           std::uint32_t splits, hipStream_t stream) {
  if (candidate) {
    LaunchKvarnCandidate(q, gate, out_context, layer_idx, start_pos, rows,
                         num_heads, num_kv_heads, kvarn, split_k_scratch,
                         splits, stream);
  } else {
    LaunchKvarnControlExplicit(q, gate, out_context, layer_idx, start_pos,
                               rows, num_heads, num_kv_heads, kvarn,
                               split_k_scratch, splits, stream);
  }
}
/// Row group a depth supports: the fixture is always built for the widest group
/// (`kKvarnMaxRows`), clamped by the depth, so one fixture per depth serves
/// every row count the gate reports.
struct KvarnGroup {
  std::uint32_t base{0};
  std::uint32_t rows{0};
};

[[nodiscard]] KvarnGroup KvarnGroupFor(std::uint32_t depth) {
  const std::uint32_t rows = std::min(depth, kKvarnMaxRows);
  return {depth - rows, rows};
}

[[nodiscard]] std::vector<std::uint32_t> KvarnDepths(const Options& options) {
  if (!options.series_filter.empty()) {
    return options.series_filter;
  }
  return {32768, 131072};
}

/// The row counts of the G0K grid (R=1 and the verification group R=8), plus an
/// explicit `--rows` when it is neither.
[[nodiscard]] std::vector<std::uint32_t> KvarnRowCounts(const Options& options) {
  std::vector<std::uint32_t> rows{1, 8};
  if (options.rows != 1 && options.rows != 8) {
    rows.push_back(options.rows);
  }
  return rows;
}

void PrintKvarnTiming(const char* who, const char* launcher,
                      std::uint32_t rows, const Timing& timing) {
  printf("  %-9s %-6s rows=%u median %.4f ms  samples", who, launcher, rows,
         timing.median_ms);
  for (double sample : timing.samples) {
    printf(" %.4f", sample);
  }
  printf("\n");
}

void RunKvarnCold(const Options& options, const KvarnDevice& device) {
  for (const std::uint32_t depth : KvarnDepths(options)) {
    const KvarnGroup group = KvarnGroupFor(depth);
    const KvarnFixture fixture(group.base, group.rows, options.kvarn_k_bits,
                               options.kvarn_v_bits, options.kvarn_tail);
    printf("kvarn-cold depth=%u preset=kvarn-k%uv%u tail=%u tiles=%u\n", depth,
           static_cast<unsigned>(options.kvarn_k_bits),
           static_cast<unsigned>(options.kvarn_v_bits), options.kvarn_tail,
           fixture.tile_count);
    for (const std::uint32_t rows : KvarnRowCounts(options)) {
      if (rows > group.rows) {
        continue;
      }
      // Every row ends at depth-1: row r sits at start + r.
      const std::uint32_t start = depth - rows;
      if (options.control) {
        const auto decode = [&] {
          LaunchKvarnRead(false, device.d_query, device.d_gate, device.d_out,
                          0, start, rows, kKvarnHeads, kKvarnKvHeads,
                          fixture.read, device.d_scratch, nullptr);
          HIP_CHECK(hipGetLastError());
        };
        PrintKvarnTiming("control", "decode", rows,
                         TimeLaunch(decode, options.iters));
        const auto causal = [&] {
          BaseLaunchKvarnCausalDecodeAttention(
              device.d_query, device.d_gate, device.d_out, 0, start, rows,
              kKvarnHeads, kKvarnKvHeads, fixture.read, device.d_scratch,
              nullptr);
          HIP_CHECK(hipGetLastError());
        };
        PrintKvarnTiming("control", "causal", rows,
                         TimeLaunch(causal, options.iters));
      }
      if (options.candidate) {
        const auto candidate = [&] {
          LaunchKvarnRead(true, device.d_query, device.d_gate, device.d_out, 0,
                          start, rows, kKvarnHeads, kKvarnKvHeads,
                          fixture.read, device.d_scratch, nullptr);
          HIP_CHECK(hipGetLastError());
        };
        PrintKvarnTiming("candidate", "block", rows,
                         TimeLaunch(candidate, options.iters));
      }
    }
  }
}

int RunKvarnAccuracy(const Options& options, const KvarnDevice& device) {
  const std::uint32_t rows = std::min(options.rows, kKvarnMaxRows);
  for (const std::uint32_t depth : KvarnDepths(options)) {
    const KvarnGroup group = KvarnGroupFor(depth);
    const KvarnFixture fixture(group.base, group.rows, options.kvarn_k_bits,
                               options.kvarn_v_bits, options.kvarn_tail);
    const std::uint32_t start = depth - rows;
    printf("kvarn-accuracy depth=%u preset=kvarn-k%uv%u tail=%u rows=%u\n",
           depth, static_cast<unsigned>(options.kvarn_k_bits),
           static_cast<unsigned>(options.kvarn_v_bits), options.kvarn_tail,
           rows);
    for (const bool candidate : {false, true}) {
      if (candidate ? !options.candidate : !options.control) {
        continue;
      }
      LaunchKvarnRead(candidate, device.d_query, device.d_gate, device.d_out,
                      0, start, rows, kKvarnHeads, kKvarnKvHeads, fixture.read,
                      device.d_scratch, nullptr);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      const std::vector<float> output = device.ReadOutput(rows);
      for (std::uint32_t row = 0; row < rows; ++row) {
        // The gate of the row being compared (the device gate is per row).
        const std::vector<float> row_gate(
            device.gate.begin() + static_cast<std::ptrdiff_t>(row) * kKvarnWidth,
            device.gate.begin() +
                static_cast<std::ptrdiff_t>(row + 1) * kKvarnWidth);
        const std::vector<double> reference =
            KvarnReferenceRow(fixture, start + row, device.query, row_gate);
        const std::vector<float> row_output(
            output.begin() + static_cast<std::ptrdiff_t>(row) * kKvarnWidth,
            output.begin() +
                static_cast<std::ptrdiff_t>(row + 1) * kKvarnWidth);
        const ErrorStats stats = CompareWithReference(row_output, reference);
        printf("  %-9s row=%u pos=%u max-abs %.6e rms %.6e\n",
               candidate ? "candidate" : "control", row, start + row,
               stats.max_abs, stats.rms);
      }
    }
  }
  return 0;
}

/// Byte comparison of every row of an R-row KVarN launch against the same row
/// launched alone (design D3/D14 row independence).
int RunKvarnRowIndependence(const Options& options,
                            const KvarnDevice& device) {
  // The shallow set of tests/models/qwen/hip/attention/kv_storage_ops_test.cpp
  // plus the split-K threshold, and the depth the gate reports.
  std::vector<std::uint32_t> positions = {0,    17,   120,   124,   127,
                                          128,  504,  508,   511,   512,
                                          4093, 4095, 4096,  4097};
  positions.push_back(options.start_pos);

  int mismatches = 0;
  for (const std::uint32_t base : positions) {
    // One fixture per base position: the F16 tail window of design D2 is
    // rooted at that row's own view, and the ring cannot hold the windows of
    // two far-apart bases at once.
    const KvarnFixture fixture(base, kKvarnMaxRows, options.kvarn_k_bits,
                               options.kvarn_v_bits, options.kvarn_tail);
    for (const std::uint32_t rows : {2U, 3U, 8U}) {
      // The partition structure belongs to the launch, not to the row: the
      // production launcher derives the split count from the group's last row
      // (`start_pos + batch_size`), so a row launched alone would otherwise run
      // a different partition structure and disagree in the last bits. Pin the
      // group's count for the row-alone runs of both targets.
      const std::uint32_t splits = KvarnGroupSplits(base, rows);
      for (const bool candidate : {false, true}) {
        if (candidate ? !options.candidate : !options.control) {
          continue;
        }
        LaunchKvarnReadPinned(candidate, device.d_query, device.d_gate,
                              device.d_out, 0, base, rows, kKvarnHeads,
                              kKvarnKvHeads, fixture.read, device.d_scratch,
                              splits, nullptr);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        const std::vector<float> group_output = device.ReadOutput(rows);
        int row_mismatches = 0;
        for (std::uint32_t row = 0; row < rows; ++row) {
          LaunchKvarnReadPinned(candidate, device.d_query, device.d_gate,
                                device.d_out, 0, base + row, 1, kKvarnHeads,
                                kKvarnKvHeads, fixture.read, device.d_scratch,
                                splits, nullptr);
          HIP_CHECK(hipGetLastError());
          HIP_CHECK(hipDeviceSynchronize());
          const std::vector<float> single_output = device.ReadOutput(1);
          if (std::memcmp(group_output.data() + row * kKvarnWidth,
                          single_output.data(),
                          kKvarnWidth * sizeof(float)) != 0) {
            ++row_mismatches;
          }
        }
        mismatches += row_mismatches;
        printf("kvarn-r1-check %-9s start=%u rows=%u splits=%u mismatches=%d\n",
               candidate ? "candidate" : "control", base, rows, splits,
               row_mismatches);
      }
    }
  }
  return mismatches == 0 ? 0 : 1;
}

void PrintUsage() {
  printf(
      "usage: qwen_decode_attention_bench [options]\n"
      "  --mode cold|resident|accuracy|r1-check|stats|kvarn-cold|kvarn-accuracy|"
      "kvarn-r1-check (default cold)\n"
      "  --rows N              verification rows          (default 1)\n"
      "  --start-pos P         last position of the rows  (default 131071)\n"
      "  --partitions a|b      partition method           (default a)\n"
      "  --candidate-mode N    bench candidate variant (7 = production kernel)\n"
      "  --split-qp off|q|p|both  high+residual split      (default both)\n"
      "  --only control|candidate|both                    (default both)\n"
      "  --input formula|subnormal|all   accuracy inputs   (default all)\n"
      "  --series N[,N...]     accuracy series lengths      (default all)\n"
      "  --preset kvarn-k5v4   KVarN K/V bit widths       (default kvarn-k5v4)\n"
      "  --tail N              KVarN F16 tail rows        (default 2048)\n"
      "  --iters N             timed iterations           (default 5)\n"
      "  --sweep LO HI STEP    resident sweep             (default 2304 9216 "
      "1152)\n");
}

}  // namespace
}  // namespace gufo::hip

int main(int argc, char** argv) {
  using gufo::hip::Options;
  using gufo::hip::HostInputs;
  using gufo::hip::DeviceState;

  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", flag.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (flag == "--mode") {
      options.mode = next();
    } else if (flag == "--rows") {
      options.rows = static_cast<std::uint32_t>(std::stoul(next()));
    } else if (flag == "--start-pos") {
      options.start_pos = static_cast<std::uint32_t>(std::stoul(next()));
    } else if (flag == "--partitions") {
      const std::string value = next();
      options.method_b = value == "b";
    } else if (flag == "--split-qp") {
      const std::string value = next();
      options.split_mode = value == "off"   ? 0
                           : value == "q"   ? 1
                           : value == "p"   ? 2
                                            : 3;
    } else if (flag == "--only") {
      const std::string value = next();
      options.control = value != "candidate";
      options.candidate = value != "control";
    } else if (flag == "--iters") {
      options.iters = std::stoi(next());
    } else if (flag == "--input") {
      const std::string value = next();
      options.input_filter =
          value == "small"    ? 1
          : value == "zero-q" ? 2
          : value == "small0" ? 3
          : value == "subnormal" ? 4
                                 : 0;
    } else if (flag == "--series") {
      std::string value = next();
      options.series_filter.clear();
      std::size_t begin = 0;
      while (begin <= value.size()) {
        const std::size_t comma = value.find(',', begin);
        const std::string item =
            value.substr(begin, comma == std::string::npos ? std::string::npos
                                                           : comma - begin);
        if (!item.empty()) {
          options.series_filter.push_back(
              static_cast<std::uint32_t>(std::stoul(item)));
        }
        if (comma == std::string::npos) {
          break;
        }
        begin = comma + 1;
      }
    } else if (flag == "--preset") {
      const std::string value = next();
      // kvarn-k<K>v<V>, e.g. kvarn-k5v4.
      if (value.size() != 10 || value.compare(0, 7, "kvarn-k") != 0 ||
          value[8] != 'v' || value[7] < '0' || value[7] > '9' ||
          value[9] < '0' || value[9] > '9') {
        fprintf(stderr, "unknown preset %s (want kvarn-k<K>v<V>)\n",
                value.c_str());
        gufo::hip::PrintUsage();
        return 2;
      }
      options.kvarn_k_bits = static_cast<std::uint8_t>(value[7] - '0');
      options.kvarn_v_bits = static_cast<std::uint8_t>(value[9] - '0');
    } else if (flag == "--tail") {
      options.kvarn_tail = static_cast<std::uint32_t>(std::stoul(next()));
    } else if (flag == "--candidate-mode") {
      options.split_mode = std::stoi(next());
    } else if (flag == "--sweep") {
      options.sweep_lo = static_cast<std::uint32_t>(std::stoul(next()));
      options.sweep_hi = static_cast<std::uint32_t>(std::stoul(next()));
      options.sweep_step = static_cast<std::uint32_t>(std::stoul(next()));
    } else if (flag == "--help" || flag == "-h") {
      gufo::hip::PrintUsage();
      return 0;
    } else {
      fprintf(stderr, "unknown option %s\n", flag.c_str());
      gufo::hip::PrintUsage();
      return 2;
    }
  }

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, device));
  printf("device %s (gfx1151 target, %d CUs)\n", properties.name,
         properties.multiProcessorCount);
  printf("base revision %s\n", gufo::hip::kBaseRevision);
  printf("control: verbatim copy of QwenDecodeSplitKAttentionHalfPartialsKernel "
         "and its launcher\n");
  printf("candidate: block=(KV head, row, partition), 6 heads in WMMA M, "
         "16-token LDS stages, 4 warps\n");
  if (options.mode.compare(0, 5, "kvarn") == 0) {
    printf(
        "kvarn control: verbatim copy of the base KVarN read kernels "
        "(%s)\n",
        "tools/bench/detail/qwen_kvarn_read_base_801abd3.inc");
    printf(
        "kvarn candidate: block=(KV head, partition) over the whole row "
        "group, 6 heads in WMMA M, one 16-key tile stage in LDS shared by "
        "every row,\n"
        "  exact f16 codes as the B operand with the RTN fold on the A side "
        "(design D2/D14)\n");
    printf(
        "kvarn preset kvarn-k%uv%u tail=%u; the model configuration the "
        "G0K gate fixes\n",
        static_cast<unsigned>(options.kvarn_k_bits),
        static_cast<unsigned>(options.kvarn_v_bits), options.kvarn_tail);
  }
  printf("mode %s\n", options.mode.c_str());

  HostInputs inputs;
  DeviceState state;
  // Headroom for the largest row count: an R-row launch reads positions up to
  // start_pos + R - 1.
  inputs.max_context = 131072 + 8;
  gufo::hip::BuildInputs(options.input_filter >= 0 ? options.input_filter
                                                   : options.input,
                         24, 4, inputs.max_context, inputs.query, inputs.gate,
                         inputs.cache);
  const std::size_t width = 24 * gufo::hip::kCandHeadDim;
  const std::size_t cache_elements =
      static_cast<std::size_t>(inputs.max_context) * 4 * gufo::hip::kCandHeadDim;
  const std::size_t max_rows = 8;
  HIP_CHECK(hipMalloc(&state.q, max_rows * width * sizeof(float)));
  HIP_CHECK(hipMalloc(&state.gate, max_rows * width * sizeof(float)));
  HIP_CHECK(hipMalloc(&state.k_cache, cache_elements * sizeof(__half)));
  HIP_CHECK(hipMalloc(&state.v_cache, cache_elements * sizeof(__half)));
  HIP_CHECK(hipMalloc(&state.out, max_rows * width * sizeof(float)));
  HIP_CHECK(hipMalloc(&state.reference_out, width * sizeof(float)));
  const std::size_t row_scratch =
      gufo::hip::detail::DecodeAttentionScratchElements(
          24, gufo::hip::kCandHeadDim);
  const std::size_t scratch_elements = row_scratch * max_rows;
  HIP_CHECK(hipMalloc(&state.scratch, scratch_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&state.reference_scratch,
                      row_scratch * sizeof(float)));
  for (std::uint32_t row = 0; row < max_rows; ++row) {
    HIP_CHECK(hipMemcpy(state.q + (row * width), inputs.query.data(),
                        width * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(state.gate + (row * width), inputs.gate.data(),
                        width * sizeof(float), hipMemcpyHostToDevice));
  }
  HIP_CHECK(hipMemcpy(state.k_cache, inputs.cache.data(),
                      cache_elements * sizeof(__half),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(state.v_cache, inputs.cache.data(),
                      cache_elements * sizeof(__half),
                      hipMemcpyHostToDevice));

  int status = 0;
  if (options.mode == "cold") {
    gufo::hip::RunCold(options, inputs, state);
  } else if (options.mode == "resident") {
    status = gufo::hip::RunResident(options, inputs, state);
  } else if (options.mode == "accuracy") {
    status = gufo::hip::RunAccuracy(options, inputs, state);
  } else if (options.mode == "stats") {
    status = gufo::hip::RunStatsCheck(options, inputs, state);
  } else if (options.mode == "r1-check") {
    status = gufo::hip::RunRowIndependence(options, inputs, state);
  } else if (options.mode == "kvarn-cold") {
    gufo::hip::KvarnDevice kvarn_device;
    gufo::hip::RunKvarnCold(options, kvarn_device);
  } else if (options.mode == "kvarn-accuracy") {
    gufo::hip::KvarnDevice kvarn_device;
    status = gufo::hip::RunKvarnAccuracy(options, kvarn_device);
  } else if (options.mode == "kvarn-r1-check") {
    gufo::hip::KvarnDevice kvarn_device;
    status = gufo::hip::RunKvarnRowIndependence(options, kvarn_device);
  } else {
    fprintf(stderr, "unknown mode %s\n", options.mode.c_str());
    status = 2;
  }
  return status;
}
#else
int main() {
  return 77;
}
#endif  // defined(ENGINE_ENABLE_HIP)
