#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_BASELINE_QUANT_DEVICE_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_BASELINE_QUANT_DEVICE_HPP_

// Baseline probes of the KV quantization quality evaluation (design D1):
// "rotate, quantize in blocks of 32 with an FP16 scale, dequantize, inverse
// rotate" over one head row held in shared memory. Only
// `GUFO_QWEN_PSEUDO_KVARN` builds compile this file, so no production binary
// carries the probe path.

#if defined(ENGINE_ENABLE_HIP) && defined(GUFO_QWEN_PSEUDO_KVARN)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "src/models/qwen/kvarn_reference.hpp"

namespace gufo::hip {

/// Rotation of the baseline probes (design D1): the normalized Hadamard
/// transform, kept as a probe-private copy. The baselines and their teachers
/// are measured once and reused, so a later KVarN rotation change (design D7)
/// must never move this ruler even though both start out identical.
/// `row` holds `models::qwen::kKvarnHeadDim` floats and every thread of the
/// block must call it (`tid` indexes the row). The transform is symmetric, so
/// this same call inverse-rotates.
__device__ __forceinline__ void BaselineFwhtRowShared(float* row, int tid) {
  constexpr int kCols = static_cast<int>(models::qwen::kKvarnHeadDim);
  for (int span = 1; span < kCols; span <<= 1) {
    __syncthreads();
    if (tid < kCols / 2) {
      const int j = tid & (span - 1);
      const int base = (tid - j) * 2;
      const float a = row[base + j];
      const float b = row[base + j + span];
      row[base + j] = a + b;
      row[base + j + span] = a - b;
    }
  }
  __syncthreads();
  row[tid] *= 0.0625F;
  __syncthreads();
}

/// One element of the E_q8 / E_q4 round trip: the signed code nearest to
/// `value / scale`, clamped to the code range, reconstructed with `scale`.
__device__ __forceinline__ void DequantizeBaselineValue(float* dequantized,
                                                        float value,
                                                        float scale,
                                                        float code_min,
                                                        float code_max) {
  const float code = rintf(value / scale);
  *dequantized = fminf(fmaxf(code, code_min), code_max) * scale;
}

/// Quantizes the shared row with the E_q8 / E_q4 block quantizer and
/// dequantizes it back in place (design D1): 32-value blocks, an FP16 scale,
/// and a signed code. `kBits == 8` uses the q8_0 convention (scale = amax/127,
/// codes in [-127, 127]) and `kBits == 4` the q4_0 one (scale = the signed
/// maximum divided by -8, codes in [-8, 7]). One warp handles one of the eight
/// blocks, so every thread of the block must call it once per row.
template<int kBits>
__device__ __forceinline__ void BaselineQuantizeRowShared(float* row, int tid) {
  static_assert(kBits == 8 || kBits == 4);
  constexpr int kBlockValues = 32;
  constexpr float kCodeMin = kBits == 8 ? -127.0F : -8.0F;
  constexpr float kCodeMax = kBits == 8 ? 127.0F : 7.0F;
  const int lane = tid & (kBlockValues - 1);
  const int index = ((tid / kBlockValues) * kBlockValues) + lane;
  const float value = row[index];

  // The block maximum: q8_0 scales by the largest magnitude, q4_0 by the
  // signed value that carries it.
  float amax = fabsf(value);
  float signed_max = value;
#pragma unroll
  for (int offset = kBlockValues / 2; offset > 0; offset >>= 1) {
    const float other_amax = __shfl_xor(amax, offset);
    const float other_signed = __shfl_xor(signed_max, offset);
    if (other_amax > amax) {
      amax = other_amax;
      signed_max = other_signed;
    }
  }

  const float scale = kBits == 8 ? amax / kCodeMax : signed_max / kCodeMin;
  float dequantized = 0.0F;
  if (scale != 0.0F) {
    // The block scale is stored as FP16, so the codes and the reconstruction
    // both use the rounded value.
    const float rounded_scale = __half2float(__float2half_rn(scale));
    DequantizeBaselineValue(&dequantized, value, rounded_scale, kCodeMin,
                            kCodeMax);
  }
  row[index] = dequantized;
}

/// Stores one dense FP16 row the way the baseline probes define it (design
/// D1): the sink rows (absolute position below `kKvarnSinkRows`) keep the
/// plain row, every later row is rotated, quantized and dequantized in blocks
/// of 32 and inverse-rotated back. `row` holds the full head row and every
/// thread of the block must call it. `bits` is 8 for E_q8 and 4 for E_q4.
__device__ __forceinline__ void BaselineRoundTripRow(float* row, int tid,
                                                     std::uint8_t bits,
                                                     bool quantize) {
  if (!quantize) {
    return;
  }
  BaselineFwhtRowShared(row, tid);
  if (bits == 8) {
    BaselineQuantizeRowShared<8>(row, tid);
  } else {
    BaselineQuantizeRowShared<4>(row, tid);
  }
  // The quantizer wrote shared memory: the inverse rotation must see it.
  BaselineFwhtRowShared(row, tid);
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP) && defined(GUFO_QWEN_PSEUDO_KVARN)

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_BASELINE_QUANT_DEVICE_HPP_
