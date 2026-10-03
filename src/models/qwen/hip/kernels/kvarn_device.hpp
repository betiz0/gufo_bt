#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_KVARN_DEVICE_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_KVARN_DEVICE_HPP_

// KVarN device helpers shared by the write, seal and read kernels (design D3).
// Qwen-only: nothing here is shared with another model.

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/models/qwen/kvarn_reference.hpp"

namespace gufo::hip {

/// Normalized Hadamard rotation of one head_dim 256 row held in shared memory
/// (design D3): eight butterfly stages over the whole row, then the 1/16 scale.
/// The transform is symmetric, so this same call inverse-rotates. `row` holds
/// `models::qwen::kKvarnHeadDim` floats and every thread in the block must call
/// it (`tid` indexes the row).
__device__ __forceinline__ void KvarnFwhtRowShared(float* row, int tid) {
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

/// Normalized Hadamard rotation of one head_dim 256 row held in shared memory
/// for a single-warp block (design D3): eight butterfly stages over the whole
/// row, then the 1/16 scale. The transform is symmetric, so this same call
/// inverse-rotates. `row` holds `models::qwen::kKvarnHeadDim` floats and every
/// lane of the warp must call it; the caller stages the row in shared memory.
__device__ __forceinline__ void KvarnFwhtRowWarp(float* row, int lane) {
  constexpr int kCols = static_cast<int>(models::qwen::kKvarnHeadDim);
  for (int span = 1; span < kCols; span <<= 1) {
    __syncwarp();
    for (int index = lane; index < kCols / 2; index += 32) {
      const int j = index & (span - 1);
      const int base = (index - j) * 2;
      const float a = row[base + j];
      const float b = row[base + j + span];
      row[base + j] = a + b;
      row[base + j + span] = a - b;
    }
  }
  __syncwarp();
  for (int index = lane; index < kCols; index += 32) {
    row[index] *= 0.0625F;
  }
  __syncwarp();
}

/// Shared-memory F16 row of one KVarN sink or tail region, indexed by the
/// absolute position (design D1/D2).
__device__ __forceinline__ std::size_t KvarnF16RowOffset(
    std::uint32_t layer, std::uint32_t kv_heads, std::uint32_t head,
    std::uint32_t position, std::uint32_t ring_rows) {
  if (position < models::qwen::kKvarnSinkRows) {
    return models::qwen::KvarnSinkRowOffset(layer, kv_heads, head, position);
  }
  return models::qwen::KvarnTailRowOffset(layer, kv_heads, head, position,
                                          ring_rows);
}

/// Chooses the K or V plane of a KVarN region pair from the row offset.
__device__ __forceinline__ __half* KvarnPlaneRow(__half* first_plane,
                                                 std::uint64_t plane_elements,
                                                 std::size_t row_offset,
                                                 bool is_value) {
  return first_plane + (is_value ? plane_elements : 0) + row_offset;
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_KVARN_DEVICE_HPP_
