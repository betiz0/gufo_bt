#ifndef GUFO_MODELS_QWEN_HIP_KVARN_HPP_
#define GUFO_MODELS_QWEN_HIP_KVARN_HPP_

// Production KVarN KV storage for Qwen3.8 27B full attention (design D1-D7).
// Qwen-only: the kernels, state and tests live in the model tree.
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"

namespace gufo::hip {

/// Per-session KVarN seal state: one sealed-tile counter per (layer, KV head),
/// the FP32 tile work area of the seal, and the device non-finite error flag
/// (design D4/D15). Never synchronizes on its own.
class KvarnSealState {
public:
  KvarnSealState(std::uint32_t attention_layers, std::uint32_t kv_heads);
  ~KvarnSealState();
  KvarnSealState(const KvarnSealState&) = delete;
  KvarnSealState& operator=(const KvarnSealState&) = delete;

  /// Forgets every sealed tile and clears the error flag.
  void Reset(hipStream_t stream);
  /// Reads the non-finite error flag at an existing synchronisation point
  /// (design D15). Never adds a synchronization.
  [[nodiscard]] bool HasError(hipStream_t stream) const;

  [[nodiscard]] std::uint32_t* sealed() const noexcept { return sealed_; }
  [[nodiscard]] float* scratch() const noexcept { return scratch_; }
  [[nodiscard]] std::uint32_t* error_flag() const noexcept { return error_; }

private:
  std::uint32_t layers_;
  std::uint32_t kv_heads_;
  std::uint32_t* sealed_{nullptr};
  float* scratch_{nullptr};
  std::uint32_t* error_{nullptr};
};

/// Seals every tile that the view rule of design D2 selects for the query rows
/// `[start, start + count)`, for one full-attention layer (design D4). The
/// launch is unconditional at the step start; a device-side predicate turns it
/// into a no-op when no tile is due, so HIP graph capture is unaffected.
///
/// `sink` and `tail` are the F16 regions, each holding the K plane followed by
/// the V plane; `body` is the sealed-tile region. A non-finite tile raises the
/// session error flag and is left unsealed. `chunk` is the bounded per-chunk
/// F16 workspace (design D4/4.1): rows of the current chunk are read from
/// there, older rows from the sink and ring, which a prefill chunk has not
/// overwritten yet.
void LaunchKvarnSeal(const KvarnSealState& state,
                     const QwenKvarnPreset& preset, void* sink, void* tail,
                     std::uint64_t sink_plane_elements,
                     std::uint64_t tail_plane_elements, std::uint32_t ring_rows,
                     std::uint8_t* body, std::uint32_t tile_count,
                     std::uint32_t layer_idx, std::uint32_t num_kv_heads,
                     std::uint32_t start, const std::uint32_t* d_start,
                     std::uint32_t count, hipStream_t stream,
                     const void* chunk = nullptr,
                     std::uint64_t chunk_plane_elements = 0);

/// Merges the bounded per-chunk F16 workspace back into the sink and the tail
/// ring (design D4/4.1), so the next step reads the same rows as a sequential
/// decode. Runs after the chunk's attention; the chunk is indexed by its row
/// within `[chunk_begin, chunk_begin + chunk_rows)`.
void LaunchKvarnChunkFlush(const KvarnWriteParams& kvarn,
                           std::uint32_t layer_idx, std::uint32_t num_kv_heads,
                           std::uint32_t chunk_begin, std::uint32_t chunk_rows,
                           hipStream_t stream);

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_KVARN_HPP_
