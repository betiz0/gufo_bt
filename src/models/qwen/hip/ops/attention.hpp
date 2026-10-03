#ifndef GUFO_MODELS_QWEN_HIP_OPS_ATTENTION_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_ATTENTION_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/vision/rope.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

namespace gufo::hip {

/// KVarN decode-write destination (design D1/D3/D7). When `sink` is non-null
/// the fused write stores the rotated K/V into the KVarN regions instead of the
/// canonical FP16 plane: positions below the sink go to `sink`, later ones wrap
/// through `tail`. K is the first plane, V the second.
struct KvarnWriteParams {
  __half* sink{nullptr};
  __half* tail{nullptr};
  std::uint64_t sink_plane_elements{0};
  std::uint64_t tail_plane_elements{0};
  std::uint32_t ring_rows{0};
  /// Bounded per-chunk F16 workspace (design D4/4.1). During prefill the
  /// fused write stores each rotated row here, indexed by its row within the
  /// chunk, instead of the sink/ring, so the sink and the L+128+W ring keep
  /// the pre-chunk rows the view rule still reads as F16. `LaunchKvarnChunkFlush`
  /// merges the chunk into the sink and ring after attention.
  __half* chunk{nullptr};
  std::uint64_t chunk_plane_elements{0};

  __host__ __device__ [[nodiscard]] bool enabled() const noexcept {
    return sink != nullptr;
  }
};

/// KVarN read source (design D1/D2/D7). The three regions replace the canonical
/// FP16 planes: the sink and the tail ring hold F16 rows, the body holds sealed
/// tiles. `tail_length` is the view-rule tail (W when `--kv-cache-tail` was not
/// given). K is the first plane, V the second.
struct KvarnReadParams {
  const __half* sink{nullptr};
  const __half* tail{nullptr};
  const std::uint8_t* body{nullptr};
  std::uint64_t sink_plane_elements{0};
  std::uint64_t tail_plane_elements{0};
  std::uint32_t ring_rows{0};
  std::uint32_t tile_count{0};
  std::uint32_t k_tile_bytes{0};
  std::uint32_t v_tile_bytes{0};
  std::uint32_t kv_heads{0};
  std::uint32_t tail_length{0};
  std::uint8_t k_bits{4};
  std::uint8_t v_bits{4};
  /// Bounded per-chunk F16 workspace (design D4/4.1), set only while a prefill
  /// chunk is being read: a position inside the current chunk is served from
  /// here, everything older from the sink or the tail ring, which the deferred
  /// flush has not overwritten yet.
  const __half* chunk{nullptr};
  std::uint64_t chunk_plane_elements{0};

  __host__ __device__ [[nodiscard]] bool enabled() const noexcept {
    return sink != nullptr;
  }
};

/// Rejects an attention launch that has neither a canonical FP16/FP32 KV plane
/// nor a KVarN read source. A KVarN policy leaves the canonical planes null, so
/// a non-KVarN launcher reached with them is a wiring error rather than a
/// silent fallback (design D7; tasks 3.13). Qwen-only helper.
inline void RequireAttentionKvSource(const void* k_cache, const void* v_cache,
                                     const void* k_cache_f16,
                                     const void* v_cache_f16, bool kvarn) {
  const bool fp32_planes = k_cache != nullptr && v_cache != nullptr;
  const bool fp16_planes = k_cache_f16 != nullptr && v_cache_f16 != nullptr;
  if (!kvarn && !fp32_planes && !fp16_planes) {
    throw std::runtime_error(
        "attention requires a KV cache plane or KVarN read parameters");
  }
}

/// Computes Fused QKV Projections for Full Attention layers in a single kernel
void LaunchFusedQKVProjections(const void* q_w, core::GgmlType q_type,
                               const void* k_w, core::GgmlType k_type,
                               const void* v_w, core::GgmlType v_type,
                               const float* x, float* q_out, float* k_out,
                               float* v_out, std::size_t q_dim,
                               std::size_t kv_dim, std::size_t hidden_size,
                               hipStream_t stream = nullptr);

/// Computes Grouped-Query Softmax Attention with KV-cache and optional gating
/// on GPU. Production supplies canonical FP16 K/V; the independent validation
/// fallback supplies FP32 K/V. Tests may supply both representations. When
/// skip_kv_write is true the selected KV cache is assumed already written
/// (e.g. by the fused QK norm+RoPE kernel, opt-c010-qk-rope-kv).
void LaunchAttention(const float* q, const float* k, const float* v,
                     const float* gate, float* k_cache, float* v_cache,
                     void* k_cache_f16, void* v_cache_f16, float* out_context,
                     std::uint32_t layer_idx, std::uint32_t pos,
                     std::uint32_t max_context, std::uint32_t num_heads,
                     std::uint32_t num_kv_heads, std::uint32_t head_dim,
                     hipStream_t stream = nullptr,
                     float* split_k_scratch = nullptr,
                     bool skip_kv_write = false);

/// Decode attention over the KVarN sink, sealed tiles and F16 tail ring
/// (design D2/D7/D8). One launch covers `batch_size` consecutive query rows;
/// the dequantized tensor is never materialized. `d_pos`, when non-null,
/// supplies the first row's absolute position from device memory so a HIP graph
/// carries no baked host position. `split_k_scratch`, when non-null and the
/// context is long enough, selects the tile-aligned split-K path whose partials
/// use the `DecodeAttentionScratchElements` layout.
void LaunchKvarnDecodeAttention(
    const float* q, const float* gate, float* out_context,
    std::uint32_t layer_idx, std::uint32_t start_pos,
    const std::uint32_t* d_pos, std::size_t batch_size, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, const KvarnReadParams& kvarn,
    float* split_k_scratch = nullptr, hipStream_t stream = nullptr);

/// Runs consecutive verification queries with scalar decode arithmetic.
/// The selected KV cache must already contain all rows. Each row sees only
/// its causal prefix. Scratch for batch_size rows batches split-K too; a
/// smaller span retains the scalar split-K fallback.
///
/// When `kvarn` is enabled the rows read the KVarN sink, sealed tiles and F16
/// tail ring through the view rule of design D2 instead of the canonical FP16
/// planes (design D7). Up to `kQwenKvarnVerificationRows` rows share the
/// dequantized tile staging in LDS, and every row keeps the tile-aligned
/// split-K partition of the single-row decoder, so its logits stay
/// bit-identical to the AR decode at the same position (T-05).
void LaunchCausalDecodeAttention(
    const float* q, const float* gate, float* k_cache, float* v_cache,
    void* k_cache_f16, void* v_cache_f16, float* out_context,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream = nullptr, std::span<float> split_k_scratch = {},
    KvarnReadParams kvarn = {});

/// KVarN verification-row implementation of `LaunchCausalDecodeAttention`
/// (T-05, design D2/D7/D8). Kept Qwen-only in `kernels/kvarn_read.hip`.
void LaunchKvarnCausalDecodeAttention(
    const float* q, const float* gate, float* out_context,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t num_heads, std::uint32_t num_kv_heads,
    const KvarnReadParams& kvarn, float* split_k_scratch,
    hipStream_t stream = nullptr);

/// Computes Grouped-Query Softmax Attention reading position from device memory
void LaunchAttention(const float* q, const float* k, const float* v,
                     const float* gate, float* k_cache, float* v_cache,
                     void* k_cache_f16, void* v_cache_f16, float* out_context,
                     std::uint32_t layer_idx, const std::uint32_t* d_pos,
                     std::uint32_t max_context, std::uint32_t num_heads,
                     std::uint32_t num_kv_heads, std::uint32_t head_dim,
                     hipStream_t stream = nullptr, bool skip_kv_write = false);

/// Fuses per-head Q/K RMSNorm, RoPE, and the KV-cache write for a single decode
/// token into one launch (opt-c010-qk-rope-kv). Writes the normed+roped Q into
/// q_out and the normed+roped K into k_out, and K/V into every non-null cache
/// representation supplied by the caller.
void LaunchFusedQKNormRoPEKvWrite(
    const float* q, const float* k, const float* v, const float* q_weight,
    const float* k_weight, float* q_out, float* k_out, float* k_cache,
    float* v_cache, void* k_cache_f16, void* v_cache_f16,
    std::uint32_t layer_idx, const std::uint32_t* d_pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float rope_theta, float eps = 1e-6F,
    hipStream_t stream = nullptr,
    const models::qwen::vision::DeviceRope* rope = nullptr,
    KvarnWriteParams kvarn = {});

/// Batched Causal Attention for B tokens with KV Cache. When skip_kv_write is
/// true the KV cache is assumed already written by the fused prefill kernel
/// (opt-c010-qk-rope-kv). When out_context_bf16 is non-null the context is also
/// written as BF16, which saves the caller a conversion launch when the next
/// projection consumes BF16.
void LaunchBatchedAttention(const float* q, const float* k, const float* v,
                            const float* gate, float* k_cache, float* v_cache,
                            void* k_cache_f16, void* v_cache_f16,
                            float* out_context, std::uint32_t layer_idx,
                            std::uint32_t start_pos, std::size_t batch_size,
                            std::uint32_t max_context, std::uint32_t num_heads,
                            std::uint32_t num_kv_heads, std::uint32_t head_dim,
                            hipStream_t stream = nullptr,
                            bool skip_kv_write = false,
                            void* out_context_bf16 = nullptr,
                            KvarnReadParams kvarn = {});

/// Batched fuse of per-head Q/K RMSNorm, RoPE, and the KV-cache write across B
/// tokens into one launch (opt-c010-qk-rope-kv). Writes the normed+roped Q into
/// q_out and the normed+roped K into k_out, and K/V into every non-null cache
/// representation supplied by the caller.
void LaunchBatchedFusedQKNormRoPEKvWrite(
    const float* q, const float* k, const float* v, const float* q_weight,
    const float* k_weight, float* q_out, float* k_out, float* k_cache,
    float* v_cache, void* k_cache_f16, void* v_cache_f16,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    std::uint32_t rotary_dim, float rope_theta, float eps = 1e-6F,
    hipStream_t stream = nullptr,
    const models::qwen::vision::DeviceRope* rope = nullptr,
    KvarnWriteParams kvarn = {});

/// Qwen3.8-specific causal GQA tile for gfx1151. The kernel processes 16 query
/// positions and two query heads per block while reusing one FP16 K/V tile.
/// Returns false for unsupported model shapes.
/// `lse_out`, when non-null, makes this one half of a split attention: the
/// kernel starts at `key_begin`, writes the partial log-sum-exp per
/// [head, token], and leaves the SiLU gate to the merge step. `skip_kv_write`
/// suppresses the KV-cache pack/sync when another half already did it.
[[nodiscard]] bool LaunchBatchedAttentionTile(
    const float* q, const float* k, const float* v, const float* gate,
    float* k_cache, float* v_cache, void* k_cache_f16, void* v_cache_f16,
    float* out_context, std::uint32_t layer_idx, std::uint32_t start_pos,
    std::size_t batch_size, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream = nullptr, float* lse_out = nullptr,
    std::uint32_t key_begin = 0, bool skip_kv_write = false,
    KvarnReadParams kvarn = {});

/// Masked prefill attention on the WMMA matrix cores.
/// Same contract as LaunchBatchedAttentionTile: it
/// packs the FP16 K/V cache unless `skip_kv_write`, masks keys in
/// [`key_begin`, min(context_end, query position]), and emits the partial
/// log-sum-exp into `lse_out` (suppressing the gate) when that is non-null.
/// Optional workspace spans hold temporary KV layouts and must not alias live
/// inputs, outputs or each other. Heads reuse the workspace in bounded groups
/// when the full layout does not fit. Persistent KV storage is unchanged.
/// Returns false when the shape is unsupported, so the caller can fall back.
[[nodiscard]] bool LaunchQwenWmmaAttention(
    const float* q, const float* k, const float* v, const float* gate,
    float* k_cache, float* v_cache, void* k_cache_f16, void* v_cache_f16,
    float* out_context, std::uint32_t layer_idx, std::uint32_t start_pos,
    std::size_t batch_size, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream = nullptr, float* lse_out = nullptr,
    std::uint32_t key_begin = 0, bool skip_kv_write = false,
    std::span<float> packed_k_workspace = {},
    std::span<float> packed_v_workspace = {},
    KvarnReadParams kvarn = {});

/// Large-batch causal attention using float32 QK/PV GEMMs and one reusable
/// [batch, context] score buffer. Test- and benchmark-only: it reads the FP32
/// planes directly and does not implement the KVarN read contract, so a KVarN
/// selection is an explicit error rather than a silent FP16/FP32 fallback
/// (design D7).
void LaunchBatchedAttentionGemm(
    hipblasHandle_t handle, const float* q, const float* k, const float* v,
    const float* gate, float* k_cache, float* v_cache, void* k_cache_f16,
    void* v_cache_f16, float* scores, float* out_context,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim,
    hipStream_t stream = nullptr, KvarnReadParams kvarn = {});

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_ATTENTION_HPP_
