#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_ATTENTION_POLICY_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_ATTENTION_POLICY_HPP_

#include <cstddef>
#include <cstdint>
#include <utility>

#include "src/models/qwen/hip/execution_policy.hpp"

namespace gufo::hip::detail {

inline constexpr std::size_t kOptimizedAttentionMinBatch{1024};
inline constexpr std::uint32_t kTiledAttentionQueryHeads{24};
inline constexpr std::uint32_t kTiledAttentionKvHeads{4};
inline constexpr std::uint32_t kTiledAttentionHeadDim{256};
inline constexpr std::size_t kSplitKDecodeAttentionMinContext{128};
inline constexpr std::uint32_t kSplitKDecodeAttentionMaxSplits{32};
inline constexpr std::uint32_t kFusedQkNormMaxHeadDim{256};

struct AttentionSupportParams {
  std::size_t batch_size{0};
  std::uint32_t start_pos{0};
  std::uint32_t max_context{0};
  std::uint32_t num_heads{0};
  std::uint32_t num_kv_heads{0};
  std::uint32_t head_dim{0};
  bool has_k_cache_f16{false};
  bool has_v_cache_f16{false};
  // KVarN selection replaces the canonical FP16 planes with the sink, the
  // sealed-tile body and the F16 tail ring (design D1/D7). The tile and WMMA
  // kernels must then read those regions instead of `*_cache_f16`.
  bool has_kvarn{false};
};

[[nodiscard]] constexpr bool ShouldAttemptOptimizedAttention(
    std::size_t visible_context) noexcept {
  return visible_context >= kOptimizedAttentionMinBatch;
}

// opt-c010-qk-rope-kv: fuse per-head Q/K RMSNorm, RoPE, and the KV-cache write
// into a single kernel per token (decode) / per token row (prefill). Flip to
// false to revert to the unfused chain (PerHeadRMSNorm x2 + RoPE +
// WriteKVCache*), which stays wired as the independent reference.
[[nodiscard]] constexpr bool ShouldFuseQKNormRoPEKvWrite(
    const QwenExecutionPolicy& policy) noexcept {
  return policy.fuse_qk_norm_rope_kv;
}

[[nodiscard]] constexpr bool ShouldFuseQKNormRoPEKvWrite() noexcept {
  return ShouldFuseQKNormRoPEKvWrite(QwenExecutionPolicy::Production());
}

[[nodiscard]] constexpr std::uint32_t SelectDecodeAttentionSplitCount(
    std::size_t sequence_length) noexcept {
  if (sequence_length < kSplitKDecodeAttentionMinContext) {
    return 1;
  }
  return kSplitKDecodeAttentionMaxSplits;
}

[[nodiscard]] constexpr bool IsFusedQkNormSupported(
    std::uint32_t head_dim) noexcept {
  return head_dim != 0 && head_dim <= kFusedQkNormMaxHeadDim;
}

[[nodiscard]] constexpr bool IsSplitKDecodeAttentionSupported(
    std::size_t sequence_length, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) noexcept {
  return SelectDecodeAttentionSplitCount(sequence_length) > 1 &&
         num_heads != 0 && num_kv_heads != 0 &&
         (num_heads % num_kv_heads) == 0 && head_dim == 256;
}

[[nodiscard]] constexpr std::size_t DecodeAttentionScratchElements(
    std::uint32_t num_heads, std::uint32_t head_dim) noexcept {
  return static_cast<std::size_t>(num_heads) * kSplitKDecodeAttentionMaxSplits *
         (static_cast<std::size_t>(head_dim) + 2);
}

/// FP16 decode and verification tile width: the block kernel stages K and V in
/// 128-token tiles (design D1/D11).
inline constexpr std::size_t kDecodeAttentionBlockTokens{128};

/// One partition of the FP16 decode/verification attention blocks: whole
/// 128-token tiles spread over `kSplitKDecodeAttentionMaxSplits` partitions,
/// the last one absorbing the remainder. This is the whole-tile split the
/// KVarN read already uses and the method design D11 selected for the FP16
/// path; it replaces the `ceil(n / 32)` split of
/// `SelectDecodeAttentionSplitCount` for the FP16 block kernel only. Below the
/// split-K threshold, split 0 covers the whole sequence and the remaining
/// splits are empty.
struct DecodeAttentionBlockRange {
  std::size_t begin{0};
  std::size_t end{0};
  /// A partition without keys contributes nothing to the merge.
  [[nodiscard]] constexpr bool empty() const noexcept { return begin >= end; }
};

[[nodiscard]] constexpr DecodeAttentionBlockRange
SelectDecodeAttentionBlockRange(std::size_t sequence_length,
                                std::uint32_t split) noexcept {
  if (SelectDecodeAttentionSplitCount(sequence_length) == 1) {
    return split == 0
               ? DecodeAttentionBlockRange{0, sequence_length}
               : DecodeAttentionBlockRange{sequence_length, sequence_length};
  }
  const std::size_t blocks =
      (sequence_length + kDecodeAttentionBlockTokens - 1) /
      kDecodeAttentionBlockTokens;
  const std::size_t begin_block = (static_cast<std::uint64_t>(split) * blocks) /
                                  kSplitKDecodeAttentionMaxSplits;
  const std::size_t end_block =
      (static_cast<std::uint64_t>(split + 1) * blocks) /
      kSplitKDecodeAttentionMaxSplits;
  const std::size_t begin = begin_block * kDecodeAttentionBlockTokens;
  const std::size_t end = end_block * kDecodeAttentionBlockTokens;
  return {begin, end < sequence_length ? end : sequence_length};
}

[[nodiscard]] constexpr bool IsTiledAttentionSupported(
    const AttentionSupportParams& params) noexcept {
  return params.batch_size != 0 &&
         params.num_heads == kTiledAttentionQueryHeads &&
         params.num_kv_heads == kTiledAttentionKvHeads &&
         params.head_dim == kTiledAttentionHeadDim &&
         static_cast<std::size_t>(params.start_pos) + params.batch_size <=
             params.max_context &&
         (params.has_kvarn ||
          (params.has_k_cache_f16 && params.has_v_cache_f16));
}

/// The row-split recurrence needs two scratch planes and a 128 x 128 state
/// tile.
[[nodiscard]] constexpr bool ShouldUseSsmRowSplitRecurrence(
    bool shape_supported, bool scratch_ready) noexcept {
  return shape_supported && scratch_ready;
}

/// Executes the existing prefill attention fallback chain without virtual
/// dispatch: tiled, then the baseline implementation.
template<typename TiledLauncher, typename BaselineLauncher>
inline void DispatchPrefillAttention(std::size_t visible_context,
                                     TiledLauncher&& launch_tiled,
                                     BaselineLauncher&& launch_baseline) {
  if (ShouldAttemptOptimizedAttention(visible_context)) {
    if (std::forward<TiledLauncher>(launch_tiled)()) {
      return;
    }
  }
  std::forward<BaselineLauncher>(launch_baseline)();
}

}  // namespace gufo::hip::detail

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_ATTENTION_POLICY_HPP_
