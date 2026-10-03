#ifndef GUFO_MODELS_QWEN_HIP_EXECUTION_POLICY_HPP_
#define GUFO_MODELS_QWEN_HIP_EXECUTION_POLICY_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace gufo::hip {

enum class QwenKvCacheStorage : std::uint8_t {
  kFp32,
  kFp16,
  kKvarn,
};

/// Range of the `--kv-cache-tail` option: full-precision rows kept behind each
/// KVarN query (design.md D1, D10). Without the option no tail is added.
inline constexpr std::uint32_t kQwenKvCacheTailMin = 256;
inline constexpr std::uint32_t kQwenKvCacheTailMax = 2048;

/// Numeric type of the KVarN scale and zero-point metadata (design.md D5,
/// resolved D-04). FP16 is the measured default; BF16 exists for a preset that
/// needs its exponent range.
enum class QwenKvarnScale : std::uint8_t {
  kFp16,
  kBf16,
};

/// KVarN preset resolved from `--kv-cache` (design.md D1, D11). `tail_length`
/// is 0 when `--kv-cache-tail` was not given, which selects the structural
/// minimum W; an explicit value is at least `kQwenKvCacheTailMin`, so the two
/// never collide and both feed the fingerprint.
struct QwenKvarnPreset {
  std::uint8_t k_bits{0};  ///< 0 when no KVarN preset is selected.
  std::uint8_t v_bits{0};
  std::uint32_t tail_length{0};
  QwenKvarnScale scale{QwenKvarnScale::kFp16};

  [[nodiscard]] constexpr bool enabled() const noexcept { return k_bits != 0; }
};

/// KVarN region geometry (design.md D1, D5, D8). W is the larger of the
/// maximum verification rows and the maximum rollback width: the rows a
/// session can still revisit in F16 behind a sealed tile.
inline constexpr std::uint32_t kQwenKvarnTileTokens = 128;
inline constexpr std::uint32_t kQwenKvarnHeadDim = 256;
inline constexpr std::uint32_t kQwenKvarnSinkRows = 128;
inline constexpr std::uint32_t kQwenKvarnVerificationRows = 8;
inline constexpr std::uint32_t kQwenKvarnRollbackRows = 9;
inline constexpr std::uint32_t kQwenKvarnW =
    kQwenKvarnVerificationRows > kQwenKvarnRollbackRows
        ? kQwenKvarnVerificationRows
        : kQwenKvarnRollbackRows;
static_assert(kQwenKvCacheTailMin >= kQwenKvarnW,
              "the smallest --kv-cache-tail must cover W");

/// F16 rows behind each KVarN query: the explicit `--kv-cache-tail` value, or
/// the structural minimum W when it was not given.
[[nodiscard]] constexpr std::uint32_t QwenKvarnTailRows(
    const QwenKvarnPreset& preset) noexcept {
  return preset.tail_length == 0 ? kQwenKvarnW : preset.tail_length;
}

/// Capacity of the F16 tail ring in rows (design.md D1): L + 128 + W.
[[nodiscard]] constexpr std::uint32_t QwenKvarnTailRingRows(
    const QwenKvarnPreset& preset) noexcept {
  return QwenKvarnTailRows(preset) + kQwenKvarnSinkRows + kQwenKvarnW;
}

enum class QwenRecurrentStateStorage : std::uint8_t {
  kFp32,
  kBf16,
};

[[nodiscard]] constexpr std::size_t QwenRecurrentStateElementBytes(
    QwenRecurrentStateStorage storage) noexcept {
  return storage == QwenRecurrentStateStorage::kBf16 ? sizeof(std::uint16_t)
                                                     : sizeof(float);
}

/// Immutable route policy for one Qwen GPU executor. Resolve this before HIP
/// graph capture. Alternate storage is used by independent operator controls.
enum class QwenExecutionMode : std::uint8_t {
  kDecode,
  kPrefill,
};

struct QwenExecutionPolicy {
  bool fuse_qk_norm_rope_kv{true};
  QwenKvCacheStorage kv_cache_storage{QwenKvCacheStorage::kFp16};
  /// Meaningful only when `kv_cache_storage == kKvarn`.
  QwenKvarnPreset kvarn;
  QwenRecurrentStateStorage recurrent_state_storage{
      QwenRecurrentStateStorage::kFp32};

  [[nodiscard]] static constexpr QwenExecutionPolicy Production() noexcept {
    return {};
  }

  [[nodiscard]] constexpr bool UsesFp16AttentionKv() const noexcept {
    return kv_cache_storage == QwenKvCacheStorage::kFp16;
  }

  [[nodiscard]] constexpr bool UsesKvarn() const noexcept {
    return kv_cache_storage == QwenKvCacheStorage::kKvarn;
  }

  [[nodiscard]] constexpr bool UsesBf16RecurrentState() const noexcept {
    return recurrent_state_storage == QwenRecurrentStateStorage::kBf16;
  }

  /// Stable bit fingerprint suitable for telemetry and graph-cache identity.
  /// The FP32/FP16 values are unchanged; KVarN uses the bits above 40 so a
  /// preset (and its tail length) can never share a graph with another.
  [[nodiscard]] constexpr std::uint64_t Fingerprint() const noexcept {
    std::uint64_t bits =
        (static_cast<std::uint64_t>(fuse_qk_norm_rope_kv) << 0U) |
        (static_cast<std::uint64_t>(UsesFp16AttentionKv()) << 8U) |
        (static_cast<std::uint64_t>(UsesBf16RecurrentState()) << 9U);
    if (UsesKvarn()) {
      bits |= (std::uint64_t{1} << 40U) |
              (static_cast<std::uint64_t>(kvarn.k_bits) << 41U) |
              (static_cast<std::uint64_t>(kvarn.v_bits) << 44U) |
              (static_cast<std::uint64_t>(kvarn.tail_length) << 47U) |
              (static_cast<std::uint64_t>(kvarn.scale == QwenKvarnScale::kBf16)
               << 59U);
    }
    return bits;
  }
};

/// The production KVarN presets of design D1: memory-first `kvarn-k4v4`,
/// intermediate `kvarn-k5v4`, and quality-first `kvarn-k6v5` (resolved D-02).
[[nodiscard]] inline bool ParseQwenKvarnPreset(
    std::string_view name, QwenKvarnPreset* preset) noexcept {
  QwenKvarnPreset parsed;
  if (name == "kvarn-k4v4") {
    parsed.k_bits = 4;
    parsed.v_bits = 4;
  } else if (name == "kvarn-k5v4") {
    parsed.k_bits = 5;
    parsed.v_bits = 4;
  } else if (name == "kvarn-k6v5") {
    parsed.k_bits = 6;
    parsed.v_bits = 5;
  } else {
    return false;
  }
  *preset = parsed;
  return true;
}

/// KVarN supports only the Qwen3.8 27B full-attention shape (design D7): 24
/// query heads, 4 KV heads, head_dim 256. Any other shape is a startup error;
/// KVarN never falls back to FP16 silently.
[[nodiscard]] constexpr bool QwenKvarnSupportsShape(
    std::uint32_t attention_heads, std::uint32_t kv_heads,
    std::uint32_t head_dim) noexcept {
  return attention_heads == 24 && kv_heads == 4 && head_dim == 256;
}

/// Error text when a KVarN policy meets a shape KVarN does not support, or an
/// empty string when KVarN is not selected or the shape matches. Callers raise
/// this before allocating the arena (design D7).
[[nodiscard]] inline std::string QwenKvarnShapeError(
    const QwenExecutionPolicy& policy, std::uint32_t attention_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim) {
  if (!policy.UsesKvarn() ||
      QwenKvarnSupportsShape(attention_heads, kv_heads, head_dim)) {
    return {};
  }
  return "KVarN KV storage requires 24 query heads, 4 KV heads and head_dim "
         "256 (Qwen3.8 27B full attention)";
}

/// Whether `--kv-cache-tail` may accompany this `--kv-cache` value. Only KVarN
/// keeps a tail; `fp16` does not.
[[nodiscard]] inline bool QwenKvCacheAcceptsTail(std::string_view name) {
  QwenKvarnPreset preset;
  return ParseQwenKvarnPreset(name, &preset);
}

/// Maps a validated `--kv-cache` value and `--kv-cache-tail` (0 = not given)
/// onto the execution policy. This is the single place the CLI names become
/// policy state; KVarN presets register here with their storage. Returns false
/// for a value this build does not support or a tail that does not apply.
[[nodiscard]] inline bool ResolveQwenKvCache(std::string_view name,
                                             std::uint32_t tail_tokens,
                                             QwenExecutionPolicy* policy) {
  if (tail_tokens != 0 &&
      (tail_tokens < kQwenKvCacheTailMin || tail_tokens > kQwenKvCacheTailMax ||
       !QwenKvCacheAcceptsTail(name))) {
    return false;
  }
  if (name == "fp16") {
    policy->kv_cache_storage = QwenKvCacheStorage::kFp16;
    return true;
  }
  if (QwenKvarnPreset preset; ParseQwenKvarnPreset(name, &preset)) {
    preset.tail_length = tail_tokens;
    policy->kvarn = preset;
    policy->kv_cache_storage = QwenKvCacheStorage::kKvarn;
    return true;
  }
  return false;
}

/// Resolved route decisions for one layer invocation. This record is pure data
/// so policy selection can be tested independently from HIP launches.
struct QwenLayerRoutePlan {
  QwenExecutionMode mode{QwenExecutionMode::kDecode};
  bool full_attention{false};
  bool fuse_qk_norm_rope_kv{false};

  [[nodiscard]] constexpr std::uint64_t Fingerprint() const noexcept {
    return (static_cast<std::uint64_t>(mode == QwenExecutionMode::kPrefill)
            << 0U) |
           (static_cast<std::uint64_t>(full_attention) << 1U) |
           (static_cast<std::uint64_t>(fuse_qk_norm_rope_kv) << 2U);
  }
};

enum class QwenRouteRejection : std::uint32_t {
  kNone = 0,
  kQkNormRopeKvRequiresAttention = 1U << 0U,
};

[[nodiscard]] constexpr QwenRouteRejection operator|(
    QwenRouteRejection lhs, QwenRouteRejection rhs) noexcept {
  return static_cast<QwenRouteRejection>(static_cast<std::uint32_t>(lhs) |
                                         static_cast<std::uint32_t>(rhs));
}

constexpr QwenRouteRejection& operator|=(QwenRouteRejection& lhs,
                                         QwenRouteRejection rhs) noexcept {
  lhs = lhs | rhs;
  return lhs;
}

[[nodiscard]] constexpr bool HasQwenRouteRejection(
    QwenRouteRejection reasons, QwenRouteRejection reason) noexcept {
  return (static_cast<std::uint32_t>(reasons) &
          static_cast<std::uint32_t>(reason)) != 0U;
}

struct QwenLayerRouteResolution {
  QwenLayerRoutePlan plan;
  QwenRouteRejection rejected{QwenRouteRejection::kNone};
};

[[nodiscard]] constexpr QwenLayerRouteResolution
ResolveQwenLayerRouteWithReasons(const QwenExecutionPolicy& policy,
                                 QwenExecutionMode mode,
                                 bool full_attention) noexcept {
  const auto rejected = policy.fuse_qk_norm_rope_kv && !full_attention
                            ? QwenRouteRejection::kQkNormRopeKvRequiresAttention
                            : QwenRouteRejection::kNone;
  return {
      .plan =
          {
              .mode = mode,
              .full_attention = full_attention,
              .fuse_qk_norm_rope_kv =
                  full_attention && policy.fuse_qk_norm_rope_kv,
          },
      .rejected = rejected,
  };
}

[[nodiscard]] constexpr QwenLayerRoutePlan ResolveQwenLayerRoute(
    const QwenExecutionPolicy& policy, QwenExecutionMode mode,
    bool full_attention) noexcept {
  return ResolveQwenLayerRouteWithReasons(policy, mode, full_attention).plan;
}

/// FNV-1a-style composition helpers for a deterministic executor-local graph
/// workload identity. Callers add configuration fields and resolved route
/// fingerprints; raw addresses must never participate.
[[nodiscard]] constexpr std::uint64_t
BeginQwenGraphWorkloadIdentity() noexcept {
  return 14695981039346656037ULL;
}

[[nodiscard]] constexpr std::uint64_t ExtendQwenGraphWorkloadIdentity(
    std::uint64_t identity, std::uint64_t value) noexcept {
  return (identity ^ value) * 1099511628211ULL;
}

enum class QwenGraphRejection : std::uint32_t {
  kNone = 0,
  kLogitsNotRequested = 1U << 0U,
  kSplitKAttentionRequired = 1U << 1U,
  kGraphDisabled = 1U << 3U,
};

[[nodiscard]] constexpr QwenGraphRejection operator|(
    QwenGraphRejection lhs, QwenGraphRejection rhs) noexcept {
  return static_cast<QwenGraphRejection>(static_cast<std::uint32_t>(lhs) |
                                         static_cast<std::uint32_t>(rhs));
}

constexpr QwenGraphRejection& operator|=(QwenGraphRejection& lhs,
                                         QwenGraphRejection rhs) noexcept {
  lhs = lhs | rhs;
  return lhs;
}

[[nodiscard]] constexpr bool HasQwenGraphRejection(
    QwenGraphRejection reasons, QwenGraphRejection reason) noexcept {
  return (static_cast<std::uint32_t>(reasons) &
          static_cast<std::uint32_t>(reason)) != 0U;
}

[[nodiscard]] constexpr QwenGraphRejection ResolveQwenGraphRejections(
    bool compute_logits, bool split_k_attention, bool graph_enabled) noexcept {
  QwenGraphRejection rejected = QwenGraphRejection::kNone;
  if (!compute_logits) {
    rejected |= QwenGraphRejection::kLogitsNotRequested;
  }
  if (split_k_attention) {
    rejected |= QwenGraphRejection::kSplitKAttentionRequired;
  }
  if (!graph_enabled) {
    rejected |= QwenGraphRejection::kGraphDisabled;
  }
  return rejected;
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_EXECUTION_POLICY_HPP_
