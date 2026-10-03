#ifndef GUFO_MODELS_QWEN_KVARN_REFERENCE_HPP_
#define GUFO_MODELS_QWEN_KVARN_REFERENCE_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace gufo::models::qwen {

// KVarN is Qwen-only: the reference oracle, the kernels and the tests live in
// the model tree and share nothing with other models (design.md Context).

/// Sealed-tile geometry (design.md D1/D5): one KVarN tile holds a single
/// attention head's 128 tokens x 256 channels for K or V.
inline constexpr std::size_t kKvarnTileTokens = 128;
inline constexpr std::size_t kKvarnTileChannels = 256;
inline constexpr std::size_t kKvarnValueGroupChannels = 128;
inline constexpr std::size_t kKvarnValueGroups =
    kKvarnTileChannels / kKvarnValueGroupChannels;
inline constexpr std::size_t kKvarnTileElements =
    kKvarnTileTokens * kKvarnTileChannels;
static_assert(kKvarnTileElements == 32768);
static_assert(kKvarnTileChannels % kKvarnValueGroupChannels == 0);

/// RTN parameter pairs per tile (design.md D5): K keeps one pair per channel
/// (shared across the 128 tokens), V one pair per (token, 128-channel group).
/// Both come to 256 pairs, so the reader addresses K and V uniformly.
inline constexpr std::size_t kKvarnRtnParams = kKvarnTileChannels;
static_assert(kKvarnTileTokens * kKvarnValueGroups == kKvarnRtnParams);

/// Head dimension the rotation and the tile channel axis are defined for
/// (design.md D3). 256 is a power of two, so the normalized Hadamard scale is
/// the exactly representable 1/16.
inline constexpr std::size_t kKvarnHeadDim = 256;
static_assert(kKvarnHeadDim == kKvarnTileChannels);
static_assert((kKvarnHeadDim & (kKvarnHeadDim - 1)) == 0);

/// Numeric contract (design.md D6). Deterministic by construction: the pass
/// count is a compile-time constant and every axis pass reduces indices in
/// ascending order, so a tile's sealed bytes are a function of its 128 KV rows
/// alone and do not depend on hardware parallelism or workgroup shape.
inline constexpr std::size_t kKvarnNormPasses = 16;
/// Floor applied to an axis RMS and to a zero-span RTN range. A constant
/// channel keeps an exact reconstruction because its code is zero either way.
inline constexpr double kKvarnNormEpsilon = 1e-5;
/// Defined clamp for extreme outliers on the raw (rotated) tile values.
inline constexpr double kKvarnValueClamp = 4096.0;
/// Clamp on each axis' accumulated log-scale. exp(±40) and their product stay
/// inside FP32 range, so normalization can never emit a non-finite scale.
inline constexpr double kKvarnLogScaleClamp = 40.0;
inline constexpr std::uint8_t kKvarnMinBits = 4;
inline constexpr std::uint8_t kKvarnMaxBits = 6;

inline constexpr std::size_t kKvarnLowPlaneBytes = kKvarnTileElements / 2;
inline constexpr std::size_t kKvarnBitPlaneBytes = kKvarnTileElements / 8;
/// 256 scale + 256 zero-point + 256 channel + 128 token, all FP16.
inline constexpr std::size_t kKvarnPayloadMetadataBytes =
    2 * (kKvarnRtnParams + kKvarnRtnParams + kKvarnTileChannels) +
    2 * kKvarnTileTokens;

#pragma pack(push, 1)
/// Bit planes of one tile's codes. Code (i, j) lives at element index
/// `i * kKvarnTileChannels + j`; `low` carries bits 0..3 with two codes per
/// byte (even element index in the low nibble), `mid` bit 4 and `high` bit 5
/// with one bit per element (bit b of byte n is element 8n + b). 5- and 6-bit
/// tiles reuse the 4-bit plane, which keeps one load width and alignment for
/// every preset (design.md D5).
struct KvarnTileLowPlane {
  std::uint8_t bytes[kKvarnLowPlaneBytes];
};

struct KvarnTileBitPlane {
  std::uint8_t bytes[kKvarnBitPlaneBytes];
};

/// Sealed payload for one tile and one of K or V. Field order is the on-device
/// contract; the assertions below pin size and offsets so the GPU kernel, the
/// snapshot size formula and the CPU oracle cannot drift apart.
struct KvarnTilePayload {
  std::uint16_t rtn_scale[kKvarnRtnParams];
  std::uint16_t rtn_zero[kKvarnRtnParams];
  std::uint16_t norm_channel_scale[kKvarnTileChannels];
  std::uint16_t norm_token_scale[kKvarnTileTokens];
  KvarnTileLowPlane low;
  KvarnTileBitPlane mid;
  KvarnTileBitPlane high;
};
#pragma pack(pop)

static_assert(sizeof(KvarnTileLowPlane) == kKvarnLowPlaneBytes);
static_assert(sizeof(KvarnTileBitPlane) == kKvarnBitPlaneBytes);
static_assert(offsetof(KvarnTilePayload, rtn_scale) == 0);
static_assert(offsetof(KvarnTilePayload, rtn_zero) == 2 * kKvarnRtnParams);
static_assert(offsetof(KvarnTilePayload, norm_channel_scale) ==
              4 * kKvarnRtnParams);
static_assert(offsetof(KvarnTilePayload, norm_token_scale) ==
              4 * kKvarnRtnParams + 2 * kKvarnTileChannels);
static_assert(offsetof(KvarnTilePayload, low) == kKvarnPayloadMetadataBytes);
static_assert(offsetof(KvarnTilePayload, mid) ==
              kKvarnPayloadMetadataBytes + kKvarnLowPlaneBytes);
static_assert(offsetof(KvarnTilePayload, high) == kKvarnPayloadMetadataBytes +
                                                      kKvarnLowPlaneBytes +
                                                      kKvarnBitPlaneBytes);
static_assert(sizeof(KvarnTilePayload) == kKvarnPayloadMetadataBytes +
                                              kKvarnLowPlaneBytes +
                                              2 * kKvarnBitPlaneBytes);

/// Axis a RTN parameter pair is shared over (design.md D5).
enum class KvarnRtnAxis : std::uint8_t {
  kChannel = 0,     ///< K: one pair per channel, ranged over the 128 tokens.
  kTokenGroup = 1,  ///< V: one pair per (token, 128-channel group).
};

/// Records how the defined floors and clamps were applied. The GPU seal kernel
/// keeps the same counters per session for the T-01 oracle comparison.
struct KvarnQuantDiagnostics {
  std::size_t non_finite_inputs = 0;
  std::size_t value_clamps = 0;
  std::size_t scale_floors = 0;
  std::size_t log_scale_clamps = 0;
};

/// Sealed values for one tile, in the order the reader consumes them.
struct KvarnTileQuantResult {
  std::array<float, kKvarnTileChannels> norm_channel_scale{};
  std::array<float, kKvarnTileTokens> norm_token_scale{};
  std::array<float, kKvarnRtnParams> rtn_scale{};
  std::array<float, kKvarnRtnParams> rtn_zero{};
  std::array<std::uint8_t, kKvarnTileElements> codes{};
  std::uint8_t bits = kKvarnMinBits;
};

enum class KvarnQuantStatus : std::uint8_t {
  kOk = 0,
  kNonFiniteInput = 1,
};

/// True when `values` holds a non-finite element. Callers use this at an
/// existing synchronisation point; it never syncs on its own.
[[nodiscard]] bool KvarnHasNonFinite(std::span<const float> values) noexcept;

/// In-place normalized Hadamard rotation of one attention head (design.md D3):
/// eight butterfly stages over `kKvarnHeadDim` elements, then the 1/16 scale.
/// The transform is symmetric, so the inverse rotation is this same call; K, Q
/// and V all rotate through it. `values` must be `kKvarnHeadDim` long.
void KvarnFwhtHeadInPlace(std::span<float> values) noexcept;

/// Dual-axis normalization of one `kKvarnTileTokens` x `kKvarnTileChannels`
/// tile (row-major), alternating `kKvarnNormPasses` axis passes starting with
/// the channel pass. Every pass scales by the root-mean-square of its axis,
/// floored at `kKvarnNormEpsilon`, and accumulates the log of that scale so the
/// reader restores the tile as `normalized[i][j] * channel_scale[j] *
/// token_scale[i]`. Raw values are clamped to `kKvarnValueClamp` first.
///
/// Writes the normalized tile and the two scale vectors; fills `diag` when
/// non-null. Returns `kNonFiniteInput` without writing results when the input
/// holds a non-finite value. `tile` and `normalized` must both be
/// `kKvarnTileElements` long.
[[nodiscard]] KvarnQuantStatus KvarnNormalizeTile(
    std::span<const float> tile, std::span<float> normalized,
    KvarnTileQuantResult* result, KvarnQuantDiagnostics* diag) noexcept;

/// Asymmetric RTN over an already-normalized tile. Each parameter pair ranges
/// its own reduction set (see `KvarnRtnAxis`), so a constant set floors its
/// scale instead of dividing by zero. Writes codes, scales and zero-points.
void KvarnQuantizeAsymmetricRtn(std::span<const float> normalized,
                                KvarnRtnAxis axis, std::uint8_t bits,
                                KvarnTileQuantResult* result,
                                KvarnQuantDiagnostics* diag) noexcept;

/// Encodes `values` with a caller-supplied scale/zero-point. The arithmetic is
/// literally the device form (float subtract, divide, round-half-away-from-
/// zero, clamp) so T-01 stage 1 can compare GPU-produced codes exactly.
void KvarnEncodeAsymmetricRtn(std::span<const float> values, std::uint8_t bits,
                              float rtn_scale, float rtn_zero,
                              std::span<std::uint8_t> codes) noexcept;

/// Reads one code from a bit plane: element `index` is `i * kKvarnTileChannels
/// + j`.
[[nodiscard]] std::uint8_t KvarnUnpackTileCode(const KvarnTilePayload& payload,
                                               std::uint8_t bits,
                                               std::size_t index) noexcept;

/// Reconstructs the normalized-domain value of one packed element:
/// `code * rtn_scale[p] + rtn_zero[p]`, where `p` is the channel for
/// `KvarnRtnAxis::kChannel` and the (token, 128-channel group) for
/// `KvarnRtnAxis::kTokenGroup`.
[[nodiscard]] float KvarnDecodeTileElement(const KvarnTilePayload& payload,
                                           KvarnRtnAxis axis, std::uint8_t bits,
                                           std::size_t token,
                                           std::size_t channel) noexcept;

/// Packs a quantized tile into its sealed byte layout.
void KvarnPackTilePayload(const KvarnTileQuantResult& result,
                          KvarnTilePayload* payload) noexcept;

/// Round-to-nearest-even FP16 conversion for the payload metadata, which is
/// stored as FP16 unless phase 0 promotes it to BF16 (design.md D5).
[[nodiscard]] std::uint16_t KvarnFloatToFp16(float value) noexcept;

/// Rows every session keeps in FP16 regardless of the format: the sink.
inline constexpr std::size_t kKvarnSinkRows = kKvarnTileTokens;
/// Rows of the FP16 tail ring beyond the `tail_length` rows of the view rule:
/// one tile of margin (design.md D1).
inline constexpr std::size_t kKvarnRingMarginRows = kKvarnTileTokens;

/// F16 element offset of one (layer, KV head, position) row inside one plane of
/// the KVarN sink or tail ring (design D1/D2). Positions below the sink go to
/// the sink; later positions wrap through the ring. `plane_elements` is
/// `sink_elements` or `tail_elements` from `KvarnRegions`; K is the first
/// plane, V the second.
[[nodiscard]] constexpr std::size_t KvarnSinkRowOffset(
    std::size_t layer, std::size_t kv_heads, std::size_t head,
    std::size_t position) noexcept {
  return ((layer * kv_heads + head) * kKvarnSinkRows + position) *
         kKvarnHeadDim;
}

[[nodiscard]] constexpr std::size_t KvarnTailRowOffset(
    std::size_t layer, std::size_t kv_heads, std::size_t head,
    std::size_t position, std::size_t ring_rows) noexcept {
  const std::size_t ring = (position - kKvarnSinkRows) % ring_rows;
  return ((layer * kv_heads + head) * ring_rows + ring) * kKvarnHeadDim;
}

/// Tile index a position belongs to: t = (position - 128) / 128 (design D2).
[[nodiscard]] constexpr std::size_t KvarnTileIndex(
    std::size_t position) noexcept {
  return (position - kKvarnSinkRows) / kKvarnTileTokens;
}

/// Byte offset of one sealed tile's K payload inside the KVarN body; the V
/// payload of the same tile follows immediately (design D5/D7).
[[nodiscard]] constexpr std::size_t KvarnBodyTileOffset(
    std::size_t layer, std::size_t kv_heads, std::size_t head, std::size_t tile,
    std::size_t tile_count, std::size_t k_tile_bytes,
    std::size_t v_tile_bytes) noexcept {
  return ((layer * kv_heads + head) * tile_count + tile) *
         (k_tile_bytes + v_tile_bytes);
}

/// Bytes of one sealed tile's payload for a given code width (design.md D5):
/// the metadata plus the low plane and, for 5- and 6-bit codes, the extra bit
/// planes. This is the exact size `EstimateKvarnMemory` counts.
[[nodiscard]] constexpr std::size_t KvarnTilePayloadBytes(
    std::uint8_t bits) noexcept {
  return kKvarnPayloadMetadataBytes + kKvarnLowPlaneBytes +
         (bits > 4 ? kKvarnBitPlaneBytes : 0) +
         (bits > 5 ? kKvarnBitPlaneBytes : 0);
}

/// Byte sizes of the three KVarN regions of one session, over every
/// full-attention layer and KV head (design.md D1/D5). The sink and the tail
/// ring hold F16 K and V; the body holds one sealed K and V payload per tile.
struct KvarnRegions {
  std::size_t tile_count{0};     ///< Tiles per (layer, KV head).
  std::size_t tail_rows{0};      ///< Ring capacity C = L + 128 + W.
  std::size_t sink_elements{0};  ///< Per K or V plane.
  std::size_t tail_elements{0};  ///< Per K or V plane.
  std::size_t sink_bytes{0};     ///< Both planes.
  std::size_t body_bytes{0};     ///< Both planes.
  std::size_t tail_bytes{0};     ///< Both planes.
  std::size_t total_bytes{0};
};

/// Region sizes from the same formula as `EstimateKvarnMemory`. Pass the
/// structural minimum W as `tail_length` when `--kv-cache-tail` was not given.
[[nodiscard]] KvarnRegions ComputeKvarnRegions(
    std::uint8_t k_bits, std::uint8_t v_bits, std::size_t tail_length,
    std::size_t verify_rows, std::size_t capacity_tokens,
    std::size_t attention_layers, std::size_t kv_heads) noexcept;

/// KV memory of one session by the size formula of design.md D5, over every
/// full-attention layer and KV head. Bits per value are for K and V together.
struct KvarnMemoryEstimate {
  double body_bpv{0.0};   ///< Sealed tiles or quantized rows only.
  double total_bpv{0.0};  ///< Sink, tail ring, scales and zero points included.
  double session_gib{0.0};
};

/// KVarN: sealed tiles of `k_bits` / `v_bits` plus the FP16 sink and a tail
/// ring of `tail_length + kKvarnRingMarginRows + verify_rows` rows. Pass the
/// structural minimum `verify_rows` (W) as `tail_length` when no tail was set.
[[nodiscard]] KvarnMemoryEstimate EstimateKvarnMemory(
    std::uint8_t k_bits, std::uint8_t v_bits, std::size_t tail_length,
    std::size_t verify_rows, std::size_t capacity_tokens,
    std::size_t attention_layers, std::size_t kv_heads) noexcept;

/// E_q8 / E_q4 baseline: 32-value blocks with an FP16 scale, no tail, FP16
/// sink rows.
[[nodiscard]] KvarnMemoryEstimate EstimateBlockQuantMemory(
    std::uint8_t bits, std::size_t capacity_tokens,
    std::size_t attention_layers, std::size_t kv_heads) noexcept;

}  // namespace gufo::models::qwen

#endif  // GUFO_MODELS_QWEN_KVARN_REFERENCE_HPP_
