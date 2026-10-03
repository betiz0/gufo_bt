#include "src/models/qwen/kvarn_reference.hpp"

#include <cmath>
#include <cstring>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::qwen {
namespace {

constexpr std::size_t kTokenStride = kKvarnTileChannels;

[[nodiscard]] constexpr std::uint32_t CodeMax(std::uint8_t bits) noexcept {
  return (1U << bits) - 1U;
}

/// Number of elements in one RTN parameter's reduction set.
[[nodiscard]] constexpr std::size_t RtnSetSize(KvarnRtnAxis axis) noexcept {
  return axis == KvarnRtnAxis::kChannel ? kKvarnTileTokens
                                        : kKvarnValueGroupChannels;
}

/// Element (token, channel) owning RTN parameter `p`.
[[nodiscard]] constexpr std::size_t RtnSetElement(KvarnRtnAxis axis,
                                                  std::size_t p,
                                                  std::size_t n) noexcept {
  if (axis == KvarnRtnAxis::kChannel) {
    return n * kTokenStride + p;
  }
  const std::size_t token = p / kKvarnValueGroups;
  const std::size_t group = p % kKvarnValueGroups;
  return token * kTokenStride + group * kKvarnValueGroupChannels + n;
}

/// RTN parameter owning element (token, channel).
[[nodiscard]] constexpr std::size_t RtnParam(KvarnRtnAxis axis,
                                             std::size_t token,
                                             std::size_t channel) noexcept {
  if (axis == KvarnRtnAxis::kChannel) {
    return channel;
  }
  return token * kKvarnValueGroups + channel / kKvarnValueGroupChannels;
}

[[nodiscard]] double ClampLogScale(double value,
                                   KvarnQuantDiagnostics* diag) noexcept {
  if (value > kKvarnLogScaleClamp) {
    if (diag != nullptr) {
      ++diag->log_scale_clamps;
    }
    return kKvarnLogScaleClamp;
  }
  if (value < -kKvarnLogScaleClamp) {
    if (diag != nullptr) {
      ++diag->log_scale_clamps;
    }
    return -kKvarnLogScaleClamp;
  }
  return value;
}

[[nodiscard]] double AxisRms(std::span<const float> normalized,
                             std::size_t base, std::size_t stride,
                             std::size_t count,
                             KvarnQuantDiagnostics* diag) noexcept {
  double sum = 0.0;
  for (std::size_t n = 0; n < count; ++n) {
    const double v = static_cast<double>(normalized[base + n * stride]);
    sum += v * v;
  }
  double rms = std::sqrt(sum / static_cast<double>(count));
  if (!(rms > kKvarnNormEpsilon)) {
    rms = kKvarnNormEpsilon;
    if (diag != nullptr) {
      ++diag->scale_floors;
    }
  }
  return rms;
}

void ScaleAxis(std::span<float> normalized, std::size_t base,
               std::size_t stride, std::size_t count, double rms) noexcept {
  for (std::size_t n = 0; n < count; ++n) {
    const std::size_t index = base + n * stride;
    normalized[index] =
        static_cast<float>(static_cast<double>(normalized[index]) / rms);
  }
}

}  // namespace

bool KvarnHasNonFinite(std::span<const float> values) noexcept {
  for (const float value : values) {
    if (!std::isfinite(value)) {
      return true;
    }
  }
  return false;
}

void KvarnFwhtHeadInPlace(std::span<float> values) noexcept {
  if (values.size() != kKvarnHeadDim) {
    return;
  }
  for (std::size_t span = 1; span < kKvarnHeadDim; span <<= 1) {
    for (std::size_t base = 0; base < kKvarnHeadDim; base += 2 * span) {
      for (std::size_t j = 0; j < span; ++j) {
        const float a = values[base + j];
        const float b = values[base + j + span];
        values[base + j] = a + b;
        values[base + j + span] = a - b;
      }
    }
  }
  // 1 / sqrt(kKvarnHeadDim) is exact in binary floating point.
  constexpr float kScale = 1.0F / 16.0F;
  for (float& value : values) {
    value *= kScale;
  }
}

KvarnQuantStatus KvarnNormalizeTile(std::span<const float> tile,
                                    std::span<float> normalized,
                                    KvarnTileQuantResult* result,
                                    KvarnQuantDiagnostics* diag) noexcept {
  if (tile.size() != kKvarnTileElements ||
      normalized.size() != kKvarnTileElements || result == nullptr) {
    return KvarnQuantStatus::kOk;
  }

  for (std::size_t index = 0; index < kKvarnTileElements; ++index) {
    const float raw = tile[index];
    if (!std::isfinite(raw)) {
      if (diag != nullptr) {
        ++diag->non_finite_inputs;
      }
      return KvarnQuantStatus::kNonFiniteInput;
    }
    double value = static_cast<double>(raw);
    if (value > kKvarnValueClamp) {
      value = kKvarnValueClamp;
      if (diag != nullptr) {
        ++diag->value_clamps;
      }
    } else if (value < -kKvarnValueClamp) {
      value = -kKvarnValueClamp;
      if (diag != nullptr) {
        ++diag->value_clamps;
      }
    }
    normalized[index] = static_cast<float>(value);
  }

  double log_channel[kKvarnTileChannels] = {};
  double log_token[kKvarnTileTokens] = {};
  for (std::size_t pass = 0; pass < kKvarnNormPasses; ++pass) {
    if (pass % 2 == 0) {
      // Channel pass: one reduction per channel over the 128 tokens, tokens
      // ascending (fixed reduction order, design.md D6).
      for (std::size_t j = 0; j < kKvarnTileChannels; ++j) {
        const double rms =
            AxisRms(normalized, j, kTokenStride, kKvarnTileTokens, diag);
        log_channel[j] = ClampLogScale(log_channel[j] + std::log(rms), diag);
        ScaleAxis(normalized, j, kTokenStride, kKvarnTileTokens, rms);
      }
    } else {
      // Token pass: one reduction per token over the 256 channels, channels
      // ascending.
      for (std::size_t i = 0; i < kKvarnTileTokens; ++i) {
        const std::size_t base = i * kTokenStride;
        const double rms =
            AxisRms(normalized, base, 1, kKvarnTileChannels, diag);
        log_token[i] = ClampLogScale(log_token[i] + std::log(rms), diag);
        ScaleAxis(normalized, base, 1, kKvarnTileChannels, rms);
      }
    }
  }

  for (std::size_t j = 0; j < kKvarnTileChannels; ++j) {
    result->norm_channel_scale[j] =
        static_cast<float>(std::exp(log_channel[j]));
  }
  for (std::size_t i = 0; i < kKvarnTileTokens; ++i) {
    result->norm_token_scale[i] = static_cast<float>(std::exp(log_token[i]));
  }
  return KvarnQuantStatus::kOk;
}

void KvarnEncodeAsymmetricRtn(std::span<const float> values, std::uint8_t bits,
                              float rtn_scale, float rtn_zero,
                              std::span<std::uint8_t> codes) noexcept {
  if (bits < kKvarnMinBits || bits > kKvarnMaxBits ||
      codes.size() < values.size()) {
    return;
  }
  const float qmax = static_cast<float>(CodeMax(bits));
  for (std::size_t n = 0; n < values.size(); ++n) {
    // float subtract/divide/round: the literal device form, so T-01 stage 1
    // compares codes exactly. round() is half-away-from-zero, matching the
    // device roundf.
    float code = std::round((values[n] - rtn_zero) / rtn_scale);
    if (!(code >= 0.0F)) {
      code = 0.0F;
    } else if (code > qmax) {
      code = qmax;
    }
    codes[n] = static_cast<std::uint8_t>(code);
  }
}

void KvarnQuantizeAsymmetricRtn(std::span<const float> normalized,
                                KvarnRtnAxis axis, std::uint8_t bits,
                                KvarnTileQuantResult* result,
                                KvarnQuantDiagnostics* diag) noexcept {
  if (normalized.size() != kKvarnTileElements || result == nullptr ||
      bits < kKvarnMinBits || bits > kKvarnMaxBits) {
    return;
  }
  result->bits = bits;
  const std::size_t set_size = RtnSetSize(axis);
  const double qmax = static_cast<double>(CodeMax(bits));
  float set_values[kKvarnTileTokens] = {};
  std::uint8_t set_codes[kKvarnTileTokens] = {};

  for (std::size_t p = 0; p < kKvarnRtnParams; ++p) {
    double minimum = static_cast<double>(normalized[RtnSetElement(axis, p, 0)]);
    double maximum = minimum;
    for (std::size_t n = 0; n < set_size; ++n) {
      const double v =
          static_cast<double>(normalized[RtnSetElement(axis, p, n)]);
      if (v < minimum) {
        minimum = v;
      }
      if (v > maximum) {
        maximum = v;
      }
      set_values[n] = static_cast<float>(v);
    }
    double scale = (maximum - minimum) / qmax;
    if (!(scale > kKvarnNormEpsilon)) {
      // Constant parameter set (variance 0): floor the scale. Every value is
      // then encoded as zero, so the reconstruction stays exact.
      scale = kKvarnNormEpsilon;
      if (diag != nullptr) {
        ++diag->scale_floors;
      }
    }
    result->rtn_scale[p] = static_cast<float>(scale);
    result->rtn_zero[p] = static_cast<float>(minimum);
    KvarnEncodeAsymmetricRtn(std::span<const float>(set_values, set_size), bits,
                             result->rtn_scale[p], result->rtn_zero[p],
                             std::span<std::uint8_t>(set_codes, set_size));
    for (std::size_t n = 0; n < set_size; ++n) {
      result->codes[RtnSetElement(axis, p, n)] = set_codes[n];
    }
  }
}

void KvarnPackTilePayload(const KvarnTileQuantResult& result,
                          KvarnTilePayload* payload) noexcept {
  if (payload == nullptr) {
    return;
  }
  std::memset(payload, 0, sizeof(*payload));
  for (std::size_t p = 0; p < kKvarnRtnParams; ++p) {
    payload->rtn_scale[p] = KvarnFloatToFp16(result.rtn_scale[p]);
    payload->rtn_zero[p] = KvarnFloatToFp16(result.rtn_zero[p]);
  }
  for (std::size_t j = 0; j < kKvarnTileChannels; ++j) {
    payload->norm_channel_scale[j] =
        KvarnFloatToFp16(result.norm_channel_scale[j]);
  }
  for (std::size_t i = 0; i < kKvarnTileTokens; ++i) {
    payload->norm_token_scale[i] = KvarnFloatToFp16(result.norm_token_scale[i]);
  }
  for (std::size_t index = 0; index < kKvarnTileElements; ++index) {
    const std::uint8_t code = result.codes[index];
    payload->low.bytes[index / 2] |=
        static_cast<std::uint8_t>((code & 0x0FU) << (4U * (index % 2)));
    payload->mid.bytes[index / 8] |=
        static_cast<std::uint8_t>(((code >> 4U) & 1U) << (index % 8));
    payload->high.bytes[index / 8] |=
        static_cast<std::uint8_t>(((code >> 5U) & 1U) << (index % 8));
  }
}

std::uint8_t KvarnUnpackTileCode(const KvarnTilePayload& payload,
                                 std::uint8_t bits,
                                 std::size_t index) noexcept {
  std::uint8_t code = static_cast<std::uint8_t>(
      (payload.low.bytes[index / 2] >> (4U * (index % 2))) & 0x0FU);
  if (bits >= 5) {
    code = static_cast<std::uint8_t>(
        code | (((payload.mid.bytes[index / 8] >> (index % 8)) & 1U) << 4));
  }
  if (bits >= 6) {
    code = static_cast<std::uint8_t>(
        code | (((payload.high.bytes[index / 8] >> (index % 8)) & 1U) << 5));
  }
  return code;
}

float KvarnDecodeTileElement(const KvarnTilePayload& payload, KvarnRtnAxis axis,
                             std::uint8_t bits, std::size_t token,
                             std::size_t channel) noexcept {
  const std::size_t index = token * kKvarnTileChannels + channel;
  const std::size_t p = RtnParam(axis, token, channel);
  const std::uint8_t code = KvarnUnpackTileCode(payload, bits, index);
  return static_cast<float>(code) *
             gufo::quant::Fp16ToFloat(payload.rtn_scale[p]) +
         gufo::quant::Fp16ToFloat(payload.rtn_zero[p]);
}

std::uint16_t KvarnFloatToFp16(float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  const std::uint32_t exponent_bits = (bits >> 23) & 0xFFU;
  const std::uint32_t mantissa = bits & 0x7FFFFFU;
  if (exponent_bits == 0xFFU) {
    // Inf stays Inf; NaN stays NaN rather than collapsing to Inf.
    const std::uint32_t nan_bit = mantissa != 0 ? 0x200U : 0U;
    return static_cast<std::uint16_t>(sign | 0x7C00U | nan_bit);
  }
  const std::int32_t exponent =
      static_cast<std::int32_t>(exponent_bits) - 127 + 15;
  if (exponent >= 0x1F) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
    std::uint32_t half = (mantissa | 0x800000U) >> shift;
    const std::uint32_t remainder =
        (mantissa | 0x800000U) & ((1U << shift) - 1U);
    const std::uint32_t halfway = 1U << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (half & 1U) != 0)) {
      ++half;
    }
    return static_cast<std::uint16_t>(sign | half);
  }
  std::uint32_t half =
      (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13);
  const std::uint32_t remainder = mantissa & 0x1FFFU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (half & 1U) != 0)) {
    ++half;  // May carry into the exponent, which is the correct rounding.
  }
  return static_cast<std::uint16_t>(sign | half);
}

namespace {

constexpr double kBitsPerFp16 = 16.0;
constexpr double kBitsPerByte = 8.0;
constexpr double kBytesPerGib = 1024.0 * 1024.0 * 1024.0;
/// K and V planes.
constexpr double kPlanes = 2.0;
constexpr std::size_t kBaselineBlockValues = 32;

[[nodiscard]] double TileBits(std::uint8_t bits) noexcept {
  return static_cast<double>(kKvarnTileElements) * bits +
         kBitsPerByte * static_cast<double>(kKvarnPayloadMetadataBytes);
}

[[nodiscard]] KvarnMemoryEstimate Finish(double head_bits,
                                         std::size_t capacity_tokens,
                                         std::size_t attention_layers,
                                         std::size_t kv_heads,
                                         double body_bpv) noexcept {
  const double values_per_head =
      kPlanes * static_cast<double>(capacity_tokens) * kKvarnHeadDim;
  KvarnMemoryEstimate out;
  out.body_bpv = body_bpv;
  out.total_bpv = head_bits / values_per_head;
  out.session_gib = head_bits / kBitsPerByte *
                    static_cast<double>(attention_layers * kv_heads) /
                    kBytesPerGib;
  return out;
}

}  // namespace

KvarnRegions ComputeKvarnRegions(
    std::uint8_t k_bits, std::uint8_t v_bits, std::size_t tail_length,
    std::size_t verify_rows, std::size_t capacity_tokens,
    std::size_t attention_layers, std::size_t kv_heads) noexcept {
  const std::size_t body_tokens =
      capacity_tokens > kKvarnSinkRows ? capacity_tokens - kKvarnSinkRows : 0;
  const std::size_t tiles =
      (body_tokens + kKvarnTileTokens - 1) / kKvarnTileTokens;
  const std::size_t ring_rows = tail_length + kKvarnRingMarginRows + verify_rows;
  const std::size_t heads = attention_layers * kv_heads;
  constexpr std::size_t kFp16Bytes = sizeof(std::uint16_t);
  constexpr std::size_t kPlaneCount = static_cast<std::size_t>(kPlanes);

  KvarnRegions regions;
  regions.tile_count = tiles;
  regions.tail_rows = ring_rows;
  regions.sink_elements = heads * kKvarnSinkRows * kKvarnHeadDim;
  regions.tail_elements = heads * ring_rows * kKvarnHeadDim;
  regions.sink_bytes = regions.sink_elements * kPlaneCount * kFp16Bytes;
  regions.tail_bytes = regions.tail_elements * kPlaneCount * kFp16Bytes;
  regions.body_bytes =
      heads * tiles *
      (KvarnTilePayloadBytes(k_bits) + KvarnTilePayloadBytes(v_bits));
  regions.total_bytes =
      regions.sink_bytes + regions.body_bytes + regions.tail_bytes;
  return regions;
}

KvarnMemoryEstimate EstimateKvarnMemory(
    std::uint8_t k_bits, std::uint8_t v_bits, std::size_t tail_length,
    std::size_t verify_rows, std::size_t capacity_tokens,
    std::size_t attention_layers, std::size_t kv_heads) noexcept {
  const std::size_t body_tokens =
      capacity_tokens > kKvarnSinkRows ? capacity_tokens - kKvarnSinkRows : 0;
  const double tiles = std::ceil(static_cast<double>(body_tokens) /
                                 static_cast<double>(kKvarnTileTokens));
  const double fp16_rows = static_cast<double>(
      kKvarnSinkRows + tail_length + kKvarnRingMarginRows + verify_rows);
  const double head_bits = tiles * (TileBits(k_bits) + TileBits(v_bits)) +
                           fp16_rows * kPlanes * kKvarnHeadDim * kBitsPerFp16;
  const double body_bpv = (TileBits(k_bits) + TileBits(v_bits)) /
                          (kPlanes * static_cast<double>(kKvarnTileElements));
  return Finish(head_bits, capacity_tokens, attention_layers, kv_heads,
                body_bpv);
}

KvarnMemoryEstimate EstimateBlockQuantMemory(std::uint8_t bits,
                                             std::size_t capacity_tokens,
                                             std::size_t attention_layers,
                                             std::size_t kv_heads) noexcept {
  const std::size_t body_tokens =
      capacity_tokens > kKvarnSinkRows ? capacity_tokens - kKvarnSinkRows : 0;
  const double bpv = bits + kBitsPerFp16 / kBaselineBlockValues;
  const double head_bits = kPlanes * kKvarnHeadDim *
                           (static_cast<double>(kKvarnSinkRows) * kBitsPerFp16 +
                            static_cast<double>(body_tokens) * bpv);
  return Finish(head_bits, capacity_tokens, attention_layers, kv_heads, bpv);
}

}  // namespace gufo::models::qwen
