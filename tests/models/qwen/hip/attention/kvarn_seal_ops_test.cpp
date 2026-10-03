// KVarN seal kernel checks (openspec add-kvarn-kv-cache-quantization 3.4):
// T-01 two-stage oracle comparison, determinism, bit-plane layout and the
// non-finite error flag (design D4/D5/D6/D15).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/kvarn.hpp"
#include "src/models/qwen/kvarn_reference.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/testing/test_common.hpp"

namespace {

namespace ref = gufo::models::qwen;

constexpr std::uint32_t kRows =
    static_cast<std::uint32_t>(ref::kKvarnTileTokens);
constexpr std::uint32_t kCols =
    static_cast<std::uint32_t>(ref::kKvarnTileChannels);
constexpr std::uint32_t kElements = kRows * kCols;
// Declared in QUALITY.md: the GPU normalizes each element as
// `raw / (channel_scale * token_scale)` with the FP16-stored scale vectors (the
// exact form the reader inverts), while the oracle keeps progressive float
// values; the two differ by the FP16 rounding of the two scale vectors, which a
// range-of-extremes can amplify. Stage 1 pins the codes exactly, so this limit
// only has to catch a wrong pass count or axis.
constexpr double kScaleLimit = 1e-2;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void ExpectNear(double actual, double expected, const std::string& message) {
  const double tolerance =
      kScaleLimit *
      std::max(1.0, std::max(std::abs(actual), std::abs(expected)));
  if (std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(message + ": " + std::to_string(actual) + " vs " +
                             std::to_string(expected));
  }
}

/// Deterministic tile with a constant channel and an extreme outlier, so the
/// epsilon floor and the clamp both engage.
std::vector<float> MakeTile() {
  std::vector<float> tile(kElements);
  for (std::uint32_t r = 0; r < kRows; ++r) {
    for (std::uint32_t c = 0; c < kCols; ++c) {
      float value = 0.35F * std::sin(static_cast<float>(r + 1) * 0.017F +
                                     static_cast<float>(c + 1) * 0.011F);
      if (c == 7) {
        value = 0.0F;  // constant channel: the RMS floor must apply
      }
      if (r == 3 && c == 9) {
        value = 5000.0F;  // outlier: the value clamp must apply
      }
      tile[r * kCols + c] = value;
    }
  }
  return tile;
}

struct SealHarness {
  gufo::hip::KvarnSealState state{1, 1};
  void* sink{nullptr};
  void* tail{nullptr};
  std::uint8_t* body{nullptr};
  std::uint32_t sink_plane = 128 * kCols;
  std::uint32_t tail_plane = (9 + 128 + 9) * kCols;
  std::uint32_t ring_rows = 9 + 128 + 9;
  std::uint32_t tile_count = 2;
  std::uint32_t k_tile_bytes = 0;
  std::uint32_t v_tile_bytes = 0;
  std::uint32_t k_bits = 4;
  std::uint32_t v_bits = 4;

  explicit SealHarness(std::uint8_t k_b, std::uint8_t v_b,
                       std::uint32_t tail_length = 0)
      : k_bits(k_b), v_bits(v_b) {
    if (tail_length != 0) {
      ring_rows = tail_length + 128 + 9;
      tail_plane = ring_rows * kCols;
    }
    k_tile_bytes = static_cast<std::uint32_t>(ref::KvarnTilePayloadBytes(k_b));
    v_tile_bytes = static_cast<std::uint32_t>(ref::KvarnTilePayloadBytes(v_b));
    HIP_CHECK(hipMalloc(&sink, 2 * sink_plane * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&tail, 2 * tail_plane * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&body, static_cast<std::size_t>(tile_count) *
                                   (k_tile_bytes + v_tile_bytes)));
    HIP_CHECK(hipMemset(sink, 0, 2 * sink_plane * sizeof(std::uint16_t)));
    HIP_CHECK(hipMemset(tail, 0, 2 * tail_plane * sizeof(std::uint16_t)));
    HIP_CHECK(hipMemset(
        body, 0,
        static_cast<std::size_t>(tile_count) * (k_tile_bytes + v_tile_bytes)));
  }

  ~SealHarness() {
    (void)hipFree(sink);
    (void)hipFree(tail);
    (void)hipFree(body);
  }

  /// Writes `tile` into the tail-ring rows of tile 0 (positions 128..255) and
  /// seals it. Tile 0 covers absolute positions 128 + 128 t, so its rows live
  /// in the ring, not the sink (design D2/D4).
  void Seal(std::span<const float> tile, bool non_finite = false,
            hipStream_t stream = nullptr) {
    std::vector<std::uint16_t> rows(kRows * kCols);
    for (std::uint32_t r = 0; r < kRows; ++r) {
      for (std::uint32_t c = 0; c < kCols; ++c) {
        rows[r * kCols + c] =
            __half_as_ushort(__float2half_rn(tile[r * kCols + c]));
      }
    }
    if (non_finite) {
      rows[5 * kCols + 11] = __half_as_ushort(
          __float2half_rn(std::numeric_limits<float>::infinity()));
    }
    const std::size_t plane_bytes = tail_plane * sizeof(std::uint16_t);
    HIP_CHECK(hipMemcpy(tail, rows.data(), rows.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(static_cast<std::uint8_t*>(tail) + plane_bytes,
                        rows.data(), rows.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));
    auto preset = gufo::hip::QwenKvarnPreset{};
    preset.k_bits = static_cast<std::uint8_t>(k_bits);
    preset.v_bits = static_cast<std::uint8_t>(v_bits);
    gufo::hip::LaunchKvarnSeal(
        state, preset, sink, tail, sink_plane, tail_plane, ring_rows, body,
        tile_count, /*layer_idx=*/0, /*num_kv_heads=*/1, /*start=*/265, nullptr,
        /*count=*/1, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  [[nodiscard]] ref::KvarnTilePayload Payload(bool is_value) const {
    ref::KvarnTilePayload payload{};
    HIP_CHECK(hipMemcpy(&payload, body + (is_value ? k_tile_bytes : 0),
                        sizeof(payload), hipMemcpyDeviceToHost));
    return payload;
  }

  [[nodiscard]] bool Sealed() const {
    std::uint32_t value = 0;
    HIP_CHECK(hipMemcpy(&value, state.sealed(), sizeof(value),
                        hipMemcpyDeviceToHost));
    return value != 0;
  }
};

/// Stage 1 of T-01: reconstruct the normalized values from the payload's FP16
/// scales and encode with the payload's FP16 RTN parameters; the codes must
/// match exactly.
void StageOne(const ref::KvarnTilePayload& payload, std::span<const float> raw,
              std::uint8_t bits, bool is_value) {
  const bool token_axis = is_value;
  const std::size_t set_size =
      token_axis ? ref::kKvarnValueGroupChannels : ref::kKvarnTileTokens;
  std::vector<float> values(set_size);
  std::vector<std::uint8_t> codes(set_size);
  for (std::size_t p = 0; p < ref::kKvarnRtnParams; ++p) {
    for (std::size_t n = 0; n < set_size; ++n) {
      std::size_t token = 0, channel = 0;
      if (!token_axis) {
        token = n;
        channel = p;
      } else {
        token = p / ref::kKvarnValueGroups;
        channel =
            (p % ref::kKvarnValueGroups) * ref::kKvarnValueGroupChannels + n;
      }
      const double channel_scale =
          gufo::quant::Fp16ToFloat(payload.norm_channel_scale[channel]);
      const double token_scale =
          gufo::quant::Fp16ToFloat(payload.norm_token_scale[token]);
      const float clamped =
          std::min(std::max(raw[token * kCols + channel],
                            -static_cast<float>(ref::kKvarnValueClamp)),
                   static_cast<float>(ref::kKvarnValueClamp));
      const double divisor = channel_scale * token_scale;
      values[n] =
          divisor > 0.0
              ? static_cast<float>(static_cast<double>(clamped) / divisor)
              : 0.0F;
    }
    ref::KvarnEncodeAsymmetricRtn(
        values, bits, gufo::quant::Fp16ToFloat(payload.rtn_scale[p]),
        gufo::quant::Fp16ToFloat(payload.rtn_zero[p]), codes);
    for (std::size_t n = 0; n < set_size; ++n) {
      std::size_t token = 0, channel = 0;
      if (!token_axis) {
        token = n;
        channel = p;
      } else {
        token = p / ref::kKvarnValueGroups;
        channel =
            (p % ref::kKvarnValueGroups) * ref::kKvarnValueGroupChannels + n;
      }
      const std::uint8_t actual =
          ref::KvarnUnpackTileCode(payload, bits, token * kCols + channel);
      Expect(actual == codes[n],
             "stage 1: the oracle must reproduce the GPU code exactly");
    }
  }
}

/// Stage 2 of T-01: the scales must match the CPU double computation within the
/// declared limit.
void StageTwo(const ref::KvarnTilePayload& payload, std::span<const float> raw,
              std::uint8_t bits, bool is_value) {
  std::vector<float> normalized(kElements);
  ref::KvarnTileQuantResult result;
  Expect(ref::KvarnNormalizeTile(raw, normalized, &result, nullptr) ==
             ref::KvarnQuantStatus::kOk,
         "the oracle must accept the finite tile");
  ref::KvarnQuantizeAsymmetricRtn(
      normalized,
      is_value ? ref::KvarnRtnAxis::kTokenGroup : ref::KvarnRtnAxis::kChannel,
      bits, &result, nullptr);
  const std::string plane = is_value ? "V" : "K";
  for (std::size_t c = 0; c < kCols; ++c) {
    ExpectNear(
        gufo::quant::Fp16ToFloat(payload.norm_channel_scale[c]),
        result.norm_channel_scale[c],
        "stage 2: " + plane + " channel scale [" + std::to_string(c) + "]");
  }
  for (std::size_t r = 0; r < kRows; ++r) {
    ExpectNear(
        gufo::quant::Fp16ToFloat(payload.norm_token_scale[r]),
        result.norm_token_scale[r],
        "stage 2: " + plane + " token scale [" + std::to_string(r) + "]");
  }
  for (std::size_t p = 0; p < ref::kKvarnRtnParams; ++p) {
    ExpectNear(gufo::quant::Fp16ToFloat(payload.rtn_scale[p]),
               result.rtn_scale[p],
               "stage 2: " + plane + " RTN scale [" + std::to_string(p) + "]");
    ExpectNear(
        gufo::quant::Fp16ToFloat(payload.rtn_zero[p]), result.rtn_zero[p],
        "stage 2: " + plane + " RTN zero point [" + std::to_string(p) + "]");
  }
}

void TestSealOracle(std::uint8_t k_bits, std::uint8_t v_bits) {
  const std::vector<float> tile = MakeTile();
  // The seal reads F16 rows, so the oracle must see the same rounded values.
  std::vector<float> stored(kElements);
  for (std::size_t i = 0; i < kElements; ++i) {
    stored[i] = __half2float(__float2half_rn(tile[i]));
  }
  SealHarness harness(k_bits, v_bits);
  harness.Seal(tile);
  Expect(harness.Sealed(), "the seal must advance the tile counter");
  for (const bool is_value : {false, true}) {
    const auto payload = harness.Payload(is_value);
    const std::uint8_t bits =
        static_cast<std::uint8_t>(is_value ? v_bits : k_bits);
    StageOne(payload, stored, bits, is_value);
    StageTwo(payload, stored, bits, is_value);
    // Constant channels and outliers stay finite after the floors and clamps.
    for (std::uint16_t value : payload.rtn_scale) {
      Expect(std::isfinite(gufo::quant::Fp16ToFloat(value)),
             "RTN scales must be finite");
    }
  }
}

void TestSealDeterminism() {
  const std::vector<float> tile = MakeTile();
  SealHarness first(6, 5);
  SealHarness second(6, 5);
  first.Seal(tile);
  second.Seal(tile);
  const auto a = first.Payload(false);
  const auto b = second.Payload(false);
  Expect(std::memcmp(&a, &b, sizeof(a)) == 0,
         "sealing the same tile twice must be bit-identical");
}

void TestBitPlaneLayout() {
  const std::vector<float> tile = MakeTile();
  for (const std::uint8_t bits : {4, 5, 6}) {
    SealHarness harness(bits, bits);
    harness.Seal(tile);
    const auto payload = harness.Payload(false);
    for (std::size_t index = 0; index < kElements; ++index) {
      const std::uint8_t low = static_cast<std::uint8_t>(
          (payload.low.bytes[index / 2] >> (4U * (index % 2))) & 0x0FU);
      std::uint8_t expected = low;
      if (bits >= 5) {
        expected = static_cast<std::uint8_t>(
            expected |
            (((payload.mid.bytes[index / 8] >> (index % 8)) & 1U) << 4));
      }
      if (bits >= 6) {
        expected = static_cast<std::uint8_t>(
            expected |
            (((payload.high.bytes[index / 8] >> (index % 8)) & 1U) << 5));
      }
      Expect(ref::KvarnUnpackTileCode(payload, bits, index) == expected,
             "the bit planes must unpack with the 4-bit alignment");
      Expect(expected <= ((1U << bits) - 1U), "codes must fit the bit width");
    }
  }
}

void TestNonFiniteFlag() {
  const std::vector<float> tile = MakeTile();
  SealHarness harness(4, 4);
  harness.Seal(tile, /*non_finite=*/true);
  Expect(harness.state.HasError(nullptr),
         "a non-finite tile must raise the session error flag");
  Expect(!harness.Sealed(), "a non-finite tile must not be recorded as sealed");
}

}  // namespace

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "Qwen KVarN seal ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }
  TestSealOracle(4, 4);
  TestSealOracle(6, 5);
  TestSealDeterminism();
  TestBitPlaneLayout();
  TestNonFiniteFlag();
  std::cout << "Qwen KVarN seal ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen KVarN seal ops test.\n";
  return 77;
#endif
}
#endif  // defined(ENGINE_ENABLE_HIP)
