#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string_view>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/kvarn_reference.hpp"

namespace {

using gufo::models::qwen::EstimateBlockQuantMemory;
using gufo::models::qwen::EstimateKvarnMemory;
using gufo::models::qwen::kKvarnBitPlaneBytes;
using gufo::models::qwen::kKvarnLowPlaneBytes;
using gufo::models::qwen::kKvarnNormEpsilon;
using gufo::models::qwen::kKvarnPayloadMetadataBytes;
using gufo::models::qwen::kKvarnRtnParams;
using gufo::models::qwen::kKvarnTileChannels;
using gufo::models::qwen::kKvarnTileElements;
using gufo::models::qwen::kKvarnTileTokens;
using gufo::models::qwen::kKvarnValueGroups;
using gufo::models::qwen::KvarnDecodeTileElement;
using gufo::models::qwen::KvarnFloatToFp16;
using gufo::models::qwen::KvarnHasNonFinite;
using gufo::models::qwen::KvarnNormalizeTile;
using gufo::models::qwen::KvarnPackTilePayload;
using gufo::models::qwen::KvarnQuantDiagnostics;
using gufo::models::qwen::KvarnQuantizeAsymmetricRtn;
using gufo::models::qwen::KvarnQuantStatus;
using gufo::models::qwen::KvarnRtnAxis;
using gufo::models::qwen::KvarnTilePayload;
using gufo::models::qwen::KvarnTileQuantResult;
using gufo::models::qwen::KvarnUnpackTileCode;

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

void ExpectNear(float a, float b, float tol, std::string_view msg) {
  if (!(std::abs(a - b) <= tol)) {
    std::cerr << "Assertion failed: " << msg << " (got " << a << ", expected "
              << b << ", diff " << std::abs(a - b) << " > " << tol << ")\n";
    std::exit(1);
  }
}

[[nodiscard]] std::vector<float> MakeTile(std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> tile(kKvarnTileElements);
  for (float& value : tile) {
    value = dist(rng);
  }
  return tile;
}

[[nodiscard]] std::size_t ParamIndex(KvarnRtnAxis axis, std::size_t token,
                                     std::size_t channel) {
  return axis == KvarnRtnAxis::kChannel
             ? channel
             : token * kKvarnValueGroups + channel / 128;
}

void ExpectFiniteScales(const KvarnTileQuantResult& result,
                        std::string_view msg) {
  for (const float value : result.norm_channel_scale) {
    Expect(std::isfinite(value), msg);
  }
  for (const float value : result.norm_token_scale) {
    Expect(std::isfinite(value), msg);
  }
}

void TestPayloadLayoutAndPacking() {
  // Runtime mirror of the header's layout assertions: the sealed payload size
  // is consumed by the snapshot size formula, so it must not drift silently.
  Expect(sizeof(KvarnTilePayload) == kKvarnPayloadMetadataBytes +
                                         kKvarnLowPlaneBytes +
                                         2 * kKvarnBitPlaneBytes,
         "payload size matches metadata + bit planes");

  KvarnTileQuantResult result{};
  result.bits = 6;
  for (std::size_t index = 0; index < kKvarnTileElements; ++index) {
    result.codes[index] = static_cast<std::uint8_t>(index % 64);
  }
  KvarnTilePayload payload{};
  KvarnPackTilePayload(result, &payload);

  for (std::size_t index = 0; index < kKvarnTileElements; ++index) {
    Expect(KvarnUnpackTileCode(payload, 6, index) == result.codes[index],
           "6-bit code round-trips through the bit planes");
    Expect(
        KvarnUnpackTileCode(payload, 5, index) == (result.codes[index] & 0x1FU),
        "5-bit path reads the same alignment");
    Expect(
        KvarnUnpackTileCode(payload, 4, index) == (result.codes[index] & 0x0FU),
        "4-bit path reads the low plane with the same alignment");
  }

  // FP16 metadata conversion rounds to nearest even, so payload scales decode
  // back to a finite value within half an ulp.
  Expect(gufo::quant::Fp16ToFloat(KvarnFloatToFp16(1.0F)) == 1.0F,
         "FP16 conversion is exact for 1.0");
  Expect(gufo::quant::Fp16ToFloat(KvarnFloatToFp16(65504.0F)) == 65504.0F,
         "FP16 conversion is exact for the largest finite value");
}

void TestNormalizationIsDeterministicAndInvertible() {
  const std::vector<float> tile = MakeTile(1);
  std::vector<float> first(kKvarnTileElements);
  std::vector<float> second(kKvarnTileElements);
  KvarnTileQuantResult result_a{};
  KvarnTileQuantResult result_b{};
  KvarnQuantDiagnostics diag_a{};
  KvarnQuantDiagnostics diag_b{};

  Expect(KvarnNormalizeTile(tile, first, &result_a, &diag_a) ==
             KvarnQuantStatus::kOk,
         "normalization succeeds for a finite tile");
  Expect(KvarnNormalizeTile(tile, second, &result_b, &diag_b) ==
             KvarnQuantStatus::kOk,
         "normalization succeeds twice");
  Expect(std::memcmp(first.data(), second.data(),
                     kKvarnTileElements * sizeof(float)) == 0,
         "normalization is deterministic");
  Expect(diag_a.value_clamps == 0 && diag_a.scale_floors == 0,
         "a well-conditioned tile needs no clamp or floor");
  ExpectFiniteScales(result_a, "normalization scales are finite");

  for (std::size_t i = 0; i < kKvarnTileTokens; ++i) {
    for (std::size_t j = 0; j < kKvarnTileChannels; ++j) {
      const std::size_t index = i * kKvarnTileChannels + j;
      const float restored = first[index] * result_a.norm_channel_scale[j] *
                             result_a.norm_token_scale[i];
      ExpectNear(restored, tile[index], 2e-4F * (std::abs(tile[index]) + 1.0F),
                 "normalization inverts through the two scale vectors");
    }
  }
}

void TestConstantAndZeroChannelStaysFinite() {
  std::vector<float> tile = MakeTile(2);
  constexpr std::size_t kZeroChannel = 3;
  constexpr std::size_t kConstantChannel = 7;
  for (std::size_t i = 0; i < kKvarnTileTokens; ++i) {
    tile[i * kKvarnTileChannels + kZeroChannel] = 0.0F;
    tile[i * kKvarnTileChannels + kConstantChannel] = 2.5F;
  }

  std::vector<float> normalized(kKvarnTileElements);
  KvarnTileQuantResult result{};
  KvarnQuantDiagnostics diag{};
  Expect(KvarnNormalizeTile(tile, normalized, &result, &diag) ==
             KvarnQuantStatus::kOk,
         "normalization accepts a zero channel");
  Expect(diag.scale_floors >= 1,
         "the epsilon floor covers the zero channel's RMS");
  Expect(diag.log_scale_clamps >= 1,
         "the accumulated log-scale clamp bounds the zero channel");
  ExpectFiniteScales(result, "constant and zero channels keep finite scales");
  for (const float value : normalized) {
    Expect(std::isfinite(value), "normalized values stay finite");
  }
  for (std::size_t i = 0; i < kKvarnTileTokens; ++i) {
    Expect(normalized[i * kKvarnTileChannels + kZeroChannel] == 0.0F,
           "the zero channel stays zero");
    const float restored =
        normalized[i * kKvarnTileChannels + kConstantChannel] *
        result.norm_channel_scale[kConstantChannel] *
        result.norm_token_scale[i];
    ExpectNear(restored, 2.5F, 2e-4F * 3.5F,
               "the constant channel restores through its floor");
  }
}

void TestExtremeOutlierIsClamped() {
  std::vector<float> tile = MakeTile(3);
  constexpr std::size_t kOutlierIndex = 5 * kKvarnTileChannels + 11;
  tile[kOutlierIndex] = 1.0e30F;

  std::vector<float> normalized(kKvarnTileElements);
  KvarnTileQuantResult result{};
  KvarnQuantDiagnostics diag{};
  Expect(KvarnNormalizeTile(tile, normalized, &result, &diag) ==
             KvarnQuantStatus::kOk,
         "normalization accepts an extreme outlier");
  Expect(diag.value_clamps == 1, "the outlier is clamped exactly once");
  ExpectFiniteScales(result, "outlier keeps finite scales");
  for (const float value : normalized) {
    Expect(std::isfinite(value), "clamped tile stays finite");
  }
}

void TestNonFiniteInputIsDetected() {
  std::vector<float> tile = MakeTile(4);
  constexpr std::size_t kIndex = 100;
  tile[kIndex] = std::numeric_limits<float>::quiet_NaN();
  Expect(KvarnHasNonFinite(tile), "NaN is detected");
  std::vector<float> normalized(kKvarnTileElements);
  KvarnTileQuantResult result{};
  KvarnQuantDiagnostics diag{};
  Expect(KvarnNormalizeTile(tile, normalized, &result, &diag) ==
             KvarnQuantStatus::kNonFiniteInput,
         "NaN input fails normalization");
  Expect(diag.non_finite_inputs == 1, "NaN is counted once");

  tile[kIndex] = std::numeric_limits<float>::infinity();
  Expect(KvarnHasNonFinite(tile), "Inf is detected");
  Expect(KvarnNormalizeTile(tile, normalized, &result, &diag) ==
             KvarnQuantStatus::kNonFiniteInput,
         "Inf input fails normalization");
  diag = KvarnQuantDiagnostics{};
  Expect(KvarnNormalizeTile(tile, normalized, &result, &diag) ==
             KvarnQuantStatus::kNonFiniteInput,
         "Inf input fails normalization on a fresh diagnostic record");
  Expect(diag.non_finite_inputs == 1, "Inf is counted once");
}

void TestAsymmetricRtnRoundTrip() {
  std::vector<float> tile = MakeTile(5);
  for (float& value : tile) {
    value += 0.5F;  // Make the parameter sets asymmetric.
  }
  std::vector<float> normalized(kKvarnTileElements);
  KvarnTileQuantResult norm{};
  Expect(KvarnNormalizeTile(tile, normalized, &norm, nullptr) ==
             KvarnQuantStatus::kOk,
         "normalization succeeds before RTN");

  for (const std::uint8_t bits : {4, 5, 6}) {
    for (const KvarnRtnAxis axis :
         {KvarnRtnAxis::kChannel, KvarnRtnAxis::kTokenGroup}) {
      KvarnTileQuantResult quant{};
      KvarnQuantDiagnostics diag{};
      KvarnQuantizeAsymmetricRtn(normalized, axis, bits, &quant, &diag);
      Expect(quant.bits == bits, "bit width is recorded");
      Expect(diag.scale_floors == 0,
             "a random tile needs no scale floor in any parameter set");

      const std::uint32_t max_code = (1U << bits) - 1U;
      for (const std::uint8_t code : quant.codes) {
        Expect(code <= max_code, "codes stay inside the bit width");
      }

      for (std::size_t i = 0; i < kKvarnTileTokens; ++i) {
        for (std::size_t j = 0; j < kKvarnTileChannels; ++j) {
          const std::size_t index = i * kKvarnTileChannels + j;
          const std::size_t p = ParamIndex(axis, i, j);
          const float scale = quant.rtn_scale[p];
          const float reconstructed =
              static_cast<float>(quant.codes[index]) * scale +
              quant.rtn_zero[p];
          ExpectNear(reconstructed, normalized[index], scale * 0.5F + 1e-6F,
                     "RTN reconstructs within half a step");
        }
      }

      KvarnTilePayload payload{};
      KvarnPackTilePayload(quant, &payload);
      for (std::size_t i = 0; i < kKvarnTileTokens; i += 7) {
        for (std::size_t j = 0; j < kKvarnTileChannels; j += 11) {
          const std::size_t index = i * kKvarnTileChannels + j;
          const std::size_t p = ParamIndex(axis, i, j);
          const float scale_16 = gufo::quant::Fp16ToFloat(payload.rtn_scale[p]);
          const float zero_16 = gufo::quant::Fp16ToFloat(payload.rtn_zero[p]);
          const float expected =
              static_cast<float>(quant.codes[index]) * scale_16 + zero_16;
          ExpectNear(KvarnDecodeTileElement(payload, axis, bits, i, j),
                     expected, 1e-6F * (std::abs(zero_16) + 1.0F),
                     "packed payload decodes the same element");
        }
      }
    }
  }
}

void TestZeroSpanFallsBackToEpsilon() {
  // A constant parameter set has zero range, so its scale is floored instead
  // of divided by zero and every value encodes as zero (exact reconstruction).
  const std::vector<float> normalized(kKvarnTileElements, 1.0F);
  for (const KvarnRtnAxis axis :
       {KvarnRtnAxis::kChannel, KvarnRtnAxis::kTokenGroup}) {
    KvarnTileQuantResult quant{};
    KvarnQuantDiagnostics diag{};
    KvarnQuantizeAsymmetricRtn(normalized, axis, 4, &quant, &diag);
    Expect(diag.scale_floors == kKvarnRtnParams,
           "every constant set floors its scale");
    ExpectNear(quant.rtn_scale[0], static_cast<float>(kKvarnNormEpsilon), 0.0F,
               "the floored scale is the declared epsilon");
    for (const std::uint8_t code : quant.codes) {
      Expect(code == 0, "a constant set encodes as zero");
    }
    ExpectNear(quant.rtn_zero[0], 1.0F, 1e-6F,
               "the zero-point keeps the constant value");
  }
}

/// The size formula of design.md D5 reproduces the design.md D1 table:
/// total bpv (= GiB per session at 262,144 tokens) for every preset and tail.
void TestMemoryEstimateMatchesDesignTable() {
  constexpr std::size_t kCapacity = 262144;
  constexpr std::size_t kLayers = 16;
  constexpr std::size_t kHeads = 4;
  constexpr std::size_t kW = 9;
  struct Row {
    std::uint8_t k, v;
    double body;
    double total[4];  // unset (L = W), 256, 1024, 2048
  };
  constexpr Row kTable[] = {
      {4, 4, 4.4375, {4.45, 4.47, 4.51, 4.58}},
      {5, 4, 4.9375, {4.95, 4.97, 5.01, 5.08}},
      {5, 5, 5.4375, {5.45, 5.47, 5.51, 5.58}},
      {6, 5, 5.9375, {5.95, 5.97, 6.01, 6.08}},
  };
  constexpr std::size_t kTails[4] = {kW, 256, 1024, 2048};
  for (const Row& row : kTable) {
    for (int i = 0; i < 4; ++i) {
      const auto estimate = EstimateKvarnMemory(row.k, row.v, kTails[i], kW,
                                                kCapacity, kLayers, kHeads);
      Expect(std::fabs(estimate.total_bpv - row.total[i]) < 0.005,
             "total bpv must match the design table");
      Expect(std::fabs(estimate.session_gib - estimate.total_bpv) < 1e-9,
             "at 262,144 tokens GiB per session equals total bpv");
    }
    const auto body =
        EstimateKvarnMemory(row.k, row.v, kW, kW, kCapacity, kLayers, kHeads);
    Expect(std::fabs(body.body_bpv - row.body) < 1e-9,
           "body bpv must match the design table");
  }
  // The quality-first budget of 6.4 bpv holds up to L = 2048 for k6v5, and the
  // memory-first 4.6 bpv budget holds for k4v4 across the whole range.
  Expect(EstimateKvarnMemory(4, 4, 2048, kW, kCapacity, kLayers, kHeads)
                 .total_bpv <= 4.6,
         "kvarn-k4v4 total must stay within 4.6 bpv at L = 2048");
  Expect(EstimateKvarnMemory(6, 5, 2048, kW, kCapacity, kLayers, kHeads)
                 .total_bpv <= 6.4,
         "k6v5 total must stay within 6.4 bpv at L = 2048");
  // Baselines: block 32 with an FP16 scale is 8.5 / 4.5 bpv plus the sink.
  const auto q8 = EstimateBlockQuantMemory(8, kCapacity, kLayers, kHeads);
  const auto q4 = EstimateBlockQuantMemory(4, kCapacity, kLayers, kHeads);
  Expect(q8.body_bpv == 8.5 && q4.body_bpv == 4.5, "baseline body bpv");
  Expect(q8.total_bpv > 8.5 && q8.total_bpv < 8.55 && q4.total_bpv > 4.5 &&
             q4.total_bpv < 4.55,
         "baseline total includes the FP16 sink");
}

}  // namespace

int main() {
  std::cout << "Running KVarN reference oracle tests...\n";
  TestPayloadLayoutAndPacking();
  TestNormalizationIsDeterministicAndInvertible();
  TestConstantAndZeroChannelStaysFinite();
  TestExtremeOutlierIsClamped();
  TestNonFiniteInputIsDetected();
  TestAsymmetricRtnRoundTrip();
  TestZeroSpanFallsBackToEpsilon();
  TestMemoryEstimateMatchesDesignTable();
  std::cout << "All KVarN reference oracle tests passed successfully!\n";
  return 0;
}
