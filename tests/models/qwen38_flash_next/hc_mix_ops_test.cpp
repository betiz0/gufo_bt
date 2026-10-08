// Hyper-connection Mixer operator checks: the fused F16 projection route and
// its exact-agreement replay, the decoded Q6_K / IQ4_NL Mixer projections, the
// mixed-format SSM gate stack, the epilogue's scalar and four-lane variants,
// the Swift-1.5 Mixer down/up projections executed from their stored Q4_K /
// Q5_0 encodings through the vendored adapter (including the down -> SiLU ->
// up -> epilogue chain), and the mixed-format model whose blocks store
// different Mixer down formats and are each checked against the FP32
// reference of their own encoding.
//
// Every expectation is compared against a CPU dequantization of the same
// packed blocks; the stored-encoding chain additionally quantizes its
// activations to Q8_1 exactly as the kernels do.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "tests/models/qwen38_flash_next/quant_fixture.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace fixture = gufo::models::qwen38_flash_next::fixture;
namespace {

constexpr std::uint32_t kTokens = 19;
constexpr std::uint32_t kHidden = 2560;
constexpr std::uint32_t kStreams = 4;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

template<typename T>
class HipBuffer {
public:
  explicit HipBuffer(std::size_t count) : count_(count) {
    void* allocation = nullptr;
    CheckHip(hipMalloc(&allocation, bytes()), "hipMalloc");
    data_ = static_cast<T*>(allocation);
  }
  ~HipBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  HipBuffer(const HipBuffer&) = delete;
  HipBuffer& operator=(const HipBuffer&) = delete;
  HipBuffer(HipBuffer&&) = delete;
  HipBuffer& operator=(HipBuffer&&) = delete;

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

private:
  T* data_{nullptr};
  std::size_t count_{0};
};

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

std::vector<float> MakeValues(std::size_t count, std::uint32_t seed,
                              float scale) {
  std::vector<float> values(count);
  for (float& value : values) {
    value = scale *
            static_cast<float>(static_cast<int>(NextRandom(&seed) & 0xFFFFU) -
                               32768) /
            32768.0F;
  }
  return values;
}

void Upload(HipBuffer<float>* destination, const std::vector<float>& source) {
  CheckHip(hipMemcpy(destination->get(), source.data(), destination->bytes(),
                     hipMemcpyHostToDevice),
           "upload");
}

std::vector<float> Download(HipBuffer<float>* source, std::size_t count) {
  std::vector<float> values(count);
  CheckHip(hipMemcpy(values.data(), source->get(), source->bytes(),
                     hipMemcpyDeviceToHost),
           "download");
  return values;
}

// Fusing the projection must preserve every downstream representation,
// including the inject partials' reduction order and quantization ties.
void CheckFusedProjection(std::uint32_t tokens) {
  constexpr std::size_t kRank = 320;
  constexpr std::size_t kDim = kStreams * kHidden;
  const std::size_t mixed_count = static_cast<std::size_t>(tokens) * kHidden;
  const std::size_t inject_count = static_cast<std::size_t>(tokens) * kStreams *
                                   q::HcInjectPartsVec4(kHidden);
  const std::size_t q8_bytes = q::Q8TiledBytes(tokens, kHidden);
  const auto low =
      MakeValues(static_cast<std::size_t>(tokens) * kRank, 0x12345678U, 1.0F);
  const auto norm =
      MakeValues(static_cast<std::size_t>(tokens) * kDim, 0x87654321U, 0.5F);
  std::vector<__half> low_half(low.size()), norm_half(norm.size());
  std::transform(low.begin(), low.end(), low_half.begin(),
                 [](float v) { return __float2half_rn(v); });
  std::transform(norm.begin(), norm.end(), norm_half.begin(),
                 [](float v) { return __float2half_rn(v); });
  std::vector<std::uint8_t> weights(kDim * (kRank / 32) * 34);
  std::uint32_t seed = 0xABCD0123U;
  for (std::size_t block = 0; block < weights.size() / 34; ++block) {
    const __half scale = __float2half_rn(
        static_cast<float>(1 + NextRandom(&seed) % 13) / 2048.0F);
    std::memcpy(weights.data() + block * 34, &scale, sizeof(scale));
    for (std::size_t j = 0; j < 32; ++j) {
      weights[block * 34 + 2 + j] = static_cast<std::uint8_t>(
          static_cast<int>(NextRandom(&seed) % 255) - 127);
    }
  }
  HipBuffer<std::uint8_t> d_up(weights.size());
  HipBuffer<__half> d_low(low.size()), d_norm(norm.size());
  HipBuffer<float> d_inject_w(kStreams * kDim);
  HipBuffer<float> d_gate(norm.size()), d_ref(mixed_count), d_out(mixed_count);
  HipBuffer<__half> d_ref_half(mixed_count), d_out_half(mixed_count);
  HipBuffer<std::uint8_t> d_ref_q8(q8_bytes), d_out_q8(q8_bytes);
  HipBuffer<float> d_ref_inject(inject_count), d_out_inject(inject_count);
  CheckHip(hipMemcpy(d_up.get(), weights.data(), d_up.bytes(),
                     hipMemcpyHostToDevice),
           "up weights");
  CheckHip(hipMemcpy(d_low.get(), low_half.data(), d_low.bytes(),
                     hipMemcpyHostToDevice),
           "low rank input");
  CheckHip(hipMemcpy(d_norm.get(), norm_half.data(), d_norm.bytes(),
                     hipMemcpyHostToDevice),
           "normalized streams");
  Upload(&d_inject_w, MakeValues(kStreams * kDim, 0xDEADBEEFU, 0.1F));
  CheckHip(hipMemset(d_ref_q8.get(), 0, q8_bytes), "clear reference Q8");
  CheckHip(hipMemset(d_out_q8.get(), 0, q8_bytes), "clear fused Q8");
  if (!q::DenseF16Gemm(d_up.get(), d_low.get(), d_gate.get(), tokens, kDim,
                       kRank, nullptr)) {
    throw std::runtime_error("HC reference projection rejected");
  }
  q::HcMixEpilogueVec4F16(d_norm.get(), d_gate.get(), d_inject_w.get(),
                          d_ref.get(), d_ref_half.get(), d_ref_q8.get(),
                          d_ref_inject.get(), tokens, kHidden, nullptr);
  const auto exact = [](auto& reference, auto& actual, const char* name) {
    std::vector<std::uint8_t> a(reference.bytes()), b(actual.bytes());
    CheckHip(
        hipMemcpy(a.data(), reference.get(), a.size(), hipMemcpyDeviceToHost),
        "reference download");
    CheckHip(hipMemcpy(b.data(), actual.get(), b.size(), hipMemcpyDeviceToHost),
             "fused download");
    if (a != b) {
      throw std::runtime_error(std::string("HC fusion changed ") + name);
    }
  };
  // Small case covers all optional outputs; the large ragged case also
  // replays the full path to catch cross-block races and tail writes.
  const unsigned first = tokens == 96 ? 0 : 7;
  const unsigned end = tokens == 96 ? 8 : 9;
  for (unsigned iteration = first; iteration < end; ++iteration) {
    const unsigned outputs = std::min(iteration, 7U);
    if (!q::HcMixF16Gemm(d_up.get(), d_low.get(), d_norm.get(),
                         outputs & 4 ? d_inject_w.get() : nullptr, d_out.get(),
                         outputs & 1 ? d_out_half.get() : nullptr,
                         outputs & 2 ? d_out_q8.get() : nullptr,
                         outputs & 4 ? d_out_inject.get() : nullptr, tokens,
                         kHidden, kRank, nullptr)) {
      throw std::runtime_error("HC fused projection rejected");
    }
    exact(d_ref, d_out, "F32 mixed row");
    if (outputs & 1)
      exact(d_ref_half, d_out_half, "F16 mixed row");
    if (outputs & 2)
      exact(d_ref_q8, d_out_q8, "Q8 mixed row");
    if (outputs & 4)
      exact(d_ref_inject, d_out_inject, "inject partials");
  }
  std::cerr << "HC projection fusion exact at " << tokens << " tokens\n";
}

/// The Mixer projections of a Q6_K / IQ4_NL artifact are decoded at load
/// (design D2), and their consumers are the fixed-size kernels: SmallGemm
/// over F16 weights at decode, SmallGemm with F32 weights for the inject
/// projection. Both must reproduce the FP32 reference over the dequantized
/// rows.
void CheckDecodedMixer() {
  constexpr std::size_t kRank = 320;
  constexpr std::size_t kDim = kStreams * kHidden;
  constexpr std::uint32_t kDecode = 4;
  const auto down = fixture::MakeQ6Weights(kRank, kDim, 0x0D0A1234U);
  const auto up = fixture::MakeIq4NlWeights(kDim, kRank, 0x0B4C5678U);
  const auto inject = fixture::MakeQ6Weights(kStreams, kDim, 0x1D3C9AB0U);
  const auto x = MakeValues(kDecode * kDim, 0x5EED1234U, 0.5F);

  HipBuffer<std::uint8_t> d_down_in(down.bytes.size());
  HipBuffer<std::uint8_t> d_up_in(up.bytes.size());
  HipBuffer<std::uint8_t> d_inject_in(inject.bytes.size());
  HipBuffer<__half> d_down(kRank * kDim), d_up(kDim * kRank);
  HipBuffer<float> d_inject(kStreams * kDim);
  HipBuffer<float> d_x(x.size()), d_lo(kDecode * kRank),
      d_mixed(kDecode * kDim), d_injected(kDecode * kStreams);
  const auto upload = [](auto& destination, const auto& source) {
    CheckHip(hipMemcpy(destination.get(), source.data(), destination.bytes(),
                       hipMemcpyHostToDevice),
             "upload");
  };
  upload(d_down_in, down.bytes);
  upload(d_up_in, up.bytes);
  upload(d_inject_in, inject.bytes);
  upload(d_x, x);
  // The decoder the uploader runs for the Mixer projections.
  q::DecodeWeights(d_down_in.get(), q::WeightType::kQ6_K, d_down.get(),
                   /*dst_half=*/true, kRank * kDim, nullptr);
  q::DecodeWeights(d_up_in.get(), q::WeightType::kIQ4_NL, d_up.get(),
                   /*dst_half=*/true, kDim * kRank, nullptr);
  q::DecodeWeights(d_inject_in.get(), q::WeightType::kQ6_K, d_inject.get(),
                   /*dst_half=*/false, kStreams * kDim, nullptr);
  CheckHip(hipDeviceSynchronize(), "decode");
  // The decoded values must equal the FP32 dequantization the reference uses.
  const auto check_row = [&](const fixture::Packed& w, auto* device,
                             std::size_t row, std::size_t cols) {
    std::vector<std::uint8_t> raw(cols * sizeof(*device));
    CheckHip(hipMemcpy(raw.data(), device + row * cols, raw.size(),
                       hipMemcpyDeviceToHost),
             "decoded row download");
    for (std::size_t i = 0; i < cols; ++i) {
      float actual = 0.0F;
      if constexpr (std::is_same_v<std::decay_t<decltype(*device)>, __half>) {
        actual = __half2float(reinterpret_cast<const __half*>(raw.data())[i]);
      } else {
        actual = reinterpret_cast<const float*>(raw.data())[i];
      }
      const float expected = w.values[row * cols + i];
      if (std::abs(actual - expected) >
          1e-3F * std::max(1.0F, std::abs(expected))) {
        throw std::runtime_error("decoded mixer weight differs at " +
                                 std::to_string(row) + "," + std::to_string(i));
      }
    }
  };
  check_row(down, d_down.get(), 7, kDim);
  check_row(up, d_up.get(), 11, kRank);
  check_row(inject, d_inject.get(), 2, kDim);

  // The executor's decode-time route: SmallGemm over the decoded weights.
  q::SmallGemm(d_down.get(), q::WeightType::kF16, d_x.get(), d_lo.get(),
               kDecode, static_cast<std::uint32_t>(kRank),
               static_cast<std::uint32_t>(kDim), nullptr);
  std::vector<float> lo = Download(&d_lo, kDecode * kRank);
  q::SmallGemm(d_up.get(), q::WeightType::kF16, d_lo.get(), d_mixed.get(),
               kDecode, static_cast<std::uint32_t>(kDim),
               static_cast<std::uint32_t>(kRank), nullptr);
  q::SmallGemm(d_inject.get(), q::WeightType::kF32, d_x.get(), d_injected.get(),
               kDecode, static_cast<std::uint32_t>(kStreams),
               static_cast<std::uint32_t>(kDim), nullptr);
  CheckHip(hipDeviceSynchronize(), "decoded mixer projections");

  // FP32 reference over the dequantized rows. The F16 projections round the
  // decoded weights to F16, which bounds their tolerance separately.
  const auto check = [&](const auto& actual, const std::string& name,
                         const fixture::Packed& w, const auto& source,
                         std::size_t rows, std::size_t cols, float tolerance) {
    double scale = 0.0;
    double worst = 0.0;
    for (std::size_t token = 0; token < kDecode; ++token) {
      for (std::size_t row = 0; row < rows; ++row) {
        double expected = 0.0;
        double magnitude = 0.0;
        for (std::size_t i = 0; i < cols; ++i) {
          const double product = static_cast<double>(w.values[row * cols + i]) *
                                 static_cast<double>(source[token * cols + i]);
          expected += product;
          magnitude += std::abs(product);
        }
        scale = std::max(scale, magnitude);
        worst = std::max(
            worst, std::abs(static_cast<double>(actual[token * rows + row]) -
                            expected));
      }
    }
    std::cerr << "decoded mixer " << name << ": worst |actual - F64| " << worst
              << " (scale " << scale << ")\n";
    if (!(worst < tolerance * std::max(scale, 1e-3))) {
      throw std::runtime_error("decoded mixer " + name +
                               " differs from the FP32 reference");
    }
  };
  check(lo, "down", down, x, kRank, kDim, 1e-3);
  check(Download(&d_mixed, kDecode * kDim), "up", up, lo, kDim, kRank, 1e-3);
  check(Download(&d_injected, kDecode * kStreams), "inject", inject, x,
        kStreams, kDim, 1e-4);
}

/// Mixed `ssm_alpha`/`ssm_beta` formats (spec: "Mixed SSM gate formats decode
/// correctly"): the uploader decodes each half through its own WeightType into
/// one F16 stack, so a Q8_0 half must not be read as F32. The stacked
/// projection is then compared against the FP32 reference.
void CheckMixedSsmGate() {
  constexpr std::size_t kRows = 8;  // one gate's rows
  constexpr std::uint32_t kGateTokens = 4;
  const auto alpha = fixture::MakeQ8Weights(kRows, kHidden, 0x51A0U);
  const auto beta = fixture::MakeQ6Weights(kRows, kHidden, 0x8BE7U);
  const auto x = MakeValues(kGateTokens * kHidden, 0x51A1U, 0.5F);

  HipBuffer<std::uint8_t> d_alpha(alpha.bytes.size());
  HipBuffer<std::uint8_t> d_beta(beta.bytes.size());
  HipBuffer<__half> d_stack(2 * kRows * kHidden);
  HipBuffer<float> d_x(x.size()), d_out(kGateTokens * 2 * kRows);
  CheckHip(hipMemcpy(d_alpha.get(), alpha.bytes.data(), d_alpha.bytes(),
                     hipMemcpyHostToDevice),
           "alpha upload");
  CheckHip(hipMemcpy(d_beta.get(), beta.bytes.data(), d_beta.bytes(),
                     hipMemcpyHostToDevice),
           "beta upload");
  Upload(&d_x, x);

  // The uploader's mixed-format path: decode each stored half into its own
  // row range of the F16 stack (StackAlphaBeta).
  q::DecodeWeights(d_alpha.get(), q::WeightType::kQ8_0, d_stack.get(),
                   /*dst_half=*/true, kRows * kHidden, nullptr);
  q::DecodeWeights(d_beta.get(), q::WeightType::kQ6_K,
                   d_stack.get() + kRows * kHidden, /*dst_half=*/true,
                   kRows * kHidden, nullptr);
  CheckHip(hipDeviceSynchronize(), "mixed gate decode");

  // The consumer: the decode-time F16 projection over the stacked rows.
  q::SmallGemm(d_stack.get(), q::WeightType::kF16, d_x.get(), d_out.get(),
               kGateTokens, static_cast<std::uint32_t>(2 * kRows), kHidden,
               nullptr);
  CheckHip(hipDeviceSynchronize(), "mixed gate projection");
  const auto out = Download(&d_out, kGateTokens * 2 * kRows);

  double worst = 0.0, scale = 0.0;
  for (std::uint32_t t = 0; t < kGateTokens; ++t) {
    for (std::size_t row = 0; row < 2 * kRows; ++row) {
      const auto& w = row < kRows ? alpha : beta;
      const std::size_t r = row < kRows ? row : row - kRows;
      double expected = 0.0, magnitude = 0.0;
      for (std::size_t i = 0; i < kHidden; ++i) {
        const double product = static_cast<double>(w.values[r * kHidden + i]) *
                               static_cast<double>(x[t * kHidden + i]);
        expected += product;
        magnitude += std::abs(product);
      }
      scale = std::max(scale, magnitude);
      worst = std::max(
          worst,
          std::abs(static_cast<double>(out[t * 2 * kRows + row]) - expected));
    }
  }
  std::cerr << "mixed ssm gate: worst |actual - F64| " << worst << " (scale "
            << scale << ")\n";
  if (!(worst < 1e-3 * std::max(scale, 1e-3))) {
    throw std::runtime_error(
        "mixed Q8_0/Q6_K ssm gate differs from the FP32 reference");
  }
}

/// The Swift-1.5 Mixer projections executed from their stored encodings (design
/// D2): `hc_*_down` is Q4_K over 10240 contracted columns -> 320 rows, and
/// `hc_*_up` is Q5_0 over 320 contracted columns -> 10240 rows, the only
/// admitted shape whose contracted dimension is not a multiple of 256. Both
/// take the vendored adapter's tiled entry and its decode GEMV over
/// pre-quantized activations, and the model's chain -- down projection, SiLU
/// scaled by 1/streams, up projection, Mixer epilogue -- is replayed over one
/// normed input with every stage compared against the F64 reference.
void CheckStoredMixerProjections(std::uint32_t tokens) {
  constexpr std::size_t kRank = 320;
  constexpr std::size_t kDim = kStreams * kHidden;
  const auto down = fixture::MakeQ4KWeights(kRank, kDim, 0x6D8C0A01U);
  const auto up = fixture::MakeQ5_0Weights(kDim, kRank, 0x6D8C0A02U);
  // The normed Mixer input is integer-scaled, so the Q8_1 quantization the
  // kernels apply is exact and the down projection measures their arithmetic
  // alone; the up projection consumes the SiLU-shaped chain row instead.
  const auto xn = fixture::MakeIntegerActivations(tokens, kDim, 0x6D8C0A03U);
  const auto inject_w = MakeValues(kStreams * kDim, 0x6D8C0A04U, 0.1F);
  const std::uint32_t parts = q::HcInjectPartsVec4(kHidden);
  const std::size_t lo_count = static_cast<std::size_t>(tokens) * kRank;
  const std::size_t gate_count = static_cast<std::size_t>(tokens) * kDim;
  const std::size_t mixed_count = static_cast<std::size_t>(tokens) * kHidden;
  const std::size_t inject_count =
      static_cast<std::size_t>(tokens) * kStreams * parts;

  HipBuffer<std::uint8_t> d_down(down.bytes.size());
  HipBuffer<std::uint8_t> d_up(up.bytes.size());
  HipBuffer<std::uint8_t> d_x_q8(
      qfn_mmq_q8_1_bytes(static_cast<int>(tokens), static_cast<int>(kDim)));
  HipBuffer<std::uint8_t> d_lo_q8(
      qfn_mmq_q8_1_bytes(static_cast<int>(tokens), static_cast<int>(kRank)));
  HipBuffer<float> d_x(xn.size()), d_lo(lo_count), d_gate(gate_count);
  HipBuffer<float> d_mixed(mixed_count), d_inject(inject_count);
  HipBuffer<float> d_inject_w(kStreams * kDim);
  CheckHip(hipMemcpy(d_down.get(), down.bytes.data(), d_down.bytes(),
                     hipMemcpyHostToDevice),
           "stored down weights");
  CheckHip(hipMemcpy(d_up.get(), up.bytes.data(), d_up.bytes(),
                     hipMemcpyHostToDevice),
           "stored up weights");
  Upload(&d_x, xn);
  Upload(&d_inject_w, inject_w);

  // Every adapter entry quantizes its activations to Q8_1 (the tiled entry
  // internally, the decode entry through `qfn_mmq_quantize_q8_1`), so over the
  // SiLU-shaped chain row the error is a random walk over the contracted
  // dimension rather than F16-rounded weights alone. That walk measures at
  // ~2e-3 of the summed |products| on these shapes, which the adapter-route
  // tolerance the operator tests enforce for these formats (5e-3) admits and
  // the decoded-F16 Mixer tolerance (1e-3) does not.
  constexpr double kTolerance = 5e-3;
  const auto check = [&](const std::vector<float>& actual,
                         const std::string& name, const fixture::Packed& w,
                         const std::vector<float>& source, std::size_t rows,
                         std::size_t cols) {
    double scale = 0.0;
    double worst = 0.0;
    for (std::uint32_t token = 0; token < tokens; ++token) {
      for (std::size_t row = 0; row < rows; ++row) {
        const double expected =
            fixture::ReferenceDot(w, source, row, token, cols, &scale);
        worst = std::max(
            worst, std::abs(static_cast<double>(actual[token * rows + row]) -
                            expected));
      }
    }
    std::cerr << "stored mixer " << name << ": worst |actual - F64| " << worst
              << " (scale " << scale << ")\n";
    if (!(worst < kTolerance * std::max(scale, 1e-3))) {
      throw std::runtime_error("stored mixer " + name +
                               " differs from the F64 reference");
    }
  };

  // Prefill route of the down projection over the normed Mixer input.
  if (qfn_mmq_q4_K_dense(d_down.get(), d_x.get(), d_lo.get(),
                         static_cast<int>(kRank), static_cast<int>(tokens),
                         static_cast<int>(kDim), nullptr) != 0) {
    throw std::runtime_error("hc down MMQ launch failed");
  }
  CheckHip(hipDeviceSynchronize(), "hc down MMQ");
  check(Download(&d_lo, lo_count), "down MMQ", down, xn, kRank, kDim);

  // The chain: the stored up projection consumes the SiLU-scaled down result,
  // and the epilogue mixes the streams through that projection's output.
  q::SiluScale(d_lo.get(), 1.0F / static_cast<float>(kStreams), lo_count,
               nullptr);
  CheckHip(hipDeviceSynchronize(), "hc silu");
  const auto lo = Download(&d_lo, lo_count);
  if (qfn_mmq_q5_0_dense(d_up.get(), d_lo.get(), d_gate.get(),
                         static_cast<int>(kDim), static_cast<int>(tokens),
                         static_cast<int>(kRank), nullptr) != 0) {
    throw std::runtime_error("hc up MMQ launch failed");
  }
  CheckHip(hipDeviceSynchronize(), "hc up MMQ");
  const auto gate = Download(&d_gate, gate_count);
  check(gate, "up MMQ", up, lo, kDim, kRank);
  q::HcMixEpilogueVec4(d_x.get(), d_gate.get(), d_inject_w.get(), d_mixed.get(),
                       d_inject.get(), tokens, kHidden, kStreams, nullptr);
  CheckHip(hipDeviceSynchronize(), "hc epilogue");

  // The epilogue's two outputs against the F64 reference over the same norm
  // and the up projection's output.
  {
    const auto mixed = Download(&d_mixed, mixed_count);
    double scale = 0.0;
    double worst = 0.0;
    for (std::uint32_t token = 0; token < tokens; ++token) {
      for (std::uint32_t i = 0; i < kHidden; ++i) {
        double expected = 0.0;
        double magnitude = 0.0;
        for (std::uint32_t s = 0; s < kStreams; ++s) {
          const std::size_t index =
              static_cast<std::size_t>(token) * kDim + s * kHidden + i;
          const double product =
              static_cast<double>(xn[index]) /
              (1.0 + std::exp(-static_cast<double>(gate[index])));
          expected += product;
          magnitude += std::abs(product);
        }
        scale = std::max(scale, magnitude);
        worst = std::max(
            worst,
            std::abs(static_cast<double>(
                         mixed[static_cast<std::size_t>(token) * kHidden + i]) -
                     expected / static_cast<double>(kStreams)));
      }
    }
    std::cerr << "stored mixer epilogue: worst |actual - F64| " << worst
              << " (scale " << scale << ")\n";
    if (!(worst < kTolerance * std::max(scale, 1e-3))) {
      throw std::runtime_error(
          "stored mixer epilogue mixed row differs from the F64 reference");
    }
  }
  {
    const auto inject = Download(&d_inject, inject_count);
    double scale = 0.0;
    double worst = 0.0;
    for (std::uint32_t token = 0; token < tokens; ++token) {
      for (std::uint32_t o = 0; o < kStreams; ++o) {
        double expected = 0.0;
        double magnitude = 0.0;
        for (std::uint32_t s = 0; s < kStreams; ++s) {
          for (std::uint32_t i = 0; i < kHidden; ++i) {
            const std::size_t index =
                static_cast<std::size_t>(token) * kDim + s * kHidden + i;
            const double product =
                static_cast<double>(inject_w[o * kDim + s * kHidden + i]) *
                static_cast<double>(xn[index]);
            expected += product;
            magnitude += std::abs(product);
          }
        }
        double partial = 0.0;
        for (std::uint32_t p = 0; p < parts; ++p) {
          partial +=
              inject[(static_cast<std::size_t>(token) * kStreams + o) * parts +
                     p];
        }
        scale = std::max(scale, magnitude);
        worst = std::max(worst, std::abs(partial - expected));
      }
    }
    std::cerr << "stored mixer inject: worst |actual - F64| " << worst
              << " (scale " << scale << ")\n";
    if (!(worst < kTolerance * std::max(scale, 1e-3))) {
      throw std::runtime_error(
          "stored mixer epilogue inject differs from the F64 reference");
    }
  }

  // Decode route over activations pre-quantized through the adapter's own
  // quantizer. The up projection runs first: the down projection's GEMV
  // overwrites the SiLU-scaled row it reads.
  if (qfn_mmq_quantize_q8_1(d_lo.get(), d_lo_q8.get(), static_cast<int>(tokens),
                            static_cast<int>(kRank), nullptr) != 0) {
    throw std::runtime_error("hc up GEMV quantization failed");
  }
  if (qfn_mmq_q5_0_dense_vec_preq(
          d_up.get(), d_lo_q8.get(), d_gate.get(), static_cast<int>(kDim),
          static_cast<int>(tokens), static_cast<int>(kRank), nullptr) != 0) {
    throw std::runtime_error("hc up GEMV launch failed");
  }
  CheckHip(hipDeviceSynchronize(), "hc up GEMV");
  check(Download(&d_gate, gate_count), "up GEMV", up, lo, kDim, kRank);

  if (qfn_mmq_quantize_q8_1(d_x.get(), d_x_q8.get(), static_cast<int>(tokens),
                            static_cast<int>(kDim), nullptr) != 0) {
    throw std::runtime_error("hc down GEMV quantization failed");
  }
  if (qfn_mmq_q4_K_dense_vec_preq(
          d_down.get(), d_x_q8.get(), d_lo.get(), static_cast<int>(kRank),
          static_cast<int>(tokens), static_cast<int>(kDim), nullptr) != 0) {
    throw std::runtime_error("hc down GEMV launch failed");
  }
  CheckHip(hipDeviceSynchronize(), "hc down GEMV");
  check(Download(&d_lo, lo_count), "down GEMV", down, xn, kRank, kDim);
  std::cerr << "stored Mixer projections match at " << tokens << " tokens\n";
}

/// One Mixer chain over the route the executor takes with the wide route off
/// (`HcMix` -> `Dense` per stored type): Q8_0 down through the W8A8 GEMM over
/// the tiled Q8 norm, the adapter formats through the vendored MMQ, the shared
/// SiLU scaling, then the up projection on the same dispatch and the four-lane
/// epilogue.
void CheckMixedBlock(gufo::core::GgmlType down_type,
                     gufo::core::GgmlType up_type, std::uint32_t tokens) {
  constexpr std::size_t kRank = 320;
  constexpr std::size_t kDim = kStreams * kHidden;
  const std::uint32_t seed = 0x9E3779B9U + tokens;
  const auto make = [](gufo::core::GgmlType type, std::size_t rows,
                       std::size_t k, std::uint32_t s) {
    switch (type) {
      case gufo::core::GgmlType::kQ8_0:
        return fixture::MakeQ8Weights(rows, k, s);
      case gufo::core::GgmlType::kQ4_K:
        return fixture::MakeQ4KWeights(rows, k, s);
      case gufo::core::GgmlType::kQ5_0:
        return fixture::MakeQ5_0Weights(rows, k, s);
      default:
        throw std::runtime_error("mixed model uses an unhandled Mixer format");
    }
  };
  const auto down = make(down_type, kRank, kDim, seed);
  const auto up = make(up_type, kDim, kRank, seed + 1);
  const auto xn = fixture::MakeIntegerActivations(tokens, kDim, seed + 2);
  const std::size_t lo_count = static_cast<std::size_t>(tokens) * kRank;
  const std::size_t gate_count = static_cast<std::size_t>(tokens) * kDim;
  const std::size_t mixed_count = static_cast<std::size_t>(tokens) * kHidden;

  HipBuffer<std::uint8_t> d_down(down.bytes.size());
  HipBuffer<std::uint8_t> d_up(up.bytes.size());
  HipBuffer<std::uint8_t> d_x_q8t(q::Q8TiledBytes(tokens, kDim));
  HipBuffer<__half> d_lo_half(lo_count);
  HipBuffer<float> d_x(xn.size()), d_lo(lo_count), d_gate(gate_count);
  HipBuffer<float> d_mixed(mixed_count);
  CheckHip(hipMemcpy(d_down.get(), down.bytes.data(), d_down.bytes(),
                     hipMemcpyHostToDevice),
           "mixed down weights");
  CheckHip(hipMemcpy(d_up.get(), up.bytes.data(), d_up.bytes(),
                     hipMemcpyHostToDevice),
           "mixed up weights");
  Upload(&d_x, xn);

  const auto adapter = [](gufo::core::GgmlType type, const void* w,
                          const float* x, float* out, std::size_t rows,
                          std::uint32_t n, std::size_t k) {
    int status = 0;
    if (type == gufo::core::GgmlType::kQ4_K) {
      status =
          qfn_mmq_q4_K_dense(w, x, out, static_cast<int>(rows),
                             static_cast<int>(n), static_cast<int>(k), nullptr);
    } else if (type == gufo::core::GgmlType::kQ5_0) {
      status =
          qfn_mmq_q5_0_dense(w, x, out, static_cast<int>(rows),
                             static_cast<int>(n), static_cast<int>(k), nullptr);
    } else {
      throw std::runtime_error("mixed model has no adapter entry for the type");
    }
    if (status != 0) {
      throw std::runtime_error("mixed model adapter launch failed");
    }
  };
  // Down projection.
  if (down_type == gufo::core::GgmlType::kQ8_0) {
    q::QuantizeQ8Tiled(d_x.get(), d_x_q8t.get(), tokens, kDim, nullptr);
    if (!q::W8A8Gemm(d_down.get(), d_x_q8t.get(), d_lo.get(), tokens, kRank,
                     kDim, nullptr)) {
      throw std::runtime_error("mixed model W8A8 down rejected");
    }
  } else {
    adapter(down_type, d_down.get(), d_x.get(), d_lo.get(), kRank, tokens,
            kDim);
  }
  CheckHip(hipDeviceSynchronize(), "mixed model down");
  const auto lo = Download(&d_lo, lo_count);

  q::SiluScale(d_lo.get(), 1.0F / static_cast<float>(kStreams), lo_count,
               nullptr);
  CheckHip(hipDeviceSynchronize(), "mixed model silu");
  const auto chain = Download(&d_lo, lo_count);
  // Up projection. Q8_0 takes the F16 route the executor's `DenseF16Route`
  // selects at this shape (10240 rows, 320 columns).
  if (up_type == gufo::core::GgmlType::kQ8_0) {
    q::NarrowActivations(d_lo.get(), d_lo_half.get(), false, lo_count, nullptr);
    if (!q::DenseF16Gemm(d_up.get(), d_lo_half.get(), d_gate.get(), tokens,
                         kDim, kRank, nullptr)) {
      throw std::runtime_error("mixed model F16 up rejected");
    }
  } else {
    adapter(up_type, d_up.get(), d_lo.get(), d_gate.get(), kDim, tokens, kRank);
  }
  CheckHip(hipDeviceSynchronize(), "mixed model up");
  const auto gate = Download(&d_gate, gate_count);

  q::HcMixEpilogueVec4(d_x.get(), d_gate.get(), nullptr, d_mixed.get(), nullptr,
                       tokens, kHidden, kStreams, nullptr);
  CheckHip(hipDeviceSynchronize(), "mixed model epilogue");
  const auto mixed = Download(&d_mixed, mixed_count);

  // Every Mixer output against the F64 reference of its own stored encoding.
  // The adapter routes quantize their activations to Q8_1 exactly (the chain
  // input is integer-scaled), so their tolerance is the operator tests' 5e-3;
  // a Q8_0 projection over these weights is exact in F16 and only rounds its
  // F16 activation input.
  const auto check = [&](const std::vector<float>& actual,
                         const std::string& name, const fixture::Packed& w,
                         const std::vector<float>& source, std::size_t rows,
                         std::size_t cols, double tolerance) {
    double scale = 0.0;
    double worst = 0.0;
    for (std::uint32_t token = 0; token < tokens; ++token) {
      for (std::size_t row = 0; row < rows; ++row) {
        const double expected =
            fixture::ReferenceDot(w, source, row, token, cols, &scale);
        worst = std::max(
            worst, std::abs(static_cast<double>(actual[token * rows + row]) -
                            expected));
      }
    }
    if (!(worst < tolerance * std::max(scale, 1e-3))) {
      throw std::runtime_error("mixed model " + name +
                               " differs from the F64 "
                               "reference (worst " +
                               std::to_string(worst) + ", scale " +
                               std::to_string(scale) + ")");
    }
  };
  const double down_tolerance =
      down_type == gufo::core::GgmlType::kQ8_0 ? 1e-3 : 5e-3;
  const double up_tolerance =
      up_type == gufo::core::GgmlType::kQ8_0 ? 1e-3 : 5e-3;
  const auto down_name = std::string(gufo::core::ToString(down_type));
  const auto up_name = std::string(gufo::core::ToString(up_type));
  check(lo, "down " + down_name, down, xn, kRank, kDim, down_tolerance);
  check(gate, "up " + up_name, up, chain, kDim, kRank, up_tolerance);
  {
    double scale = 0.0;
    double worst = 0.0;
    for (std::uint32_t token = 0; token < tokens; ++token) {
      for (std::uint32_t i = 0; i < kHidden; ++i) {
        double expected = 0.0;
        double magnitude = 0.0;
        for (std::uint32_t s = 0; s < kStreams; ++s) {
          const std::size_t index =
              static_cast<std::size_t>(token) * kDim + s * kHidden + i;
          const double product =
              static_cast<double>(xn[index]) /
              (1.0 + std::exp(-static_cast<double>(gate[index])));
          expected += product;
          magnitude += std::abs(product);
        }
        scale = std::max(scale, magnitude);
        worst = std::max(
            worst,
            std::abs(static_cast<double>(
                         mixed[static_cast<std::size_t>(token) * kHidden + i]) -
                     expected / static_cast<double>(kStreams)));
      }
    }
    if (!(worst < 1e-3 * std::max(scale, 1e-3))) {
      throw std::runtime_error(
          "mixed model mixed row differs from the F64 "
          "reference");
    }
  }
  std::cerr << "mixed model " << down_name << "/" << up_name << " matches at "
            << tokens << " tokens\n";
}

/// The mixed-format model (spec: "Mixer down formats mixed across blocks
/// decode correctly"): a model whose Mixer down projections differ across
/// blocks must not put a non-Q8_0 encoding into the wide route's Q8_0 kernels.
/// With the wide route off, `HcMix` dispatches every projection on its own
/// stored type, so this case runs the three block pairs the design names —
/// Q8_0/Q8_0, Q4_K/Q8_0 and Q4_K/Q5_0 — and checks every Mixer output of each
/// against the F64 reference of its own encoding.
void CheckMixedFormatModel(std::uint32_t tokens) {
  CheckMixedBlock(gufo::core::GgmlType::kQ8_0, gufo::core::GgmlType::kQ8_0,
                  tokens);
  CheckMixedBlock(gufo::core::GgmlType::kQ4_K, gufo::core::GgmlType::kQ8_0,
                  tokens);
  CheckMixedBlock(gufo::core::GgmlType::kQ4_K, gufo::core::GgmlType::kQ5_0,
                  tokens);
}

}  // namespace

int main() {
  try {
    CheckFusedProjection(96);
    CheckFusedProjection(2049);
    CheckDecodedMixer();
    CheckMixedSsmGate();
    CheckStoredMixerProjections(1);
    CheckStoredMixerProjections(8);
    CheckMixedFormatModel(64);
    CheckMixedFormatModel(128);
    constexpr std::size_t kHcDim = static_cast<std::size_t>(kHidden) * kStreams;
    constexpr std::size_t kRows = static_cast<std::size_t>(kTokens) * kHcDim;
    constexpr std::size_t kMixed = static_cast<std::size_t>(kTokens) * kHidden;
    const std::uint32_t scalar_parts = q::HcInjectParts(kHidden);
    const std::uint32_t vec_parts = q::HcInjectPartsVec4(kHidden);
    const std::size_t scalar_inject =
        static_cast<std::size_t>(kTokens) * kStreams * scalar_parts;
    const std::size_t vec_inject =
        static_cast<std::size_t>(kTokens) * kStreams * vec_parts;

    const auto xn = MakeValues(kRows, 0x1234ABCDU, 0.5F);
    const auto gate = MakeValues(kRows, 0xBADC0FFEU, 2.0F);
    const auto inject_w = MakeValues(kStreams * kHcDim, 0xDEADBEEFU, 0.1F);
    HipBuffer<float> d_xn(kRows);
    HipBuffer<float> d_gate(kRows);
    HipBuffer<float> d_weight(kStreams * kHcDim);
    HipBuffer<float> d_mixed_scalar(kMixed);
    HipBuffer<float> d_mixed_vec(kMixed);
    HipBuffer<float> d_inject_scalar(scalar_inject);
    HipBuffer<float> d_inject_vec(vec_inject);
    Upload(&d_xn, xn);
    Upload(&d_gate, gate);
    Upload(&d_weight, inject_w);

    q::HcMixEpilogue(d_xn.get(), d_gate.get(), d_weight.get(),
                     d_mixed_scalar.get(), d_inject_scalar.get(), kTokens,
                     kHidden, kStreams, nullptr);
    q::HcMixEpilogueVec4(d_xn.get(), d_gate.get(), d_weight.get(),
                         d_mixed_vec.get(), d_inject_vec.get(), kTokens,
                         kHidden, kStreams, nullptr);
    CheckHip(hipDeviceSynchronize(), "HC mix synchronization");

    const auto mixed_scalar = Download(&d_mixed_scalar, kMixed);
    const auto mixed_vec = Download(&d_mixed_vec, kMixed);
    const auto inject_scalar = Download(&d_inject_scalar, scalar_inject);
    const auto inject_vec = Download(&d_inject_vec, vec_inject);
    double worst_mixed = 0.0;
    for (std::size_t i = 0; i < kMixed; ++i) {
      worst_mixed = std::max(
          worst_mixed,
          std::abs(static_cast<double>(mixed_scalar[i] - mixed_vec[i])));
    }
    double worst_inject = 0.0;
    for (std::uint32_t t = 0; t < kTokens; ++t) {
      for (std::uint32_t s = 0; s < kStreams; ++s) {
        double scalar = 0.0;
        double vectorized = 0.0;
        const std::size_t scalar_base =
            (static_cast<std::size_t>(t) * kStreams + s) * scalar_parts;
        const std::size_t vec_base =
            (static_cast<std::size_t>(t) * kStreams + s) * vec_parts;
        for (std::uint32_t p = 0; p < scalar_parts; ++p) {
          scalar += inject_scalar[scalar_base + p];
        }
        for (std::uint32_t p = 0; p < vec_parts; ++p) {
          vectorized += inject_vec[vec_base + p];
        }
        const double scale = std::max(1.0, std::abs(scalar));
        worst_inject =
            std::max(worst_inject, std::abs(scalar - vectorized) / scale);
      }
    }
    std::cerr << "HC mix vec4 worst mixed absolute error " << worst_mixed
              << ", inject relative error " << worst_inject << std::endl;
    if (worst_mixed > 1e-6 || worst_inject > 5e-5) {
      return 1;
    }

    // F16 mixer input route: the combine writes the next mixer's norm as
    // F16 and the epilogue reads it back. The residual update must stay
    // bit-identical; the norm and the mixed output carry F16 rounding.
    constexpr float kEps = 1e-6F;
    const auto res = MakeValues(kRows, 0x0F0F0F0FU, 1.0F);
    const auto block_out = MakeValues(kMixed, 0xF0F0F0F0U, 1.0F);
    const auto gamma = MakeValues(kHcDim, 0x600DCAFEU, 0.5F);
    HipBuffer<float> d_res_ref(kRows);
    HipBuffer<float> d_res_f16(kRows);
    HipBuffer<float> d_block_out(kMixed);
    HipBuffer<float> d_gamma(kHcDim);
    HipBuffer<float> d_xn_ref(kRows);
    HipBuffer<__half> d_xn_f16(kRows);
    HipBuffer<float> d_inject_ref(vec_inject);
    Upload(&d_res_ref, res);
    Upload(&d_res_f16, res);
    Upload(&d_block_out, block_out);
    Upload(&d_gamma, gamma);
    q::HcCombine(d_res_ref.get(), d_block_out.get(), d_inject_vec.get(),
                 vec_parts, d_gamma.get(), d_xn_ref.get(), kTokens, kHidden,
                 kStreams, kEps, nullptr);
    const std::size_t q8_bytes = q::Q8TiledBytes(kTokens, kHcDim);
    HipBuffer<std::uint8_t> d_xn_q8(q8_bytes);
    q::HcCombineF16(d_res_f16.get(), d_block_out.get(), d_inject_vec.get(),
                    vec_parts, d_gamma.get(), d_xn_f16.get(), d_xn_q8.get(),
                    kTokens, kHidden, kStreams, kEps, nullptr);
    q::HcMixEpilogueVec4(d_xn_ref.get(), d_gate.get(), d_weight.get(),
                         d_mixed_scalar.get(), d_inject_ref.get(), kTokens,
                         kHidden, kStreams, nullptr);
    q::HcMixEpilogueVec4F16(d_xn_f16.get(), d_gate.get(), d_weight.get(),
                            d_mixed_vec.get(), nullptr, nullptr,
                            d_inject_vec.get(), kTokens, kHidden, nullptr);
    CheckHip(hipDeviceSynchronize(), "HC F16 synchronization");
    const auto res_ref = Download(&d_res_ref, kRows);
    const auto res_f16 = Download(&d_res_f16, kRows);
    const auto xn_ref = Download(&d_xn_ref, kRows);
    // CPU reference of the combine (residual update and grouped norm) over
    // the same inject partials.
    {
      double worst_cpu_res = 0.0;
      double worst_cpu_xn = 0.0;
      for (std::uint32_t t = 0; t < kTokens; ++t) {
        for (std::uint32_t s = 0; s < kStreams; ++s) {
          double logit = 0.0;
          for (std::uint32_t p = 0; p < vec_parts; ++p) {
            logit += inject_vec[(static_cast<std::size_t>(t) * kStreams + s) *
                                    vec_parts +
                                p];
          }
          const double w =
              2.0 / (1.0 + std::exp(-logit / static_cast<double>(kStreams)));
          const std::size_t base =
              (static_cast<std::size_t>(t) * kStreams + s) * kHidden;
          std::vector<double> v(kHidden);
          double ss = 0.0;
          for (std::uint32_t i = 0; i < kHidden; ++i) {
            v[i] = res[base + i] +
                   block_out[static_cast<std::size_t>(t) * kHidden + i] * w;
            ss += v[i] * v[i];
          }
          const double scale = 1.0 / std::sqrt(ss / kHidden + kEps);
          for (std::uint32_t i = 0; i < kHidden; ++i) {
            worst_cpu_res =
                std::max(worst_cpu_res, std::abs(v[i] - res_ref[base + i]));
            const double n = v[i] * scale * gamma[s * kHidden + i];
            worst_cpu_xn =
                std::max(worst_cpu_xn, std::abs(n - xn_ref[base + i]) /
                                           std::max(1e-2, std::abs(n)));
          }
        }
      }
      std::cerr << "HC combine vs CPU: residual " << worst_cpu_res << ", norm "
                << worst_cpu_xn << std::endl;
      if (worst_cpu_res > 1e-4 || worst_cpu_xn > 1e-3) {
        return 1;
      }
    }
    std::vector<__half> xn_f16(kRows);
    CheckHip(hipMemcpy(xn_f16.data(), d_xn_f16.get(), kRows * sizeof(__half),
                       hipMemcpyDeviceToHost),
             "download");
    const auto mixed_ref = Download(&d_mixed_scalar, kMixed);
    const auto mixed_f16 = Download(&d_mixed_vec, kMixed);
    const auto inject_ref = Download(&d_inject_ref, vec_inject);
    const auto inject_f16 = Download(&d_inject_vec, vec_inject);
    double worst_res = 0.0;
    double worst_xn = 0.0;
    for (std::size_t i = 0; i < kRows; ++i) {
      worst_res = std::max(
          worst_res, std::abs(static_cast<double>(res_ref[i] - res_f16[i])));
      const double scale =
          std::max(1e-2, std::abs(static_cast<double>(xn_ref[i])));
      worst_xn = std::max(
          worst_xn,
          std::abs(static_cast<double>(xn_ref[i] - __half2float(xn_f16[i]))) /
              scale);
    }
    // The mix sums four streams of either sign, so it is scored absolutely
    // on values of order one.
    double worst_mixed_f16 = 0.0;
    for (std::size_t i = 0; i < kMixed; ++i) {
      worst_mixed_f16 =
          std::max(worst_mixed_f16,
                   std::abs(static_cast<double>(mixed_ref[i] - mixed_f16[i])));
    }
    double worst_inject_f16 = 0.0;
    for (std::uint32_t t = 0; t < kTokens; ++t) {
      for (std::uint32_t s = 0; s < kStreams; ++s) {
        double reference = 0.0;
        double candidate = 0.0;
        const std::size_t base =
            (static_cast<std::size_t>(t) * kStreams + s) * vec_parts;
        for (std::uint32_t p = 0; p < vec_parts; ++p) {
          reference += inject_ref[base + p];
          candidate += inject_f16[base + p];
        }
        const double scale = std::max(1.0, std::abs(reference));
        worst_inject_f16 =
            std::max(worst_inject_f16, std::abs(reference - candidate) / scale);
      }
    }
    // The fused tiled-Q8 norm: dequantize each 32-block back and compare
    // with the F32 norm (8-bit rounding of the block's absmax).
    std::vector<std::uint8_t> xn_q8(q8_bytes);
    CheckHip(
        hipMemcpy(xn_q8.data(), d_xn_q8.get(), q8_bytes, hipMemcpyDeviceToHost),
        "download");
    double worst_q8 = 0.0;
    for (std::uint32_t t = 0; t < kTokens; ++t) {
      for (std::size_t kb = 0; kb < kHcDim / 32; ++kb) {
        const std::uint8_t* tile =
            xn_q8.data() + (((t / 16) * (kHcDim / 32)) + kb) * 576;
        float d = 0.0F;
        std::memcpy(&d, tile + 512 + (t % 16) * 4, 4);
        for (std::size_t j = 0; j < 32; ++j) {
          const auto code = static_cast<std::int8_t>(
              tile[((j / 16) * 256) + ((t % 16) * 16) + (j % 16)]);
          const double ref =
              xn_ref[static_cast<std::size_t>(t) * kHcDim + kb * 32 + j];
          const double err = std::abs(ref - d * static_cast<double>(code));
          // One rounding step of the block's absmax / 127 grid.
          worst_q8 = std::max(
              worst_q8, err / std::max(1e-3, static_cast<double>(std::abs(d))));
        }
      }
    }
    std::cout << "HC F16 mixer input: residual absolute error " << worst_res
              << ", tiled Q8 norm worst error " << worst_q8
              << " quantization steps\n";
    std::cout << "HC F16 mixer input: residual absolute error " << worst_res
              << ", norm relative error " << worst_xn
              << ", mixed absolute error " << worst_mixed_f16
              << ", inject relative error " << worst_inject_f16 << '\n';
    if (worst_res != 0.0 || worst_xn > 2e-3 || worst_mixed_f16 > 1e-3 ||
        worst_inject_f16 > 4e-3 || worst_q8 > 0.51) {
      return 1;
    }

    // Side outputs are independent: Q8 must also work without an F16
    // output or inject weights, including a one-token batch.
    for (const auto count : {1U, kTokens}) {
      const std::size_t mixed_count = static_cast<std::size_t>(count) * kHidden;
      const auto bytes = q::Q8TiledBytes(count, kHidden);
      HipBuffer<__half> d_half(mixed_count);
      HipBuffer<std::uint8_t> d_q8(bytes);
      for (unsigned outputs = 0; outputs < 8; ++outputs) {
        const bool half_output = (outputs & 1U) != 0;
        const bool q8_output = (outputs & 2U) != 0;
        const bool inject_output = (outputs & 4U) != 0;
        CheckHip(hipMemset(d_q8.get(), 0xA5, bytes), "poison Q8 output");
        q::HcMixEpilogueVec4F16(d_xn_f16.get(), d_gate.get(),
                                inject_output ? d_weight.get() : nullptr,
                                d_mixed_vec.get(),
                                half_output ? d_half.get() : nullptr,
                                q8_output ? d_q8.get() : nullptr,
                                inject_output ? d_inject_vec.get() : nullptr,
                                count, kHidden, nullptr);
        const auto actual = Download(&d_mixed_vec, kMixed);
        if (std::memcmp(actual.data(), mixed_f16.data(),
                        mixed_count * sizeof(float)) != 0) {
          throw std::runtime_error("optional outputs changed the mixed row");
        }
        if (inject_output) {
          const auto actual_inject = Download(&d_inject_vec, vec_inject);
          if (std::memcmp(actual_inject.data(), inject_f16.data(),
                          static_cast<std::size_t>(count) * kStreams *
                              vec_parts * sizeof(float)) != 0) {
            throw std::runtime_error("mixer replay changed inject partials");
          }
        }
        if (half_output) {
          std::vector<__half> half(mixed_count);
          CheckHip(hipMemcpy(half.data(), d_half.get(), d_half.bytes(),
                             hipMemcpyDeviceToHost),
                   "download mixed F16");
          for (std::size_t i = 0; i < mixed_count; ++i) {
            if (__half2float(half[i]) !=
                __half2float(__float2half(mixed_f16[i]))) {
              throw std::runtime_error("mixed F16 rounding differs");
            }
          }
        }
        if (q8_output) {
          std::vector<std::uint8_t> packed(bytes);
          CheckHip(hipMemcpy(packed.data(), d_q8.get(), bytes,
                             hipMemcpyDeviceToHost),
                   "download mixed Q8");
          for (std::uint32_t t = 0; t < count; ++t) {
            for (std::uint32_t kb = 0; kb < kHidden / 32; ++kb) {
              const auto* tile =
                  packed.data() + ((t / 16) * (kHidden / 32) + kb) * 576;
              float scale;
              std::memcpy(&scale, tile + 512 + (t % 16) * 4, 4);
              if (!std::isfinite(scale) || scale < 0.0F) {
                throw std::runtime_error("invalid mixed Q8 scale");
              }
              for (std::uint32_t j = 0; j < 32; ++j) {
                const auto code = static_cast<std::int8_t>(
                    tile[(j / 16) * 256 + (t % 16) * 16 + j % 16]);
                const float expected =
                    mixed_f16[static_cast<std::size_t>(t) * kHidden + kb * 32 +
                              j];
                if (std::abs(expected - scale * code) >
                    0.501F * scale + 1e-7F) {
                  throw std::runtime_error("mixed Q8 rounding differs");
                }
              }
            }
          }
        }
      }
    }
    for (const std::uint32_t chunk : {1U, 8U, 9U, 16U}) {
      Upload(&d_res_f16, res);
      Upload(&d_inject_vec, inject_vec);
      for (std::uint32_t off = 0; off < kTokens; off += chunk) {
        q::HcCombineF16(
            d_res_f16.get() + std::size_t(off) * kHcDim,
            d_block_out.get() + std::size_t(off) * kHidden,
            d_inject_vec.get() + std::size_t(off) * kStreams * vec_parts,
            vec_parts, d_gamma.get(),
            d_xn_f16.get() + std::size_t(off) * kHcDim, nullptr,
            std::min(chunk, kTokens - off), kHidden, kStreams, kEps, nullptr);
      }
      CheckHip(hipDeviceSynchronize(), "short HC prefill");
      const auto actual_res = Download(&d_res_f16, kRows);
      std::vector<__half> actual_norm(kRows);
      CheckHip(hipMemcpy(actual_norm.data(), d_xn_f16.get(),
                         kRows * sizeof(__half), hipMemcpyDeviceToHost),
               "short HC norm");
      if (actual_res != res_f16 ||
          std::memcmp(actual_norm.data(), xn_f16.data(),
                      kRows * sizeof(__half)) != 0)
        throw std::runtime_error("HC prefill depends on chunk width");
    }
    std::cout << "HC mixed outputs and short prefill chunks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
