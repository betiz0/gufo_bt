// Adapter-route checks for the Q6_K and IQ4_NL entries: the prefill MMQ dense
// projections, their decode GEMV counterparts over pre-quantized Q8_1
// activations, and the IQ4_NL routed down projection on the vector route.
// Every result is compared against an FP32 reference built from the CPU
// dequantization of the same packed blocks.
//
// Activations are integers times a power-of-two scale with a 127 magnitude in
// every 32-wide block, so the Q8_1 quantization the kernels apply is exact and
// the reference measures the kernels themselves, not quantizer tie-breaking.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"
#include "tests/models/qwen38_flash_next/quant_fixture.hpp"

namespace {

int DenseDispatch(int type, const void* w, const float* x, float* out,
                  std::size_t tokens, std::size_t rows, std::size_t k);
int DenseVecDispatch(int type, const void* w, const void* x_q8, float* out,
                     std::size_t tokens, std::size_t rows, std::size_t k);

using Packed = gufo::models::qwen38_flash_next::fixture::Packed;
using gufo::models::qwen38_flash_next::fixture::DequantizeRow;
using gufo::models::qwen38_flash_next::fixture::MakeIntegerActivations;
using gufo::models::qwen38_flash_next::fixture::MakeIq4NlWeights;
using gufo::models::qwen38_flash_next::fixture::MakeQ4KWeights;
using gufo::models::qwen38_flash_next::fixture::MakeQ5_0Weights;
using gufo::models::qwen38_flash_next::fixture::MakeQ5KWeights;
using gufo::models::qwen38_flash_next::fixture::MakeQ6Weights;
using gufo::models::qwen38_flash_next::fixture::NextRandom;
using gufo::models::qwen38_flash_next::fixture::ReferenceDot;
using GgmlType = gufo::core::GgmlType;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

void CheckRow(const std::string& what, double actual, double expected,
              double scale) {
  if (!std::isfinite(actual)) {
    throw std::runtime_error(what + ": output is not finite");
  }
  if (std::abs(actual - expected) > 5e-3 * std::max(scale, 1e-3)) {
    throw std::runtime_error(what + ": |" + std::to_string(actual) + " - " +
                             std::to_string(expected) +
                             "| exceeds the tolerance (scale " +
                             std::to_string(scale) + ")");
  }
}

/// IQ4_NL routed down projection on the adapter's vector route (single-token
/// and narrow-batch decode).
void CheckIq4NlRoutedVector() {
  const std::size_t rows = 2560, k = 640, experts = 4, tokens = 3, used = 1;
  const Packed w = MakeIq4NlWeights(rows * experts, k, 0x1234U);
  const auto x = MakeIntegerActivations(tokens, k, 0x9876U);
  std::vector<std::int32_t> ids(tokens * used);
  for (std::size_t t = 0; t < tokens; ++t) {
    ids[t] = static_cast<std::int32_t>(t % experts);
  }
  void* d_w = nullptr;
  float* d_x = nullptr;
  float* d_out = nullptr;
  std::int32_t* d_ids = nullptr;
  CheckHip(hipMalloc(&d_w, w.bytes.size()), "weights");
  CheckHip(hipMalloc(&d_x, x.size() * sizeof(float)), "input");
  CheckHip(hipMalloc(&d_out, tokens * used * rows * sizeof(float)), "output");
  CheckHip(hipMalloc(&d_ids, ids.size() * sizeof(std::int32_t)), "ids");
  CheckHip(
      hipMemcpy(d_w, w.bytes.data(), w.bytes.size(), hipMemcpyHostToDevice),
      "weights upload");
  CheckHip(
      hipMemcpy(d_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice),
      "input upload");
  CheckHip(hipMemcpy(d_ids, ids.data(), ids.size() * sizeof(std::int32_t),
                     hipMemcpyHostToDevice),
           "ids upload");
  if (qfn_mmq_moe_vec(20 /* GGML_TYPE_IQ4_NL */, d_w, d_x, d_ids, d_out,
                      static_cast<int>(rows), static_cast<int>(k),
                      static_cast<int>(tokens), static_cast<int>(experts),
                      static_cast<int>(used), nullptr) != 0) {
    throw std::runtime_error("IQ4_NL routed vector projection failed");
  }
  CheckHip(hipDeviceSynchronize(), "routed vector synchronize");
  std::vector<float> out(tokens * used * rows);
  CheckHip(hipMemcpy(out.data(), d_out, out.size() * sizeof(float),
                     hipMemcpyDeviceToHost),
           "output download");
  double scale = 0.0;
  for (std::size_t t = 0; t < tokens; ++t) {
    const std::size_t expert = static_cast<std::size_t>(ids[t]);
    for (std::size_t r = 0; r < rows; ++r) {
      const double expected =
          ReferenceDot(w, x, expert * rows + r, t, k, &scale);
      CheckRow("IQ4_NL routed row " + std::to_string(r), out[t * rows + r],
               expected, scale);
    }
  }
  for (void* p : {d_w, static_cast<void*>(d_x), static_cast<void*>(d_out),
                  static_cast<void*>(d_ids)}) {
    CheckHip(hipFree(p), "free");
  }
  std::cout << "IQ4_NL routed vector: " << rows << "x" << k << " x " << tokens
            << " tokens match the reference\n";
}

/// Q5_0 routed down projection on the adapter's vector route (single-token
/// and narrow-batch decode), the Swift-1.5 down-expert format.
void CheckQ5_0RoutedVector() {
  const std::size_t rows = 2560, k = 640, experts = 4, tokens = 3, used = 1;
  const Packed w = MakeQ5_0Weights(rows * experts, k, 0x0F0FU);
  const auto x = MakeIntegerActivations(tokens, k, 0x9876U);
  std::vector<std::int32_t> ids(tokens * used);
  for (std::size_t t = 0; t < tokens; ++t) {
    ids[t] = static_cast<std::int32_t>(t % experts);
  }
  void* d_w = nullptr;
  float* d_x = nullptr;
  float* d_out = nullptr;
  std::int32_t* d_ids = nullptr;
  CheckHip(hipMalloc(&d_w, w.bytes.size()), "weights");
  CheckHip(hipMalloc(&d_x, x.size() * sizeof(float)), "input");
  CheckHip(hipMalloc(&d_out, tokens * used * rows * sizeof(float)), "output");
  CheckHip(hipMalloc(&d_ids, ids.size() * sizeof(std::int32_t)), "ids");
  CheckHip(
      hipMemcpy(d_w, w.bytes.data(), w.bytes.size(), hipMemcpyHostToDevice),
      "weights upload");
  CheckHip(
      hipMemcpy(d_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice),
      "input upload");
  CheckHip(hipMemcpy(d_ids, ids.data(), ids.size() * sizeof(std::int32_t),
                     hipMemcpyHostToDevice),
           "ids upload");
  if (qfn_mmq_moe_vec(6 /* GGML_TYPE_Q5_0 */, d_w, d_x, d_ids, d_out,
                      static_cast<int>(rows), static_cast<int>(k),
                      static_cast<int>(tokens), static_cast<int>(experts),
                      static_cast<int>(used), nullptr) != 0) {
    throw std::runtime_error("Q5_0 routed vector projection failed");
  }
  CheckHip(hipDeviceSynchronize(), "routed vector synchronize");
  std::vector<float> out(tokens * used * rows);
  CheckHip(hipMemcpy(out.data(), d_out, out.size() * sizeof(float),
                     hipMemcpyDeviceToHost),
           "output download");
  double scale = 0.0;
  for (std::size_t t = 0; t < tokens; ++t) {
    const std::size_t expert = static_cast<std::size_t>(ids[t]);
    for (std::size_t r = 0; r < rows; ++r) {
      const double expected =
          ReferenceDot(w, x, expert * rows + r, t, k, &scale);
      CheckRow("Q5_0 routed row " + std::to_string(r), out[t * rows + r],
               expected, scale);
    }
  }
  for (void* p : {d_w, static_cast<void*>(d_x), static_cast<void*>(d_out),
                  static_cast<void*>(d_ids)}) {
    CheckHip(hipFree(p), "free");
  }
  std::cout << "Q5_0 routed vector: " << rows << "x" << k << " x " << tokens
            << " tokens match the reference\n";
}

/// Dense prefill MMQ and decode GEMV for one format. `ggml_type` selects the
/// row-dequantization used when the reference is sampled; `reference_rows`
/// limits the checked rows (0 checks every row), so the vocabulary-sized
/// shapes sample a spread instead of materializing the whole FP32 matrix.
void CheckDenseRoutes(const std::string& name, int type, GgmlType ggml_type,
                      const Packed& w, std::size_t rows, std::size_t k,
                      std::size_t reference_rows = 0) {
  const std::size_t wide_tokens = 8;
  const auto x_wide = MakeIntegerActivations(wide_tokens, k, 0x2468U);
  const auto x_one = MakeIntegerActivations(1, k, 0x1357U);
  const bool keep_all = reference_rows == 0 || reference_rows >= rows;
  std::vector<bool> keep(rows, true);
  if (!keep_all) {
    std::fill(keep.begin(), keep.end(), false);
    for (std::size_t s = 0; s < reference_rows; ++s) {
      keep[reference_rows > 1 ? s * (rows - 1) / (reference_rows - 1) : 0] =
          true;
    }
  }

  void* d_w = nullptr;
  float* d_x = nullptr;
  float* d_out = nullptr;
  void* d_q8 = nullptr;
  CheckHip(hipMalloc(&d_w, w.bytes.size()), "weights");
  CheckHip(hipMalloc(&d_x, x_wide.size() * sizeof(float)), "input");
  CheckHip(hipMalloc(&d_out, wide_tokens * rows * sizeof(float)), "output");
  const std::size_t q8_bytes = qfn_mmq_q8_1_bytes(4, static_cast<int>(k));
  CheckHip(hipMalloc(&d_q8, q8_bytes), "q8 input");
  CheckHip(
      hipMemcpy(d_w, w.bytes.data(), w.bytes.size(), hipMemcpyHostToDevice),
      "weights upload");
  CheckHip(hipMemcpy(d_x, x_wide.data(), x_wide.size() * sizeof(float),
                     hipMemcpyHostToDevice),
           "input upload");

  double scale = 0.0;
  std::vector<float> out(wide_tokens * rows);

  // Prefill MMQ over F32 activations, for a full tile and for the single-token
  // catch-up case the executor also sends through the tiled entry.
  for (const std::size_t tokens : {wide_tokens, std::size_t{1}}) {
    float* dst = d_out;
    const std::string label = name + " MMQ N=" + std::to_string(tokens);
    if (DenseDispatch(type, d_w, d_x, dst, tokens, rows, k) != 0) {
      throw std::runtime_error(label + " failed");
    }
    CheckHip(hipDeviceSynchronize(), "MMQ synchronize");
    CheckHip(hipMemcpy(out.data(), dst, tokens * rows * sizeof(float),
                       hipMemcpyDeviceToHost),
             "output download");
    for (std::size_t t = 0; t < tokens; ++t) {
      for (std::size_t r = 0; r < rows; ++r) {
        if (!keep[r]) {
          continue;
        }
        const std::vector<double> values =
            keep_all ? std::vector<double>()
                     : DequantizeRow(w, r, k, ggml_type);
        double expected = 0.0;
        double magnitude = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
          const double weight =
              keep_all ? static_cast<double>(w.values[r * k + i]) : values[i];
          const double product =
              weight * static_cast<double>(x_wide[t * k + i]);
          expected += product;
          magnitude += std::abs(product);
        }
        scale = std::max(scale, magnitude);
        CheckRow(label + " row " + std::to_string(r), out[t * rows + r],
                 expected, scale);
      }
    }
  }

  // Decode GEMV over pre-quantized Q8_1 activations, one and four tokens.
  for (const std::size_t tokens : {std::size_t{1}, std::size_t{4}}) {
    const std::string label = name + " GEMV N=" + std::to_string(tokens);
    std::vector<float> x_gemv;
    for (std::size_t t = 0; t < tokens; ++t) {
      x_gemv.insert(x_gemv.end(), x_one.begin(), x_one.end());
    }
    CheckHip(hipMemcpy(d_x, x_gemv.data(), x_gemv.size() * sizeof(float),
                       hipMemcpyHostToDevice),
             "input upload");
    if (qfn_mmq_quantize_q8_1(d_x, d_q8, static_cast<int>(tokens),
                              static_cast<int>(k), nullptr) != 0) {
      throw std::runtime_error(label + " activation quantization failed");
    }
    if (DenseVecDispatch(type, d_w, d_q8, d_out, tokens, rows, k) != 0) {
      throw std::runtime_error(label + " failed");
    }
    CheckHip(hipDeviceSynchronize(), "GEMV synchronize");
    CheckHip(hipMemcpy(out.data(), d_out, tokens * rows * sizeof(float),
                       hipMemcpyDeviceToHost),
             "output download");
    for (std::size_t t = 0; t < tokens; ++t) {
      for (std::size_t r = 0; r < rows; ++r) {
        if (!keep[r]) {
          continue;
        }
        const std::vector<double> values =
            keep_all ? std::vector<double>()
                     : DequantizeRow(w, r, k, ggml_type);
        double expected = 0.0;
        double magnitude = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
          const double weight =
              keep_all ? static_cast<double>(w.values[r * k + i]) : values[i];
          const double product = weight * static_cast<double>(x_one[i]);
          expected += product;
          magnitude += std::abs(product);
        }
        scale = std::max(scale, magnitude);
        CheckRow(label + " row " + std::to_string(r), out[t * rows + r],
                 expected, scale);
      }
    }
  }

  for (void* p :
       {d_w, static_cast<void*>(d_x), static_cast<void*>(d_out), d_q8}) {
    CheckHip(hipFree(p), "free");
  }
  std::cout << name << ": " << rows << "x" << k
            << " MMQ and GEMV match the reference\n";
}

/// GGML type ids of the adapter formats (matching qfn_ggml_stubs.h).
constexpr int kQ5_0 = 6;
constexpr int kQ4K = 12;
constexpr int kQ5K = 13;
constexpr int kQ6K = 14;
constexpr int kIq4Nl = 20;

int DenseDispatch(int type, const void* w, const float* x, float* out,
                  std::size_t tokens, std::size_t rows, std::size_t k) {
  const int m = static_cast<int>(rows);
  const int n = static_cast<int>(tokens);
  const int width = static_cast<int>(k);
  switch (type) {
    case kQ4K:
      return qfn_mmq_q4_K_dense(w, x, out, m, n, width, nullptr);
    case kQ5K:
      return qfn_mmq_q5_K_dense(w, x, out, m, n, width, nullptr);
    case kQ5_0:
      return qfn_mmq_q5_0_dense(w, x, out, m, n, width, nullptr);
    case kQ6K:
      return qfn_mmq_q6_K_dense(w, x, out, m, n, width, nullptr);
    default:
      return qfn_mmq_iq4_nl_dense(w, x, out, m, n, width, nullptr);
  }
}

int DenseVecDispatch(int type, const void* w, const void* x_q8, float* out,
                     std::size_t tokens, std::size_t rows, std::size_t k) {
  const int m = static_cast<int>(rows);
  const int n = static_cast<int>(tokens);
  const int width = static_cast<int>(k);
  switch (type) {
    case kQ4K:
      return qfn_mmq_q4_K_dense_vec_preq(w, x_q8, out, m, n, width, nullptr);
    case kQ5K:
      return qfn_mmq_q5_K_dense_vec_preq(w, x_q8, out, m, n, width, nullptr);
    case kQ5_0:
      return qfn_mmq_q5_0_dense_vec_preq(w, x_q8, out, m, n, width, nullptr);
    case kQ6K:
      return qfn_mmq_q6_K_dense_vec_preq(w, x_q8, out, m, n, width, nullptr);
    default:
      return qfn_mmq_iq4_nl_dense_vec_preq(w, x_q8, out, m, n, width, nullptr);
  }
}

}  // namespace

int main() {
  try {
    CheckIq4NlRoutedVector();
    CheckQ5_0RoutedVector();
    CheckDenseRoutes("Q6_K", kQ6K, GgmlType::kQ6_K,
                     MakeQ6Weights(1024, 2560, 0xABCDU), 1024, 2560);
    CheckDenseRoutes("IQ4_NL", kIq4Nl, GgmlType::kIQ4_NL,
                     MakeIq4NlWeights(320, 1024, 0xFEEDU), 320, 1024);
    // The Swift-1.5 trunk's dense shapes: the SSM/attention projections as
    // Q4_K, attn_output as Q5_K, the vocabulary head sampled instead of
    // materializing its 2.5 GiB FP32 reference.
    CheckDenseRoutes("Q4_K", kQ4K, GgmlType::kQ4_K,
                     MakeQ4KWeights(10240, 2560, 0x1111U), 10240, 2560);
    CheckDenseRoutes("Q4_K", kQ4K, GgmlType::kQ4_K,
                     MakeQ4KWeights(12288, 2560, 0x2222U), 12288, 2560);
    CheckDenseRoutes("Q4_K", kQ4K, GgmlType::kQ4_K,
                     MakeQ4KWeights(2560, 6144, 0x3333U), 2560, 6144);
    CheckDenseRoutes("Q5_K", kQ5K, GgmlType::kQ5_K,
                     MakeQ5KWeights(2560, 6144, 0x4444U), 2560, 6144);
    CheckDenseRoutes("Q4_K", kQ4K, GgmlType::kQ4_K,
                     MakeQ4KWeights(248320, 2560, 0x5555U, false), 248320, 2560,
                     192);
    // The hyper-connection Mixer shapes: 10240 columns -> 320 rows and the
    // only admitted shape with a 320-wide contracted dimension.
    CheckDenseRoutes("Q4_K", kQ4K, GgmlType::kQ4_K,
                     MakeQ4KWeights(320, 10240, 0x6666U), 320, 10240);
    CheckDenseRoutes("Q5_0", kQ5_0, GgmlType::kQ5_0,
                     MakeQ5_0Weights(320, 10240, 0x7777U), 320, 10240);
    CheckDenseRoutes("Q5_0", kQ5_0, GgmlType::kQ5_0,
                     MakeQ5_0Weights(10240, 320, 0x8888U), 10240, 320);
    std::cout << "Flash-Next MMQ adapter checks passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "MMQ adapter check failed: " << error.what() << "\n";
    return 1;
  }
}
