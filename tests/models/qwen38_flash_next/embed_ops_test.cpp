// Embedding lookup operator check: EmbedTokens must reproduce the CPU
// dequantization for every table format the loader accepts, including the
// Q6_K and Q4_K rows the AP and Swift trunks store (the 256-element block
// decode that RowBytes/RowElement provide).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "tests/models/qwen38_flash_next/quant_fixture.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace fixture = gufo::models::qwen38_flash_next::fixture;

namespace {

constexpr std::size_t kRows = 12;
constexpr std::size_t kHidden = 2560;  // Ten Q6_K blocks, five Q8_0 rows' worth
constexpr std::size_t kStreams = 4;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

/// F32 reference rows for the table, one vector per row.
using Rows = std::vector<std::vector<float>>;

/// Q8_0 table rows plus their packed blocks: one scale per 32-element block.
Rows MakeQ8Rows(std::uint32_t seed, std::vector<std::uint8_t>* packed) {
  Rows rows(kRows, std::vector<float>(kHidden));
  packed->assign(kRows * (kHidden / 32) * 34, 0);
  std::uint32_t block_seed = seed;
  for (std::size_t r = 0; r < kRows; ++r) {
    for (std::size_t b = 0; b < kHidden / 32; ++b) {
      std::uint8_t* block = packed->data() + (r * (kHidden / 32) + b) * 34;
      const float scale =
          0.0009765625F * static_cast<float>(1 + (block_seed++ % 7));
      const __half d = __float2half(scale);
      std::memcpy(block, &d, sizeof(d));
      const float stored = __half2float(d);
      for (std::size_t i = 0; i < 32; ++i) {
        const auto code = static_cast<std::int8_t>(
            static_cast<int>(fixture::NextRandom(&seed) % 255) - 127);
        block[2 + i] = static_cast<std::uint8_t>(code);
        rows[r][b * 32 + i] = stored * static_cast<float>(code);
      }
    }
  }
  return rows;
}

void CheckTable(q::WeightType type, const std::string& name,
                const std::vector<std::uint8_t>& packed, const Rows& rows,
                const std::vector<std::int32_t>& tokens) {
  std::uint8_t* table = nullptr;
  std::int32_t* d_tokens = nullptr;
  float* out = nullptr;
  CheckHip(hipMalloc(&table, packed.size()), "table");
  CheckHip(hipMalloc(&d_tokens, tokens.size() * sizeof(std::int32_t)),
           "tokens");
  CheckHip(hipMalloc(&out, tokens.size() * kStreams * kHidden * sizeof(float)),
           "output");
  CheckHip(
      hipMemcpy(table, packed.data(), packed.size(), hipMemcpyHostToDevice),
      "table upload");
  CheckHip(
      hipMemcpy(d_tokens, tokens.data(), tokens.size() * sizeof(std::int32_t),
                hipMemcpyHostToDevice),
      "token upload");
  q::EmbedTokens(table, type, d_tokens, out,
                 static_cast<std::uint32_t>(tokens.size()),
                 static_cast<std::uint32_t>(kHidden),
                 static_cast<std::uint32_t>(kStreams), nullptr);
  CheckHip(hipDeviceSynchronize(), "embedding");
  std::vector<float> actual(tokens.size() * kStreams * kHidden);
  CheckHip(hipMemcpy(actual.data(), out, actual.size() * sizeof(float),
                     hipMemcpyDeviceToHost),
           "output download");
  for (std::size_t t = 0; t < tokens.size(); ++t) {
    const auto& expected = rows[static_cast<std::size_t>(tokens[t])];
    for (std::size_t s = 0; s < kStreams; ++s) {
      for (std::size_t i = 0; i < kHidden; ++i) {
        const float value = actual[(t * kStreams + s) * kHidden + i];
        if (std::abs(value - expected[i]) >
            1e-3F * std::max(1.0F, std::abs(expected[i]))) {
          throw std::runtime_error(name + " row " + std::to_string(tokens[t]) +
                                   " stream " + std::to_string(s) +
                                   " element " + std::to_string(i) + " is " +
                                   std::to_string(value) + ", expected " +
                                   std::to_string(expected[i]));
        }
      }
    }
  }
  for (void* p : {static_cast<void*>(table), static_cast<void*>(d_tokens),
                  static_cast<void*>(out)}) {
    CheckHip(hipFree(p), "free");
  }
  std::cout << name << ": " << tokens.size() << " rows x " << kHidden
            << " match the CPU dequantization\n";
}

}  // namespace

int main() {
  try {
    const std::vector<std::int32_t> tokens = {0, 3, 7, 11, 5, 5};

    // Q8_0 rows, packed by hand from the same dequantized values.
    {
      std::vector<std::uint8_t> packed;
      const auto rows = MakeQ8Rows(0x51A1U, &packed);
      CheckTable(q::WeightType::kQ8_0, "Q8_0", packed, rows, tokens);
    }

    // Q6_K rows: the new 256-element block decode.
    {
      const auto weights = fixture::MakeQ6Weights(kRows, kHidden, 0x6B10U);
      std::vector<std::vector<float>> rows(kRows, std::vector<float>(kHidden));
      for (std::size_t r = 0; r < kRows; ++r) {
        std::copy_n(weights.values.begin() + r * kHidden, kHidden,
                    rows[r].begin());
      }
      CheckTable(q::WeightType::kQ6_K, "Q6_K", weights.bytes, rows, tokens);
    }

    // Q4_K rows: the Swift-1.5 token embedding's format, whose group
    // scale/minimum pair the row decode has to unpack.
    {
      const auto weights = fixture::MakeQ4KWeights(kRows, kHidden, 0x4B10U);
      std::vector<std::vector<float>> rows(kRows, std::vector<float>(kHidden));
      for (std::size_t r = 0; r < kRows; ++r) {
        std::copy_n(weights.values.begin() + r * kHidden, kHidden,
                    rows[r].begin());
      }
      CheckTable(q::WeightType::kQ4_K, "Q4_K", weights.bytes, rows, tokens);
    }

    // Q5_K rows: the same class's other admitted 256-element format, which
    // also carries the fifth-bit word.
    {
      const auto weights = fixture::MakeQ5KWeights(kRows, kHidden, 0x5B10U);
      std::vector<std::vector<float>> rows(kRows, std::vector<float>(kHidden));
      for (std::size_t r = 0; r < kRows; ++r) {
        std::copy_n(weights.values.begin() + r * kHidden, kHidden,
                    rows[r].begin());
      }
      CheckTable(q::WeightType::kQ5_K, "Q5_K", weights.bytes, rows, tokens);
    }

    // F32 and F16 rows keep their stored values.
    {
      Rows rows(kRows, std::vector<float>(kHidden));
      std::uint32_t seed = 0xF32U;
      for (auto& row : rows) {
        for (float& v : row) {
          v = 0.5F *
              static_cast<float>(
                  static_cast<int>(fixture::NextRandom(&seed) & 0xFFFFU) -
                  32768) /
              32768.0F;
        }
      }
      std::vector<std::uint8_t> packed(kRows * kHidden * sizeof(float));
      std::memcpy(packed.data(), rows[0].data(), 0);
      for (std::size_t r = 0; r < kRows; ++r) {
        std::memcpy(packed.data() + r * kHidden * sizeof(float), rows[r].data(),
                    kHidden * sizeof(float));
      }
      CheckTable(q::WeightType::kF32, "F32", packed, rows, tokens);

      std::vector<std::uint8_t> half(kRows * kHidden * sizeof(__half));
      Rows rounded(kRows, std::vector<float>(kHidden));
      for (std::size_t r = 0; r < kRows; ++r) {
        for (std::size_t i = 0; i < kHidden; ++i) {
          const __half h = __float2half(rows[r][i]);
          std::memcpy(half.data() + (r * kHidden + i) * sizeof(__half), &h,
                      sizeof(h));
          rounded[r][i] = __half2float(h);
        }
      }
      CheckTable(q::WeightType::kF16, "F16", half, rounded, tokens);
    }

    std::cout << "Flash-Next embedding operator checks passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "embedding check failed: " << error.what() << "\n";
    return 1;
  }
}
