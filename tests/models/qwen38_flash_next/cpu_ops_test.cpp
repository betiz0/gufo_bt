// Host reference row decode: `DequantizeRow` must reproduce the shared
// dequantization for every format the reference path reads, so an operator
// whose weights are Q5_0 is scored against the real row instead of the
// zero-fill of an unhandled type. Runs without a GPU or a model file.

#include "src/models/qwen38_flash_next/cpu_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace quant = gufo::quant;
using GgmlType = gufo::core::GgmlType;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

/// A Q5_0 row payload of `blocks` blocks with a recognizable, non-zero
/// pattern: low nibbles count up, high nibbles down, and every other fifth
/// bit is set.
std::vector<std::uint8_t> MakeQ5_0Row(std::size_t blocks, std::uint32_t seed) {
  std::vector<std::uint8_t> bytes(blocks * 22, 0);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::uint8_t* block = bytes.data() + b * 22;
    const std::uint16_t d =
        static_cast<std::uint16_t>(0x3000 + (seed % 4) * 0x0800);
    std::memcpy(block, &d, sizeof(d));
    for (std::size_t i = 0; i < 16; ++i) {
      const std::uint8_t low = static_cast<std::uint8_t>((i + b) & 0x0FU);
      const std::uint8_t high = static_cast<std::uint8_t>((15U - i) & 0x0FU);
      block[6 + i] = static_cast<std::uint8_t>(low | (high << 4U));
    }
    // Bits 0 and 17 of the high-bit word, so both halves exercise the
    // fifth bit.
    block[2] = 0x01;
    block[4] = 0x02;
  }
  return bytes;
}

void CheckQ5_0Row() {
  constexpr std::uint64_t kCols = 64;
  const std::uint64_t rows = 3;
  const std::vector<std::uint8_t> payload = MakeQ5_0Row(rows * (kCols / 32), 1);
  const qfn::TensorRef t{.data = payload.data(),
                         .type = GgmlType::kQ5_0,
                         .cols = kCols,
                         .rows = rows,
                         .experts = 1,
                         .name = "blk.0.hc_ffn_up.weight",
                         .cls = qfn::WeightClass::kMixerDownUp};
  Require(t.RowBytes() == (kCols / 32) * 22,
          "the Q5_0 row geometry is not the stored encoding's");
  for (std::uint64_t row = 0; row < rows; ++row) {
    std::vector<float> actual(kCols);
    qfn::cpu::DequantizeRow(t, 0, row, actual.data());
    std::vector<float> expected(kCols);
    quant::DequantizeQ5_0(payload.data() + row * t.RowBytes(), expected.data(),
                          kCols);
    double magnitude = 0.0;
    for (std::size_t i = 0; i < kCols; ++i) {
      magnitude += std::abs(expected[i]);
      Require(actual[i] == expected[i],
              "Q5_0 row " + std::to_string(row) + " element " +
                  std::to_string(i) + " decoded as " +
                  std::to_string(actual[i]) + ", expected " +
                  std::to_string(expected[i]));
    }
    Require(magnitude > 0.0, "the Q5_0 reference row is all zero");
  }

  // The reference matrix-vector path reads the same rows: a non-zero weight
  // row must contribute to every output row.
  const std::vector<float> x(kCols, 1.0F);
  std::vector<float> y(rows, 0.0F);
  qfn::cpu::MatVec(t, 0, x, y);
  for (std::uint64_t row = 0; row < rows; ++row) {
    Require(y[row] != 0.0F,
            "MatVec produced a zero row for a non-zero Q5_0 weight row");
  }
}

/// The previously handled 32-element formats keep their own decode; the check
/// guards against a Q5_0 case being wired to the wrong layout.
void CheckQ5_1RowStays() {
  constexpr std::uint64_t kCols = 32;
  std::vector<std::uint8_t> payload(24, 0);
  const std::uint16_t d = 0x3C00;  // fp16 1.0
  const std::uint16_t m = 0x3800;  // fp16 0.5
  std::memcpy(payload.data(), &d, sizeof(d));
  std::memcpy(payload.data() + 2, &m, sizeof(m));
  for (std::size_t i = 0; i < 16; ++i) {
    payload[8 + i] =
        static_cast<std::uint8_t>((i & 0x0FU) | (((15U - i) & 0x0FU) << 4U));
  }
  const std::uint32_t high = 0x00020001U;
  std::memcpy(payload.data() + 4, &high, sizeof(high));
  const qfn::TensorRef t{.data = payload.data(),
                         .type = GgmlType::kQ5_1,
                         .cols = kCols,
                         .rows = 1,
                         .name = "blk.0.ssm_alpha.weight",
                         .cls = qfn::WeightClass::kSsmGate};
  std::vector<float> actual(kCols);
  qfn::cpu::DequantizeRow(t, 0, 0, actual.data());
  // Layout: element i of the first half is (qs[i] & 0xF) | (bit i << 4),
  // times d, plus m; the second half takes the high nibble and bit i + 16.
  for (std::size_t i = 0; i < 16; ++i) {
    const float lo = static_cast<float>((payload[8 + i] & 0x0FU) |
                                        (((high >> i) & 1U) << 4U));
    const float hi = static_cast<float>((payload[8 + i] >> 4U) |
                                        (((high >> (i + 16)) & 1U) << 4U));
    Require(std::abs(actual[i] - (lo * 1.0F + 0.5F)) < 1e-6F,
            "Q5_1 low-nibble decode changed");
    Require(std::abs(actual[16 + i] - (hi * 1.0F + 0.5F)) < 1e-6F,
            "Q5_1 high-nibble decode changed");
  }
}

}  // namespace

int main() {
  try {
    CheckQ5_0Row();
    CheckQ5_1RowStays();
    std::cout << "Flash-Next host row-decode checks passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "host row-decode check failed: " << error.what() << "\n";
    return 1;
  }
}
