#ifndef GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_QUANT_FIXTURE_HPP_
#define GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_QUANT_FIXTURE_HPP_

// Host-side Q6_K / IQ4_NL weight fixtures for the operator tests: packed
// blocks that follow the GGUF layouts, the CPU dequantization of those same
// blocks as the FP32 reference, and activation rows whose Q8_1 quantization is
// exact (integers times 2^-10 with a 127 magnitude in every 32-wide block), so
// a comparison isolates the kernel's arithmetic from quantizer tie-breaking.

#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::qwen38_flash_next::fixture {

inline std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

/// Packed weights plus their CPU-dequantized values, row-major [rows][k].
struct Packed {
  std::vector<std::uint8_t> bytes;
  std::vector<float> values;
  std::size_t row_bytes{0};
};

/// `with_values` keeps the FP32 reference matrix; the trunk's wide shapes
/// dequantize the few reference rows they check instead.
inline Packed MakeQ6Weights(std::size_t rows, std::size_t k, std::uint32_t seed,
                            bool with_values = true) {
  using gufo::quant::block_q6_K;
  using gufo::quant::DequantizeQ6_K;
  Packed w;
  const std::size_t blocks = k / 256;
  w.row_bytes = blocks * sizeof(block_q6_K);
  w.bytes.assign(rows * w.row_bytes, 0);
  if (with_values) {
    w.values.resize(rows * k);
  }
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0; b < blocks; ++b) {
      auto* block = reinterpret_cast<block_q6_K*>(
          w.bytes.data() + r * w.row_bytes + b * sizeof(block_q6_K));
      const float d = 0.00390625F * static_cast<float>(1 + (seed % 5));
      const __half dh = __float2half(d);
      std::memcpy(&block->d, &dh, sizeof(dh));
      std::uint8_t codes[256];
      for (std::size_t i = 0; i < 256; ++i) {
        codes[i] = static_cast<std::uint8_t>(NextRandom(&seed) % 64);
      }
      for (std::size_t s = 0; s < 16; ++s) {
        block->scales[s] = static_cast<std::int8_t>(1 + (seed >> s) % 6);
      }
      // Pack the codes exactly as the dequantizer reads them: two 128-element
      // halves, four 32-element segments per half, low nibbles in ql[0..63]
      // (segments 0 and 2), high nibbles in ql[64..127], two bits per segment
      // in qh.
      for (std::size_t half = 0; half < 2; ++half) {
        for (std::size_t segment = 0; segment < 4; ++segment) {
          for (std::size_t lane = 0; lane < 32; ++lane) {
            const std::size_t index = half * 128 + segment * 32 + lane;
            const std::uint8_t code = codes[index];
            std::uint8_t* ql =
                block->ql + half * 64 + ((segment % 2 == 0) ? 0 : 32) + lane;
            if (segment < 2) {
              *ql = static_cast<std::uint8_t>((*ql & 0xF0U) | (code & 0x0FU));
            } else {
              *ql = static_cast<std::uint8_t>((*ql & 0x0FU) |
                                              ((code & 0x0FU) << 4U));
            }
            std::uint8_t* qh = block->qh + half * 32 + lane;
            const std::uint8_t shift = static_cast<std::uint8_t>(2 * segment);
            *qh = static_cast<std::uint8_t>((*qh & ~(0x03U << shift)) |
                                            (((code >> 4U) & 0x03U) << shift));
          }
        }
      }
      if (with_values) {
        DequantizeQ6_K(block, w.values.data() + r * k + b * 256, 256);
      }
    }
  }
  return w;
}

/// Q8_0 blocks (fp16 scale + 32 int8) with their CPU dequantization. The
/// scale is a power of two, so the dequantized values are exact in F16 and a
/// mixed-format decode can be compared without a rounding allowance.
inline Packed MakeQ8Weights(std::size_t rows, std::size_t k,
                            std::uint32_t seed) {
  using gufo::quant::block_q8_0;
  using gufo::quant::DequantizeQ8_0;
  Packed w;
  const std::size_t blocks = k / 32;
  w.row_bytes = blocks * sizeof(block_q8_0);
  w.bytes.assign(rows * w.row_bytes, 0);
  w.values.resize(rows * k);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0; b < blocks; ++b) {
      auto* block = reinterpret_cast<block_q8_0*>(
          w.bytes.data() + r * w.row_bytes + b * sizeof(block_q8_0));
      const float d = 0.0009765625F * static_cast<float>(1 + (seed % 3));
      const __half dh = __float2half(d);
      std::memcpy(&block->d, &dh, sizeof(dh));
      for (std::size_t i = 0; i < 32; ++i) {
        const int magnitude = 1 + static_cast<int>(NextRandom(&seed) % 126);
        const bool sign = (NextRandom(&seed) & 1U) != 0;
        block->qs[i] = static_cast<std::int8_t>(sign ? -magnitude : magnitude);
      }
      DequantizeQ8_0(block, w.values.data() + r * k + b * 32, 32);
    }
  }
  return w;
}

inline Packed MakeIq4NlWeights(std::size_t rows, std::size_t k,
                               std::uint32_t seed) {
  using gufo::quant::block_iq4_nl;
  using gufo::quant::DequantizeIQ4_NL;
  Packed w;
  const std::size_t blocks = k / 32;
  w.row_bytes = blocks * sizeof(block_iq4_nl);
  w.bytes.assign(rows * w.row_bytes, 0);
  w.values.resize(rows * k);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0; b < blocks; ++b) {
      auto* block = reinterpret_cast<block_iq4_nl*>(
          w.bytes.data() + r * w.row_bytes + b * sizeof(block_iq4_nl));
      const float d = 0.0078125F * static_cast<float>(1 + (seed % 3));
      const __half dh = __float2half(d);
      std::memcpy(&block->d, &dh, sizeof(dh));
      for (std::size_t i = 0; i < 16; ++i) {
        const std::uint8_t low =
            static_cast<std::uint8_t>(NextRandom(&seed) % 16);
        const std::uint8_t high =
            static_cast<std::uint8_t>(NextRandom(&seed) % 16);
        block->qs[i] = static_cast<std::uint8_t>(low | (high << 4U));
      }
      DequantizeIQ4_NL(block, w.values.data() + r * k + b * 32, 32);
    }
  }
  return w;
}

/// Integer activations scaled by 2^-10, 127 magnitude in every 32-wide block.
inline std::vector<float> MakeIntegerActivations(std::size_t tokens,
                                                 std::size_t k,
                                                 std::uint32_t seed) {
  std::vector<float> x(tokens * k);
  const float scale = 0.0009765625F;  // 2^-10, exact in FP16
  for (std::size_t t = 0; t < tokens; ++t) {
    for (std::size_t b = 0; b < k / 32; ++b) {
      for (std::size_t i = 0; i < 32; ++i) {
        const bool sign = (NextRandom(&seed) & 1U) != 0;
        const int magnitude =
            i == 0 ? 127 : 1 + static_cast<int>(NextRandom(&seed) % 126);
        x[t * k + b * 32 + i] =
            scale * static_cast<float>(sign ? -magnitude : magnitude);
      }
    }
  }
  return x;
}

/// Q4_K super-blocks (fp16 d/dmin, 12 scale bytes, 128 nibble bytes) with
/// their CPU dequantization.
inline Packed MakeQ4KWeights(std::size_t rows, std::size_t k,
                             std::uint32_t seed, bool with_values = true) {
  using gufo::quant::block_q4_K;
  using gufo::quant::DequantizeQ4_K;
  Packed w;
  const std::size_t blocks = k / 256;
  w.row_bytes = blocks * sizeof(block_q4_K);
  w.bytes.assign(rows * w.row_bytes, 0);
  if (with_values) {
    w.values.resize(rows * k);
  }
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0; b < blocks; ++b) {
      auto* block = reinterpret_cast<block_q4_K*>(
          w.bytes.data() + r * w.row_bytes + b * sizeof(block_q4_K));
      // Exact half scales: the dequantized values stay comparable without a
      // per-element rounding allowance.
      const __half dh =
          __float2half(0.001953125F * static_cast<float>(1 + (seed % 5)));
      const __half mh =
          __float2half(0.0009765625F * static_cast<float>(1 + (seed % 3)));
      std::memcpy(&block->d, &dh, sizeof(dh));
      std::memcpy(&block->dmin, &mh, sizeof(mh));
      for (std::size_t i = 0; i < sizeof(block->scales); ++i) {
        block->scales[i] = static_cast<std::uint8_t>(NextRandom(&seed));
      }
      for (std::size_t i = 0; i < sizeof(block->qs); ++i) {
        block->qs[i] = static_cast<std::uint8_t>(NextRandom(&seed));
      }
      if (with_values) {
        DequantizeQ4_K(block, w.values.data() + r * k + b * 256, 256);
      }
    }
  }
  return w;
}

/// Q5_K super-blocks: the Q4_K header plus the 32 high-bit bytes.
inline Packed MakeQ5KWeights(std::size_t rows, std::size_t k,
                             std::uint32_t seed, bool with_values = true) {
  using gufo::quant::block_q5_K;
  using gufo::quant::DequantizeQ5_K;
  Packed w;
  const std::size_t blocks = k / 256;
  w.row_bytes = blocks * sizeof(block_q5_K);
  w.bytes.assign(rows * w.row_bytes, 0);
  if (with_values) {
    w.values.resize(rows * k);
  }
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0; b < blocks; ++b) {
      auto* block = reinterpret_cast<block_q5_K*>(
          w.bytes.data() + r * w.row_bytes + b * sizeof(block_q5_K));
      const __half dh =
          __float2half(0.001953125F * static_cast<float>(1 + (seed % 5)));
      const __half mh =
          __float2half(0.0009765625F * static_cast<float>(1 + (seed % 3)));
      std::memcpy(&block->d, &dh, sizeof(dh));
      std::memcpy(&block->dmin, &mh, sizeof(mh));
      for (std::size_t i = 0; i < sizeof(block->scales); ++i) {
        block->scales[i] = static_cast<std::uint8_t>(NextRandom(&seed));
      }
      for (std::size_t i = 0; i < sizeof(block->qh); ++i) {
        block->qh[i] = static_cast<std::uint8_t>(NextRandom(&seed));
      }
      for (std::size_t i = 0; i < sizeof(block->qs); ++i) {
        block->qs[i] = static_cast<std::uint8_t>(NextRandom(&seed));
      }
      if (with_values) {
        DequantizeQ5_K(block, w.values.data() + r * k + b * 256, 256);
      }
    }
  }
  return w;
}

/// Q5_0 blocks (fp16 scale, 4 high-bit bytes, 16 nibble bytes) with their
/// CPU dequantization.
inline Packed MakeQ5_0Weights(std::size_t rows, std::size_t k,
                              std::uint32_t seed, bool with_values = true) {
  using gufo::quant::block_q5_0;
  using gufo::quant::DequantizeQ5_0;
  Packed w;
  const std::size_t blocks = k / 32;
  w.row_bytes = blocks * sizeof(block_q5_0);
  w.bytes.assign(rows * w.row_bytes, 0);
  if (with_values) {
    w.values.resize(rows * k);
  }
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0; b < blocks; ++b) {
      auto* block = reinterpret_cast<block_q5_0*>(
          w.bytes.data() + r * w.row_bytes + b * sizeof(block_q5_0));
      const __half dh =
          __float2half(0.0009765625F * static_cast<float>(1 + (seed % 4)));
      std::memcpy(&block->d, &dh, sizeof(dh));
      for (std::size_t i = 0; i < sizeof(block->qh); ++i) {
        block->qh[i] = static_cast<std::uint8_t>(NextRandom(&seed));
      }
      for (std::size_t i = 0; i < sizeof(block->qs); ++i) {
        block->qs[i] = static_cast<std::uint8_t>(NextRandom(&seed));
      }
      if (with_values) {
        DequantizeQ5_0(block, w.values.data() + r * k + b * 32, 32);
      }
    }
  }
  return w;
}

/// F64 dot product of one dequantized weight row with one activation row.
/// `scale` accumulates the sum of |products| so callers can bound the
/// tolerance relative to cancellation.
inline double ReferenceDot(const Packed& w, const std::vector<float>& x,
                           std::size_t row, std::size_t token, std::size_t k,
                           double* scale) {
  double sum = 0.0;
  double magnitude = 0.0;
  for (std::size_t i = 0; i < k; ++i) {
    const double product = static_cast<double>(w.values[row * k + i]) *
                           static_cast<double>(x[token * k + i]);
    sum += product;
    magnitude += std::abs(product);
  }
  *scale = std::max(*scale, magnitude);
  return sum;
}

/// Dequantizes one weight row without materializing the whole matrix; used
/// for the shapes whose full FP32 reference would not fit.
inline std::vector<double> DequantizeRow(const Packed& w, std::size_t row,
                                         std::size_t k,
                                         gufo::core::GgmlType type) {
  std::vector<float> values(k);
  const std::uint8_t* src = w.bytes.data() + row * w.row_bytes;
  switch (type) {
    case gufo::core::GgmlType::kQ4_K:
      gufo::quant::DequantizeQ4_K(src, values.data(), k);
      break;
    case gufo::core::GgmlType::kQ5_K:
      gufo::quant::DequantizeQ5_K(src, values.data(), k);
      break;
    case gufo::core::GgmlType::kQ5_0:
      gufo::quant::DequantizeQ5_0(src, values.data(), k);
      break;
    case gufo::core::GgmlType::kQ6_K:
      gufo::quant::DequantizeQ6_K(src, values.data(), k);
      break;
    default:
      gufo::quant::DequantizeIQ4_NL(src, values.data(), k);
      break;
  }
  return std::vector<double>(values.begin(), values.end());
}

inline void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace gufo::models::qwen38_flash_next::fixture

#endif  // GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_QUANT_FIXTURE_HPP_
