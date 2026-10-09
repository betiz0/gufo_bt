#ifndef GUFO_TESTS_MODELS_QWEN_HIP_SUPPORT_FP16_HPP_
#define GUFO_TESTS_MODELS_QWEN_HIP_SUPPORT_FP16_HPP_

#include <cstdint>
#include <cstring>

namespace gufo::test {

/// Host-side round-to-nearest FP16 encoding. The HIP headers declare
/// `__half_as_ushort` only for the clang HIP compiler, so host test code that
/// must also build with GCC encodes through `_Float16` instead.
[[nodiscard]] inline std::uint16_t FloatToFp16Bits(float value) noexcept {
  const _Float16 half = static_cast<_Float16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

}  // namespace gufo::test

#endif  // GUFO_TESTS_MODELS_QWEN_HIP_SUPPORT_FP16_HPP_
