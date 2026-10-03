#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

#include "src/models/qwen/kvarn_reference.hpp"

namespace {

using gufo::models::qwen::kKvarnHeadDim;
using gufo::models::qwen::KvarnFwhtHeadInPlace;

// Declared limits (spec "数値一致の定義"). The same values and their derivation
// are recorded in docs/models/qwen3.8-27b/QUALITY.md; they are FP32 rounding
// bounds for eight butterfly stages and must not be loosened to pass.
//
// Round trip: |x - H(Hx)|_inf <= limit, for ||x||_inf = 1
// QK score:   |<q,k> - <Hq,Hk>| <= limit, for ||q||_2 = ||k||_2 = 1
inline constexpr float kFwhtRoundTripLimit = 1e-4F;
inline constexpr float kFwhtQkScoreLimit = 1e-4F;

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

[[nodiscard]] std::vector<float> RandomHead(std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> values(kKvarnHeadDim);
  for (float& value : values) {
    value = dist(rng);
  }
  return values;
}

void NormalizeInf(std::vector<float>& values) {
  float maximum = 0.0F;
  for (const float value : values) {
    maximum = std::max(maximum, std::abs(value));
  }
  for (float& value : values) {
    value /= maximum;
  }
}

void NormalizeL2(std::vector<float>& values) {
  double sum = 0.0;
  for (const float value : values) {
    sum += static_cast<double>(value) * static_cast<double>(value);
  }
  const float norm = static_cast<float>(std::sqrt(sum));
  for (float& value : values) {
    value /= norm;
  }
}

[[nodiscard]] double Dot(const std::vector<float>& a,
                         const std::vector<float>& b) {
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return sum;
}

/// H . H^T = I: the transform matrix is symmetric and applying it twice is the
/// identity. Both properties are checked directly rather than through a
/// hand-written Hadamard matrix, so the test covers the shipped call.
void TestRotationMatrixIsOrthogonal() {
  std::array<float, kKvarnHeadDim * kKvarnHeadDim> matrix{};
  for (std::size_t i = 0; i < kKvarnHeadDim; ++i) {
    std::vector<float> basis(kKvarnHeadDim, 0.0F);
    basis[i] = 1.0F;
    KvarnFwhtHeadInPlace(basis);
    for (std::size_t j = 0; j < kKvarnHeadDim; ++j) {
      matrix[i * kKvarnHeadDim + j] = basis[j];
    }
  }

  float worst_asymmetry = 0.0F;
  for (std::size_t i = 0; i < kKvarnHeadDim; ++i) {
    for (std::size_t j = 0; j < kKvarnHeadDim; ++j) {
      worst_asymmetry =
          std::max(worst_asymmetry, std::abs(matrix[i * kKvarnHeadDim + j] -
                                             matrix[j * kKvarnHeadDim + i]));
    }
  }
  Expect(worst_asymmetry == 0.0F, "the rotation matrix is exactly symmetric");

  float worst_identity = 0.0F;
  for (std::size_t i = 0; i < kKvarnHeadDim; ++i) {
    std::vector<float> basis(kKvarnHeadDim, 0.0F);
    basis[i] = 1.0F;
    KvarnFwhtHeadInPlace(basis);
    KvarnFwhtHeadInPlace(basis);
    for (std::size_t j = 0; j < kKvarnHeadDim; ++j) {
      const float expected = i == j ? 1.0F : 0.0F;
      worst_identity = std::max(worst_identity, std::abs(basis[j] - expected));
    }
  }
  Expect(worst_identity <= kFwhtRoundTripLimit,
         "H . H^T = I within the declared limit");
  std::cout << "  rotation matrix: symmetry " << worst_asymmetry
            << ", identity error " << worst_identity << "\n";
}

void TestRoundTripPreservesVector() {
  std::mt19937 rng(11);
  float worst = 0.0F;
  for (int trial = 0; trial < 64; ++trial) {
    std::vector<float> original = RandomHead(rng);
    NormalizeInf(original);
    std::vector<float> rotated = original;
    KvarnFwhtHeadInPlace(rotated);
    KvarnFwhtHeadInPlace(rotated);
    for (std::size_t i = 0; i < kKvarnHeadDim; ++i) {
      worst = std::max(worst, std::abs(rotated[i] - original[i]));
    }
  }
  Expect(worst <= kFwhtRoundTripLimit,
         "an attached pair of rotations returns the input");
  std::cout << "  round trip over 64 unit-infinity heads: worst " << worst
            << "\n";
}

void TestQkScoreIsInvariant() {
  std::mt19937 rng(12);
  double worst = 0.0;
  for (int trial = 0; trial < 256; ++trial) {
    std::vector<float> query = RandomHead(rng);
    std::vector<float> key = RandomHead(rng);
    NormalizeL2(query);
    NormalizeL2(key);
    const double plain = Dot(query, key);

    KvarnFwhtHeadInPlace(query);
    KvarnFwhtHeadInPlace(key);
    const double rotated = Dot(query, key);
    worst = std::max(worst, std::abs(plain - rotated));
  }
  Expect(worst <= static_cast<double>(kFwhtQkScoreLimit),
         "QK scores are invariant under rotation");
  std::cout << "  QK score drift over 256 unit-norm pairs: worst " << worst
            << "\n";
}

void TestWrongLengthIsIgnored() {
  std::vector<float> too_short(kKvarnHeadDim - 1, 3.0F);
  KvarnFwhtHeadInPlace(too_short);
  for (const float value : too_short) {
    Expect(value == 3.0F, "a wrong-sized vector is left untouched");
  }
}

}  // namespace

int main() {
  std::cout << "Running KVarN FWHT oracle tests...\n";
  TestRotationMatrixIsOrthogonal();
  TestRoundTripPreservesVector();
  TestQkScoreIsInvariant();
  TestWrongLengthIsIgnored();
  std::cout << "All KVarN FWHT oracle tests passed successfully!\n";
  return 0;
}
