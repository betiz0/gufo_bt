// Test-side policy for design D21: the normal ctest presets and the normal
// `tools/qwen27b/check.py` suites never run more than one session at a time.
// Tests that batch or interleave several sessions set GUFO_MULTI_SESSION_TESTS
// (the dedicated ctest preset/label `multi-session` and the matching check.py
// suite do this) and skip themselves otherwise.
#ifndef GUFO_TESTS_MODELS_QWEN27B_MULTI_SESSION_HPP_
#define GUFO_TESTS_MODELS_QWEN27B_MULTI_SESSION_HPP_

#include <cstdlib>
#include <iostream>

namespace gufo::tests {

[[nodiscard]] inline bool MultiSessionEnabled() {
  static const bool enabled =
      std::getenv("GUFO_MULTI_SESSION_TESTS") != nullptr;
  return enabled;
}

/// Prints the standard notice for a skipped multi-session check.
inline void ReportMultiSessionSkipped(const char* what) {
  std::cout << "[multi-session] skipped: " << what
            << " (set GUFO_MULTI_SESSION_TESTS=1)\n";
}

}  // namespace gufo::tests

#endif  // GUFO_TESTS_MODELS_QWEN27B_MULTI_SESSION_HPP_
