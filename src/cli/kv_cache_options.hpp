#ifndef GUFO_CLI_KV_CACHE_OPTIONS_HPP_
#define GUFO_CLI_KV_CACHE_OPTIONS_HPP_

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

#include "src/models/qwen/hip/execution_policy.hpp"

namespace gufo::cli {

/// Selection carried by the canonical `--kv-cache` long option. An empty
/// `value` means the option was not given, so every model keeps its own
/// default KV format. There is exactly one option name and no alias.
struct KvCacheSelection {
  std::string value;

  [[nodiscard]] bool specified() const noexcept { return !value.empty(); }
};

/// The single value this build accepts, and Qwen3.8 27B's default format.
/// KVarN presets are registered here together with their storage.
inline constexpr std::string_view kKvCacheFp16{"fp16"};

/// Value list printed in `--help`.
[[nodiscard]] inline std::string KvCacheValueList() {
  std::string values =
      std::string(kKvCacheFp16) + ", kvarn-k4v4, kvarn-k5v4, kvarn-k6v5";
#if defined(GUFO_QWEN_PSEUDO_KVARN)
  // gpu-test quality probes (design D1).
  values += ", baseline-q8, baseline-q4";
#endif
  return values;
}

/// Validates one `--kv-cache` value. Callers run this after parsing and before
/// opening a model, so an unknown value exits non-zero without loading.
[[nodiscard]] inline bool ValidateKvCacheValue(std::string_view value,
                                               std::string* error) {
  if (value == kKvCacheFp16) {
    return true;
  }
#if defined(GUFO_QWEN_PSEUDO_KVARN)
  if (value == "baseline-q8" || value == "baseline-q4") {
    return true;
  }
#endif
  if (hip::QwenKvarnPreset preset; hip::ParseQwenKvarnPreset(value, &preset)) {
    return true;
  }
  if (error != nullptr) {
    *error =
        "--kv-cache value '" + std::string(value) +
        "' is not supported by this build (supported: " + KvCacheValueList() +
        ")";
  }
  return false;
}

/// Help text of `--kv-cache-tail`, shared by every parser that registers it.
[[nodiscard]] inline std::string KvCacheTailHelp() {
  return "Full-precision tail tokens kept behind each KVarN query, " +
         std::to_string(hip::kQwenKvCacheTailMin) + "-" +
         std::to_string(hip::kQwenKvCacheTailMax) +
         " (default: no tail); requires a KVarN --kv-cache";
}

/// Parses `--kv-cache-tail`. Empty text means the option was not given and
/// yields 0. Anything but a plain integer in the supported range is an error.
[[nodiscard]] inline bool ParseKvCacheTail(std::string_view text,
                                           std::uint32_t* tail,
                                           std::string* error) {
  *tail = 0;
  if (text.empty()) {
    return true;
  }
  std::uint64_t value = 0;
  bool numeric = true;
  for (const char c : text) {
    if (c < '0' || c > '9' || value > hip::kQwenKvCacheTailMax) {
      numeric = false;
      break;
    }
    value = value * 10 + static_cast<std::uint64_t>(c - '0');
  }
  if (!numeric || value < hip::kQwenKvCacheTailMin ||
      value > hip::kQwenKvCacheTailMax) {
    if (error != nullptr) {
      *error = "--kv-cache-tail value '" + std::string(text) +
               "' must be an integer from " +
               std::to_string(hip::kQwenKvCacheTailMin) + " to " +
               std::to_string(hip::kQwenKvCacheTailMax);
    }
    return false;
  }
  *tail = static_cast<std::uint32_t>(value);
  return true;
}

/// Custom parse callback for `--kv-cache-tail`: stores the text and rejects an
/// empty value, which a plain string target could not tell from "not given".
[[nodiscard]] inline std::function<bool(std::string_view, std::string_view,
                                        std::string*)>
KvCacheTailParser(std::string* target) {
  return
      [target](std::string_view, std::string_view value, std::string* error) {
        if (value.empty()) {
          if (error != nullptr) {
            *error = "--kv-cache-tail needs a value";
          }
          return false;
        }
        *target = std::string(value);
        return true;
      };
}

/// Tail tokens of an already validated `--kv-cache-tail` (0 when not given).
[[nodiscard]] inline std::uint32_t KvCacheTailTokens(std::string_view text) {
  std::uint32_t tail = 0;
  (void)ParseKvCacheTail(text, &tail, nullptr);
  return tail;
}

/// Validates `--kv-cache` together with `--kv-cache-tail` after parsing and
/// before any model is opened. Only a KVarN preset keeps a tail; `fp16` and an
/// unset `--kv-cache` keep no tail.
[[nodiscard]] inline bool ValidateKvCacheSelection(std::string_view value,
                                                   std::string_view tail_text,
                                                   std::string* error) {
  if (!value.empty() && !ValidateKvCacheValue(value, error)) {
    return false;
  }
  std::uint32_t tail = 0;
  if (!ParseKvCacheTail(tail_text, &tail, error)) {
    return false;
  }
  if (tail != 0 && !hip::QwenKvCacheAcceptsTail(value)) {
    if (error != nullptr) {
      *error = "--kv-cache-tail requires a KVarN --kv-cache value";
    }
    return false;
  }
  return true;
}

/// KV formats the startup log reports for models that do not accept
/// `--kv-cache`. Qwen3.8 27B and Qwen3.8-Flash-Next keep F16; DeepSeek V4
/// Flash keeps FP8.
inline constexpr std::string_view kQwenKvFormat{"fp16"};
inline constexpr std::string_view kFlashNextKvFormat{"fp16"};
inline constexpr std::string_view kDeepSeekKvFormat{"fp8"};

/// Error text for an explicit `--kv-cache` on a model that does not offer the
/// option. The value is rejected even when it names that model's own format.
[[nodiscard]] inline std::string KvCacheUnsupportedModelError(
    std::string_view model, std::string_view format) {
  return std::string(model) + " does not support --kv-cache (KV format is '" +
         std::string(format) + "')";
}

}  // namespace gufo::cli

#endif  // GUFO_CLI_KV_CACHE_OPTIONS_HPP_
