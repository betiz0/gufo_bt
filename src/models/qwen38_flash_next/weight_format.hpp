#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHT_FORMAT_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHT_FORMAT_HPP_

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen38_flash_next {

/// Weight classes of the qwen4exp trunk. Format legality is a property of the
/// pair (class, GGUF type), never of the file or its producer: the CPU bind
/// step and the device uploader both validate against this one table, so a
/// file the loader accepts is exactly one the kernels can execute.
///
/// Routed experts are two classes because gate/up and down projections have
/// different kernel coverage: only the down projection has an IQ4_NL
/// specialization, and no routed kernel reads Q6_K.
enum class WeightClass : std::uint8_t {
  kDense = 0,     ///< Dense projections, token embedding and output head.
  kRoutedGateUp,  ///< `ffn_gate_exps` / `ffn_up_exps`.
  kRoutedDown,    ///< `ffn_down_exps`.
  kMixerDownUp,   ///< Hyper-connection down/up projections.
  kMixerInject,   ///< Hyper-connection inject projection.
  kSsmGate,       ///< `ssm_alpha` / `ssm_beta`.
  kPleTable,      ///< `per_layer_token_embd` (n-gram hash table).
  kPleConv1d,     ///< `ple_conv1d`.
  kF32Only,       ///< Norm/scale/bias vectors and the F32 router rows.
};

/// The single weight-class format table: the accepted `GgmlType`s per class.
namespace formats {

inline constexpr core::GgmlType kDense[] = {
    core::GgmlType::kQ4_K, core::GgmlType::kQ5_K, core::GgmlType::kQ8_0,
    core::GgmlType::kQ6_K, core::GgmlType::kBF16, core::GgmlType::kF16,
    core::GgmlType::kF32};
inline constexpr core::GgmlType kRoutedGateUp[] = {
    core::GgmlType::kQ4_K, core::GgmlType::kQ5_K, core::GgmlType::kQ5_1,
    core::GgmlType::kQ8_0};
inline constexpr core::GgmlType kRoutedDown[] = {
    core::GgmlType::kQ4_K, core::GgmlType::kQ5_K, core::GgmlType::kQ5_1,
    core::GgmlType::kQ5_0, core::GgmlType::kQ8_0, core::GgmlType::kIQ4_NL};
inline constexpr core::GgmlType kMixerDownUp[] = {
    core::GgmlType::kQ4_K, core::GgmlType::kQ5_0,   core::GgmlType::kQ8_0,
    core::GgmlType::kQ6_K, core::GgmlType::kIQ4_NL, core::GgmlType::kBF16,
    core::GgmlType::kF16,  core::GgmlType::kF32};
inline constexpr core::GgmlType kMixerInject[] = {
    core::GgmlType::kF32, core::GgmlType::kBF16, core::GgmlType::kQ8_0,
    core::GgmlType::kQ6_K};
inline constexpr core::GgmlType kSsmGate[] = {
    core::GgmlType::kF32, core::GgmlType::kBF16, core::GgmlType::kQ8_0,
    core::GgmlType::kQ6_K};
inline constexpr core::GgmlType kPleTable[] = {
    core::GgmlType::kQ5_0, core::GgmlType::kIQ4_NL, core::GgmlType::kBF16};
inline constexpr core::GgmlType kPleConv1d[] = {core::GgmlType::kF32,
                                                core::GgmlType::kF16};
/// Norm gammas, convolution kernels, DT/A biases, router rows and the
/// shared-expert gate row: read as F32 by the fused epilogues and the
/// hipBLAS router GEMM, never dequantized.
inline constexpr core::GgmlType kF32Only[] = {core::GgmlType::kF32};

}  // namespace formats

/// The formats accepted for `cls`, in no particular order.
[[nodiscard]] inline std::span<const core::GgmlType> AcceptedFormats(
    WeightClass cls) noexcept {
  switch (cls) {
    case WeightClass::kDense:
      return formats::kDense;
    case WeightClass::kRoutedGateUp:
      return formats::kRoutedGateUp;
    case WeightClass::kRoutedDown:
      return formats::kRoutedDown;
    case WeightClass::kMixerDownUp:
      return formats::kMixerDownUp;
    case WeightClass::kMixerInject:
      return formats::kMixerInject;
    case WeightClass::kSsmGate:
      return formats::kSsmGate;
    case WeightClass::kPleTable:
      return formats::kPleTable;
    case WeightClass::kPleConv1d:
      return formats::kPleConv1d;
    case WeightClass::kF32Only:
      return formats::kF32Only;
  }
  return {};
}

[[nodiscard]] inline bool FormatAccepted(WeightClass cls,
                                         core::GgmlType type) noexcept {
  for (const auto accepted : AcceptedFormats(cls)) {
    if (accepted == type) {
      return true;
    }
  }
  return false;
}

/// Human-readable class name for diagnostics.
[[nodiscard]] inline const char* WeightClassName(WeightClass cls) noexcept {
  switch (cls) {
    case WeightClass::kDense:
      return "dense projection";
    case WeightClass::kRoutedGateUp:
      return "routed gate/up expert";
    case WeightClass::kRoutedDown:
      return "routed down expert";
    case WeightClass::kMixerDownUp:
      return "hyper-connection down/up projection";
    case WeightClass::kMixerInject:
      return "hyper-connection inject projection";
    case WeightClass::kSsmGate:
      return "SSM gate projection";
    case WeightClass::kPleTable:
      return "PLE n-gram table";
    case WeightClass::kPleConv1d:
      return "PLE conv1d";
    case WeightClass::kF32Only:
      return "F32-only tensor";
  }
  return "unknown weight class";
}

/// The single refusal message of the loader and the device uploader: it names
/// the offending tensor, its GGUF format, and the class it failed.
[[nodiscard]] inline std::string UnsupportedFormatMessage(
    std::string_view tensor, core::GgmlType type, WeightClass cls) {
  return "tensor " + std::string(tensor) + " has unsupported format " +
         std::string(core::ToString(type)) + " (" + WeightClassName(cls) + ")";
}

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_WEIGHT_FORMAT_HPP_
