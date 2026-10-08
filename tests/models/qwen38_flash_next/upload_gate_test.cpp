// Device upload gate: DeviceModel::Upload must revalidate every tensor's
// format against the weight-class table (weight_format.hpp) before it starts
// any allocation, name the offending tensor and format, and leave the device
// untouched. Crafted ModelWeights are used because the loader already refuses
// such a file; this covers the second, device-side line of defense.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/weight_format.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace rocm = gufo::models::qwen38_flash_next::rocm;
namespace core = gufo::core;
using GgmlType = gufo::core::GgmlType;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

/// A GGUF header with no metadata and no tensors: enough for Upload, which
/// only needs the reader for the shards of a file that passes the format
/// gate.
std::vector<std::uint8_t> EmptyGguf() {
  std::vector<std::uint8_t> blob(32, 0);
  const std::uint32_t magic = 0x46554747;
  const std::uint32_t version = 3;
  std::memcpy(blob.data(), &magic, sizeof(magic));
  std::memcpy(blob.data() + 4, &version, sizeof(version));
  return blob;
}

qfn::TensorRef Bound(std::string_view name, GgmlType type,
                     qfn::WeightClass cls) {
  return qfn::TensorRef{.data = reinterpret_cast<const void*>(0x10),
                        .type = type,
                        .name = name,
                        .cls = cls};
}

qfn::ModelWeights WeightsWith(const qfn::TensorRef& tensor) {
  qfn::ModelWeights weights;
  weights.layers.resize(1);
  weights.layers[0].ffn_gate_exps = tensor;
  return weights;
}

/// Uploads crafted weights and returns the error, asserting nothing was
/// allocated on the device.
std::string ExpectRefusal(const qfn::ModelWeights& weights,
                          const std::vector<std::uint8_t>& reader_bytes,
                          const std::string& expected) {
  std::string parse_error;
  auto reader = core::GgufReader::OpenMemory(reader_bytes.data(),
                                             reader_bytes.size(), &parse_error);
  Require(reader != nullptr, "reader failed: " + parse_error);

  std::size_t free_before = 0;
  std::size_t total = 0;
  Require(hipMemGetInfo(&free_before, &total) == hipSuccess,
          "hipMemGetInfo failed; this check needs a HIP device");
  std::string error;
  const auto device =
      rocm::DeviceModel::Upload(weights, *reader, nullptr, nullptr, &error);
  Require(device == nullptr, "Upload accepted unaccepted weights");
  std::size_t free_after = 0;
  Require(hipMemGetInfo(&free_after, &total) == hipSuccess,
          "hipMemGetInfo failed after Upload");
  Require(free_before <= free_after,
          "Upload allocated " + std::to_string(free_before - free_after) +
              " bytes before refusing the weights");
  Require(error == expected, "unexpected refusal: got \"" + error +
                                 "\", want \"" + expected + "\"");
  return error;
}

}  // namespace

int main() {
  try {
    const auto reader_bytes = EmptyGguf();

    ExpectRefusal(
        WeightsWith(Bound("token_embd.weight", GgmlType::kIQ3_S,
                          qfn::WeightClass::kDense)),
        reader_bytes,
        "tensor token_embd.weight has unsupported format IQ3_S (dense "
        "projection)");
    ExpectRefusal(
        WeightsWith(Bound("blk.0.ffn_gate_exps.weight", GgmlType::kQ6_K,
                          qfn::WeightClass::kRoutedGateUp)),
        reader_bytes,
        "tensor blk.0.ffn_gate_exps.weight has unsupported format Q6_K "
        "(routed gate/up expert)");
    ExpectRefusal(
        WeightsWith(Bound("blk.0.ffn_gate_exps.weight", GgmlType::kIQ4_NL,
                          qfn::WeightClass::kRoutedGateUp)),
        reader_bytes,
        "tensor blk.0.ffn_gate_exps.weight has unsupported format IQ4_NL "
        "(routed gate/up expert)");
    ExpectRefusal(
        WeightsWith(Bound("blk.0.ffn_down_exps.weight", GgmlType::kQ6_K,
                          qfn::WeightClass::kRoutedDown)),
        reader_bytes,
        "tensor blk.0.ffn_down_exps.weight has unsupported format Q6_K "
        "(routed down expert)");
    ExpectRefusal(WeightsWith(Bound("token_embd.weight", GgmlType::kQ5_0,
                                    qfn::WeightClass::kDense)),
                  reader_bytes,
                  "tensor token_embd.weight has unsupported format Q5_0 (dense "
                  "projection)");
    ExpectRefusal(
        WeightsWith(Bound("blk.0.ffn_gate_exps.weight", GgmlType::kQ5_0,
                          qfn::WeightClass::kRoutedGateUp)),
        reader_bytes,
        "tensor blk.0.ffn_gate_exps.weight has unsupported format Q5_0 "
        "(routed gate/up expert)");
    ExpectRefusal(WeightsWith(Bound("blk.0.hc_ffn_down.weight", GgmlType::kQ5_K,
                                    qfn::WeightClass::kMixerDownUp)),
                  reader_bytes,
                  "tensor blk.0.hc_ffn_down.weight has unsupported format Q5_K "
                  "(hyper-connection down/up projection)");
    ExpectRefusal(
        WeightsWith(Bound("per_layer_token_embd.weight", GgmlType::kQ4_K,
                          qfn::WeightClass::kPleTable)),
        reader_bytes,
        "tensor per_layer_token_embd.weight has unsupported format Q4_K "
        "(PLE n-gram table)");
    ExpectRefusal(
        WeightsWith(Bound("blk.0.ple_conv1d.weight", GgmlType::kQ8_0,
                          qfn::WeightClass::kPleConv1d)),
        reader_bytes,
        "tensor blk.0.ple_conv1d.weight has unsupported format Q8_0 (PLE "
        "conv1d)");

    // The formats this change adds pass the same gate, including the Q6_K
    // dense tensor the previous blanket check refused.
    qfn::ModelWeights accepted;
    accepted.token_embd =
        Bound("token_embd.weight", GgmlType::kQ4_K, qfn::WeightClass::kDense);
    accepted.layers.resize(1);
    accepted.layers[0].ffn_down_exps =
        Bound("blk.0.ffn_down_exps.weight", GgmlType::kIQ4_NL,
              qfn::WeightClass::kRoutedDown);
    accepted.layers[0].ffn_gate_exps =
        Bound("blk.0.ffn_gate_exps.weight", GgmlType::kQ4_K,
              qfn::WeightClass::kRoutedGateUp);
    accepted.layers[0].shexp_down =
        Bound("blk.0.ffn_down_shexp.weight", GgmlType::kQ6_K,
              qfn::WeightClass::kDense);
    accepted.layers[0].attn_out = Bound(
        "blk.0.attn_output.weight", GgmlType::kQ5_K, qfn::WeightClass::kDense);
    accepted.layers[0].hc_ffn.down =
        Bound("blk.0.hc_ffn_down.weight", GgmlType::kQ4_K,
              qfn::WeightClass::kMixerDownUp);
    accepted.layers[0].hc_ffn.up =
        Bound("blk.0.hc_ffn_up.weight", GgmlType::kQ5_0,
              qfn::WeightClass::kMixerDownUp);
    accepted.layers[0].ple_conv1d =
        Bound("blk.0.ple_conv1d.weight", GgmlType::kF16,
              qfn::WeightClass::kPleConv1d);
    accepted.ple_table = Bound("per_layer_token_embd.weight", GgmlType::kQ5_0,
                               qfn::WeightClass::kPleTable);
    // The Swift-1.5 trunk stores half its routed down-experts as Q8_0.
    accepted.layers[0].ffn_down_exps.type = GgmlType::kQ5_0;
    std::string error;
    Require(qfn::ValidateTensorFormats(accepted, &error),
            "the Swift-1.5 weight-format mix failed validation: " + error);

    std::cout << "Flash-Next device upload gate checks passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "upload gate check failed: " << error.what() << "\n";
    return 1;
  }
}
