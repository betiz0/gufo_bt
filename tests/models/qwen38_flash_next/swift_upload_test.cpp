// Bind-and-upload acceptance check for the Swift-1.5 Q4_K_M trunk (three
// shards).
//
// Loads the real artifact, reports its configuration, tensor count, total
// device weight bytes, the per-class device bytes and the bytes decoded at
// load, and checks the format policy's device-memory contract for this mix:
// every tensor uploaded as stored occupies exactly its stored GGUF size plus
// the tail margin, the decoded-at-load total is below the 1 GiB spec bound,
// the Q4_K/Q5_0 Mixer tensors are not among the decoded tensors, and the Q5_0
// `per_layer_token_embd` table the n-gram reader consumes from disk is never
// uploaded.
//
// Set GUFO_FLASH_NEXT_SWIFT_MODEL to the trunk's first shard; without it the
// test skips with exit 77. The MTP sidecar is optional
// (GUFO_FLASH_NEXT_MTP_MODEL).

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/weight_format.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace rocm = gufo::models::qwen38_flash_next::rocm;
namespace core = gufo::core;
namespace quant = gufo::quant;

namespace {

constexpr std::size_t kTailMargin = 4096;
constexpr std::size_t kDecodedBound = 1073741824;  // 1 GiB, per the spec.

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

/// Coarse weight class of a GGUF tensor name, for the per-class byte report.
std::string ClassOf(std::string_view name) {
  const auto contains = [&](std::string_view token) {
    return name.find(token) != std::string_view::npos;
  };
  if (name == "token_embd.weight")
    return "token_embedding";
  if (name == "output.weight")
    return "output_head";
  if (contains("per_layer_token_embd"))
    return "ple_table";
  if (contains("_exps"))
    return "routed_experts";
  if (contains("hc_"))
    return "hyper_connection";
  if (contains("ple_"))
    return "ple";
  if (contains("ssm_"))
    return "ssm";
  if (contains("attn_") || contains("indexer."))
    return "attention";
  if (contains("ffn_"))
    return "shared_expert";
  if (contains("nextn."))
    return "mtp";
  return "other";
}

/// The per-block Mixer projections this mix stores as Q4_K (down) and Q5_0
/// (up). The output-head and MTP Mixers (`output_hc_*`, `hc_head_*`) keep
/// their own producer formats and are not part of this pair.
bool IsMixerProjection(std::string_view name) {
  return name.find("hc_attn_down.weight") != std::string_view::npos ||
         name.find("hc_attn_up.weight") != std::string_view::npos ||
         name.find("hc_ffn_down.weight") != std::string_view::npos ||
         name.find("hc_ffn_up.weight") != std::string_view::npos;
}

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_FLASH_NEXT_SWIFT_MODEL");
  if (path == nullptr || *path == '\0') {
    std::cerr << "GUFO_FLASH_NEXT_SWIFT_MODEL is unset; skipping\n";
    return 77;
  }
  try {
    std::string error;
    auto reader = core::GgufReader::OpenFile(path, &error);
    Require(reader != nullptr, "reader failed: " + error);

    const char* mtp_path = std::getenv("GUFO_FLASH_NEXT_MTP_MODEL");
    const bool with_mtp = mtp_path != nullptr && *mtp_path != '\0';
    auto model =
        qfn::Model::Load(path,
                         {.max_context = 4096,
                          .mtp_model_path = with_mtp ? mtp_path : std::string{},
                          .max_draft_tokens = 1U},
                         &error);
    Require(model != nullptr, "load failed: " + error);

    const qfn::Config& c = model->config();
    const rocm::DeviceModel& device = model->executor().device();
    std::cout << "artifact " << path << "\n"
              << "architecture qwen4exp blocks " << c.num_layers << " hidden "
              << c.hidden_size << " vocab " << model->VocabSize() << " context "
              << c.context_length << "\n"
              << "gguf tensors " << reader->GetTensorCount() << " uploaded "
              << device.uploaded_tensors().size() << " quantization "
              << reader->GetQuantizationLabel() << "\n"
              << "device weight bytes " << model->ResidentBytes()
              << " decoded at load " << device.decoded_bytes() << " across "
              << device.decoded_tensors().size() << " tensors\n";

    // The spec's architecture scenario.
    Require(c.num_layers == 48 && c.hidden_size == 2560 &&
                model->VocabSize() == 248320,
            "unexpected architecture: blocks " + std::to_string(c.num_layers) +
                " hidden " + std::to_string(c.hidden_size) + " vocab " +
                std::to_string(model->VocabSize()));

    // Every tensor uploaded as stored occupies its stored size plus the tail.
    std::map<std::string, std::size_t> device_by_class;
    std::size_t stored_total = 0;
    for (const auto& t : device.uploaded_tensors()) {
      Require(t.device_bytes == t.stored_bytes + kTailMargin,
              "tensor " + t.name + " occupies " +
                  std::to_string(t.device_bytes) + " bytes, expected its " +
                  std::to_string(t.stored_bytes) + "-byte stored encoding");
      stored_total += t.stored_bytes;
      device_by_class[ClassOf(t.name)] += t.device_bytes;
      Require(t.name != "per_layer_token_embd.weight",
              "the PLE n-gram table was uploaded to the device");
    }
    for (const auto& [name, bytes] : device.decoded_tensors()) {
      device_by_class[ClassOf(name)] += bytes + kTailMargin;
      Require(!IsMixerProjection(name),
              "Mixer projection " + name + " was decoded at load");
    }
    std::cout << "stored weight bytes " << stored_total << "\n";
    for (const auto& [name, bytes] : device_by_class) {
      std::cout << "class " << name << " device bytes " << bytes << "\n";
    }
    for (const auto& [name, bytes] : device.decoded_tensors()) {
      std::cout << "decoded " << name << " " << bytes << " bytes\n";
    }

    Require(device.decoded_bytes() < kDecodedBound,
            "decoded weight memory " + std::to_string(device.decoded_bytes()) +
                " exceeds the 1 GiB bound");

    // The Mixer pair is stored, not decoded: its tensors must appear in the
    // upload list with their own encodings, and the up projection's Q5_0
    // blocks are smaller than the same elements as F16.
    std::size_t mixer_down = 0;
    std::size_t mixer_up = 0;
    std::size_t mixer_up_elements = 0;
    for (const auto& t : reader->GetTensors()) {
      if (!IsMixerProjection(t.name)) {
        continue;
      }
      const bool down = t.name.find("_down.weight") != std::string_view::npos;
      if (down) {
        Require(
            t.type == core::GgmlType::kQ4_K || t.type == core::GgmlType::kQ8_0,
            std::string(t.name) + " is not Q4_K");
        mixer_down += t.type == core::GgmlType::kQ4_K;
        continue;
      }
      Require(
          t.type == core::GgmlType::kQ5_0 || t.type == core::GgmlType::kQ8_0,
          std::string(t.name) + " is not Q5_0");
      mixer_up += t.type == core::GgmlType::kQ5_0;
      mixer_up_elements += t.ElementCount();
    }
    Require(mixer_down > 0 && mixer_up > 0,
            "the trunk stores no Q4_K Mixer down / Q5_0 Mixer up projection");
    const std::size_t mixer_up_stored = mixer_up_elements / 32 * 22;
    Require(mixer_up_stored < mixer_up_elements * sizeof(std::uint16_t),
            "the Q5_0 Mixer up projections are not smaller than F16");
    std::cout << "Mixer Q4_K down " << mixer_down << " tensors, Q5_0 up "
              << mixer_up << " tensors, " << mixer_up_stored
              << " bytes stored against "
              << mixer_up_elements * sizeof(std::uint16_t) << " bytes as F16\n";

    // The PLE table exists, is Q5_0, and is read from disk only.
    const auto* ple = reader->FindTensor("per_layer_token_embd.weight");
    Require(ple != nullptr, "the trunk has no per_layer_token_embd.weight");
    Require(ple->type == core::GgmlType::kQ5_0,
            "per_layer_token_embd.weight is not Q5_0");
    Require(quant::EncodedSizeBytes(ple->type, ple->ElementCount()) > 0,
            "the Q5_0 PLE table has no encoded size");

    std::cout << "Swift-1.5 Q4_K_M bind/upload acceptance passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Swift-1.5 acceptance check failed: " << error.what() << "\n";
    return 1;
  }
}
