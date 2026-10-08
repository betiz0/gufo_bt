// Bind-and-upload acceptance check for the Agention AP-Q4_K_XL trunk.
//
// Loads the real artifact, reports its configuration, tensor count, total
// device weight bytes, the per-class device bytes and the bytes decoded at
// load, and checks the format policy's device-memory contract: every tensor
// uploaded as stored occupies exactly its stored GGUF size plus the tail
// margin, the decoded total is below the 1 GiB spec bound, and the IQ4_NL
// routed down-experts occupy fewer bytes than the same tensors in Q8_0.
//
// Set GUFO_FLASH_NEXT_AP_MODEL to the trunk path; without it the test skips
// with exit 77. The MTP sidecar is optional (GUFO_FLASH_NEXT_MTP_MODEL).

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
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

}  // namespace

int main() {
  const char* path = std::getenv("GUFO_FLASH_NEXT_AP_MODEL");
  if (path == nullptr || *path == '\0') {
    std::cerr << "GUFO_FLASH_NEXT_AP_MODEL is unset; skipping\n";
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
    }
    for (const auto& [name, bytes] : device.decoded_tensors()) {
      device_by_class[ClassOf(name)] += bytes + kTailMargin;
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

    // The routed down-experts are IQ4_NL: fewer bytes than a Q8_0 copy of the
    // same tensors, which is what "the stored encoding is executed" means.
    std::size_t down_stored = 0;
    std::size_t down_elements = 0;
    for (const auto& t : reader->GetTensors()) {
      if (t.name.find("ffn_down_exps.weight") == std::string_view::npos) {
        continue;
      }
      Require(t.type == core::GgmlType::kIQ4_NL,
              std::string(t.name) + " is not IQ4_NL");
      down_elements += t.ElementCount();
      down_stored += quant::EncodedSizeBytes(t.type, t.ElementCount());
    }
    const std::size_t q8_bytes = down_elements / 32 * 34;
    Require(down_stored < q8_bytes,
            "IQ4_NL routed down-experts are not smaller than Q8_0");
    std::cout << "routed down-experts " << down_stored << " bytes stored vs "
              << q8_bytes << " bytes as Q8_0\n";
    Require(down_stored > 0, "no routed down-expert tensors found");

    std::cout << "AP-Q4_K_XL bind/upload acceptance passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "AP acceptance check failed: " << error.what() << "\n";
    return 1;
  }
}
