// Route pinning: every dense, Mixer and routed projection must take the same
// executor route the parent commit (d9a84f1) took for the formats a file
// already used. The expected route is derived per call from the parent's
// dispatch conditions and the call's own (format, shape, batch class), so the
// check is an equality, not a membership test: a Q8_0 dense projection is
// required to take exactly the route its shape selected before this change.
//
// The formats this change adds (Q6_K/IQ4_NL dense, IQ4_NL routed down) had no
// pre-change route; they must take the adapter route under the same
// batch-class rules.
//
// Recording is opt-in (design D8): the check first proves that an ordinary
// session records nothing, then enables recording for the pinned runs, so a
// production `prompt`/`chat`/`serve` executor never grows the log.
//
// Set GUFO_FLASH_NEXT_MODEL to the target's first shard; without it the test
// skips with exit 77.

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace rocm = gufo::models::qwen38_flash_next::rocm;
namespace core = gufo::core;
using GgmlType = gufo::core::GgmlType;
using rocm::WeightRoute;

namespace {

// The parent commit's dense-route constants (executor.cpp).
constexpr std::uint32_t kDenseF16MinRows = 2048;
constexpr std::uint32_t kDenseF16MaxCols = 2560;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

const char* RouteName(WeightRoute route) {
  switch (route) {
    case WeightRoute::kSmall:
      return "small";
    case WeightRoute::kQ8Gemv:
      return "q8-gemv";
    case WeightRoute::kQ8F16Gemm:
      return "q8-f16-gemm";
    case WeightRoute::kQ8W8A8:
      return "q8-w8a8";
    case WeightRoute::kAdapterMmq:
      return "adapter-mmq";
    case WeightRoute::kAdapterGemv:
      return "adapter-gemv";
    case WeightRoute::kBlasGemm:
      return "blas-gemm";
    case WeightRoute::kMixerFusedDown:
      return "mixer-fused-down";
    case WeightRoute::kMixerW8A8Down:
      return "mixer-w8a8-down";
    case WeightRoute::kMixerFusedUp:
      return "mixer-fused-up";
    case WeightRoute::kMoeWmma:
      return "moe-wmma";
    case WeightRoute::kMoeAdapter:
      return "moe-adapter";
    case WeightRoute::kMoeVector:
      return "moe-vector";
  }
  return "unknown";
}

bool IsRouted(WeightRoute route) {
  return route == WeightRoute::kMoeWmma || route == WeightRoute::kMoeAdapter ||
         route == WeightRoute::kMoeVector;
}

bool IsMixer(WeightRoute route) {
  return route == WeightRoute::kMixerFusedDown ||
         route == WeightRoute::kMixerW8A8Down ||
         route == WeightRoute::kMixerFusedUp;
}

/// The route the parent commit selected for one dense projection call.
WeightRoute ExpectedDense(GgmlType type, bool wide, std::uint32_t rows,
                          std::uint32_t cols, std::size_t max_half_cols) {
  switch (type) {
    case GgmlType::kQ8_0:
      if (!wide) {
        return WeightRoute::kQ8Gemv;
      }
      return (rows >= kDenseF16MinRows && cols <= kDenseF16MaxCols &&
              cols <= max_half_cols)
                 ? WeightRoute::kQ8F16Gemm
                 : WeightRoute::kQ8W8A8;
    case GgmlType::kQ4_K:
    case GgmlType::kQ5_K:
    case GgmlType::kQ5_0:
    case GgmlType::kQ6_K:
    case GgmlType::kIQ4_NL:
      return wide ? WeightRoute::kAdapterMmq : WeightRoute::kAdapterGemv;
    case GgmlType::kF32:
    case GgmlType::kF16:
    case GgmlType::kBF16:
      return wide ? WeightRoute::kBlasGemm : WeightRoute::kSmall;
    default:
      break;
  }
  throw std::runtime_error("unexpected dense format " +
                           std::string(core::ToString(type)));
}

/// Per-layer routed expert formats read from the file header.
struct LayerFormats {
  GgmlType gate{GgmlType::kF32};
  GgmlType up{GgmlType::kF32};
  GgmlType down{GgmlType::kF32};
  bool seen{false};
};

/// The tiled routed decision for one layer: the formats whose tiled kernel
/// exists, unchanged for the pre-change formats.
bool WmmaRouted(const LayerFormats& l, const qfn::Config& c) {
  return l.seen && (l.gate == GgmlType::kQ4_K || l.gate == GgmlType::kQ5_K) &&
         l.up == l.gate &&
         (l.down == GgmlType::kQ5_1 || l.down == GgmlType::kQ5_0 ||
          l.down == GgmlType::kQ8_0 || l.down == GgmlType::kIQ4_NL) &&
         c.hidden_size % 256 == 0 && c.expert_ff % 64 == 0;
}

/// Tensors the uploader decodes to F16/F32 (or never uploads): their stored
/// Q6_K/IQ4_NL never reaches a dense kernel, so they do not imply the adapter
/// route. Mirrors `device_model.cpp`'s decode set.
bool DecodedAtLoad(std::string_view name) {
  constexpr std::string_view kDecoded[] = {
      "ssm_alpha.weight",        "ssm_beta.weight",
      "ple_conv1d.weight",       "per_layer_token_embd.weight",
      "hc_attn_inject.weight",   "hc_attn_down.weight",
      "hc_attn_up.weight",       "hc_ffn_inject.weight",
      "hc_ffn_down.weight",      "hc_ffn_up.weight",
      "output_hc_inject.weight", "output_hc_down.weight",
      "output_hc_up.weight",     "hc_head_inject.weight",
      "hc_head_down.weight",     "hc_head_up.weight",
  };
  for (const std::string_view suffix : kDecoded) {
    if (name.ends_with(suffix)) {
      return true;
    }
  }
  return false;
}

/// Parses `blk.<N>.ffn_{gate,up,down}_exps.weight`; returns false otherwise.
bool ParseExpertName(std::string_view name, std::size_t* layer, int* kind) {
  constexpr std::string_view kSuffixes[] = {
      "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
  for (int k = 0; k < 3; ++k) {
    const std::string_view suffix = kSuffixes[k];
    if (!name.ends_with(suffix)) {
      continue;
    }
    const std::string_view prefix = name.substr(0, name.size() - suffix.size());
    if (!prefix.starts_with("blk.") || !prefix.ends_with(".")) {
      return false;
    }
    const std::string_view digits = prefix.substr(4, prefix.size() - 5);
    if (digits.empty()) {
      return false;
    }
    std::size_t value = 0;
    for (const char ch : digits) {
      if (ch < '0' || ch > '9') {
        return false;
      }
      value = value * 10 + static_cast<std::size_t>(ch - '0');
    }
    *layer = value;
    *kind = k;
    return true;
  }
  return false;
}

struct Summary {
  std::size_t dense = 0;
  std::size_t routed = 0;
  std::size_t mixer = 0;
  std::size_t mixer_down = 0;
  std::size_t mixer_up = 0;
  std::vector<std::string> pairs;
};

/// Checks every recorded call against the parent commit's route for its
/// (format, shape, batch class), and the Mixer/routed rules.
Summary Check(const std::vector<rocm::RouteRecord>& routes,
              const std::map<std::size_t, LayerFormats>& layers,
              const qfn::Config& c, std::size_t max_half_cols, bool wide,
              bool wide_mixer, const std::string& label) {
  Summary summary;
  std::size_t routed_index = 0;
  std::size_t routed_tiled = 0;
  std::size_t routed_decode = 0;
  // A mismatch prints the whole record so the failing call can be placed in
  // the batch it belongs to.
  const auto fail = [&](const std::string& message) {
    std::cerr << label << ": " << routes.size() << " recorded calls:\n";
    for (const auto& r : routes) {
      std::cerr << "  " << RouteName(r.route) << " " << core::ToString(r.type)
                << " " << r.rows << "x" << r.cols << " tokens " << r.tokens
                << "\n";
    }
    throw std::runtime_error(message);
  };
  const auto require = [&](bool condition, const std::string& message) {
    if (!condition) {
      fail(message);
    }
  };
  for (const auto& r : routes) {
    const std::string pair = std::string(RouteName(r.route)) + "/" +
                             std::string(core::ToString(r.type));
    summary.pairs.push_back(pair);
    // The recorded batch class is the parent's MatrixRows(n): the prefill
    // phase or more than the vector-width tokens. It is per call, so a
    // prefill's logit head (a short decode-phase batch) is checked as decode.
    const bool call_wide = r.tiled;
    if (IsRouted(r.route)) {
      ++summary.routed;
      if (!call_wide) {
        ++routed_decode;
        require(r.route == WeightRoute::kMoeVector,
                label + ": routed experts took " + RouteName(r.route) +
                    " at decode");
        continue;
      }
      ++routed_tiled;
      const auto it = layers.find(routed_index);
      require(it != layers.end(),
              label + ": more routed calls than trunk layers");
      const WeightRoute expected = WmmaRouted(it->second, c)
                                       ? WeightRoute::kMoeWmma
                                       : WeightRoute::kMoeAdapter;
      require(r.route == expected,
              label + ": layer " + std::to_string(routed_index) + " down " +
                  std::string(core::ToString(r.type)) + " took " +
                  RouteName(r.route) + ", expected " + RouteName(expected));
      ++routed_index;
      continue;
    }
    if (IsMixer(r.route)) {
      ++summary.mixer;
      if (r.route == WeightRoute::kMixerFusedDown) {
        ++summary.mixer_down;
      } else if (r.route == WeightRoute::kMixerFusedUp) {
        ++summary.mixer_up;
      }
      // The fused down/up pair is the parent's wide route for a Q8_0-mixer
      // file; a decode batch or a file with decoded Mixer weights records no
      // Mixer route at all (its projections go through Dense).
      require(wide && wide_mixer,
              label + ": " + RouteName(r.route) +
                  " recorded outside the wide Q8_0-mixer route");
      require(r.route == WeightRoute::kMixerFusedDown ||
                  r.route == WeightRoute::kMixerFusedUp,
              label + ": unexpected Mixer route " + RouteName(r.route));
      continue;
    }
    ++summary.dense;
    const WeightRoute expected =
        ExpectedDense(r.type, call_wide, r.rows, r.cols, max_half_cols);
    require(r.route == expected,
            label + ": " + std::string(core::ToString(r.type)) + " " +
                std::to_string(r.rows) + "x" + std::to_string(r.cols) +
                " tokens " + std::to_string(r.tokens) + " took " +
                RouteName(r.route) + ", expected " + RouteName(expected));
  }
  // Each batch class records exactly one routed down projection per layer; the
  // prefill's logit head is a decode batch and records none.
  require(routed_tiled == (wide ? c.num_layers : 0) &&
              routed_decode == (wide ? 0 : c.num_layers) &&
              routed_index == routed_tiled,
          label + ": recorded " + std::to_string(routed_tiled) + " tiled and " +
              std::to_string(routed_decode) +
              " decode routed calls, expected " +
              std::to_string(wide ? c.num_layers : 0) + " tiled and " +
              std::to_string(wide ? 0 : c.num_layers) + " decode");
  require(summary.mixer_down == summary.mixer_up,
          label + ": " + std::to_string(summary.mixer_down) +
              " fused Mixer down routes against " +
              std::to_string(summary.mixer_up) + " up routes");
  return summary;
}

}  // namespace

/// Pins the routes of one artifact. The file's own formats decide which
/// routes are expected, so every accepted mix runs the same checks.
void CheckArtifact(const std::string& path, const std::string& mtp_path,
                   const std::string& tag) {
  std::string error;
  auto reader = core::GgufReader::OpenFile(path.c_str(), &error);
  Require(reader != nullptr, "reader failed: " + error);
  // The file's own formats decide which routes are expected. The PLE
  // convolution is `blk.<N>.ple_conv1d.weight`, so only a suffix match sees
  // it; a prefix mismatch silently disabled the old detection.
  std::map<std::size_t, LayerFormats> layers;
  GgmlType ple_conv{GgmlType::kF32};
  bool has_ple_conv = false;
  bool has_q6_dense = false;
  bool has_iq4_down = false;
  // A dense or Mixer tensor stored in a format only the vendored adapter
  // executes keeps the file on the adapter route: the Q4_K/Q5_K/Q5_0 dense
  // projections and the Q4_K/Q5_0 Mixer down/up pair. The Mixer weights
  // stored as Q6_K/IQ4_NL are still decoded to F16 at load, so they reach
  // no dense kernel and record nothing.
  bool has_adapter_format = false;
  for (const auto& t : reader->GetTensors()) {
    if (t.name.ends_with("ple_conv1d.weight")) {
      has_ple_conv = true;
      ple_conv = t.type;
    }
    std::size_t layer = 0;
    int kind = 0;
    if (ParseExpertName(t.name, &layer, &kind)) {
      LayerFormats& l = layers[layer];
      l.seen = true;
      if (kind == 0) {
        l.gate = t.type;
      } else if (kind == 1) {
        l.up = t.type;
      } else {
        l.down = t.type;
        has_iq4_down = has_iq4_down || t.type == GgmlType::kIQ4_NL;
      }
      continue;
    }
    // A Q6_K tensor that stays on the device takes the adapter dense path; a
    // decoded one (Mixer/norm-side) never reaches a dense kernel.
    if (DecodedAtLoad(t.name)) {
      has_adapter_format = has_adapter_format || t.type == GgmlType::kQ4_K ||
                           t.type == GgmlType::kQ5_0;
      continue;
    }
    // The embedding lookup reads its rows itself, never through the
    // adapter GEMM.
    if (t.name != "token_embd.weight") {
      has_adapter_format = has_adapter_format || t.type == GgmlType::kQ4_K ||
                           t.type == GgmlType::kQ5_K ||
                           t.type == GgmlType::kQ5_0;
    }
    has_q6_dense = has_q6_dense || t.type == GgmlType::kQ6_K;
  }
  Require(!layers.empty(), "no routed expert tensors in the file");
  const bool expects_adapter =
      has_q6_dense || has_iq4_down || has_adapter_format;

  auto model = qfn::Model::Load(path,
                                {.max_context = 4096,
                                 .mtp_model_path = mtp_path,
                                 .max_draft_tokens = 1U,
                                 .vision_model_path = std::string{}},
                                &error);
  Require(model != nullptr, "load failed: " + error);
  const qfn::Config& c = model->config();
  const rocm::Executor& executor = model->executor();
  const std::size_t max_half_cols = executor.device().max_half_cols();
  // The executor's wide-Mixer condition (design D7): every Mixer it can
  // reach — each trunk block's, the output head's and, when bound, the MTP
  // draft block's — stores its down projection as Q8_0.
  const auto& device_layers = executor.device().layers();
  bool device_wide_mixer =
      c.hc_count == 4 && c.hidden_size % 32 == 0 && !device_layers.empty() &&
      executor.device().hc_head().down.type == GgmlType::kQ8_0;
  for (const auto& layer : device_layers) {
    device_wide_mixer = device_wide_mixer &&
                        layer.hc_attn.down.type == GgmlType::kQ8_0 &&
                        layer.hc_ffn.down.type == GgmlType::kQ8_0;
  }
  if (executor.device().has_mtp()) {
    const auto& mtp_layer = executor.device().mtp();
    device_wide_mixer = device_wide_mixer &&
                        mtp_layer.hc_attn.down.type == GgmlType::kQ8_0 &&
                        mtp_layer.hc_ffn.down.type == GgmlType::kQ8_0 &&
                        mtp_layer.nextn_head.down.type == GgmlType::kQ8_0;
  }

  auto session =
      model->CreateSession(core::SessionMode::kAutoregressive, 4096, &error);
  Require(session != nullptr, "session failed: " + error);

  const auto prompt = model->Tokenize("The capital of France is");
  Require(session->Sync(prompt, &error), "prefill failed: " + error);
  // Recording is off until a test enables it (design D8): an ordinary
  // prefill must leave the log empty.
  Require(executor.routes().empty(),
          "route log grew before a test enabled recording");
  executor.EnableRouteRecording();

  // Decode: one token through the vector and GEMV routes.
  executor.ResetRoutes();
  Require(session->Evaluate(prompt.back(), &error), "decode failed: " + error);
  const auto decode_routes = executor.routes();
  const Summary decode =
      Check(decode_routes, layers, c, max_half_cols, /*wide=*/false,
            device_wide_mixer, "decode (1 token)");
  Require(decode.mixer == 0,
          "decode recorded " + std::to_string(decode.mixer) +
              " Mixer routes; the decode Mixer runs through Dense");
  std::cout << tag << " decode (1 token): dense " << decode.dense << " mixer "
            << decode.mixer << " routed " << decode.routed << " [";
  for (const auto& pair : decode.pairs)
    std::cout << pair << ' ';
  std::cout << "]\n";

  // Tiled prefill: 128 tokens through the wide routes.
  std::vector<std::int32_t> wide_tokens(128);
  for (std::size_t i = 0; i < wide_tokens.size(); ++i) {
    wide_tokens[i] = prompt[i % prompt.size()];
  }
  auto wide_session =
      model->CreateSession(core::SessionMode::kAutoregressive, 4096, &error);
  Require(wide_session != nullptr, "session failed: " + error);
  executor.ResetRoutes();
  Require(wide_session->Sync(wide_tokens, &error),
          "tiled prefill failed: " + error);
  const auto wide_routes = executor.routes();
  const Summary tiled =
      Check(wide_routes, layers, c, max_half_cols,
            /*wide=*/true, device_wide_mixer, "tiled (128 tokens)");
  std::cout << tag << " tiled (128 tokens): dense " << tiled.dense << " mixer "
            << tiled.mixer << " routed " << tiled.routed << " [";
  for (const auto& pair : tiled.pairs)
    std::cout << pair << ' ';
  std::cout << "]\n";
  if (device_wide_mixer) {
    Require(tiled.mixer >= 2 && tiled.mixer_down == tiled.mixer_up,
            "tiled prefill recorded " + std::to_string(tiled.mixer) +
                " Mixer routes, expected the fused down/up pair");
  } else {
    Require(tiled.mixer == 0,
            "tiled prefill recorded " + std::to_string(tiled.mixer) +
                " Mixer routes for a file whose Mixer weights are not Q8_0");
  }

  // The adapter route appears exactly when the file stores a format that
  // only it executes: Q6_K dense weights or IQ4_NL routed down-experts.
  const auto uses_adapter = [](const std::vector<std::string>& pairs) {
    for (const auto& pair : pairs) {
      if (pair.find("adapter") != std::string::npos) {
        return true;
      }
    }
    return false;
  };
  Require(uses_adapter(tiled.pairs) == expects_adapter,
          "tiled adapter-route use disagrees with the file's stored formats");
  Require(uses_adapter(decode.pairs) == expects_adapter,
          "decode adapter-route use disagrees with the file's stored "
          "formats");

  // The PLE convolution's non-F32 form is decoded at load, so the device
  // tensor the kernel reads is always F32 (the suffix match above is what
  // locates it).
  if (has_ple_conv && ple_conv != GgmlType::kF32) {
    Require(!device_layers.empty() &&
                device_layers[0].ple_conv1d.type == GgmlType::kF32,
            "a non-F32 ple_conv1d was not decoded to F32 at load");
  }
  std::cout << tag << ": Flash-Next route pinning passed.\n";
}

int main() {
  const char* path = std::getenv("GUFO_FLASH_NEXT_MODEL");
  if (path == nullptr || *path == '\0') {
    std::cerr << "GUFO_FLASH_NEXT_MODEL is unset; skipping\n";
    return 77;
  }
  const char* mtp = std::getenv("GUFO_FLASH_NEXT_MTP_MODEL");
  const std::string sidecar = mtp != nullptr ? mtp : std::string{};
  const char* ap = std::getenv("GUFO_FLASH_NEXT_AP_MODEL");
  const char* swift = std::getenv("GUFO_FLASH_NEXT_SWIFT_MODEL");
  try {
    CheckArtifact(path, sidecar, "UD-Q4_K_XL");
    if (ap != nullptr && *ap != '\0') {
      CheckArtifact(ap, sidecar, "AP-Q4_K_XL");
    }
    if (swift != nullptr && *swift != '\0') {
      CheckArtifact(swift, sidecar, "Swift-1.5 Q4_K_M");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "route check failed: " << error.what() << "\n";
    return 1;
  }
}
