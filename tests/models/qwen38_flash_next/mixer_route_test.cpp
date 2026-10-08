// Wide-Mixer-route guard (design D7): the executor may enable the wide Mixer
// route — the F16 norm, the tiled-Q8 norm and the Q8_0-only W8A8/fused Mixer
// down GEMMs — only when every Mixer `HcMix` can reach stores its down
// projection as Q8_0. A synthetic trunk whose Mixer down projections differ
// across blocks, or whose bound MTP sidecar Mixer is not Q8_0, must therefore
// take the per-tensor `Dense` route for every Mixer, which records no
// kMixerW8A8Down/kMixerFusedDown/kMixerFusedUp call, while an all-Q8_0 trunk
// still takes the wide route (the control case).
//
// The trunk is the sparse `gguf_fixture` artifact: the executor's routes are
// the subject here, and the numeric correctness of each projection's format
// is asserted by `mmq_ops`/`hc_mix_ops` against the FP32 reference (the
// mixed-format model case in `hc_mix_ops` runs every per-block Mixer chain
// against it).

#include <hip/hip_runtime.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/weight_format.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"
#include "tests/models/qwen38_flash_next/gguf_fixture.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace rocm = gufo::models::qwen38_flash_next::rocm;
namespace fixture = gufo::models::qwen38_flash_next::fixture;
namespace core = gufo::core;
using GgmlType = gufo::core::GgmlType;
using rocm::WeightRoute;

namespace {

/// A synthetic trunk whose Mixer formats are the case's.
struct Case {
  std::string name;
  std::vector<fixture::ClassFormat> formats;
  /// The MTP sidecar's Mixer formats when the case binds one; the sidecar
  /// itself is the fixture's `nextn` block. `HcMix` reaches it, so the wide
  /// route's guard has to look at it too.
  std::optional<std::vector<fixture::ClassFormat>> sidecar_formats;
  /// True when the trunk's own Mixer down projections are all Q8_0, so the
  /// only reason the wide route must be off is the sidecar.
  bool trunk_mixer_down_q8_0{false};
  /// True when every Mixer down projection is Q8_0, so the wide route is the
  /// expected one.
  bool wide_expected{false};
};

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

bool IsQ8OnlyMixerRoute(WeightRoute route) {
  return route == WeightRoute::kMixerW8A8Down ||
         route == WeightRoute::kMixerFusedDown ||
         route == WeightRoute::kMixerFusedUp;
}

/// Removes a fixture even when a check throws.
class TempPath {
public:
  TempPath() {
    char path[] = "/tmp/qfn-mixer-route-XXXXXX";
    const int fd = ::mkstemp(path);
    if (fd < 0) {
      throw std::runtime_error("could not create a fixture path");
    }
    ::close(fd);
    path_ = path;
  }
  ~TempPath() { ::unlink(path_.c_str()); }
  TempPath(const TempPath&) = delete;
  TempPath& operator=(const TempPath&) = delete;
  [[nodiscard]] const std::string& str() const noexcept { return path_; }

private:
  std::string path_;
};

/// Runs one synthetic trunk at `rows`-token tiled prefill and returns the
/// recorded routes.
std::vector<rocm::RouteRecord> RunCase(const Case& test_case,
                                       std::uint32_t rows) {
  fixture::Options opt;
  opt.with_ple = false;
  opt.vocab_size = 4096;  // keep the sparse fixture's extent small
  opt.class_formats = test_case.formats;
  const TempPath path;
  const std::string write_error =
      fixture::WriteTrunk(path.str(), fixture::FixtureConfig(opt), opt);
  if (!write_error.empty()) {
    throw std::runtime_error("fixture write failed: " + write_error);
  }
  std::string error;
  auto reader = core::GgufReader::OpenFile(path.str(), &error);
  if (reader == nullptr) {
    throw std::runtime_error("fixture did not open: " + error);
  }
  auto weights = qfn::ModelWeights::Bind(*reader, &error);
  if (!weights.has_value()) {
    throw std::runtime_error("fixture did not bind: " + error);
  }
  std::shared_ptr<core::GgufReader> sidecar_reader;
  std::optional<qfn::MtpWeights> sidecar;
  const TempPath sidecar_path;
  if (test_case.sidecar_formats.has_value()) {
    fixture::Options sidecar_opt;
    sidecar_opt.with_ple = false;
    sidecar_opt.vocab_size = opt.vocab_size;
    sidecar_opt.class_formats = *test_case.sidecar_formats;
    const std::string sidecar_error =
        fixture::WriteSidecar(sidecar_path.str(), weights->config, sidecar_opt);
    if (!sidecar_error.empty()) {
      throw std::runtime_error("sidecar write failed: " + sidecar_error);
    }
    sidecar_reader = core::GgufReader::OpenFile(sidecar_path.str(), &error);
    if (sidecar_reader == nullptr) {
      throw std::runtime_error("sidecar did not open: " + error);
    }
    sidecar = qfn::MtpWeights::Bind(*sidecar_reader, weights->config, &error);
    if (!sidecar.has_value()) {
      throw std::runtime_error("sidecar did not bind: " + error);
    }
  }
  auto device = rocm::DeviceModel::Upload(*weights, *reader,
                                          sidecar ? &*sidecar : nullptr,
                                          sidecar_reader.get(), &error);
  if (device == nullptr) {
    throw std::runtime_error("upload failed: " + error);
  }
  // The Mixer's own formats are the subject; the rest of the trunk keeps the
  // fixture's smallest accepted formats.
  rocm::Executor::Options options;
  options.max_batch = rows;
  options.max_logit_rows = rows;
  auto executor = rocm::Executor::Create(*device, nullptr, options, &error);
  if (executor == nullptr) {
    throw std::runtime_error("executor failed: " + error);
  }
  auto session =
      executor->CreateSession(core::SessionMode::kAutoregressive, 4096, &error);
  if (session == nullptr) {
    throw std::runtime_error("session failed: " + error);
  }
  std::vector<std::int32_t> tokens(rows);
  for (std::uint32_t i = 0; i < rows; ++i) {
    tokens[i] = static_cast<std::int32_t>(i % opt.vocab_size);
  }
  std::vector<float> logits(static_cast<std::size_t>(rows) * 4096);
  executor->EnableRouteRecording();
  executor->ResetRoutes();
  if (!executor->Forward(*session, tokens, rows, logits.data(),
                         rocm::Executor::ForwardMode::kPrefill, &error)) {
    throw std::runtime_error("prefill failed: " + error);
  }
  for (const float value : logits) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("prefill produced a non-finite logit");
    }
  }
  return executor->routes();
}

/// The pinned geometry of every fixture Mixer: `hc_*_down` is
/// `HcDim = 4 * 2560 = 10240` columns -> `hc_low_rank = 320` rows.
constexpr std::uint32_t kMixerCols = 10240;
constexpr std::uint32_t kMixerRank = 320;

/// One mixed-format scenario: no Mixer route that reads Q8_0 may be recorded
/// (the wide route's down kernels assume that encoding), and the Mixer must
/// still run — through the adapter when its own block is not Q8_0, through the
/// trunk's own Q8_0 dense route when the guard was turned off by the sidecar.
void CheckMixed(const Case& test_case, std::uint32_t rows) {
  const auto routes = RunCase(test_case, rows);
  std::size_t mixer_q8_routes = 0;
  std::size_t adapter_down = 0;
  std::size_t dense_down = 0;
  for (const auto& record : routes) {
    if (IsQ8OnlyMixerRoute(record.route)) {
      ++mixer_q8_routes;
    }
    if (record.rows == kMixerRank && record.cols == kMixerCols) {
      if (record.route == WeightRoute::kAdapterMmq ||
          record.route == WeightRoute::kAdapterGemv) {
        ++adapter_down;
      } else if (record.route == WeightRoute::kQ8W8A8 ||
                 record.route == WeightRoute::kQ8F16Gemm) {
        ++dense_down;
      }
    }
  }
  if (mixer_q8_routes != 0) {
    for (const auto& record : routes) {
      std::cerr << "  " << RouteName(record.route) << " "
                << core::ToString(record.type) << " " << record.rows << "x"
                << record.cols << " tokens " << record.tokens << "\n";
    }
    throw std::runtime_error(test_case.name + " at " + std::to_string(rows) +
                             " tokens: " + std::to_string(mixer_q8_routes) +
                             " Mixer route(s) that read Q8_0 were recorded "
                             "for a model with a non-Q8_0 Mixer down "
                             "projection");
  }
  if (test_case.trunk_mixer_down_q8_0) {
    if (dense_down == 0) {
      throw std::runtime_error(test_case.name + " at " + std::to_string(rows) +
                               " tokens: the all-Q8_0 trunk recorded no dense "
                               "Mixer down projection, so the sidecar Mixer "
                               "did not turn the wide route off");
    }
  } else if (adapter_down == 0) {
    throw std::runtime_error(test_case.name + " at " + std::to_string(rows) +
                             " tokens: no non-Q8_0 Mixer down projection took "
                             "the adapter route");
  }
}

/// The control: an all-Q8_0 trunk keeps the wide Mixer route, so the guard
/// above is a format check rather than an unconditional off switch.
void CheckControl(std::uint32_t rows, bool fused_expected) {
  Case test_case{.name = "all Q8_0",
                 .formats = {},
                 .sidecar_formats = std::nullopt,
                 .trunk_mixer_down_q8_0 = false,
                 .wide_expected = true};
  const auto routes = RunCase(test_case, rows);
  std::size_t w8a8 = 0;
  std::size_t fused_down = 0;
  std::size_t fused_up = 0;
  for (const auto& record : routes) {
    w8a8 += record.route == WeightRoute::kMixerW8A8Down;
    fused_down += record.route == WeightRoute::kMixerFusedDown;
    fused_up += record.route == WeightRoute::kMixerFusedUp;
  }
  if (fused_expected) {
    if (fused_down == 0 || fused_down != fused_up) {
      throw std::runtime_error("all-Q8_0 trunk at " + std::to_string(rows) +
                               " tokens recorded " +
                               std::to_string(fused_down) + " fused down and " +
                               std::to_string(fused_up) + " fused up routes");
    }
  } else if (w8a8 == 0) {
    throw std::runtime_error("all-Q8_0 trunk at " + std::to_string(rows) +
                             " tokens recorded no W8A8 Mixer down route");
  }
  std::cout << "all Q8_0 at " << rows << " tokens: mixer-w8a8 " << w8a8
            << " fused-down " << fused_down << " fused-up " << fused_up << "\n";
}

/// Block 0 Q8_0 / block 1 Q4_K down with that block's up as Q8_0, and
/// separately as Q5_0; block 0 Q4_K / block 1 Q8_0; and the two sidecar
/// directions. Pre-change the wide route was enabled from block 0's
/// feed-forward Mixer alone and the sidecar was not consulted at all, so the
/// first two, the fourth and especially the last case (an all-Q8_0 trunk the
/// sidecar has to turn off) are exactly the ones the old condition
/// mis-enabled.
std::vector<Case> MixedCases() {
  const auto mix_formats = [](const std::string& block, bool up_q5_0) {
    // Suffix entries precede the class-wide default: the fixture matches
    // them in order, so a class-wide entry first would shadow them.
    std::vector<fixture::ClassFormat> formats{
        {qfn::WeightClass::kMixerDownUp, GgmlType::kQ4_K,
         block + "hc_attn_down.weight"},
        {qfn::WeightClass::kMixerDownUp, GgmlType::kQ4_K,
         block + "hc_ffn_down.weight"}};
    if (up_q5_0) {
      formats.push_back({qfn::WeightClass::kMixerDownUp, GgmlType::kQ5_0,
                         block + "hc_attn_up.weight"});
      formats.push_back({qfn::WeightClass::kMixerDownUp, GgmlType::kQ5_0,
                         block + "hc_ffn_up.weight"});
    }
    formats.push_back({qfn::WeightClass::kMixerDownUp, GgmlType::kQ8_0, ""});
    return formats;
  };
  const auto sidecar_q4k_down = [] {
    return std::vector<fixture::ClassFormat>{
        {qfn::WeightClass::kMixerDownUp, GgmlType::kQ4_K, "_down.weight"},
        {qfn::WeightClass::kMixerDownUp, GgmlType::kQ8_0, ""}};
  };
  const auto sidecar_q8_0 = [] {
    return std::vector<fixture::ClassFormat>{
        {qfn::WeightClass::kMixerDownUp, GgmlType::kQ8_0, ""}};
  };
  std::vector<Case> cases;
  cases.push_back({"block 0 Q8_0 / block 1 Q4_K, up Q8_0",
                   mix_formats("blk.1.", false), std::nullopt, false, false});
  cases.push_back({"block 0 Q8_0 / block 1 Q4_K, up Q5_0",
                   mix_formats("blk.1.", true), std::nullopt, false, false});
  cases.push_back({"block 0 Q4_K / block 1 Q8_0", mix_formats("blk.0.", false),
                   std::nullopt, false, false});
  // A Q4_K trunk with a Q8_0 sidecar stays off for the trunk's own sake.
  cases.push_back({"Q4_K trunk with a Q8_0 sidecar",
                   mix_formats("blk.0.", true), sidecar_q8_0(), false, false});
  // An all-Q8_0 trunk with a Q4_K sidecar is the only case that reaches the
  // guard's sidecar branch: without it the wide route would be enabled from
  // the trunk alone and read the sidecar's Q4_K bytes as Q8_0.
  cases.push_back(
      {"Q8_0 trunk with a Q4_K sidecar", {}, sidecar_q4k_down(), true, false});
  return cases;
}

}  // namespace

int main() {
  try {
    for (const Case& test_case : MixedCases()) {
      for (const std::uint32_t rows : {64U, 128U}) {
        CheckMixed(test_case, rows);
      }
      std::cout << test_case.name << ": no Q8_0 Mixer route at 64 or 128 "
                << "tokens\n";
    }
    CheckControl(64, /*fused_expected=*/false);
    CheckControl(128, /*fused_expected=*/true);
    std::cout << "Flash-Next wide-Mixer-route guard checks passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "wide-Mixer-route guard check failed: " << error.what()
              << "\n";
    return 1;
  }
}
