// Weight-class format policy: the accepted/rejected matrix, the AP-Q4_K_XL
// per-class histogram, the shared refusal message, and the loader's behavior
// on a synthetic `qwen4exp` fixture whose formats are overridden one at a
// time. Runs without a GPU or a model file (the fixture payloads are sparse).

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/models/qwen38_flash_next/weight_format.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"
#include "tests/models/qwen38_flash_next/gguf_fixture.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace fixture = gufo::models::qwen38_flash_next::fixture;
namespace core = gufo::core;
using GgmlType = gufo::core::GgmlType;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

/// Every format this runtime has an opinion about, including ones no qwen4exp
/// kernel executes.
const std::vector<std::pair<GgmlType, const char*>> kProbed = {
    {GgmlType::kF32, "F32"},     {GgmlType::kF16, "F16"},
    {GgmlType::kBF16, "BF16"},   {GgmlType::kQ8_0, "Q8_0"},
    {GgmlType::kQ6_K, "Q6_K"},   {GgmlType::kQ4_K, "Q4_K"},
    {GgmlType::kQ5_K, "Q5_K"},   {GgmlType::kQ5_1, "Q5_1"},
    {GgmlType::kQ5_0, "Q5_0"},   {GgmlType::kIQ4_NL, "IQ4_NL"},
    {GgmlType::kIQ3_S, "IQ3_S"}, {GgmlType::kIQ2_XXS, "IQ2_XXS"},
};

/// The accepted set of every class, as the specification states it.
void CheckAcceptedSets() {
  const std::vector<std::pair<qfn::WeightClass, std::vector<GgmlType>>> table =
      {
          {qfn::WeightClass::kDense,
           {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ8_0, GgmlType::kQ6_K,
            GgmlType::kBF16, GgmlType::kF16, GgmlType::kF32}},
          {qfn::WeightClass::kRoutedGateUp,
           {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ5_1,
            GgmlType::kQ8_0}},
          {qfn::WeightClass::kRoutedDown,
           {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ5_1, GgmlType::kQ5_0,
            GgmlType::kQ8_0, GgmlType::kIQ4_NL}},
          {qfn::WeightClass::kMixerDownUp,
           {GgmlType::kQ4_K, GgmlType::kQ5_0, GgmlType::kQ8_0, GgmlType::kQ6_K,
            GgmlType::kIQ4_NL, GgmlType::kBF16, GgmlType::kF16,
            GgmlType::kF32}},
          {qfn::WeightClass::kMixerInject,
           {GgmlType::kF32, GgmlType::kBF16, GgmlType::kQ8_0, GgmlType::kQ6_K}},
          {qfn::WeightClass::kSsmGate,
           {GgmlType::kF32, GgmlType::kBF16, GgmlType::kQ8_0, GgmlType::kQ6_K}},
          {qfn::WeightClass::kPleTable,
           {GgmlType::kQ5_0, GgmlType::kIQ4_NL, GgmlType::kBF16}},
          {qfn::WeightClass::kPleConv1d, {GgmlType::kF32, GgmlType::kF16}},
          {qfn::WeightClass::kF32Only, {GgmlType::kF32}},
      };
  for (const auto& [cls, accepted] : table) {
    for (const auto& [type, name] : kProbed) {
      bool expected = false;
      for (const auto candidate : accepted) {
        expected = expected || candidate == type;
      }
      Require(
          qfn::FormatAccepted(cls, type) == expected,
          std::string(qfn::WeightClassName(cls)) + " " + name +
              (expected ? " unexpectedly refused" : " unexpectedly accepted"));
    }
    Require(qfn::AcceptedFormats(cls).size() == accepted.size(),
            std::string(qfn::WeightClassName(cls)) +
                " table size disagrees with the accepted set");
  }
  Require(
      !qfn::FormatAccepted(static_cast<qfn::WeightClass>(0xF0), GgmlType::kF32),
      "an unknown class must accept nothing");
}

/// The (class, format) pairs of the Agention AP-Q4_K_XL trunk. Every one of
/// them must be in the table, or the tier cannot load.
void CheckApQ4KXlHistogram() {
  const std::vector<std::pair<qfn::WeightClass, GgmlType>> histogram = {
      {qfn::WeightClass::kDense, GgmlType::kQ6_K},  // attention/SSM/shexp
      {qfn::WeightClass::kDense, GgmlType::kBF16},  // indexer q/k
      {qfn::WeightClass::kDense, GgmlType::kQ8_0},  // ffn_down_shexp
      {qfn::WeightClass::kRoutedGateUp, GgmlType::kQ4_K},
      {qfn::WeightClass::kRoutedDown, GgmlType::kIQ4_NL},
      {qfn::WeightClass::kMixerDownUp, GgmlType::kQ6_K},    // hc_*_down
      {qfn::WeightClass::kMixerDownUp, GgmlType::kIQ4_NL},  // output_hc_*
      {qfn::WeightClass::kMixerInject, GgmlType::kQ6_K},
      {qfn::WeightClass::kSsmGate, GgmlType::kQ6_K},  // ssm_alpha/beta
      {qfn::WeightClass::kPleTable, GgmlType::kIQ4_NL},
      {qfn::WeightClass::kPleConv1d, GgmlType::kF16},
      {qfn::WeightClass::kF32Only, GgmlType::kF32},
  };
  for (const auto& [cls, type] : histogram) {
    Require(qfn::FormatAccepted(cls, type),
            std::string("AP-Q4_K_XL pair refused: ") +
                qfn::WeightClassName(cls) + " " +
                std::string(core::ToString(type)));
  }
}

/// The (class, format) pairs of the Swift-1.5 Q4_K_M trunk. Every one of them
/// must be in the table, or the tier cannot load.
void CheckSwiftQ4KmHistogram() {
  const std::vector<std::pair<qfn::WeightClass, GgmlType>> histogram = {
      {qfn::WeightClass::kDense, GgmlType::kQ4_K},  // token_embd, attn_gate,
                                                    // ssm_out, attn_q, ple_*
      {qfn::WeightClass::kDense, GgmlType::kQ5_K},  // attn_output
      {qfn::WeightClass::kDense, GgmlType::kQ6_K},  // indexer, shexp
      {qfn::WeightClass::kDense, GgmlType::kQ8_0},  // ffn_down_shexp
      {qfn::WeightClass::kDense, GgmlType::kBF16},  // indexer q/k
      {qfn::WeightClass::kRoutedGateUp, GgmlType::kQ4_K},
      {qfn::WeightClass::kRoutedDown, GgmlType::kQ5_0},   // half the layers
      {qfn::WeightClass::kRoutedDown, GgmlType::kQ8_0},   // the other half
      {qfn::WeightClass::kMixerDownUp, GgmlType::kQ4_K},  // hc_*_down
      {qfn::WeightClass::kMixerDownUp, GgmlType::kQ5_0},  // hc_*_up
      {qfn::WeightClass::kMixerDownUp, GgmlType::kBF16},  // output_hc_*
      {qfn::WeightClass::kMixerInject, GgmlType::kBF16},
      {qfn::WeightClass::kSsmGate, GgmlType::kF32},
      {qfn::WeightClass::kPleTable, GgmlType::kQ5_0},
      {qfn::WeightClass::kPleConv1d, GgmlType::kF16},
      {qfn::WeightClass::kF32Only, GgmlType::kF32},
  };
  for (const auto& [cls, type] : histogram) {
    Require(qfn::FormatAccepted(cls, type),
            std::string("Swift-1.5 Q4_K_M pair refused: ") +
                qfn::WeightClassName(cls) + " " +
                std::string(core::ToString(type)));
  }
}

/// The refusal message names the tensor, the format and the class.
void CheckRefusalMessage() {
  const std::string message = qfn::UnsupportedFormatMessage(
      "blk.7.ffn_gate_exps.weight", GgmlType::kQ6_K,
      qfn::WeightClass::kRoutedGateUp);
  Require(message.find("blk.7.ffn_gate_exps.weight") != std::string::npos,
          "refusal message lost the tensor name");
  Require(message.find("Q6_K") != std::string::npos,
          "refusal message lost the format");
  Require(message.find("routed gate/up expert") != std::string::npos,
          "refusal message lost the class");
}

/// The upload gate's validation: accepted weights pass, a class/format
/// mismatch is reported with the tensor name.
void CheckTensorValidator() {
  qfn::Config c;
  qfn::ModelWeights weights;
  weights.config = c;
  weights.layers.resize(1);
  // A bound tensor carries a non-null payload pointer; empty optional
  // tensors are skipped by the validator. Values are never dereferenced.
  const auto bound = [](std::string_view name, GgmlType type,
                        qfn::WeightClass cls) {
    return qfn::TensorRef{.data = reinterpret_cast<const void*>(0x10),
                          .type = type,
                          .name = name,
                          .cls = cls};
  };
  weights.layers[0].ffn_down_exps =
      bound("blk.0.ffn_down_exps.weight", GgmlType::kIQ4_NL,
            qfn::WeightClass::kRoutedDown);
  weights.layers[0].ffn_gate_exps =
      bound("blk.0.ffn_gate_exps.weight", GgmlType::kQ4_K,
            qfn::WeightClass::kRoutedGateUp);
  std::string error;
  Require(qfn::ValidateTensorFormats(weights, &error),
          "accepted weights failed format validation: " + error);
  Require(error.empty(), "validator wrote an error for accepted weights");

  // Same tensor, a format only the down projection accepts.
  weights.layers[0].ffn_gate_exps.type = GgmlType::kIQ4_NL;
  Require(!qfn::ValidateTensorFormats(weights, &error),
          "IQ4_NL routed gate expert passed format validation");
  Require(error.find("blk.0.ffn_gate_exps.weight") != std::string::npos &&
              error.find("IQ4_NL") != std::string::npos,
          "validator error lost the tensor or format: " + error);

  // Q5_0 is accepted for the routed down projection and the PLE table only.
  weights.layers[0].ffn_gate_exps.type = GgmlType::kQ5_0;
  Require(!qfn::ValidateTensorFormats(weights, &error),
          "Q5_0 routed gate expert passed format validation");
  Require(error.find("blk.0.ffn_gate_exps.weight") != std::string::npos &&
              error.find("Q5_0") != std::string::npos,
          "validator error lost the tensor or format: " + error);
  weights.layers[0].ffn_gate_exps.type = GgmlType::kQ6_K;
  Require(!qfn::ValidateTensorFormats(weights, &error),
          "Q6_K routed gate expert passed format validation");
  Require(error.find("blk.0.ffn_gate_exps.weight") != std::string::npos &&
              error.find("Q6_K") != std::string::npos,
          "validator error lost the tensor or format: " + error);

  // The Swift-1.5 mix passes the same gate.
  qfn::ModelWeights swift;
  swift.token_embd =
      bound("token_embd.weight", GgmlType::kQ4_K, qfn::WeightClass::kDense);
  swift.layers.resize(1);
  swift.layers[0].attn_out = bound("blk.0.attn_output.weight", GgmlType::kQ5_K,
                                   qfn::WeightClass::kDense);
  swift.layers[0].hc_ffn.down =
      bound("blk.0.hc_ffn_down.weight", GgmlType::kQ4_K,
            qfn::WeightClass::kMixerDownUp);
  swift.layers[0].hc_ffn.up = bound("blk.0.hc_ffn_up.weight", GgmlType::kQ5_0,
                                    qfn::WeightClass::kMixerDownUp);
  swift.layers[0].ffn_down_exps =
      bound("blk.0.ffn_down_exps.weight", GgmlType::kQ5_0,
            qfn::WeightClass::kRoutedDown);
  swift.layers[0].ffn_gate_exps =
      bound("blk.0.ffn_gate_exps.weight", GgmlType::kQ4_K,
            qfn::WeightClass::kRoutedGateUp);
  swift.ple_table = bound("per_layer_token_embd.weight", GgmlType::kQ5_0,
                          qfn::WeightClass::kPleTable);
  Require(qfn::ValidateTensorFormats(swift, &error),
          "the Swift-1.5 weight-format mix failed validation: " + error);
  swift.ple_table.type = GgmlType::kQ4_K;
  Require(!qfn::ValidateTensorFormats(swift, &error),
          "Q4_K PLE table passed format validation");
  Require(error.find("per_layer_token_embd.weight") != std::string::npos &&
              error.find("Q4_K") != std::string::npos,
          "validator error lost the tensor or format: " + error);
}

/// Removes a fixture even when a check throws.
class TempPath {
public:
  TempPath() {
    char path[] = "/tmp/qfn-format-fixture-XXXXXX";
    const int fd = ::mkstemp(path);
    Require(fd >= 0, "could not create a fixture path");
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

/// Writes a fixture with one overridden format and loads it.
void CheckBindRefusal(const qfn::Config& config, const fixture::Options& opt,
                      const std::string& expected) {
  const TempPath path;
  const std::string write_error = fixture::WriteTrunk(path.str(), config, opt);
  Require(write_error.empty(), "fixture write failed: " + write_error);

  std::string error;
  auto reader = core::GgufReader::OpenFile(path.str(), &error);
  Require(reader != nullptr, "fixture did not open: " + error);
  const auto weights = qfn::ModelWeights::Bind(*reader, &error);
  Require(!weights.has_value(),
          "loader accepted a fixture with an unaccepted format");
  Require(error == expected, "unexpected refusal: got \"" + error +
                                 "\", want \"" + expected + "\"");
}

/// A synthetic trunk written in the Swift-1.5 class formats binds: the new
/// (class, format) pairs are legal end to end, not only in the table.
void CheckSwiftFixtureBinds() {
  fixture::Options opt;
  opt.vocab_size = 4096;  // keep the sparse fixture's extent small
  opt.class_formats = fixture::SwiftMixFormats();
  const TempPath path;
  const std::string write_error =
      fixture::WriteTrunk(path.str(), fixture::FixtureConfig(opt), opt);
  Require(write_error.empty(), "fixture write failed: " + write_error);
  std::string error;
  auto reader = core::GgufReader::OpenFile(path.str(), &error);
  Require(reader != nullptr, "fixture did not open: " + error);
  const auto weights = qfn::ModelWeights::Bind(*reader, &error);
  Require(weights.has_value(),
          "the Swift-format fixture trunk did not bind: " + error);
  Require(weights->layers.size() == 48,
          "the Swift-format fixture trunk bound a wrong block count");
}

void CheckBindRefusals() {
  const fixture::Options ple;
  const qfn::Config config = fixture::FixtureConfig(ple);
  const auto refuse = [&](std::string tensor, GgmlType type) {
    fixture::Options opt = ple;
    opt.override = {std::move(tensor), type};
    return opt;
  };
  const std::string dense =
      "tensor token_embd.weight has unsupported format IQ3_S (dense "
      "projection)";
  CheckBindRefusal(config, refuse("token_embd.weight", GgmlType::kIQ3_S),
                   dense);

  // The PLE table's 160-wide rows only fit 32-element blocks, so an
  // overridden format must be one the reader can size at all.
  CheckBindRefusal(
      config, refuse("per_layer_token_embd.weight", GgmlType::kQ8_0),
      "tensor per_layer_token_embd.weight has unsupported format Q8_0 "
      "(PLE n-gram table)");

  const std::string gate_q6k =
      "tensor blk.0.ffn_gate_exps.weight has unsupported format Q6_K "
      "(routed gate/up expert)";
  CheckBindRefusal(
      config, refuse("blk.0.ffn_gate_exps.weight", GgmlType::kQ6_K), gate_q6k);
  CheckBindRefusal(
      config, refuse("blk.0.ffn_up_exps.weight", GgmlType::kQ6_K),
      "tensor blk.0.ffn_up_exps.weight has unsupported format Q6_K "
      "(routed gate/up expert)");
  CheckBindRefusal(
      config, refuse("blk.0.ffn_gate_exps.weight", GgmlType::kIQ4_NL),
      "tensor blk.0.ffn_gate_exps.weight has unsupported format IQ4_NL "
      "(routed gate/up expert)");
  CheckBindRefusal(
      config, refuse("blk.0.ffn_up_exps.weight", GgmlType::kIQ4_NL),
      "tensor blk.0.ffn_up_exps.weight has unsupported format IQ4_NL "
      "(routed gate/up expert)");
  // Q5_0 is legal for the routed down projection but not for gate/up.
  CheckBindRefusal(
      config, refuse("blk.0.ffn_gate_exps.weight", GgmlType::kQ5_0),
      "tensor blk.0.ffn_gate_exps.weight has unsupported format Q5_0 "
      "(routed gate/up expert)");
  CheckBindRefusal(
      config, refuse("blk.0.ffn_up_exps.weight", GgmlType::kQ5_0),
      "tensor blk.0.ffn_up_exps.weight has unsupported format Q5_0 "
      "(routed gate/up expert)");
  // The Q4_K/Q5_K dense formats are accepted, but Q5_0 is not a dense
  // projection format.
  CheckBindRefusal(
      config, refuse("token_embd.weight", GgmlType::kQ5_0),
      "tensor token_embd.weight has unsupported format Q5_0 (dense "
      "projection)");
  // The down projection's 640-wide rows fit 32-element blocks only, so a
  // 256-element format (Q6_K, IQ2_XXS) is refused by the reader's size check
  // before the class check; the policy-level cases above cover those, and
  // here the same class refusal is exercised through the loader.
  CheckBindRefusal(config,
                   refuse("blk.0.ffn_down_exps.weight", GgmlType::kQ4_0),
                   "tensor blk.0.ffn_down_exps.weight has unsupported format "
                   "Q4_0 (routed down expert)");

  // Mixer and SSM-gate classes: a format another class accepts is still
  // refused here.
  CheckBindRefusal(
      config, refuse("blk.0.hc_attn_inject.weight", GgmlType::kIQ4_NL),
      "tensor blk.0.hc_attn_inject.weight has unsupported format IQ4_NL "
      "(hyper-connection inject projection)");
  // Q4_K and Q5_0 are now legal for the Mixer projections; the routed
  // formats that no dense or Mixer kernel reads still are not.
  CheckBindRefusal(
      config, refuse("blk.0.hc_ffn_down.weight", GgmlType::kQ5_1),
      "tensor blk.0.hc_ffn_down.weight has unsupported format Q5_1 "
      "(hyper-connection down/up projection)");
  CheckBindRefusal(
      config, refuse("blk.0.hc_ffn_down.weight", GgmlType::kQ5_K),
      "tensor blk.0.hc_ffn_down.weight has unsupported format Q5_K "
      "(hyper-connection down/up projection)");
  CheckBindRefusal(config, refuse("blk.0.ssm_alpha.weight", GgmlType::kF16),
                   "tensor blk.0.ssm_alpha.weight has unsupported format F16 "
                   "(SSM gate projection)");
}

/// Writes a fixture for the CLI load-error check and prints its path.
int EmitFixture(const std::string& variant, const std::string& path) {
  if (variant == "cli") {
    // One declared tensor: the loader refuses the embedding format before it
    // reads any payload, so the fixture is a sparse file of the declared
    // extent with no data blocks.
    fixture::Options opt;
    opt.override = {"token_embd.weight", GgmlType::kIQ3_S};
    const std::string error =
        fixture::WriteTrunk(path, fixture::FixtureConfig(opt), opt);
    if (!error.empty()) {
      std::cerr << "fixture write failed: " << error << "\n";
      return 1;
    }
    return 0;
  }
  std::cerr << "unknown fixture variant: " << variant << "\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "--emit") {
      return EmitFixture(argv[2], argv[3]);
    }
    if (argc != 1) {
      std::cerr << "usage: " << argv[0] << " [--emit <variant> <path>]\n";
      return 2;
    }
    CheckAcceptedSets();
    CheckApQ4KXlHistogram();
    CheckSwiftQ4KmHistogram();
    CheckRefusalMessage();
    CheckTensorValidator();
    CheckSwiftFixtureBinds();
    CheckBindRefusals();
    std::cout << "Flash-Next weight-format policy checks passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "format policy check failed: " << error.what() << "\n";
    return 1;
  }
}
