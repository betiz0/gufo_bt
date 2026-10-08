#ifndef GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_GGUF_FIXTURE_HPP_
#define GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_GGUF_FIXTURE_HPP_

// Synthetic `qwen4exp` GGUF fixtures for loader tests. The payload regions are
// sparse holes: `GgufReader` validates every declared tensor extent at open
// but the bind step only reads headers, so a fixture that fails on a format
// never touches the weight bytes. Tensors are declared in the order
// `ModelWeights::Bind` reads them and the emitter stops at the first tensor
// whose format its class refuses, which keeps each fixture as small as the
// case allows.

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/config.hpp"
#include "src/models/qwen38_flash_next/weight_format.hpp"

namespace gufo::models::qwen38_flash_next::fixture {

/// One declared tensor of the synthetic trunk.
struct TensorSpec {
  std::string name;
  std::uint64_t cols{1};     ///< GGUF ne[0], the contiguous dimension.
  std::uint64_t rows{1};     ///< GGUF ne[1].
  std::uint64_t experts{1};  ///< GGUF ne[2].
  core::GgmlType type{core::GgmlType::kF32};
  WeightClass cls{WeightClass::kDense};
};

/// Replaces one tensor's format. The emitted prefix stops there when the
/// replacement is not accepted for the tensor's class.
struct Override {
  std::string tensor;
  core::GgmlType type{core::GgmlType::kQ6_K};
};

/// Replaces the default storage format of one weight class, so a fixture can
/// carry a producer mix whose classes use a different (still accepted) format
/// than `DefaultFormat` picks, such as the Swift-1.5 Q4_K_M pair set.
struct ClassFormat {
  WeightClass cls{WeightClass::kDense};
  core::GgmlType type{core::GgmlType::kF32};
  /// Restricts the entry to tensors whose name ends with `suffix`; empty
  /// applies to every tensor of the class. Entries are matched in order, so a
  /// suffix entry must precede the class-wide one it overrides. Owned (not a
  /// view) so a caller may build names at run time.
  std::string suffix;
};

struct Options {
  /// Rows of `token_embd`/`output`; a small value keeps a CLI fixture tiny.
  std::uint64_t vocab_size{248320};
  /// PLE metadata plus its table and `ple_conv1d` tensor on layer 1.
  bool with_ple{true};
  /// `block_count` metadata. The MTP sidecar declares one more layer than the
  /// trunk it matches.
  std::uint32_t block_count{48};
  /// `nextn_predict_layers` metadata: 1 for the MTP sidecar, 0 for a trunk.
  std::uint32_t nextn_layers{0};
  Override override;
  /// Per-class storage formats; a class absent here uses `DefaultFormat`.
  std::vector<ClassFormat> class_formats;
};

/// The pinned metadata `Config::FromGguf` requires, mirroring the 48-block
/// Qwen3.8-Flash-Next geometry. Returns the number of metadata keys written,
/// so the header can declare it after the fact.
std::uint64_t AppendMetadata(std::vector<std::uint8_t>& out,
                             const Options& opt) {
  std::uint64_t keys = 0;
  const auto pod = [&]<class T>(T value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
  };
  const auto text = [&](std::string_view value) {
    pod(std::uint64_t{value.size()});
    out.insert(out.end(), value.begin(), value.end());
  };
  const auto named = [&](std::string_view name) {
    text(name);
    ++keys;
  };
  const auto u32_value = [&](std::uint32_t value) {
    pod(std::uint32_t{4});
    pod(value);
  };
  const auto f32_value = [&](float value) {
    pod(std::uint32_t{6});
    pod(value);
  };
  const auto string_value = [&](std::string_view value) {
    pod(std::uint32_t{8});
    text(value);
  };
  const auto u32_key = [&](std::string_view suffix, std::uint32_t value) {
    named(std::string("qwen4exp.") + std::string(suffix));
    u32_value(value);
  };
  const auto u64_array = [&](const std::vector<std::uint64_t>& values) {
    pod(std::uint32_t{9});   // array
    pod(std::uint32_t{10});  // of uint64
    pod(std::uint64_t{values.size()});
    for (const auto value : values) {
      pod(value);
    }
  };

  named("general.architecture");
  string_value("qwen4exp");
  // Not a Qwen3.8 artifact name: the chat-template hash check stays out of
  // the way of a format check.
  named("general.name");
  string_value("Flash-Next format fixture");

  u32_key("block_count", opt.block_count);
  if (opt.nextn_layers != 0) {
    u32_key("nextn_predict_layers", opt.nextn_layers);
  }
  u32_key("embedding_length", 2560);
  u32_key("context_length", 262144);
  u32_key("hyper_connection.count", 4);
  u32_key("hyper_connection.low_rank", 320);
  u32_key("full_attention_interval", 4);
  u32_key("attention.head_count", 24);
  u32_key("attention.head_count_kv", 2);
  u32_key("attention.key_length", 256);
  u32_key("attention.indexer.head_count", 4);
  u32_key("attention.indexer.key_length", 128);
  u32_key("attention.indexer.top_k", 2048);
  u32_key("ssm.conv_kernel", 4);
  u32_key("ssm.state_size", 128);
  u32_key("ssm.group_count", 16);
  u32_key("ssm.time_step_rank", 48);
  u32_key("ssm.inner_size", 6144);
  u32_key("expert_count", 512);
  u32_key("expert_used_count", 10);
  u32_key("expert_feed_forward_length", 640);
  u32_key("expert_shared_feed_forward_length", 640);

  named("qwen4exp.attention.layer_norm_rms_epsilon");
  f32_value(1e-6F);
  named("qwen4exp.rope.freq_base");
  f32_value(1e7F);
  named("qwen4exp.rope.dimension_count");
  u32_value(64);
  named("qwen4exp.rope.dimension_sections");
  u64_array({11, 11, 10, 0});

  // One ratio per declared layer; every layer of the nextn block keeps the
  // zero the trunk profile uses there.
  std::vector<std::uint64_t> ratios(opt.block_count, 0);
  for (std::size_t layer = 3; layer < ratios.size(); layer += 4) {
    ratios[layer] = 4;
  }
  named("qwen4exp.attention.compress_ratios");
  u64_array(ratios);

  if (opt.with_ple) {
    named("qwen4exp.ple.layers");
    u64_array({1});
    u32_key("ple.ngram_size", 3);
    u32_key("ple.heads_per_ngram", 8);
    u32_key("ple.conv_kernel", 4);
    u32_key("ple.eos_token_id", 248046);
    // Unlike the other PLE keys, the embedding width keeps llama.cpp's
    // top-level name.
    named("qwen4exp.embedding_length_per_layer_input");
    u32_value(160);
    named("qwen4exp.ple.layer_multipliers");
    u64_array({1, 2, 3});
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint64_t> vocab;
    for (std::uint64_t head = 0; head < 16; ++head) {
      offsets.push_back(head * 64);
      vocab.push_back(64);
    }
    named("qwen4exp.ple.head_offsets");
    u64_array(offsets);
    named("qwen4exp.ple.head_vocab_sizes");
    u64_array(vocab);
  }
  return keys;
}

/// Smallest format each class accepts, so an accepted fixture stays compact.
constexpr core::GgmlType DefaultFormat(WeightClass cls) {
  switch (cls) {
    case WeightClass::kDense:
    case WeightClass::kRoutedDown:
    case WeightClass::kMixerDownUp:
    case WeightClass::kMixerInject:
    case WeightClass::kSsmGate:
      return core::GgmlType::kQ8_0;
    case WeightClass::kRoutedGateUp:
      return core::GgmlType::kQ4_K;
    case WeightClass::kPleTable:
      return core::GgmlType::kIQ4_NL;
    case WeightClass::kPleConv1d:
    case WeightClass::kF32Only:
      return core::GgmlType::kF32;
  }
  return core::GgmlType::kF32;
}

/// The format a tensor of `cls` is written with: the first matching per-class
/// override for its name, else the smallest accepted format.
inline core::GgmlType SelectedFormat(std::string_view name, WeightClass cls,
                                     const Options& opt) {
  for (const ClassFormat& entry : opt.class_formats) {
    if (entry.cls == cls &&
        (entry.suffix.empty() || name.ends_with(entry.suffix))) {
      return entry.type;
    }
  }
  return DefaultFormat(cls);
}

/// The Swift-1.5 Q4_K_M pair set on the tensors that mix stores it on: the
/// dense projections Q4_K with `attn_output` Q5_K and the 640-wide shared
/// expert down projection Q8_0, Mixer down Q4_K with up Q5_0, routed gate/up
/// Q4_K, routed down Q5_0, the PLE table Q5_0 and `ple_conv1d` F16.
inline std::vector<ClassFormat> SwiftMixFormats() {
  return {
      {WeightClass::kDense, core::GgmlType::kQ5_K, "attn_output.weight"},
      {WeightClass::kDense, core::GgmlType::kQ8_0, "ffn_down_shexp.weight"},
      {WeightClass::kDense, core::GgmlType::kQ4_K, ""},
      {WeightClass::kRoutedGateUp, core::GgmlType::kQ4_K, ""},
      {WeightClass::kRoutedDown, core::GgmlType::kQ5_0, ""},
      // The Mixer up projection's 320-wide rows fit 32-element blocks only,
      // so one class-wide entry cannot express the down/up pair.
      {WeightClass::kMixerDownUp, core::GgmlType::kQ4_K, "_down.weight"},
      {WeightClass::kMixerDownUp, core::GgmlType::kQ5_0, "_up.weight"},
      {WeightClass::kPleTable, core::GgmlType::kQ5_0, ""},
      {WeightClass::kPleConv1d, core::GgmlType::kF16, ""},
  };
}
/// The config the fixture metadata describes, so a caller can emit a trunk
/// without a real artifact. Throws when the metadata does not parse.
inline Config FixtureConfig(const Options& opt) {
  std::vector<std::uint8_t> blob;
  const auto pod = [&]<class T>(T value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    blob.insert(blob.end(), bytes, bytes + sizeof(T));
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{0});  // no tensors
  pod(std::uint64_t{0});  // patched below
  const std::size_t count_at = blob.size() - sizeof(std::uint64_t);
  std::vector<std::uint8_t> metadata;
  const std::uint64_t count = AppendMetadata(metadata, opt);
  blob.insert(blob.end(), metadata.begin(), metadata.end());
  std::memcpy(blob.data() + count_at, &count, sizeof(count));
  // The reader aligns the payload base to 32 bytes; a metadata-only blob
  // must still leave room for that padding.
  blob.resize((blob.size() + 31) / 32 * 32);
  std::string error;
  auto reader = core::GgufReader::OpenMemory(blob.data(), blob.size(), &error);
  if (reader == nullptr) {
    throw std::runtime_error("fixture metadata did not parse: " + error);
  }
  auto config = Config::FromGguf(*reader, true, &error);
  if (!config.has_value()) {
    throw std::runtime_error("fixture metadata is not a valid config: " +
                             error);
  }
  return *config;
}

/// Appends one tensor in bind order. Returns true when the emitted format is
/// refused for its class, so the loader never reads past it.
inline bool AppendTensor(std::vector<TensorSpec>& tensors, std::string name,
                         std::uint64_t cols, std::uint64_t rows,
                         std::uint64_t experts, WeightClass cls,
                         const Options& opt) {
  const core::GgmlType type = name == opt.override.tensor
                                  ? opt.override.type
                                  : SelectedFormat(name, cls, opt);
  tensors.push_back({std::move(name), cols, rows, experts, type, cls});
  return !FormatAccepted(cls, type);
}

/// Appends one hyper-connection Mixer (`<prefix>_norm/_down/_up`, plus
/// `_inject` when requested) in bind order.
inline bool AppendMixer(std::vector<TensorSpec>& tensors,
                        const std::string& prefix, bool inject, const Config& c,
                        const Options& opt) {
  if (AppendTensor(tensors, prefix + "_norm.weight", c.HcDim(), 1, 1,
                   WeightClass::kF32Only, opt)) {
    return true;
  }
  if (AppendTensor(tensors, prefix + "_down.weight", c.HcDim(), c.hc_low_rank,
                   1, WeightClass::kMixerDownUp, opt)) {
    return true;
  }
  if (AppendTensor(tensors, prefix + "_up.weight", c.hc_low_rank, c.HcDim(), 1,
                   WeightClass::kMixerDownUp, opt)) {
    return true;
  }
  return inject && AppendTensor(tensors, prefix + "_inject.weight", c.HcDim(),
                                c.hc_count, 1, WeightClass::kMixerInject, opt);
}

/// The synthetic trunk's tensors in bind order, up to and including the first
/// tensor whose emitted format its class refuses.
std::vector<TensorSpec> TrunkTensors(const Config& c, const Options& opt) {
  std::vector<TensorSpec> tensors;
  const auto add = [&](std::string name, std::uint64_t cols, std::uint64_t rows,
                       std::uint64_t experts, WeightClass cls) {
    return AppendTensor(tensors, std::move(name), cols, rows, experts, cls,
                        opt);
  };
  const auto add_mixer = [&](const std::string& prefix, bool inject) {
    return AppendMixer(tensors, prefix, inject, c, opt);
  };

  if (add("token_embd.weight", c.hidden_size, opt.vocab_size, 1,
          WeightClass::kDense)) {
    return tensors;
  }
  if (add_mixer("output_hc", false)) {
    return tensors;
  }
  if (opt.with_ple && add("per_layer_token_embd.weight", c.ple_head_dim, 1024,
                          1, WeightClass::kPleTable)) {
    return tensors;
  }
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    if (add_mixer(p + "hc_attn", true) || add_mixer(p + "hc_ffn", true)) {
      return tensors;
    }
    if (c.IsLinearLayer(il)) {
      if (add(p + "attn_qkv.weight", c.hidden_size, c.SsmConvChannels(), 1,
              WeightClass::kDense) ||
          add(p + "attn_gate.weight", c.hidden_size, c.SsmValueDim(), 1,
              WeightClass::kDense) ||
          add(p + "ssm_conv1d.weight", c.ssm_conv_kernel, c.SsmConvChannels(),
              1, WeightClass::kF32Only) ||
          add(p + "ssm_alpha.weight", c.hidden_size, c.ssm_num_v_heads, 1,
              WeightClass::kSsmGate) ||
          add(p + "ssm_beta.weight", c.hidden_size, c.ssm_num_v_heads, 1,
              WeightClass::kSsmGate) ||
          add(p + "ssm_dt.bias", c.ssm_num_v_heads, 1, 1,
              WeightClass::kF32Only) ||
          add(p + "ssm_a", c.ssm_num_v_heads, 1, 1, WeightClass::kF32Only) ||
          add(p + "ssm_norm.weight", c.ssm_head_dim, 1, 1,
              WeightClass::kF32Only) ||
          add(p + "ssm_out.weight", c.SsmValueDim(), c.hidden_size, 1,
              WeightClass::kDense)) {
        return tensors;
      }
    } else if (add(p + "attn_q.weight", c.hidden_size, 2 * c.AttentionQDim(), 1,
                   WeightClass::kDense) ||
               add(p + "attn_k.weight", c.hidden_size, c.AttentionKvDim(), 1,
                   WeightClass::kDense) ||
               add(p + "attn_v.weight", c.hidden_size, c.AttentionKvDim(), 1,
                   WeightClass::kDense) ||
               add(p + "attn_output.weight", c.AttentionQDim(), c.hidden_size,
                   1, WeightClass::kDense) ||
               add(p + "attn_q_norm.weight", c.head_dim, 1, 1,
                   WeightClass::kF32Only) ||
               add(p + "attn_k_norm.weight", c.head_dim, 1, 1,
                   WeightClass::kF32Only) ||
               add(p + "indexer.q_proj.weight", c.hidden_size,
                   c.indexer_heads * c.indexer_head_dim, 1,
                   WeightClass::kDense) ||
               add(p + "indexer.k_proj.weight", c.hidden_size,
                   c.indexer_head_dim, 1, WeightClass::kDense) ||
               add(p + "indexer.q_norm.weight", c.indexer_head_dim, 1, 1,
                   WeightClass::kF32Only) ||
               add(p + "indexer.k_norm.weight", c.indexer_head_dim, 1, 1,
                   WeightClass::kF32Only)) {
      return tensors;
    }
    if (c.IsPleLayer(il) && (add(p + "ple_key.weight", c.PleEmbeddingDim(),
                                 c.HcDim(), 1, WeightClass::kDense) ||
                             add(p + "ple_value.weight", c.PleEmbeddingDim(),
                                 c.hidden_size, 1, WeightClass::kDense) ||
                             add(p + "ple_norm_key.weight", c.HcDim(), 1, 1,
                                 WeightClass::kF32Only) ||
                             add(p + "ple_norm_query.weight", c.HcDim(), 1, 1,
                                 WeightClass::kF32Only) ||
                             add(p + "ple_norm_conv.weight", c.HcDim(), 1, 1,
                                 WeightClass::kF32Only) ||
                             add(p + "ple_conv1d.weight", c.ple_conv_kernel,
                                 c.HcDim(), 1, WeightClass::kPleConv1d))) {
      return tensors;
    }
    if (add(p + "ffn_gate_inp.weight", c.hidden_size, c.num_experts, 1,
            WeightClass::kF32Only) ||
        add(p + "ffn_gate_exps.weight", c.hidden_size, c.expert_ff,
            c.num_experts, WeightClass::kRoutedGateUp) ||
        add(p + "ffn_up_exps.weight", c.hidden_size, c.expert_ff, c.num_experts,
            WeightClass::kRoutedGateUp) ||
        add(p + "ffn_down_exps.weight", c.expert_ff, c.hidden_size,
            c.num_experts, WeightClass::kRoutedDown) ||
        add(p + "ffn_gate_inp_shexp.weight", c.hidden_size, 1, 1,
            WeightClass::kF32Only) ||
        add(p + "ffn_gate_shexp.weight", c.hidden_size, c.shared_expert_ff, 1,
            WeightClass::kDense) ||
        add(p + "ffn_up_shexp.weight", c.hidden_size, c.shared_expert_ff, 1,
            WeightClass::kDense) ||
        add(p + "ffn_down_shexp.weight", c.shared_expert_ff, c.hidden_size, 1,
            WeightClass::kDense)) {
      return tensors;
    }
  }
  return tensors;
}

/// Writes the fixture to `path` as a sparse file and returns an empty string,
/// or the OS error that prevented it.
std::string WriteGguf(const std::string& path,
                      const std::vector<TensorSpec>& tensors,
                      const Options& opt) {
  std::vector<std::uint8_t> header;
  const auto pod = [&]<class T>(T value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    header.insert(header.end(), bytes, bytes + sizeof(T));
  };
  const auto text = [&](std::string_view value) {
    pod(std::uint64_t{value.size()});
    header.insert(header.end(), value.begin(), value.end());
  };

  std::vector<std::uint8_t> metadata;
  const std::uint64_t metadata_count = AppendMetadata(metadata, opt);

  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{tensors.size()});
  pod(metadata_count);
  header.insert(header.end(), metadata.begin(), metadata.end());

  // Tensor infos: extents are sequential from the data base, 32-byte aligned.
  std::vector<std::uint64_t> offsets;
  offsets.reserve(tensors.size());
  std::uint64_t extent = 0;
  for (const auto& tensor : tensors) {
    offsets.push_back(extent);
    const std::size_t size = quant::EncodedSizeBytes(
        tensor.type, tensor.cols * tensor.rows * tensor.experts);
    extent += (size + 31) / 32 * 32;
  }
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    text(tensor.name);
    // GGUF stores the significant dimensions only; the loader reads a
    // missing dimension as one.
    const std::uint32_t dims = tensor.experts > 1 ? 3U
                               : tensor.rows > 1  ? 2U
                                                  : 1U;
    pod(dims);
    pod(tensor.cols);
    if (dims > 1) {
      pod(tensor.rows);
    }
    if (dims > 2) {
      pod(tensor.experts);
    }
    pod(static_cast<std::uint32_t>(tensor.type));
    pod(offsets[i]);
  }
  const std::size_t alignment = 32;
  const std::uint64_t data_base =
      (header.size() + alignment - 1) / alignment * alignment;
  header.resize(data_base, 0);

  const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
  if (fd < 0) {
    return std::string(std::strerror(errno));
  }
  std::string error;
  ssize_t written = 0;
  while (written < static_cast<ssize_t>(header.size())) {
    const ssize_t n =
        ::write(fd, header.data() + written, header.size() - written);
    if (n <= 0) {
      error = std::string(std::strerror(errno));
      break;
    }
    written += n;
  }
  if (error.empty() &&
      ::ftruncate(fd, static_cast<off_t>(data_base + extent)) != 0) {
    error = std::string(std::strerror(errno));
  }
  ::close(fd);
  return error;
}

/// Writes the synthetic trunk described by `c` and `opt`.
std::string WriteTrunk(const std::string& path, const Config& c,
                       const Options& opt) {
  return WriteGguf(path, TrunkTensors(c, opt), opt);
}

/// The MTP sidecar's tensors in bind order: the single `nextn` full-attention
/// block at index `c.num_layers`, mirroring
/// `Binder::Layer(c, c.num_layers, false, false, true)`.
std::vector<TensorSpec> SidecarTensors(const Config& c, const Options& opt) {
  std::vector<TensorSpec> tensors;
  const std::string p = "blk." + std::to_string(c.num_layers) + ".";
  if (AppendMixer(tensors, p + "hc_attn", true, c, opt) ||
      AppendMixer(tensors, p + "hc_ffn", true, c, opt)) {
    return tensors;
  }
  if (AppendTensor(tensors, p + "attn_q.weight", c.hidden_size,
                   2 * c.AttentionQDim(), 1, WeightClass::kDense, opt) ||
      AppendTensor(tensors, p + "attn_k.weight", c.hidden_size,
                   c.AttentionKvDim(), 1, WeightClass::kDense, opt) ||
      AppendTensor(tensors, p + "attn_v.weight", c.hidden_size,
                   c.AttentionKvDim(), 1, WeightClass::kDense, opt) ||
      AppendTensor(tensors, p + "attn_output.weight", c.AttentionQDim(),
                   c.hidden_size, 1, WeightClass::kDense, opt) ||
      AppendTensor(tensors, p + "attn_q_norm.weight", c.head_dim, 1, 1,
                   WeightClass::kF32Only, opt) ||
      AppendTensor(tensors, p + "attn_k_norm.weight", c.head_dim, 1, 1,
                   WeightClass::kF32Only, opt) ||
      AppendTensor(tensors, p + "indexer.q_proj.weight", c.hidden_size,
                   c.indexer_heads * c.indexer_head_dim, 1, WeightClass::kDense,
                   opt) ||
      AppendTensor(tensors, p + "indexer.k_proj.weight", c.hidden_size,
                   c.indexer_head_dim, 1, WeightClass::kDense, opt) ||
      AppendTensor(tensors, p + "indexer.q_norm.weight", c.indexer_head_dim, 1,
                   1, WeightClass::kF32Only, opt) ||
      AppendTensor(tensors, p + "indexer.k_norm.weight", c.indexer_head_dim, 1,
                   1, WeightClass::kF32Only, opt)) {
    return tensors;
  }
  if (AppendTensor(tensors, p + "ffn_gate_inp.weight", c.hidden_size,
                   c.num_experts, 1, WeightClass::kF32Only, opt) ||
      AppendTensor(tensors, p + "ffn_gate_exps.weight", c.hidden_size,
                   c.expert_ff, c.num_experts, WeightClass::kRoutedGateUp,
                   opt) ||
      AppendTensor(tensors, p + "ffn_up_exps.weight", c.hidden_size,
                   c.expert_ff, c.num_experts, WeightClass::kRoutedGateUp,
                   opt) ||
      AppendTensor(tensors, p + "ffn_down_exps.weight", c.expert_ff,
                   c.hidden_size, c.num_experts, WeightClass::kRoutedDown,
                   opt) ||
      AppendTensor(tensors, p + "ffn_gate_inp_shexp.weight", c.hidden_size, 1,
                   1, WeightClass::kF32Only, opt) ||
      AppendTensor(tensors, p + "ffn_gate_shexp.weight", c.hidden_size,
                   c.shared_expert_ff, 1, WeightClass::kDense, opt) ||
      AppendTensor(tensors, p + "ffn_up_shexp.weight", c.hidden_size,
                   c.shared_expert_ff, 1, WeightClass::kDense, opt) ||
      AppendTensor(tensors, p + "ffn_down_shexp.weight", c.shared_expert_ff,
                   c.hidden_size, 1, WeightClass::kDense, opt)) {
    return tensors;
  }
  if (AppendTensor(tensors, p + "nextn.enorm.weight", c.hidden_size, 1, 1,
                   WeightClass::kF32Only, opt) ||
      AppendTensor(tensors, p + "nextn.hnorm.weight", c.HcDim(), 1, 1,
                   WeightClass::kF32Only, opt) ||
      AppendTensor(tensors, p + "nextn.eh_proj.weight", 2 * c.hidden_size,
                   c.hidden_size, 1, WeightClass::kDense, opt) ||
      AppendMixer(tensors, p + "nextn.hc_head", false, c, opt)) {
    return tensors;
  }
  return tensors;
}

/// Writes the MTP sidecar matching `trunk` (one extra `nextn` block) so
/// `MtpWeights::Bind` and `Config::MtpMatches` accept it.
std::string WriteSidecar(const std::string& path, const Config& trunk,
                         const Options& opt) {
  Options sidecar = opt;
  sidecar.block_count = static_cast<std::uint32_t>(trunk.num_layers) + 1;
  sidecar.nextn_layers = 1;
  return WriteGguf(path, SidecarTensors(trunk, sidecar), sidecar);
}

}  // namespace gufo::models::qwen38_flash_next::fixture

#endif  // GUFO_TESTS_MODELS_QWEN38_FLASH_NEXT_GGUF_FIXTURE_HPP_
