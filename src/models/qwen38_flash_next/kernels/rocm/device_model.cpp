#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <initializer_list>

#include "src/core/hip/weight_upload.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

/// Elements of one stored block of `type` (the stored byte count is
/// `elements / block * ggml_type_size`, which the reader already validated).
std::size_t WeightBlockElements(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kQ6_K:
      return 256;
    case core::GgmlType::kQ5_0:
    case core::GgmlType::kQ8_0:
    case core::GgmlType::kIQ4_NL:
      return 32;
    default:
      return 1;
  }
}

std::size_t WeightBlockBytes(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kQ6_K:
      return 210;
    case core::GgmlType::kQ5_0:
      return 22;
    case core::GgmlType::kQ8_0:
      return 34;
    case core::GgmlType::kIQ4_NL:
      return 18;
    case core::GgmlType::kF16:
    case core::GgmlType::kBF16:
      return 2;
    default:
      return 4;
  }
}

/// The kernels' WeightType for a stored GGML type. Every format the uploader
/// may decode at load must map to its own WeightType: a mixed `ssm_alpha`/
/// `ssm_beta` pair decodes a Q8_0 half through the same path as a Q6_K half,
/// and reading Q8_0 bytes as F32 would run past the staging buffer.
WeightType SmallType(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kQ8_0:
      return WeightType::kQ8_0;
    case core::GgmlType::kQ5_0:
      return WeightType::kQ5_0;
    case core::GgmlType::kQ6_K:
      return WeightType::kQ6_K;
    case core::GgmlType::kIQ4_NL:
      return WeightType::kIQ4_NL;
    case core::GgmlType::kF16:
      return WeightType::kF16;
    case core::GgmlType::kBF16:
      return WeightType::kBF16;
    default:
      return WeightType::kF32;
  }
}

struct Conversion {
  void* source;
  void* destination;
  std::size_t count;
};

/// One tensor decoded at load: a staging copy of its stored encoding and the
/// destination its consumers read (F16 or F32).
struct Decode {
  void* source;
  void* destination;
  core::GgmlType type;
  bool to_half;
  std::size_t elements;
};

struct Uploader {
  hip::WeightUpload& stager;
  std::vector<Conversion>& conversions;
  std::vector<Decode>& decodes;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  std::size_t& decoded_bytes;
  std::vector<std::pair<std::string, std::size_t>>* decoded_ptr;
  std::vector<DeviceModel::UploadedTensor>& uploaded;
  std::string* error;
  bool ok{true};
  std::uint32_t shard_base{0};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (!stager.Copy(shard_base + t.shard, t.file_offset, size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    uploaded.push_back({std::string(t.name), size, size + kTailMargin});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if (t.type == core::GgmlType::kQ8_0 && t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  /// Load-time decode of one tensor whose consumers read F32 or F16: the
  /// stored bytes are staged, expanded by DecodeWeights after the disk
  /// pipeline drains, and the staging copy is released. A tensor already in
  /// the destination format is uploaded as stored.
  DeviceTensor DecodeTensor(const TensorRef& t, bool to_half) {
    if (t.empty()) {
      return {};
    }
    const core::GgmlType target =
        to_half ? core::GgmlType::kF16 : core::GgmlType::kF32;
    if (t.type == target) {
      return Copy(t);
    }
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* staging = nullptr;
    if (hipMalloc(&staging, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    allocations.push_back(staging);
    bytes += size + kTailMargin;
    if (!stager.Copy(shard_base + t.shard, t.file_offset, size, staging,
                     error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    const std::size_t element_bytes =
        to_half ? sizeof(std::uint16_t) : sizeof(float);
    const std::size_t decoded = t.ElementCount() * element_bytes;
    void* dst = nullptr;
    if (hipMalloc(&dst, decoded + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for decoded " + std::string(t.name));
      return d;
    }
    allocations.push_back(dst);
    bytes += decoded + kTailMargin;
    decoded_bytes += decoded;
    if (decoded_ptr != nullptr) {
      decoded_ptr->emplace_back(std::string(t.name), decoded);
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(dst) + decoded, 0,
                         kTailMargin, nullptr);
    decodes.push_back({staging, dst, t.type, to_half, t.ElementCount()});
    d.data = dst;
    d.type = target;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (to_half) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    return d;
  }

  /// Hyper-connection down/up projections: Q6_K and IQ4_NL are decoded to
  /// F16 (their consumers are fixed-size F16 kernels); the formats with a
  /// dense path of their own stay in their stored encoding.
  DeviceTensor MixerProjection(const TensorRef& t) {
    if (t.type == core::GgmlType::kQ6_K || t.type == core::GgmlType::kIQ4_NL) {
      return DecodeTensor(t, true);
    }
    return Copy(t);
  }

  DeviceMixer Mixer(const HcMixer& m) {
    return {Copy(m.norm), MixerProjection(m.down), MixerProjection(m.up),
            m.inject.type == core::GgmlType::kQ6_K
                ? DecodeTensor(m.inject, false)
                : Copy(m.inject)};
  }

  /// ssm_alpha | ssm_beta as one F16 block of rows. Q8_0 pairs keep their
  /// storable encoding (the decode GEMV reads them merged anyway).
  DeviceTensor StackAlphaBeta(const TensorRef& a, const TensorRef& b) {
    if (a.type == core::GgmlType::kQ8_0 && b.type == core::GgmlType::kQ8_0) {
      return Stack({&a, &b});
    }
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    if (a.cols != b.cols || a.empty() || b.empty()) {
      Fail("stacked alpha/beta tensors disagree");
      return d;
    }
    const std::size_t rows = a.rows + b.rows;
    const std::size_t payload =
        (a.ElementCount() + b.ElementCount()) * sizeof(std::uint16_t);
    void* dst = nullptr;
    if (hipMalloc(&dst, payload + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for the alpha/beta stack");
      return d;
    }
    allocations.push_back(dst);
    bytes += payload + kTailMargin;
    decoded_bytes += payload;
    if (decoded_ptr != nullptr) {
      decoded_ptr->emplace_back(std::string(a.name) + "+" + std::string(b.name),
                                payload);
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(dst) + payload, 0,
                         kTailMargin, nullptr);
    const auto stage = [&](const TensorRef& t, std::size_t row_offset) {
      const std::size_t size = t.SizeBytes();
      void* staging = nullptr;
      if (hipMalloc(&staging, size + kTailMargin) != hipSuccess) {
        Fail("hipMalloc failed for " + std::string(t.name));
        return;
      }
      allocations.push_back(staging);
      bytes += size + kTailMargin;
      if (stager.Copy(shard_base + t.shard, t.file_offset, size, staging,
                      error)) {
        auto* destination = static_cast<std::uint8_t*>(dst) +
                            row_offset * t.cols * sizeof(std::uint16_t);
        decodes.push_back(
            {staging, destination, t.type, true, t.ElementCount()});
      } else {
        Fail("upload failed for " + std::string(t.name));
      }
    };
    stage(a, 0);
    stage(b, a.rows);
    d.data = dst;
    d.type = core::GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>(a.cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  // GGUF packs [fc_embedding | fc_hidden] across each row. Split on a
  // quantization-block boundary without dequantizing or changing any weight.
  void SplitMtpProjection(const TensorRef& t, DeviceTensor& embedding,
                          DeviceTensor& hidden) {
    if (t.empty() || !ok)
      return;
    const auto combined = Copy(t);
    if (!ok || !stager.Finish(error)) {
      Fail("MTP projection upload failed");
      return;
    }
    const std::size_t row_bytes = t.SizeBytes() / t.rows / 2;
    const std::size_t part_bytes = row_bytes * t.rows;
    for (std::uint32_t part = 0; part < 2; ++part) {
      auto& dst = part == 0 ? embedding : hidden;
      dst = combined;
      dst.cols /= 2;
      if (hipMalloc(&dst.data, part_bytes + kTailMargin) != hipSuccess) {
        Fail("MTP split projection allocation failed");
        return;
      }
      allocations.push_back(dst.data);
      bytes += part_bytes + kTailMargin;
      const auto* src =
          static_cast<const std::uint8_t*>(combined.data) + part * row_bytes;
      if (hipMemcpy2D(dst.data, row_bytes, src, 2 * row_bytes, row_bytes,
                      t.rows, hipMemcpyDeviceToDevice) != hipSuccess ||
          hipMemset(static_cast<std::uint8_t*>(dst.data) + part_bytes, 0,
                    kTailMargin) != hipSuccess) {
        Fail("MTP split projection copy failed");
        return;
      }
    }
    std::erase(allocations, combined.data);
    (void)hipFree(combined.data);
    bytes -= t.SizeBytes() + kTailMargin;
  }

  /// Uploads matrices of one type stacked along rows; every input shares
  /// `cols`. An F32 stack (router logits, GDN alpha/beta: the only
  /// unquantized projections) is narrowed to F16, which the wide-batch GEMM
  /// tier runs at speed. A Q8_0 stack merges projections of one input into
  /// a single decode GEMV.
  DeviceTensor Stack(std::initializer_list<const TensorRef*> parts) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    std::size_t rows = 0;
    std::size_t size = 0;
    const core::GgmlType type = (*parts.begin())->type;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != type || t->cols != (*parts.begin())->cols ||
          (type != core::GgmlType::kF32 && type != core::GgmlType::kQ8_0)) {
        Fail("stacked upload needs F32 or Q8_0 tensors of one shape");
        return d;
      }
      rows += t->rows;
      size += t->SizeBytes();
    }
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    std::size_t offset = 0;
    for (const TensorRef* t : parts) {
      if (!stager.Copy(shard_base + t->shard, t->file_offset, t->SizeBytes(),
                       static_cast<std::uint8_t*>(ptr) + offset, error)) {
        Fail("upload failed for " + std::string(t->name));
        return d;
      }
      offset += t->SizeBytes();
    }
    if (type == core::GgmlType::kQ8_0) {
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                           kTailMargin, nullptr);
      d.data = ptr;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      return d;
    }
    const std::size_t count = size / sizeof(float);
    void* half = nullptr;
    if (hipMalloc(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    // Keep the small F32 stacks until the disk pipeline drains. Converting
    // each router immediately would serialize every layer's uploads.
    conversions.push_back({ptr, half, count});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(half) + count * 2, 0,
                         kTailMargin, nullptr);
    d.data = half;
    d.type = core::GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.linear = l.linear;
    d.hc_attn = Mixer(l.hc_attn);
    d.hc_ffn = Mixer(l.hc_ffn);
    // Projections of one input are stacked into one Q8_0 GEMV where the
    // quantization allows; otherwise they stay separate.
    const auto stackable = [](std::initializer_list<const TensorRef*> parts) {
      for (const TensorRef* t : parts) {
        if (t->empty() || t->type != core::GgmlType::kQ8_0 ||
            t->cols != (*parts.begin())->cols) {
          return false;
        }
      }
      return true;
    };
    if (l.linear && stackable({&l.ssm_qkv, &l.ssm_gate})) {
      d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
    } else {
      d.ssm_qkv = Copy(l.ssm_qkv);
      d.ssm_gate = Copy(l.ssm_gate);
    }
    d.ssm_conv1d = Copy(l.ssm_conv1d);
    if (l.linear) {
      d.ssm_alpha_beta = StackAlphaBeta(l.ssm_alpha, l.ssm_beta);
    }
    d.ssm_dt = Copy(l.ssm_dt);
    d.ssm_a = Copy(l.ssm_a);
    d.ssm_norm = Copy(l.ssm_norm);
    d.ssm_out = Copy(l.ssm_out);
    if (!l.linear && stackable({&l.attn_q, &l.attn_k, &l.attn_v})) {
      d.attn_qkv = Stack({&l.attn_q, &l.attn_k, &l.attn_v});
    } else {
      d.attn_q = Copy(l.attn_q);
      d.attn_k = Copy(l.attn_k);
      d.attn_v = Copy(l.attn_v);
    }
    d.attn_out = Copy(l.attn_out);
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.indexer_q = Copy(l.indexer_q);
    d.indexer_k = Copy(l.indexer_k);
    d.indexer_q_norm = Copy(l.indexer_q_norm);
    d.indexer_k_norm = Copy(l.indexer_k_norm);
    d.ple_key = Copy(l.ple_key);
    d.ple_value = Copy(l.ple_value);
    d.ple_norm_key = Copy(l.ple_norm_key);
    d.ple_norm_query = Copy(l.ple_norm_query);
    d.ple_norm_conv = Copy(l.ple_norm_conv);
    // The convolution reads F32 taps; an F16-stored kernel is decoded here.
    d.ple_conv1d = DecodeTensor(l.ple_conv1d, false);
    d.router = Stack({&l.router, &l.shexp_gate_inp});
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
    d.shexp_gate = Copy(l.shexp_gate);
    d.shexp_up = Copy(l.shexp_up);
    d.shexp_down = Copy(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    SplitMtpProjection(l.nextn_eh_proj, d.nextn_fc_embedding,
                       d.nextn_fc_hidden);
    d.nextn_head = Mixer(l.nextn_head);
    return d;
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const core::GgufReader& reader,
    const MtpWeights* mtp, const core::GgufReader* mtp_reader,
    std::string* error_msg) {
  // Every tensor must use a format its weight class accepts (the one table
  // in weight_format.hpp, applied per tensor at bind time). Revalidating here
  // keeps the failure ahead of any allocation, names the tensor and format,
  // and covers a ModelWeights that was not produced by Bind.
  if (!ValidateTensorFormats(w, error_msg)) {
    return nullptr;
  }
  if (mtp != nullptr && !ValidateTensorFormats(*mtp, error_msg)) {
    return nullptr;
  }
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (mtp != nullptr) {
    if (mtp_reader == nullptr) {
      if (error_msg)
        *error_msg = "MTP weights require their bound reader";
      return nullptr;
    }
    const auto extra = mtp_reader->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  std::vector<Conversion> conversions;
  std::vector<Decode> decodes;
  Uploader up{*stager,         conversions,       decodes,
              m->allocations_, m->bytes_,         m->max_half_cols_,
              m->max_q8_cols_, m->decoded_bytes_, &m->decoded_,
              m->uploaded_,    error_msg};
  m->token_embd_ = up.Copy(w.token_embd);
  m->output_ =
      w.output.data == w.token_embd.data ? m->token_embd_ : up.Copy(w.output);
  m->hc_head_ = up.Mixer(w.hc_head);
  m->layers_.reserve(w.layers.size());
  for (const auto& l : w.layers) {
    m->layers_.push_back(up.Layer(l));
    if (!up.ok) {
      return nullptr;
    }
  }
  if (mtp != nullptr) {
    // The sidecar has its own shard index; reuse the target's readers and
    // staging pool.
    up.shard_base = shard_count;
    m->mtp_ = up.Layer(mtp->block);
    m->has_mtp_ = true;
  }
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  for (const auto& c : conversions) {
    NarrowActivations(static_cast<const float*>(c.source), c.destination, false,
                      c.count, nullptr);
  }
  // Decode the tensors whose consumers are fixed-size F32/F16 kernels. The
  // staging copies stay resident until here, exactly like the F32 narrowings.
  for (const auto& d : decodes) {
    DecodeWeights(d.source, SmallType(d.type), d.destination, d.to_half,
                  d.elements, nullptr);
  }
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg =
          "weight conversion failed: " + std::string(hipGetErrorString(status));
    }
    return nullptr;
  }
  for (const auto& c : conversions) {
    std::erase(m->allocations_, c.source);
    (void)hipFree(c.source);
    m->bytes_ -= c.count * sizeof(float) + kTailMargin;
  }
  for (const auto& d : decodes) {
    std::erase(m->allocations_, d.source);
    (void)hipFree(d.source);
    m->bytes_ -=
        d.elements * WeightBlockBytes(d.type) / WeightBlockElements(d.type) +
        kTailMargin;
  }
  return m;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
