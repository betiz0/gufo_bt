// KVarN decode read checks (openspec add-kvarn-kv-cache-quantization 3.5,
// T-03): the sink, the sealed tiles and the F16 tail ring are read through one
// online softmax with the view rule of design D2, and the boundary context
// lengths of design D1 match an FP64 reference within a declared limit.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/kvarn.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/kvarn_reference.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/fp16.hpp"
#include "tests/testing/test_common.hpp"

namespace {

namespace ref = gufo::models::qwen;

constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kHeads = 4;
constexpr std::uint32_t kKvHeads = 1;
constexpr std::uint32_t kLayers = 1;
// Declared in QUALITY.md: the GPU keeps the online softmax in FP32 while the
// reference is FP64, so the outputs agree within this absolute limit.
constexpr double kOutputLimit = 2e-4;

float KeyValue(std::uint32_t position, std::uint32_t channel, bool is_value) {
  const float p = static_cast<float>(position);
  const float c = static_cast<float>(channel);
  return is_value ? 0.25F * std::cos(p * 0.05F + c * 0.017F)
                  : 0.3F * std::sin(p * 0.07F + c * 0.013F);
}

float QueryValue(std::uint32_t head, std::uint32_t channel) {
  return 0.2F * std::sin(static_cast<float>(head) * 0.3F +
                         static_cast<float>(channel) * 0.021F);
}

struct ReadHarness {
  gufo::hip::KvarnSealState seal{kLayers, kKvHeads};
  void* sink{nullptr};
  void* tail{nullptr};
  std::uint8_t* body{nullptr};
  std::uint32_t ring_rows{0};
  std::uint32_t tail_length{0};
  std::uint32_t tile_count{0};
  std::uint32_t k_tile_bytes{0};
  std::uint32_t v_tile_bytes{0};
  std::uint8_t k_bits{4};
  std::uint8_t v_bits{4};
  gufo::hip::QwenKvarnPreset preset{};

  ReadHarness(std::uint8_t k_b, std::uint8_t v_b, std::uint32_t tail_tokens,
              std::uint32_t max_context)
      : k_bits(k_b), v_bits(v_b) {
    preset.k_bits = k_b;
    preset.v_bits = v_b;
    preset.tail_length = tail_tokens;  // 0 = unset (L = W)
    tail_length = gufo::hip::QwenKvarnTailRows(preset);
    ring_rows = gufo::hip::QwenKvarnTailRingRows(preset);
    tile_count = (max_context - 128 + 127) / 128;
    k_tile_bytes = static_cast<std::uint32_t>(ref::KvarnTilePayloadBytes(k_b));
    v_tile_bytes = static_cast<std::uint32_t>(ref::KvarnTilePayloadBytes(v_b));
    const std::size_t sink_plane =
        static_cast<std::size_t>(kLayers) * kKvHeads * 128 * kHeadDim;
    const std::size_t tail_plane =
        static_cast<std::size_t>(kLayers) * kKvHeads * ring_rows * kHeadDim;
    HIP_CHECK(hipMalloc(&sink, 2 * sink_plane * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&tail, 2 * tail_plane * sizeof(std::uint16_t)));
    void* body_raw = nullptr;
    HIP_CHECK(hipMalloc(&body_raw, static_cast<std::size_t>(tile_count) *
                                       (k_tile_bytes + v_tile_bytes)));
    body = static_cast<std::uint8_t*>(body_raw);
    HIP_CHECK(hipMemset(sink, 0, 2 * sink_plane * sizeof(std::uint16_t)));
    HIP_CHECK(hipMemset(tail, 0, 2 * tail_plane * sizeof(std::uint16_t)));
    HIP_CHECK(hipMemset(
        body_raw, 0,
        static_cast<std::size_t>(tile_count) * (k_tile_bytes + v_tile_bytes)));
  }

  ~ReadHarness() {
    (void)hipFree(sink);
    (void)hipFree(tail);
    (void)hipFree(body);
  }

  /// Writes positions `[0, count)` into the sink or the ring, in both planes.
  void Fill(std::uint32_t count) {
    const std::size_t sink_plane =
        static_cast<std::size_t>(kLayers) * kKvHeads * 128 * kHeadDim;
    const std::size_t tail_plane =
        static_cast<std::size_t>(kLayers) * kKvHeads * ring_rows * kHeadDim;
    for (std::uint32_t position = 0; position < count; ++position) {
      for (const bool is_value : {false, true}) {
        std::vector<std::uint16_t> row(kHeadDim);
        for (std::uint32_t c = 0; c < kHeadDim; ++c) {
          row[c] = gufo::test::FloatToFp16Bits(KeyValue(position, c, is_value));
        }
        std::uint8_t* destination = nullptr;
        std::size_t plane = 0;
        std::size_t offset = 0;
        if (position < 128) {
          destination = static_cast<std::uint8_t*>(sink);
          plane = is_value ? sink_plane : 0;
          offset = static_cast<std::size_t>(position) * kHeadDim;
        } else {
          destination = static_cast<std::uint8_t*>(tail);
          plane = is_value ? tail_plane : 0;
          offset = ((position - 128) % ring_rows) * kHeadDim;
        }
        HIP_CHECK(hipMemcpy(
            destination + (plane + offset) * sizeof(std::uint16_t), row.data(),
            row.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice));
      }
    }
  }

  [[nodiscard]] std::size_t PlaneElements(bool is_value) const {
    const std::size_t plane = is_value ? static_cast<std::size_t>(kLayers) *
                                             kKvHeads * ring_rows * kHeadDim
                                       : static_cast<std::size_t>(kLayers) *
                                             kKvHeads * 128 * kHeadDim;
    return plane;
  }

  void Seal(std::uint32_t pos) {
    gufo::hip::LaunchKvarnSeal(seal, preset, sink, tail, PlaneElements(false),
                               PlaneElements(true), ring_rows, body, tile_count,
                               /*layer_idx=*/0, kKvHeads, pos, nullptr,
                               /*count=*/1, nullptr);
  }

  [[nodiscard]] ref::KvarnTilePayload Payload(std::uint32_t tile,
                                              bool is_value) const {
    ref::KvarnTilePayload payload{};
    const std::size_t offset =
        (static_cast<std::size_t>(tile)) * (k_tile_bytes + v_tile_bytes) +
        (is_value ? k_tile_bytes : 0);
    HIP_CHECK(hipMemcpy(&payload, body + offset, sizeof(payload),
                        hipMemcpyDeviceToHost));
    return payload;
  }

  [[nodiscard]] gufo::hip::KvarnReadParams ReadParams() const {
    gufo::hip::KvarnReadParams params;
    params.sink = static_cast<const __half*>(sink);
    params.tail = static_cast<const __half*>(tail);
    params.body = body;
    params.sink_plane_elements = PlaneElements(false);
    params.tail_plane_elements = PlaneElements(true);
    params.ring_rows = ring_rows;
    params.tile_count = tile_count;
    params.k_tile_bytes = k_tile_bytes;
    params.v_tile_bytes = v_tile_bytes;
    params.kv_heads = kKvHeads;
    params.tail_length = tail_length;
    params.k_bits = k_bits;
    params.v_bits = v_bits;
    return params;
  }
};

/// FP64 reference for one query position: the same three-region view rule and
/// the same decoded values, but exact softmax arithmetic.
std::vector<double> Reference(const ReadHarness& harness, std::uint32_t pos,
                              std::uint32_t head) {
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  std::vector<double> scores(pos + 1);
  double running_max = -std::numeric_limits<double>::infinity();
  for (std::uint32_t position = 0; position <= pos; ++position) {
    const bool quantized =
        position >= 128 &&
        (static_cast<double>(128 + 128 * ((position - 128) / 128) + 127) +
             harness.tail_length + 1.0 <=
         static_cast<double>(pos));
    double score = 0.0;
    if (!quantized) {
      for (std::uint32_t c = 0; c < kHeadDim; ++c) {
        const float key = KeyValue(position, c, false);
        score += static_cast<double>(QueryValue(head, c)) *
                 static_cast<double>(__half2float(__float2half_rn(key)));
      }
    } else {
      const std::uint32_t tile = (position - 128) / 128;
      const std::uint32_t token = position - 128 - 128 * tile;
      const auto payload = harness.Payload(tile, false);
      for (std::uint32_t c = 0; c < kHeadDim; ++c) {
        const double decoded = static_cast<double>(ref::KvarnDecodeTileElement(
            payload, ref::KvarnRtnAxis::kChannel, harness.k_bits, token, c));
        const double channel_scale =
            gufo::quant::Fp16ToFloat(payload.norm_channel_scale[c]);
        const double token_scale =
            gufo::quant::Fp16ToFloat(payload.norm_token_scale[token]);
        score += static_cast<double>(QueryValue(head, c)) * decoded *
                 channel_scale * token_scale;
      }
    }
    scores[position] = score * static_cast<double>(scale);
    running_max = std::max(running_max, scores[position]);
  }
  double sum = 0.0;
  for (std::uint32_t position = 0; position <= pos; ++position) {
    scores[position] = std::exp(scores[position] - running_max);
    sum += scores[position];
  }
  std::vector<double> out(kHeadDim, 0.0);
  for (std::uint32_t position = 0; position <= pos; ++position) {
    const double weight = scores[position] / sum;
    const bool quantized =
        position >= 128 &&
        (static_cast<double>(128 + 128 * ((position - 128) / 128) + 127) +
             harness.tail_length + 1.0 <=
         static_cast<double>(pos));
    if (!quantized) {
      for (std::uint32_t c = 0; c < kHeadDim; ++c) {
        const float value = KeyValue(position, c, true);
        out[c] +=
            weight * static_cast<double>(__half2float(__float2half_rn(value)));
      }
    } else {
      const std::uint32_t tile = (position - 128) / 128;
      const std::uint32_t token = position - 128 - 128 * tile;
      const auto payload = harness.Payload(tile, true);
      for (std::uint32_t c = 0; c < kHeadDim; ++c) {
        const double decoded = static_cast<double>(ref::KvarnDecodeTileElement(
            payload, ref::KvarnRtnAxis::kTokenGroup, harness.v_bits, token, c));
        const double channel_scale =
            gufo::quant::Fp16ToFloat(payload.norm_channel_scale[c]);
        const double token_scale =
            gufo::quant::Fp16ToFloat(payload.norm_token_scale[token]);
        out[c] += weight * decoded * channel_scale * token_scale;
      }
    }
  }
  // The KVarN read inverse-rotates the merged context before the SiLU gate
  // (design D3), so the reference applies the same normalized Hadamard.
  std::vector<float> rotated(kHeadDim);
  for (std::uint32_t c = 0; c < kHeadDim; ++c) {
    rotated[c] = static_cast<float>(out[c]);
  }
  ref::KvarnFwhtHeadInPlace(rotated);
  for (std::uint32_t c = 0; c < kHeadDim; ++c) {
    out[c] = static_cast<double>(rotated[c]);
  }
  return out;
}

/// Runs one decode query and returns its context. `split_k` supplies the
/// `DecodeAttentionScratchElements` scratch, selecting the tile-aligned split-K
/// path; `device_pos` supplies the row position from device memory, as the HIP
/// graph capture path does.
std::vector<float> RunQuery(const ReadHarness& harness, std::uint32_t pos,
                            bool split_k, bool device_pos) {
  std::vector<float> q(static_cast<std::size_t>(kHeads) * kHeadDim);
  for (std::uint32_t head = 0; head < kHeads; ++head) {
    for (std::uint32_t c = 0; c < kHeadDim; ++c) {
      q[head * kHeadDim + c] = QueryValue(head, c);
    }
  }
  float* d_q = nullptr;
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_q, q.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, q.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_q, q.data(), q.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  float* d_split = nullptr;
  if (split_k) {
    HIP_CHECK(hipMalloc(
        &d_split,
        gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kHeadDim) *
            sizeof(float)));
  }
  std::uint32_t* d_pos = nullptr;
  if (device_pos) {
    HIP_CHECK(hipMalloc(&d_pos, sizeof(std::uint32_t)));
    HIP_CHECK(hipMemcpy(d_pos, &pos, sizeof(pos), hipMemcpyHostToDevice));
  }
  gufo::hip::LaunchKvarnDecodeAttention(
      d_q, nullptr, d_out, /*layer_idx=*/0, pos, d_pos, /*batch_size=*/1,
      kHeads, kKvHeads, harness.ReadParams(), d_split, nullptr);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> out(q.size());
  HIP_CHECK(hipMemcpy(out.data(), d_out, out.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_out));
  if (d_split != nullptr) {
    HIP_CHECK(hipFree(d_split));
  }
  if (d_pos != nullptr) {
    HIP_CHECK(hipFree(d_pos));
  }
  return out;
}

void RunCase(std::uint8_t k_bits, std::uint8_t v_bits, std::uint32_t tail,
             std::uint32_t pos) {
  const std::uint32_t max_context = pos + 1;
  ReadHarness harness(k_bits, v_bits, tail, max_context);
  harness.Fill(pos + 1);
  harness.Seal(pos);

  const std::vector<float> out = RunQuery(harness, pos, /*split_k=*/false,
                                          /*device_pos=*/false);

  double worst = 0.0;
  for (std::uint32_t head = 0; head < kHeads; ++head) {
    const std::vector<double> expected = Reference(harness, pos, head);
    for (std::uint32_t c = 0; c < kHeadDim; ++c) {
      worst = std::max(worst,
                       std::abs(static_cast<double>(out[head * kHeadDim + c]) -
                                expected[c]));
    }
  }
  if (worst > kOutputLimit) {
    throw std::runtime_error("T-03: context " + std::to_string(pos) + " tail " +
                             std::to_string(tail) +
                             " differs from the FP64 reference by " +
                             std::to_string(worst));
  }
  // The split-K path and the device-supplied position must reproduce the same
  // context within the declared limit (design D8; graph capture supplies pos
  // from device memory).
  const std::vector<float> split = RunQuery(harness, pos, /*split_k=*/true,
                                            /*device_pos=*/false);
  const std::vector<float> device = RunQuery(harness, pos, /*split_k=*/false,
                                             /*device_pos=*/true);
  double worst_split = 0.0;
  double worst_device = 0.0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    worst_split = std::max(
        worst_split,
        std::abs(static_cast<double>(out[i]) - static_cast<double>(split[i])));
    worst_device = std::max(
        worst_device,
        std::abs(static_cast<double>(out[i]) - static_cast<double>(device[i])));
  }
  if (worst_split > kOutputLimit || worst_device > kOutputLimit) {
    throw std::runtime_error(
        "T-03: context " + std::to_string(pos) + " tail " +
        std::to_string(tail) + " split-K/device-pos disagree by " +
        std::to_string(std::max(worst_split, worst_device)));
  }
  std::cout << "T-03 context " << pos << " tail " << tail << " worst " << worst
            << " split " << worst_split << " device_pos " << worst_device
            << '\n';
}

/// Runs `batch` consecutive verification rows through the KVarN causal path
/// (T-05). The rows share one LDS staging block but each keeps the AR decode's
/// tile-aligned split partition.
std::vector<float> RunCausal(const ReadHarness& harness, std::uint32_t start,
                             std::uint32_t batch) {
  const std::size_t width = static_cast<std::size_t>(kHeads) * kHeadDim;
  std::vector<float> q(batch * width);
  for (std::uint32_t row = 0; row < batch; ++row) {
    for (std::uint32_t head = 0; head < kHeads; ++head) {
      for (std::uint32_t c = 0; c < kHeadDim; ++c) {
        q[row * width + head * kHeadDim + c] = QueryValue(head, c);
      }
    }
  }
  float* d_q = nullptr;
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_q, q.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, q.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_q, q.data(), q.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  const auto row_scratch =
      gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kHeadDim);
  float* d_scratch = nullptr;
  HIP_CHECK(hipMalloc(&d_scratch, batch * row_scratch * sizeof(float)));
  gufo::hip::LaunchCausalDecodeAttention(
      d_q, nullptr, nullptr, nullptr, nullptr, nullptr, d_out, /*layer_idx=*/0,
      start, batch, /*max_context=*/start + batch, kHeads, kKvHeads, kHeadDim,
      nullptr, {d_scratch, batch * row_scratch}, harness.ReadParams());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> out(q.size());
  HIP_CHECK(hipMemcpy(out.data(), d_out, out.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_scratch));
  return out;
}

/// T-05: a batch of verification rows must produce the AR decode's logits bit
/// for bit, including split boundaries and rows spanning the seal boundary.
void RunCausalCase(std::uint8_t k_bits, std::uint8_t v_bits, std::uint32_t tail,
                   std::uint32_t start, std::uint32_t batch) {
  ReadHarness harness(k_bits, v_bits, tail, start + batch);
  harness.Fill(start + batch);
  harness.Seal(start + batch - 1);

  const std::size_t width = static_cast<std::size_t>(kHeads) * kHeadDim;
  std::vector<float> expected(batch * width);
  for (std::uint32_t row = 0; row < batch; ++row) {
    const auto ar = RunQuery(harness, start + row, /*split_k=*/true,
                             /*device_pos=*/false);
    std::copy(ar.begin(), ar.end(), expected.begin() + row * width);
  }
  const std::vector<float> causal = RunCausal(harness, start, batch);
  if (std::memcmp(expected.data(), causal.data(),
                  expected.size() * sizeof(float)) != 0) {
    double worst = 0.0;
    std::size_t worst_index = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const double diff = std::abs(static_cast<double>(expected[i]) -
                                   static_cast<double>(causal[i]));
      if (diff > worst) {
        worst = diff;
        worst_index = i;
      }
    }
    throw std::runtime_error(
        "T-05: causal verification rows differ from AR decode (start " +
        std::to_string(start) + " batch " + std::to_string(batch) + " tail " +
        std::to_string(tail) + ") worst " + std::to_string(worst) + " at " +
        std::to_string(worst_index) + " ar " +
        std::to_string(expected[worst_index]) + " causal " +
        std::to_string(causal[worst_index]));
  }
  std::cout << "T-05 start " << start << " batch " << batch << " tail " << tail
            << " bit-identical\n";
}

/// T-05: the seal predicate must seal only committed tiles, never a tile that
/// holds a row of the verification window (rejected rows must not be sealed).
void RunCausalSealBoundary(std::uint8_t k_bits, std::uint8_t v_bits,
                           std::uint32_t tail, std::uint32_t start,
                           std::uint32_t batch) {
  ReadHarness harness(k_bits, v_bits, tail, start + batch);
  harness.Fill(start + batch);
  harness.Seal(start + batch - 1);

  std::uint32_t sealed = 0;
  HIP_CHECK(hipMemcpy(&sealed, harness.seal.sealed(), sizeof(sealed),
                      hipMemcpyDeviceToHost));
  const std::uint32_t q_max = start + batch - 1;
  const std::uint32_t length = harness.tail_length;
  const std::uint32_t expected =
      q_max >= 256 + length ? (q_max - length - 256) / 128 + 1 : 0;
  if (sealed != expected) {
    throw std::runtime_error(
        "T-05: seal boundary tail " + std::to_string(tail) + " start " +
        std::to_string(start) + " sealed " + std::to_string(sealed) +
        " expected " + std::to_string(expected));
  }
  if (sealed > 0) {
    const std::uint32_t e_t = 128 + 128 * (sealed - 1) + 127;
    if (e_t + length + 1 > q_max) {
      throw std::runtime_error("T-05: sealed tile beyond the view boundary");
    }
    if (e_t >= start) {
      throw std::runtime_error(
          "T-05: sealed tile contains a verification-window row");
    }
    if (sealed < harness.tile_count) {
      const auto payload = harness.Payload(sealed, false);
      if (payload.rtn_scale[0] != 0 || payload.norm_channel_scale[0] != 0 ||
          payload.norm_token_scale[0] != 0) {
        throw std::runtime_error(
            "T-05: tile holding window rows was written by the seal");
      }
    }
  }
  std::cout << "T-05 seal boundary start " << start << " batch " << batch
            << " tail " << tail << " sealed " << sealed << '\n';
}

/// Design D7 / tasks 3.13: a KVarN policy leaves the canonical FP16/FP32
/// planes null, so reaching a non-KVarN attention launcher is a wiring error
/// that must throw rather than silently read a null plane. `kvarn` here is a
/// disabled `KvarnReadParams` (null sink), which cannot substitute for them.
void RunNullPlaneGuard() {
  float* d_q = nullptr;
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_q, 4 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, 4 * sizeof(float)));
  const auto expect_rejected = [](auto&& launch, const char* label) {
    bool rejected = false;
    try {
      launch();
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error(
          std::string("null KV planes must be rejected by ") + label);
  };
  // Position form, device-position (graph) form, and the verification-row form
  // with a disabled KVarN source.
  expect_rejected(
      [&] {
        gufo::hip::LaunchAttention(
            d_q, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            d_out, /*layer_idx=*/0, /*pos=*/0, /*max_context=*/4, kHeads,
            kKvHeads, kHeadDim, nullptr, nullptr, /*skip_kv_write=*/false);
      },
      "LaunchAttention");
  expect_rejected(
      [&] {
        gufo::hip::LaunchAttention(
            d_q, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            d_out, /*layer_idx=*/0, /*d_pos=*/nullptr, /*max_context=*/4,
            kHeads, kKvHeads, kHeadDim, nullptr, /*skip_kv_write=*/false);
      },
      "LaunchAttention(d_pos)");
  expect_rejected(
      [&] {
        gufo::hip::LaunchCausalDecodeAttention(
            d_q, nullptr, nullptr, nullptr, nullptr, nullptr, d_out,
            /*layer_idx=*/0, /*start_pos=*/0, /*batch_size=*/1,
            /*max_context=*/4, kHeads, kKvHeads, kHeadDim, nullptr, {},
            gufo::hip::KvarnReadParams{});
      },
      "LaunchCausalDecodeAttention");
  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_out));
  std::cout << "null KV planes rejected by every attention launcher\n";
}

}  // namespace

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "Qwen KVarN read ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }
  RunNullPlaneGuard();
  // Design D1 boundary cases: the sink edge, the first tail row, and the first
  // seal boundary for the unset tail (L = W = 9) and L = 2048.
  RunCase(4, 4, 0, 0);
  for (const std::uint32_t pos : {1U, 2U, 4U, 8U, 16U, 32U, 64U}) {
    RunCase(4, 4, 0, pos);
  }
  for (const std::uint32_t tail : {0U, 2048U}) {
    const std::uint32_t L = tail == 0 ? 9 : tail;
    // Design D1 boundary contexts: the sink edge, the first tail row, and the
    // first seal boundary at 128+L and 256+L, including the split boundary at
    // 256 (a multiple of the 128-token tile).
    for (const std::uint32_t pos :
         {127U, 128U, 129U, 128 + L + 127, 128 + L + 128, 128 + L + 129,
          256 + L + 127, 256 + L + 128, 256 + L + 129}) {
      RunCase(4, 4, tail, pos);
    }
    // T-05 verification rows: bit-identical to AR decode across split and seal
    // boundaries, for batch widths 2 through 8.
    for (const std::uint32_t batch : {1U, 2U, 3U, 5U, 8U}) {
      for (const std::uint32_t start :
           {0U, 17U, 100U, 120U, 124U, 127U, 128U, 129U, 200U, 255U, 256U, 384U,
            128 + L + 120, 128 + L + 127, 128 + L + 128, 256 + L + 120,
            256 + L + 127, 256 + L + 128}) {
        RunCausalCase(4, 4, tail, start, batch);
      }
      RunCausalSealBoundary(4, 4, tail, 128 + L + 64, batch);
    }
    for (const std::uint32_t batch : {1U, 8U}) {
      for (const std::uint32_t start : {1024U, 2048U, 4096U}) {
        RunCausalCase(4, 4, tail, start, batch);
      }
    }
  }
  std::cout << "Qwen KVarN read ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen KVarN read ops test.\n";
  return 77;
#endif
}
#endif  // defined(ENGINE_ENABLE_HIP)
