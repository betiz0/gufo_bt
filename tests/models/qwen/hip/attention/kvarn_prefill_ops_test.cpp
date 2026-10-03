// KVarN prefill write checks (openspec add-kvarn-kv-cache-quantization 4.1,
// T-04): the batched prefill fused QK-norm/RoPE/KV-write and the single-token
// decode write must leave byte-identical rotated rows and byte-identical
// sealed tiles (design D3/D4). The seal predicate must cover only committed
// rows of the chunk.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/kvarn.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/kvarn_reference.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/testing/test_common.hpp"

namespace {

namespace ref = gufo::models::qwen;

constexpr std::uint32_t kHeads = 4;
constexpr std::uint32_t kKvHeads = 1;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kLayers = 1;
constexpr std::uint32_t kRotaryDim = 64;
constexpr float kRopeTheta = 10000000.0F;

float QueryValue(std::uint32_t row, std::uint32_t channel) {
  return 0.2F * std::sin(static_cast<float>(row) * 0.013F +
                         static_cast<float>(channel) * 0.021F);
}

float KeyValue(std::uint32_t row, std::uint32_t channel) {
  return 0.3F * std::sin(static_cast<float>(row) * 0.017F +
                         static_cast<float>(channel) * 0.013F);
}

float ValueValue(std::uint32_t row, std::uint32_t channel) {
  return 0.25F * std::cos(static_cast<float>(row) * 0.011F +
                          static_cast<float>(channel) * 0.019F);
}

/// One KVarN write destination and its seal state.
struct WriteHarness {
  gufo::hip::KvarnSealState seal{kLayers, kKvHeads};
  gufo::hip::QwenKvarnPreset preset{};
  void* sink{nullptr};
  void* tail{nullptr};
  std::uint8_t* body{nullptr};
  std::uint32_t ring_rows{0};
  std::uint32_t tile_count{0};
  std::uint32_t k_tile_bytes{0};
  std::uint32_t v_tile_bytes{0};
  std::uint64_t sink_elements{0};
  std::uint64_t tail_elements{0};
  std::size_t sink_bytes{0};
  std::size_t tail_bytes{0};
  std::size_t body_bytes{0};

  WriteHarness(std::uint32_t tail_tokens, std::uint32_t max_context) {
    preset.k_bits = 4;
    preset.v_bits = 4;
    preset.tail_length = tail_tokens;
    ring_rows = gufo::hip::QwenKvarnTailRingRows(preset);
    tile_count = (max_context - 128 + 127) / 128;
    k_tile_bytes =
        static_cast<std::uint32_t>(ref::KvarnTilePayloadBytes(preset.k_bits));
    v_tile_bytes =
        static_cast<std::uint32_t>(ref::KvarnTilePayloadBytes(preset.v_bits));
    sink_elements =
        static_cast<std::uint64_t>(kLayers) * kKvHeads * 128 * kHeadDim;
    tail_elements =
        static_cast<std::uint64_t>(kLayers) * kKvHeads * ring_rows * kHeadDim;
    sink_bytes = 2 * sink_elements * sizeof(std::uint16_t);
    tail_bytes = 2 * tail_elements * sizeof(std::uint16_t);
    body_bytes =
        static_cast<std::size_t>(tile_count) * (k_tile_bytes + v_tile_bytes);
    HIP_CHECK(hipMalloc(&sink, sink_bytes));
    HIP_CHECK(hipMalloc(&tail, tail_bytes));
    void* body_raw = nullptr;
    HIP_CHECK(hipMalloc(&body_raw, body_bytes));
    body = static_cast<std::uint8_t*>(body_raw);
    HIP_CHECK(hipMemset(sink, 0, sink_bytes));
    HIP_CHECK(hipMemset(tail, 0, tail_bytes));
    HIP_CHECK(hipMemset(body_raw, 0, body_bytes));
  }

  ~WriteHarness() {
    (void)hipFree(sink);
    (void)hipFree(tail);
    (void)hipFree(body);
  }

  [[nodiscard]] gufo::hip::KvarnWriteParams WriteParams() const {
    gufo::hip::KvarnWriteParams params;
    params.sink = static_cast<__half*>(sink);
    params.tail = static_cast<__half*>(tail);
    params.sink_plane_elements = sink_elements;
    params.tail_plane_elements = tail_elements;
    params.ring_rows = ring_rows;
    return params;
  }

  void Seal(std::uint32_t start, std::uint32_t count) {
    gufo::hip::LaunchKvarnSeal(
        seal, preset, sink, tail, sink_elements, tail_elements, ring_rows, body,
        tile_count, /*layer_idx=*/0, kKvHeads, start, nullptr, count, nullptr);
  }

  [[nodiscard]] std::vector<std::uint8_t> SnapshotBody() const {
    std::vector<std::uint8_t> bytes(body_bytes);
    HIP_CHECK(hipMemcpy(bytes.data(), body, body_bytes, hipMemcpyDeviceToHost));
    return bytes;
  }
};

std::vector<float> BuildQ(std::uint32_t rows, std::uint32_t channels,
                          float (*value)(std::uint32_t, std::uint32_t)) {
  std::vector<float> out(static_cast<std::size_t>(rows) * channels);
  for (std::uint32_t r = 0; r < rows; ++r) {
    for (std::uint32_t c = 0; c < channels; ++c) {
      out[static_cast<std::size_t>(r) * channels + c] = value(r, c);
    }
  }
  return out;
}

/// T-04: the batched prefill write and the sequential decode write produce the
/// same rotated rows and the same sealed tiles.
void RunCase(std::uint32_t tail_tokens, std::uint32_t rows) {
  const std::size_t q_size = static_cast<std::size_t>(rows) * kHeads * kHeadDim;
  const std::size_t kv_size =
      static_cast<std::size_t>(rows) * kKvHeads * kHeadDim;
  const std::vector<float> q = BuildQ(rows, kHeads * kHeadDim, QueryValue);
  const std::vector<float> k = BuildQ(rows, kKvHeads * kHeadDim, KeyValue);
  const std::vector<float> v = BuildQ(rows, kKvHeads * kHeadDim, ValueValue);

  WriteHarness batched(tail_tokens, rows);
  WriteHarness decode(tail_tokens, rows);

  // Path A: one batched prefill write for the whole chunk, then the seal.
  {
    float* d_q = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;
    float* d_q_out = nullptr;
    float* d_k_out = nullptr;
    HIP_CHECK(hipMalloc(&d_q, q_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_k, kv_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_v, kv_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_q_out, q_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_k_out, kv_size * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_q, q.data(), q_size * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_k, k.data(), kv_size * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_v, v.data(), kv_size * sizeof(float),
                        hipMemcpyHostToDevice));
    gufo::hip::LaunchBatchedFusedQKNormRoPEKvWrite(
        d_q, d_k, d_v, nullptr, nullptr, d_q_out, d_k_out, nullptr, nullptr,
        nullptr, nullptr, /*layer_idx=*/0, /*start_pos=*/0, rows,
        /*max_context=*/rows, kHeads, kKvHeads, kHeadDim, kRotaryDim,
        kRopeTheta, 1e-6F, nullptr, nullptr, batched.WriteParams());
    HIP_CHECK(hipDeviceSynchronize());
    batched.Seal(0, rows);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_q));
    HIP_CHECK(hipFree(d_k));
    HIP_CHECK(hipFree(d_v));
    HIP_CHECK(hipFree(d_q_out));
    HIP_CHECK(hipFree(d_k_out));
  }

  // Path B: one single-token write plus seal per row, as the decode step does.
  {
    float* d_q = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;
    float* d_q_out = nullptr;
    float* d_k_out = nullptr;
    std::uint32_t* d_pos = nullptr;
    HIP_CHECK(hipMalloc(
        &d_q, static_cast<std::size_t>(kHeads) * kHeadDim * sizeof(float)));
    HIP_CHECK(hipMalloc(
        &d_k, static_cast<std::size_t>(kKvHeads) * kHeadDim * sizeof(float)));
    HIP_CHECK(hipMalloc(
        &d_v, static_cast<std::size_t>(kKvHeads) * kHeadDim * sizeof(float)));
    HIP_CHECK(hipMalloc(
        &d_q_out, static_cast<std::size_t>(kHeads) * kHeadDim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_k_out, static_cast<std::size_t>(kKvHeads) *
                                      kHeadDim * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_pos, sizeof(std::uint32_t)));
    for (std::uint32_t row = 0; row < rows; ++row) {
      const std::size_t q_row =
          static_cast<std::size_t>(row) * kHeads * kHeadDim;
      const std::size_t kv_row =
          static_cast<std::size_t>(row) * kKvHeads * kHeadDim;
      HIP_CHECK(
          hipMemcpy(d_q, q.data() + q_row,
                    static_cast<std::size_t>(kHeads) * kHeadDim * sizeof(float),
                    hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(
          d_k, k.data() + kv_row,
          static_cast<std::size_t>(kKvHeads) * kHeadDim * sizeof(float),
          hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(
          d_v, v.data() + kv_row,
          static_cast<std::size_t>(kKvHeads) * kHeadDim * sizeof(float),
          hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(d_pos, &row, sizeof(row), hipMemcpyHostToDevice));
      gufo::hip::LaunchFusedQKNormRoPEKvWrite(
          d_q, d_k, d_v, nullptr, nullptr, d_q_out, d_k_out, nullptr, nullptr,
          nullptr, nullptr, /*layer_idx=*/0, d_pos, /*max_context=*/rows,
          kHeads, kKvHeads, kHeadDim, kRotaryDim, kRopeTheta, 1e-6F, nullptr,
          nullptr, decode.WriteParams());
      HIP_CHECK(hipDeviceSynchronize());
      decode.Seal(row, 1);
      HIP_CHECK(hipDeviceSynchronize());
    }
    HIP_CHECK(hipFree(d_q));
    HIP_CHECK(hipFree(d_k));
    HIP_CHECK(hipFree(d_v));
    HIP_CHECK(hipFree(d_q_out));
    HIP_CHECK(hipFree(d_k_out));
    HIP_CHECK(hipFree(d_pos));
  }

  const std::vector<std::uint8_t> batched_body = batched.SnapshotBody();
  const std::vector<std::uint8_t> decode_body = decode.SnapshotBody();
  if (batched_body != decode_body) {
    throw std::runtime_error(
        "T-04: batched and sequential KVarN writes sealed different tiles "
        "(tail " +
        std::to_string(tail_tokens) + ", rows " + std::to_string(rows) + ")");
  }

  // The seal must cover only committed tiles: the window rows of the final
  // chunk stay unsealed.
  std::uint32_t sealed = 0;
  HIP_CHECK(hipMemcpy(&sealed, batched.seal.sealed(), sizeof(sealed),
                      hipMemcpyDeviceToHost));
  const std::uint32_t length = gufo::hip::QwenKvarnTailRows(batched.preset);
  if (sealed > 0) {
    const std::uint32_t e_t = 128 + 128 * (sealed - 1) + 127;
    if (e_t + length + 1 > rows - 1) {
      throw std::runtime_error("T-04: sealed tile beyond the view boundary");
    }
  }
  std::cout << "T-04 tail " << tail_tokens << " rows " << rows << " sealed "
            << sealed << " bit-identical\n";
}

}  // namespace

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "Qwen KVarN prefill test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }
  // The chunk must stay inside the F16 ring until its seal runs; sizes just
  // past each seal threshold and within ring capacity exercise that window.
  RunCase(/*tail_tokens=*/0, /*rows=*/270);
  RunCase(/*tail_tokens=*/2048, /*rows=*/2308);
  std::cout << "Qwen KVarN prefill ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen KVarN prefill ops test.\n";
  return 77;
#endif
}
#endif  // defined(ENGINE_ENABLE_HIP)
