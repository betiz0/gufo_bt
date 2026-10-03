#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/detail/hip_graph_decode_executor.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/hip/snapshot_transfer.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/kvarn.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/kvarn_reference.hpp"
#include "src/models/qwen/modules/ffn.hpp"
#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/modules/norm.hpp"
#include "src/models/qwen/modules/quant_gemm.hpp"
#include "src/models/qwen/modules/residual.hpp"
#include "tests/models/qwen/hip/support/bfloat16.hpp"
#include "tests/models/qwen/hip/support/comparisons.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

void TestHipGraphDecodeStep() {
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  constexpr std::size_t hidden_size = 256;
  constexpr std::size_t vocab_size = 1024;
  std::vector<float> h_embd(vocab_size * hidden_size, 1.0F);
  for (std::size_t i = 0; i < vocab_size * hidden_size; ++i) {
    h_embd[i] = 0.01F * static_cast<float>(i % 37);
  }

  void* d_embd = nullptr;
  float* d_hidden = nullptr;
  float* d_normed = nullptr;
  float* d_weight = nullptr;
  std::uint32_t* d_params = nullptr;

  HIP_CHECK(hipMalloc(&d_embd, vocab_size * hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_hidden, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_normed, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_weight, hidden_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_params, 4 * sizeof(std::uint32_t)));

  std::vector<float> h_w(hidden_size, 1.5F);
  HIP_CHECK(hipMemcpy(d_embd, h_embd.data(),
                      vocab_size * hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_weight, h_w.data(), hidden_size * sizeof(float),
                      hipMemcpyHostToDevice));

  gufo::hip::detail::HipGraphDecodeExecutor executor;
  constexpr gufo::hip::detail::HipGraphCaptureKey graph_key{
      .execution_identity = 11,
      .workload_identity = 29,
  };
  constexpr gufo::hip::detail::HipGraphCaptureKey mismatched_key{
      .execution_identity = 11,
      .workload_identity = 31,
  };
  gufo::test::Expect(executor.IsEnabled(), "HIP graph executor disabled");
  gufo::test::Expect(!executor.IsCaptured(),
                     "new HIP graph executor is already captured");

  bool threw = false;
  try {
    executor.TryCapture(stream, graph_key,
                        [] { HIP_CHECK(hipErrorOutOfMemory); });
  } catch (const std::runtime_error&) {
    threw = true;
  }
  gufo::test::Expect(threw, "capture failure must propagate");
  hipStreamCaptureStatus capture = hipStreamCaptureStatusActive;
  HIP_CHECK(hipStreamIsCapturing(stream, &capture));
  gufo::test::Expect(capture == hipStreamCaptureStatusNone,
                     "failed capture must release the stream");

  // Test across 5 consecutive tokens with graph replay
  for (std::uint32_t step = 0; step < 5; ++step) {
    const std::uint32_t token_id = step * 10 + 3;
    const std::uint32_t pos = step;
    const std::uint32_t in_params[2] = {token_id, pos};
    HIP_CHECK(hipMemcpyAsync(d_params, in_params, sizeof(in_params),
                             hipMemcpyHostToDevice, stream));

    auto StepOps = [&]() {
      gufo::hip::LaunchEmbeddingLookup(d_embd, gufo::core::GgmlType::kF32,
                                       d_params + 0, d_hidden, hidden_size,
                                       stream);
      gufo::hip::LaunchRMSNorm(d_hidden, d_weight, d_normed, hidden_size, 1e-6F,
                               stream);
    };

    if (!executor.IsCaptured()) {
      const bool ok = executor.TryCapture(stream, graph_key, [&] {
        StepOps();
        // A different request may allocate/copy a snapshot during capture.
        // Global capture or a blocking stream makes these independent copies
        // invalidate the decode graph (previously hidden by HIP_CHECK).
        std::async(std::launch::async, [&] {
          void* independent = nullptr;
          HIP_CHECK(hipMalloc(&independent, hidden_size * sizeof(float)));
          try {
            gufo::hip::SnapshotTransfer transfer;
            transfer.Copy(independent, h_w.data(), hidden_size * sizeof(float),
                          hipMemcpyHostToDevice);
            std::vector<float> restored(hidden_size);
            transfer.Copy2D(restored.data(), 64 * sizeof(float), independent,
                            64 * sizeof(float), 64 * sizeof(float),
                            hidden_size / 64);
            gufo::test::Expect(restored == h_w,
                               "snapshot copy during capture changed values");
          } catch (...) {
            (void)hipFree(independent);
            throw;
          }
          HIP_CHECK(hipFree(independent));
        }).get();
      });
      gufo::test::Expect(ok, "HIP graph capture failed");
      gufo::test::Expect(executor.IsCapturedFor(graph_key),
                         "HIP graph capture key was not retained");
      gufo::test::Expect(!executor.IsCapturedFor(mismatched_key),
                         "mismatched HIP graph key was accepted");
      gufo::test::Expect(!executor.Launch(stream, mismatched_key),
                         "HIP graph replay accepted a mismatched key");
      const bool launch_ok = executor.Launch(stream, graph_key);
      gufo::test::Expect(launch_ok, "HIP graph launch failed");
    } else {
      const bool ok = executor.Launch(stream, graph_key);
      gufo::test::Expect(ok, "HIP graph replay failed");
    }

    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> h_out(hidden_size);
    HIP_CHECK(hipMemcpy(h_out.data(), d_normed, hidden_size * sizeof(float),
                        hipMemcpyDeviceToHost));

    // Verify RMSNorm output
    float sum_sq = 0.0F;
    for (std::size_t i = 0; i < hidden_size; ++i) {
      const float val = h_embd[token_id * hidden_size + i];
      sum_sq += val * val;
    }
    const float rms =
        std::sqrt(sum_sq / static_cast<float>(hidden_size) + 1e-6F);
    for (std::size_t i = 0; i < hidden_size; ++i) {
      const float expected = (h_embd[token_id * hidden_size + i] / rms) * 1.5F;
      gufo::test::ExpectNear(expected, h_out[i], 1e-4F,
                             "captured RMSNorm result mismatch");
    }
  }

  executor.Reset();
  gufo::test::Expect(!executor.IsCaptured(),
                     "HIP graph reset retained capture state");
  gufo::test::Expect(!executor.IsCapturedFor(graph_key),
                     "HIP graph reset retained capture identity");

  HIP_CHECK(hipFree(d_embd));
  HIP_CHECK(hipFree(d_hidden));
  HIP_CHECK(hipFree(d_normed));
  HIP_CHECK(hipFree(d_weight));
  HIP_CHECK(hipFree(d_params));
  HIP_CHECK(hipStreamDestroy(stream));
}

/// T-05/graph (tasks 3.9): the KVarN seal and decode launches are HIP-graph
/// eligible. Capture spans a seal boundary with the query position supplied
/// from device memory, no host synchronization is added for the seal, and the
/// replay reproduces the uncaptured output bit for bit. The rejection reasons
/// must not grow a KVarN entry.
void TestKvarnGraphCapture() {
  using gufo::hip::QwenGraphRejection;
  using gufo::hip::QwenKvarnPreset;
  static_assert(
      static_cast<std::uint32_t>(QwenGraphRejection::kGraphDisabled) == 8U,
      "HIP graph rejection reasons must not grow a KVarN entry");
  gufo::test::Expect(gufo::hip::ResolveQwenGraphRejections(
                         /*compute_logits=*/true,
                         /*split_k_attention=*/false,
                         /*graph_enabled=*/true) == QwenGraphRejection::kNone,
                     "KVarN decode must stay graph eligible");

  constexpr std::uint32_t kHeads = 4;
  constexpr std::uint32_t kKvHeads = 1;
  constexpr std::uint32_t kHeadDim = 256;
  constexpr std::uint32_t kLayers = 1;
  constexpr std::uint32_t kPosLow = 264;   // below the first seal boundary
  constexpr std::uint32_t kPosHigh = 265;  // crosses it (e_t + L + 1 = 265)
  const std::uint32_t max_context = kPosHigh + 1;
  const std::uint32_t sink_plane = kLayers * kKvHeads * 128 * kHeadDim;

  QwenKvarnPreset preset;
  preset.k_bits = 4;
  preset.v_bits = 4;
  const std::uint32_t tail_length = gufo::hip::QwenKvarnTailRows(preset);
  const std::uint32_t ring_rows = gufo::hip::QwenKvarnTailRingRows(preset);
  const std::uint32_t tile_count = (max_context - 128 + 127) / 128;
  const std::uint32_t k_bytes = static_cast<std::uint32_t>(
      gufo::models::qwen::KvarnTilePayloadBytes(preset.k_bits));
  const std::uint32_t v_bytes = static_cast<std::uint32_t>(
      gufo::models::qwen::KvarnTilePayloadBytes(preset.v_bits));
  const std::uint32_t tail_plane = kLayers * kKvHeads * ring_rows * kHeadDim;

  void* sink = nullptr;
  void* tail = nullptr;
  std::uint8_t* body = nullptr;
  HIP_CHECK(hipMalloc(&sink, 2 * sink_plane * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&tail, 2 * tail_plane * sizeof(std::uint16_t)));
  void* body_raw = nullptr;
  HIP_CHECK(hipMalloc(
      &body_raw, static_cast<std::size_t>(tile_count) * (k_bytes + v_bytes)));
  body = static_cast<std::uint8_t*>(body_raw);
  HIP_CHECK(hipMemset(sink, 0, 2 * sink_plane * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(tail, 0, 2 * tail_plane * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(
      body_raw, 0, static_cast<std::size_t>(tile_count) * (k_bytes + v_bytes)));

  for (std::uint32_t position = 0; position < max_context; ++position) {
    for (const bool is_value : {false, true}) {
      std::vector<std::uint16_t> row(kHeadDim);
      for (std::uint32_t c = 0; c < kHeadDim; ++c) {
        const float value =
            is_value ? 0.25F * std::cos(position * 0.05F + c * 0.017F)
                     : 0.3F * std::sin(position * 0.07F + c * 0.013F);
        row[c] = __half_as_ushort(__float2half_rn(value));
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

  gufo::hip::KvarnSealState seal(kLayers, kKvHeads);
  gufo::hip::KvarnReadParams read;
  read.sink = static_cast<const __half*>(sink);
  read.tail = static_cast<const __half*>(tail);
  read.body = body;
  read.sink_plane_elements = sink_plane;
  read.tail_plane_elements = tail_plane;
  read.ring_rows = ring_rows;
  read.tile_count = tile_count;
  read.k_tile_bytes = k_bytes;
  read.v_tile_bytes = v_bytes;
  read.kv_heads = kKvHeads;
  read.tail_length = tail_length;
  read.k_bits = preset.k_bits;
  read.v_bits = preset.v_bits;

  std::vector<float> query(static_cast<std::size_t>(kHeads) * kHeadDim);
  for (std::uint32_t head = 0; head < kHeads; ++head) {
    for (std::uint32_t c = 0; c < kHeadDim; ++c) {
      query[head * kHeadDim + c] = 0.2F * std::sin(head * 0.3F + c * 0.021F);
    }
  }
  float* d_q = nullptr;
  float* d_out = nullptr;
  std::uint32_t* d_pos = nullptr;
  HIP_CHECK(hipMalloc(&d_q, query.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, query.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_pos, sizeof(std::uint32_t)));
  HIP_CHECK(hipMemcpy(d_q, query.data(), query.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  const auto run = [&]() {
    gufo::hip::LaunchKvarnSeal(seal, preset, sink, tail, sink_plane, tail_plane,
                               ring_rows, body, tile_count, kLayers, kKvHeads,
                               /*start=*/0, d_pos, /*count=*/1, stream);
    gufo::hip::LaunchKvarnDecodeAttention(
        d_q, nullptr, d_out, kLayers, /*start_pos=*/0, d_pos, /*batch_size=*/1,
        kHeads, kKvHeads, read, nullptr, stream);
  };

  // Uncaptured reference at the high position (seals the boundary tile).
  HIP_CHECK(
      hipMemcpy(d_pos, &kPosHigh, sizeof(kPosHigh), hipMemcpyHostToDevice));
  run();
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<float> reference(query.size());
  HIP_CHECK(hipMemcpy(reference.data(), d_out, reference.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  // Capture at the low position, then replay at the high position. The capture
  // must succeed: no host synchronization was introduced for the seal.
  HIP_CHECK(hipMemcpy(d_pos, &kPosLow, sizeof(kPosLow), hipMemcpyHostToDevice));
  gufo::hip::detail::HipGraphDecodeExecutor executor;
  constexpr gufo::hip::detail::HipGraphCaptureKey graph_key{
      .execution_identity = 71,
      .workload_identity = 97,
  };
  const bool captured = executor.TryCapture(stream, graph_key, run);
  gufo::test::Expect(captured, "KVarN seal/decode capture failed");
  HIP_CHECK(
      hipMemcpy(d_pos, &kPosHigh, sizeof(kPosHigh), hipMemcpyHostToDevice));
  gufo::test::Expect(executor.Launch(stream, graph_key),
                     "KVarN graph replay failed");
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<float> replayed(query.size());
  HIP_CHECK(hipMemcpy(replayed.data(), d_out, replayed.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  gufo::test::Expect(std::memcmp(reference.data(), replayed.data(),
                                 reference.size() * sizeof(float)) == 0,
                     "KVarN graph replay changed the decode output");

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_pos));
  HIP_CHECK(hipFree(sink));
  HIP_CHECK(hipFree(tail));
  HIP_CHECK(hipFree(body_raw));
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int gate = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "Qwen graph ops test");
  if (gate != gufo::test::kHipTestSuccess) {
    return gate;
  }

  TestHipGraphDecodeStep();
  TestKvarnGraphCapture();
  std::cout << "Qwen graph ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen graph ops test.\n";
  return 77;
#endif
}
