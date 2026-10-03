// KVarN prefill read checks (openspec add-kvarn-kv-cache-quantization 4.2,
// T-03): the tile, WMMA and batched prefill reads merge the sink, the sealed
// tiles and the F16 rows held in the bounded per-chunk workspace (design D4),
// per query row, and agree with an FP64 reference and with each other within a
// declared limit. The chunk flush must land the chunk rows in the sink and ring
// byte-identically (the persisted rows T-04 compares).
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
#include "src/models/qwen/hip/kvarn.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/kvarn_reference.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/testing/test_common.hpp"

namespace {

namespace ref = gufo::models::qwen;

// Production full-attention shape of Qwen3.8 27B, the only shape the tile and
// WMMA prefill kernels accept.
constexpr std::uint32_t kQueryHeads = 24;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kGqa = kQueryHeads / kKvHeads;
// Declared in QUALITY.md: the tile and WMMA paths stage the decoded K/V as
// FP16 in shared memory and WMMA, while the reference is FP64 over the exact
// decoded values, so they agree within this absolute limit.
constexpr double kOutputLimit = 4e-3;
// Above this many rows the scalar batched fallback is quadratic and too slow
// to check; the production WMMA path and the tiled fallback still cover it.
constexpr std::uint32_t kMaxBatchedFallbackRows = 640;

float KeyValue(std::uint32_t position, std::uint32_t head,
               std::uint32_t channel, bool is_value) {
  const float p = static_cast<float>(position);
  const float h = static_cast<float>(head);
  const float c = static_cast<float>(channel);
  // Head-dependent so a KVarN payload read from the wrong KV head cannot pass.
  return is_value ? 0.25F * std::cos(p * 0.05F + h * 0.013F + c * 0.017F)
                  : 0.3F * std::sin(p * 0.07F + h * 0.011F + c * 0.013F);
}

float QueryValue(std::uint32_t head, std::uint32_t channel) {
  return 0.2F * std::sin(static_cast<float>(head) * 0.3F +
                         static_cast<float>(channel) * 0.021F);
}

float GateValue(std::uint32_t row, std::uint32_t head, std::uint32_t channel) {
  return 0.5F * std::sin(static_cast<float>(row) * 0.011F +
                         static_cast<float>(head) * 0.07F +
                         static_cast<float>(channel) * 0.003F);
}

/// One KVarN session held directly on the device: the sink, the sealed tiles,
/// the tail ring and the bounded per-chunk F16 workspace (design D1/D4).
struct Harness {
  gufo::hip::KvarnSealState seal{1, kKvHeads};
  gufo::hip::QwenKvarnPreset preset{};
  void* sink{nullptr};
  void* tail{nullptr};
  void* chunk{nullptr};
  std::uint8_t* body{nullptr};
  std::uint32_t ring_rows{0};
  std::uint32_t tail_length{0};
  std::uint32_t tile_count{0};
  std::uint32_t k_tile_bytes{0};
  std::uint32_t v_tile_bytes{0};
  std::size_t sink_plane{0};
  std::size_t tail_plane{0};
  std::size_t chunk_plane{0};
  std::uint8_t k_bits{4};
  std::uint8_t v_bits{4};

  Harness(std::uint8_t k_b, std::uint8_t v_b, std::uint32_t tail_tokens,
          std::uint32_t context)
      : k_bits(k_b), v_bits(v_b) {
    preset.k_bits = k_b;
    preset.v_bits = v_b;
    preset.tail_length = tail_tokens;
    tail_length = gufo::hip::QwenKvarnTailRows(preset);
    ring_rows = gufo::hip::QwenKvarnTailRingRows(preset);
    tile_count = (context - 128 + 127) / 128;
    k_tile_bytes = static_cast<std::uint32_t>(ref::KvarnTilePayloadBytes(k_b));
    v_tile_bytes = static_cast<std::uint32_t>(ref::KvarnTilePayloadBytes(v_b));
    sink_plane = static_cast<std::size_t>(kKvHeads) * 128 * kHeadDim;
    tail_plane = static_cast<std::size_t>(kKvHeads) * ring_rows * kHeadDim;
    chunk_plane = static_cast<std::size_t>(context) * kKvHeads * kHeadDim;
    const std::size_t body_bytes = std::max<std::size_t>(
        static_cast<std::size_t>(tile_count) * kKvHeads *
            (k_tile_bytes + v_tile_bytes),
        1);
    HIP_CHECK(hipMalloc(&sink, 2 * sink_plane * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&tail, 2 * tail_plane * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&chunk, 2 * chunk_plane * sizeof(std::uint16_t)));
    void* body_raw = nullptr;
    HIP_CHECK(hipMalloc(&body_raw, body_bytes));
    body = static_cast<std::uint8_t*>(body_raw);
    // Garbage everywhere: a read that picks the wrong region must fail the
    // reference comparison instead of silently succeeding.
    HIP_CHECK(hipMemset(sink, 0xA5, 2 * sink_plane * sizeof(std::uint16_t)));
    HIP_CHECK(hipMemset(tail, 0xA5, 2 * tail_plane * sizeof(std::uint16_t)));
    HIP_CHECK(hipMemset(chunk, 0xA5, 2 * chunk_plane * sizeof(std::uint16_t)));
    HIP_CHECK(hipMemset(body_raw, 0xA5, body_bytes));
  }

  ~Harness() {
    (void)hipFree(sink);
    (void)hipFree(tail);
    (void)hipFree(chunk);
    (void)hipFree(body);
  }

  /// Writes positions [begin, begin + count) into the per-chunk workspace as
  /// the fused write does (rotated rows already rounded to F16).
  void FillChunk(std::uint32_t begin, std::uint32_t count) {
    for (std::uint32_t row = 0; row < count; ++row) {
      const std::uint32_t position = begin + row;
      for (std::uint32_t head = 0; head < kKvHeads; ++head) {
        for (const bool is_value : {false, true}) {
          std::vector<std::uint16_t> values(kHeadDim);
          for (std::uint32_t c = 0; c < kHeadDim; ++c) {
            values[c] = __half_as_ushort(
                __float2half_rn(KeyValue(position, head, c, is_value)));
          }
          const std::size_t offset =
              (is_value ? chunk_plane : 0) +
              (static_cast<std::size_t>(row) * kKvHeads + head) * kHeadDim;
          HIP_CHECK(hipMemcpy(
              static_cast<std::uint8_t*>(chunk) +
                  offset * sizeof(std::uint16_t),
              values.data(), values.size() * sizeof(std::uint16_t),
              hipMemcpyHostToDevice));
        }
      }
    }
  }

  [[nodiscard]] gufo::hip::KvarnWriteParams WriteParams() const {
    gufo::hip::KvarnWriteParams params;
    params.sink = static_cast<__half*>(sink);
    params.tail = static_cast<__half*>(tail);
    params.sink_plane_elements = sink_plane;
    params.tail_plane_elements = tail_plane;
    params.ring_rows = ring_rows;
    params.chunk = static_cast<__half*>(chunk);
    params.chunk_plane_elements = chunk_plane;
    return params;
  }

  [[nodiscard]] gufo::hip::KvarnReadParams ReadParams() const {
    gufo::hip::KvarnReadParams params;
    params.sink = static_cast<const __half*>(sink);
    params.tail = static_cast<const __half*>(tail);
    params.body = body;
    params.sink_plane_elements = sink_plane;
    params.tail_plane_elements = tail_plane;
    params.ring_rows = ring_rows;
    params.tile_count = tile_count;
    params.k_tile_bytes = k_tile_bytes;
    params.v_tile_bytes = v_tile_bytes;
    params.kv_heads = kKvHeads;
    params.tail_length = tail_length;
    params.k_bits = k_bits;
    params.v_bits = v_bits;
    params.chunk = static_cast<const __half*>(chunk);
    params.chunk_plane_elements = chunk_plane;
    return params;
  }

  void Seal(std::uint32_t start, std::uint32_t count) {
    gufo::hip::LaunchKvarnSeal(
        seal, preset, sink, tail, sink_plane, tail_plane, ring_rows, body,
        std::max<std::uint32_t>(tile_count, 1), /*layer_idx=*/0, kKvHeads,
        start, nullptr, count, nullptr, chunk, chunk_plane);
  }

  void Flush(std::uint32_t begin, std::uint32_t count) {
    gufo::hip::LaunchKvarnChunkFlush(WriteParams(), /*layer_idx=*/0, kKvHeads,
                                     begin, count, nullptr);
  }

  [[nodiscard]] ref::KvarnTilePayload Payload(std::uint32_t tile,
                                              std::uint32_t head,
                                              bool is_value) const {
    ref::KvarnTilePayload payload{};
    const std::size_t offset =
        (static_cast<std::size_t>(head) * tile_count + tile) *
            (k_tile_bytes + v_tile_bytes) +
        (is_value ? k_tile_bytes : 0);
    HIP_CHECK(hipMemcpy(&payload, body + offset, sizeof(payload),
                        hipMemcpyDeviceToHost));
    return payload;
  }
};

/// FP64 reference of one query row: the D2 view rule evaluated for this row,
/// the same decoded values the decode read uses, exact softmax arithmetic, one
/// inverse Hadamard rotation and the SiLU gate (design D2/D3/D5). `gate_row`
/// and `q_head` are the absolute row's gate and this head's query.
std::vector<double> Reference(const Harness& harness, std::uint32_t pos,
                              std::uint32_t head, const float* gate_row,
                              const float* q_head) {
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
  const std::uint32_t kv_head = head / kGqa;
  const auto quantized_at = [&](std::uint32_t position) {
    if (position < 128) {
      return false;
    }
    const std::uint32_t tile = (position - 128) / 128;
    const std::uint32_t e_t = 128 + 128 * tile + 127;
    return static_cast<std::uint64_t>(e_t) + harness.tail_length + 1 <= pos;
  };
  std::vector<double> scores(pos + 1);
  double running_max = -std::numeric_limits<double>::infinity();
  for (std::uint32_t position = 0; position <= pos; ++position) {
    double score = 0.0;
    if (!quantized_at(position)) {
      for (std::uint32_t c = 0; c < kHeadDim; ++c) {
        score += static_cast<double>(q_head[c]) *
                 static_cast<double>(__half2float(
                     __float2half_rn(KeyValue(position, kv_head, c, false))));
      }
    } else {
      const std::uint32_t tile = (position - 128) / 128;
      const std::uint32_t token = position - 128 - 128 * tile;
      const auto payload = harness.Payload(tile, kv_head, false);
      const double token_scale =
          gufo::quant::Fp16ToFloat(payload.norm_token_scale[token]);
      for (std::uint32_t c = 0; c < kHeadDim; ++c) {
        const double decoded = static_cast<double>(ref::KvarnDecodeTileElement(
            payload, ref::KvarnRtnAxis::kChannel, harness.k_bits, token, c));
        const double channel_scale =
            gufo::quant::Fp16ToFloat(payload.norm_channel_scale[c]);
        score += static_cast<double>(q_head[c]) * decoded * channel_scale *
                 token_scale;
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
    if (!quantized_at(position)) {
      for (std::uint32_t c = 0; c < kHeadDim; ++c) {
        out[c] += weight * static_cast<double>(__half2float(
                              __float2half_rn(KeyValue(position, kv_head, c, true))));
      }
    } else {
      const std::uint32_t tile = (position - 128) / 128;
      const std::uint32_t token = position - 128 - 128 * tile;
      const auto payload = harness.Payload(tile, kv_head, true);
      const double token_scale =
          gufo::quant::Fp16ToFloat(payload.norm_token_scale[token]);
      for (std::uint32_t c = 0; c < kHeadDim; ++c) {
        const double decoded = static_cast<double>(ref::KvarnDecodeTileElement(
            payload, ref::KvarnRtnAxis::kTokenGroup, harness.v_bits, token, c));
        const double channel_scale =
            gufo::quant::Fp16ToFloat(payload.norm_channel_scale[c]);
        out[c] += weight * decoded * channel_scale * token_scale;
      }
    }
  }
  std::vector<float> rotated(kHeadDim);
  for (std::uint32_t c = 0; c < kHeadDim; ++c) {
    rotated[c] = static_cast<float>(out[c]);
  }
  ref::KvarnFwhtHeadInPlace(rotated);
  for (std::uint32_t c = 0; c < kHeadDim; ++c) {
    out[c] = static_cast<double>(rotated[c]) *
             (1.0 / (1.0 + std::exp(-static_cast<double>(gate_row[c]))));
  }
  return out;
}

struct PathOutputs {
  std::vector<float> wmma;
  std::vector<float> tile;
  std::vector<float> batched;
};

/// Runs the three prefill reads over the chunk; `q` and `gate` hold exactly the
/// chunk's `rows` entries (local row major).
PathOutputs RunPaths(Harness& harness, std::uint32_t start, std::uint32_t rows,
                     std::uint32_t max_context, const std::vector<float>& q,
                     const std::vector<float>& gate) {
    const std::size_t width =
      static_cast<std::size_t>(rows) * kQueryHeads * kHeadDim;
  float* d_q = nullptr;
  float* d_gate = nullptr;
  HIP_CHECK(hipMalloc(&d_q, width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, width * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_q, q.data(), width * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, gate.data(), width * sizeof(float),
                      hipMemcpyHostToDevice));

  const auto read_into = [&](auto launch) {
    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, width * sizeof(float)));
    HIP_CHECK(hipMemset(d_out, 0, width * sizeof(float)));
    launch(d_out);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> out(width);
    HIP_CHECK(hipMemcpy(out.data(), d_out, width * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_out));
    return out;
  };

  PathOutputs outputs;

  {
    // Sized for at most three heads per group, so the bounded-workspace
    // head grouping (design D7) is exercised on every case.
    const std::size_t words =
        (static_cast<std::size_t>(start + rows) * kHeadDim * 3) / 2 + 16;
    std::vector<float*> storage(2, nullptr);
    for (auto*& buffer : storage) {
      HIP_CHECK(hipMalloc(&buffer, words * sizeof(float)));
    }
    const std::span<float> key_workspace{storage[0], words};
    const std::span<float> value_workspace{storage[1], words};
    outputs.wmma = read_into([&](float* out) {
      if (!gufo::hip::LaunchQwenWmmaAttention(
              d_q, nullptr, nullptr, d_gate, nullptr, nullptr, nullptr, nullptr,
              out, /*layer_idx=*/0, start, rows, max_context, kQueryHeads,
              kKvHeads, kHeadDim, nullptr, nullptr, 0, /*skip_kv_write=*/true,
              key_workspace, value_workspace, harness.ReadParams())) {
        throw std::runtime_error("WMMA prefill rejected the production shape");
      }
    });
    for (auto* buffer : storage) {
      HIP_CHECK(hipFree(buffer));
    }
  }
  outputs.tile = read_into([&](float* out) {
    if (!gufo::hip::LaunchBatchedAttentionTile(
            d_q, nullptr, nullptr, d_gate, nullptr, nullptr, nullptr, nullptr,
            out, /*layer_idx=*/0, start, rows, max_context, kQueryHeads,
            kKvHeads, kHeadDim, nullptr, nullptr, 0, /*skip_kv_write=*/true,
            harness.ReadParams())) {
      throw std::runtime_error("tiled prefill rejected the production shape");
    }
  });
  // The scalar batched kernel is a fallback for shapes the WMMA tile rejects;
  // it is O(rows * context * heads), so only the cases where it is realistic
  // are exercised here (production always takes WMMA for this shape).
  if (rows <= kMaxBatchedFallbackRows) {
    outputs.batched = read_into([&](float* out) {
      gufo::hip::LaunchBatchedAttention(
          d_q, nullptr, nullptr, d_gate, nullptr, nullptr, nullptr, nullptr, out,
          /*layer_idx=*/0, start, rows, max_context, kQueryHeads, kKvHeads,
          kHeadDim, nullptr, /*skip_kv_write=*/true, nullptr,
          harness.ReadParams());
    });
  }

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_gate));
  return outputs;
}

/// Rows to check with the FP64 reference: the chunk's first and last query plus
/// the rows next to each view threshold 256 + 128 t + L.
std::vector<std::uint32_t> SelectedRows(std::uint32_t start,
                                        std::uint32_t rows,
                                        std::uint32_t tail_length) {
  std::vector<std::uint32_t> selected;
  if (rows == 0) {
    return selected;
  }
  selected.push_back(start);
  selected.push_back(start + rows - 1);
  for (std::uint64_t threshold =
           static_cast<std::uint64_t>(256) + tail_length;
       threshold <= static_cast<std::uint64_t>(start) + rows;
       threshold += 128) {
    if (threshold > start) {
      selected.push_back(static_cast<std::uint32_t>(threshold - 1));
    }
    if (threshold >= start && threshold < start + rows) {
      selected.push_back(static_cast<std::uint32_t>(threshold));
    }
  }
  std::sort(selected.begin(), selected.end());
  selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
  return selected;
}

/// Compares each path's selected rows against the FP64 reference, the paths
/// against each other, then flushes and checks the persisted F16 rows.
void CheckChunk(Harness& harness, std::uint32_t start, std::uint32_t rows,
                const std::vector<float>& q, const std::vector<float>& gate,
                const std::vector<float>& gate_all,
                const PathOutputs& outputs) {
  const auto selected = SelectedRows(start, rows, harness.tail_length);
  double worst = 0.0;
  double worst_wmma = 0.0, worst_tile = 0.0, worst_batched = 0.0;
  for (const std::uint32_t row : selected) {
    const std::uint32_t local = row - start;
    const float* gate_row = gate_all.data() +
                            static_cast<std::size_t>(row) * kQueryHeads *
                                kHeadDim;
    for (std::uint32_t head = 0; head < kQueryHeads; ++head) {
      const float* q_head =
          q.data() +
          (static_cast<std::size_t>(local) * kQueryHeads + head) * kHeadDim;
      const auto expected =
          Reference(harness, row, head, gate_row + head * kHeadDim, q_head);
      for (std::uint32_t c = 0; c < kHeadDim; ++c) {
        const std::size_t index =
            (static_cast<std::size_t>(local) * kQueryHeads + head) * kHeadDim +
            c;
        const double dw = std::abs(static_cast<double>(outputs.wmma[index]) -
                                   expected[c]);
        const double dt = std::abs(static_cast<double>(outputs.tile[index]) -
                                   expected[c]);
        const double db =
            outputs.batched.empty()
                ? 0.0
                : std::abs(static_cast<double>(outputs.batched[index]) -
                           expected[c]);
        worst_wmma = std::max(worst_wmma, dw);
        worst_tile = std::max(worst_tile, dt);
        worst_batched = std::max(worst_batched, db);
        worst = std::max({worst, dw, dt, db});
      }
    }
  }
  const auto cross = [&](const std::vector<float>& a,
                         const std::vector<float>& b, std::size_t* where) {
    double local_worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double diff = std::abs(static_cast<double>(a[i]) -
                                   static_cast<double>(b[i]));
      if (diff > local_worst) {
        local_worst = diff;
        *where = i;
      }
    }
    return local_worst;
  };
  std::size_t where_wt = 0, where_wb = 0, where_tb = 0;
  const double cwt = cross(outputs.wmma, outputs.tile, &where_wt);
  const double cwb = outputs.batched.empty()
                         ? 0.0
                         : cross(outputs.wmma, outputs.batched, &where_wb);
  const double ctb = outputs.batched.empty()
                         ? 0.0
                         : cross(outputs.tile, outputs.batched, &where_tb);
  const double cross_worst = std::max({cwt, cwb, ctb});
  const auto describe = [&](std::size_t index) {
    const std::size_t per_row = kQueryHeads * kHeadDim;
    return "row " + std::to_string(index / per_row) + " head " +
           std::to_string((index % per_row) / kHeadDim) + " channel " +
           std::to_string(index % kHeadDim);
  };
  if (worst > kOutputLimit || cross_worst > kOutputLimit) {
    throw std::runtime_error(
        "T-03 prefill: start " + std::to_string(start) + " rows " +
        std::to_string(rows) + " tail " + std::to_string(harness.tail_length) +
        " reference worst " + std::to_string(worst) + " (wmma " +
        std::to_string(worst_wmma) + " tile " + std::to_string(worst_tile) +
        " batched " + std::to_string(worst_batched) + ") cross " +
        std::to_string(cross_worst) + " (wmma-tile " + std::to_string(cwt) +
        " @ " + describe(where_wt) + ", wmma-batched " + std::to_string(cwb) +
        " @ " + describe(where_wb) + ", tile-batched " + std::to_string(ctb) +
        " @ " + describe(where_tb) + ")");
  }
  std::cout << "T-03 prefill start " << start << " rows " << rows << " tail "
            << harness.tail_length << " reference worst " << worst
            << " cross " << cross_worst << '\n';
}

void CheckFlush(Harness& harness, std::uint32_t begin, std::uint32_t rows) {
  harness.Flush(begin, rows);
  HIP_CHECK(hipDeviceSynchronize());
  const std::uint32_t ring_first =
      rows > harness.ring_rows ? begin + rows - harness.ring_rows : begin;
  std::uint32_t checked = 0;
  for (std::uint32_t row = 0; row < rows; ++row) {
    const std::uint32_t position = begin + row;
    const bool to_sink = position < 128;
    if (!to_sink && position < ring_first) {
      continue;
    }
    ++checked;
    for (std::uint32_t head = 0; head < kKvHeads; ++head) {
      std::vector<std::uint16_t> persisted(kHeadDim);
      const std::size_t offset =
          to_sink ? ref::KvarnSinkRowOffset(0, kKvHeads, head, position)
                  : ref::KvarnTailRowOffset(0, kKvHeads, head, position,
                                            harness.ring_rows);
      const void* source =
          to_sink ? static_cast<const void*>(static_cast<const std::uint8_t*>(
                                                harness.sink) +
                                            offset * sizeof(std::uint16_t))
                  : static_cast<const void*>(static_cast<const std::uint8_t*>(
                                                harness.tail) +
                                            offset * sizeof(std::uint16_t));
      HIP_CHECK(hipMemcpy(persisted.data(), source,
                          persisted.size() * sizeof(std::uint16_t),
                          hipMemcpyDeviceToHost));
      std::vector<std::uint16_t> expected(kHeadDim);
      HIP_CHECK(hipMemcpy(
          expected.data(),
          static_cast<const std::uint8_t*>(harness.chunk) +
              ((static_cast<std::size_t>(row) * kKvHeads + head) * kHeadDim) *
                  sizeof(std::uint16_t),
          expected.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
      if (persisted != expected) {
        throw std::runtime_error(
            "T-04 flush: begin " + std::to_string(begin) + " rows " +
            std::to_string(rows) + " position " + std::to_string(position) +
            " head " + std::to_string(head) + " differs from the chunk row");
      }
    }
  }
  std::cout << "T-04 flush begin " << begin << " rows " << rows << " persisted "
            << checked << " bit-identical\n";
}

std::vector<float> MakeQuery(std::uint32_t rows) {
  std::vector<float> q(static_cast<std::size_t>(rows) * kQueryHeads * kHeadDim);
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t head = 0; head < kQueryHeads; ++head) {
      for (std::uint32_t c = 0; c < kHeadDim; ++c) {
        q[(static_cast<std::size_t>(row) * kQueryHeads + head) * kHeadDim + c] =
            QueryValue(head, c);
      }
    }
  }
  return q;
}

std::vector<float> MakeGate(std::uint32_t rows) {
  std::vector<float> gate(static_cast<std::size_t>(rows) * kQueryHeads *
                          kHeadDim);
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t head = 0; head < kQueryHeads; ++head) {
      for (std::uint32_t c = 0; c < kHeadDim; ++c) {
        gate[(static_cast<std::size_t>(row) * kQueryHeads + head) * kHeadDim +
             c] = GateValue(row, head, c);
      }
    }
  }
  return gate;
}

/// T-03: one prefill chunk at every view boundary, for both tail settings.
void RunSingleChunk(std::uint8_t k_bits, std::uint8_t v_bits,
                    std::uint32_t tail, std::uint32_t context) {
  std::cerr << "[case] single context=" << context << " tail=" << tail
            << std::endl;
  Harness harness(k_bits, v_bits, tail, context);
    harness.FillChunk(0, context);
    harness.Seal(0, context);
      const auto q = MakeQuery(context);
  const auto gate = MakeGate(context);
  const auto outputs = RunPaths(harness, 0, context, context, q, gate);
    CheckChunk(harness, 0, context, q, gate, gate, outputs);
  CheckFlush(harness, 0, context);
}

/// T-04: a second chunk reads its pre-chunk F16 rows from the ring the first
/// chunk flushed, and the whole run matches the FP64 reference.
void RunMultiChunk(std::uint8_t k_bits, std::uint8_t v_bits, std::uint32_t tail,
                   std::uint32_t first_rows, std::uint32_t second_rows) {
  const std::uint32_t context = first_rows + second_rows;
  std::cerr << "[case] multi first=" << first_rows << " second=" << second_rows
            << " tail=" << tail << std::endl;
  Harness harness(k_bits, v_bits, tail, context);
  const auto gate_all = MakeGate(context);
  const auto q_all = MakeQuery(context);
  {
    const std::vector<float> gate_chunk(
        gate_all.begin(),
        gate_all.begin() +
            static_cast<std::size_t>(first_rows) * kQueryHeads * kHeadDim);
    const std::vector<float> q_chunk(
        q_all.begin(), q_all.begin() + static_cast<std::size_t>(first_rows) *
                                            kQueryHeads * kHeadDim);
    harness.FillChunk(0, first_rows);
    harness.Seal(0, first_rows);
    const auto outputs =
        RunPaths(harness, 0, first_rows, context, q_chunk, gate_chunk);
    CheckChunk(harness, 0, first_rows, q_chunk, gate_chunk, gate_all, outputs);
    harness.Flush(0, first_rows);
    HIP_CHECK(hipDeviceSynchronize());
  }
  const std::size_t offset =
      static_cast<std::size_t>(first_rows) * kQueryHeads * kHeadDim;
  const std::vector<float> q_chunk(
      q_all.begin() + offset, q_all.begin() + offset +
                                  static_cast<std::size_t>(second_rows) *
                                      kQueryHeads * kHeadDim);
  const std::vector<float> gate_chunk(
      gate_all.begin() + offset,
      gate_all.begin() + offset +
          static_cast<std::size_t>(second_rows) * kQueryHeads * kHeadDim);
  harness.FillChunk(first_rows, second_rows);
  harness.Seal(first_rows, second_rows);
  const auto outputs = RunPaths(harness, first_rows, second_rows, context,
                                q_chunk, gate_chunk);
  CheckChunk(harness, first_rows, second_rows, q_chunk, gate_chunk, gate_all,
             outputs);
  CheckFlush(harness, first_rows, second_rows);
}

}  // namespace

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional,
      "Qwen KVarN prefill read ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }
  for (const std::uint32_t tail : {0U, 2048U}) {
    const std::uint32_t L = tail == 0 ? 9 : tail;
    for (const std::uint32_t context :
         {127U, 128U, 129U, 128 + L + 127, 128 + L + 128, 128 + L + 129,
          256 + L + 127, 256 + L + 128, 256 + L + 129}) {
      RunSingleChunk(4, 4, tail, context);
    }
    RunMultiChunk(4, 4, tail, 256, 128);
  }
  std::cout << "Qwen KVarN prefill read ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout << "HIP disabled, skipping Qwen KVarN prefill read ops test.\n";
  return 77;
#endif
}
#endif  // defined(ENGINE_ENABLE_HIP)
