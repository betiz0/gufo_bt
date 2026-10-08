// KVarN decode-read compute-only microbenchmark (openspec change
// add-kvarn-kv-cache-quantization, design D18 follow-up).
//
// The production depth cost is dominated by KV memory traffic (the live KV does
// not fit in cache), so it cannot separate compute from bandwidth. Here one
// full-attention layer's KV is small enough to stay resident in the SoC cache
// (2 MiB FP16 for the unset tail, ~10 MiB for tail 2048), its tiles are sealed
// with the CPU reference encoder, and the read kernel is timed over it. The
// per-(query head, key) unit cost is scaled to the 128K-token, 16-layer decode
// to give the compute-only floor of one generated token, next to the bandwidth
// floor B(depth)/BW. That decides whether 1.00x FP16 at 128K is reachable.
//
// Build (system hipcc):
//   hipcc -O3 --offload-arch=gfx1151 -std=c++20 -I. -c \
//     tools/qwen27b/kvarn_read_compute_bench.hip -o /tmp/kvrcb_main.o
//   hipcc -O3 -std=c++20 -I. -c src/models/qwen/kvarn_reference.cpp \
//     -o /tmp/kvrcb_ref.o
//   hipcc --offload-arch=gfx1151 /tmp/kvrcb_main.o /tmp/kvrcb_ref.o \
//     -o /tmp/kvarn_read_compute_bench
#define ENGINE_ENABLE_HIP 1
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/hip/kernels/attention_decode.hip"
#include "src/models/qwen/hip/kernels/kvarn_read.hip"
#include "src/models/qwen/kvarn_reference.hpp"

namespace {

namespace ref = gufo::models::qwen;

constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kLayers = 1;
constexpr std::uint32_t kKeys128K = 131072;
constexpr std::uint32_t kLayers128K = 16;
constexpr std::uint32_t kIterations = 400;
constexpr std::uint32_t kWarmup = 20;
// Measured FP16 KV read bandwidth of the same measurement series (design D18).
constexpr double kBandwidthGBs = 241.0;

float KeyValue(std::uint32_t position, std::uint32_t channel, bool is_value) {
  const float p = static_cast<float>(position);
  const float c = static_cast<float>(channel);
  return is_value ? 0.25F * std::cos(p * 0.05F + c * 0.017F)
                  : 0.3F * std::sin(p * 0.07F + c * 0.013F);
}

std::size_t TilePayloadBytes(std::uint8_t bits) {
  return ref::KvarnTilePayloadBytes(bits);
}

/// One cache-resident full-attention layer of KVarN storage, sealed with the
/// CPU reference encoder so the read kernel exercises the quantized path.
struct KvarnFixture {
  std::uint32_t context{0};
  gufo::hip::QwenKvarnPreset preset{};
  std::uint32_t tail_rows{0};
  std::uint32_t ring_rows{0};
  std::uint32_t tile_count{0};
  std::size_t sink_plane{0};
  std::size_t tail_plane{0};
  std::size_t k_tile_bytes{0};
  std::size_t v_tile_bytes{0};
  void* sink{nullptr};
  void* tail{nullptr};
  std::uint8_t* body{nullptr};
  gufo::hip::KvarnReadParams read{};

  KvarnFixture(std::uint32_t context_tokens, std::uint8_t k_bits,
               std::uint8_t v_bits, std::uint32_t tail_tokens)
      : context(context_tokens) {
    preset.k_bits = k_bits;
    preset.v_bits = v_bits;
    preset.tail_length = tail_tokens;
    tail_rows = gufo::hip::QwenKvarnTailRows(preset);
    ring_rows = gufo::hip::QwenKvarnTailRingRows(preset);
    tile_count = (context - ref::kKvarnSinkRows + ref::kKvarnTileTokens - 1) /
                 ref::kKvarnTileTokens;
    k_tile_bytes = TilePayloadBytes(k_bits);
    v_tile_bytes = TilePayloadBytes(v_bits);
    sink_plane = static_cast<std::size_t>(kLayers) * kKvHeads *
                 ref::kKvarnSinkRows * kHeadDim;
    tail_plane =
        static_cast<std::size_t>(kLayers) * kKvHeads * ring_rows * kHeadDim;
    HIP_CHECK(hipMalloc(&sink, 2 * sink_plane * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&tail, 2 * tail_plane * sizeof(std::uint16_t)));
    void* body_raw = nullptr;
    HIP_CHECK(hipMalloc(&body_raw, static_cast<std::size_t>(kLayers) *
                                       kKvHeads * tile_count *
                                       (k_tile_bytes + v_tile_bytes)));
    body = static_cast<std::uint8_t*>(body_raw);

    // Tiles the view rule reads quantized for the query at `context - 1`.
    const std::uint32_t query = context - 1;
    const std::uint32_t quantized_tiles =
        query >= ref::kKvarnSinkRows + tail_rows + 1
            ? (query - ref::kKvarnSinkRows - tail_rows) / ref::kKvarnTileTokens
            : 0;
    const std::uint32_t first_f16 =
        ref::kKvarnSinkRows + ref::kKvarnTileTokens * quantized_tiles;

    // F16 sink and the unquantized tail rows; everything the query reads as
    // F16.
    std::vector<std::uint16_t> sink_host(2 * sink_plane, 0);
    std::vector<std::uint16_t> tail_host(2 * tail_plane, 0);
    const auto write_f16 = [&](std::uint32_t position) {
      for (std::uint32_t head = 0; head < kKvHeads; ++head) {
        for (const bool is_value : {false, true}) {
          for (std::uint32_t c = 0; c < kHeadDim; ++c) {
            const std::uint16_t encoded = __half_as_ushort(
                __float2half_rn(KeyValue(position, c, is_value)));
            std::size_t offset = 0;
            std::size_t plane = 0;
            if (position < ref::kKvarnSinkRows) {
              offset = ref::KvarnSinkRowOffset(0, kKvHeads, head, position);
              plane = is_value ? sink_plane : 0;
              sink_host[plane + offset + c] = encoded;
            } else {
              offset = ref::KvarnTailRowOffset(0, kKvHeads, head, position,
                                               ring_rows);
              plane = is_value ? tail_plane : 0;
              tail_host[plane + offset + c] = encoded;
            }
          }
        }
      }
    };
    for (std::uint32_t position = 0; position < ref::kKvarnSinkRows;
         ++position) {
      write_f16(position);
    }
    for (std::uint32_t position = first_f16; position < context; ++position) {
      write_f16(position);
    }
    HIP_CHECK(hipMemcpy(sink, sink_host.data(),
                        2 * sink_plane * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(tail, tail_host.data(),
                        2 * tail_plane * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));

    // Seal the quantized tiles with the CPU reference encoder.
    std::vector<float> tile(ref::kKvarnTileElements);
    std::vector<float> normalized(ref::kKvarnTileElements);
    for (std::uint32_t head = 0; head < kKvHeads; ++head) {
      for (const bool is_value : {false, true}) {
        for (std::uint32_t t = 0; t < quantized_tiles; ++t) {
          for (std::uint32_t token = 0; token < ref::kKvarnTileTokens;
               ++token) {
            const std::uint32_t position =
                ref::kKvarnSinkRows + ref::kKvarnTileTokens * t + token;
            float* row = tile.data() + token * ref::kKvarnTileChannels;
            for (std::uint32_t c = 0; c < ref::kKvarnTileChannels; ++c) {
              row[c] = KeyValue(position, c, is_value);
            }
            ref::KvarnFwhtHeadInPlace(
                std::span<float>(row, ref::kKvarnTileChannels));
          }
          ref::KvarnTileQuantResult result{};
          ref::KvarnQuantDiagnostics diag{};
          const auto status =
              ref::KvarnNormalizeTile(tile, normalized, &result, &diag);
          if (status != ref::KvarnQuantStatus::kOk) {
            throw std::runtime_error("reference tile normalization failed");
          }
          ref::KvarnQuantizeAsymmetricRtn(
              normalized,
              is_value ? ref::KvarnRtnAxis::kTokenGroup
                       : ref::KvarnRtnAxis::kChannel,
              is_value ? v_bits : k_bits, &result, &diag);
          ref::KvarnTilePayload payload{};
          ref::KvarnPackTilePayload(result, &payload);
          const std::size_t offset =
              ref::KvarnBodyTileOffset(0, kKvHeads, head, t, tile_count,
                                       k_tile_bytes, v_tile_bytes) +
              (is_value ? k_tile_bytes : 0);
          HIP_CHECK(hipMemcpy(body + offset, &payload, sizeof(payload),
                              hipMemcpyHostToDevice));
        }
      }
    }

    read.sink = static_cast<const __half*>(sink);
    read.tail = static_cast<const __half*>(tail);
    read.body = body;
    read.sink_plane_elements = sink_plane;
    read.tail_plane_elements = tail_plane;
    read.ring_rows = ring_rows;
    read.tile_count = tile_count;
    read.k_tile_bytes = static_cast<std::uint32_t>(k_tile_bytes);
    read.v_tile_bytes = static_cast<std::uint32_t>(v_tile_bytes);
    read.kv_heads = kKvHeads;
    read.tail_length = tail_rows;
    read.k_bits = k_bits;
    read.v_bits = v_bits;
  }

  ~KvarnFixture() {
    (void)hipFree(sink);
    (void)hipFree(tail);
    (void)hipFree(body);
  }
};

double TimeMs(const std::function<void()>& launch) {
  for (std::uint32_t i = 0; i < kWarmup; ++i) {
    launch();
  }
  hipEvent_t begin{};
  hipEvent_t end{};
  HIP_CHECK(hipEventCreate(&begin));
  HIP_CHECK(hipEventCreate(&end));
  HIP_CHECK(hipEventRecord(begin));
  for (std::uint32_t i = 0; i < kIterations; ++i) {
    launch();
  }
  HIP_CHECK(hipEventRecord(end));
  HIP_CHECK(hipEventSynchronize(end));
  float ms = 0.0F;
  HIP_CHECK(hipEventElapsedTime(&ms, begin, end));
  HIP_CHECK(hipEventDestroy(begin));
  HIP_CHECK(hipEventDestroy(end));
  return static_cast<double>(ms) / kIterations;
}

/// Bytes of the whole 27B KVarN state at `capacity` tokens (design D5).
double KvarnBytes(std::uint8_t k_bits, std::uint8_t v_bits,
                  std::uint32_t tail_tokens, std::uint32_t capacity) {
  const std::uint32_t tail_rows =
      tail_tokens == 0 ? gufo::hip::kQwenKvarnW : tail_tokens;
  const std::uint32_t ring_rows =
      tail_rows + ref::kKvarnSinkRows + gufo::hip::kQwenKvarnW;
  const std::size_t tiles =
      (capacity - ref::kKvarnSinkRows + ref::kKvarnTileTokens - 1) /
      ref::kKvarnTileTokens;
  const std::size_t heads = static_cast<std::size_t>(kLayers128K) * kKvHeads;
  const std::size_t sink = heads * ref::kKvarnSinkRows * kHeadDim * 2 * 2;
  const std::size_t tail = heads * ring_rows * kHeadDim * 2 * 2;
  const std::size_t body =
      heads * tiles * (TilePayloadBytes(k_bits) + TilePayloadBytes(v_bits));
  return static_cast<double>(sink + tail + body);
}

int Run() {
  const std::uint32_t attention_width = kHeads * kHeadDim;
  std::vector<float> q(attention_width), gate(attention_width),
      out(attention_width);
  for (std::uint32_t i = 0; i < attention_width; ++i) {
    q[i] = 0.2F * std::sin(static_cast<float>(i) * 0.021F);
    gate[i] = 0.5F + 0.1F * std::cos(static_cast<float>(i) * 0.017F);
  }
  float* d_q = nullptr;
  float* d_gate = nullptr;
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_q, q.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, gate.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, out.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_q, q.data(), q.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, gate.data(), gate.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  const std::size_t row_scratch =
      gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kHeadDim);
  float* d_split = nullptr;
  HIP_CHECK(hipMalloc(&d_split, row_scratch * sizeof(float)));

  // FP16 reference for each context, laid out like the production layer.
  const auto time_fp16 = [&](std::uint32_t context) {
    const std::size_t total_plane =
        static_cast<std::size_t>(kLayers) * kKvHeads * context * kHeadDim;
    std::vector<std::uint16_t> host(2 * total_plane);
    for (std::uint32_t position = 0; position < context; ++position) {
      for (std::uint32_t head = 0; head < kKvHeads; ++head) {
        for (const bool is_value : {false, true}) {
          for (std::uint32_t c = 0; c < kHeadDim; ++c) {
            const std::size_t off =
                (is_value ? total_plane : 0) +
                (static_cast<std::size_t>(head) * context + position) *
                    kHeadDim +
                c;
            host[off] = __half_as_ushort(
                __float2half_rn(KeyValue(position, c, is_value)));
          }
        }
      }
    }
    __half* d_cache = nullptr;
    HIP_CHECK(hipMalloc(&d_cache, host.size() * sizeof(std::uint16_t)));
    HIP_CHECK(hipMemcpy(d_cache, host.data(),
                        host.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));
    const std::uint32_t position = context - 1;
    const double ms = TimeMs([&] {
      gufo::hip::LaunchAttention(d_q, nullptr, nullptr, d_gate, nullptr,
                                 nullptr, d_cache, d_cache + total_plane, d_out,
                                 0, position, context, kHeads, kKvHeads,
                                 kHeadDim, nullptr, d_split,
                                 /*skip_kv_write=*/true);
    });
    HIP_CHECK(hipFree(d_cache));
    return ms;
  };

  // A single small context is launch-overhead dominated, so sweep several
  // cache-resident contexts and least-squares fit time against the
  // (query head x key) work to get the per-unit compute cost.
  const auto fit = [](const std::vector<double>& units,
                      const std::vector<double>& ms) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    const double n = static_cast<double>(units.size());
    for (std::size_t i = 0; i < units.size(); ++i) {
      sx += units[i];
      sy += ms[i];
      sxx += units[i] * units[i];
      sxy += units[i] * ms[i];
    }
    const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const double intercept = (sy - slope * sx) / n;
    return std::pair<double, double>{slope, intercept};
  };

  const auto report = [&](const char* label, std::uint32_t base_context,
                          std::uint8_t k_bits, std::uint8_t v_bits,
                          std::uint32_t tail_tokens) {
    std::vector<double> units, fp16_ms, kvarn_ms;
    for (std::uint32_t step = 1; step <= 4; ++step) {
      const std::uint32_t context = base_context * step;
      const double work = static_cast<double>(kHeads) * context;
      fp16_ms.push_back(time_fp16(context));
      KvarnFixture fixture(context, k_bits, v_bits, tail_tokens);
      kvarn_ms.push_back(TimeMs([&] {
        gufo::hip::LaunchKvarnDecodeAttention(
            d_q, d_gate, d_out, 0, context - 1, nullptr, 1, kHeads, kKvHeads,
            fixture.read, d_split, nullptr);
      }));
      units.push_back(work);
    }
    const auto [kvarn_slope_ms, kvarn_b] = fit(units, kvarn_ms);
    const auto [fp16_slope_ms, fp16_b] = fit(units, fp16_ms);
    const double total_units =
        static_cast<double>(kHeads) * kKeys128K * kLayers128K;
    const double kvarn_ns = kvarn_slope_ms * 1e6;
    const double fp16_ns = fp16_slope_ms * 1e6;
    const double compute_128k = kvarn_ns * total_units / 1e6;
    const double fp16_compute_128k = fp16_ns * total_units / 1e6;
    const double cap = kKeys128K + 64;
    const double kvarn_bytes = KvarnBytes(k_bits, v_bits, tail_tokens, cap);
    printf(
        "%s contexts=[%u..%u] kvarn=%.4f ns/unit (b=%.1f us) "
        "fp16=%.4f ns/unit (b=%.1f us)\n"
        "  compute_128k_ms=%.2f  fp16_compute_128k_ms=%.2f  "
        "bw_floor_ms=%.2f\n",
        label, base_context, base_context * 4, kvarn_ns, kvarn_b * 1e3, fp16_ns,
        fp16_b * 1e3, compute_128k, fp16_compute_128k,
        kvarn_bytes / kBandwidthGBs / 1e6);
    for (std::size_t i = 0; i < units.size(); ++i) {
      printf("    context=%u kvarn_ms=%.6f fp16_ms=%.6f\n",
             base_context * static_cast<std::uint32_t>(i + 1), kvarn_ms[i],
             fp16_ms[i]);
    }
  };

  // Base contexts: 128 sink + 128 token + 2048 fixed rows, scaled by the
  // sweep, so both presets span the same (4x) context range while the KVarN
  // body stays cache-resident.
  report("kvarn-k4v4-none", ref::kKvarnSinkRows + ref::kKvarnTileTokens + 2048,
         4, 4, 0);
  report("kvarn-k6v5-tail2048",
         ref::kKvarnSinkRows + ref::kKvarnTileTokens + 2048, 6, 5, 2048);

  HIP_CHECK(hipFree(d_q));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_split));
  return 0;
}

}  // namespace

int main() {
  return Run();
}
