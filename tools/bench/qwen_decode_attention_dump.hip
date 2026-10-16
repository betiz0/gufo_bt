// Dumps the outputs of every decode attention path this change must not
// alter, so the base and the candidate source trees can be compared byte for
// byte (openspec change update-qwen-decode-attention-blocks, task 1.5, design
// D12/G0U).
//
// Covered paths, all through the public launch functions only:
//   * the below-threshold online path at sequence length 127,
//   * the FP32 reference-KV path with the MTP argument shape (both the FP32
//     and the FP16 planes) at sequence length 128, 4096 and 32768,
//   * the FP32 reference-KV verification rows,
//   * the KVarN decode and verification-row reads over a fixed fixture sealed
//     with the CPU reference encoder (the same sealing framework the KVarN
//     compute bench uses).
//
// Build and run (from the repository root):
//   nix develop -c tools/bench/build.sh tools/bench/qwen_decode_attention_dump.hip
//   /tmp/qwen_decode_attention_dump --output /tmp/attention_dump.bin
//
// The output file is a plain concatenation of the dumped float buffers; the
// manifest printed to stdout lists every record's name, element count and byte
// offset. Running the same source twice must produce identical files (cmp).
#define ENGINE_ENABLE_HIP 1
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
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
constexpr std::uint32_t kAttentionWidth = kHeads * kHeadDim;
constexpr std::uint32_t kKvWidth = kKvHeads * kHeadDim;

std::FILE* g_output = nullptr;
std::size_t g_offset = 0;

void DumpRecord(const char* name, const void* data, std::size_t bytes) {
  printf("%-28s offset %10zu elements %8zu bytes %8zu\n", name, g_offset,
         bytes / sizeof(float), bytes);
  if (std::fwrite(data, 1, bytes, g_output) != bytes) {
    std::perror("fwrite");
    std::exit(1);
  }
  g_offset += bytes;
}

std::vector<float> ReadDevice(const float* source, std::size_t elements) {
  std::vector<float> host(elements);
  HIP_CHECK(hipMemcpy(host.data(), source, elements * sizeof(float),
                      hipMemcpyDeviceToHost));
  return host;
}

void DumpDevice(const char* name, const float* source, std::size_t elements) {
  const auto host = ReadDevice(source, elements);
  DumpRecord(name, host.data(), host.size() * sizeof(float));
}

/// Deterministic query/gate/KV inputs in the production layer layout.
struct LayerInputs {
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> gate;
  std::vector<float> cache32;
  std::vector<__half> cache16;
  std::uint32_t context{0};

  LayerInputs(std::uint32_t context_tokens, std::uint32_t rows)
      : context(context_tokens) {
    query.resize(static_cast<std::size_t>(rows) * kAttentionWidth);
    gate.resize(query.size());
    key.resize(kKvWidth);
    value.resize(kKvWidth);
    const std::size_t cache_elements =
        static_cast<std::size_t>(context) * kKvWidth;
    cache32.resize(cache_elements);
    cache16.resize(cache_elements);
    for (std::size_t i = 0; i < query.size(); ++i) {
      query[i] = 0.3F * std::sin(static_cast<float>(i) * 0.073F);
      gate[i] = std::cos(static_cast<float>(i) * 0.019F);
    }
    for (std::size_t i = 0; i < key.size(); ++i) {
      key[i] = 0.2F * std::sin(static_cast<float>(i) * 0.041F);
      value[i] = 0.25F * std::cos(static_cast<float>(i) * 0.037F);
    }
    for (std::size_t i = 0; i < cache_elements; ++i) {
      cache32[i] = 0.4F * std::sin(static_cast<float>(i) * 0.011F);
      cache16[i] = __float2half_rn(cache32[i]);
    }
  }
};

float KeyValue(std::uint32_t position, std::uint32_t channel, bool is_value,
               std::uint32_t kv_head) {
  const float p = static_cast<float>(position);
  const float c = static_cast<float>(channel);
  const float h = static_cast<float>(kv_head);
  return is_value ? 0.25F * std::cos(p * 0.05F + c * 0.017F + h * 0.11F)
                  : 0.3F * std::sin(p * 0.07F + c * 0.013F + h * 0.07F);
}

float QueryValue(std::uint32_t head, std::uint32_t channel) {
  return 0.2F * std::sin(static_cast<float>(head) * 0.3F +
                         static_cast<float>(channel) * 0.021F);
}

/// One cache-resident full-attention layer of KVarN storage sealed with the
/// CPU reference encoder (the framework from
/// tools/qwen27b/kvarn_read_compute_bench.hip).
struct KvarnFixture {
  std::uint32_t context{0};
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
    gufo::hip::QwenKvarnPreset preset{};
    preset.k_bits = k_bits;
    preset.v_bits = v_bits;
    preset.tail_length = tail_tokens;
    tail_rows = gufo::hip::QwenKvarnTailRows(preset);
    ring_rows = gufo::hip::QwenKvarnTailRingRows(preset);
    // Contexts shorter than the sink hold no sealed tiles at all.
    tile_count =
        context > ref::kKvarnSinkRows
            ? static_cast<std::uint32_t>(
                  (context - ref::kKvarnSinkRows + ref::kKvarnTileTokens - 1) /
                  ref::kKvarnTileTokens)
            : 0;
    k_tile_bytes = ref::KvarnTilePayloadBytes(k_bits);
    v_tile_bytes = ref::KvarnTilePayloadBytes(v_bits);
    sink_plane = static_cast<std::size_t>(kLayers) * kKvHeads *
                 ref::kKvarnSinkRows * kHeadDim;
    tail_plane = static_cast<std::size_t>(kLayers) * kKvHeads * ring_rows *
                 kHeadDim;
    HIP_CHECK(hipMalloc(&sink, 2 * sink_plane * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&tail, 2 * tail_plane * sizeof(std::uint16_t)));
    const std::size_t body_bytes = static_cast<std::size_t>(kLayers) *
                                   kKvHeads * tile_count *
                                   (k_tile_bytes + v_tile_bytes);
    if (body_bytes != 0) {
      void* body_raw = nullptr;
      HIP_CHECK(hipMalloc(&body_raw, body_bytes));
      body = static_cast<std::uint8_t*>(body_raw);
    }

    const std::uint32_t query = context - 1;
    const std::uint32_t quantized_tiles =
        query >= ref::kKvarnSinkRows + tail_rows + 1
            ? static_cast<std::uint32_t>(
                  (query - ref::kKvarnSinkRows - tail_rows) /
                  ref::kKvarnTileTokens)
            : 0;
    const std::uint32_t first_f16 =
        ref::kKvarnSinkRows + ref::kKvarnTileTokens * quantized_tiles;

    std::vector<std::uint16_t> sink_host(2 * sink_plane, 0);
    std::vector<std::uint16_t> tail_host(2 * tail_plane, 0);
    const auto write_f16 = [&](std::uint32_t position) {
      for (std::uint32_t head = 0; head < kKvHeads; ++head) {
        for (const bool is_value : {false, true}) {
          for (std::uint32_t c = 0; c < kHeadDim; ++c) {
            const std::uint16_t encoded = __half_as_ushort(
                __float2half_rn(KeyValue(position, c, is_value, head)));
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
              row[c] = KeyValue(position, c, is_value, head);
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
          HIP_CHECK(hipMemcpy(body + offset, &payload,
                              is_value ? v_tile_bytes : k_tile_bytes,
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

}  // namespace

int main(int argc, char** argv) {
  std::string output_path = "/tmp/qwen_decode_attention_dump.bin";
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--output" && i + 1 < argc) {
      output_path = argv[++i];
    } else {
      fprintf(stderr, "unknown option %s\n", flag.c_str());
      return 2;
    }
  }
  g_output = std::fopen(output_path.c_str(), "wb");
  if (g_output == nullptr) {
    std::perror("fopen");
    return 1;
  }

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDeviceProperties(&properties, device));
  printf("device %s\n", properties.name);
  printf("output %s\n", output_path.c_str());

  const std::size_t row_scratch =
      gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kHeadDim);
  // The verification calls span `rows` rows of scratch.
  constexpr std::uint32_t kMaxRows = 2;
  float* d_scratch = nullptr;
  HIP_CHECK(hipMalloc(&d_scratch, kMaxRows * row_scratch * sizeof(float)));

  // ---- online path, sequence length 127 (below the split-K threshold)
  {
    LayerInputs inputs(128, 1);
    float* d_query = nullptr;
    float* d_key = nullptr;
    float* d_value_input = nullptr;
    float* d_gate = nullptr;
    void* d_cache16 = nullptr;
    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_query, kAttentionWidth * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_key, kKvWidth * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_value_input, kKvWidth * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gate, kAttentionWidth * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_cache16, 2 * inputs.cache16.size() *
                                        sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&d_out, kAttentionWidth * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_query, inputs.query.data(),
                        kAttentionWidth * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_key, inputs.key.data(), kKvWidth * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_value_input, inputs.value.data(),
                        kKvWidth * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_gate, inputs.gate.data(),
                        kAttentionWidth * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cache16, inputs.cache16.data(),
                        inputs.cache16.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));
    auto* const cache16 = static_cast<__half*>(d_cache16);
    gufo::hip::LaunchAttention(
        d_query, d_key, d_value_input, d_gate, nullptr, nullptr, cache16,
        cache16 + inputs.cache16.size() / 2, d_out, 0, 126, inputs.context,
        kHeads, kKvHeads, kHeadDim, nullptr, d_scratch);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    DumpDevice("online_n127_output", d_out, kAttentionWidth);
    {
      const std::size_t row_halves = kKvWidth;
      std::vector<std::uint16_t> cache_row(row_halves);
      HIP_CHECK(hipMemcpy(cache_row.data(),
                          cache16 + (126 * kKvWidth),
                          row_halves * sizeof(std::uint16_t),
                          hipMemcpyDeviceToHost));
      DumpRecord("online_n127_cache_row", cache_row.data(),
                 cache_row.size() * sizeof(std::uint16_t));
    }
    (void)hipFree(d_query);
    (void)hipFree(d_key);
    (void)hipFree(d_value_input);
    (void)hipFree(d_gate);
    (void)hipFree(d_cache16);
    (void)hipFree(d_out);
  }

  // ---- FP32 reference-KV path with the MTP argument shape (both planes)
  for (const std::uint32_t context : {128U, 4096U, 32768U}) {
    constexpr std::uint32_t rows = kMaxRows;
    LayerInputs inputs(context, rows);
    float* d_query = nullptr;
    float* d_key = nullptr;
    float* d_value_input = nullptr;
    float* d_gate = nullptr;
    float* d_cache32 = nullptr;
    void* d_cache16 = nullptr;
    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_query, rows * kAttentionWidth * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_key, kKvWidth * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_value_input, kKvWidth * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gate, rows * kAttentionWidth * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_cache32,
                        2 * inputs.cache32.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_cache16, 2 * inputs.cache16.size() *
                                        sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&d_out, rows * kAttentionWidth * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_query, inputs.query.data(),
                        rows * kAttentionWidth * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_key, inputs.key.data(), kKvWidth * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_value_input, inputs.value.data(),
                        kKvWidth * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_gate, inputs.gate.data(),
                        rows * kAttentionWidth * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cache32, inputs.cache32.data(),
                        inputs.cache32.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cache16, inputs.cache16.data(),
                        inputs.cache16.size() * sizeof(std::uint16_t),
                        hipMemcpyHostToDevice));
    auto* const cache16 = static_cast<__half*>(d_cache16);
    // MTP argument shape: both the FP32 and the FP16 planes are supplied, so
    // the FP32 reference-KV path is selected. The KV write is skipped so the
    // dump depends only on the attention arithmetic.
    gufo::hip::LaunchAttention(
        d_query, d_key, d_value_input, d_gate, d_cache32,
        d_cache32 + inputs.cache32.size() / 2, cache16,
        cache16 + inputs.cache16.size() / 2, d_out, 0, context - 1,
        inputs.context, kHeads, kKvHeads, kHeadDim, nullptr, d_scratch,
        /*skip_kv_write=*/true);
    {
      // The last written cache row, so a KV-write difference is visible next
      // to the attention output.
      std::vector<float> written(kKvWidth);
      HIP_CHECK(hipMemcpy(written.data(),
                          d_cache32 + static_cast<std::size_t>(context - 1) *
                                           kKvWidth,
                          kKvWidth * sizeof(float), hipMemcpyDeviceToHost));
      char cache_name[64];
      std::snprintf(cache_name, sizeof(cache_name), "fp32_kv_n%u_cache_row",
                    context);
      DumpRecord(cache_name, written.data(), written.size() * sizeof(float));
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    char name[64];
    std::snprintf(name, sizeof(name), "fp32_kv_n%u_output", context);
    DumpDevice(name, d_out, kAttentionWidth);

    gufo::hip::LaunchCausalDecodeAttention(
        d_query, d_gate, d_cache32, d_cache32 + inputs.cache32.size() / 2,
        cache16, cache16 + inputs.cache16.size() / 2, d_out, 0, context - rows,
        rows, inputs.context, kHeads, kKvHeads, kHeadDim, nullptr,
        {d_scratch, rows * row_scratch});
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::snprintf(name, sizeof(name), "fp32_kv_n%u_rows_output", context);
    DumpDevice(name, d_out, rows * kAttentionWidth);

    (void)hipFree(d_query);
    (void)hipFree(d_key);
    (void)hipFree(d_value_input);
    (void)hipFree(d_gate);
    (void)hipFree(d_cache32);
    (void)hipFree(d_cache16);
    (void)hipFree(d_out);
  }

  // ---- KVarN decode and verification rows over a sealed fixed fixture
  for (const std::uint32_t context : {128U, 4096U, 32768U}) {
    constexpr std::uint32_t rows = 2;
    KvarnFixture fixture(context, 5, 4, 2048);
    float* d_query = nullptr;
    float* d_gate = nullptr;
    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_query, rows * kAttentionWidth * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gate, rows * kAttentionWidth * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, rows * kAttentionWidth * sizeof(float)));
    std::vector<float> query(rows * kAttentionWidth);
    std::vector<float> gate(rows * kAttentionWidth);
    for (std::uint32_t row = 0; row < rows; ++row) {
      for (std::uint32_t head = 0; head < kHeads; ++head) {
        for (std::uint32_t c = 0; c < kHeadDim; ++c) {
          const std::size_t index = (row * kHeads + head) * kHeadDim + c;
          query[index] = QueryValue(head, c);
          gate[index] = std::cos(static_cast<float>(index) * 0.013F);
        }
      }
    }
    HIP_CHECK(hipMemcpy(d_query, query.data(),
                        query.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_gate, gate.data(), gate.size() * sizeof(float),
                        hipMemcpyHostToDevice));
    gufo::hip::LaunchKvarnDecodeAttention(
        d_query, d_gate, d_out, 0, context - 1, nullptr, 1, kHeads, kKvHeads,
        fixture.read, d_scratch, nullptr);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    char name[64];
    std::snprintf(name, sizeof(name), "kvarn_k5v4_n%u_decode", context);
    DumpDevice(name, d_out, kAttentionWidth);

    gufo::hip::LaunchKvarnCausalDecodeAttention(
        d_query, d_gate, d_out, 0, context - rows, rows, kHeads, kKvHeads,
        fixture.read, d_scratch, nullptr);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::snprintf(name, sizeof(name), "kvarn_k5v4_n%u_rows", context);
    DumpDevice(name, d_out, rows * kAttentionWidth);
    (void)hipFree(d_query);
    (void)hipFree(d_gate);
    (void)hipFree(d_out);
  }

  HIP_CHECK(hipFree(d_scratch));
  if (std::fclose(g_output) != 0) {
    std::perror("fclose");
    return 1;
  }
  printf("wrote %zu bytes\n", g_offset);
  return 0;
}
#else
int main() {
  return 77;
}
#endif  // defined(ENGINE_ENABLE_HIP)
