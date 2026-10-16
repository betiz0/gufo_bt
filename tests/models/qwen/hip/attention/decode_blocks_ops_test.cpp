// FP16 decode attention over the block partitions the new kernel uses: the
// series lengths whose whole-tile split leaves partitions without keys must
// still produce finite output that matches the FP64 reference within the
// declared limit (openspec change update-qwen-decode-attention-blocks, tasks
// 2.1/2.4, design D11). The limit below is the same contract as
// tests/models/qwen/hip/attention/long_context_ops_test.cpp and is recorded in
// docs/models/qwen3.8-27b/QUALITY.md.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

/// Declared limit for "matches within the declared limit" in this test and in
/// the block kernel's FP64 comparisons (QUALITY.md records the same value).
constexpr float kDecodeAttentionFp64MaxAbsTolerance = 5.0e-4F;
constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 4;
constexpr std::uint32_t kHeadDim = 256;
// Row 1 of the mixed-partition check reads up to token 4096.
constexpr std::uint32_t kMaxContext = 4097;

using gufo::hip::detail::DecodeAttentionBlockRange;
using gufo::hip::detail::kSplitKDecodeAttentionMaxSplits;
using gufo::hip::detail::SelectDecodeAttentionBlockRange;

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "decode attention blocks: %s\n", message);
    std::abort();
  }
}

/// FP64 oracle over exactly the FP16 cache values, like the long-context test.
void ReferenceForRow(const std::vector<float>& query,
                     const std::vector<float>& gate_values,
                     const std::vector<__half>& cache, std::uint32_t position,
                     std::vector<double>& reference) {
  const std::size_t kv_width = static_cast<std::size_t>(kKvHeads) * kHeadDim;
  const std::size_t tokens = static_cast<std::size_t>(position) + 1;
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
  reference.assign(static_cast<std::size_t>(kHeads) * kHeadDim, 0.0);
  std::vector<double> scores(tokens);
  for (std::uint32_t head = 0; head < kHeads; ++head) {
    const __half* const cache_head =
        cache.data() + (static_cast<std::size_t>(head / 6) * kHeadDim);
    const float* const query_head =
        query.data() + (static_cast<std::size_t>(head) * kHeadDim);
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t token = 0; token < tokens; ++token) {
      double dot = 0.0;
      const __half* const key = cache_head + (token * kv_width);
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        dot += static_cast<double>(query_head[d]) *
               static_cast<double>(__half2float(key[d]));
      }
      scores[token] = dot * scale;
      maximum = std::max(maximum, scores[token]);
    }
    double denominator = 0.0;
    for (std::size_t token = 0; token < tokens; ++token) {
      scores[token] = std::exp(scores[token] - maximum);
      denominator += scores[token];
    }
    for (std::uint32_t d = 0; d < kHeadDim; ++d) {
      double weighted = 0.0;
      for (std::size_t token = 0; token < tokens; ++token) {
        weighted += scores[token] * static_cast<double>(__half2float(
                                        cache_head[(token * kv_width) + d]));
      }
      const double sigmoid =
          1.0 /
          (1.0 +
           std::exp(-static_cast<double>(
               gate_values[(static_cast<std::size_t>(head) * kHeadDim) + d])));
      reference[(static_cast<std::size_t>(head) * kHeadDim) + d] =
          (weighted / denominator) * sigmoid;
    }
  }
}

void CheckSequenceLength(std::uint32_t sequence_length) {
  const std::size_t width = static_cast<std::size_t>(kHeads) * kHeadDim;
  const std::size_t kv_width = static_cast<std::size_t>(kKvHeads) * kHeadDim;
  const std::size_t cache_elements =
      static_cast<std::size_t>(kMaxContext) * kv_width;
  std::vector<float> query(width);
  std::vector<float> gate_values(width);
  std::vector<float> cache_values(cache_elements);
  for (std::size_t i = 0; i < width; ++i) {
    query[i] = 0.1F * std::sin(static_cast<float>((i % 257) + 1) * 0.013F);
    gate_values[i] =
        0.4F * std::cos(static_cast<float>((i % 193) + 1) * 0.017F);
  }
  for (std::size_t i = 0; i < cache_elements; ++i) {
    const float value =
        0.03F * std::sin(static_cast<float>((i % 509) + 1) * 0.011F);
    cache_values[i] = __half2float(__float2half_rn(value));
  }
  std::vector<__half> cache(cache_elements);
  for (std::size_t i = 0; i < cache_elements; ++i) {
    cache[i] = __float2half_rn(cache_values[i]);
  }

  // The split the kernel uses must leave the documented key-less partitions.
  std::uint32_t empty_partitions = 0;
  for (std::uint32_t split = 0; split < kSplitKDecodeAttentionMaxSplits;
       ++split) {
    const DecodeAttentionBlockRange range =
        SelectDecodeAttentionBlockRange(sequence_length, split);
    if (range.empty()) {
      ++empty_partitions;
    }
  }
  // Whole tiles are spread over the 32 partitions, so a partition is empty
  // exactly while the row has fewer 128-token tiles than partitions.
  const std::uint32_t blocks = static_cast<std::uint32_t>(
      (sequence_length + gufo::hip::detail::kDecodeAttentionBlockTokens - 1) /
      gufo::hip::detail::kDecodeAttentionBlockTokens);
  Expect(empty_partitions == kSplitKDecodeAttentionMaxSplits -
                                 std::min<std::uint32_t>(
                                     blocks, kSplitKDecodeAttentionMaxSplits),
         "the number of key-less partitions does not match the tile count");

  float* d_query = nullptr;
  float* d_gate = nullptr;
  void* d_k_cache = nullptr;
  void* d_v_cache = nullptr;
  float* d_out = nullptr;
  float* d_scratch = nullptr;
  const std::size_t cache_bytes = cache_elements * sizeof(__half);
  const std::size_t row_scratch =
      gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kHeadDim);
  HIP_CHECK(hipMalloc(&d_query, width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_cache, cache_bytes));
  HIP_CHECK(hipMalloc(&d_v_cache, cache_bytes));
  HIP_CHECK(hipMalloc(&d_out, width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_scratch, row_scratch * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_query, query.data(), width * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, gate_values.data(), width * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_k_cache, cache.data(), cache_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_v_cache, cache.data(), cache_bytes, hipMemcpyHostToDevice));

  gufo::hip::LaunchAttention(
      d_query, nullptr, nullptr, d_gate, nullptr, nullptr, d_k_cache, d_v_cache,
      d_out, 0, sequence_length - 1, kMaxContext, kHeads, kKvHeads, kHeadDim,
      nullptr, d_scratch, /*skip_kv_write=*/true);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> output(width);
  HIP_CHECK(hipMemcpy(output.data(), d_out, width * sizeof(float),
                      hipMemcpyDeviceToHost));

  std::vector<double> reference;
  ReferenceForRow(query, gate_values, cache, sequence_length - 1, reference);
  float max_abs = 0.0F;
  for (std::size_t i = 0; i < width; ++i) {
    Expect(std::isfinite(output[i]),
           "a key-less partition produced a non-finite output");
    max_abs = std::max(
        max_abs, static_cast<float>(
                     std::abs(static_cast<double>(output[i]) - reference[i])));
  }
  std::printf("n=%u key-less partitions=%u max FP64 diff %.6e\n",
              sequence_length, empty_partitions, max_abs);
  Expect(max_abs < kDecodeAttentionFp64MaxAbsTolerance,
         "output left the declared FP64 limit");
  Expect(max_abs > 0.0F, "output is identically zero");

  HIP_CHECK(hipFree(d_query));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_k_cache));
  HIP_CHECK(hipFree(d_v_cache));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_scratch));
}

/// A launch that mixes 4096- and 4097-token rows: the two rows use different
/// partition counts, and each row must keep the bits it produces alone.
void CheckMixedPartitionRows() {
  const std::size_t width = static_cast<std::size_t>(kHeads) * kHeadDim;
  const std::size_t kv_width = static_cast<std::size_t>(kKvHeads) * kHeadDim;
  const std::size_t cache_elements =
      static_cast<std::size_t>(kMaxContext) * kv_width;
  std::vector<float> query(2 * width);
  std::vector<float> gate_values(2 * width);
  std::vector<float> cache_values(cache_elements);
  for (std::size_t i = 0; i < query.size(); ++i) {
    query[i] = 0.1F * std::sin(static_cast<float>((i % 257) + 1) * 0.013F);
    gate_values[i] =
        0.4F * std::cos(static_cast<float>((i % 193) + 1) * 0.017F);
  }
  std::vector<__half> cache(cache_elements);
  for (std::size_t i = 0; i < cache_elements; ++i) {
    cache_values[i] =
        0.03F * std::sin(static_cast<float>((i % 509) + 1) * 0.011F);
    cache[i] = __float2half_rn(cache_values[i]);
  }
  float* d_query = nullptr;
  float* d_gate = nullptr;
  void* d_k_cache = nullptr;
  void* d_v_cache = nullptr;
  float* d_out = nullptr;
  float* d_single = nullptr;
  float* d_scratch = nullptr;
  const std::size_t cache_bytes = cache_elements * sizeof(__half);
  const std::size_t row_scratch =
      gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kHeadDim);
  HIP_CHECK(hipMalloc(&d_query, query.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, gate_values.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_cache, cache_bytes));
  HIP_CHECK(hipMalloc(&d_v_cache, cache_bytes));
  HIP_CHECK(hipMalloc(&d_out, query.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_single, width * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_scratch, 2 * row_scratch * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_query, query.data(), query.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, gate_values.data(),
                      gate_values.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_k_cache, cache.data(), cache_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_v_cache, cache.data(), cache_bytes, hipMemcpyHostToDevice));
  // Row 0 sees 4096 tokens, row 1 sees 4097: 32 partitions of 128 tiles
  // against 32 partitions where the last holds two tiles.
  gufo::hip::LaunchCausalDecodeAttention(
      d_query, d_gate, nullptr, nullptr, d_k_cache, d_v_cache, d_out, 0, 4095,
      2, kMaxContext, kHeads, kKvHeads, kHeadDim, nullptr,
      {d_scratch, 2 * row_scratch});
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> mixed(2 * width);
  HIP_CHECK(hipMemcpy(mixed.data(), d_out, mixed.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  for (std::uint32_t row = 0; row < 2; ++row) {
    gufo::hip::LaunchCausalDecodeAttention(
        d_query + (row * width), d_gate + (row * width), nullptr, nullptr,
        d_k_cache, d_v_cache, d_single, 0, 4095 + row, 1, kMaxContext, kHeads,
        kKvHeads, kHeadDim, nullptr, {d_scratch, row_scratch});
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> single(width);
    HIP_CHECK(hipMemcpy(single.data(), d_single, width * sizeof(float),
                        hipMemcpyDeviceToHost));
    Expect(std::memcmp(mixed.data() + (row * width), single.data(),
                       width * sizeof(float)) == 0,
           "a mixed-partition row changed bits");
  }
  std::printf("mixed-partition rows 4096/4097 match their single-row runs\n");
  HIP_CHECK(hipFree(d_query));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_k_cache));
  HIP_CHECK(hipFree(d_v_cache));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_single));
  HIP_CHECK(hipFree(d_scratch));
}

}  // namespace

int main() {
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen decode attention blocks test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }
  // Series lengths whose whole-tile split leaves partitions without keys.
  for (std::uint32_t sequence_length : {128U, 129U, 3968U, 4095U, 4096U}) {
    CheckSequenceLength(sequence_length);
  }
  CheckMixedPartitionRows();
  std::printf("Qwen decode attention blocks test passed.\n");
  return 0;
}
#else
int main() {
  std::printf("HIP disabled, skipping Qwen decode attention blocks test.\n");
  return 77;
}
#endif  // defined(ENGINE_ENABLE_HIP)
