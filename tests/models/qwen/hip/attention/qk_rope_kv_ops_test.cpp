#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/detail/hip_graph_decode_executor.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/modules/ffn.hpp"
#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/modules/norm.hpp"
#include "src/models/qwen/modules/quant_gemm.hpp"
#include "src/models/qwen/modules/residual.hpp"
#include "tests/models/qwen/hip/support/bfloat16.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

// DFlash2 uses in-place Q/K and no target KV-cache outputs. Context injection
// also uses the key-only path with no query heads.
void TestNormRoPEWithoutCache() {
  using gufo::test::DeviceBuffer;
  for (const unsigned dim : {128U, 256U}) {
    for (const unsigned rows : {1U, 7U, 8U, 16U}) {
      for (const unsigned position : {0U, 4096U, 16384U}) {
        for (const unsigned heads : {0U, 32U}) {
          constexpr unsigned kv_heads = 8;
          std::vector<float> q(rows * 32 * dim), k(rows * kv_heads * dim),
              weights(dim);
          for (std::size_t i = 0; i < q.size(); ++i)
            q[i] = 0.4F * std::sin(static_cast<float>(i) * 0.021F);
          for (std::size_t i = 0; i < k.size(); ++i)
            k[i] = 0.3F * std::cos(static_cast<float>(i) * 0.031F);
          for (std::size_t i = 0; i < weights.size(); ++i)
            weights[i] = 0.5F + static_cast<float>(i) * 0.001F;
          DeviceBuffer<float> q_ref(q), k_ref(k), q_fused(q), k_fused(k),
              weight(weights);
          if (heads != 0) {
            gufo::hip::LaunchBatchedPerHeadRMSNorm(
                q_ref.data(), weight.data(), q_ref.data(), rows, heads, dim);
          }
          gufo::hip::LaunchBatchedPerHeadRMSNorm(
              k_ref.data(), weight.data(), k_ref.data(), rows, kv_heads, dim);
          gufo::hip::LaunchBatchedRoPE(q_ref.data(), k_ref.data(), rows, heads,
                                       kv_heads, dim, 64, position,
                                       10000000.0F);
          gufo::hip::LaunchBatchedFusedQKNormRoPEKvWrite(
              heads != 0 ? q_fused.data() : nullptr, k_fused.data(),
              k_fused.data(), weight.data(), weight.data(), q_fused.data(),
              k_fused.data(), nullptr, nullptr, nullptr, nullptr, 0, position,
              rows, 0, heads, kv_heads, dim, 64, 10000000.0F);
          const auto check = [](const auto& expected, const auto& actual) {
            const auto a = expected.CopyToHost(), b = actual.CopyToHost();
            if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float))) {
              throw std::runtime_error("DFlash2 QK norm/RoPE bits differ");
            }
          };
          check(q_ref, q_fused);
          check(k_ref, k_fused);
        }
      }
    }
  }
}

void TestFusedQKNormRoPEKvWriteEquivalence() {
  constexpr std::uint32_t num_heads = 4;
  constexpr std::uint32_t num_kv_heads = 2;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t rotary_dim = 64;
  constexpr float rope_theta = 1000000.0F;
  constexpr float eps = 1e-6F;
  constexpr std::uint32_t layer_idx = 1;
  constexpr std::uint32_t max_context = 16;
  constexpr std::uint32_t pos = 3;

  const std::size_t q_size = num_heads * head_dim;
  const std::size_t kv_size = num_kv_heads * head_dim;
  const std::size_t kv_width = num_kv_heads * head_dim;
  const std::size_t f32_cache_elems =
      (static_cast<std::size_t>(layer_idx) + 1) * num_kv_heads * max_context *
      head_dim;
  const std::size_t f16_cache_elems =
      (static_cast<std::size_t>(layer_idx) + 1) * max_context * kv_width;

  std::vector<float> h_q(q_size), h_k(kv_size), h_v(kv_size);
  std::vector<float> h_q_w(head_dim), h_k_w(head_dim);
  std::vector<float> h_gate(q_size);
  for (std::size_t i = 0; i < q_size; ++i) {
    h_q[i] = 0.18F * std::sin(static_cast<float>(i + 1) * 0.05F);
    h_gate[i] = 0.3F * std::cos(static_cast<float>(i + 1) * 0.03F);
  }
  for (std::size_t i = 0; i < kv_size; ++i) {
    h_k[i] = 0.11F * std::cos(static_cast<float>(i + 1) * 0.07F);
    h_v[i] = 0.24F * std::sin(static_cast<float>(i + 1) * 0.04F);
  }
  for (std::size_t i = 0; i < head_dim; ++i) {
    h_q_w[i] = 0.9F + 0.04F * static_cast<float>(i % 17);
    h_k_w[i] = 1.1F - 0.03F * static_cast<float>(i % 13);
  }

  float *d_q_ref = nullptr, *d_k_ref = nullptr, *d_v_ref = nullptr;
  float *d_q_fus = nullptr, *d_k_fus = nullptr, *d_v_fus = nullptr;
  float *d_wq = nullptr, *d_wk = nullptr, *d_gate = nullptr;
  float *d_kc_ref = nullptr, *d_vc_ref = nullptr;
  float *d_kc_fus = nullptr, *d_vc_fus = nullptr;
  void *d_kc16_ref = nullptr, *d_vc16_ref = nullptr;
  void *d_kc16_fus = nullptr, *d_vc16_fus = nullptr;
  float *d_out_ref = nullptr, *d_out_fus = nullptr;
  std::uint32_t* d_pos = nullptr;

  HIP_CHECK(hipMalloc(&d_q_ref, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_ref, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_ref, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q_fus, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_fus, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_fus, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_wq, head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_wk, head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc_ref, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_vc_ref, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc_fus, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_vc_fus, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc16_ref, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_vc16_ref, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_kc16_fus, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_vc16_fus, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out_ref, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_fus, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_pos, sizeof(std::uint32_t)));

  HIP_CHECK(hipMemcpy(d_q_ref, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_ref, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v_ref, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_q_fus, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_fus, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v_fus, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_wq, h_q_w.data(), head_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_wk, h_k_w.data(), head_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_pos, &pos, sizeof(std::uint32_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_kc_ref, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_vc_ref, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_kc_fus, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_vc_fus, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_kc16_ref, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_vc16_ref, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_kc16_fus, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_vc16_fus, 0, f16_cache_elems * sizeof(std::uint16_t)));

  // Unfused reference chain
  gufo::hip::LaunchPerHeadRMSNorm(d_q_ref, d_wq, d_q_ref, num_heads, head_dim,
                                  eps);
  gufo::hip::LaunchPerHeadRMSNorm(d_k_ref, d_wk, d_k_ref, num_kv_heads,
                                  head_dim, eps);
  gufo::hip::LaunchRoPE(d_q_ref, d_k_ref, num_heads, num_kv_heads, head_dim,
                        rotary_dim, d_pos, rope_theta);
  gufo::hip::LaunchAttention(d_q_ref, d_k_ref, d_v_ref, d_gate, d_kc_ref,
                             d_vc_ref, d_kc16_ref, d_vc16_ref, d_out_ref,
                             layer_idx, d_pos, max_context, num_heads,
                             num_kv_heads, head_dim, nullptr,
                             /*skip_kv_write=*/false);

  // Fused chain
  gufo::hip::LaunchFusedQKNormRoPEKvWrite(
      d_q_fus, d_k_fus, d_v_fus, d_wq, d_wk, d_q_fus, d_k_fus, d_kc_fus,
      d_vc_fus, d_kc16_fus, d_vc16_fus, layer_idx, d_pos, max_context,
      num_heads, num_kv_heads, head_dim, rotary_dim, rope_theta, eps);
  gufo::hip::LaunchAttention(d_q_fus, d_k_fus, d_v_fus, d_gate, d_kc_fus,
                             d_vc_fus, d_kc16_fus, d_vc16_fus, d_out_fus,
                             layer_idx, d_pos, max_context, num_heads,
                             num_kv_heads, head_dim, nullptr,
                             /*skip_kv_write=*/true);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_q_ref(q_size), res_q_fus(q_size);
  std::vector<float> res_k_ref(kv_size), res_k_fus(kv_size);
  std::vector<float> res_kc_ref(f32_cache_elems), res_kc_fus(f32_cache_elems);
  std::vector<float> res_vc_ref(f32_cache_elems), res_vc_fus(f32_cache_elems);
  std::vector<std::uint16_t> res_kc16_ref(f16_cache_elems),
      res_kc16_fus(f16_cache_elems);
  std::vector<std::uint16_t> res_vc16_ref(f16_cache_elems),
      res_vc16_fus(f16_cache_elems);
  std::vector<float> res_out_ref(q_size), res_out_fus(q_size);
  HIP_CHECK(hipMemcpy(res_q_ref.data(), d_q_ref, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_q_fus.data(), d_q_fus, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_ref.data(), d_k_ref, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_fus.data(), d_k_fus, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc_ref.data(), d_kc_ref,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc_fus.data(), d_kc_fus,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc_ref.data(), d_vc_ref,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc_fus.data(), d_vc_fus,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc16_ref.data(), d_kc16_ref,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc16_fus.data(), d_kc16_fus,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc16_ref.data(), d_vc16_ref,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc16_fus.data(), d_vc16_fus,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_ref.data(), d_out_ref, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_fus.data(), d_out_fus, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t i = 0; i < q_size; ++i) {
    if (res_q_ref[i] != res_q_fus[i]) {
      std::cerr << "Fused Q mismatch at " << i << ": " << res_q_ref[i] << " vs "
                << res_q_fus[i] << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < kv_size; ++i) {
    if (res_k_ref[i] != res_k_fus[i]) {
      std::cerr << "Fused K mismatch at " << i << ": " << res_k_ref[i] << " vs "
                << res_k_fus[i] << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < f32_cache_elems; ++i) {
    if (res_kc_ref[i] != res_kc_fus[i] || res_vc_ref[i] != res_vc_fus[i]) {
      std::cerr << "Fused FP32 KV-cache mismatch at " << i << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < f16_cache_elems; ++i) {
    if (res_kc16_ref[i] != res_kc16_fus[i] ||
        res_vc16_ref[i] != res_vc16_fus[i]) {
      std::cerr << "Fused FP16 KV-cache mismatch at " << i << "\n";
      std::abort();
    }
  }
  float max_out_diff = 0.0F;
  for (std::size_t i = 0; i < q_size; ++i) {
    max_out_diff =
        std::max(max_out_diff, std::abs(res_out_ref[i] - res_out_fus[i]));
  }
  std::cout << "FusedQKNormRoPEKvWrite decode: exact Q/K/cache match, max out "
               "diff "
            << max_out_diff << "\n";
  if (max_out_diff >= 1e-4F) {
    std::cerr << "Fused decode attention output mismatch: " << max_out_diff
              << "\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_q_ref));
  HIP_CHECK(hipFree(d_k_ref));
  HIP_CHECK(hipFree(d_v_ref));
  HIP_CHECK(hipFree(d_q_fus));
  HIP_CHECK(hipFree(d_k_fus));
  HIP_CHECK(hipFree(d_v_fus));
  HIP_CHECK(hipFree(d_wq));
  HIP_CHECK(hipFree(d_wk));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_kc_ref));
  HIP_CHECK(hipFree(d_vc_ref));
  HIP_CHECK(hipFree(d_kc_fus));
  HIP_CHECK(hipFree(d_vc_fus));
  HIP_CHECK(hipFree(d_kc16_ref));
  HIP_CHECK(hipFree(d_vc16_ref));
  HIP_CHECK(hipFree(d_kc16_fus));
  HIP_CHECK(hipFree(d_vc16_fus));
  HIP_CHECK(hipFree(d_out_ref));
  HIP_CHECK(hipFree(d_out_fus));
  HIP_CHECK(hipFree(d_pos));
}

void TestBatchedFusedQKNormRoPEKvWriteEquivalence() {
  constexpr std::size_t batch = 8;
  constexpr std::uint32_t num_heads = 4;
  constexpr std::uint32_t num_kv_heads = 2;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t rotary_dim = 64;
  constexpr float rope_theta = 1000000.0F;
  constexpr float eps = 1e-6F;
  constexpr std::uint32_t layer_idx = 2;
  constexpr std::uint32_t max_context = 64;
  constexpr std::uint32_t start_pos = 5;

  const std::size_t head_total = num_heads * head_dim;
  const std::size_t kv_total = num_kv_heads * head_dim;
  const std::size_t q_size = batch * head_total;
  const std::size_t kv_size = batch * kv_total;
  const std::size_t kv_width = num_kv_heads * head_dim;
  const std::size_t f32_cache_elems =
      (static_cast<std::size_t>(layer_idx) + 1) * num_kv_heads * max_context *
      head_dim;
  const std::size_t f16_cache_elems =
      (static_cast<std::size_t>(layer_idx) + 1) * max_context * kv_width;

  std::vector<float> h_q(q_size), h_k(kv_size), h_v(kv_size);
  std::vector<float> h_q_w(head_dim), h_k_w(head_dim);
  std::vector<float> h_gate(q_size);
  for (std::size_t i = 0; i < q_size; ++i) {
    h_q[i] = 0.16F * std::sin(static_cast<float>(i + 1) * 0.04F);
    h_gate[i] = 0.25F * std::cos(static_cast<float>(i + 1) * 0.06F);
  }
  for (std::size_t i = 0; i < kv_size; ++i) {
    h_k[i] = 0.13F * std::cos(static_cast<float>(i + 1) * 0.05F);
    h_v[i] = 0.21F * std::sin(static_cast<float>(i + 1) * 0.08F);
  }
  for (std::size_t i = 0; i < head_dim; ++i) {
    h_q_w[i] = 0.95F + 0.03F * static_cast<float>(i % 19);
    h_k_w[i] = 1.05F - 0.02F * static_cast<float>(i % 11);
  }

  float *d_q_ref = nullptr, *d_k_ref = nullptr, *d_v_ref = nullptr;
  float *d_q_fus = nullptr, *d_k_fus = nullptr, *d_v_fus = nullptr;
  float *d_wq = nullptr, *d_wk = nullptr, *d_gate = nullptr;
  float *d_kc_ref = nullptr, *d_vc_ref = nullptr;
  float *d_kc_fus = nullptr, *d_vc_fus = nullptr;
  void *d_kc16_ref = nullptr, *d_vc16_ref = nullptr;
  void *d_kc16_fus = nullptr, *d_vc16_fus = nullptr;
  float *d_out_ref = nullptr, *d_out_fus = nullptr;

  HIP_CHECK(hipMalloc(&d_q_ref, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_ref, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_ref, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_q_fus, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k_fus, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_fus, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_wq, head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_wk, head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc_ref, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_vc_ref, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc_fus, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_vc_fus, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc16_ref, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_vc16_ref, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_kc16_fus, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_vc16_fus, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_out_ref, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out_fus, q_size * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_q_ref, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_ref, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v_ref, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_q_fus, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_fus, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v_fus, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_wq, h_q_w.data(), head_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_wk, h_k_w.data(), head_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate, h_gate.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_kc_ref, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_vc_ref, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_kc_fus, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_vc_fus, 0, f32_cache_elems * sizeof(float)));
  HIP_CHECK(hipMemset(d_kc16_ref, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_vc16_ref, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_kc16_fus, 0, f16_cache_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_vc16_fus, 0, f16_cache_elems * sizeof(std::uint16_t)));

  // Per-token unfused reference chain
  for (std::size_t t = 0; t < batch; ++t) {
    gufo::hip::LaunchPerHeadRMSNorm(d_q_ref + t * head_total, d_wq,
                                    d_q_ref + t * head_total, num_heads,
                                    head_dim, eps);
    gufo::hip::LaunchPerHeadRMSNorm(d_k_ref + t * kv_total, d_wk,
                                    d_k_ref + t * kv_total, num_kv_heads,
                                    head_dim, eps);
    gufo::hip::LaunchRoPE(d_q_ref + t * head_total, d_k_ref + t * kv_total,
                          num_heads, num_kv_heads, head_dim, rotary_dim,
                          static_cast<std::uint32_t>(start_pos + t),
                          rope_theta);
    gufo::hip::LaunchAttention(
        d_q_ref + t * head_total, d_k_ref + t * kv_total,
        d_v_ref + t * kv_total, d_gate + t * head_total, d_kc_ref, d_vc_ref,
        d_kc16_ref, d_vc16_ref, d_out_ref + t * head_total, layer_idx,
        static_cast<std::uint32_t>(start_pos + t), max_context, num_heads,
        num_kv_heads, head_dim, nullptr, nullptr, /*skip_kv_write=*/false);
  }

  // Batched fused chain
  gufo::hip::LaunchBatchedFusedQKNormRoPEKvWrite(
      d_q_fus, d_k_fus, d_v_fus, d_wq, d_wk, d_q_fus, d_k_fus, d_kc_fus,
      d_vc_fus, d_kc16_fus, d_vc16_fus, layer_idx, start_pos, batch,
      max_context, num_heads, num_kv_heads, head_dim, rotary_dim, rope_theta,
      eps);
  gufo::hip::LaunchBatchedAttention(d_q_fus, d_k_fus, d_v_fus, d_gate, d_kc_fus,
                                    d_vc_fus, d_kc16_fus, d_vc16_fus, d_out_fus,
                                    layer_idx, start_pos, batch, max_context,
                                    num_heads, num_kv_heads, head_dim, nullptr,
                                    /*skip_kv_write=*/true);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> res_q_ref(q_size), res_q_fus(q_size);
  std::vector<float> res_k_ref(kv_size), res_k_fus(kv_size);
  std::vector<float> res_kc_ref(f32_cache_elems), res_kc_fus(f32_cache_elems);
  std::vector<float> res_vc_ref(f32_cache_elems), res_vc_fus(f32_cache_elems);
  std::vector<std::uint16_t> res_kc16_ref(f16_cache_elems),
      res_kc16_fus(f16_cache_elems);
  std::vector<std::uint16_t> res_vc16_ref(f16_cache_elems),
      res_vc16_fus(f16_cache_elems);
  std::vector<float> res_out_ref(q_size), res_out_fus(q_size);
  HIP_CHECK(hipMemcpy(res_q_ref.data(), d_q_ref, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_q_fus.data(), d_q_fus, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_ref.data(), d_k_ref, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_k_fus.data(), d_k_fus, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc_ref.data(), d_kc_ref,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc_fus.data(), d_kc_fus,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc_ref.data(), d_vc_ref,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc_fus.data(), d_vc_fus,
                      f32_cache_elems * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc16_ref.data(), d_kc16_ref,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_kc16_fus.data(), d_kc16_fus,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc16_ref.data(), d_vc16_ref,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_vc16_fus.data(), d_vc16_fus,
                      f16_cache_elems * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_ref.data(), d_out_ref, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(res_out_fus.data(), d_out_fus, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (std::size_t i = 0; i < q_size; ++i) {
    if (res_q_ref[i] != res_q_fus[i]) {
      std::cerr << "Batched fused Q mismatch at " << i << ": " << res_q_ref[i]
                << " vs " << res_q_fus[i] << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < kv_size; ++i) {
    if (res_k_ref[i] != res_k_fus[i]) {
      std::cerr << "Batched fused K mismatch at " << i << ": " << res_k_ref[i]
                << " vs " << res_k_fus[i] << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < f32_cache_elems; ++i) {
    if (res_kc_ref[i] != res_kc_fus[i] || res_vc_ref[i] != res_vc_fus[i]) {
      std::cerr << "Batched fused FP32 KV-cache mismatch at " << i << "\n";
      std::abort();
    }
  }
  for (std::size_t i = 0; i < f16_cache_elems; ++i) {
    if (res_kc16_ref[i] != res_kc16_fus[i] ||
        res_vc16_ref[i] != res_vc16_fus[i]) {
      std::cerr << "Batched fused FP16 KV-cache mismatch at " << i << "\n";
      std::abort();
    }
  }
  float max_out_diff = 0.0F;
  for (std::size_t i = 0; i < q_size; ++i) {
    max_out_diff =
        std::max(max_out_diff, std::abs(res_out_ref[i] - res_out_fus[i]));
  }
  std::cout << "BatchedFusedQKNormRoPEKvWrite prefill: exact Q/K/cache match, "
               "max out diff "
            << max_out_diff << "\n";
  if (max_out_diff >= 1e-4F) {
    std::cerr << "Batched fused prefill attention output mismatch: "
              << max_out_diff << "\n";
    std::abort();
  }

  HIP_CHECK(hipFree(d_q_ref));
  HIP_CHECK(hipFree(d_k_ref));
  HIP_CHECK(hipFree(d_v_ref));
  HIP_CHECK(hipFree(d_q_fus));
  HIP_CHECK(hipFree(d_k_fus));
  HIP_CHECK(hipFree(d_v_fus));
  HIP_CHECK(hipFree(d_wq));
  HIP_CHECK(hipFree(d_wk));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_kc_ref));
  HIP_CHECK(hipFree(d_vc_ref));
  HIP_CHECK(hipFree(d_kc_fus));
  HIP_CHECK(hipFree(d_vc_fus));
  HIP_CHECK(hipFree(d_kc16_ref));
  HIP_CHECK(hipFree(d_vc16_ref));
  HIP_CHECK(hipFree(d_kc16_fus));
  HIP_CHECK(hipFree(d_vc16_fus));
  HIP_CHECK(hipFree(d_out_ref));
  HIP_CHECK(hipFree(d_out_fus));
}

// T-02 (design D3): the fused write rotates Q and K with the same normalized
// Hadamard, so the FP32 QK score is unchanged, and the rotated K lands in the
// KVarN sink. The declared limit is recorded in QUALITY.md.
void TestKvarnRotationPreservesQkScore() {
  constexpr std::uint32_t num_heads = 4;
  constexpr std::uint32_t num_kv_heads = 2;
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t rotary_dim = 64;
  constexpr float rope_theta = 1000000.0F;
  constexpr float eps = 1e-6F;
  constexpr std::uint32_t layer_idx = 1;
  constexpr std::uint32_t max_context = 16;
  constexpr std::uint32_t pos = 3;
  constexpr std::uint32_t ring_rows = 256 + 128 + 9;
  // Declared limit of QUALITY.md (T-02): rotation preserves the QK score.
  constexpr double kQkScoreLimit = 1e-4;

  const std::size_t q_size = num_heads * head_dim;
  const std::size_t kv_size = num_kv_heads * head_dim;
  const std::size_t layers = layer_idx + 1;
  const std::size_t f16_elems = layers * max_context * kv_size;
  const std::size_t sink_plane = layers * num_kv_heads * 128 * head_dim;
  const std::size_t tail_plane = layers * num_kv_heads * ring_rows * head_dim;

  std::vector<float> h_q(q_size), h_k(kv_size), h_v(kv_size);
  std::vector<float> h_q_w(head_dim), h_k_w(head_dim);
  for (std::size_t i = 0; i < q_size; ++i)
    h_q[i] = 0.18F * std::sin(static_cast<float>(i + 1) * 0.05F);
  for (std::size_t i = 0; i < kv_size; ++i) {
    h_k[i] = 0.11F * std::cos(static_cast<float>(i + 1) * 0.07F);
    h_v[i] = 0.24F * std::sin(static_cast<float>(i + 1) * 0.04F);
  }
  for (std::size_t i = 0; i < head_dim; ++i) {
    h_q_w[i] = 0.9F + 0.04F * static_cast<float>(i % 17);
    h_k_w[i] = 1.1F - 0.03F * static_cast<float>(i % 13);
  }

  float *d_q16 = nullptr, *d_k16 = nullptr, *d_v16 = nullptr;
  float *d_qk = nullptr, *d_kk = nullptr, *d_vk = nullptr;
  float *d_qw = nullptr, *d_kw = nullptr;
  void *d_kc16 = nullptr, *d_vc16 = nullptr;
  void *d_sink = nullptr, *d_tail = nullptr;
  std::uint32_t* d_pos = nullptr;
  HIP_CHECK(hipMalloc(&d_q16, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k16, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v16, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_qk, q_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kk, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_vk, kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_qw, head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kw, head_dim * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_kc16, f16_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_vc16, f16_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_sink, 2 * sink_plane * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_tail, 2 * tail_plane * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&d_pos, sizeof(std::uint32_t)));

  HIP_CHECK(hipMemcpy(d_q16, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k16, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v16, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qk, h_q.data(), q_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_kk, h_k.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_vk, h_v.data(), kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_qw, h_q_w.data(), head_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_kw, h_k_w.data(), head_dim * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(
      hipMemcpy(d_pos, &pos, sizeof(std::uint32_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_kc16, 0, f16_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_vc16, 0, f16_elems * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_sink, 0, 2 * sink_plane * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemset(d_tail, 0, 2 * tail_plane * sizeof(std::uint16_t)));

  // FP16 path: no rotation.
  gufo::hip::LaunchFusedQKNormRoPEKvWrite(
      d_q16, d_k16, d_v16, d_qw, d_kw, d_q16, d_k16, nullptr, nullptr, d_kc16,
      d_vc16, layer_idx, d_pos, max_context, num_heads, num_kv_heads, head_dim,
      rotary_dim, rope_theta, eps);

  // KVarN path: rotate Q/K/V and store K/V in the sink.
  gufo::hip::KvarnWriteParams kvarn;
  kvarn.sink = static_cast<__half*>(d_sink);
  kvarn.tail = static_cast<__half*>(d_tail);
  kvarn.sink_plane_elements = sink_plane;
  kvarn.tail_plane_elements = tail_plane;
  kvarn.ring_rows = ring_rows;
  gufo::hip::LaunchFusedQKNormRoPEKvWrite(
      d_qk, d_kk, d_vk, d_qw, d_kw, d_qk, d_kk, nullptr, nullptr, nullptr,
      nullptr, layer_idx, d_pos, max_context, num_heads, num_kv_heads,
      head_dim, rotary_dim, rope_theta, eps, nullptr, nullptr, kvarn);

  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> q16(q_size), k16(kv_size);
  std::vector<float> qk(q_size), kk(kv_size);
  HIP_CHECK(hipMemcpy(q16.data(), d_q16, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(k16.data(), d_k16, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(qk.data(), d_qk, q_size * sizeof(float),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(kk.data(), d_kk, kv_size * sizeof(float),
                      hipMemcpyDeviceToHost));

  const std::size_t group = num_heads / num_kv_heads;
  double worst = 0.0;
  for (std::uint32_t head = 0; head < num_heads; ++head) {
    const std::uint32_t kv_h = head / static_cast<std::uint32_t>(group);
    double reference = 0.0, rotated = 0.0;
    for (std::uint32_t d = 0; d < head_dim; ++d) {
      reference += static_cast<double>(q16[head * head_dim + d]) *
                   static_cast<double>(k16[kv_h * head_dim + d]);
      rotated += static_cast<double>(qk[head * head_dim + d]) *
                 static_cast<double>(kk[kv_h * head_dim + d]);
    }
    worst = std::max(worst, std::abs(reference - rotated) /
                                std::max(1.0, std::abs(reference)));
  }
  if (worst > kQkScoreLimit) {
    throw std::runtime_error(
        "the KVarN rotation must preserve the QK score within the declared "
        "limit");
  }

  // The sink holds the rotated K; the tail stays untouched at pos < 128.
  std::vector<std::uint16_t> sink(2 * sink_plane);
  std::vector<std::uint16_t> tail(2 * tail_plane);
  HIP_CHECK(hipMemcpy(sink.data(), d_sink, sink.size() * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tail.data(), d_tail, tail.size() * sizeof(std::uint16_t),
                      hipMemcpyDeviceToHost));
  for (std::uint32_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
    const std::size_t row =
        ((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) * 128 +
         pos) *
        head_dim;
    for (std::uint32_t d = 0; d < head_dim; ++d) {
      const float rotated = kk[kv_h * head_dim + d];
      if (sink[row + d] != __half_as_ushort(__float2half_rn(rotated))) {
        throw std::runtime_error("the KVarN sink must hold the rotated K");
      }
    }
  }
  for (std::uint16_t value : tail) {
    if (value != 0) {
      throw std::runtime_error("the KVarN tail must stay untouched");
    }
  }

  HIP_CHECK(hipFree(d_q16));
  HIP_CHECK(hipFree(d_k16));
  HIP_CHECK(hipFree(d_v16));
  HIP_CHECK(hipFree(d_qk));
  HIP_CHECK(hipFree(d_kk));
  HIP_CHECK(hipFree(d_vk));
  HIP_CHECK(hipFree(d_qw));
  HIP_CHECK(hipFree(d_kw));
  HIP_CHECK(hipFree(d_kc16));
  HIP_CHECK(hipFree(d_vc16));
  HIP_CHECK(hipFree(d_sink));
  HIP_CHECK(hipFree(d_tail));
  HIP_CHECK(hipFree(d_pos));
  std::cout << "KVarN rotation worst relative QK score change " << worst
            << " (limit " << kQkScoreLimit << ")\n";
}

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen QK/RoPE/KV fusion ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }

  TestFusedQKNormRoPEKvWriteEquivalence();
  TestBatchedFusedQKNormRoPEKvWriteEquivalence();
  TestKvarnRotationPreservesQkScore();
  TestNormRoPEWithoutCache();
  std::cout
      << "Qwen QK norm, RoPE, and KV fusion ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout
      << "HIP disabled, skipping Qwen QK norm, RoPE, and KV fusion ops test.\n";
  return 77;
#endif
}
