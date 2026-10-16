#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops/attention.hpp"

namespace gufo::hip {

__global__ void WriteKVCacheKernel(
    const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    std::uint32_t layer_idx, std::uint32_t pos, std::uint32_t max_context,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::uint32_t kv_h = blockIdx.x;
  if (kv_h >= num_kv_heads)
    return;
  const std::size_t tid = threadIdx.x;
  if (tid >= head_dim)
    return;

  const std::size_t kv_offset =
      (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
            max_context +
        pos) *
       head_dim);
  const float key = k[static_cast<std::size_t>(kv_h) * head_dim + tid];
  const float val = v[static_cast<std::size_t>(kv_h) * head_dim + tid];
  if (k_cache != nullptr) {
    k_cache[kv_offset + tid] = key;
  }
  if (v_cache != nullptr) {
    v_cache[kv_offset + tid] = val;
  }

  if (k_cache_f16 != nullptr || v_cache_f16 != nullptr) {
    const std::size_t kv_width =
        static_cast<std::size_t>(num_kv_heads) * head_dim;
    const std::size_t f16_offset =
        ((static_cast<std::size_t>(layer_idx) * max_context + pos) * kv_width) +
        (static_cast<std::size_t>(kv_h) * head_dim) + tid;
    if (k_cache_f16 != nullptr) {
      k_cache_f16[f16_offset] = __float2half_rn(key);
    }
    if (v_cache_f16 != nullptr) {
      v_cache_f16[f16_offset] = __float2half_rn(val);
    }
  }
}

__device__ __forceinline__ float4 LoadHalf4(const __half* source) {
  const auto* pairs = reinterpret_cast<const __half2*>(source);
  const float2 lower = __half22float2(pairs[0]);
  const float2 upper = __half22float2(pairs[1]);
  return make_float4(lower.x, lower.y, upper.x, upper.y);
}

__global__ void AttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx, std::uint32_t pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::uint32_t h = blockIdx.x;
  if (h >= num_heads)
    return;

  const std::uint32_t group_size =
      (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const std::uint32_t kv_h = h / group_size;
  const std::size_t tid = threadIdx.x;

  const float* q_h = q + (static_cast<std::size_t>(h) * head_dim);
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dim));
  const std::size_t seq_len = static_cast<std::size_t>(pos) + 1;

  extern __shared__ float s_scores[];
  __shared__ float s_scratch[256];

  // Step 1: Compute scores for each position p = 0..seq_len-1
  for (std::size_t p = 0; p < seq_len; ++p) {
    const std::size_t p_offset =
        (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
              max_context +
          p) *
         head_dim);
    const float* k_p = k_cache + p_offset;

    float dot = 0.0F;
    for (std::size_t d = tid; d < head_dim; d += blockDim.x) {
      dot += q_h[d] * k_p[d];
    }
    s_scratch[tid] = dot;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (tid < stride) {
        s_scratch[tid] += s_scratch[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      s_scores[p] = s_scratch[0] * scale;
    }
    __syncthreads();
  }

  // Step 2: Softmax across s_scores[0..seq_len-1]
  float local_max = -1e30F;
  for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
    local_max = fmaxf(local_max, s_scores[p]);
  }
  s_scratch[tid] = local_max;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_scratch[tid] = fmaxf(s_scratch[tid], s_scratch[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = s_scratch[0];
  __syncthreads();

  float local_sum = 0.0F;
  for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
    const float e = expf(s_scores[p] - max_score);
    s_scores[p] = e;
    local_sum += e;
  }
  s_scratch[tid] = local_sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      s_scratch[tid] += s_scratch[tid + stride];
    }
    __syncthreads();
  }
  const float inv_sum_exp = 1.0F / (s_scratch[0] + 1e-9F);
  __syncthreads();

  for (std::size_t p = tid; p < seq_len; p += blockDim.x) {
    s_scores[p] *= inv_sum_exp;
  }
  __syncthreads();

  // Step 3: Weighted sum of V
  if (tid < head_dim) {
    float ctx = 0.0F;
    for (std::size_t p = 0; p < seq_len; ++p) {
      const std::size_t p_offset =
          (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_h) *
                max_context +
            p) *
           head_dim);
      const float* v_p = v_cache + p_offset;
      ctx += s_scores[p] * v_p[tid];
    }

    if (gate != nullptr) {
      const float g = gate[static_cast<std::size_t>(h) * head_dim + tid];
      const float sig = 1.0F / (1.0F + expf(-g));
      ctx *= sig;
    }
    out_context[static_cast<std::size_t>(h) * head_dim + tid] = ctx;
  }
}

__global__ void AttentionHalfKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx, std::uint32_t pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim) {
  const std::uint32_t head = blockIdx.x;
  if (head >= num_heads) {
    return;
  }

  const std::uint32_t group_size =
      (num_kv_heads > 0) ? (num_heads / num_kv_heads) : 1;
  const std::uint32_t kv_head = head / group_size;
  const std::size_t tid = threadIdx.x;
  const float* query = q + (static_cast<std::size_t>(head) * head_dim);
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;

  extern __shared__ float scores[];
  __shared__ float scratch[256];

  for (std::size_t position = 0; position < sequence_length; ++position) {
    const std::size_t cache_offset =
        ((static_cast<std::size_t>(layer_idx) * max_context + position) *
         kv_width) +
        (static_cast<std::size_t>(kv_head) * head_dim);
    const __half* key = k_cache + cache_offset;
    float dot = 0.0F;
    for (std::size_t dim = tid; dim < head_dim; dim += blockDim.x) {
      dot += query[dim] * __half2float(key[dim]);
    }
    scratch[tid] = dot;
    __syncthreads();
    for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (tid < stride) {
        scratch[tid] += scratch[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0) {
      scores[position] = scratch[0] * scale;
    }
    __syncthreads();
  }

  float local_max = -INFINITY;
  for (std::size_t position = tid; position < sequence_length;
       position += blockDim.x) {
    local_max = fmaxf(local_max, scores[position]);
  }
  scratch[tid] = local_max;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    }
    __syncthreads();
  }
  const float max_score = scratch[0];

  float local_sum = 0.0F;
  for (std::size_t position = tid; position < sequence_length;
       position += blockDim.x) {
    const float exponential = expf(scores[position] - max_score);
    scores[position] = exponential;
    local_sum += exponential;
  }
  scratch[tid] = local_sum;
  __syncthreads();
  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float inverse_sum = 1.0F / (scratch[0] + 1e-9F);
  for (std::size_t position = tid; position < sequence_length;
       position += blockDim.x) {
    scores[position] *= inverse_sum;
  }
  __syncthreads();

  if (tid < head_dim) {
    float context = 0.0F;
    for (std::size_t position = 0; position < sequence_length; ++position) {
      const std::size_t cache_offset =
          ((static_cast<std::size_t>(layer_idx) * max_context + position) *
           kv_width) +
          (static_cast<std::size_t>(kv_head) * head_dim);
      context += scores[position] * __half2float(v_cache[cache_offset + tid]);
    }
    if (gate != nullptr) {
      const float gate_value =
          gate[(static_cast<std::size_t>(head) * head_dim) + tid];
      context *= 1.0F / (1.0F + expf(-gate_value));
    }
    out_context[(static_cast<std::size_t>(head) * head_dim) + tid] = context;
  }
}

__launch_bounds__(32) __global__ void QwenDecodeOnlineAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx, std::uint32_t pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads) {
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / 32;

  const std::size_t row_offset =
      static_cast<std::size_t>(blockIdx.y) * num_heads * head_dim;
  q += row_offset;
  if (gate != nullptr) {
    gate += row_offset;
  }
  out_context += row_offset;
  pos += blockIdx.y;

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || lane >= 32) {
    return;
  }

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * head_dim));
  const std::size_t cache_head_offset =
      ((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_head) *
       max_context * head_dim);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float4 query_values[vectors_per_lane];
  float4 output_values[vectors_per_lane] = {};
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    query_values[vector] = query[lane + (vector * 32)];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  for (std::size_t position = 0; position < sequence_length; ++position) {
    const auto* key = reinterpret_cast<const float4*>(
        k_cache + cache_head_offset + (position * head_dim));
    float score = 0.0F;
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 query_value = query_values[vector];
      const float4 key_value = key[lane + (vector * 32)];
      score += (query_value.x * key_value.x) + (query_value.y * key_value.y) +
               (query_value.z * key_value.z) + (query_value.w * key_value.w);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      score += __shfl_down(score, offset);
    }
    score = __shfl(score, 0) * scale;

    const float new_max = fmaxf(running_max, score);
    const float previous_scale = expf(running_max - new_max);
    const float value_scale = expf(score - new_max);
    running_sum = (running_sum * previous_scale) + value_scale;
    running_max = new_max;

    const auto* value = reinterpret_cast<const float4*>(
        v_cache + cache_head_offset + (position * head_dim));
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 value_data = value[lane + (vector * 32)];
      float4& output = output_values[vector];
      output.x = (output.x * previous_scale) + (value_data.x * value_scale);
      output.y = (output.y * previous_scale) + (value_data.y * value_scale);
      output.z = (output.z * previous_scale) + (value_data.z * value_scale);
      output.w = (output.w * previous_scale) + (value_data.w * value_scale);
    }
  }

  auto* output = reinterpret_cast<float4*>(
      out_context + (static_cast<std::size_t>(head) * head_dim));
  const auto* gate_values =
      gate == nullptr ? nullptr
                      : reinterpret_cast<const float4*>(
                            gate + (static_cast<std::size_t>(head) * head_dim));
  const float inverse_sum = 1.0F / running_sum;
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    float4 result = output_values[vector];
    result.x *= inverse_sum;
    result.y *= inverse_sum;
    result.z *= inverse_sum;
    result.w *= inverse_sum;
    if (gate_values != nullptr) {
      const float4 gate_value = gate_values[lane + (vector * 32)];
      result.x *= 1.0F / (1.0F + expf(-gate_value.x));
      result.y *= 1.0F / (1.0F + expf(-gate_value.y));
      result.z *= 1.0F / (1.0F + expf(-gate_value.z));
      result.w *= 1.0F / (1.0F + expf(-gate_value.w));
    }
    output[lane + (vector * 32)] = result;
  }
}

__launch_bounds__(32) __global__ void QwenDecodeOnlineAttentionHalfKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const __half* __restrict__ k_cache, const __half* __restrict__ v_cache,
    float* __restrict__ out_context, std::uint32_t layer_idx, std::uint32_t pos,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads) {
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / 32;

  const std::size_t row_offset =
      static_cast<std::size_t>(blockIdx.y) * num_heads * head_dim;
  q += row_offset;
  if (gate != nullptr) {
    gate += row_offset;
  }
  out_context += row_offset;
  pos += blockIdx.y;

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || lane >= 32) {
    return;
  }

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * head_dim));
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t cache_head_offset =
      (static_cast<std::size_t>(layer_idx) * max_context * kv_width) +
      (static_cast<std::size_t>(kv_head) * head_dim);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float4 query_values[vectors_per_lane];
  float4 output_values[vectors_per_lane] = {};
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    query_values[vector] = query[lane + (vector * 32)];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  // Load neighboring keys and values together, then retain the scalar
  // position order for every softmax and output update.
  constexpr std::size_t positions_per_load = 2;
  for (std::size_t position = 0; position < sequence_length;
       position += positions_per_load) {
    float scores[positions_per_load]{};
    float4 values[positions_per_load][vectors_per_lane];
#pragma unroll
    for (std::size_t step = 0; step < positions_per_load; ++step) {
      if (position + step >= sequence_length)
        break;
      const auto offset = cache_head_offset + ((position + step) * kv_width);
      const __half* key = k_cache + offset;
      const __half* value = v_cache + offset;
      float score = 0.0F;
#pragma unroll
      for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
        const float4 query_value = query_values[vector];
        const auto element = (lane + (vector * 32)) * 4;
        const float4 key_value = LoadHalf4(key + element);
        values[step][vector] = LoadHalf4(value + element);
        score += (query_value.x * key_value.x) + (query_value.y * key_value.y) +
                 (query_value.z * key_value.z) + (query_value.w * key_value.w);
      }
      // Lane zero keeps the descending reduction tree. Inside each
      // 16-lane row, DPP avoids the LDS path used by shuffle instructions.
      score += __shfl_down(score, 16);
      score += __builtin_bit_cast(
          float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, score),
                                             0x168, 0xF, 0xF, false));
      score += __builtin_bit_cast(
          float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, score),
                                             0x164, 0xF, 0xF, false));
      score += __builtin_bit_cast(
          float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, score),
                                             0x162, 0xF, 0xF, false));
      score += __builtin_bit_cast(
          float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, score),
                                             0x161, 0xF, 0xF, false));
      scores[step] = __shfl(score, 0) * scale;
    }

#pragma unroll
    for (std::size_t step = 0; step < positions_per_load; ++step) {
      if (position + step >= sequence_length)
        break;
      const float new_max = fmaxf(running_max, scores[step]);
      const float previous_scale = expf(running_max - new_max);
      const float value_scale = expf(scores[step] - new_max);
      running_sum = (running_sum * previous_scale) + value_scale;
      running_max = new_max;

#pragma unroll
      for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
        const float4 value_data = values[step][vector];
        float4& output = output_values[vector];
        // Retain the original half-pair kernel's rounded product and FMA.
        output.x = __fmaf_rn(output.x, previous_scale,
                             __fmul_rn(value_data.x, value_scale));
        output.y = __fmaf_rn(output.y, previous_scale,
                             __fmul_rn(value_data.y, value_scale));
        output.z = __fmaf_rn(value_data.z, value_scale,
                             __fmul_rn(output.z, previous_scale));
        output.w = __fmaf_rn(value_data.w, value_scale,
                             __fmul_rn(output.w, previous_scale));
      }
    }
  }

  auto* output = reinterpret_cast<float4*>(
      out_context + (static_cast<std::size_t>(head) * head_dim));
  const auto* gate_values =
      gate == nullptr ? nullptr
                      : reinterpret_cast<const float4*>(
                            gate + (static_cast<std::size_t>(head) * head_dim));
  const float inverse_sum = 1.0F / running_sum;
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    float4 result = output_values[vector];
    result.x *= inverse_sum;
    result.y *= inverse_sum;
    result.z *= inverse_sum;
    result.w *= inverse_sum;
    if (gate_values != nullptr) {
      const float4 gate_value = gate_values[lane + (vector * 32)];
      result.x *= 1.0F / (1.0F + expf(-gate_value.x));
      result.y *= 1.0F / (1.0F + expf(-gate_value.y));
      result.z *= 1.0F / (1.0F + expf(-gate_value.z));
      result.w *= 1.0F / (1.0F + expf(-gate_value.w));
    }
    output[lane + (vector * 32)] = result;
  }
}

__launch_bounds__(32) __global__ void QwenDecodeSplitKAttentionPartialsKernel(
    const float* __restrict__ q, const float* __restrict__ k_cache,
    const float* __restrict__ v_cache, float* __restrict__ scratch,
    std::uint32_t layer_idx, std::uint32_t pos, std::uint32_t max_context,
    std::uint32_t num_heads, std::uint32_t num_kv_heads,
    std::uint32_t split_count) {
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / 32;

  const std::size_t row = blockIdx.y;
  q += row * num_heads * head_dim;
  scratch += row * num_heads * detail::kSplitKDecodeAttentionMaxSplits *
             (head_dim + 2);
  pos += blockIdx.y;

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t split = blockIdx.z;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || split >= split_count || lane >= 32) {
    return;
  }

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * head_dim));
  const std::size_t cache_head_offset =
      ((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_head) *
       max_context * head_dim);
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const std::size_t partition_length =
      (sequence_length + split_count - 1) / split_count;
  const std::size_t partition_begin =
      static_cast<std::size_t>(split) * partition_length;
  const std::size_t partition_end =
      min(partition_begin + partition_length, sequence_length);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float4 query_values[vectors_per_lane];
  float4 output_values[vectors_per_lane] = {};
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    query_values[vector] = query[lane + (vector * 32)];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0F;
  for (std::size_t position = partition_begin; position < partition_end;
       ++position) {
    const auto* key = reinterpret_cast<const float4*>(
        k_cache + cache_head_offset + (position * head_dim));
    float score = 0.0F;
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 query_value = query_values[vector];
      const float4 key_value = key[lane + (vector * 32)];
      score += (query_value.x * key_value.x) + (query_value.y * key_value.y) +
               (query_value.z * key_value.z) + (query_value.w * key_value.w);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      score += __shfl_down(score, offset);
    }
    score = __shfl(score, 0) * scale;

    const float new_max = fmaxf(running_max, score);
    const float previous_scale = expf(running_max - new_max);
    const float value_scale = expf(score - new_max);
    running_sum = (running_sum * previous_scale) + value_scale;
    running_max = new_max;

    const auto* value = reinterpret_cast<const float4*>(
        v_cache + cache_head_offset + (position * head_dim));
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 value_data = value[lane + (vector * 32)];
      float4& output = output_values[vector];
      output.x = (output.x * previous_scale) + (value_data.x * value_scale);
      output.y = (output.y * previous_scale) + (value_data.y * value_scale);
      output.z = (output.z * previous_scale) + (value_data.z * value_scale);
      output.w = (output.w * previous_scale) + (value_data.w * value_scale);
    }
  }

  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;
  const std::size_t partition_index =
      (static_cast<std::size_t>(head) * max_splits) + split;
  if (lane == 0) {
    scratch[(partition_index * 2) + 0] = running_max;
    scratch[(partition_index * 2) + 1] = running_sum;
  }
  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
  auto* partial_outputs = reinterpret_cast<float4*>(scratch + stats_elements);
  auto* partial_output = partial_outputs + (partition_index * vector_width);
#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    partial_output[lane + (vector * 32)] = output_values[vector];
  }
}

// ---------------------------------------------------------------------------
// decode-blocks (openspec change update-qwen-decode-attention-blocks, design
// D1/D3/D11): FP16 split-K decode and verification over whole 128-token block
// partitions. One block carries one (KV head, row, partition); the six GQA
// query heads of that KV head sit in the WMMA M dimension, each K/V tile is
// read from global memory once and placed in LDS once for all six heads, and
// the partition ranges come from
// detail::SelectDecodeAttentionBlockRange (the whole-tile split design D11
// selected). Q and P are split into an f16 high part and a scaled f16 residual
// (2^11 keeps the residual a normal f16), and every product outside the WMMA
// is folded with an explicit round-to-nearest fmaf because the WMMA's own
// accumulate path rounds the incoming accumulator.
inline constexpr std::uint32_t kDecodeBlocksStageTokens = 16;
inline constexpr std::uint32_t kDecodeBlocksWarps = 8;
inline constexpr std::uint32_t kDecodeBlocksThreads = 32 * kDecodeBlocksWarps;
inline constexpr std::uint32_t kDecodeBlocksDimChunks = 16;
inline constexpr std::uint32_t kDecodeBlocksChunksPerWarp =
    kDecodeBlocksDimChunks / kDecodeBlocksWarps;
inline constexpr std::uint32_t kDecodeBlocksKStride = 256 + 8;
inline constexpr std::uint32_t kDecodeBlocksVStride =
    kDecodeBlocksStageTokens + 8;
inline constexpr std::uint32_t kDecodeBlocksPStride =
    kDecodeBlocksStageTokens + 8;
inline constexpr std::uint32_t kDecodeBlocksGqa = 6;
inline constexpr float kDecodeBlocksResidualScale = 2048.0F;
inline constexpr float kDecodeBlocksResidualScaleInverse =
    1.0F / kDecodeBlocksResidualScale;

using DecodeBlocksHalf16 =
    __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using DecodeBlocksFloat8 =
    __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ DecodeBlocksFloat8 DecodeBlocksWmma(
    DecodeBlocksHalf16 a, DecodeBlocksHalf16 b, DecodeBlocksFloat8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ DecodeBlocksHalf16 DecodeBlocksZeroFrag() {
  union {
    DecodeBlocksHalf16 f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = make_uint4(0u, 0u, 0u, 0u);
  cvt.u[1] = make_uint4(0u, 0u, 0u, 0u);
  return cvt.f;
}

__device__ __forceinline__ DecodeBlocksHalf16 DecodeBlocksLoadFrag(
    const __half* p) {
  union {
    DecodeBlocksHalf16 f;
    uint4 u[2];
  } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}

/// A-side fragment rows are 2i + lane/16, so this lane's eight accumulator
/// elements carry the rescale of query head 2i + lane/16; padding rows stay at
/// 1.0 because their probability rows are zero.
__device__ __forceinline__ DecodeBlocksFloat8 DecodeBlocksRescaleVector(
    const float* row_rescale, std::uint32_t lane) {
  DecodeBlocksFloat8 scale;
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    const std::uint32_t row = (2 * i) + (lane / 16);
    scale[i] = row < kDecodeBlocksGqa ? row_rescale[row] : 1.0F;
  }
  return scale;
}

__launch_bounds__(kDecodeBlocksThreads) __global__
    void QwenDecodeBlockAttentionPartialsKernel(
        const float* __restrict__ q, const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache, float* __restrict__ scratch,
        std::uint32_t layer_idx, std::uint32_t start_pos,
        std::uint32_t max_context, std::uint32_t num_heads,
        std::uint32_t num_kv_heads) {
  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;
  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t row = blockIdx.y;
  const std::uint32_t split = blockIdx.z;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid % 32;
  const std::uint32_t warp = tid / 32;
  if (kv_head >= num_kv_heads || split >= max_splits) {
    return;
  }

  const std::size_t sequence_length =
      static_cast<std::size_t>(start_pos) + row + 1;
  const detail::DecodeAttentionBlockRange range =
      detail::SelectDecodeAttentionBlockRange(sequence_length, split);
  const bool empty = range.empty();

  float* const row_scratch =
      scratch + (static_cast<std::size_t>(row) * num_heads * max_splits *
                 (256 + 2));
  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
  auto* const partial_outputs =
      reinterpret_cast<float*>(row_scratch + stats_elements);
  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t head_begin = kv_head * group_size;

  if (empty) {
    if (tid < group_size) {
      const std::size_t index =
          ((static_cast<std::size_t>(head_begin) + tid) * max_splits) + split;
      row_scratch[index * 2] = -INFINITY;
      row_scratch[index * 2 + 1] = 0.0F;
    }
    for (std::size_t i = tid; i < static_cast<std::size_t>(group_size) * 256;
         i += kDecodeBlocksThreads) {
      const std::size_t head = i / 256;
      const std::size_t dim = i % 256;
      const std::size_t index =
          ((static_cast<std::size_t>(head_begin) + head) * max_splits) + split;
      partial_outputs[(index * 256) + dim] = 0.0F;
    }
    return;
  }

  __shared__ __align__(16) __half k_lds[kDecodeBlocksStageTokens *
                                        kDecodeBlocksKStride];
  __shared__ __align__(16) __half v_lds[256 * kDecodeBlocksVStride];
  __shared__ __align__(16) __half p_hi[kDecodeBlocksGqa *
                                       kDecodeBlocksPStride];
  __shared__ __align__(16) __half p_lo[kDecodeBlocksGqa *
                                       kDecodeBlocksPStride];
  __shared__ float s_lds[kDecodeBlocksWarps][kDecodeBlocksGqa]
                        [kDecodeBlocksStageTokens];
  __shared__ float row_max[kDecodeBlocksGqa];
  __shared__ float row_sum[kDecodeBlocksGqa];
  __shared__ float row_rescale[kDecodeBlocksGqa];

  const std::size_t kv_width = static_cast<std::size_t>(num_kv_heads) * 256;
  const std::size_t head_offset =
      (static_cast<std::size_t>(layer_idx) * max_context * kv_width) +
      (static_cast<std::size_t>(kv_head) * 256);
  const __half* const k_base = k_cache + head_offset;
  const __half* const v_base = v_cache + head_offset;
  const float scale = rsqrtf(256.0F);

  // Q high + residual fragments straight from global memory into registers.
  DecodeBlocksHalf16 q_frag_hi[kDecodeBlocksChunksPerWarp];
  DecodeBlocksHalf16 q_frag_lo[kDecodeBlocksChunksPerWarp];
  {
    const std::uint32_t head_row = lane & 15u;
    const float* const base =
        q + (static_cast<std::size_t>(row) * num_heads * 256) +
        (static_cast<std::size_t>(head_begin + head_row) * 256);
#pragma unroll
    for (std::uint32_t c = 0; c < kDecodeBlocksChunksPerWarp; ++c) {
      const std::uint32_t chunk = (warp * kDecodeBlocksChunksPerWarp) + c;
      if (head_row < kDecodeBlocksGqa) {
        union {
          DecodeBlocksHalf16 f;
          uint4 u[2];
        } high;
        union {
          DecodeBlocksHalf16 f;
          uint4 u[2];
        } low;
        const auto* const source =
            reinterpret_cast<const float*>(base + (chunk * 16));
#pragma unroll
        for (std::uint32_t half = 0; half < 2; ++half) {
          uint4 packed_high;
          uint4 packed_low;
          auto* const high_words = reinterpret_cast<__half*>(&packed_high);
          auto* const low_words = reinterpret_cast<__half*>(&packed_low);
#pragma unroll
          for (std::uint32_t i = 0; i < 8; ++i) {
            const float value = source[(half * 8) + i] * scale;
            const __half high_value = __float2half_rn(value);
            high_words[i] = high_value;
            low_words[i] = __float2half_rn(
                (value - __half2float(high_value)) * kDecodeBlocksResidualScale);
          }
          high.u[half] = packed_high;
          low.u[half] = packed_low;
        }
        q_frag_hi[c] = high.f;
        q_frag_lo[c] = low.f;
      } else {
        q_frag_hi[c] = DecodeBlocksZeroFrag();
        q_frag_lo[c] = DecodeBlocksZeroFrag();
      }
    }
  }
  if (tid < kDecodeBlocksGqa) {
    row_max[tid] = -INFINITY;
    row_sum[tid] = 0.0F;
    row_rescale[tid] = 0.0F;
  }
  __syncthreads();

  DecodeBlocksFloat8 output_acc[kDecodeBlocksChunksPerWarp];
#pragma unroll
  for (std::uint32_t c = 0; c < kDecodeBlocksChunksPerWarp; ++c) {
    output_acc[c] = DecodeBlocksFloat8{};
  }

  // Software pipeline: one stage's K/V sits in registers while the previous
  // stage is computed, so the global loads stay behind the compute.
  uint4 k_regs[2];
  std::uint32_t v_regs[8];
  const auto load_stage = [&](std::size_t begin, std::uint32_t count,
                              uint4 (&k_out)[2], std::uint32_t (&v_out)[8]) {
    {
      const std::uint32_t token = tid / 16;
      const std::uint32_t dim = (tid % 16) * 16;
      if (token < count) {
        const auto* const source = reinterpret_cast<const uint4*>(
            k_base + ((begin + token) * kv_width) + dim);
        k_out[0] = source[0];
        k_out[1] = source[1];
      } else {
        k_out[0] = make_uint4(0u, 0u, 0u, 0u);
        k_out[1] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    {
      const std::uint32_t dim = 2 * (tid / 2);
      const std::uint32_t token_begin = (tid % 2) * 8;
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t token = token_begin + i;
        v_out[i] = token < count
                       ? *reinterpret_cast<const std::uint32_t*>(
                             v_base + ((begin + token) * kv_width) + dim)
                       : 0U;
      }
    }
  };
  const auto store_stage = [&](uint4 (&k_in)[2], std::uint32_t (&v_in)[8]) {
    {
      const std::uint32_t token = tid / 16;
      const std::uint32_t dim = (tid % 16) * 16;
      auto* const target =
          reinterpret_cast<uint4*>(&k_lds[(token * kDecodeBlocksKStride) + dim]);
      target[0] = k_in[0];
      target[1] = k_in[1];
    }
    {
      const std::uint32_t dim = 2 * (tid / 2);
      const std::uint32_t token_begin = (tid % 2) * 8;
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        v_lds[((dim + 0) * kDecodeBlocksVStride) + token_begin + i] =
            __ushort_as_half(static_cast<unsigned short>(v_in[i] & 0xFFFFU));
        v_lds[((dim + 1) * kDecodeBlocksVStride) + token_begin + i] =
            __ushort_as_half(static_cast<unsigned short>(v_in[i] >> 16));
      }
    }
  };

  load_stage(range.begin,
             static_cast<std::uint32_t>(std::min<std::size_t>(
                 kDecodeBlocksStageTokens, range.end - range.begin)),
             k_regs, v_regs);
  for (std::size_t stage_begin = range.begin; stage_begin < range.end;
       stage_begin += kDecodeBlocksStageTokens) {
    const std::uint32_t stage_count = static_cast<std::uint32_t>(
        std::min<std::size_t>(kDecodeBlocksStageTokens,
                              range.end - stage_begin));
    __syncthreads();
    store_stage(k_regs, v_regs);
    __syncthreads();
    const std::size_t next_begin = stage_begin + kDecodeBlocksStageTokens;
    if (next_begin < range.end) {
      load_stage(next_begin,
                 static_cast<std::uint32_t>(std::min<std::size_t>(
                     kDecodeBlocksStageTokens, range.end - next_begin)),
                 k_regs, v_regs);
    }

    // S = Q K^T over this warp's 16-dim chunks.
    DecodeBlocksFloat8 scores = DecodeBlocksFloat8{};
#pragma unroll
    for (std::uint32_t c = 0; c < kDecodeBlocksChunksPerWarp; ++c) {
      const std::uint32_t chunk = (warp * kDecodeBlocksChunksPerWarp) + c;
      const DecodeBlocksHalf16 key_frag = DecodeBlocksLoadFrag(
          &k_lds[((lane & 15u) * kDecodeBlocksKStride) + (chunk * 16)]);
      scores = DecodeBlocksWmma(q_frag_hi[c], key_frag, scores);
      const DecodeBlocksFloat8 residual =
          DecodeBlocksWmma(q_frag_lo[c], key_frag, DecodeBlocksFloat8{});
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        scores[i] = fmaf(residual[i], kDecodeBlocksResidualScaleInverse,
                         scores[i]);
      }
    }
    {
      const std::uint32_t head_row = lane / 16;
      const std::uint32_t token = lane & 15u;
#pragma unroll
      for (std::uint32_t i = 0; i < kDecodeBlocksGqa / 2; ++i) {
        s_lds[warp][(2 * i) + head_row][token] = scores[i];
      }
    }
    __syncthreads();

    // Online softmax: heads 2w and 2w+1 share a warp; the 16 lanes of a half
    // wave reduce the staged tokens.
    if (warp < kDecodeBlocksGqa / 2) {
      const std::uint32_t head_row = (2 * warp) + (lane / 16);
      const std::uint32_t token = lane & 15u;
      float score = (token < stage_count) ? 0.0F : -INFINITY;
#pragma unroll
      for (std::uint32_t slot = 0; slot < kDecodeBlocksWarps; ++slot) {
        score += s_lds[slot][head_row][token];
      }
      float tile_max = score;
#pragma unroll
      for (int offset = 8; offset > 0; offset /= 2) {
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, offset));
      }
      const float previous_max = row_max[head_row];
      const float stage_max = fmaxf(previous_max, tile_max);
      const float rescale =
          isfinite(previous_max) ? expf(previous_max - stage_max) : 0.0F;
      float weight = 0.0F;
      if (token < stage_count) {
        weight = expf(score - stage_max);
      }
      float part_sum = weight;
#pragma unroll
      for (int offset = 8; offset > 0; offset /= 2) {
        part_sum = fmaf(1.0F, part_sum, __shfl_xor(part_sum, offset));
      }
      const float previous_sum = row_sum[head_row];
      const float carried = previous_sum * rescale;
      const float sum_new = carried + part_sum;
      const float inverse_sum = (sum_new > 0.0F) ? 1.0F / sum_new : 0.0F;
      const float scaled_weight = weight * inverse_sum;
      const __half weight_high = __float2half_rn(scaled_weight);
      p_hi[(head_row * kDecodeBlocksPStride) + token] = weight_high;
      p_lo[(head_row * kDecodeBlocksPStride) + token] = __float2half_rn(
          (scaled_weight - __half2float(weight_high)) *
          kDecodeBlocksResidualScale);
      if ((lane & 15u) == 0) {
        row_rescale[head_row] = carried * inverse_sum;
        row_max[head_row] = stage_max;
        row_sum[head_row] = sum_new;
      }
    }
    __syncthreads();

    // O += P V, with the running rescale fused into the round-to-nearest fold.
    const std::uint32_t probability_row = lane & 15u;
    DecodeBlocksHalf16 probabilities = DecodeBlocksZeroFrag();
    DecodeBlocksHalf16 residual_probabilities = DecodeBlocksZeroFrag();
    if (probability_row < kDecodeBlocksGqa) {
      probabilities =
          DecodeBlocksLoadFrag(&p_hi[(probability_row * kDecodeBlocksPStride)]);
      residual_probabilities =
          DecodeBlocksLoadFrag(&p_lo[(probability_row * kDecodeBlocksPStride)]);
    }
    const DecodeBlocksFloat8 rescale_vector =
        DecodeBlocksRescaleVector(row_rescale, lane);
#pragma unroll
    for (std::uint32_t c = 0; c < kDecodeBlocksChunksPerWarp; ++c) {
      const std::uint32_t chunk = (warp * kDecodeBlocksChunksPerWarp) + c;
      const DecodeBlocksHalf16 value_frag = DecodeBlocksLoadFrag(
          &v_lds[((chunk * 16) + probability_row) * kDecodeBlocksVStride]);
      const DecodeBlocksFloat8 product =
          DecodeBlocksWmma(probabilities, value_frag, DecodeBlocksFloat8{});
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        output_acc[c][i] = fmaf(output_acc[c][i], rescale_vector[i],
                                product[i]);
      }
      const DecodeBlocksFloat8 scaled =
          DecodeBlocksWmma(residual_probabilities, value_frag,
                           DecodeBlocksFloat8{});
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        output_acc[c][i] = fmaf(scaled[i], kDecodeBlocksResidualScaleInverse,
                                output_acc[c][i]);
      }
    }
  }

  __syncthreads();
  if ((warp < kDecodeBlocksGqa / 2) && ((lane & 15u) == 0)) {
    const std::uint32_t head = (2 * warp) + (lane / 16);
    const std::size_t index =
        (static_cast<std::size_t>(head_begin + head) * max_splits) + split;
    row_scratch[index * 2] = row_max[head];
    row_scratch[index * 2 + 1] = row_sum[head];
  }
#pragma unroll
  for (std::uint32_t c = 0; c < kDecodeBlocksChunksPerWarp; ++c) {
    const std::uint32_t chunk = (warp * kDecodeBlocksChunksPerWarp) + c;
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      const std::uint32_t head = (2 * i) + (lane / 16);
      if (head < kDecodeBlocksGqa) {
        const std::size_t index =
            (static_cast<std::size_t>(head_begin + head) * max_splits) + split;
        const std::uint32_t dim = (chunk * 16) + (lane & 15u);
        partial_outputs[(index * 256) + dim] =
            output_acc[c][i] * row_sum[head];
      }
    }
  }
}

static void LaunchDecodeBlockAttentionPartials(
    const float* q, const __half* k_cache, const __half* v_cache,
    float* scratch, std::uint32_t layer_idx, std::uint32_t start_pos,
    unsigned rows, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, hipStream_t stream) {
  QwenDecodeBlockAttentionPartialsKernel<<<
      dim3(num_kv_heads, rows, detail::kSplitKDecodeAttentionMaxSplits),
      kDecodeBlocksThreads, 0, stream>>>(q, k_cache, v_cache, scratch,
                                        layer_idx, start_pos, max_context,
                                        num_heads, num_kv_heads);
}

__launch_bounds__(32) __global__ void QwenDecodeSplitKAttentionReduceKernel(
    const float* __restrict__ gate, const float* __restrict__ scratch,
    float* __restrict__ out_context, std::uint32_t num_heads,
    std::uint32_t split_count) {
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / 32;
  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;

  const std::size_t row = blockIdx.y;
  if (gate != nullptr)
    gate += row * num_heads * head_dim;
  out_context += row * num_heads * head_dim;
  scratch += row * num_heads * max_splits * (head_dim + 2);

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || lane >= 32) {
    return;
  }

  const std::size_t head_partition_offset =
      static_cast<std::size_t>(head) * max_splits;
  float global_max = -INFINITY;
  for (std::uint32_t split = 0; split < split_count; ++split) {
    global_max =
        fmaxf(global_max, scratch[((head_partition_offset + split) * 2) + 0]);
  }

  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
  const auto* partial_outputs =
      reinterpret_cast<const float4*>(scratch + stats_elements);
  auto* output = reinterpret_cast<float4*>(
      out_context + (static_cast<std::size_t>(head) * head_dim));
  const auto* gate_values =
      gate == nullptr ? nullptr
                      : reinterpret_cast<const float4*>(
                            gate + (static_cast<std::size_t>(head) * head_dim));
  // Each output vector and the denominator visit partitions in the same
  // order. Share their scale instead of recomputing its exponential.
  float global_sum = 0.0F;
  float4 results[vectors_per_lane]{};
  for (std::uint32_t split = 0; split < split_count; ++split) {
    const std::size_t partition_index = head_partition_offset + split;
    const float partition_scale =
        expf(scratch[partition_index * 2] - global_max);
    global_sum += scratch[partition_index * 2 + 1] * partition_scale;
#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 partial = partial_outputs[(partition_index * vector_width) +
                                             lane + (vector * 32)];
      auto& result = results[vector];
      result.x += partial.x * partition_scale;
      result.y += partial.y * partition_scale;
      result.z += partial.z * partition_scale;
      result.w += partial.w * partition_scale;
    }
  }
  const float inverse_sum = 1.0F / global_sum;

#pragma unroll
  for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
    auto result = results[vector];
    result.x *= inverse_sum;
    result.y *= inverse_sum;
    result.z *= inverse_sum;
    result.w *= inverse_sum;
    if (gate_values != nullptr) {
      const float4 gate_value = gate_values[lane + (vector * 32)];
      result.x *= 1.0F / (1.0F + expf(-gate_value.x));
      result.y *= 1.0F / (1.0F + expf(-gate_value.y));
      result.z *= 1.0F / (1.0F + expf(-gate_value.z));
      result.w *= 1.0F / (1.0F + expf(-gate_value.w));
    }
    output[lane + (vector * 32)] = result;
  }
}

void LaunchAttention(const float* q, const float* k, const float* v,
                     const float* gate, float* k_cache, float* v_cache,
                     void* k_cache_f16, void* v_cache_f16, float* out_context,
                     std::uint32_t layer_idx, std::uint32_t pos,
                     std::uint32_t max_context, std::uint32_t num_heads,
                     std::uint32_t num_kv_heads, std::uint32_t head_dim,
                     hipStream_t stream, float* split_k_scratch,
                     bool skip_kv_write) {
  RequireAttentionKvSource(k_cache, v_cache, k_cache_f16, v_cache_f16, false);
  // 1. Write K and V into every non-null selected KV representation across all
  // num_kv_heads. May be skipped when a fused kernel already wrote the cache
  // (opt-c010-qk-rope-kv).
  if (!skip_kv_write) {
    WriteKVCacheKernel<<<num_kv_heads, 256, 0, stream>>>(
        k, v, k_cache, v_cache, static_cast<__half*>(k_cache_f16),
        static_cast<__half*>(v_cache_f16), layer_idx, pos, max_context,
        num_kv_heads, head_dim);
  }

  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const bool split_k_supported = detail::IsSplitKDecodeAttentionSupported(
      sequence_length, num_heads, num_kv_heads, head_dim);
  if (split_k_supported && split_k_scratch != nullptr) {
    const std::uint32_t split_count =
        detail::SelectDecodeAttentionSplitCount(sequence_length);
    const dim3 partial_grid(num_heads, 1, split_count);
    if (k_cache != nullptr && v_cache != nullptr) {
      detail::EmitAttentionDispatch("decode_split_k_fp32", "");
      QwenDecodeSplitKAttentionPartialsKernel<<<partial_grid, 32, 0, stream>>>(
          q, k_cache, v_cache, split_k_scratch, layer_idx, pos, max_context,
          num_heads, num_kv_heads, split_count);
    } else {
      // decode-blocks: the FP16 path runs the (KV head, row, partition) block
      // kernel; the reduce above and below it is unchanged.
      detail::EmitAttentionDispatch("decode_split_k_fp16", "");
      LaunchDecodeBlockAttentionPartials(
          q, static_cast<const __half*>(k_cache_f16),
          static_cast<const __half*>(v_cache_f16), split_k_scratch, layer_idx,
          pos, 1, max_context, num_heads, num_kv_heads, stream);
    }
    QwenDecodeSplitKAttentionReduceKernel<<<num_heads, 32, 0, stream>>>(
        gate, split_k_scratch, out_context, num_heads, split_count);
  } else if (head_dim == 256 && num_kv_heads > 0 &&
             (num_heads % num_kv_heads) == 0) {
    if (k_cache != nullptr && v_cache != nullptr) {
      detail::EmitAttentionDispatch(
          "decode_online_fp32", split_k_supported
                                    ? "decode_split_k_fp32: missing_scratch"
                                    : "decode_split_k_fp32: below_threshold");
      QwenDecodeOnlineAttentionKernel<<<num_heads, 32, 0, stream>>>(
          q, gate, k_cache, v_cache, out_context, layer_idx, pos, max_context,
          num_heads, num_kv_heads);
    } else {
      detail::EmitAttentionDispatch(
          "decode_online_fp16", split_k_supported
                                    ? "decode_split_k_fp16: missing_scratch"
                                    : "decode_split_k_fp16: below_threshold");
      QwenDecodeOnlineAttentionHalfKernel<<<num_heads, 32, 0, stream>>>(
          q, gate, static_cast<const __half*>(k_cache_f16),
          static_cast<const __half*>(v_cache_f16), out_context, layer_idx, pos,
          max_context, num_heads, num_kv_heads);
    }
  } else {
    const std::size_t seq_len = static_cast<std::size_t>(pos) + 1;
    const std::size_t shared_scores_bytes = seq_len * sizeof(float);
    if (k_cache != nullptr && v_cache != nullptr) {
      detail::EmitAttentionDispatch(
          "decode_baseline_fp32",
          "decode_split_k_fp32: unsupported_shape; decode_online_fp32: "
          "unsupported_shape");
      AttentionKernel<<<num_heads, 256, shared_scores_bytes, stream>>>(
          q, gate, k_cache, v_cache, out_context, layer_idx, pos, max_context,
          num_heads, num_kv_heads, head_dim);
    } else {
      detail::EmitAttentionDispatch(
          "decode_baseline_fp16",
          "decode_split_k_fp16: unsupported_shape; decode_online_fp16: "
          "unsupported_shape");
      AttentionHalfKernel<<<num_heads, 256, shared_scores_bytes, stream>>>(
          q, gate, static_cast<const __half*>(k_cache_f16),
          static_cast<const __half*>(v_cache_f16), out_context, layer_idx, pos,
          max_context, num_heads, num_kv_heads, head_dim);
    }
  }
}

void LaunchCausalDecodeAttention(
    const float* q, const float* gate, float* k_cache, float* v_cache,
    void* k_cache_f16, void* v_cache_f16, float* out_context,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, hipStream_t stream,
    std::span<float> scratch, KvarnReadParams kvarn) {
  RequireAttentionKvSource(k_cache, v_cache, k_cache_f16, v_cache_f16,
                           kvarn.enabled());
  const std::size_t row_scratch =
      detail::DecodeAttentionScratchElements(num_heads, head_dim);
  float* const split_k_scratch =
      scratch.size() >= row_scratch ? scratch.data() : nullptr;
  // KVarN verification rows read the sink, sealed tiles and tail ring through
  // the shared LDS staging kernel, keeping every row bit-identical to the AR
  // decode at the same position (design D7, T-05).
  if (kvarn.enabled() && head_dim == 256 && batch_size > 0) {
    if (split_k_scratch != nullptr &&
        scratch.size() / batch_size >= row_scratch) {
      LaunchKvarnCausalDecodeAttention(
          q, gate, out_context, layer_idx, start_pos, batch_size, num_heads,
          num_kv_heads, kvarn, split_k_scratch, stream);
      return;
    }
    const std::size_t width = static_cast<std::size_t>(num_heads) * head_dim;
    for (std::size_t row = 0; row < batch_size; ++row) {
      LaunchKvarnDecodeAttention(
          q + row * width, gate == nullptr ? nullptr : gate + row * width,
          out_context + row * width, layer_idx,
          start_pos + static_cast<std::uint32_t>(row), nullptr, 1, num_heads,
          num_kv_heads, kvarn, split_k_scratch, stream);
    }
    return;
  }
  if (batch_size > 1 && scratch.size() / batch_size >= row_scratch &&
      split_k_scratch != nullptr &&
      detail::IsSplitKDecodeAttentionSupported(
          static_cast<std::size_t>(start_pos) + 1, num_heads, num_kv_heads,
          head_dim)) {
    const auto splits = detail::SelectDecodeAttentionSplitCount(
        static_cast<std::size_t>(start_pos) + 1);
    // Keep a partition's KV resident while visiting every query row. With the
    // row as the slow grid dimension, each row rereads the entire long cache.
    const dim3 grid(num_heads, static_cast<unsigned>(batch_size), splits);
    if (k_cache != nullptr && v_cache != nullptr) {
      QwenDecodeSplitKAttentionPartialsKernel<<<grid, 32, 0, stream>>>(
          q, k_cache, v_cache, split_k_scratch, layer_idx, start_pos,
          max_context, num_heads, num_kv_heads, splits);
    } else {
      // decode-blocks: every row of the verification launch uses the same
      // block partitions, so each row keeps the scalar decode bits.
      LaunchDecodeBlockAttentionPartials(
          q, static_cast<const __half*>(k_cache_f16),
          static_cast<const __half*>(v_cache_f16), split_k_scratch, layer_idx,
          start_pos, static_cast<unsigned>(batch_size), max_context, num_heads,
          num_kv_heads, stream);
    }
    QwenDecodeSplitKAttentionReduceKernel<<<
        dim3(num_heads, static_cast<unsigned>(batch_size)), 32, 0, stream>>>(
        gate, split_k_scratch, out_context, num_heads, splits);
    return;
  }
  // Each query keeps the scalar decode reduction and its own causal endpoint.
  // Crossing the split-K threshold retains the per-row route and scratch.
  if (batch_size > 1 && head_dim == 256 && num_kv_heads > 0 &&
      num_heads % num_kv_heads == 0 &&
      static_cast<std::size_t>(start_pos) + batch_size <
          detail::kSplitKDecodeAttentionMinContext) {
    const dim3 grid(num_heads, static_cast<unsigned>(batch_size));
    if (k_cache != nullptr && v_cache != nullptr) {
      QwenDecodeOnlineAttentionKernel<<<grid, 32, 0, stream>>>(
          q, gate, k_cache, v_cache, out_context, layer_idx, start_pos,
          max_context, num_heads, num_kv_heads);
    } else {
      QwenDecodeOnlineAttentionHalfKernel<<<grid, 32, 0, stream>>>(
          q, gate, static_cast<const __half*>(k_cache_f16),
          static_cast<const __half*>(v_cache_f16), out_context, layer_idx,
          start_pos, max_context, num_heads, num_kv_heads);
    }
    return;
  }
  const std::size_t width = static_cast<std::size_t>(num_heads) * head_dim;
  for (std::size_t row = 0; row < batch_size; ++row) {
    LaunchAttention(
        q + row * width, nullptr, nullptr,
        gate == nullptr ? nullptr : gate + row * width, k_cache, v_cache,
        k_cache_f16, v_cache_f16, out_context + row * width, layer_idx,
        start_pos + static_cast<std::uint32_t>(row), max_context, num_heads,
        num_kv_heads, head_dim, stream, split_k_scratch, true);
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
