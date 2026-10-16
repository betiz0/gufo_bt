template<bool ShareKv = false, unsigned Lanes = 32>
__launch_bounds__(ShareKv ? 6 * Lanes : 32) __global__
    void QwenDecodeSplitKAttentionHalfPartialsKernel(
        const float* __restrict__ q, const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache, float* __restrict__ scratch,
        std::uint32_t layer_idx, std::uint32_t pos, std::uint32_t max_context,
        std::uint32_t num_heads, std::uint32_t num_kv_heads,
        std::uint32_t split_count) {
  static_assert(Lanes == 32 || (ShareKv && Lanes == 16));
  constexpr std::uint32_t head_dim = 256;
  constexpr std::uint32_t vector_width = head_dim / 4;
  constexpr std::uint32_t vectors_per_lane = vector_width / Lanes;

  const std::size_t row = blockIdx.y;
  q += row * num_heads * head_dim;
  scratch += row * num_heads * detail::kSplitKDecodeAttentionMaxSplits *
             (head_dim + 2);
  pos += blockIdx.y;

  constexpr std::uint32_t heads_per_block = ShareKv ? 6 : 1;
  const std::uint32_t head = blockIdx.x * heads_per_block + threadIdx.x / Lanes;
  const std::uint32_t split = blockIdx.z;
  const std::uint32_t lane = threadIdx.x % Lanes;
  if (head >= num_heads || split >= split_count) {
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
    query_values[vector] =
        query[lane + (vector / 2) * Lanes + (vector % 2) * 32];
  }

  float running_max = -INFINITY;
  float running_sum = 0.0F;
  const auto accumulate = [&](const __half* key, const __half* value) {
    float score = 0.0F;
    if constexpr (Lanes == 32) {
#pragma unroll
      for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
        const float4 query_value = query_values[vector];
        const float4 key_value = LoadHalf4(key + ((lane + (vector * 32)) * 4));
        score += (query_value.x * key_value.x) + (query_value.y * key_value.y) +
                 (query_value.z * key_value.z) + (query_value.w * key_value.w);
      }
    } else {
      // Two query heads share a hardware wave. Keep both original lane
      // partials independent until the original offset-16 reduction stage.
      float partial[2] = {};
#pragma unroll
      for (unsigned half = 0; half < 2; ++half) {
#pragma unroll
        for (unsigned vector = 0; vector < 2; ++vector) {
          const float4 query_value = query_values[half * 2 + vector];
          const float4 key_value =
              LoadHalf4(key + ((lane + half * 16 + vector * 32) * 4));
          partial[half] +=
              (query_value.x * key_value.x) + (query_value.y * key_value.y) +
              (query_value.z * key_value.z) + (query_value.w * key_value.w);
        }
      }
      score = partial[0] + partial[1];
    }
#pragma unroll
    for (int offset = Lanes / 2; offset > 0; offset /= 2) {
      score += __shfl_down(score, offset, Lanes);
    }
    score = __shfl(score, 0, Lanes) * scale;

    const float new_max = fmaxf(running_max, score);
    const float previous_scale = expf(running_max - new_max);
    const float value_scale = expf(score - new_max);
    running_sum = (running_sum * previous_scale) + value_scale;
    running_max = new_max;

#pragma unroll
    for (std::uint32_t vector = 0; vector < vectors_per_lane; ++vector) {
      const float4 value_data = LoadHalf4(
          value + ((lane + (vector / 2) * Lanes + (vector % 2) * 32) * 4));
      float4& output = output_values[vector];
      // Preserve the scalar kernel's rounded product and FMA operands.
      output.x =
          fmaf(output.x, previous_scale, __fmul_rn(value_data.x, value_scale));
      output.y =
          fmaf(output.y, previous_scale, __fmul_rn(value_data.y, value_scale));
      output.z =
          fmaf(value_data.z, value_scale, __fmul_rn(output.z, previous_scale));
      output.w =
          fmaf(value_data.w, value_scale, __fmul_rn(output.w, previous_scale));
    }
  };
  if constexpr (ShareKv) {
    // Six query heads share each KV head. Stage the encoded KV once without
    // changing any head's score, softmax or value accumulation order.
    constexpr unsigned tile = 4;
    __shared__ __align__(16) __half keys[tile][head_dim];
    __shared__ __align__(16) __half values[tile][head_dim];
    for (std::size_t begin = partition_begin; begin < partition_end;
         begin += tile) {
      const unsigned count =
          static_cast<unsigned>(min(std::size_t{tile}, partition_end - begin));
      for (unsigned vector = threadIdx.x; vector < count * 32;
           vector += heads_per_block * Lanes) {
        const unsigned token = vector / 32;
        const unsigned dim = (vector % 32) * 8;
        const std::size_t offset =
            cache_head_offset + (begin + token) * kv_width + dim;
        *reinterpret_cast<uint4*>(keys[token] + dim) =
            *reinterpret_cast<const uint4*>(k_cache + offset);
        *reinterpret_cast<uint4*>(values[token] + dim) =
            *reinterpret_cast<const uint4*>(v_cache + offset);
      }
      __syncthreads();
#pragma unroll 1
      for (unsigned token = 0; token < count; ++token) {
        accumulate(keys[token], values[token]);
      }
      __syncthreads();
    }
  } else {
    for (std::size_t position = partition_begin; position < partition_end;
         ++position) {
      const std::size_t offset = cache_head_offset + position * kv_width;
      accumulate(k_cache + offset, v_cache + offset);
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
    partial_output[lane + (vector / 2) * Lanes + (vector % 2) * 32] =
        output_values[vector];
  }
}

static void LaunchSplitKAttentionHalfPartials(
    const float* q, const __half* k_cache, const __half* v_cache,
    float* scratch, std::uint32_t layer_idx, std::uint32_t start_pos,
    unsigned rows, std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t splits, hipStream_t stream) {
  // Share each KV tile across its six query heads in both scalar decoding
  // and verification. Rows retain independent causal endpoints and scratch.
  if (num_heads == 6 * num_kv_heads &&
      static_cast<std::size_t>(start_pos) + 1 >= 512) {
    if (rows > 1) {
      QwenDecodeSplitKAttentionHalfPartialsKernel<true, 16>
          <<<dim3(num_kv_heads, rows, splits), 96, 0, stream>>>(
              q, k_cache, v_cache, scratch, layer_idx, start_pos, max_context,
              num_heads, num_kv_heads, splits);
    } else {
      QwenDecodeSplitKAttentionHalfPartialsKernel<true>
          <<<dim3(num_kv_heads, rows, splits), 192, 0, stream>>>(
              q, k_cache, v_cache, scratch, layer_idx, start_pos, max_context,
              num_heads, num_kv_heads, splits);
    }
  } else {
    QwenDecodeSplitKAttentionHalfPartialsKernel<>
        <<<dim3(num_heads, rows, splits), 32, 0, stream>>>(
            q, k_cache, v_cache, scratch, layer_idx, start_pos, max_context,
            num_heads, num_kv_heads, splits);
  }
}
