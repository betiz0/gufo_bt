// Verbatim copy of the KVarN decode *read* kernels from the baseline 801abd3.
//
// The baseline file `src/models/qwen/hip/kernels/kvarn_read.hip` is included by
// this bench's translation unit as production source, so the copy below only
// renames the top-level identifiers (prefix `Base`) and leaves every body, every
// expression and every launch argument byte-identical to the baseline. It gives
// the KVarN bench modes a control to compare candidate block structures against.
//
// Extracted with:
//   git show 801abd3:src/models/qwen/hip/kernels/kvarn_read.hip \
//     | sed -n '615,700p;702,794p;796,1104p;1108,1168p'
//
// Region map (baseline line numbers):
//   615-700   QwenKvarnSplitKPartialsKernel (+ doc comment)
//   702-794   QwenKvarnSplitKReduceKernel (+ doc comment)
//   796-1104  kCausalMaxRows/kCausalChunk/kCausalThreads and
//             QwenKvarnCausalDecodeKernel (+ doc comment)
//   1108-1168 LaunchKvarnCausalDecodeAttention and LaunchKvarnDecodeAttention
//
// Renames (top level only):
//   QwenKvarnSplitKPartialsKernel  -> BaseQwenKvarnSplitKPartialsKernel
//   QwenKvarnSplitKReduceKernel    -> BaseQwenKvarnSplitKReduceKernel
//   QwenKvarnCausalDecodeKernel    -> BaseQwenKvarnCausalDecodeKernel
//   LaunchKvarnCausalDecodeAttention -> BaseLaunchKvarnCausalDecodeAttention
//   LaunchKvarnDecodeAttention     -> BaseLaunchKvarnDecodeAttention
//   kCausalMaxRows                 -> BaseKvarnCausalMaxRows
//   kCausalChunk                   -> BaseKvarnCausalChunk
//   kCausalThreads                 -> BaseKvarnCausalThreads
//
// Verified with (report from the machine that produced this file: identical,
// `cmp` exit status 0):
//   git show 801abd3:src/models/qwen/hip/kernels/kvarn_read.hip \
//     | sed -n '615,700p;702,794p;796,1104p;1108,1168p' > /tmp/kvarn_base_raw.txt
//   tail -n +39 tools/bench/detail/qwen_kvarn_read_base_801abd3.inc \
//     | sed -E 's/\bBaseQwenKvarn/QwenKvarn/g;
//               s/\bBaseLaunchKvarn/LaunchKvarn/g;
//               s/\bBaseKvarnCausal/kCausal/g' \
//     | cmp - /tmp/kvarn_base_raw.txt
/// One split of the KVarN decode. The split's key range is snapped to whole
/// 128-token tiles, so a split never straddles a sealed tile and the last split
/// absorbs the F16 tail rows (design D8). Partial max/sum and partial outputs
/// use the same scratch layout as the FP16 split-K decode, so the split count
/// (<= 32) and `DecodeAttentionScratchElements` stay valid. The walk is the one
/// `KvarnWalkPartition`, shared with the online kernel (design D7/T-05).
__launch_bounds__(kLanes) __global__ void BaseQwenKvarnSplitKPartialsKernel(
    const float* __restrict__ q, float* __restrict__ scratch,
    std::uint32_t layer_idx, std::uint32_t start_pos,
    const std::uint32_t* d_pos, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t split_count,
    KvarnReadParams kvarn) {
  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;
  const std::size_t row = blockIdx.y;
  q += row * num_heads * kHeadDim;
  scratch += row * num_heads * max_splits * (kHeadDim + 2);
  const std::uint32_t pos =
      (d_pos != nullptr ? *d_pos : start_pos) + static_cast<std::uint32_t>(row);

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t split = blockIdx.z;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || split >= split_count || lane >= kLanes) {
    return;
  }

  const std::size_t partition_index =
      static_cast<std::size_t>(head) * max_splits + split;
  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
  auto* partial_outputs = reinterpret_cast<float4*>(scratch + stats_elements);
  auto* partial_output = partial_outputs + partition_index * kVectorWidth;

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  const auto* query = reinterpret_cast<const float4*>(
      q + (static_cast<std::size_t>(head) * kHeadDim));
  const float scale = rsqrtf(static_cast<float>(kHeadDim));

  float4 query_values[kVectorsPerLane];
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
    query_values[vector] = query[lane + (vector * kLanes)];
  }

  const std::uint32_t sequence_length = pos + 1;
  // Whole-tile split: boundaries land on 128-token tiles and the last split
  // absorbs the remainder (design D8).
  const std::uint32_t blocks =
      (sequence_length + kTileTokens - 1) / kTileTokens;
  const std::uint32_t begin_block = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(split) * blocks) / split_count);
  const std::uint32_t end_block = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(split + 1) * blocks) / split_count);
  const std::uint32_t partition_begin = begin_block * kTileTokens;
  const std::uint32_t partition_end =
      min(end_block * kTileTokens, sequence_length);

  KvarnOnlineState state;
  state.running_max = -INFINITY;
  state.running_sum = 0.0F;
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      state.output[vector][k] = 0.0F;
    }
  }

  KvarnWalkPartition(kvarn, layer_idx, kv_head, query_values, lane, scale, pos,
                     partition_begin, partition_end, &state);

  if (lane == 0) {
    scratch[partition_index * 2] = state.running_max;
    scratch[partition_index * 2 + 1] = state.running_sum;
  }
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
    float4 result;
    result.x = state.output[vector][0];
    result.y = state.output[vector][1];
    result.z = state.output[vector][2];
    result.w = state.output[vector][3];
    partial_output[lane + (vector * kLanes)] = result;
  }
}
/// Merges the KVarN split-K partials into the context. Whole-tile rounding can
/// leave some of the 32 partitions without keys; those hold a sum of 0 and are
/// skipped, so the merge never depends on an empty partition's scale.
__launch_bounds__(kLanes) __global__ void BaseQwenKvarnSplitKReduceKernel(
    const float* __restrict__ gate, const float* __restrict__ scratch,
    float* __restrict__ out_context, std::uint32_t num_heads,
    std::uint32_t split_count) {
  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;
  const std::size_t row = blockIdx.y;
  if (gate != nullptr) {
    gate += row * num_heads * kHeadDim;
  }
  out_context += row * num_heads * kHeadDim;
  scratch += row * num_heads * max_splits * (kHeadDim + 2);

  const std::uint32_t head = blockIdx.x;
  const std::uint32_t lane = threadIdx.x;
  if (head >= num_heads || lane >= kLanes) {
    return;
  }

  const std::size_t head_partition_offset =
      static_cast<std::size_t>(head) * max_splits;
  float global_max = -INFINITY;
  for (std::uint32_t split = 0; split < split_count; ++split) {
    global_max = fmaxf(
        global_max, scratch[((head_partition_offset + split) * 2) + 0]);
  }

  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
  const auto* partial_outputs =
      reinterpret_cast<const float4*>(scratch + stats_elements);
  auto* output = reinterpret_cast<float4*>(
      out_context + (static_cast<std::size_t>(head) * kHeadDim));
  const auto* gate_values =
      gate == nullptr ? nullptr
                      : reinterpret_cast<const float4*>(
                            gate + (static_cast<std::size_t>(head) * kHeadDim));

  float global_sum = 0.0F;
  float4 results[kVectorsPerLane]{};
  for (std::uint32_t split = 0; split < split_count; ++split) {
    const std::size_t partition_index = head_partition_offset + split;
    const float partition_sum = scratch[(partition_index * 2) + 1];
    if (partition_sum <= 0.0F) {
      continue;
    }
    const float partition_scale =
        expf(scratch[partition_index * 2] - global_max);
    global_sum = fmaf(partition_sum, partition_scale, global_sum);
#pragma unroll
    for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
      const float4 partial = partial_outputs[(partition_index * kVectorWidth) +
                                             lane + (vector * kLanes)];
      auto& result = results[vector];
      result.x = fmaf(partial.x, partition_scale, result.x);
      result.y = fmaf(partial.y, partition_scale, result.y);
      result.z = fmaf(partial.z, partition_scale, result.z);
      result.w = fmaf(partial.w, partition_scale, result.w);
    }
  }
  const float inverse_sum = 1.0F / global_sum;
  // The inverse rotation is applied once after the partial merge, before the
  // SiLU gate (design D3).
  __shared__ float s_context[kHeadDim];
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
    auto result = results[vector];
    result.x *= inverse_sum;
    result.y *= inverse_sum;
    result.z *= inverse_sum;
    result.w *= inverse_sum;
    *reinterpret_cast<float4*>(
        &s_context[(lane + (vector * kLanes)) * 4]) = result;
  }
  __syncwarp();
  KvarnFwhtRowWarp(s_context, static_cast<int>(lane));
  __syncwarp();
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
    float4 result =
        *reinterpret_cast<const float4*>(&s_context[(lane + (vector * kLanes)) * 4]);
    if (gate_values != nullptr) {
      const float4 gate_value = gate_values[lane + (vector * kLanes)];
      result.x *= 1.0F / (1.0F + expf(-gate_value.x));
      result.y *= 1.0F / (1.0F + expf(-gate_value.y));
      result.z *= 1.0F / (1.0F + expf(-gate_value.z));
      result.w *= 1.0F / (1.0F + expf(-gate_value.w));
    }
    output[lane + (vector * kLanes)] = result;
  }
}
// T-05 verification rows (design D2/D7/D8). One block covers one query head and
// one tile-aligned split across up to `BaseKvarnCausalMaxRows` rows, one warp per row.
// A chunk of key positions has its raw codes staged once in LDS and consumed by
// every row that reads it quantized; each row then runs the same folded
// arithmetic, in the same order, as the single-row decoders, so its logits stay
// bit-identical to the AR decode at the same position.
constexpr std::uint32_t BaseKvarnCausalMaxRows = 8;
constexpr std::uint32_t BaseKvarnCausalChunk = kKeyBlock;
constexpr std::uint32_t BaseKvarnCausalThreads = BaseKvarnCausalMaxRows * kLanes;

__launch_bounds__(BaseKvarnCausalThreads) __global__ void BaseQwenKvarnCausalDecodeKernel(
    const float* __restrict__ q, float* __restrict__ scratch,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::uint32_t rows,
    std::uint32_t num_heads, std::uint32_t num_kv_heads,
    std::uint32_t split_count, KvarnReadParams kvarn) {
  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;
  const std::size_t stats_elements =
      static_cast<std::size_t>(num_heads) * max_splits * 2;
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t split = blockIdx.y;
  const std::uint32_t warp = threadIdx.x / kLanes;
  const std::uint32_t lane = threadIdx.x % kLanes;

  __shared__ std::uint32_t s_begin[BaseKvarnCausalMaxRows];
  __shared__ std::uint32_t s_end[BaseKvarnCausalMaxRows];
  __shared__ std::uint32_t s_pos[BaseKvarnCausalMaxRows];
  // Raw codes of the chunk, one byte per (token, channel), staged once per
  // block. Staging the codes instead of decoded FP32 values keeps the LDS
  // footprint at 4 KiB (the FP32 staging cost occupancy; see EXPERIMENTS.md)
  // while the rows still run the identical folded arithmetic.
  __shared__ std::uint32_t s_codes_k[BaseKvarnCausalChunk * kHeadDim / 4];
  __shared__ std::uint32_t s_codes_v[BaseKvarnCausalChunk * kHeadDim / 4];

  // Per-row partition: rows below the split-K threshold use one partition
  // [0, sequence) exactly like the single-row online decoder; longer rows use
  // the tile-aligned split of the single-row split-K decoder (design D8).
  const bool row_active = warp < rows;
  const std::uint32_t pos = start_pos + warp;
  std::uint32_t begin = 0;
  std::uint32_t end = 0;
  if (row_active) {
    const std::uint32_t sequence = pos + 1;
    if (sequence < detail::kSplitKDecodeAttentionMinContext) {
      if (split == 0) {
        end = sequence;
      } else {
        begin = sequence;
        end = sequence;
      }
    } else {
      const std::uint32_t blocks = (sequence + kTileTokens - 1) / kTileTokens;
      const std::uint32_t begin_block = (split * blocks) / split_count;
      const std::uint32_t end_block = ((split + 1) * blocks) / split_count;
      begin = begin_block * kTileTokens;
      end = min(end_block * kTileTokens, sequence);
    }
  }
  if (lane == 0) {
    if (warp < rows) {
      s_begin[warp] = begin;
      s_end[warp] = end;
      s_pos[warp] = pos;
    }
  }
  __syncthreads();
  const bool active = row_active && s_begin[warp] < s_end[warp];

  const std::uint32_t group_size = num_heads / num_kv_heads;
  const std::uint32_t kv_head = head / group_size;
  float* const row_scratch =
      scratch + static_cast<std::size_t>(warp) * num_heads * max_splits *
                    (kHeadDim + 2);

  float4 query_values[kVectorsPerLane];
  if (active) {
    const auto* const query = reinterpret_cast<const float4*>(
        q + (static_cast<std::size_t>(warp) * num_heads + head) * kHeadDim);
#pragma unroll
    for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
      query_values[vector] = query[lane + (vector * kLanes)];
    }
  }

  KvarnOnlineState state;
  state.running_max = -INFINITY;
  state.running_sum = 0.0F;
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      state.output[vector][k] = 0.0F;
    }
  }
  KvarnValueFold value_fold{};
  bool tile_open = false;
  std::uint32_t open_tile = 0;

  std::uint32_t union_begin = 0xFFFFFFFFU;
  std::uint32_t union_end = 0;
  for (std::uint32_t r = 0; r < rows; ++r) {
    if (s_begin[r] < s_end[r]) {
      union_begin = min(union_begin, s_begin[r]);
      union_end = max(union_end, s_end[r]);
    }
  }

  const float scale = rsqrtf(static_cast<float>(kHeadDim));
  for (std::uint32_t chunk = union_begin; chunk < union_end;
       chunk += BaseKvarnCausalChunk) {
    const std::uint32_t count = min(BaseKvarnCausalChunk, union_end - chunk);
    // Stage the raw codes of this chunk once: only positions at or beyond the
    // sink that at least one row reads quantized are decoded.
    for (std::uint32_t i = threadIdx.x; i < count * kHeadDim;
         i += BaseKvarnCausalThreads) {
      const std::uint32_t token = i / kHeadDim;
      const std::uint32_t channel = i % kHeadDim;
      const std::uint32_t position = chunk + token;
      std::uint8_t code_k = 0;
      std::uint8_t code_v = 0;
      if (position >= kTileTokens) {
        const std::uint32_t tile = (position - kTileTokens) / kTileTokens;
        const std::uint32_t e_t = kTileTokens + kTileTokens * tile + 127;
        bool staged = false;
        for (std::uint32_t r = 0; r < rows; ++r) {
          if (position >= s_begin[r] && position < s_end[r] &&
              e_t + kvarn.tail_length + 1 <= s_pos[r]) {
            staged = true;
            break;
          }
        }
        if (staged) {
          const std::uint32_t token_in_tile = position - kTileTokens -
                                              kTileTokens * tile;
          const std::uint8_t* const payload_k = KvarnTilePayload(
              kvarn.body, layer_idx, kvarn.kv_heads, kv_head, tile,
              kvarn.tile_count, kvarn.k_tile_bytes, kvarn.v_tile_bytes, false);
          const std::uint8_t* const payload_v = KvarnTilePayload(
              kvarn.body, layer_idx, kvarn.kv_heads, kv_head, tile,
              kvarn.tile_count, kvarn.k_tile_bytes, kvarn.v_tile_bytes, true);
          const std::size_t index =
              static_cast<std::size_t>(token_in_tile) * kHeadDim + channel;
          code_k = KvarnUnpackCode(payload_k, kvarn.k_bits, index);
          code_v = KvarnUnpackCode(payload_v, kvarn.v_bits, index);
        }
      }
      reinterpret_cast<std::uint8_t*>(s_codes_k)[i] = code_k;
      reinterpret_cast<std::uint8_t*>(s_codes_v)[i] = code_v;
    }
    __syncthreads();

    if (active) {
      const std::uint32_t lo = max(chunk, s_begin[warp]);
      const std::uint32_t hi = min(chunk + count, s_end[warp]);
      if (lo < hi) {
        // A chunk never straddles a tile, so the row's whole block belongs to
        // one region: a sealed tile or F16 rows (design D2).
        const bool quantized =
            lo >= kTileTokens &&
            KvarnTileQuantized(s_pos[warp], kvarn.tail_length,
                               (lo - kTileTokens) / kTileTokens);
        const std::uint32_t row_count = hi - lo;
        if (quantized) {
          const std::uint32_t tile = (lo - kTileTokens) / kTileTokens;
          if (tile_open && tile != open_tile) {
            KvarnMergeOpenTile(kvarn, layer_idx, kv_head, open_tile, lane,
                               &value_fold, state.output);
            tile_open = false;
          }
          const std::uint8_t* const payload_k = KvarnTilePayload(
              kvarn.body, layer_idx, kvarn.kv_heads, kv_head, tile,
              kvarn.tile_count, kvarn.k_tile_bytes, kvarn.v_tile_bytes, false);
          const std::uint8_t* const payload_v = KvarnTilePayload(
              kvarn.body, layer_idx, kvarn.kv_heads, kv_head, tile,
              kvarn.tile_count, kvarn.k_tile_bytes, kvarn.v_tile_bytes, true);
          if (!tile_open) {
            KvarnValueFoldLoad(&value_fold);
            tile_open = true;
            open_tile = tile;
          }
          const KvarnKeyFold key_fold =
              KvarnFoldKeyQuery(payload_k, query_values, lane);
          float scores[kKeyBlock];
#pragma unroll
          for (std::uint32_t i = 0; i < kKeyBlock; ++i) {
            if (i < row_count) {
              const std::uint32_t token =
                  lo + i - kTileTokens - kTileTokens * tile;
              float score = key_fold.zero;
#pragma unroll
              for (std::uint32_t vector = 0; vector < kVectorsPerLane;
                   ++vector) {
                // The staged plane is indexed by the token inside this chunk
                // (`i`, since a row's block starts at the chunk edge).
                const std::uint32_t word =
                    s_codes_k[i * (kHeadDim / 4) + lane + (vector * kLanes)];
                const std::uint32_t codes[4] = {
                    word & 0xFFU, (word >> 8) & 0xFFU, (word >> 16) & 0xFFU,
                    (word >> 24) & 0xFFU};
                score = KvarnKeyFoldVector(key_fold, vector, codes, score);
              }
              scores[i] = KvarnReduceKeyScore(
                  score * KvarnTileTokenScale(payload_k, token), scale);
            }
          }
          KvarnBlockOpen(scores, row_count, &state, &value_fold, true);
#pragma unroll
          for (std::uint32_t i = 0; i < kKeyBlock; ++i) {
            if (i < row_count) {
              const std::uint32_t token =
                  lo + i - kTileTokens - kTileTokens * tile;
              const float weight = KvarnBlockWeight(
                  scores[i], &state, KvarnTileTokenScale(payload_v, token));
#pragma unroll
              for (std::uint32_t vector = 0; vector < kVectorsPerLane;
                   ++vector) {
                const std::uint32_t word =
                    s_codes_v[i * (kHeadDim / 4) + lane + (vector * kLanes)];
                const std::uint32_t codes[4] = {
                    word & 0xFFU, (word >> 8) & 0xFFU, (word >> 16) & 0xFFU,
                    (word >> 24) & 0xFFU};
                float rtn_scale = 0.0F;
                float rtn_zero = 0.0F;
                KvarnValueRtnPair(payload_v, token, vector, &rtn_scale,
                                  &rtn_zero);
                KvarnValueFoldVector(&value_fold, vector, codes, weight,
                                     rtn_scale, rtn_zero);
              }
            }
          }
        } else {
          if (tile_open) {
            KvarnMergeOpenTile(kvarn, layer_idx, kv_head, open_tile, lane,
                               &value_fold, state.output);
            tile_open = false;
          }
          float scores[kKeyBlock];
#pragma unroll
          for (std::uint32_t i = 0; i < kKeyBlock; ++i) {
            if (i < row_count) {
              const __half* const row = KvarnF16Row(
                  kvarn.sink, kvarn.tail, kvarn.sink_plane_elements,
                  kvarn.tail_plane_elements, layer_idx, kvarn.kv_heads, kv_head,
                  lo + i, kvarn.ring_rows, false);
              scores[i] = KvarnReduceKeyScore(
                  KvarnF16KeyScore(row, query_values, lane), scale);
            }
          }
          KvarnBlockOpen(scores, row_count, &state, nullptr, false);
#pragma unroll
          for (std::uint32_t i = 0; i < kKeyBlock; ++i) {
            if (i < row_count) {
              const float weight = KvarnBlockWeight(scores[i], &state, 1.0F);
              const __half* const row = KvarnF16Row(
                  kvarn.sink, kvarn.tail, kvarn.sink_plane_elements,
                  kvarn.tail_plane_elements, layer_idx, kvarn.kv_heads, kv_head,
                  lo + i, kvarn.ring_rows, true);
#pragma unroll
              for (std::uint32_t vector = 0; vector < kVectorsPerLane;
                   ++vector) {
                const std::uint32_t base = (lane + (vector * kLanes)) * 4;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                  state.output[vector][k] = fmaf(
                      weight,
                      __half2float(row[base + static_cast<std::uint32_t>(k)]),
                      state.output[vector][k]);
                }
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }

  if (active && tile_open) {
    KvarnMergeOpenTile(kvarn, layer_idx, kv_head, open_tile, lane, &value_fold,
                       state.output);
    tile_open = false;
  }

  // Empty partitions keep a zero sum so the reduce step skips them, exactly as
  // the single-row decoder's empty partitions are skipped. Idle warps (beyond
  // the group's row count) own no scratch slice.
  if (warp >= rows) {
    return;
  }
  if (lane == 0) {
    row_scratch[(static_cast<std::size_t>(head) * max_splits + split) * 2] =
        active ? state.running_max : -INFINITY;
    row_scratch[(static_cast<std::size_t>(head) * max_splits + split) * 2 + 1] =
        active ? state.running_sum : 0.0F;
  }
  auto* const partial_outputs =
      reinterpret_cast<float4*>(row_scratch + stats_elements);
  float4* const partial_output =
      partial_outputs +
      (static_cast<std::size_t>(head) * max_splits + split) * kVectorWidth;
#pragma unroll
  for (std::uint32_t vector = 0; vector < kVectorsPerLane; ++vector) {
    float4 result;
    result.x = state.output[vector][0];
    result.y = state.output[vector][1];
    result.z = state.output[vector][2];
    result.w = state.output[vector][3];
    partial_output[lane + (vector * kLanes)] = result;
  }
}
void BaseLaunchKvarnCausalDecodeAttention(
    const float* q, const float* gate, float* out_context,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t num_heads, std::uint32_t num_kv_heads,
    const KvarnReadParams& kvarn, float* split_k_scratch, hipStream_t stream) {
  if (split_k_scratch == nullptr || batch_size == 0 || num_kv_heads == 0) {
    return;
  }
  constexpr std::size_t max_splits = detail::kSplitKDecodeAttentionMaxSplits;
  const std::size_t row_stride =
      static_cast<std::size_t>(num_heads) * max_splits * (kHeadDim + 2);
  const std::size_t width = static_cast<std::size_t>(num_heads) * kHeadDim;
  for (std::size_t row = 0; row < batch_size; row += BaseKvarnCausalMaxRows) {
    const std::uint32_t rows = static_cast<std::uint32_t>(
        std::min<std::size_t>(BaseKvarnCausalMaxRows, batch_size - row));
    const std::uint32_t group_start = start_pos + static_cast<std::uint32_t>(row);
    const std::uint32_t splits =
        (group_start + rows >= detail::kSplitKDecodeAttentionMinContext)
            ? detail::kSplitKDecodeAttentionMaxSplits
            : 1;
    BaseQwenKvarnCausalDecodeKernel<<<dim3(num_heads, splits), BaseKvarnCausalThreads, 0,
                                  stream>>>(
        q + row * width, split_k_scratch + row * row_stride, layer_idx,
        group_start, rows, num_heads, num_kv_heads, splits, kvarn);
    BaseQwenKvarnSplitKReduceKernel<<<dim3(num_heads, rows), kLanes, 0, stream>>>(
        gate == nullptr ? nullptr : gate + row * width,
        split_k_scratch + row * row_stride, out_context + row * width,
        num_heads, splits);
  }
}

void BaseLaunchKvarnDecodeAttention(const float* q, const float* gate,
                                float* out_context, std::uint32_t layer_idx,
                                std::uint32_t start_pos,
                                const std::uint32_t* d_pos,
                                std::size_t batch_size,
                                std::uint32_t num_heads,
                                std::uint32_t num_kv_heads,
                                const KvarnReadParams& kvarn,
                                float* split_k_scratch, hipStream_t stream) {
  const std::uint32_t sequence_length =
      start_pos + static_cast<std::uint32_t>(batch_size);
  if (split_k_scratch != nullptr && detail::IsSplitKDecodeAttentionSupported(
          sequence_length, num_heads, num_kv_heads, kHeadDim)) {
    const std::uint32_t splits =
        detail::SelectDecodeAttentionSplitCount(sequence_length);
    const dim3 partial_grid(num_heads, static_cast<unsigned>(batch_size),
                            splits);
    BaseQwenKvarnSplitKPartialsKernel<<<partial_grid, kLanes, 0, stream>>>(
        q, split_k_scratch, layer_idx, start_pos, d_pos, num_heads,
        num_kv_heads, splits, kvarn);
    const dim3 reduce_grid(num_heads, static_cast<unsigned>(batch_size));
    BaseQwenKvarnSplitKReduceKernel<<<reduce_grid, kLanes, 0, stream>>>(
        gate, split_k_scratch, out_context, num_heads, splits);
    return;
  }
  const dim3 grid(num_heads, static_cast<unsigned>(batch_size));
  QwenKvarnDecodeOnlineKernel<<<grid, kLanes, 0, stream>>>(
      q, gate, out_context, layer_idx, start_pos, d_pos, num_heads,
      num_kv_heads, kvarn);
}
