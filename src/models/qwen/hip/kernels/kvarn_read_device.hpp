// Shared device-side KVarN read helpers (openspec
// add-kvarn-kv-cache-quantization, design D2/D5/D7). Included by the prefill
// attention kernels for the view rule, the tile payload layout and the
// stored-element arithmetic; the decode read arithmetic lives in
// `kernels/kvarn_read.hip` (design D7). Qwen-only; never include from other
// model trees.
#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_KVARN_READ_DEVICE_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_KVARN_READ_DEVICE_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops/attention.hpp"
#include "src/models/qwen/kvarn_reference.hpp"

namespace gufo::hip {

inline constexpr std::uint32_t kKvarnHeadDim = 256;
inline constexpr std::uint32_t kKvarnLanes = 32;
inline constexpr std::uint32_t kKvarnVectorWidth = kKvarnHeadDim / 4;
inline constexpr std::uint32_t kKvarnVectorsPerLane =
    kKvarnVectorWidth / kKvarnLanes;
inline constexpr std::size_t kKvarnLowOffset =
    models::qwen::kKvarnPayloadMetadataBytes;
inline constexpr std::size_t kKvarnMidOffset =
    kKvarnLowOffset + models::qwen::kKvarnLowPlaneBytes;
inline constexpr std::size_t kKvarnHighOffset =
    kKvarnMidOffset + models::qwen::kKvarnBitPlaneBytes;

__device__ __forceinline__ std::uint8_t KvarnUnpackCode(
    const std::uint8_t* payload, std::uint8_t bits, std::size_t index) {
  const std::uint8_t* const low = payload + kKvarnLowOffset;
  std::uint8_t code =
      static_cast<std::uint8_t>((low[index / 2] >> (4U * (index % 2))) & 0x0FU);
  if (bits >= 5) {
    code = static_cast<std::uint8_t>(
        code | (((payload + kKvarnMidOffset)[index / 8] >> (index % 8)) & 1U)
                   << 4);
  }
  if (bits >= 6) {
    code = static_cast<std::uint8_t>(
        code | (((payload + kKvarnHighOffset)[index / 8] >> (index % 8)) & 1U)
                   << 5);
  }
  return code;
}

/// True when the view rule of design D2 reads tile `t` as quantized for a query
/// at absolute position `q`: e_t <= q - L - 1.
__device__ __forceinline__ bool KvarnTileQuantized(std::uint32_t q,
                                                   std::uint32_t tail_length,
                                                   std::uint32_t tile) {
  const std::uint32_t e_t = 128 + 128 * tile + 127;
  return e_t + tail_length + 1 <= q;
}

/// Payload of tile `t` for the K or V plane of one (layer, KV head).
__device__ __forceinline__ const std::uint8_t* KvarnTilePayload(
    const std::uint8_t* body, std::uint32_t layer, std::uint32_t kv_heads,
    std::uint32_t head, std::uint32_t tile, std::uint32_t tile_count,
    std::uint32_t k_tile_bytes, std::uint32_t v_tile_bytes, bool is_value) {
  const std::size_t offset = models::qwen::KvarnBodyTileOffset(
      layer, kv_heads, head, tile, tile_count, k_tile_bytes, v_tile_bytes);
  return body + offset + (is_value ? k_tile_bytes : 0);
}

/// F16 row of one (layer, KV head, position) in the sink or tail ring.
__device__ __forceinline__ const __half* KvarnF16Row(
    const __half* sink, const __half* tail, std::uint64_t sink_plane_elements,
    std::uint64_t tail_plane_elements, std::uint32_t layer,
    std::uint32_t kv_heads, std::uint32_t head, std::uint32_t position,
    std::uint32_t ring_rows, bool is_value) {
  if (position < models::qwen::kKvarnSinkRows) {
    const std::size_t row =
        models::qwen::KvarnSinkRowOffset(layer, kv_heads, head, position);
    return sink + (is_value ? sink_plane_elements : 0) + row;
  }
  const std::size_t row = models::qwen::KvarnTailRowOffset(
      layer, kv_heads, head, position, ring_rows);
  return tail + (is_value ? tail_plane_elements : 0) + row;
}

/// One element's raw rotated value from a sealed tile, before the channel
/// normalization scale: `code * rtn_scale[p] + rtn_zero[p]` with `p` the
/// channel for K and the (token, 128-channel group) for V (design D5). The
/// prefill reader multiplies the channel scale afterwards, so staging the raw
/// value keeps the tile bits for every query row it serves.
__device__ __forceinline__ float KvarnRawDecodedElement(
    const std::uint8_t* payload, std::uint8_t bits, bool is_value,
    std::uint32_t token, std::uint32_t channel) {
  const float code = static_cast<float>(KvarnUnpackCode(
      payload, bits,
      static_cast<std::size_t>(token) * kKvarnHeadDim + channel));
  const std::size_t parameter =
      is_value
          ? (static_cast<std::size_t>(token) * models::qwen::kKvarnValueGroups +
             channel / models::qwen::kKvarnValueGroupChannels)
          : channel;
  const auto* const rtn_scale = reinterpret_cast<const __half*>(payload);
  const auto* const rtn_zero = rtn_scale + models::qwen::kKvarnRtnParams;
  return code * __half2float(rtn_scale[parameter]) +
         __half2float(rtn_zero[parameter]);
}

/// Channel normalization scale of a sealed tile (design D5).
__device__ __forceinline__ float KvarnTileChannelScale(
    const std::uint8_t* payload, std::uint32_t channel) {
  return __half2float(
      reinterpret_cast<const __half*>(payload + 4 * 256)[channel]);
}

/// Token normalization scale of one tile payload (design D5). K scores fold the
/// K tile's token scale and V accumulates the V tile's, so the two planes keep
/// their own scales.
__device__ __forceinline__ float KvarnTileTokenScale(
    const std::uint8_t* payload, std::uint32_t token) {
  return __half2float(
      reinterpret_cast<const __half*>(payload + 4 * 256 + 2 * 256)[token]);
}

// The decode read arithmetic (folded K query, normalized-space V and key
// blocks) is not shared with the prefill kernels: it lives in
// `kernels/kvarn_read.hip` next to the three decode paths that must stay
// bit-identical (design D7).

/// Number of tiles the view rule of design D2 reads quantized for a query at
/// absolute position `q`: e_t + L + 1 <= q, i.e. t <= (q - 256 - L) / 128.
/// Prefill launches one kernel per query-row group that shares this count, so
/// each row keeps the AR decode's view (design D2/D4).
__host__ __device__ __forceinline__ std::uint32_t KvarnQuantizedTilesForQuery(
    std::uint32_t q, std::uint32_t tail_length) {
  const std::uint32_t first = 256U + tail_length;
  return q < first ? 0U : ((q - first) / 128U) + 1U;
}

/// The D2 view rule with the per-row count already resolved: a position belongs
/// to a quantized tile when its tile index is below `quantized_tiles`.
__device__ __forceinline__ bool KvarnTileInView(std::uint32_t position,
                                                std::uint32_t quantized_tiles) {
  return position >= models::qwen::kKvarnSinkRows &&
         ((position - models::qwen::kKvarnSinkRows) / 128) < quantized_tiles;
}

/// F16 row of one (layer, KV head, position) during a prefill chunk: a position
/// inside `[chunk_begin, chunk_end)` is served from the bounded per-chunk
/// workspace (design D4/4.1), anything older from the sink or the tail ring,
/// which the deferred flush has not overwritten yet.
__device__ __forceinline__ const __half* KvarnF16RowForChunk(
    const KvarnReadParams& kvarn, std::uint32_t layer, std::uint32_t kv_head,
    std::uint32_t position, std::uint32_t chunk_begin, std::uint32_t chunk_end,
    bool is_value) {
  if (kvarn.chunk != nullptr && position >= chunk_begin &&
      position < chunk_end) {
    const std::size_t row =
        (static_cast<std::size_t>(position - chunk_begin) * kvarn.kv_heads +
         kv_head) *
        kKvarnHeadDim;
    return kvarn.chunk + (is_value ? kvarn.chunk_plane_elements : 0) + row;
  }
  return KvarnF16Row(kvarn.sink, kvarn.tail, kvarn.sink_plane_elements,
                     kvarn.tail_plane_elements, layer, kvarn.kv_heads, kv_head,
                     position, kvarn.ring_rows, is_value);
}

/// Stored K element of `position` at `channel` for a fixed row view: F16 rows
/// contribute as stored, a sealed tile decodes and folds the channel
/// normalization and the tile's token scale (design D5), exactly as the decode
/// read folds `KvarnKeyScore`'s trailing token scale.
__device__ __forceinline__ float KvarnStoredKElement(
    const KvarnReadParams& kvarn, std::uint32_t layer, std::uint32_t kv_head,
    std::uint32_t position, std::uint32_t quantized_tiles,
    std::uint32_t chunk_begin, std::uint32_t chunk_end, std::uint32_t channel) {
  if (!KvarnTileInView(position, quantized_tiles)) {
    const __half* const row = KvarnF16RowForChunk(
        kvarn, layer, kv_head, position, chunk_begin, chunk_end, false);
    return __half2float(row[channel]);
  }
  const std::uint32_t tile = (position - 128) / 128;
  const std::uint32_t token = position - 128 - 128 * tile;
  const std::uint8_t* const payload = KvarnTilePayload(
      kvarn.body, layer, kvarn.kv_heads, kv_head, tile, kvarn.tile_count,
      kvarn.k_tile_bytes, kvarn.v_tile_bytes, false);
  const float raw =
      KvarnRawDecodedElement(payload, kvarn.k_bits, false, token, channel);
  return raw * KvarnTileChannelScale(payload, channel) *
         KvarnTileTokenScale(payload, token);
}

/// Stored V element of `position` at `channel` for a fixed row view: F16 rows
/// contribute as stored, a sealed tile decodes and folds the channel
/// normalization and the tile's token scale (design D5), the same trailing
/// token scale the decode read folds into the softmax weight.
__device__ __forceinline__ float KvarnStoredVElement(
    const KvarnReadParams& kvarn, std::uint32_t layer, std::uint32_t kv_head,
    std::uint32_t position, std::uint32_t quantized_tiles,
    std::uint32_t chunk_begin, std::uint32_t chunk_end, std::uint32_t channel) {
  if (!KvarnTileInView(position, quantized_tiles)) {
    const __half* const row = KvarnF16RowForChunk(
        kvarn, layer, kv_head, position, chunk_begin, chunk_end, true);
    return __half2float(row[channel]);
  }
  const std::uint32_t tile = (position - 128) / 128;
  const std::uint32_t token = position - 128 - 128 * tile;
  const std::uint8_t* const payload = KvarnTilePayload(
      kvarn.body, layer, kvarn.kv_heads, kv_head, tile, kvarn.tile_count,
      kvarn.k_tile_bytes, kvarn.v_tile_bytes, true);
  const float raw =
      KvarnRawDecodedElement(payload, kvarn.v_bits, true, token, channel);
  return raw * KvarnTileChannelScale(payload, channel) *
         KvarnTileTokenScale(payload, token);
}

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_KVARN_READ_DEVICE_HPP_
