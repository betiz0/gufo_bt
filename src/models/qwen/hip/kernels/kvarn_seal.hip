// KVarN seal kernel (openspec add-kvarn-kv-cache-quantization, design D4-D6).
// Turns the rotated F16 rows of one 128-token tile into the sealed payload:
// dual-axis normalization (16 fixed-order passes, channel first), asymmetric
// RTN over the K channel axis or the V token-group axis, then the bit-plane
// layout of design D5.
//
// Determinism: every reduction runs in ascending index order, so the scales are
// a function of the tile alone. The final normalized values are recomputed as
// `clamped_raw / (channel_scale * token_scale)`, exactly the form the reader
// inverts, so a tile's payload is reproducible from its rows and its stored
// scales (T-01 stage 1).
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "src/models/qwen/hip/kvarn.hpp"
#include "src/models/qwen/kvarn_reference.hpp"

namespace gufo::hip {
namespace {

constexpr int kThreads = 256;
constexpr int kRows = static_cast<int>(models::qwen::kKvarnTileTokens);
constexpr int kCols = static_cast<int>(models::qwen::kKvarnTileChannels);
constexpr int kValueGroups =
    static_cast<int>(models::qwen::kKvarnValueGroups);
constexpr int kValueGroupChannels =
    static_cast<int>(models::qwen::kKvarnValueGroupChannels);
constexpr int kPasses = static_cast<int>(models::qwen::kKvarnNormPasses);
constexpr std::size_t kTileElements =
    static_cast<std::size_t>(kRows) * kCols;
constexpr float kValueClamp =
    static_cast<float>(models::qwen::kKvarnValueClamp);
// Payload field offsets (design D5); the CPU oracle pins the same layout.
constexpr std::size_t kRtnScaleOffset = 0;
constexpr std::size_t kRtnZeroOffset =
    kRtnScaleOffset + 2 * models::qwen::kKvarnRtnParams;
constexpr std::size_t kNormChannelOffset =
    kRtnZeroOffset + 2 * models::qwen::kKvarnRtnParams;
constexpr std::size_t kNormTokenOffset =
    kNormChannelOffset + 2 * models::qwen::kKvarnTileChannels;
constexpr std::size_t kLowOffset = models::qwen::kKvarnPayloadMetadataBytes;
constexpr std::size_t kMidOffset =
    kLowOffset + models::qwen::kKvarnLowPlaneBytes;
constexpr std::size_t kHighOffset =
    kMidOffset + models::qwen::kKvarnBitPlaneBytes;

void Check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string("KVarN seal ") + what + ": " +
                             hipGetErrorString(status));
  }
}

/// F16 row of one (layer, KV head, position) in the sink or tail ring, or in
/// the bounded per-chunk workspace while the row belongs to the current chunk
/// (design D4/4.1).
__device__ __forceinline__ const __half* KvarnRowPtr(
    const __half* sink, const __half* tail, std::uint64_t sink_plane_elements,
    std::uint64_t tail_plane_elements, const __half* chunk,
    std::uint64_t chunk_plane_elements, std::uint32_t chunk_begin,
    std::uint32_t chunk_end, std::uint32_t layer, std::uint32_t kv_heads,
    std::uint32_t head, std::uint32_t position, std::uint32_t ring_rows,
    bool is_value) {
  if (chunk != nullptr && position >= chunk_begin && position < chunk_end) {
    const std::size_t row =
        (static_cast<std::size_t>(position - chunk_begin) * kv_heads + head) *
        kCols;
    return chunk + (is_value ? chunk_plane_elements : 0) + row;
  }
  if (position < models::qwen::kKvarnSinkRows) {
    const std::size_t row =
        models::qwen::KvarnSinkRowOffset(layer, kv_heads, head, position);
    return sink + (is_value ? sink_plane_elements : 0) + row;
  }
  const std::size_t row = models::qwen::KvarnTailRowOffset(
      layer, kv_heads, head, position, ring_rows);
  return tail + (is_value ? tail_plane_elements : 0) + row;
}

/// Element (token, channel) owning RTN parameter `p`, mirroring
/// `RtnSetElement` in kvarn_reference.cpp.
__device__ __forceinline__ std::size_t RtnSetElement(bool token_axis,
                                                     int p, int n) {
  if (!token_axis) {
    return static_cast<std::size_t>(n) * kCols + p;
  }
  return static_cast<std::size_t>(p / kValueGroups) * kCols +
         static_cast<std::size_t>(p % kValueGroups) * kValueGroupChannels + n;
}

__device__ __forceinline__ double ClampLogScale(double value) {
  if (value > models::qwen::kKvarnLogScaleClamp) {
    return models::qwen::kKvarnLogScaleClamp;
  }
  if (value < -models::qwen::kKvarnLogScaleClamp) {
    return -models::qwen::kKvarnLogScaleClamp;
  }
  return value;
}

__device__ __forceinline__ float ClampValue(float value) {
  if (value > kValueClamp) {
    return kValueClamp;
  }
  if (value < -kValueClamp) {
    return -kValueClamp;
  }
  return value;
}

/// Clamps the tile and runs the 16 dual-axis passes, leaving the accumulated
/// per-channel and per-token scales in `s_channel` / `s_token`.
__device__ void NormalizeScales(float* w, float* s_channel, float* s_token,
                                double* s_log_channel, double* s_log_token,
                                int tid) {
  if (tid < kCols) {
    s_log_channel[tid] = 0.0;
  }
  if (tid < kRows) {
    s_log_token[tid] = 0.0;
  }
  for (std::size_t index = tid; index < kTileElements;
       index += static_cast<std::size_t>(kThreads)) {
    w[index] = ClampValue(w[index]);
  }
  __syncthreads();

  for (int pass = 0; pass < kPasses; ++pass) {
    if (pass % 2 == 0) {
      double sum = 0.0;
      for (int r = 0; r < kRows; ++r) {
        const double v = static_cast<double>(w[r * kCols + tid]);
        sum += v * v;
      }
      double rms = sqrt(sum / static_cast<double>(kRows));
      if (!(rms > models::qwen::kKvarnNormEpsilon)) {
        rms = models::qwen::kKvarnNormEpsilon;
      }
      s_log_channel[tid] = ClampLogScale(s_log_channel[tid] + log(rms));
      for (int r = 0; r < kRows; ++r) {
        w[r * kCols + tid] = static_cast<float>(
            static_cast<double>(w[r * kCols + tid]) / rms);
      }
    } else if (tid < kRows) {
      float* row = w + static_cast<std::size_t>(tid) * kCols;
      double sum = 0.0;
      for (int c = 0; c < kCols; ++c) {
        const double v = static_cast<double>(row[c]);
        sum += v * v;
      }
      double rms = sqrt(sum / static_cast<double>(kCols));
      if (!(rms > models::qwen::kKvarnNormEpsilon)) {
        rms = models::qwen::kKvarnNormEpsilon;
      }
      s_log_token[tid] = ClampLogScale(s_log_token[tid] + log(rms));
      for (int c = 0; c < kCols; ++c) {
        row[c] = static_cast<float>(static_cast<double>(row[c]) / rms);
      }
    }
    __syncthreads();
  }

  s_channel[tid] = static_cast<float>(exp(s_log_channel[tid]));
  if (tid < kRows) {
    s_token[tid] = static_cast<float>(exp(s_log_token[tid]));
  }
  __syncthreads();
}

/// RTN-quantizes the normalized tile in `w` and packs it into `payload`. The
/// metadata is stored as FP16 and the codes are encoded from those stored
/// values, so the stage-1 oracle reproduces them exactly from the payload.
__device__ void QuantizeAndPack(float* w, const float* s_channel,
                                const float* s_token, std::uint8_t* payload,
                                std::uint8_t bits, bool token_axis, int tid) {
  const int set_size = token_axis ? kValueGroupChannels : kRows;
  const double qmax = static_cast<double>((1U << bits) - 1U);
  double minimum = static_cast<double>(w[RtnSetElement(token_axis, tid, 0)]);
  double maximum = minimum;
  for (int n = 0; n < set_size; ++n) {
    const double v = static_cast<double>(w[RtnSetElement(token_axis, tid, n)]);
    if (v < minimum) {
      minimum = v;
    }
    if (v > maximum) {
      maximum = v;
    }
  }
  double scale = (maximum - minimum) / qmax;
  if (!(scale > models::qwen::kKvarnNormEpsilon)) {
    scale = models::qwen::kKvarnNormEpsilon;
  }
  const __half stored_scale = __float2half_rn(static_cast<float>(scale));
  const __half stored_zero = __float2half_rn(static_cast<float>(minimum));
  reinterpret_cast<std::uint16_t*>(payload + kRtnScaleOffset)[tid] =
      __half_as_ushort(stored_scale);
  reinterpret_cast<std::uint16_t*>(payload + kRtnZeroOffset)[tid] =
      __half_as_ushort(stored_zero);
  reinterpret_cast<std::uint16_t*>(payload + kNormChannelOffset)[tid] =
      __half_as_ushort(__float2half_rn(s_channel[tid]));
  if (tid < kRows) {
    reinterpret_cast<std::uint16_t*>(payload + kNormTokenOffset)[tid] =
        __half_as_ushort(__float2half_rn(s_token[tid]));
  }
  const float encode_scale = __half2float(stored_scale);
  const float encode_zero = __half2float(stored_zero);
  const float code_max = static_cast<float>((1U << bits) - 1U);
  for (int n = 0; n < set_size; ++n) {
    const std::size_t index = RtnSetElement(token_axis, tid, n);
    float code = roundf((w[index] - encode_zero) / encode_scale);
    if (!(code >= 0.0F)) {
      code = 0.0F;
    } else if (code > code_max) {
      code = code_max;
    }
    w[index] = code;
  }
  __syncthreads();

  std::uint8_t* const low = payload + kLowOffset;
  for (int b = tid; b < kRows * kCols / 2; b += kThreads) {
    const auto lo = static_cast<std::uint8_t>(w[2 * b] + 0.5F);
    const auto hi = static_cast<std::uint8_t>(w[2 * b + 1] + 0.5F);
    low[b] = static_cast<std::uint8_t>((hi << 4U) | (lo & 0x0FU));
  }
  if (bits >= 5) {
    std::uint8_t* const mid = payload + kMidOffset;
    for (int b = tid; b < kRows * kCols / 8; b += kThreads) {
      std::uint8_t value = 0;
      for (int k = 0; k < 8; ++k) {
        const auto code = static_cast<std::uint8_t>(w[8 * b + k] + 0.5F);
        value = static_cast<std::uint8_t>(value | (((code >> 4U) & 1U) << k));
      }
      mid[b] = value;
    }
  }
  if (bits >= 6) {
    std::uint8_t* const high = payload + kHighOffset;
    for (int b = tid; b < kRows * kCols / 8; b += kThreads) {
      std::uint8_t value = 0;
      for (int k = 0; k < 8; ++k) {
        const auto code = static_cast<std::uint8_t>(w[8 * b + k] + 0.5F);
        value = static_cast<std::uint8_t>(value | (((code >> 5U) & 1U) << k));
      }
      high[b] = value;
    }
  }
  __syncthreads();
}

/// Seals the tiles the view rule selects for the newest query row. Tiles below
/// `sealed` are already sealed and are never revisited; a tile's rows are in
/// the sink, the tail ring or the current chunk workspace whenever the view
/// selects it (design D2/D4/4.1), so no ring-window clamp is needed.
__device__ __forceinline__ bool PendingTiles(std::uint32_t sealed,
                                             std::uint32_t q_max,
                                             std::uint32_t tail_length,
                                             std::uint32_t* begin,
                                             std::uint32_t* end) {
  if (q_max < tail_length + 256) {
    *begin = 0;
    *end = 0;
    return false;
  }
  const std::uint32_t last = (q_max - tail_length - 256) / 128 + 1;
  *begin = sealed;
  *end = last;
  return *begin < *end;
}

__global__ __launch_bounds__(kThreads) void KvarnSealKernel(
    __half* sink, __half* tail, std::uint64_t sink_plane_elements,
    std::uint64_t tail_plane_elements, const __half* chunk,
    std::uint64_t chunk_plane_elements, std::uint32_t ring_rows,
    std::uint8_t* body, std::uint32_t tile_count, std::uint32_t k_tile_bytes,
    std::uint32_t v_tile_bytes, std::uint32_t layer, std::uint32_t kv_heads,
    std::uint32_t start_host, const std::uint32_t* d_start,
    std::uint32_t count, std::uint32_t tail_length, std::uint32_t* sealed,
    float* scratch, std::uint32_t* error_flag, std::uint8_t k_bits,
    std::uint8_t v_bits) {
  const int tid = threadIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t start = d_start != nullptr ? *d_start : start_host;
  const std::uint32_t q_max = start + count - 1;
  std::uint32_t* const sealed_ptr =
      sealed + static_cast<std::size_t>(layer) * kv_heads + head;
  float* const w = scratch + (static_cast<std::size_t>(layer) * kv_heads +
                              head) * kTileElements;

  __shared__ float s_channel[kCols];
  __shared__ float s_token[kRows];
  __shared__ double s_log_channel[kCols];
  __shared__ double s_log_token[kRows];
  __shared__ int s_bad;

  std::uint32_t begin = 0;
  std::uint32_t end = 0;
  if (!PendingTiles(*sealed_ptr, q_max, tail_length, &begin, &end)) {
    return;
  }
  for (std::uint32_t t = begin; t < end; ++t) {
    const std::uint32_t first_position = 128 + 128 * t;
    for (int plane = 0; plane < 2; ++plane) {
      const bool is_value = plane != 0;
      const std::uint8_t bits = is_value ? v_bits : k_bits;
      const std::uint32_t tile_bytes = is_value ? v_tile_bytes : k_tile_bytes;
      if (tid == 0) {
        s_bad = 0;
      }
      __syncthreads();
      for (int r = 0; r < kRows; ++r) {
        const __half* row = KvarnRowPtr(
            sink, tail, sink_plane_elements, tail_plane_elements, chunk,
            chunk_plane_elements, start, start + count, layer, kv_heads, head,
            first_position + r, ring_rows, is_value);
        const float element = __half2float(row[tid]);
        if (!isfinite(element)) {
          s_bad = 1;
        }
        w[static_cast<std::size_t>(r) * kCols + tid] = element;
      }
      __syncthreads();
      if (s_bad != 0) {
        if (tid == 0) {
          atomicAdd(error_flag, 1U);
        }
        return;
      }
      NormalizeScales(w, s_channel, s_token, s_log_channel, s_log_token, tid);
      __syncthreads();

      // Recompute the normalized values from the stored (FP16-rounded) scales
      // and the clamped rows: the reader inverts exactly this form, so the
      // stage-1 oracle reproduces the codes from the payload alone.
      for (int r = 0; r < kRows; ++r) {
        const __half* row = KvarnRowPtr(
            sink, tail, sink_plane_elements, tail_plane_elements, chunk,
            chunk_plane_elements, start, start + count, layer, kv_heads, head,
            first_position + r, ring_rows, is_value);
        const double divisor =
            static_cast<double>(
                __half2float(__float2half_rn(s_channel[tid]))) *
            static_cast<double>(
                __half2float(__float2half_rn(s_token[r])));
        // An all-zero axis floors its RMS and its accumulated scale can
        // underflow FP16 to 0; the reader reconstructs 0 for that element
        // either way, so store 0 instead of 0/0 (design D5/D6).
        w[static_cast<std::size_t>(r) * kCols + tid] =
            divisor > 0.0
                ? static_cast<float>(
                      static_cast<double>(ClampValue(__half2float(row[tid]))) /
                      divisor)
                : 0.0F;
      }
      __syncthreads();

      std::uint8_t* const payload =
          body + models::qwen::KvarnBodyTileOffset(
                     layer, kv_heads, head, t, tile_count, k_tile_bytes,
                     v_tile_bytes) +
          (is_value ? k_tile_bytes : 0);
      QuantizeAndPack(w, s_channel, s_token, payload, bits, is_value, tid);
    }
    if (tid == 0) {
      *sealed_ptr = t + 1;
    }
    __syncthreads();
  }
}

}  // namespace

KvarnSealState::KvarnSealState(std::uint32_t attention_layers,
                               std::uint32_t kv_heads)
    : layers_(attention_layers), kv_heads_(kv_heads) {
  try {
    Check(hipMalloc(reinterpret_cast<void**>(&sealed_),
                    static_cast<std::size_t>(layers_) * kv_heads_ *
                        sizeof(std::uint32_t)),
          "sealed counters");
    Check(hipMalloc(reinterpret_cast<void**>(&scratch_),
                    static_cast<std::size_t>(layers_) * kv_heads_ *
                        kTileElements * sizeof(float)),
          "tile scratch");
    Check(hipMalloc(reinterpret_cast<void**>(&error_), sizeof(std::uint32_t)),
          "error flag");
    Check(hipMemset(sealed_, 0,
                    static_cast<std::size_t>(layers_) * kv_heads_ *
                        sizeof(std::uint32_t)),
          "counter reset");
    Check(hipMemset(error_, 0, sizeof(std::uint32_t)), "error reset");
  } catch (...) {
    (void)hipFree(sealed_);
    (void)hipFree(scratch_);
    (void)hipFree(error_);
    throw;
  }
}

KvarnSealState::~KvarnSealState() {
  (void)hipFree(sealed_);
  (void)hipFree(scratch_);
  (void)hipFree(error_);
}

void KvarnSealState::Reset(hipStream_t stream) {
  Check(hipMemsetAsync(sealed_, 0,
                       static_cast<std::size_t>(layers_) * kv_heads_ *
                           sizeof(std::uint32_t),
                       stream),
        "counter reset");
  Check(hipMemsetAsync(error_, 0, sizeof(std::uint32_t), stream),
        "error reset");
}

bool KvarnSealState::HasError(hipStream_t stream) const {
  std::uint32_t flag = 0;
  Check(hipMemcpyAsync(&flag, error_, sizeof(flag), hipMemcpyDeviceToHost,
                       stream),
        "error read");
  Check(hipStreamSynchronize(stream), "error sync");
  return flag != 0;
}

void LaunchKvarnSeal(const KvarnSealState& state,
                     const QwenKvarnPreset& preset, void* sink, void* tail,
                     std::uint64_t sink_plane_elements,
                     std::uint64_t tail_plane_elements, std::uint32_t ring_rows,
                     std::uint8_t* body, std::uint32_t tile_count,
                     std::uint32_t layer_idx, std::uint32_t num_kv_heads,
                     std::uint32_t start, const std::uint32_t* d_start,
                     std::uint32_t count, hipStream_t stream,
                     const void* chunk, std::uint64_t chunk_plane_elements) {
  if (!preset.enabled() || count == 0 || tile_count == 0) {
    return;
  }
  KvarnSealKernel<<<dim3(1, num_kv_heads), kThreads, 0, stream>>>(
      static_cast<__half*>(sink), static_cast<__half*>(tail),
      sink_plane_elements, tail_plane_elements,
      static_cast<const __half*>(chunk), chunk_plane_elements, ring_rows, body,
      tile_count,
      static_cast<std::uint32_t>(
          models::qwen::KvarnTilePayloadBytes(preset.k_bits)),
      static_cast<std::uint32_t>(
          models::qwen::KvarnTilePayloadBytes(preset.v_bits)),
      layer_idx, num_kv_heads, start, d_start, count,
      QwenKvarnTailRows(preset), state.sealed(), state.scratch(),
      state.error_flag(), preset.k_bits, preset.v_bits);
}

namespace {

/// Copies the last `ring_rows` rows of one chunk from the per-chunk workspace
/// into the sink and tail ring, so the next decode or prefill step reads them
/// exactly as if they had been written there directly (design D4/4.1). Only
/// the rows whose slots are distinct after the merge are written, so the
/// result does not depend on block scheduling.
__global__ __launch_bounds__(kThreads) void KvarnChunkFlushKernel(
    const __half* __restrict__ chunk, std::uint64_t chunk_plane_elements,
    __half* __restrict__ sink, __half* __restrict__ tail,
    std::uint64_t sink_plane_elements, std::uint64_t tail_plane_elements,
    std::uint32_t ring_rows, std::uint32_t layer, std::uint32_t kv_heads,
    std::uint32_t chunk_begin, std::uint32_t chunk_rows) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t i = threadIdx.x;
  if (row >= chunk_rows || head >= kv_heads || i >= kCols) {
    return;
  }
  const std::uint32_t position = chunk_begin + row;
  // The sink keeps every position below the sink, once, and the ring keeps the
  // newest ring_rows positions of the chunk. Anything older is superseded by a
  // later chunk or decode step and must not race.
  const bool to_sink = position < models::qwen::kKvarnSinkRows;
  const std::uint32_t ring_first =
      chunk_rows > ring_rows ? chunk_begin + chunk_rows - ring_rows
                             : chunk_begin;
  if (!to_sink && position < ring_first) {
    return;
  }
  for (int plane = 0; plane < 2; ++plane) {
    const bool is_value = plane != 0;
    const __half* const src =
        chunk + (is_value ? chunk_plane_elements : 0) +
        (static_cast<std::size_t>(row) * kv_heads + head) * kCols;
    if (to_sink) {
      const std::size_t dst_row = models::qwen::KvarnSinkRowOffset(
          layer, kv_heads, head, position);
      sink[(is_value ? sink_plane_elements : 0) + dst_row + i] = src[i];
    } else {
      const std::size_t dst_row = models::qwen::KvarnTailRowOffset(
          layer, kv_heads, head, position, ring_rows);
      tail[(is_value ? tail_plane_elements : 0) + dst_row + i] = src[i];
    }
  }
}

}  // namespace

void LaunchKvarnChunkFlush(const KvarnWriteParams& kvarn,
                           std::uint32_t layer_idx, std::uint32_t num_kv_heads,
                           std::uint32_t chunk_begin, std::uint32_t chunk_rows,
                           hipStream_t stream) {
  if (!kvarn.enabled() || kvarn.chunk == nullptr || chunk_rows == 0) {
    return;
  }
  const std::uint32_t ring_rows = kvarn.ring_rows;
  // The kernel skips rows that are neither below the sink nor within the
  // newest ring_rows positions, so a whole chunk can be handed to it.
  KvarnChunkFlushKernel<<<dim3(chunk_rows, num_kv_heads), kThreads, 0, stream>>>(
      kvarn.chunk, kvarn.chunk_plane_elements,
      static_cast<__half*>(kvarn.sink), static_cast<__half*>(kvarn.tail),
      kvarn.sink_plane_elements, kvarn.tail_plane_elements, ring_rows,
      layer_idx, num_kv_heads, chunk_begin, chunk_rows);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
