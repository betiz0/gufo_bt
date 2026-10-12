// KVarN tile-quantization variant comparison (openspec change
// improve-kvarn-k6v5-quality-and-speed, design D7 phase 1a, task 4.4).
//
// Compares seven quantization variants over one 128-token x 256-channel K tile
// and the matching V tile, using only the CPU oracle
// (src/models/qwen/kvarn_reference.cpp). The input is synthetic by default:
// the KeyValue pattern of tests/models/qwen/hip/attention/
// kvarn_read_ops_test.cpp, rounded to FP16, and the table is labelled
// "synthetic input". An optional argument reads a raw little-endian FP16 dump
// (128x256 halves for K followed by 128x256 for V) so real tiles can be fed
// later. Both stores are read in the original (pre-rotation) domain.
//
// Variants:
//   0 current         Hadamard rotation, KvarnNormalizeTile, asymmetric RTN
//                     (K 6 bits, V 5 bits), FP16 metadata payload.
//   1 fp32-metadata   the same pipeline with FP32 scales/zero-points/scales
//                     instead of the FP16 payload encoding.
//   2 k-mean          the per-channel mean of K is subtracted before the
//                     rotation and added back after the inverse rotation.
//   3 sign-hadamard   a fixed pseudo-random +-1 pattern multiplies the rotated
//                     row (D*H*x) before quantization; the inverse undoes it.
//   4 bias-corrected  the current pipeline plus the quantization-noise bias
//                     correction of the assignment (see BiasCorrection).
//   5 q8_0            32-value blocks, symmetric, FP16 scale = amax/127, codes
//                     in [-127, 127] (the gpu-test baseline convention), K and
//                     V both 8 bits.
//   6 k8              K codes widened to 8 bits through the current
//                     normalization path; V stays at the current 5 bits.
//
// Metrics, both sides evaluated in the original domain:
//   score error    max/mean |q.k_hat - q.k_fp16| over a fixed seeded query set.
//   attention mass the softmax mass the quantized tile receives against a
//                  synthetic FP16 sink (MakeSink), versus the mass the FP16
//                  tile receives under the same queries.
//   weight change  max/mean |x_hat - x_fp16| of the dequantized K and V.
//
// Build (CPU only; the oracle needs ggml_dequant.cpp for Fp16ToFloat):
//   g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -I. <sources> -o <binary>
// where <sources> is
//   tools/qwen27b/kvarn_variant_compare.cpp
//   src/models/qwen/kvarn_reference.cpp
//   src/core/quant/ggml_dequant.cpp
// and <binary> is e.g. /tmp/kvarn_variant_compare.
// Run:
//   /tmp/kvarn_variant_compare                       # synthetic input
//   /tmp/kvarn_variant_compare tile.fp16             # real 128x256 K + 128x256
//   /tmp/kvarn_variant_compare tile.fp16 --queries q.f32
// A real K tile and its V tile concatenate into the tile argument because the
// reader consumes 128x256 halves for K and then 128x256 for V:
//   cat <stem>-k.f16 <stem>-v.f16 > tile.bin
// --queries reads group_size (6) rows of 256 little-endian float32 values --
// the query dump <stem>-q.f32 written by
// tests/models/qwen/hip/quality/kvarn_quality_eval --dump-kv L:T -- and uses
// them instead of the seeded synthetic queries.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/kvarn_reference.hpp"

namespace {

namespace ref = gufo::models::qwen;
namespace quant = gufo::quant;

constexpr std::size_t kTokens = ref::kKvarnTileTokens;      // 128
constexpr std::size_t kChannels = ref::kKvarnTileChannels;  // 256
constexpr std::size_t kElements = ref::kKvarnTileElements;  // 32768
constexpr std::size_t kGroups = ref::kKvarnValueGroups;     // 2
constexpr std::size_t kGroupChannels = ref::kKvarnValueGroupChannels;
constexpr std::size_t kQ8Block = 32;
constexpr std::size_t kQueryRows = 16;
constexpr std::uint64_t kQuerySeed = 0x5EED20261009ULL;
// Attention logits stay realistic: with the synthetic K amplitude of 0.3 and
// the 1/sqrt(256) head scale, N(0, 8) queries give logit sigma ~ 2.4.
constexpr double kQuerySigma = 8.0;
constexpr double kHeadScale = 1.0 / 16.0;  // 1 / sqrt(256)

constexpr std::uint8_t kCurrentKBits = 6;
constexpr std::uint8_t kCurrentVBits = 5;
constexpr std::uint8_t kKBits8 = 8;

/// The synthetic pattern of kvarn_read_ops_test.cpp.
float KeyValue(std::size_t position, std::size_t channel, bool is_value) {
  const float p = static_cast<float>(position);
  const float c = static_cast<float>(channel);
  return is_value ? 0.25F * std::cos(p * 0.05F + c * 0.017F)
                  : 0.3F * std::sin(p * 0.07F + c * 0.013F);
}

float Fp16Round(float value) {
  return quant::Fp16ToFloat(ref::KvarnFloatToFp16(value));
}

/// Element (token, channel) owning RTN parameter `p`, mirroring the oracle's
/// private RtnSetElement/RtnParam (they are not part of the public header).
std::size_t RtnParam(ref::KvarnRtnAxis axis, std::size_t token,
                     std::size_t channel) {
  if (axis == ref::KvarnRtnAxis::kChannel) {
    return channel;
  }
  return token * kGroups + channel / kGroupChannels;
}

std::size_t RtnSetElement(ref::KvarnRtnAxis axis, std::size_t p,
                          std::size_t n) {
  if (axis == ref::KvarnRtnAxis::kChannel) {
    return n * kChannels + p;
  }
  return (p / kGroups) * kChannels + (p % kGroups) * kGroupChannels + n;
}

std::size_t RtnSetSize(ref::KvarnRtnAxis axis) {
  return axis == ref::KvarnRtnAxis::kChannel ? kTokens : kGroupChannels;
}

/// The normalized Hadamard is symmetric, so the same call is the inverse.
void RotateRows(std::vector<float>& tile) {
  for (std::size_t i = 0; i < kTokens; ++i) {
    ref::KvarnFwhtHeadInPlace(
        std::span<float>(tile.data() + i * kChannels, kChannels));
  }
}

/// Fixed pseudo-random +-1 pattern for the sign-rotated Hadamard variant. It
/// is a pure function of the channel index and a fixed seed, so runs compare.
std::array<float, kChannels> MakeSignPattern() {
  std::array<float, kChannels> sign{};
  std::uint64_t state = 0x9E3779B97F4A7C15ULL;
  for (std::size_t j = 0; j < kChannels; ++j) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    sign[j] = (state & 1U) != 0 ? 1.0F : -1.0F;
  }
  return sign;
}

void ApplyForwardRotation(std::vector<float>& tile, bool signed_rotation,
                          const std::array<float, kChannels>& sign) {
  RotateRows(tile);
  if (signed_rotation) {
    for (std::size_t i = 0; i < kTokens; ++i) {
      for (std::size_t j = 0; j < kChannels; ++j) {
        tile[i * kChannels + j] *= sign[j];
      }
    }
  }
}

void ApplyInverseRotation(std::vector<float>& tile, bool signed_rotation,
                          const std::array<float, kChannels>& sign) {
  if (signed_rotation) {
    for (std::size_t i = 0; i < kTokens; ++i) {
      for (std::size_t j = 0; j < kChannels; ++j) {
        tile[i * kChannels + j] *= sign[j];
      }
    }
  }
  RotateRows(tile);
}

/// Asymmetric RTN at 8 bits. The oracle caps bits at kKvarnMaxBits (6) because
/// the sealed payload has only six bit planes, so the k8 variant re-runs the
/// same per-parameter min/max arithmetic here; nothing else is duplicated.
void EncodeAsymmetric8(std::span<const float> normalized,
                       ref::KvarnRtnAxis axis,
                       ref::KvarnTileQuantResult* result) {
  constexpr double kQmax = 255.0;  // (1 << 8) - 1
  const std::size_t set_size = RtnSetSize(axis);
  for (std::size_t p = 0; p < ref::kKvarnRtnParams; ++p) {
    double minimum = static_cast<double>(normalized[RtnSetElement(axis, p, 0)]);
    double maximum = minimum;
    for (std::size_t n = 0; n < set_size; ++n) {
      const double v =
          static_cast<double>(normalized[RtnSetElement(axis, p, n)]);
      minimum = std::min(minimum, v);
      maximum = std::max(maximum, v);
    }
    double scale = (maximum - minimum) / kQmax;
    if (!(scale > ref::kKvarnNormEpsilon)) {
      scale = ref::kKvarnNormEpsilon;
    }
    result->rtn_scale[p] = static_cast<float>(scale);
    result->rtn_zero[p] = static_cast<float>(minimum);
    for (std::size_t n = 0; n < set_size; ++n) {
      const std::size_t index = RtnSetElement(axis, p, n);
      float code =
          std::round((normalized[index] - result->rtn_zero[p]) / scale);
      code = std::clamp(code, 0.0F, static_cast<float>(kQmax));
      result->codes[index] = static_cast<std::uint8_t>(code);
    }
  }
  result->bits = kKBits8;
}

void EncodeAsymmetric(std::span<const float> normalized, ref::KvarnRtnAxis axis,
                      std::uint8_t bits, ref::KvarnTileQuantResult* result) {
  if (bits <= ref::kKvarnMaxBits) {
    ref::KvarnQuantizeAsymmetricRtn(normalized, axis, bits, result, nullptr);
    return;
  }
  EncodeAsymmetric8(normalized, axis, result);
}

/// Reconstructs one quantized tile in the rotated (tile) domain:
/// `(code * rtn_scale + rtn_zero) * channel_scale * token_scale`. With
/// `fp16_metadata` and a payload-representable width the decode goes through
/// the oracle's packed payload, so the current variant exercises exactly the
/// production read decode.
std::vector<float> DecodeResult(const ref::KvarnTileQuantResult& result,
                                ref::KvarnRtnAxis axis, std::uint8_t bits,
                                bool fp16_metadata) {
  std::vector<float> out(kElements, 0.0F);
  const bool packed = fp16_metadata && bits <= ref::kKvarnMaxBits;
  ref::KvarnTilePayload payload{};
  if (packed) {
    ref::KvarnPackTilePayload(result, &payload);
  }
  for (std::size_t t = 0; t < kTokens; ++t) {
    for (std::size_t c = 0; c < kChannels; ++c) {
      const std::size_t index = t * kChannels + c;
      const std::size_t p = RtnParam(axis, t, c);
      float value = 0.0F;
      float chn = result.norm_channel_scale[c];
      float tok = result.norm_token_scale[t];
      if (packed) {
        value = ref::KvarnDecodeTileElement(payload, axis, bits, t, c);
        chn = quant::Fp16ToFloat(payload.norm_channel_scale[c]);
        tok = quant::Fp16ToFloat(payload.norm_token_scale[t]);
      } else {
        float scale = result.rtn_scale[p];
        float zero = result.rtn_zero[p];
        if (fp16_metadata) {
          scale = Fp16Round(scale);
          zero = Fp16Round(zero);
          chn = Fp16Round(chn);
          tok = Fp16Round(tok);
        }
        value = static_cast<float>(result.codes[index]) * scale + zero;
      }
      out[index] = value * chn * tok;
    }
  }
  return out;
}

/// Normalize + encode + decode of one rotated tile; `values` is the
/// reconstruction in the rotated domain.
struct Pipeline {
  std::vector<float> values;
  ref::KvarnTileQuantResult result;
};

Pipeline NormalizeQuantize(std::span<const float> tile, ref::KvarnRtnAxis axis,
                           std::uint8_t bits, bool fp16_metadata) {
  Pipeline pipe;
  pipe.values.assign(kElements, 0.0F);
  const ref::KvarnQuantStatus status =
      ref::KvarnNormalizeTile(tile, pipe.values, &pipe.result, nullptr);
  if (status != ref::KvarnQuantStatus::kOk) {
    throw std::runtime_error("non-finite value in the tile");
  }
  EncodeAsymmetric(pipe.values, axis, bits, &pipe.result);
  pipe.values = DecodeResult(pipe.result, axis, bits, fp16_metadata);
  return pipe;
}

/// One plane's reconstruction in the original domain plus the K scales the
/// bias correction needs.
struct Plane {
  std::vector<float> original;
  std::array<float, kChannels> rtn_scale{};
  std::array<float, kChannels> chn_scale{};
  std::array<float, kTokens> tok_scale{};
};

Plane QuantizePlane(std::span<const float> input, ref::KvarnRtnAxis axis,
                    std::uint8_t bits, bool fp16_metadata, bool signed_rotation,
                    const std::array<float, kChannels>& sign) {
  std::vector<float> stored(input.begin(), input.end());
  ApplyForwardRotation(stored, signed_rotation, sign);
  Pipeline pipe = NormalizeQuantize(stored, axis, bits, fp16_metadata);
  ApplyInverseRotation(pipe.values, signed_rotation, sign);
  Plane plane;
  plane.original = std::move(pipe.values);
  plane.rtn_scale = pipe.result.rtn_scale;
  plane.chn_scale = pipe.result.norm_channel_scale;
  plane.tok_scale = pipe.result.norm_token_scale;
  return plane;
}

/// q8_0: symmetric 32-value blocks with an FP16 scale = amax/127 (the gpu-test
/// baseline convention), codes in [-127, 127].
std::vector<float> Q8Reconstruct(std::span<const float> rotated) {
  std::vector<float> out(rotated.begin(), rotated.end());
  for (std::size_t i = 0; i < kTokens; ++i) {
    float* row = out.data() + i * kChannels;
    for (std::size_t b = 0; b < kChannels; b += kQ8Block) {
      float amax = 0.0F;
      for (std::size_t j = 0; j < kQ8Block; ++j) {
        amax = std::max(amax, std::abs(row[b + j]));
      }
      const float scale = amax > 0.0F ? amax / 127.0F : 1.0F;
      const float stored = Fp16Round(scale);
      for (std::size_t j = 0; j < kQ8Block; ++j) {
        float code = std::round(row[b + j] / scale);
        code = std::clamp(code, -127.0F, 127.0F);
        row[b + j] = code * stored;
      }
    }
  }
  return out;
}

struct VariantOutput {
  std::string name;
  std::vector<float> k;  // original (pre-rotation) domain
  std::vector<float> v;
  bool bias = false;
  std::array<float, kChannels> rtn_scale{};
  std::array<float, kChannels> chn_scale{};
  std::array<float, kTokens> tok_scale{};
};

VariantOutput BuildVariant(int id, std::span<const float> k_in,
                           std::span<const float> v_in,
                           const std::array<float, kChannels>& sign) {
  VariantOutput out;
  switch (id) {
    case 0:
      out.name = "current";
      out.k =
          QuantizePlane(k_in, ref::KvarnRtnAxis::kChannel, kCurrentKBits,
                        /*fp16_metadata=*/true, /*signed_rotation=*/false, sign)
              .original;
      out.v = QuantizePlane(v_in, ref::KvarnRtnAxis::kTokenGroup, kCurrentVBits,
                            /*fp16_metadata=*/true,
                            /*signed_rotation=*/false, sign)
                  .original;
      break;
    case 1:
      out.name = "fp32-metadata";
      out.k = QuantizePlane(k_in, ref::KvarnRtnAxis::kChannel, kCurrentKBits,
                            /*fp16_metadata=*/false, /*signed_rotation=*/false,
                            sign)
                  .original;
      out.v = QuantizePlane(v_in, ref::KvarnRtnAxis::kTokenGroup, kCurrentVBits,
                            /*fp16_metadata=*/false,
                            /*signed_rotation=*/false, sign)
                  .original;
      break;
    case 2: {
      out.name = "k-mean";
      std::array<float, kChannels> mean{};
      for (std::size_t t = 0; t < kTokens; ++t) {
        for (std::size_t c = 0; c < kChannels; ++c) {
          mean[c] += k_in[t * kChannels + c];
        }
      }
      std::vector<float> centered(kElements);
      for (std::size_t c = 0; c < kChannels; ++c) {
        mean[c] /= static_cast<float>(kTokens);
      }
      for (std::size_t t = 0; t < kTokens; ++t) {
        for (std::size_t c = 0; c < kChannels; ++c) {
          centered[t * kChannels + c] = k_in[t * kChannels + c] - mean[c];
        }
      }
      Plane k_plane = QuantizePlane(
          centered, ref::KvarnRtnAxis::kChannel, kCurrentKBits,
          /*fp16_metadata=*/true, /*signed_rotation=*/false, sign);
      for (std::size_t t = 0; t < kTokens; ++t) {
        for (std::size_t c = 0; c < kChannels; ++c) {
          k_plane.original[t * kChannels + c] += mean[c];
        }
      }
      out.k = std::move(k_plane.original);
      out.v = QuantizePlane(v_in, ref::KvarnRtnAxis::kTokenGroup, kCurrentVBits,
                            /*fp16_metadata=*/true,
                            /*signed_rotation=*/false, sign)
                  .original;
      break;
    }
    case 3:
      out.name = "sign-hadamard";
      out.k =
          QuantizePlane(k_in, ref::KvarnRtnAxis::kChannel, kCurrentKBits,
                        /*fp16_metadata=*/true, /*signed_rotation=*/true, sign)
              .original;
      out.v = QuantizePlane(v_in, ref::KvarnRtnAxis::kTokenGroup, kCurrentVBits,
                            /*fp16_metadata=*/true,
                            /*signed_rotation=*/true, sign)
                  .original;
      break;
    case 4: {
      out.name = "bias-corrected";
      Plane k_plane = QuantizePlane(
          k_in, ref::KvarnRtnAxis::kChannel, kCurrentKBits,
          /*fp16_metadata=*/true, /*signed_rotation=*/false, sign);
      out.bias = true;
      out.rtn_scale = k_plane.rtn_scale;
      out.chn_scale = k_plane.chn_scale;
      out.tok_scale = k_plane.tok_scale;
      out.k = std::move(k_plane.original);
      out.v = QuantizePlane(v_in, ref::KvarnRtnAxis::kTokenGroup, kCurrentVBits,
                            /*fp16_metadata=*/true,
                            /*signed_rotation=*/false, sign)
                  .original;
      break;
    }
    case 5: {
      out.name = "q8_0";
      std::vector<float> stored_k(k_in.begin(), k_in.end());
      RotateRows(stored_k);
      out.k = Q8Reconstruct(stored_k);
      ApplyInverseRotation(out.k, false, sign);
      std::vector<float> stored_v(v_in.begin(), v_in.end());
      RotateRows(stored_v);
      out.v = Q8Reconstruct(stored_v);
      ApplyInverseRotation(out.v, false, sign);
      break;
    }
    case 6: {
      out.name = "k8";
      Plane k_plane = QuantizePlane(k_in, ref::KvarnRtnAxis::kChannel, kKBits8,
                                    /*fp16_metadata=*/true,
                                    /*signed_rotation=*/false, sign);
      out.k = std::move(k_plane.original);
      out.v = QuantizePlane(v_in, ref::KvarnRtnAxis::kTokenGroup, kCurrentVBits,
                            /*fp16_metadata=*/true,
                            /*signed_rotation=*/false, sign)
                  .original;
      break;
    }
    default:
      break;
  }
  return out;
}

/// One synthetic FP16 sink block of 128 rows (positions 0..127), the
/// non-quantized context the attention-mass metric competes against.
std::vector<float> MakeSink() {
  std::vector<float> sink(kTokens * kChannels);
  for (std::size_t p = 0; p < kTokens; ++p) {
    for (std::size_t c = 0; c < kChannels; ++c) {
      sink[p * kChannels + c] = Fp16Round(KeyValue(p, c, false));
    }
  }
  return sink;
}

std::vector<std::vector<float>> MakeQueries(std::size_t rows) {
  std::mt19937_64 rng(kQuerySeed);
  std::normal_distribution<double> normal(0.0, kQuerySigma);
  std::vector<std::vector<float>> queries(rows, std::vector<float>(kChannels));
  for (std::vector<float>& q : queries) {
    for (float& value : q) {
      value = static_cast<float>(normal(rng));
    }
  }
  return queries;
}

/// The raw float32 query dump of the gpu-test harness: `group_size` rows of
/// kChannels little-endian values. Any whole number of rows is accepted.
std::vector<std::vector<float>> ReadQueries(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open query dump " + path);
  }
  in.seekg(0, std::ios::end);
  const std::streamoff bytes = in.tellg();
  in.seekg(0, std::ios::beg);
  const std::streamoff row_bytes =
      static_cast<std::streamoff>(kChannels * sizeof(float));
  if (bytes <= 0 || bytes % row_bytes != 0) {
    throw std::runtime_error(
        "query dump is not a whole number of 256-value "
        "float32 rows: " +
        path);
  }
  const std::size_t rows = static_cast<std::size_t>(bytes / row_bytes);
  std::vector<std::vector<float>> queries(rows, std::vector<float>(kChannels));
  for (std::vector<float>& q : queries) {
    in.read(reinterpret_cast<char*>(q.data()),
            static_cast<std::streamsize>(row_bytes));
    if (in.gcount() != static_cast<std::streamsize>(row_bytes)) {
      throw std::runtime_error("short read from query dump " + path);
    }
  }
  return queries;
}

double KeyScore(const std::vector<float>& q, const float* key) {
  double sum = 0.0;
  for (std::size_t c = 0; c < kChannels; ++c) {
    sum += static_cast<double>(q[c]) * static_cast<double>(key[c]);
  }
  return sum * kHeadScale;
}

/// Softmax mass on the 128 quantized tile rows of a context whose logits are
/// the 128 FP16 sink rows followed by the 128 tile rows.
double MassOnTile(const std::vector<double>& logit) {
  double maximum = *std::max_element(logit.begin(), logit.end());
  double total = 0.0;
  double tile_sum = 0.0;
  for (std::size_t i = 0; i < logit.size(); ++i) {
    const double weight = std::exp(logit[i] - maximum);
    total += weight;
    if (i >= kTokens) {
      tile_sum += weight;
    }
  }
  return tile_sum / total;
}

/// The assignment's bias term, applied to the score because a rounding-noise
/// variance is a logit-level (Jensen) bias:
///   delta(t) = 1/2 * sum_c (rtn_scale_c * chn_scale_c * tok_scale_t)^2
///                     * (q_rot_c / sqrt(256))^2 / 12
/// Var(e_c) = (rtn_scale_c * chn_scale_c * tok_scale_t)^2 / 12 is the variance
/// of the dequantized rotated element and q_rot is the query in that same
/// domain (the codes are stored rotated); the Hadamard is orthonormal, so the
/// corrected score matches the device's q_rot . k_hat_rot.
double BiasCorrection(const std::vector<float>& q_rot, std::size_t token,
                      const VariantOutput& variant) {
  double sum = 0.0;
  for (std::size_t c = 0; c < kChannels; ++c) {
    const double scale = static_cast<double>(variant.rtn_scale[c]) *
                         static_cast<double>(variant.chn_scale[c]) *
                         static_cast<double>(variant.tok_scale[token]);
    const double q = static_cast<double>(q_rot[c]) * kHeadScale;
    sum += scale * scale * q * q;
  }
  return 0.5 * sum / 12.0;
}

struct Metrics {
  double score_max = 0.0;
  double score_mean = 0.0;
  double mass_fp16 = 0.0;
  double mass_variant = 0.0;
  double mass_max = 0.0;
  double dk_max = 0.0;
  double dk_mean = 0.0;
  double dv_max = 0.0;
  double dv_mean = 0.0;
};

Metrics Evaluate(const VariantOutput& variant, const std::vector<float>& k_fp,
                 const std::vector<float>& v_fp, const std::vector<float>& sink,
                 const std::vector<std::vector<float>>& queries) {
  Metrics m;
  double score_sum = 0.0;
  std::size_t score_count = 0;
  double mass_variant_sum = 0.0;
  double mass_fp16_sum = 0.0;
  for (const std::vector<float>& q : queries) {
    std::vector<float> q_rot;
    if (variant.bias) {
      q_rot = q;
      ref::KvarnFwhtHeadInPlace(q_rot);
    }
    std::vector<double> logit_fp(2 * kTokens);
    std::vector<double> logit_variant(2 * kTokens);
    for (std::size_t r = 0; r < kTokens; ++r) {
      const double score = KeyScore(q, sink.data() + r * kChannels);
      logit_fp[r] = score;
      logit_variant[r] = score;
    }
    for (std::size_t t = 0; t < kTokens; ++t) {
      const double base = KeyScore(q, k_fp.data() + t * kChannels);
      double value = KeyScore(q, variant.k.data() + t * kChannels);
      if (variant.bias) {
        value -= BiasCorrection(q_rot, t, variant);
      }
      logit_fp[kTokens + t] = base;
      logit_variant[kTokens + t] = value;
      const double error = std::abs(value - base);
      m.score_max = std::max(m.score_max, error);
      score_sum += error;
      ++score_count;
    }
    const double mass_variant = MassOnTile(logit_variant);
    const double mass_fp16 = MassOnTile(logit_fp);
    mass_variant_sum += mass_variant;
    mass_fp16_sum += mass_fp16;
    m.mass_max = std::max(m.mass_max, std::abs(mass_variant - mass_fp16));
  }
  m.score_mean = score_sum / static_cast<double>(score_count);
  m.mass_variant = mass_variant_sum / static_cast<double>(queries.size());
  m.mass_fp16 = mass_fp16_sum / static_cast<double>(queries.size());
  double dk_sum = 0.0;
  double dv_sum = 0.0;
  for (std::size_t i = 0; i < kElements; ++i) {
    const double dk = std::abs(static_cast<double>(variant.k[i]) -
                               static_cast<double>(k_fp[i]));
    const double dv = std::abs(static_cast<double>(variant.v[i]) -
                               static_cast<double>(v_fp[i]));
    m.dk_max = std::max(m.dk_max, dk);
    m.dv_max = std::max(m.dv_max, dv);
    dk_sum += dk;
    dv_sum += dv;
  }
  m.dk_mean = dk_sum / static_cast<double>(kElements);
  m.dv_mean = dv_sum / static_cast<double>(kElements);
  return m;
}

std::vector<float> ReadTile(std::ifstream& in) {
  std::vector<std::uint16_t> halves(kElements);
  in.read(reinterpret_cast<char*>(halves.data()),
          static_cast<std::streamsize>(kElements * sizeof(std::uint16_t)));
  if (in.gcount() !=
      static_cast<std::streamsize>(kElements * sizeof(std::uint16_t))) {
    throw std::runtime_error("input dump is shorter than one 128x256 tile");
  }
  std::vector<float> out(kElements);
  for (std::size_t i = 0; i < kElements; ++i) {
    out[i] = quant::Fp16ToFloat(halves[i]);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string tile_path;
  std::string query_path;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--queries") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "--queries needs a path\n");
        return 1;
      }
      query_path = argv[++i];
    } else if (tile_path.empty()) {
      tile_path = arg;
    } else {
      std::fprintf(stderr, "unexpected argument %s\n", argv[i]);
      return 1;
    }
  }

  try {
    std::vector<float> k_in;
    std::vector<float> v_in;
    std::string source = "synthetic input (KeyValue pattern, FP16)";
    if (!tile_path.empty()) {
      std::ifstream in(tile_path, std::ios::binary);
      if (!in) {
        std::fprintf(stderr, "cannot open %s\n", tile_path.c_str());
        return 1;
      }
      k_in = ReadTile(in);
      v_in = ReadTile(in);
      source = "FP16 tile dump " + tile_path;
    } else {
      k_in.resize(kElements);
      v_in.resize(kElements);
      for (std::size_t t = 0; t < kTokens; ++t) {
        for (std::size_t c = 0; c < kChannels; ++c) {
          k_in[t * kChannels + c] = Fp16Round(KeyValue(kTokens + t, c, false));
          v_in[t * kChannels + c] = Fp16Round(KeyValue(kTokens + t, c, true));
        }
      }
    }

    std::vector<std::vector<float>> queries;
    std::string query_source;
    if (!query_path.empty()) {
      queries = ReadQueries(query_path);
      query_source = "file " + query_path;
    } else {
      queries = MakeQueries(kQueryRows);
      char label[64] = {};
      std::snprintf(label, sizeof(label), "seeded synthetic (seed %llu)",
                    static_cast<unsigned long long>(kQuerySeed));
      query_source = label;
    }

    const std::array<float, kChannels> sign = MakeSignPattern();
    const std::vector<float> sink = MakeSink();

    std::fprintf(stdout,
                 "# KVarN tile-quantization variants (design D7 phase 1a)\n");
    std::fprintf(stdout, "# input: %s; tile %zux%zu; queries %zu (%s)\n",
                 source.c_str(), kTokens, kChannels, queries.size(),
                 query_source.c_str());
    std::fprintf(stdout, "%-18s %10s %11s %10s %10s %10s %9s %9s %9s %9s\n",
                 "variant", "score_max", "score_mean", "mass16", "mass_var",
                 "dmass_max", "dk_max", "dk_mean", "dv_max", "dv_mean");

    std::vector<std::string> names;
    std::vector<double> score_means;
    for (int id = 0; id < 7; ++id) {
      const VariantOutput variant = BuildVariant(id, k_in, v_in, sign);
      const Metrics m = Evaluate(variant, k_in, v_in, sink, queries);
      std::fprintf(stdout,
                   "%-18s %10.4g %11.4g %10.4g %10.4g %10.4g %9.4g %9.4g "
                   "%9.4g %9.4g\n",
                   variant.name.c_str(), m.score_max, m.score_mean, m.mass_fp16,
                   m.mass_variant, m.mass_max, m.dk_max, m.dk_mean, m.dv_max,
                   m.dv_mean);
      names.push_back(variant.name);
      score_means.push_back(m.score_mean);
    }

    std::vector<std::size_t> order(names.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
      order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) {
                       return score_means[a] < score_means[b];
                     });
    std::fprintf(stdout, "\nranking by score error (mean):");
    for (std::size_t i = 0; i < order.size(); ++i) {
      std::fprintf(stdout, " %zu)%s=%.4g", i + 1, names[order[i]].c_str(),
                   score_means[order[i]]);
    }
    std::fprintf(stdout, "\n");
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
  return 0;
}
