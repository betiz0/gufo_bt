// Graph identity must separate production execution from operator references.
#include "src/models/qwen/hip/execution_policy.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>

#include "src/models/qwen/kvarn_reference.hpp"

int main() {
  using namespace gufo::hip;
  const auto production = QwenExecutionPolicy::Production();
  assert(production.UsesFp16AttentionKv());
  // The FP16 fingerprint is unchanged by the KVarN bits (0x101 = fuse | FP16).
  assert(production.Fingerprint() == 0x101U);
  assert(!production.UsesBf16RecurrentState());
  assert(QwenRecurrentStateElementBytes(QwenRecurrentStateStorage::kFp32) ==
         sizeof(float));
  assert(QwenRecurrentStateElementBytes(QwenRecurrentStateStorage::kBf16) ==
         sizeof(std::uint16_t));

  auto fp32_kv = production;
  fp32_kv.kv_cache_storage = QwenKvCacheStorage::kFp32;
  auto bf16_recurrent = production;
  bf16_recurrent.recurrent_state_storage = QwenRecurrentStateStorage::kBf16;
  auto separate_qk_norm = production;
  separate_qk_norm.fuse_qk_norm_rope_kv = false;
  assert(fp32_kv.Fingerprint() != production.Fingerprint());
  assert(bf16_recurrent.Fingerprint() != production.Fingerprint());
  assert(separate_qk_norm.Fingerprint() != production.Fingerprint());

  // KVarN presets (design D1/D11): the FP16 fingerprint above is unchanged,
  // every preset and tail length is distinct, and the shape gate rejects any
  // other geometry.
  auto kvarn44 = production;
  assert(ResolveQwenKvCache("kvarn-k4v4", 0, &kvarn44));
  assert(kvarn44.UsesKvarn() && !kvarn44.UsesFp16AttentionKv());
  assert(kvarn44.kvarn.k_bits == 4 && kvarn44.kvarn.v_bits == 4);
  assert(kvarn44.kvarn.tail_length == 0);
  assert(kvarn44.kvarn.scale == QwenKvarnScale::kFp16);
  assert(kvarn44.Fingerprint() != production.Fingerprint());
  assert(kvarn44.Fingerprint() != fp32_kv.Fingerprint());
  assert(QwenKvarnTailRows(kvarn44.kvarn) == kQwenKvarnW);
  assert(QwenKvarnTailRingRows(kvarn44.kvarn) ==
         kQwenKvarnW + kQwenKvarnSinkRows + kQwenKvarnW);

  auto kvarn44_tail = production;
  assert(ResolveQwenKvCache("kvarn-k4v4", 2048, &kvarn44_tail));
  assert(kvarn44_tail.kvarn.tail_length == 2048);
  assert(kvarn44_tail.Fingerprint() != kvarn44.Fingerprint());
  assert(QwenKvarnTailRows(kvarn44_tail.kvarn) == 2048);

  auto kvarn54 = production;
  auto kvarn65 = production;
  assert(ResolveQwenKvCache("kvarn-k5v4", 0, &kvarn54));
  assert(ResolveQwenKvCache("kvarn-k6v5", 0, &kvarn65));
  assert(kvarn65.kvarn.k_bits == 6 && kvarn65.kvarn.v_bits == 5);
  assert(kvarn54.Fingerprint() != kvarn44.Fingerprint());
  assert(kvarn65.Fingerprint() != kvarn54.Fingerprint());

  auto with_tail = production;
  assert(!ResolveQwenKvCache("fp16", 512, &with_tail));
  assert(!ResolveQwenKvCache("kvarn-k9v9", 0, &with_tail));
  assert(!ResolveQwenKvCache("kvarn-k4v4", 255, &with_tail));
  assert(ResolveQwenKvCache("kvarn-k4v4", 256, &with_tail));

  assert(QwenKvarnSupportsShape(24, 4, 256));
  assert(!QwenKvarnSupportsShape(24, 4, 128));
  assert(QwenKvarnShapeError(production, 8, 2, 128).empty());
  assert(QwenKvarnShapeError(kvarn44, 24, 4, 256).empty());
  assert(!QwenKvarnShapeError(kvarn44, 8, 2, 128).empty());

  // The region sizes and the bpv report come from the same formula.
  const auto regions = gufo::models::qwen::ComputeKvarnRegions(
      4, 4, kQwenKvarnW, kQwenKvarnW, 262144, 16, 4);
  const auto estimate = gufo::models::qwen::EstimateKvarnMemory(
      4, 4, kQwenKvarnW, kQwenKvarnW, 262144, 16, 4);
  assert(regions.tile_count == 2047);
  const double values = 2.0 * 262144.0 * 256.0 * 16.0 * 4.0;
  const double total_bpv =
      static_cast<double>(regions.total_bytes) * 8.0 / values;
  assert(total_bpv > estimate.total_bpv - 1e-9 &&
         total_bpv < estimate.total_bpv + 1e-9);
  assert(total_bpv <= 4.6);

  // Acceptance bpv of design D1/D5: k4v4 body <= 4.5 and total <= 4.6 at every
  // tail, and the quality-first k6v5 total <= 6.4 at L = 2048.
  const auto bpv = [](std::uint8_t k_bits, std::uint8_t v_bits,
                      std::size_t tail) {
    return gufo::models::qwen::EstimateKvarnMemory(k_bits, v_bits, tail,
                                                   kQwenKvarnW, 262144, 16, 4);
  };
  for (std::size_t tail : {std::size_t{kQwenKvarnW}, std::size_t{256},
                           std::size_t{1024}, std::size_t{2048}}) {
    const auto memory = bpv(4, 4, tail);
    assert(memory.body_bpv <= 4.5);
    assert(memory.total_bpv <= 4.6);
  }
  assert(bpv(6, 5, 2048).total_bpv <= 6.4);

  const auto decode = ResolveQwenLayerRouteWithReasons(
      production, QwenExecutionMode::kDecode, true);
  const auto prefill = ResolveQwenLayerRouteWithReasons(
      production, QwenExecutionMode::kPrefill, true);
  const auto ssm = ResolveQwenLayerRouteWithReasons(
      production, QwenExecutionMode::kDecode, false);
  assert(decode.plan.fuse_qk_norm_rope_kv);
  assert(prefill.plan.fuse_qk_norm_rope_kv);
  assert(!ssm.plan.fuse_qk_norm_rope_kv);
  assert(decode.rejected == QwenRouteRejection::kNone);
  assert(HasQwenRouteRejection(
      ssm.rejected, QwenRouteRejection::kQkNormRopeKvRequiresAttention));
  assert(decode.plan.Fingerprint() != prefill.plan.Fingerprint());
  assert(decode.plan.Fingerprint() != ssm.plan.Fingerprint());

  const auto rejections = ResolveQwenGraphRejections(false, true, false);
  assert(HasQwenGraphRejection(rejections,
                               QwenGraphRejection::kLogitsNotRequested));
  assert(HasQwenGraphRejection(rejections,
                               QwenGraphRejection::kSplitKAttentionRequired));
  assert(HasQwenGraphRejection(rejections, QwenGraphRejection::kGraphDisabled));
  assert(ResolveQwenGraphRejections(true, false, true) ==
         QwenGraphRejection::kNone);

  const auto seed = BeginQwenGraphWorkloadIdentity();
  const auto first =
      ExtendQwenGraphWorkloadIdentity(seed, decode.plan.Fingerprint());
  assert(first ==
         ExtendQwenGraphWorkloadIdentity(seed, decode.plan.Fingerprint()));
  assert(first !=
         ExtendQwenGraphWorkloadIdentity(seed, prefill.plan.Fingerprint()));
  std::cout << "Qwen graph identity and reference policy checks passed.\n";
}
