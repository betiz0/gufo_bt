# Qwen3.8 27B experiments

| Experiment | Decision / qualification |
| --- | --- |
| KVarN KV-cache quantization quality probe (phase 0) | Retained for phase 1: G0 passes under design D17. Teacher-forced KL against the FP16-KV teacher over 4K/32K/128K (3 corpora, 256 positions each) puts `kvarn-k6v5` at tail 2048 at worst q8→q4 position 0.143 (≤ 0.15) and `kvarn-k4v4` at tail 256 at 0.971 (≤ 1); `kvarn-k5v4` fits the memory-first budget with no tail. FP16 scales, no overflow and no non-finite tiles. Probe measurements (design D13); the production read path is phase 1. Reproduce: `python3 tools/qwen27b/kvarn_quality.py report --label g0-tail --gate`. |
| Bounded prefill submission for cancellation | Retained: Q4/Q8 AR resume in 302/412 ms after a prefill disconnect; DFlash2 in 214/281 ms. Exact image/snapshot logits and seeded replay; matched 2,504-token C1 prefill times remain within 1.7% across all four modes. Two reusable events, no tensor allocation or arithmetic change. |
| Quantized verification row groups | Retained per shape; scalar FP32 bits, full target logits and private acceptance/RNG must match. |
| Shared DFlash2 body/context injection | Retained across requests; independent attention, convolution, history and selector state. |
| Partial verification after rejection | Retained with the complete original proposal and unchanged consumed-prefix feedback. |
| BF16 draft gate/up reuse and tiled argmax | Retained; complete head, finite filtering and lowest-ID ties; no extra persistent buffer. |
| Large-chunk KV packing | Retained at shallow and long depths in idle FFN scratch with bounded head groups; exact attention output/log-sum-exp, cache bytes unchanged. |
| BF16 target projection reduction | Fixed per-row FP32 order retained for chunk/cache/continued-image equivalence. |
| Register-cached vision softmax | Retained; byte-identical embeddings with the shared Q4/Q8 projector, unchanged reduction order and memory allocation. |
| 4096-patch vision attention tiles | Retained; byte-identical full embeddings, lower latency and 24 MiB less attention scratch. |
| Alternate tiles/waves/pipeline depths (2026-09-19) | Rejected: no release throughput improvement. |
| Dynamic verification chunk/controller alternatives | No new default retained; seeded private-acceptance policy remains. |
| Parallel mapped-weight reads and larger DFlash packing chunks | Retained: faster cold startup, unchanged encoded weights. |
| Compact active/saved recurrence and valid-prefix KV | Retained: unused attention-layer state and future KV rows excluded; Q4/Q8 sampling, rollback, image and disk replay pass. |
| Verification queries grouped by KV partition | Retained: unchanged arithmetic and storage; eight-token attention 2.95× faster at 32K and 3.66× at 64K. Matched d32K C1 HTTP TG improves 15.7% on Q4_K_XL and 20.2% on Q8_K_XL with Q4 DFlash2; shallow TG and PP remain comparable. |
| Split-K two-position KV prefetch and DPP score reductions | Rejected: small single-row component gains did not consistently help eight-row verification. |
| Shallow attention paired KV loads and DPP reduction | Retained: byte-exact; 12.6% less attention time in the complete AR profile. Matched d0 pp2048/tg128 controls improve TG by 1.4–2.6% across Q4/Q8 AR and Q4 DFlash2, with unchanged outputs and acceptance; PP remains comparable. No extra allocation. |
| Parallel attention from 128 tokens | Retained for Q4 C1 AR and Q4 DFlash2: greedy outputs match; lower error against FP64 in all 12 component controls. Other targets/concurrency remain to qualify. |
| Wider shallow pipeline, removed broadcast and separate score/value passes | Rejected: worse cold-KV or eight-row costs than the retained two-position pipeline. |
| Additional GEMV format specialization | Not promoted: Q8 cold components unchanged; Q6 gains at most 3% in the cold component test. |
| Paired-row activation reuse and shared input sums | Not promoted: dominant Q5_K cold components were unchanged or slower. |
| Two-block Q5 weight prefetch | Rejected: byte-exact, but cold-weight throughput is 8–12% lower. |
| Seven-row Q5 token scheduling and output-row groups | Not promoted: byte-exact across three projection shapes and input scales, but cold-weight gains are at most 1.5%; several variants regress. |
| Compact seven-row Q5 activation staging | Rejected: byte-exact and lower register/LDS use, but cold-weight gains are under 1% on gate/up; down and attention projections regress. Token-step and wave-count variants do not recover a useful gain. |
| Seven-row Q5 row-first FMA scheduling | Rejected: byte-exact on three cold-weight projection shapes; most variants are 1–3% slower. |
| Seven-row Q5 wave32 and row-first scheduling | Rejected: byte-exact on three cold-weight shapes and three input scales; no consistent useful gain over wave64. |
| Fused SSM format specialization and larger output-row groups | Not promoted: no useful cold-weight gain; larger groups were slower despite byte-exact outputs. |
| 64–256 attention partitions | Rejected: no improvement at 32K, with different FP32 rounding. |
| Split-attention graph replay | Rejected: matched Q4 AR C1 d0 remains 11.90 tok/s. The existing launch path is already 97.5% GPU-busy. |
| Graph identity includes target feature taps | Correctness fix: changing captured layers or their order must replace the graph's feature-copy operations. |
| Residual/RMSNorm fusion and fewer reduction barriers | Rejected: byte-exact component gains did not survive the full model. Q4 AR C1 remains 11.90 tok/s and profile GPU time slightly increases. |
| Multiple attention heads per thread block | Not promoted: byte-exact through 32K, but cold-KV gains are at most 1.5%. |
| Shared four-position KV tile across six query heads | Retained: byte-exact; Q4 AR C1 d32K TG improves 2.9%, with shallow TG and PP retained. Existing scalar/batched and full-logit replay checks pass. |
| Shared KV tile for verification rows | Retained: byte-exact; Q4 DFlash2 C1 d32K TG improves 4.4%. Shallow TG, PP, output hashes and acceptance counts are retained. |
| Two 16-lane verification heads per wave | Retained for multi-row attention: preserves both original lane partials and their reduction tree. Q4 DFlash2 C1 d32K TG improves 5.8%, with shallow TG, PP, greedy output and acceptance retained. Scalar AR keeps 32 lanes. |
| Draft-attention value prefetch and paired query heads | Retained: 32-value prefetch preserves the original FMA order; wider blocks share K/V across two query heads. Byte-exact through 32K; C1 d0/d32K TG improves 0.8–1.0%, with unchanged outputs/acceptance and comparable PP. No persistent allocation. |
| Wider draft prefetch and four query heads per block | Rejected: 64/128-value prefetch and four-head sharing lose to the retained 32-value/two-head layout. |
| Context-aware Q4 draft cost and censored full acceptance | Retained: measured attention growth improves C1 d32K TG by 3.0%; shallow TG, PP, greedy AR agreement and repetitive throughput are retained. Choices remain deterministic from private history and position. |
| Width-weighted acceptance feedback | Retained: removes block-width bias under the controller's geometric model. Q4 C1 d0 TG improves 7.7%, with modest d32K/d64K gains, matched PP and exact greedy/cached sampled replay. Repetition retains full acceptance. |
| Fixed three-proposal default | Rejected: helps the ordinary d32K prompt but slows d0. The adaptive controller remains the default. |
| Context cost without handling saturated acceptance | Rejected: improves ordinary d32K output but slows perfect-acceptance repetition; saturation must allow wider-block probes. |
| Eight lanes per verification head | Not promoted: higher register use and less consistent gains across draft widths than the 16-lane variant. |
| KV sharing between adjacent verification rows | Rejected: byte-exact, but slower at 2K/32K than independent row tiles. |
| Eight-position verification tiles and LDS-only barriers | Not promoted: byte-exact, but gains are small or mixed across 3/7/8 rows, with shallow regressions. |
| Next-tile KV prefetch | Rejected: 144 byte-exact controls through 64K; scalar AR is flat and verification is slower. |
| Query-row workgroup ordering | Not promoted: 144 byte-exact controls; component gains are at most 1.5%, and padded row groups regress. |
| FP32 KV staging in shared memory | Rejected: 192 byte-exact controls; sharing conversions does not offset the extra shared-memory cost. |
| Distributed encoded K/V loads | Rejected: 144 byte-exact partial/output comparisons through 64K; plane-, token- and vector-interleaved loads are slower. |
| Sixteen-row Q5 activation staging in smaller token groups | Rejected: byte-exact on three cold-weight shapes; halving or quartering shared memory adds more work than it saves. |
| Sixteen-row Q5 row-first FMA operand schedules | Rejected: byte-exact on three cold-weight shapes; no throughput gain. |
| Sixteen-row Q5 packed or lane-broadcast headers and scalar scales | Rejected: byte-exact on three cold-weight shapes, but slower than the existing shared-scale decoder. |
| Unsigned Q4/Q5 coefficient conversion | Rejected: byte-exact on three sixteen-row cold-weight shapes, but no useful speed gain or register reduction. |
| Sixteen-row Q5 wave32 | Rejected: byte-exact on three cold-weight shapes, but substantially slower than wave64; the smaller row group also spills registers. |
| Native 32-row Q5 tiles | Rejected: byte-exact on three cold-weight shapes, but slower than paired 16-row tiles across one-to-four output rows per lane; wider accumulators also spill or use private memory. |
| Compact sixteen-row Q5 staging and register activation sums | Rejected: byte-exact on three cold-weight shapes; smaller LDS allocations and alternate bank layouts do not offset the extra work. |
| Batched DFlash2 selector | Retained: exact private token chains/probabilities, two launches per saturated C4 cycle instead of 56, and reused FFN scratch. The focused release pair gains 1.3% TG; C1 PP/TG and output are retained. |
| Six fixed drafts at Q4 C2 | Rejected on perfect-acceptance repetition: exact output, but slower than adaptive. |
| Paired Q4 greedy verification costs | Retained: accounts for the projection cost jump above eight rows. C2 generation improves 20.0% on the Italian/Chinese pair, 12.1% on the pangram/train pair and 21.3% at d32K; repetition, AR equality, C1 PP/TG and private sampled replay are retained. |
| Four-request Q4 greedy verification costs | Retained: accounts for the projection jump above sixteen rows. Ordinary C4 generation improves 26.9% and 13.0%; repetition, d32K PP/TG, AR output and private sampled replay are retained. C1 DFlash2 remains unchanged. |
| Joint C4 draft allocation | Rejected: the Italian/Chinese control retains AR output but proposes 603 tokens for 297 accepted, versus 527 previously, and runs slower. Private sampled/RNG/snapshot tests pass; no production change retained. |
| Two interleaved ten-row Q5 groups | Rejected: byte-exact with production compiler settings, but slower than the existing sixteen-plus-four split on cold-weight FFN projections. |
| IQ4_XS staging both phases and compact shared memory | Not promoted: byte-exact on cold-weight FFN projections, but no consistent gain; the down projection regresses. |
| Exact FP16 integer coefficients with mixed FP32 FMA | Rejected: 2,097,152 instruction controls and both 32-row Q5 FFN projections are byte-exact, but packing coefficients and using mixed FMA runs 4–6% slower. Activations and accumulation remain FP32; no production change. |
| Draft blocks wider than seven proposals | Not attempted: the cached Q4 DFlash2 artifact declares an eight-token block, including the anchor. Its metadata limit is retained. |
| Six-request Q4 greedy verification costs | Retained: ordinary C6 controls improve 6.3% and 16.5%, with a focused 4.3% d32K gain. PP, full acceptance, AR output and private sampled replay are retained; saturated history keeps the full-block probe. |
| Eight-request Q4 greedy verification costs | Retained: ordinary C8 controls improve 10.4% and 13.6%; repetition, d32K, PP and AR output are retained. Other cohort/mode decisions and sampled replay are unchanged. AR remains faster on the harder pair, so profitable fallback still needs work. |
| Three fixed drafts at Q4 concurrency | Rejected as a default: helps C2 low acceptance but slows the higher-acceptance pair and C4 repetition. |
| Two/three query heads per shared KV tile | Rejected: byte-exact, but slower than six-head sharing at 2K and 32K. |
| Wave64 scalar projections and uniform row addresses | Not promoted: cold-weight gains are flat or mixed; some Q4/Q6 shapes regress. |
| Copied weights, including a complete GGUF allocation | Not promoted: the complete-copy control has 3–8% lower cold-weight throughput than mapped weights, with identical outputs. |
| Exact zero-exponent fast path, scalar score broadcast and LDS-only barriers | Not promoted: byte-exact, but no useful gain after shared KV staging. |
| Register-cached decode RMSNorm, tiled C1 argmax and snapshot sizing | Retained with explicit fused square accumulation after expanded tests caught a contraction difference. Existing exact argmax and no logits download merely to count them; full-logit replay now passes. |
| Streaming Q4/Q5 payload loads | Rejected: byte-exact cold-kernel gains regress full-model AR by about 1%. |
| Q6 format specialization and paired/split SSM projections | Not promoted: byte-exact, but negligible gains or regressions; split SSM adds launches. |
| Precomputed affine input sums | Rejected: byte-exact, but no consistent cold-projection benefit after including the preparation pass. |
| Gate/up input sharing, including four simultaneous dots | Not retained: exact kernel outputs and verification checks, but the paired implementation stays flat in full-model Q4 C1 AR despite cold-kernel gains. |
| Huge-page-backed immutable weights | Retained for Q4 C1 AR/DFlash2: identical encoded bytes and greedy/cache results, approximately 2% more AR TG at d0/d32K, comparable PP and unchanged process RSS. Warm AR readiness stays below one second; cold loading remains unmeasured; Q8 execution checks now pass. |
| Residual/RMSNorm fusion after register caching | Rejected: byte-exact and faster as a component, but full-model AR is slower. |
| Separate gate/up waves, staged inputs, fixed FFN geometry and `-O3` | Not retained: byte-exact controls, but no useful isolated mapped-weight gain. Stage weights as production does; device-allocated microbenchmarks overstated earlier gains. |
| Cached small-batch RMSNorm and shared attention partition scales | Retained: explicit FMA and unchanged sum trees pass independent FP64 and byte-exact controls. Q4 C1 DFlash2 gains about 1.1% at d0/d32K; AR, PP, greedy output and sampled cache replay are retained. |
| C1 batched selector and FFN scratch reuse | Retained as simplification: exact token chains/probabilities, fewer launches and 15,616 fewer allocated bytes. The focused profile does not show an isolated selector speed gain. |
| Decode fallback for missing verification replay records | Correctness fix: reuse recorded rows and decode unrecorded rows with the original arithmetic. Full mixed-context and C1–C8 target replay checks pass; the complete recorded fast path is unchanged. |
| Generated-frontier replay and cancellation rollback | Correctness fix: retain decode arithmetic for pending output IDs; reuse the existing verification backup when cancellation interrupts publication. Q4/Q8 live, disk and seeded continuation checks pass without adding a normal-cycle GPU copy. |
| Q8 scalar row interleaving and packed coefficients | Rejected: byte-exact but slower than the existing read-only mapped-weight kernel; static type specialization is speed-neutral. |
| Q8 C1 measured width costs | Rejected: exact d0/d32K output, but no end-to-end gain; the existing policy remains. |
| Q8 verification staging four/eight activation rows | Rejected: byte-exact on 32-row gate/up and down projections, but extra staging phases outweigh the smaller shared-memory allocation. |
| Private Q8 C4 width costs and per-request full-block probes | Rejected: private costs slow repetition; per-request probes recover it but regress ordinary output. |
| Shared Q8 C4 greedy width | Retained: summed private acceptance histories avoid costly ragged verification. Difficult-pair TG improves 12.3%, deep full-checkpoint TG 13.2%; full acceptance, AR output, private sampled replay and C1 pp/tg are retained. |
| Shared Q8 C8 greedy width | Retained: complete measured cycles account for ragged projection costs. Difficult-pair TG improves 21.5% and d32K generated-history forks 17.5%; repetition, AR output and private sampled replay are retained. C1 PP ranges overlap in forward/reverse controls, with TG around 16.1 tok/s. |
| Shared Q8 C2/C6 greedy widths | Retained: measured complete-cycle costs improve difficult-pair TG by 6.1%/28.9% and d32K generated-history forks by 12.9%/10.2%. Repetition, AR output, private sampled replay and C1 pp/tg are retained. |
| Ragged Q8 projection groups and odd-cohort width selection | Retained: 17–36 rows share weight reads with bounded final-group loads/stores. C8 mixed reaches 65.96 tok/s, repetition 116.90 and d32K forks 41.52; outputs match AR. Scalar bits, full logits/features, rollback and private sampled state pass; C1 is retained. C6 remains sensitive to admission order. |
| Q4 costs for shrinking odd cohorts | Retained: avoids oversized verification after a peer finishes. Mixed C4/C6/C8 reaches 71.76/79.78/85.27 tok/s, C8 repetition 121.68 and d32K C4 39.87. Outputs, private sampled replay and C1 behavior are retained. |
| Saturated-cycle cost retune without accounting for partial verification | Rejected: faster full blocks increased wasted proposals and slowed ordinary workloads. |
| Neutral-history probes after isolated short successes | Rejected: recovers repetition but can over-propose on difficult C6 prompts. Retained probes require stronger established acceptance histories. |
| Native 38-row groups using 13/14-token tiles | Rejected: negligible gains or slower cold-weight projections; only the qualified 17–36-row groups are retained. |
| Concurrent generated-reply forks | Retained: freeze the live generated checkpoint before branching, within the snapshot budget. C1 avoids the copy; Q4/Q8 greedy/seeded forks, disk restoration and cancellation pass. |
| Visible causal attention tails | Retained: FP32 FMA over visible tail keys removes chunk-boundary drift while complete tiles keep WMMA. Q8 full logits/features match at 8K across scheduling budgets; four 32K DFlash2 continuations match isolated AR. Later packed V loads and row-first scale reuse recover PP speed without spills; large shallow chunks use the same packed route. |
| Separate causal-tail passes, helper functions and deeper V prefetch | Rejected: exact outputs, but extra work or register pressure makes attention slower. |
| KVarN split-K decode tile walk with hoisted tile parameters | Retained: the AR decode walks a split's keys one 128-key tile at a time, so the tile payload pointers, the per-lane RTN scale/zero and the channel scales load once per tile instead of once per key, and a lane's four low-plane nibbles come from one 16-bit word with the 5-/6-bit extras from one bit-plane byte. `#pragma clang fp contract(off)` in `kvarn_read.hip` keeps every multiply and add rounding explicit: with contraction on, the machine combiner fuses a different product subset in the rewritten loop than in the unchanged verification rows and T-05 stops being bit-identical, even though both bodies have the same source expression tree. All KVarN operator tests, `speculative_verification_test` and the KV storage tests pass with no tolerance or golden change. AR tg at d32K 5.80 → 6.73 tok/s and at d128K 3.35 → 4.90 (2.3× on the read path); see [benchmarks](BENCHMARKS.md) for the ratio against FP16. |
| Shared-KV tile staging for the KVarN decode | Rejected: decoding each sealed tile once per KV head into shared memory (the FP16 decode's six-head pattern) cuts the hot loop from 370 to 144 warp instructions per query head and key and stays byte-identical (T-03/T-05 green), but staging raises the kernel from 74 to 107 VGPRs and adds 16 KiB of shared memory per block, so occupancy falls from 12 to 5 waves per CU and the measured AR tg at d32K is 6.54 against 6.73 for the per-head decoder. Reverted. |
| Chunked split-K tile walk keeping the per-key helper calls | Rejected: bit-exact, but only +8.8% at d32K (6.31 against 5.80 tok/s) because the helper still recomputes the tile payload address per key and its parameter loads cannot leave the token loop; superseded by the inlined tile walk. |
| KVarN decode read: folded K/V arithmetic, explicit `fmaf`, 8-key blocks (2026-10-03) | Rejected by measurement: T-03/T-05 stay bit-exact, but the fold/block state raises the decode kernels to 111 VGPRs (2 waves/SIMD against ~74 VGPRs / 3 waves for the retained walk) and the AR tg ratio is lower at every long depth (`kvarn-k4v4` d32K 0.926 vs 0.936, d128K 0.787 vs 0.813). The retained phase-1 inlined tile walk remains the faster variant; reverting this rewrite is a candidate pending the G1 decision. |
| G1 performance judgment, second run (2026-10-03, Q8_0, host ROCm 7.14) | **Not met; stopped per design D18.** Depth 0 and 4K pass (0.974–0.996); 32K is 0.839–0.926 and 128K is 0.605–0.787 against the 1.00 floor, and `kvarn-k4v4` at 128K is R = 0.787/0.780 against the D18 threshold 1.092/1.090. Every preset misses, so design D18's quality-first exception (3.16, H1–H3) does not apply. Measured with the host toolchain (ROCm 7.14, host glibc 2.43): the flake `nix develop` build links ROCm's raw `hipClang` without the nixpkgs compiler wrapper and mixes host glibc 2.43 with the store's glibc 2.42, producing a binary that cannot start on this host. Reproduce: `python3 tools/qwen27b/g1.py evaluate --artifacts artifacts/g1sys`. |
| KVarN decode key-block sweep: (a)+(b) per key vs (a)+(b)+(c) at 2/4/8 keys (2026-10-03, the single D18 user-authorized 3.15 follow-up) | Retained `kKeyBlock = 1` (per key, no blocking): it is the best of the four at long depth. `kvarn-k4v4` tail unset 128K FP16 ratio 0.843 (depth cost 65.9 ms) against 0.811 / 0.796 / 0.787 for 2 / 4 / 8 keys; `kvarn-k6v5` tail 2048 0.631 (131.6 ms) against 0.609 / 0.606 / 0.605. Blocking barely lowers VGPRs (online 90 / 96 / 111 against 91 at 1 key; split-K partials 89 / 95 / 110; causal 75 / 79 / 90; occupancy reported 16 waves/SIMD for all) and loses more to the per-block maximum and rescale than it saves. T-03/T-05 stayed bit-exact for every variant. The 32K/128K 1.00 floors and the 128K D18 threshold remain out of reach. |
| KVarN compute-only read cost, cache-resident context sweep (2026-10-03) | With one full-attention layer's KV resident and a 4x context sweep (2304..9216) fitted against the query-head x key work, KVarN compute alone is 54 ms (`kvarn-k4v4`) / 122 ms (`kvarn-k6v5`) per 128K-token, 16-layer token against FP16's 34 ms; the measured production depth cost is 65.9 ms / 131.6 ms. Even the compute-only lower bound is above the ~36 ms a 1.00x FP16 at 128K needs, so no (a)–(c) variant can reach it. Tool: `tools/qwen27b/kvarn_read_compute_bench.hip` (hipcc, cache-resident KV, CPU reference seal). |

Q4 C1 AR leads the pinned d0/d32K controls; Q8 C1 AR matches them.
DFlash2 retains AR output in the focused shallow/deep and concurrent checks.
Current comparisons and unmeasured cells are in [benchmarks](BENCHMARKS.md);
[quality](QUALITY.md) records the quality scope.
Keep the work on one PR branch with incremental, reviewable commits.

Overall target: match llama.cpp generation speed, aiming for a further 10%,
on Q4_K_XL and Q8_K_XL with AR and **Q4_K_M DFlash2** at C1/C2/C4/C6/C8.
Cover both shallow (d0–d16K) and long-context (d32K–d128K) token generation:
screen d0 and d32K, then expand where needed to isolate or qualify the change.
Preserve prompt-processing speed, greedy AR/speculative agreement and sampled
replay. Check cross-engine differences against each engine's AR output.

Iterate with one affected shape and one control. Do not refresh the
full benchmark sweep during exploration. Repeat only to resolve noise or a
failure. Qualify the other configurations after the focused optimization phase,
or earlier only when a specific correctness concern requires it. Remote GPU time
is limited.

Published workload numbers live only in [benchmarks](BENCHMARKS.md); source and
model qualification live in [quality](QUALITY.md).
