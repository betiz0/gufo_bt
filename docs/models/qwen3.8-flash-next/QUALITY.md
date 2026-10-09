# Qwen3.8 Flash-Next quality

**Retained concurrency qualification: 63 tg128 requests match fresh AR completions:**
21 AR, 21 mixed MTP and 21 repetitive MTP at C1/2/4/6/8. Unsloth UD-Q4_K_XL target,
shared Q8_0 MTP; [identities](artifacts/model-identities.json).
These are consistency checks, not original unquantized-model or GGUF-conversion
qualification. Concurrency checks: September 20–23, 2026; attention review:
September 27–28.

| Check | Result |
| --- | --- |
| Unused final-layer outputs | Full-vocabulary logits match unpruned execution byte-for-byte through 133,120 tokens, also with mixed code/Unicode text and image input. Mixed batching and sampled RAM/serialized snapshot replay preserve logits and RNG state. [Evidence](artifacts/prefill-final-rows.json). |
| Bounded selector scratch | After 133,824 prompt tokens, 15,892,480 logits are byte-identical to the previous build. Ragged chunks retain exact masks and the independent FP64 score check. [Evidence](artifacts/prefill-deep-context.json). |
| Unused normalization removal | 15,892,480 logits are byte-identical to main; ragged residual-only fusion matches normalized fusion and the separate epilogue/combine. [Evidence](artifacts/prefill-normalization.json). |
| MTP versus scalar CPU formulas, eight text/image states | Fusion/attention relative RMS <0.0008 (limit 0.002); full-width normalization, split projections, recursive carry and full Q8 head checked |
| Batched MTP/AR, C2/C4/C6/C8 | Logits, tokens, acceptance, RNG and every 1–8-token rollback prefix match isolated execution |
| Sampling | 25 AR/MTP configurations, including top-p zero; FP64 target filtering/CDF, p/q acceptance, residual correction, seeded replay and short token budgets pass |
| Greedy penalties | GPU selection matches CPU at every 1–7-row verification width, including FP32-sensitive ties; native tool/JSON constraints remain active. C2/4/6/8 outputs match the previous implementation; image/tool cancellation, RAM/disk restore and OpenAI SDK checks pass. [Evidence](artifacts/penalty-verification.json). |
| Prefill and cache | Full logits match across tested chunk boundaries, short tails and restored state through 4096 tokens |
| Seeded MTP cache rebuilding | Two seeds × 200 tokens replay exactly after different prefill splits, cache bypass and replacement. K/V-only prefill and compact catch-up preserve full-head candidates across 1/8/9/32/33-row chunks. [Evidence](artifacts/mtp-cache-replay.json). |
| Scalar versus bulk prefill, 2176 tokens | Same top-1; logit RMSE 0.18, not bit-identical |
| Serving | Cancellation, three-turn continuation, reasoning/tool history, concurrent image/text isolation and disk restart pass |
| Prompt checkpoints | In-pass and borrowed checkpoints restore exactly in AR/MTP, including rewinds, branches, resets and destruction; 14 functional cache jobs pass correctness. [Details](PROMPT-CHECKPOINTS.md) |
| Sparse attention | Independent FP64 operator error ≤2.83e-7 (limit 1e-6). At 32K/128K, 256 fixed-token code/prose rows: mean KL 5.82e-4 and 256/256 top-1 agreement with an FP64-attention diagnostic. [Evidence](artifacts/attention-tiles-review.json). |

The attention diagnostic retains the quantized weights and other native
operators. Regrouping FP32 sums can change long-context text across builds;
matched-prefill AR/MTP logits and sampled snapshot replay are exact at
32K/128K. Session tests also cover image/text restoration and C2/4/6/8.

Sampled MTP can consume different RNG draws from AR. Seeded replay requires
the same build, request budget, capacity and sampling configuration; live cost
timings only steer greedy decoding. Draft sampling uses the full Q8 head's
top 64 logits; upstream draft-sampler equivalence is not claimed.
MTP cache projections now keep the same arithmetic across prompt splits.
Older Flash-Next disk checkpoints are rejected and rebuilt once.

## Agention AP-Q4_K_XL tier

**Not qualified: the teacher-forced gate fails.** The runtime accepts and
executes the AP-Q4_K_XL encodings (load, device bytes and route pinning below),
but the tier is **not listed as a supported weight-format mix**: on the
specification's teacher-forced comparison it is worse than the unsloth
`UD-Q4_K_XL` baseline on one of the two retained prompts.

Measured 2026-09-29 with the local llama.cpp Vulkan build (`b11197-a1de614ba`)
and the retained synthetic pp2048-class prompts (seed 0 / 1690 words, seed 1 /
1700 words), greedy, thinking off. Only the positions that predict the 128-token
llama.cpp greedy continuation are scored; prompt positions are excluded, as the
specification defines. File hashes and the source revision are in
[identities](artifacts/model-identities.json); the per-prompt summaries are
`artifacts/teacher-forced-{ap,ud}-d{0,1}.json`.

| Prompt | File | top-1 agreement | mean KL (Gufo→llama.cpp) | worst KL (position) |
| --- | --- | ---: | ---: | ---: |
| d0 (seed 0) | AP-Q4_K_XL | **0.984** | **0.0815** | 1.44 (2059) |
| d0 (seed 0) | UD-Q4_K_XL | 0.977 | 0.0140 | 0.47 (2057) |
| d1 (seed 1) | AP-Q4_K_XL | **1.000** | **0.0074** | 0.17 (2078) |
| d1 (seed 1) | UD-Q4_K_XL | 0.961 | 0.0105 | 0.05 (1979) |

The gate is per prompt and paired: on d0 the AP tier's mean KL (0.0815) is
higher than the baseline's (0.0140), so the tier is not qualified. Pooled it
also fails (AP 0.0444 against the baseline's 0.0123). The earlier
"pooled passes" result came from scoring 365 positions per prompt of which
~300 were prompt tokens; that dilution is why the scored range is now the
continuation only.

Scope limit found while re-running: the AP trunk's logits, unlike the
baseline's, depend on the `gpu_probe` prefill chunk width (the vendored Q6_K
dense MMQ accumulates in chunk-sized tiles). Re-scoring d0 with `--batch 32`
gives AP 0.0430 and the baseline 0.0140, so the d0 ordering holds at both
widths even though the exact KL is chunk-dependent.

| Check | Result |
| --- | --- |
| Load | `gufo prompt` 11.4 s; `gufo serve llm --context 262144` 11.0 s, 72,613,737,984 device weight bytes, 675,840,000 bytes decoded at load across 231 tensors (under the 1 GiB bound) |
| Greedy generation | completes at the native 262144-token context: 59-token prefill 170 tok/s, 32-token decode 25.0 tok/s, peak GTT 76.8 GiB |
| Teacher-forced agreement (llama.cpp `b11197-a1de614ba`, Vulkan; 128 + 128 continuation positions) | see the table above; d0 fails the per-prompt gate |
| Free-running greedy divergence (recorded, not gated) | first divergent position, with the top-1 margin: AP d0 1969 (0.09 logits), AP d1 none in the scored window, baseline d0 1966 (0.012), baseline d1 1958 (0.017) |
| Device bytes | every tensor uploaded as stored occupies its GGUF size plus the 4 KiB tail; the IQ4_NL routed down-experts occupy 22,649,241,600 bytes against 42,781,900,800 as Q8_0 |
| Route pinning | AP decode: Q6_K adapter GEMV, IQ4_NL vector-routed down-experts, Q8_0 GEMV, BF16/F16 SmallGemm. AP tiled: Q6_K adapter MMQ, IQ4_NL routed WMMA, Q8_0 F16 WMMA. The unsloth target keeps its pre-change routes, greedy outputs and measured rates |
| MTP smoke | the local `Qwen3.8-Flash-Next-MTP-Q8_0.gguf` sidecar does not bind (`missing tensor blk.48.nextn.hc_head_norm.weight`); the documented shared Q8_0 sidecar (`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, SHA-256 in the identities) loads on the AP trunk and runs the full `single-mtp` procedure (`gufo serve --context 133760 --speculative mtp --mtp-model …`, pp2048/tg128, 3 samples): mixed d0 pp 938.81 ± 16.57 / tg 32.17 ± 0.63, mixed d4096 pp 896.14 ± 2.36 / tg 31.73 ± 3.98, repetitive d0 pp 945.34 ± 8.32 / tg 58.73 ± 5.40, repetitive d4096 pp 897.88 ± 1.82 / tg 47.55 ± 6.56, against AP AR d0 tg 24.06. The draft is accepted at the baseline's rate (repetitive d0 3.79 accepted/step against the unsloth target's 3.85) |

No Signal-specific draft acceptance is claimed: the sidecar is the unsloth
shared head, not Agention's, and the pairing is a smoke check only.

Reproduce (the comparison needs the local Vulkan build at `$LLAMA`):

```sh
g++ -O2 -std=c++17 -I$LLAMA/include -I$LLAMA/ggml/include \
  tools/qwen-flash/logit_dump.cpp -L$LLAMA/build-vulkan/bin \
  -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build-vulkan/bin -o /tmp/logit_dump
python3 tools/qwen-flash/teacher_forced.py --logit-dump /tmp/logit_dump \
  --gufo-probe build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model "$TARGET" --tag ap --seed 0 --words 1690 --continuation 128 \
  --context 4096 --out teacher-forced-ap-d0.json   # seed 1 --words 1700 for d1
python3 tools/gguf_formats.py "$TARGET" --sha256   # identity and format histogram
```

The comparison scores only the positions that predict the 128-token llama.cpp
greedy continuation; run all four summaries (`{ap,ud}-d{0,1}`) with the same
build and prompts before evaluating the per-prompt gate.

Scope limits: text only (the Signal `mmproj-F16.gguf` vision projector is not
supported; Gufo's sidecar path expects BF16), no unquantized-model or
producer-parity claim, and the teacher-forced gate compares two different
artifacts of the same base architecture (the Signal fine-tune and the unsloth
re-quantization), so it bounds the Gufo/llama.cpp gap of each file rather than
isolating quantization error.

## Swift-1.5 Q4_K_M tier

**Not qualified: the teacher-forced gate fails at two of six configurations.**
The runtime accepts and executes the Swift-1.5 Q4_K_M encodings (load, device
bytes, route pinning and the context ceiling below), but the tier is **not
listed as a supported weight-format mix**: on the specification's
per-configuration teacher-forced comparison it fails the per-prompt paired gate
at two of the six (prompt, configuration) pairs — d0 at C1 has a higher mean KL
divergence than the unsloth `UD-Q4_K_XL` baseline, and d1 at C2 has both a
lower top-1 agreement and a higher mean KL divergence.

Measured 2026-09-29 with the local llama.cpp Vulkan build (`b11197-a1de614ba`,
0.5.0-dev build 11197, commit `a1de614ba`, RADV STRIX_HALO) and the retained
synthetic pp2048-class prompts (seed 0 / 1690 words → 1966 prompt tokens, seed
1 / 1700 words → 1958 prompt tokens), greedy, thinking off. Only the positions
that predict the 128-token llama.cpp greedy continuation are scored (128 per
run). File hashes and the source revision are in
[identities](artifacts/model-identities.json); the per-run summaries are
`artifacts/teacher-forced-{swift,ud}-d{0,1}-ub{2048,1024,512}.json`, each
recording the requested and effective batch values.

| Configuration | llama.cpp requested | llama.cpp effective | Gufo prefill chunk |
| --- | --- | --- | ---: |
| C1 | `-b 8192 -ub 2048` | `n_batch 4096`, `n_ubatch 2048` | 2048 |
| C2 | `-b 4096 -ub 1024` | `n_batch 4096`, `n_ubatch 1024` | 1024 |
| C3 | `-b 2048 -ub 512` | `n_batch 2048`, `n_ubatch 512` | 512 |

llama.cpp clamps its logical batch to the 4096-token context, so C1 runs at
`n_batch 4096`; the requested value is recorded beside it in every summary.

| Configuration | Prompt | Swift-1.5 top-1 | Swift-1.5 mean KL | UD-Q4_K_XL top-1 | UD-Q4_K_XL mean KL | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| C1 (`ub 2048`) | d0 | 0.992 | **0.0228** | 0.969 | 0.0195 | **fail** |
| C1 (`ub 2048`) | d1 | 0.992 | 0.0082 | 0.992 | 0.0082 | pass |
| C2 (`ub 1024`) | d0 | 0.984 | 0.0064 | 0.984 | 0.0343 | pass |
| C2 (`ub 1024`) | d1 | **0.992** | **0.0158** | 1.000 | 0.0103 | **fail** |
| C3 (`ub 512`) | d0 | 0.977 | 0.0059 | 0.977 | 0.0140 | pass |
| C3 (`ub 512`) | d1 | 0.992 | 0.0091 | 0.961 | 0.0105 | pass |

The gate is per prompt and paired at one configuration: d0 fails at C1 (mean KL
0.0228 against 0.0195) and d1 fails at C2 on both criteria (top-1 0.992 against
1.000 and mean KL 0.0158 against 0.0103), so the tier is not qualified. Both
files' agreement is configuration-dependent (the
baseline's d0 mean KL is 0.0195 / 0.0343 / 0.0140 at C1 / C2 / C3), which is
why every pair is compared at one configuration on both sides. Worst KL
positions: Swift d0 0.701 at 2071, d1 0.518 at 2082.

Free-running greedy divergence (recorded, not gated; Gufo's first position
whose argmax differs from the forced continuation, with its top-1/top-2 logit
margin): Swift d0 at 1970 (0.076), 1966 (0.072) and 1970 (0.013) for C1/C2/C3,
Swift d1 at 1962 (0.105/0.046/0.081); baseline d0 at 1966 (0.012), 1972 (0.042)
and 1966 (0.012), baseline d1 at 1962 (0.143), none in the scored window at C2,
and 1958 (0.124). The divergences are the same positions the AP tier shows,
within one token.

| Check | Result |
| --- | --- |
| Load | `gufo prompt` 12.18 s; `gufo serve llm --context 262144` 12.43 s, 89,407 MiB device memory in use at load (total 112,640 MiB), 84,248,815,104 device weight bytes, 17,858,560 bytes decoded at load across 37 tensors (under the 1 GiB bound), no Mixer tensor among them |
| Greedy generation | completes at the native 262144-token context (17-token prompt, 8 generated tokens, 27.4 tok/s decode): the measured context ceiling is the model's native maximum, and nothing larger exists to test |
| Teacher-forced agreement (llama.cpp `b11197-a1de614ba`, Vulkan; 128 continuation positions per run) | see the table above; two of six pairs fail the per-configuration gate |
| Free-running greedy divergence (recorded, not gated) | see above |
| Device bytes | every tensor uploaded as stored occupies its GGUF size plus the 4 KiB tail; the 96 Q5_0 Mixer up projections occupy 216,268,800 bytes against 629,145,600 as F16; the 35,201,689,600-byte Q5_0 `per_layer_token_embd` table is read per row from disk and never uploaded |
| Route pinning | Swift decode: Q4_K/Q5_0/Q6_K adapter GEMV, Q5_0 routed down-experts on the vector route, Q8_0 GEMV, BF16/F16 SmallGemm. Swift tiled: Q4_K/Q5_0/Q6_K adapter MMQ, Q5_0 routed down-experts on routed WMMA, Q8_0 F16 WMMA, and no Mixer fused/W8A8 route because the Mixer pair is stored, not decoded. The unsloth target and the Agention trunk keep byte-identical route records |
| MTP smoke | the shared Q8_0 sidecar (`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, SHA-256 in the identities) binds on the Swift trunk and one served greedy request at context 4096 completed with 8 drafted and 5 accepted draft tokens. No producer-specific draft acceptance is claimed: the sidecar is the unsloth shared head, not this producer's |
| Entry points | the same prompt and greedy configuration produce byte-identical text through `prompt` (two runs), `chat` and `serve` (two served requests) at context 4096, and the CLI's CPU fallback is unavailable for this model by design (`gufo prompt --cpu` is refused with "Flash-Next requires ROCm"). The host row decoder and the Q5_0 n-gram table the CPU path would use are covered by `qwen38_flash_next.cpu_ops`, `qwen38_flash_next.ngram` and the adapter operator tests. Known gap, not introduced here: the F32 oracle behind `gpu_probe --reference` returns non-finite logits on this trunk, and the same probe reproduces it on the Agention trunk with the pre-change binary, so it predates this change |
| Previously accepted files | unsloth `UD-Q4_K_XL`: routes byte-identical to the pre-change records, greedy completions with and without the shared Q8_0 sidecar byte-identical, session and snapshot replay identical, and the `single-ar` procedure (pp2048/tg128, greedy, thinking off, `--depths 0,4096`, four warmed samples per side with the same `gpu-test` binary) within the pre-change spread: d0 pp 1588.58 ± 41.42 → 1587.77 ± 42.34, tg 26.78 ± 0.05 → 26.78 ± 0.04; d4096 pp 1510.80 ± 1.29 → 1510.39 ± 1.64, tg 26.72 ± 0.01 → 26.71 ± 0.01, with every greedy completion hash matching across the two sides. Agention `AP-Q4_K_XL`: routes byte-identical and bind/upload device bytes and decoded total unchanged (75,386,855,936 and 675,840,000) |

Reproduce (the comparison needs the local Vulkan build at `$LLAMA`):

```sh
g++ -O2 -std=c++17 -I$LLAMA/include -I$LLAMA/ggml/include \
  tools/qwen-flash/logit_dump.cpp -L$LLAMA/build-vulkan/bin \
  -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build-vulkan/bin -o /tmp/logit_dump
python3 tools/qwen-flash/teacher_forced.py --logit-dump /tmp/logit_dump \
  --gufo-probe build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model "$SWIFT" --tag swift --seed 0 --words 1690 --continuation 128 \
  --context 4096 --llama-batch 8192 --llama-ubatch 2048 --batch 2048 \
  --out teacher-forced-swift-d0-ub2048.json   # seed 1 --words 1700 for d1
python3 tools/qwen-flash/teacher_forced.py --logit-dump /tmp/logit_dump \
  --gufo-probe build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model "$UD" --tag ud --seed 0 --words 1690 --continuation 128 \
  --context 4096 --llama-batch 8192 --llama-ubatch 2048 --batch 2048 \
  --out teacher-forced-ud-d0-ub2048.json      # and (4096,1024),(2048,512)
python3 tools/gguf_formats.py "$SWIFT" --sha256   # identity and format histogram
```

Scope limits: text only (this producer's vision projector is not supported),
no unquantized-model or producer-parity claim, and the teacher-forced gate
compares two different artifacts of the same base architecture (this
producer's fine-tune and the unsloth re-quantization), so it bounds the
Gufo/llama.cpp gap of each file rather than isolating quantization error.
A Mixer kernel faster than the per-tensor adapter projection this file's Mixer
pair uses is a separate change.



## Vision

**One Gufo encoder comparison fails:** relative L2 **6.47%** versus official
Transformers BF16, above the **5%** limit, on a 1024×1024 synthetic texture.
Both use the same converted GGUF weights. Against FP32, Gufo and Transformers
BF16 differ by **7.83% / 8.10%**, respectively. This is numerical drift, not an
image-answer score; **llama.cpp was not tested**. Optimizations retain native
embedding bytes but do not resolve this gap. [Evidence](artifacts/vision-parity.json).

## Reproduce

Tests live in [`tests/models/qwen38_flash_next`](../../../tests/models/qwen38_flash_next).
Use `--batch-only`, `--sampling-only`, `--prefill-only` or `--cache-only` on the
session test; the snapshot test covers persistent image/text state.

`--execution-only 133120` runs the focused deep AR/MTP logit and snapshot-replay
check without the other session suites.

For independent MTP checks:

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen38_flash_next_model_tests qwen38_flash_next_gpu_probe
nix develop -c build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model "$MODEL" --mtp-model "$MTP" --mtp-audit
```

The [vLLM](https://github.com/vllm-project/vllm/blob/751f6807d9cb3de50c27a5f27188c4fb04fe0e2b/vllm/models/qwen4_exp/amd/mtp.py)
and [SGLang](https://github.com/sgl-project/sglang/blob/993d1fccbaafe3e79d91567d2fc1d665cc94fa50/python/sglang/srt/models/qwen4_exp_mtp.py)
formulas supply independent predictor checks; pinned Transformers ignores MTP
weights. [Vision reproduction](../qwen3.8-27b/QUALITY.md#vision).

## Benchmark method

Gufo single-user pp/tg refreshed October 7, 2026 (`def2ed1e`), Performance power
profile. Reference, concurrency, loading and memory retain September 22–23
provenance.
One warmed sample per point, greedy, thinking off, penalties disabled.
Single-user uses pp2048/tg128; MTP pp is the maximum across mixed/repetitive
workloads. Gufo capacity is 133760; reference capacity is 35456 through 32K,
68224 at 64K and 133760 at 128K. C1/2/4/6/8 use the same d0 prompts; every
session prefills before measured tg128, with at most four prompt-tail tokens
reevaluated. Rates sum individual decode rates.
Depth calibration depends on the ordered sweep. Paired speed controls use
the same depth list. Deep AR/MTP HTTP cache frontiers differ by one token;
exact replay is checked separately with identical prefill boundaries.
AR reference is llama.cpp b11069; MTP uses pinned `6fcaa16f`.
Loading: cold files, C1/MTP/capacity 262144. Memory: C1/AR/capacity 133121,
peak global HIP allocation including idle memory. Full commands, counts and
identities remain in [artifacts](artifacts/bench.json) and the
[benchmark workflow](../../../.agents/skills/benchmark-model/SKILL.md).
