# Qwen3.8 27B quality

**Greedy DFlash2 matches isolated AR on all 160 measured mixed/repetitive
requests across Q4/Q8 and C1/2/4/6/8.** Targets are Unsloth UD-Q4_K_XL /
UD-Q8_K_XL; the draft is Q4_K_M. [Identities and dated evidence](artifacts/model-identities.json).
These are execution-consistency checks; original-target and GGUF-conversion
parity remain unqualified.

| Check | Q4_K_XL | Q8_K_XL |
| --- | ---: | ---: |
| AR/DFlash2 mixed completions, eight depths 0–128K | 8/8 match | 8/8 match |
| AR repetitive C2/C4/C6/C8 requests versus C1 | 20/20 match | 20/20 match |
| DFlash2 mixed requests versus isolated AR, C1–C8 | 59/59 match | 59/59 match |
| DFlash2 repetitive requests versus isolated AR, C1–C8 | 21/21 match | 21/21 match |
| Target verification, widths 2–8 and ragged cohorts | Full logits/features match scalar | Full logits/features match scalar |
| Serving and state | Cancellation, continuation, RAM/disk restore and seeded replay pass | Same; also eight 32K continuation forks |

## Quantization comparison

September 24, 2026: short teacher-forced comparison against **Gufo's BF16 target**, with identical
prefixes: three prose/code fixtures, nine logit rows each, full 248,320-token
vocabulary, temperature 1 without filtering. This measures quantization and
native arithmetic together; BF16 here is not an independent upstream oracle.
Results and reproducible identities are in [the compact record](artifacts/quantization-quality.json).

| Target versus BF16 | Mean KL divergence (nats) | Mean total variation | Top-1 agreement |
| --- | ---: | ---: | ---: |
| Q4_K_XL | 0.001650 | 0.01566 | 27/27 |
| Q8_K_XL | 0.00006537 | 0.003128 | 27/27 |

## KVarN KV quantization (phase 0)

The KVarN KV-cache quantization change
(`openspec/changes/add-kvarn-kv-cache-quantization`) passed its phase-0 quality
gate (G0) on the pre-registered q8→q4 interpolated-position rule (design D17).
Teacher-forced comparison against the FP16-KV teacher, three corpora, the 256
positions after each 4K/32K/128K prefix, one tokenization and one history per
candidate (design D14). Corpus 1 / 32K was excluded by the pre-registered
teacher-mean-NLL rule (> 5). The probes are phase-0 approximations (design D13);
phase 1 implements the production read path. The per-depth metrics are in the
[model README](README.md#kv-cache-quantization-phase-0).

Baselines under the same framework (block 32, FP16 scale, rotated, no tail;
every row quantized right after the sink):

| Baseline | Depth | Mean KL | Median KL | P99.9 KL | Top-1 | ΔNLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E_q8 | 4K | 0.02075 | 0.000024 | 2.618 | 0.9831 | +0.02043 |
| E_q8 | 32K | 0.06490 | 0.000073 | 12.331 | 0.9766 | +0.03987 |
| E_q8 | 128K | 0.05247 | 0.000019 | 5.833 | 0.9701 | +0.04238 |
| E_q4 | 4K | 0.26810 | 0.000598 | 20.372 | 0.9414 | +0.03354 |
| E_q4 | 32K | 0.39844 | 0.001992 | 21.898 | 0.9180 | +0.04883 |
| E_q4 | 128K | 0.43462 | 0.000367 | 22.086 | 0.9154 | -0.08130 |

Gate result (worst p over depths and the four judged metrics):

| Preset | Fitting tail | Worst p | Limiting | Budget |
| --- | ---: | ---: | --- | ---: |
| `kvarn-k4v4` (memory-first) | 256 | 0.971 | 128K large-deviation rate | ≤ 1 |
| `kvarn-k6v5` (quality-first) | 2048 | 0.143 | 128K large-deviation rate | ≤ 0.15 |

`kvarn-k4v4` without a tail fails at 128K (p = 1.086); `kvarn-k5v4` and
`kvarn-k5v5` fit the memory-first budget at every tail. Scale metadata stays
FP16: the probe saw no FP16 overflow, and the 167–215 subnormal scales per
configuration are the ε floor (1e-5 is below FP16's 6.1e-5 normal minimum) and
zero points, not a dynamic-range shortfall of real data.

Reproduce (design D16):

```sh
python3 tools/qwen27b/kvarn_quality.py report --label g0-tail --gate
```

## DFlash2 and sampling

The pinned [original DFlash2 operators](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a)
were checked with independently decoded GGUF weights: worst stage relative
RMSE **5.83e-6**, full-logit maximum error **7.72e-5**, proposal total variation
**1.60e-5** on a 24-token prefix/seven proposals. Limits are 1e-4, 1e-3 and 1e-4.
This qualifies draft execution, not weight conversion.

AR and verification apply penalties → temperature → top-k → top-p → min-p.
Tests cover 216 GPU sampling cases and 23 served strategies, including p/q
acceptance, full-vocabulary residual correction, nonfinite inputs and replay.
**Sampled DFlash2 need not match AR's same-seed sequence**; within-mode seeded
replay requires the same configuration. GPU/CPU distributions have numerical
tolerances, not bit-identical upstream sampler parity.

**Re-measured (2026-10-05, tasks 7.8):** with the verification-store
fix the same corpus reports no AR/DFlash2 mismatch at depth 0 or
~3.9K and the depth-0 acceptance returns to parity (FP16 0.844
against KVarN 0.844); the pre-fix numbers below stay recorded as
invalid.

KVarN KV changes where the target's greedy decisions land, so a KVarN DFlash2
run need not match an FP16-KV DFlash2 run. The cause was checked without code
changes (design D20, tasks 4.8). At depth 0 the two runs emit the same first two
tokens **exactly** — identical sha256, acceptance 0.000 and the same decode
step counts — and separate at the third token, which points to the
rotated-space rounding of the target rather than to the draft wiring. Across the
11-case corpus (32 generated tokens, `speculative-corpus.py`) the
pre-fix DFlash2 acceptance difference was not significant: depth 0 was
FP16 0.876 against KVarN 0.814 (paired bootstrap 95% CI of the mean
difference [-0.198, +0.056]) and depth 4096 was 0.765 against 0.695
([-0.169, +0.009]); with the store fixed the depth-0 acceptance is at
parity (0.844 against 0.844), so no systematic acceptance loss was
found. **Outcome (2026-10-05, G2 re-measured):** the DFlash2 depth-0
tg floor (0.95× FP16) is met — 0.994–0.995× for all six presets and
tails — so G2 passes under D19; the depth 4K and 32K+ ratios are the
recorded reference values and the D19 AR reference values are
unchanged. Measurements and reproduce commands are
in [experiments](EXPERIMENTS.md).

## Vision

Q4/Q8 AR and DFlash2 pass multi-image, sampled/greedy C4, cancellation,
generated-history continuation and disk-restart checks. Q8 also requires exact
cold/live/restored continuation logits. Encoder controls use official
Transformers operators over the **same converted BF16 GGUF**; they cannot detect
conversion errors. Flash-Next has a separate [vision limit](../qwen3.8-flash-next/QUALITY.md#vision).
Q4/Q8 image snapshots restored after cancellation inside the image encoder or
text prefill match all target logits exactly; token-limit endings and stripped
tool reasoning retain the compatible conversation prefix.

## Reproduce

Run only the affected [model suite](../../../tools/qwen27b/check.py):

```sh
nix develop -c python3 tools/qwen27b/check.py fast
nix develop -c python3 tools/qwen27b/check.py reference \
  --model "$MODEL" --reference-model "$BF16_REFERENCE"
nix develop -c python3 tools/qwen27b/check.py serving \
  --model "$MODEL" --dflash-model "$DRAFT"
```

[Draft oracle](../../../tools/qwen27b/dflash_reference.py),
[vision oracle](../../../tools/qwen27b/vision_reference.py) and
[vision serving checks](../../../tools/qwen27b/vision_check.py) retain source
pins and compare operators, full logits, RNG, image/state isolation and replay.
Full logits stay outside Git. A missing-weight skip is not a pass.

## KVarN rotation limits

The KVarN Hadamard rotation is defined on 256-element heads with the exact 1/16
scale (`KvarnFwhtHeadInPlace`, `src/models/qwen/kvarn_reference.cpp`).
`qwen_kvarn_fwht_cpu_test` declares these FP32 limits as source constants and
asserts them against that implementation.

| Check | Worst case | Declared limit | Measured |
| --- | --- | ---: | ---: |
| `max abs(x - H(Hx))`, `max abs(x) = 1` | `2*gamma_255 = 3.05e-5` | 1e-4 | 1.8e-7 |
| `abs(<q,k> - <Hq,Hk>)`, unit-norm q and k | `2*gamma_255 = 3.05e-5` | 1e-4 | 2.1e-8 |
| `H[i][j] - H[j][i]` | exact symmetry | 0 | 0 |

`gamma_255 = 255u / (1 - 255u)` with `u = 2^-24`: each output sums 256 signed
inputs through eight butterfly stages, so at most 255 roundings contribute, and
a round trip or a rotated inner product doubles that. The declared limit sits
above the worst case rather than at it, and holds only for FP32 arithmetic. The
matrix is built by applying the shipped call to the 256 basis vectors, so exact
symmetry plus the measured round trip establish `H * H^T = I`.

The same 1e-4 relative limit is declared for the GPU side (T-02): the decode
fused write applies the identical rotation to Q and K
(`KvarnFwhtRowShared`, `src/models/qwen/hip/kernels/kvarn_device.hpp`), and
`qwen_attention_fusion_ops_test` asserts the relative change of the FP32 QK
score stays under it (measured 1.4e-7) while the FP16 path keeps its exact
Q/K/cache match and the KVarN sink holds the rotated K.

## KVarN seal limits

`qwen_attention_kvarn_seal_ops_test` declares these limits as source constants.

| Check | Basis | Declared limit | Measured |
| --- | --- | ---: | ---: |
| T-01 stage 1: codes from the payload's FP16 scales and RTN parameters | exact | 0 | 0 |
| T-01 stage 2: scales versus the CPU double oracle | FP16 metadata storage plus the reader's `raw / (channel_scale * token_scale)` form versus the oracle's progressive float values | 1e-2 relative (absolute floor 1e-2) | ≤ 0.4% |

The seal stores its metadata as FP16 and encodes the codes from those stored
values, so stage 1 reproduces every code exactly from the payload alone. Stage 2
compares the stored scales with the CPU oracle (`KvarnNormalizeTile` +
`KvarnQuantizeAsymmetricRtn`); the GPU normalizes with the FP16-rounded scale
vectors that the reader also uses, so the two differ by that rounding, which a
range-of-extremes can amplify. A wrong pass count or RTN axis would exceed the
limit by orders of magnitude.

An axis whose values are all zero floors its RMS and its accumulated scale can
underflow FP16 to 0. The seal then stores a normalized value of 0 for that
element (the reader reconstructs 0 from a zero scale either way) instead of
`0/0`, so a constant channel or an outlier never produces a non-finite code or
scale.

## KVarN decode read limits

`qwen_attention_kvarn_read_ops_test` declares one limit as a source constant
(`kOutputLimit`).

| Check | Basis | Declared limit | Measured |
| --- | --- | ---: | ---: |
| T-03: three-region decode read (sink, sealed tiles, F16 tail ring) versus the FP64 reference | FP32 online softmax with per-position warp reductions versus exact FP64 arithmetic; the reconstructed values themselves agree because both sides read the same FP16 payload metadata | 2e-4 absolute | 1.02e-07 |
| T-03: tile-aligned split-K versus the online path | same kernels, different summation order over whole 128-token tiles | 2e-4 absolute | 4.84e-08 |
| T-03: device-supplied query position versus the host position | identical arithmetic, position read from device memory for HIP graph capture | 2e-4 absolute | 0 |

The contexts are the design D1 boundary set for the unset tail (L = W = 9) and
L = 2048: 127, 128, 129, 128+L+127 through 128+L+129, and 256+L+127 through
256+L+129. A mis-selected region, a wrong tile index or a wrong view-rule
comparison changes which rows enter the softmax and moves the output by orders
of magnitude, so this limit separates the read contract from a routing error;
the largest observed error is three orders of magnitude below it. The limit was
not relaxed to pass.

## KVarN prefill read limits

`qwen_attention_kvarn_prefill_read_ops_test` declares one limit as a source
constant (`kOutputLimit`).

| Check | Basis | Declared limit | Measured |
| --- | --- | ---: | ---: |
| T-03: tile / WMMA / batched prefill read over the sink, the sealed tiles and the per-chunk F16 workspace versus the FP64 reference | the tile and WMMA paths stage the decoded K/V as FP16 in shared memory and WMMA while the reference is FP64 over exact decoded values | 4e-3 absolute | ≤ 6.6e-5 |
| T-03: the three prefill paths versus each other | same inputs, different staging and reduction orders (FP16 WMMA/tile versus FP32 batched) | 4e-3 absolute | ≤ 2.3e-4 |
| T-04: chunk flush versus the chunk workspace | the flush copies halves verbatim into the sink and the tail ring | bit-identical | 0 |

The contexts are the same design D1 boundary set, as one prefill chunk (up to
256+L+129 = 2433 rows) and as two sequential chunks whose second chunk reads its
pre-chunk F16 rows from the ring the first chunk flushed. The FP16 staging in
the tile and WMMA paths sets the limit; the batched path keeps FP32 and is
three orders of magnitude closer to the reference. The limit was not relaxed to
pass.

## Meaning of exact

Matching Gufo AR/speculative tokens or completion hashes tests consistency.
Matching llama.cpp hashes also does not establish original-model accuracy;
quantized implementations can disagree. Keep the reference, weights and history
fixed before interpreting any numerical difference.

## Benchmark method

September 23, 2026; one warmed sample per point, greedy, thinking off.
Single-user rows use pp2048/tg128 and cached depths 0–128K. Each mode creates
its own eight-token prefix reply; histories can differ across engines.
Concurrency uses the older short-prompt corpus, not matching pp2048 prompts;
its C1 is not directly comparable to single-user d0. AR uses repetitive text;
DFlash2 uses mixed/repetitive cases. Rates sum individual decode rates.
Loading uses cold files, C1/DFlash2/capacity 262144; memory uses C1/AR peak HIP
allocation. Commands, hashes and counts remain in [artifacts](artifacts/bench.json)
and the [benchmark workflow](../../../.agents/skills/benchmark-model/SKILL.md).
