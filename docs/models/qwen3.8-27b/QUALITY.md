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

#### Re-measured baselines on the G0 windows (change task 1.4)

The re-judgment change (`improve-kvarn-k6v5-quality-and-speed`) restores the
baselines as a gpu-test-only `--kv-cache baseline-q8|baseline-q4` write path
(design D1: rotate, quantize every non-sink row in blocks of 32 with an FP16
scale, dequantize, inverse-rotate, store in the dense FP16 plane) and
re-measures them on G0's own windows: label `g0-baseline-check`, G0's three
corpora, G0's cached 4K teachers, depth 4K, 256 positions, run through the test
ledger (`artifacts/test-ledger.jsonl`, `T1.4-BASELINE-G0`: exit 0, 274 s, all
six units with 0 non-finite logits). A fresh label repeats one unit
bit-identically (`T1.4-DET`, `baseline-q8` corpus 2: every field but `corpus`
and `seconds` compares equal) and the teacher logits hash is unchanged, so the
differences below come from the write path, not from measurement noise.

| Baseline | Depth | Mean KL | Median KL | P99.9 KL | Top-1 | ΔNLL | Mean KL vs G0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E_q8 | 4K | 0.06401 | 0.000024 | 13.338 | 0.9753 | +0.06011 | +0.04326 |
| E_q4 | 4K | 0.31170 | 0.000613 | 18.906 | 0.9492 | +0.00643 | +0.04360 |

The median KL and the E_q8 top-1 rate stay within 0.008 of the G0 record, so
the block-32 definition agrees at the typical position; the difference sits in
the tail. Of the 50 window positions whose KL moves by more than 0.5 nats, 44
are in the second half of the window, where the 4,352-token context is longest,
and single positions carry spikes of up to +18.7 nats that the G0 record does
not have. The per-corpus mean KL therefore moves in both directions (E_q8
corpus 0: 0.00256 → 0.00182, lower; corpus 2: 0.05820 → 0.18639, higher).

Hypothesis (the design's, not verified): phase 0 kept the rotated domain — it
stored the quantized rotated row and rotated Q the same way — while design D1
inverse-rotates the quantized row back and stores the *unrotated* row in FP16.
The inverse rotation spreads each block's quantization error over all 256
channels, and the stored row is rounded to FP16 once more; the extra error
accumulates over the whole history and shows up where the teacher distribution
is most sensitive. D1's estimate of that rounding (about an eighth of the E_q8
error) is smaller than the measured difference. The difference is recorded, not
judged (design D1): the ruler of this change is a baseline measured in the same
label and the same windows, and the G0 numbers stay above as the phase-0
record.

### Re-judgment windows (tasks 2.5-2.7)

The re-judgment evaluates three window groups, all of them different from G0's:

| Depth | Label | Windows | Source |
| --- | --- | ---: | --- |
| 4K, 32K | `t08x` | 14 | `t08-reduced`'s eight (seeds 11-18, `~/.cache/gufo/kvarn-quality/t08/corpus-*.txt`) plus six fresh ones (`ext32k/corpus-*.txt`, seeds 21-26) |
| 128K development | `g0dev-128k` | 3 | G0's corpora 0-2 (`~/.cache/gufo/kvarn-quality/corpus-*.txt`) |
| 128K verdict | `verdict-128k` | 10 | fresh corpora, seeds 31-40 (`verdict128k/corpus-*.txt`) |

The 32K extension exists because a label's corpus set cannot change after its
first run while the 32K depth must keep six or more windows after the
pre-registered teacher-mean-NLL rule (design D14); `t08-reduced` itself is left
untouched. The 128K development windows reuse G0's cache: the corpus files are
unchanged since 2026-09-30, their encoded-token SHA-256 values equal `g0-tail`'s
manifest entries (and `merge-upstream-quality`'s, measured later on the same
files), and the three 128K teacher files were copied into `g0dev-128k/teacher/`
byte-identically (`c0..c2-d131072.logits`, SHA-256 equal to the `g0-tail`
originals; the harness revalidates the token hash when it loads them). The
verdict windows are measured for the baselines once (task 3.3) and for the
candidates exactly once, in the final judgment (task 9.1); no improvement check
uses them. The 4K and 32K judgment reads the `t08x` units of the final build
(no chapter-6 or chapter-7 change was adopted, so those candidate units are the
final ones), the 128K judgment reads `verdict-128k`. Exclusions under the pre-registered teacher-mean-NLL rule (D14):

| Depth | Windows | Surviving | Excluded (teacher mean NLL) |
| --- | ---: | ---: | --- |
| 4K | 14 | 14 | none |
| 32K | 14 | 8 | corpus 2 (11.3), 3 (6.0), 4 (5.5), 7 (8.5), 9 (6.2), 10 (11.2) |
| 128K development | 3 | 3 | none |
| 128K verdict | 10 | 9 | corpus 0 (11.1) |

The five-window extension was enough: six of the fourteen 32K windows fall to
the rule and eight survive, so the depth keeps well above the six-window
target (the earlier `t08-reduced` set alone would have left four).

### Reference positions of this build (task 3.4)

q8->q4 positions with same-label baselines (window bootstrap, 2000 resamples,
seed 20261005). 4K and 32K come from `t08x` (14 and 8 surviving windows), 128K
development from `g0dev-128k` (3 windows).

| Metric | 4K | 32K | 128K development |
| --- | ---: | ---: | ---: |
| mean KL | -0.025 [-0.077, 0.044] | 0.104 [-0.006, 0.145] | **0.178 [0.080, 0.254]** |
| median KL | 0.002 [-0.009, 0.010] | -0.009 [-0.059, 0.014] | 0.008 [0.007, 0.022] |
| large-deviation rate | 0.031 [0.000, 0.158] | 0.125 [0.000, 0.164] | 0.088 [-0.125, 0.167] |
| top-1 disagreement | -0.024 [-0.195, 0.100] | 0.000 [-0.227, 0.088] | -0.056 [-0.125, 0.000] |

`kvarn-k6v5` tail 2048 passes the 0.13 budget at 4K and 32K (worst 0.125 at
32K, the large-deviation rate over 8 windows) but **fails on the 128K
development windows** on mean KL (0.178 > 0.13, interval [0.080, 0.254] over
three windows); the machine's spike mechanism is the production prefill (WMMA)
path's f16 staging rounding (task 4.2 verdict in
[experiments](EXPERIMENTS.md)). The change's judgment therefore rests on the
verdict windows (`verdict-128k`, nine surviving windows), measured exactly once
in task 9.1. `kvarn-k4v4` tail 256 stays at or below 1 everywhere measured so
far (worst 0.875 at 32K).

### Final judgment of this change (tasks 9.1-9.2)

Two labels carry the judgment, each judged with its own same-label baselines
(window bootstrap, 2000 resamples, seed 20261005; `--gate`):

* `t08x` — 4K and 32K, 14 and 8 surviving windows (D14 excludes six 32K
  windows). `kvarn-k6v5` tail 2048 passes (worst p 0.125, the 32K
  large-deviation rate over 8 windows, budget 0.13) and `kvarn-k4v4` tail 256
  passes (worst p 0.875, the 32K large-deviation rate, budget 1). The stretch
  goal p <= 0.10 is not met (recorded, not judged).
* `verdict-128k` — 128K, ten fresh-seed windows, corpus 0 excluded by D14, so
  nine survive; the candidates were measured on these windows exactly once
  (task 9.1, `T9.1-128K-K6V5-1..2` and `T9.1-128K-K4V4-1..2`).

| Metric | `kvarn-k6v5` tail 2048 (budget 0.13) | `kvarn-k4v4` tail 256 (budget 1) |
| --- | ---: | ---: |
| mean KL | **0.161 [-0.002, 0.345]** | 0.834 [0.390, 1.321] |
| median KL | 0.025 [0.009, 0.045] | 0.317 [0.224, 0.517] |
| large-deviation rate | **0.195 [0.000, 0.457]** | 0.829 [0.429, 1.071] |
| top-1 disagreement | 0.136 [-0.046, 0.380] | 0.739 [0.495, 0.976] |

The gate fails for `kvarn-k6v5` tail 2048 at depth 128K (worst p 0.195, the
large-deviation rate) and passes for `kvarn-k4v4` tail 256 (worst p 0.834, mean
KL). The raw 128K means behind the ratios: `kvarn-k6v5` 0.03195,
`baseline-q8` 0.01250, `baseline-q4` 0.13295; non-finite logits are 0 in every
unit. The 128K point estimate is consistent with the three-window development
result above (0.178) and with the G0 anchor record (0.143): the production
prefill path's f16 staging rounding (task 4.2) still sets the position, and no
adopted change moved it (chapter 7 records the decision of 2026-10-11).

The quality contract's 128K clause ("p <= 0.13 at every depth") is therefore
**not met**, and the design's fallback applies: adopt a bpv-increasing
improvement or revise the target, with the decision returned to the user. The
verdict windows are measured once, so a revised candidate cannot be re-judged
on them; it would need fresh windows or the development windows.

**Decision (2026-10-12, user):** record the outcome and revise the target. The
change's quality contract now fixes the 0.13 budget at 4K and 32K and requires
the 128K position to be recorded as measured; `kvarn-k6v5` tail 2048 is
documented as a 4K/32K configuration and its 128K use is not sanctioned. No
bpv-increasing improvement (the `k8` / `q8_0`-class variants of task 4.4) is
adopted.

Reproduce (from the repository root):

```sh
python3 tools/qwen27b/kvarn_quality.py report \
  --out ~/.cache/gufo/kvarn-quality --label t08x --gate
python3 tools/qwen27b/kvarn_quality.py report \
  --out ~/.cache/gufo/kvarn-quality --label verdict-128k --gate
```

Anchor-method record for comparison (the previous change, design D-27: G0's
E_q8/E_q4 on G0's windows as a fixed ruler): `kvarn-k6v5` tail 2048 worst 0.087
and `kvarn-k4v4` tail 256 worst 0.478 on `t08-reduced`; re-reading the archived
G0 label with its own baselines gives `kvarn-k6v5` tail 2048 worst 0.143 (128K
large-deviation rate). The two rulers differ by window and implementation, so
the numbers are recorded side by side rather than compared.

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

**Re-judgment scope (design D23).** The production read path is re-judged on
fresh windows (`t08-reduced`: eight corpora, seeds 11–18, the same 256
positions after each 4K/32K prefix) at depths 4K and 32K only. Depth 128K keeps
the G0 measurement above, which is therefore **provisional**: the quality-first
preset's limiting metric (the 128K large-deviation rate at p = 0.143) is not
re-measured here. The q8→q4 positions of the re-judgment use G0's E_q8/E_q4
windows as a fixed ruler (design D17 as amended on 2026-10-08, D-27), because
the baseline implementation was removed with task 3.10 and is not restored;
the window dependence of the baseline statistics therefore remains in those
positions. Statistical task evaluation (decoding tasks and the long-distance
reference non-inferiority test with δ/δ_ret/N/N_ret) is deferred to a follow-up
change; this change records a short 32K long-distance reference check. KVarN
with image tokens and M-RoPE positions is unverified (task 5.4: the vision
sidecar is not available on the measurement host), so image requests keep the
default `fp16` KV.
The 32K depth has **four surviving windows** (4K has eight): the spec's
re-judgment scenario asks for six or more per depth, and the difference is
recorded here as a documented deviation — the pre-registered teacher-mean-NLL
exclusion (design D14) drops corpora 2, 3, 4 and 7 at 32K, and the judgment
therefore rests on the 4-window point estimate with its interval beside it.



### Long-distance reference check (32K, task 6.2)

One 32K context per configuration: 64 invented facts sit after the sink
(first 128 tokens) and 28,551 tokens before the questions, so the answer
has to come from sealed tiles. All 64 questions are asked in one request,
so each configuration prefills the 32K context exactly once (design D23;
per-question requests re-prefilled it, because the continuation cache only
reuses stored checkpoints whose prompt matches exactly). Requests use
`reasoning_effort: low` and the answers are graded by exact match of the
4-digit code from the numbered answer lines; there is no pass criterion.
Problem set: `docs/models/qwen3.8-27b/artifacts/longref-32k.json`
(SHA-256 `2c2dfca1eee530fd…`).

| Configuration | Correct | Empty answers | Questions | Seconds |
| --- | ---: | ---: | ---: | ---: |
| `fp16` | 64/64 | 0 | 64 | 492 |
| `kvarn-k5v4` (tail 2048) | 64/64 | 0 | 64 | 604 |
| `kvarn-k6v5` (tail 2048) | 64/64 | 0 | 64 | 323 |

Settings: `reasoning_effort: low`, `max_tokens: 160`, temperature 0,
`cache_prompt: true`, one `serve llm` per configuration on the release
build (Q8_0, `--sessions 1`, `--context 40960`, continuation cache).
Calibration: prefix 32081 tokens, facts at
tokens 2058..3530, distance to the question
28551 tokens.
Reproduce:

```sh
python3 tools/qwen27b/longref_check.py --config kvarn-k5v4 --tail 2048 \
  --label t62-k5v4-2048 --max-tokens 160 --binary build/release/gufo \
  --model /home/betiz/models/Qwen3.8-27B-Q8_0.gguf \
  --output ~/.cache/gufo/longref
```

### Production-path re-judgment (T-08 reduced, 2026-10-08)

Fresh windows (`t08-reduced`: eight corpora, seeds 11–18) at depths 4K and
32K, measured through the production read path (`qwen_kvarn_quality_eval`
with `--kv-cache` / `--kv-cache-tail`). The q8→q4 positions use G0's
E_q8/E_q4 windows as the fixed ruler (design D17 as amended 2026-10-08,
D-27). Windows excluded by the pre-registered teacher-mean-NLL rule:
corpora 2, 3, 4 and 7 at 32K (4K keeps all eight). No non-finite logits.

| Preset | Tail | Depth | Windows | Mean KL | Median KL | P99.9 KL | Top-1 | ΔNLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `kvarn-k4v4` | 256 | 4K | 8 | 0.02181 | 0.000171 | 1.614 | 0.9834 | -0.00286 |
| `kvarn-k4v4` | 256 | 32K | 4 | 0.14711 | 0.000278 | 13.003 | 0.9658 | -0.08478 |
| `kvarn-k5v4` | 2048 | 4K | 8 | 0.00799 | 0.000048 | 2.898 | 0.9907 | +0.00631 |
| `kvarn-k5v4` | 2048 | 32K | 4 | 0.06486 | 0.000068 | 12.713 | 0.9746 | +0.03960 |
| `kvarn-k6v5` | 2048 | 4K | 8 | 0.00389 | 0.000034 | 0.413 | 0.9897 | +0.00618 |
| `kvarn-k6v5` | 2048 | 32K | 4 | 0.06358 | 0.000037 | 10.909 | 0.9834 | -0.02267 |

| Preset | Tail | Depth | Metric | p | 95% CI |
| --- | ---: | ---: | --- | ---: | ---: |
| `kvarn-k4v4` | 256 | 4K | mean KL | +0.004 | [-0.085, 1.351] |
| `kvarn-k4v4` | 256 | 4K | median KL | +0.255 | [-0.001, 2.298] |
| `kvarn-k4v4` | 256 | 4K | large-deviation rate | -0.070 | [-0.174, 1.127] |
| `kvarn-k4v4` | 256 | 4K | top-1 disagreement rate | -0.008 | [-0.301, 1.557] |
| `kvarn-k4v4` | 256 | 32K | mean KL | +0.246 | [-0.221, 1.504] |
| `kvarn-k4v4` | 256 | 32K | median KL | +0.107 | [-0.021, 0.514] |
| `kvarn-k4v4` | 256 | 32K | large-deviation rate | +0.478 | [-0.130, 2.125] |
| `kvarn-k4v4` | 256 | 32K | top-1 disagreement rate | +0.183 | [-0.306, 0.983] |
| `kvarn-k5v4` | 2048 | 4K | mean KL | -0.052 | [-0.087, 0.378] |
| `kvarn-k5v4` | 2048 | 4K | median KL | +0.041 | [-0.032, 0.646] |
| `kvarn-k5v4` | 2048 | 4K | large-deviation rate | -0.115 | [-0.174, 0.525] |
| `kvarn-k5v4` | 2048 | 4K | top-1 disagreement rate | -0.184 | [-0.354, 0.725] |
| `kvarn-k5v4` | 2048 | 32K | mean KL | -0.000 | [-0.243, 0.653] |
| `kvarn-k5v4` | 2048 | 32K | median KL | -0.003 | [-0.032, 0.051] |
| `kvarn-k5v4` | 2048 | 32K | large-deviation rate | +0.109 | [-0.167, 0.750] |
| `kvarn-k5v4` | 2048 | 32K | top-1 disagreement rate | +0.033 | [-0.361, 0.604] |
| `kvarn-k6v5` | 2048 | 4K | mean KL | -0.068 | [-0.088, 0.145] |
| `kvarn-k6v5` | 2048 | 4K | median KL | +0.017 | [-0.036, 0.437] |
| `kvarn-k6v5` | 2048 | 4K | large-deviation rate | -0.145 | [-0.178, 0.225] |
| `kvarn-k6v5` | 2048 | 4K | top-1 disagreement rate | -0.160 | [-0.350, 0.875] |
| `kvarn-k6v5` | 2048 | 32K | mean KL | -0.004 | [-0.244, 0.641] |
| `kvarn-k6v5` | 2048 | 32K | median KL | -0.019 | [-0.036, 0.018] |
| `kvarn-k6v5` | 2048 | 32K | large-deviation rate | +0.087 | [-0.167, 0.667] |
| `kvarn-k6v5` | 2048 | 32K | top-1 disagreement rate | -0.117 | [-0.431, 0.400] |

Worst position per preset: `kvarn-k4v4` tail 256 p = +0.478 (32K large-deviation rate; budget ≤ 1, memory-first), `kvarn-k5v4` tail 2048 p = +0.109 (32K large-deviation rate; recorded, no budget), `kvarn-k6v5` tail 2048 p = +0.087 (32K large-deviation rate; budget ≤ 0.15, quality-first). Depth 128K keeps the G0 measurement above (not re-judged).

Artifacts: `~/.cache/gufo/kvarn-quality/t08/t08-reduced/{units,summary.json}`
(the `ruler` field records the anchor label). Reproduce:

```sh
python3 tools/qwen27b/kvarn_quality.py report \
  --out ~/.cache/gufo/kvarn-quality/t08 --label t08-reduced \
  --anchors-out ~/.cache/gufo/kvarn-quality --anchors-label g0-tail --gate
```

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

## Declared limits for the decode attention blocks

The FP16 decode and verification block kernel (change
`update-qwen-decode-attention-blocks`) decides "matches within the declared
limit" with the values below; the same values are source constants in the
tests, and this change does not loosen any of them.

| Check | Limit | Source | Rationale |
| --- | --- | --- | --- |
| Block partitions against the FP64 reference, including partitions without keys | max abs 5e-4 | `kDecodeAttentionFp64MaxAbsTolerance` in `tests/models/qwen/hip/attention/decode_blocks_ops_test.cpp` | The same contract the FP16 long-context decode test already holds; the block kernel must not relax it. |
| Partition invariants (128-token boundaries, at most 32 partitions, one partition per position) | exact | `tests/models/qwen/hip/attention/decode_partition_policy_test.cpp` | Policy arithmetic only; no tolerance is meaningful. |
| Rows of one verification launch against the same row run alone, and mixed-partition rows | byte-identical | `qwen_attention_kv_storage_ops_test`, `qwen_attention_decode_blocks_ops_test` | The scalar-decode bit contract; a tolerance would hide an ordering change. |

Reproduce: `cmake --build --preset gpu-test --target qwen_attention_decode_partition_test qwen_attention_decode_blocks_ops_test` then `ctest --preset gpu-full -R '^qwen_attention_(decode_partition|decode_blocks)'` (the partition test is CPU-only; add `-DENGINE_ENABLE_HIP=OFF` to run it without a GPU).
