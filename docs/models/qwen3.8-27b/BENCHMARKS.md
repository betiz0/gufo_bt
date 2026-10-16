# Qwen27B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Gufo `628ed18e`
versus llama.cpp `b11069`, using the same Q4_K_XL / Q8_K_XL targets.
Both use the Q4_K_M DFlash2 draft; Gufo uses its adaptive controller.
HTTP, greedy, thinking off. Measured September 23, 2026.

Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar-q4 -->
| Qwen27B Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 656.33 | 357.54 | +83.6% | 12.37 | 12.06 | +2.6% |
| 4,096 | 633.16 | 330.42 | +91.6% | 12.21 | 11.88 | +2.8% |
| 8,192 | 615.06 | 316.09 | +94.6% | 12.03 | 11.73 | +2.6% |
| 12,288 | 594.61 | 302.54 | +96.5% | 11.88 | 11.59 | +2.5% |
| 16,384 | 575.50 | 290.23 | +98.3% | 11.72 | 11.44 | +2.4% |
| 32,768 | 505.07 | 246.97 | +104.5% | 11.13 | 10.89 | +2.2% |
| 65,536 | 386.92 | 193.61 | +99.8% | 10.13 | 9.93 | +2.0% |
| 131,072 | 287.12 | 135.95 | +111.2% | 8.52 | 8.46 | +0.7% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q4.svg)

---

<!-- bench:single-ar-q8 -->
| Qwen27B Q8 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 485.26 | 337.08 | +44.0% | 7.20 | 7.20 | +0.0% |
| 4,096 | 470.80 | 317.96 | +48.1% | 7.14 | 7.14 | +0.0% |
| 8,192 | 450.84 | 306.22 | +47.2% | 7.08 | 7.08 | +0.0% |
| 12,288 | 436.37 | 289.19 | +50.9% | 7.03 | 7.02 | +0.1% |
| 16,384 | 423.19 | 276.91 | +52.8% | 6.97 | 6.97 | +0.0% |
| 32,768 | 381.43 | 236.32 | +61.4% | 6.76 | 6.76 | +0.0% |
| 65,536 | 311.47 | 186.67 | +66.9% | 6.38 | 6.38 | +0.0% |
| 131,072 | 248.60 | 133.34 | +86.4% | 5.71 | 5.74 | -0.5% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar-q8.svg)

## Single user, DFlash2

pp is the highest measured rate per engine and depth across mixed/repetitive
text. Gufo retains AR output; llama.cpp differs in some controls
([quality details](QUALITY.md#meaning-of-exact)).

<!-- bench:single-dflash2-q4 -->
| Qwen27B Q4 DFlash2<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 607.27 | 347.51 | +74.7% | 26.13 | 23.40 | +11.7% | 63.12 | 35.86 | +76.0% |
| 4,096 | 559.04 | 316.58 | +76.6% | 27.06 | 23.03 | +17.5% | 52.24 | 34.47 | +51.6% |
| 8,192 | 537.00 | 302.47 | +77.5% | 26.77 | 21.73 | +23.2% | 61.55 | 34.73 | +77.2% |
| 12,288 | 520.81 | 289.98 | +79.6% | 23.92 | 23.05 | +3.8% | 53.18 | 34.19 | +55.5% |
| 16,384 | 504.59 | 278.53 | +81.2% | 23.78 | 21.84 | +8.9% | 49.67 | 33.75 | +47.2% |
| 32,768 | 448.96 | 239.21 | +87.7% | 19.25 | 18.88 | +2.0% | 47.76 | 31.75 | +50.4% |
| 65,536 | 347.29 | 190.92 | +81.9% | 17.70 | 18.53 | -4.5% | 29.44 | 26.33 | +11.8% |
| 131,072 | 274.07 | 133.37 | +105.5% | 14.70 | 14.58 | +0.8% | 26.83 | 22.93 | +17.0% |
<!-- /bench -->

![Single user, DFlash2](artifacts/charts/single-dflash2-q4.svg)

---

<!-- bench:single-dflash2-q8 -->
| Qwen27B Q8 DFlash2<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 457.05 | 333.85 | +36.9% | 16.08 | 15.00 | +7.2% | 46.84 | 23.83 | +96.6% |
| 4,096 | 423.09 | 310.23 | +36.4% | 16.36 | 16.12 | +1.5% | 40.51 | 23.59 | +71.7% |
| 8,192 | 408.06 | 292.86 | +39.3% | 15.57 | 15.27 | +2.0% | 41.53 | 23.21 | +78.9% |
| 12,288 | 401.07 | 280.40 | +43.0% | 14.99 | 13.21 | +13.5% | 40.45 | 22.10 | +83.0% |
| 16,384 | 390.69 | 269.03 | +45.2% | 14.28 | 13.83 | +3.3% | 39.39 | 22.56 | +74.6% |
| 32,768 | 355.59 | 231.89 | +53.3% | 14.41 | 14.44 | -0.2% | 35.87 | 21.16 | +69.5% |
| 65,536 | 293.81 | 185.73 | +58.2% | 10.87 | 12.62 | -13.9% | 25.45 | 19.03 | +33.7% |
| 131,072 | 237.24 | 131.80 | +80.0% | 8.87 | 11.62 | -23.7% | 23.41 | 17.23 | +35.9% |
<!-- /bench -->

![Single user, DFlash2](artifacts/charts/single-dflash2-q8.svg)

## Multiple users, autoregressive

Context 4096 per user, tg128. Throughput sums individual request decode rates,
excluding prefill and scheduling. One AR workload per concurrency.

<!-- bench:multi-ar-q4 -->
| Qwen27B Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 12.42 | 12.20 | +1.8% |
| 2 | 23.84 | 22.58 | +5.6% |
| 4 | 44.23 | 38.94 | +13.6% |
| 6 | 60.99 | 47.45 | +28.5% |
| 8 | 74.46 | 50.14 | +48.5% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q4.svg)

---

<!-- bench:multi-ar-q8 -->
| Qwen27B Q8 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 7.22 | 7.24 | -0.3% |
| 2 | 14.45 | 13.71 | +5.4% |
| 4 | 27.81 | 25.32 | +9.8% |
| 6 | 39.96 | 34.64 | +15.4% |
| 8 | 51.62 | 41.57 | +24.2% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar-q8.svg)

## Multiple users, DFlash2

Mixed prompts and repetitive output, context 4096 per user, up to 128 output
tokens. Rates sum individual request decode rates. No prompt-cache hits.

<!-- bench:multi-dflash2-q4 -->
| Qwen27B Q4 DFlash2<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36.30 | 27.15 | +33.7% | 70.56 | 37.56 | +87.9% |
| 2 | 56.34 | 40.04 | +40.7% | 105.51 | 51.65 | +104.3% |
| 4 | 70.87 | 74.43 | -4.8% | 110.49 | 93.38 | +18.3% |
| 6 | 79.83 | 77.19 | +3.4% | 113.52 | 97.52 | +16.4% |
| 8 | 85.39 | 71.71 | +19.1% | 123.00 | 113.56 | +8.3% |
<!-- /bench -->

![Multiple users, DFlash2](artifacts/charts/multi-dflash2-q4.svg)

---

<!-- bench:multi-dflash2-q8 -->
| Qwen27B Q8 DFlash2<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 24.65 | 17.54 | +40.5% | 51.23 | 24.09 | +112.7% |
| 2 | 42.87 | 31.64 | +35.5% | 86.58 | 41.60 | +108.1% |
| 4 | 48.44 | 43.88 | +10.4% | 102.98 | 58.92 | +74.8% |
| 6 | 54.77 | 53.14 | +3.1% | 109.42 | 72.40 | +51.1% |
| 8 | 65.52 | 60.64 | +8.0% | 118.07 | 86.25 | +36.9% |
<!-- /bench -->

![Multiple users, DFlash2](artifacts/charts/multi-dflash2-q8.svg)

## Loading time

C1, context capacity 262144, DFlash2. Cold target/draft files to HTTP readiness.

<!-- bench:loading -->
| Qwen27B<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 5.87 | 23.33 | +297.4% |
| Q8 | 8.17 | 36.47 | +346.4% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## KVarN KV read path

AMD Strix Halo `gfx1151`, Qwen27B Q8_0
(`Qwen3.8-27B-Q8_0.gguf`, SHA-256 `d17816b6…`), `gufo bench --n-prompt 2048
--n-gen 128 --n-depth …`, three measurement series, one warmed run per point.
`pp` is the median of the three series, `tg` comes from the same series, and
FP16 KV is measured inside every series (designs D16/D19).

Contract (D19): the floors are AR tg ≥ 0.95× FP16 at depth 0–4K, DFlash2 tg
≥ 0.95× at depth 0, C8 aggregate throughput ≥ 0.95×, startup time worsening
≤ 5%, and generation token-to-token p99/p50 ≤ 1.05× FP16's. Depth-4K
DFlash2 tg is record-only (user decision D24): no floor, no baseline.
Depth 32K and above AR tg and all-depth pp have no floor: the FP16
ratios below are the recorded **reference values**, and later judgments
(G2, G4/T-09) must keep the same point
at ≥ 0.97× of them. KVarN is a KV-memory feature and is expected to be slower
than FP16 KV for long-context single-session generation.

**Deferred judgments (design D8, user decision 2026-10-09).** The change
`improve-kvarn-k6v5-quality-and-speed` judges none of the existing speed
contracts: the depth 0–4K AR tg floor, the depth 32K and above tg and
all-depth pp reference-value regressions, DFlash2's depth-0 floor,
token-to-token p99/p50, C8 aggregate throughput and startup time are recorded
there and re-measured on the post-change build by a follow-up change. The
change records the depth 8K and 128K points for `kvarn-k6v5` tail 2048 with
FP16 in the same series, and every reading or quantization change it adopts
must keep 8K and 128K tg and pp at ≥ 0.97× of that record.

**Change record (task 3.5, published by task 9.3).** Three serial series of
`g1.py run --foreground --depths 8192,131072 --configs k6v5-t2048` with FP16 KV
measured inside every series (Q8_0, SHA-256
`d17816b6bafbe3f5c572ed94d9cee62516cf901628d4372fb73792a4a175f4cf`), ledger
`T3.5-r1..r3` in `artifacts/g1-k6v5-8k128k-r*`. The depth-8K point is new; the
128K tg ratio reproduces the G1 value exactly, and the 128K pp ratio is below
the G1 record (0.947× of 0.830), which the deferral above leaves unjudged.
`fp16` is the mean of the two bracket runs of the series, `tg` is the tg128
ratio and `pp` the pp2048 ratio of `kvarn-k6v5` tail 2048 against that bracket.

| `kvarn-k6v5`, tail 2048 | 8K tg | 8K pp | 128K tg | 128K pp |
| --- | ---: | ---: | ---: | ---: |
| series r1 | 0.960 | 0.863 | 0.631 | 0.787 |
| series r2 | 0.959 | 0.865 | 0.631 | 0.786 |
| series r3 | 0.959 | 0.865 | 0.631 | 0.784 |
| median (published method) | 0.959 | 0.865 | 0.631 | 0.786 |

<!-- bench:kvarn-ar -->
| Qwen27B Q8 AR tg128 (tok/s) / FP16 ratio | d0 | d4,096 | d32,768 | d131,072 |
| --- | ---: | ---: | ---: | ---: |
| fp16 (reference) | 7.70 | 7.64 | 7.20 | 6.04 |
| kvarn-k4v4, tail unset | 7.67 / 0.996 | 7.53 / 0.986 | 6.83 / 0.949 | 5.09 / 0.843 |
| kvarn-k4v4, tail 2048 | 7.67 / 0.996 | 7.54 / 0.987 | 6.79 / 0.943 | 5.06 / 0.838 |
| kvarn-k5v4, tail unset | 7.67 / 0.996 | 7.50 / 0.982 | 6.58 / 0.914 | 4.55 / 0.753 |
| kvarn-k5v4, tail 2048 | 7.67 / 0.996 | 7.52 / 0.984 | 6.55 / 0.910 | 4.54 / 0.752 |
| kvarn-k6v5, tail unset | 7.67 / 0.996 | 7.44 / 0.974 | 6.19 / 0.860 | 3.82 / 0.632 |
| kvarn-k6v5, tail 2048 | 7.67 / 0.996 | 7.48 / 0.979 | 6.15 / 0.854 | 3.81 / 0.631 |

| Qwen27B Q8 pp2048 (tok/s) / FP16 ratio | d0 | d4,096 | d32,768 | d131,072 |
| --- | ---: | ---: | ---: | ---: |
| fp16 (reference) | 328.89 | 320.63 | 283.93 | 204.50 |
| kvarn-k4v4, tail unset | 304.09 / 0.925 | 295.78 / 0.922 | 252.69 / 0.890 | 168.97 / 0.826 |
| kvarn-k4v4, tail 2048 | 323.85 / 0.985 | 295.51 / 0.922 | 253.91 / 0.894 | 171.74 / 0.840 |
| kvarn-k5v4, tail unset | 304.05 / 0.924 | 295.46 / 0.921 | 252.41 / 0.889 | 168.13 / 0.822 |
| kvarn-k5v4, tail 2048 | 324.11 / 0.985 | 295.39 / 0.921 | 253.77 / 0.894 | 170.85 / 0.835 |
| kvarn-k6v5, tail unset | 303.88 / 0.924 | 295.30 / 0.921 | 251.71 / 0.887 | 166.88 / 0.816 |
| kvarn-k6v5, tail 2048 | 323.91 / 0.985 | 295.10 / 0.920 | 253.00 / 0.891 | 169.65 / 0.830 |

The tg floor passes for every preset and tail (0.974–0.996 at depth 0–4K), so
G1 is met under D19. The depth 32K/128K tg ratios (0.632–0.949) and the
all-depth pp ratios (0.816–0.985) are the reference values G1 records; later
judgments must stay at ≥ 0.97× of the same point. The depth-0 `pp` spread
between tail unset and tail 2048 (0.925 against 0.985) is recorded as measured.

**G1R: the tg reference values re-measured in the block-kernel build (2026-10-15,
task 2.15, design D17).** The FP16 decode and verification arithmetic changed
(change `update-qwen-decode-attention-blocks`, Phase 1), so the tg denominator
of the performance contract is re-measured before any KVarN read change. Three
serial `g1.py` series at depths 32,768 and 131,072 with FP16 KV bracketed inside
every series (AR: `artifacts/g1r-ar-r{1,2,3}`; DFlash2:
`artifacts/g1r-g2-r{1,2,3}`, `--speculative dflash2`), greedy, Q8_0 weights
SHA-256 `d17816b6bafbe3f5c572ed94d9cee62516cf901628d4372fb73792a4a175f4cf`,
candidate binary `build/release-fresh/gufo`, medians of the three series. The pp
reference values stay as measured above (prefill attention is out of the
change's scope). FP16 tg in these series: AR 7.17 (d32,768) and 6.00 (d131,072)
tok/s; DFlash2 39.62 and 7.82 tok/s.

| AR tg128 FP16 ratio (new reference) | k4v4 none | k4v4 2048 | k5v4 none | k5v4 2048 | k6v5 none | k6v5 2048 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d32,768, previous G1 value | 0.949 | 0.943 | 0.914 | 0.910 | 0.860 | 0.854 |
| d32,768, G1R | 0.9484 | 0.9428 | 0.9163 | 0.9121 | 0.8619 | 0.8577 |
| d131,072, previous G1 value | 0.843 | 0.838 | 0.753 | 0.752 | 0.632 | 0.631 |
| d131,072, G1R | 0.8450 | 0.8400 | 0.7567 | 0.7550 | 0.6367 | 0.6350 |

| DFlash2 tg128 FP16 ratio (new reference) | k4v4 none | k4v4 2048 | k5v4 none | k5v4 2048 | k6v5 none | k6v5 2048 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d32,768, previous G2 value | 0.628 | 0.617 | 0.626 | 0.606 | 0.579 | 0.581 |
| d32,768, G1R | 0.6123 | 0.6032 | 0.6098 | 0.5914 | 0.5644 | 0.5671 |
| d131,072, previous G2 value | 0.807 | 0.767 | 0.741 | 0.737 | 0.347 | 0.667 |
| d131,072, G1R | 0.7685 | 0.7302 | 0.7059 | 0.7020 | 0.3312 | 0.6394 |

The AR ratios move by at most 0.005 (the AR decode is bandwidth-bound, design
D13). The DFlash2 ratios fall by 2.2–4.8% relative, mostly because the same
block kernel makes the eight-row FP16 verification faster: FP16 DFlash2 tg
rises from 38.36 to 39.62 tok/s at 32,768 and from 7.56 to 7.82 tok/s at
131,072 (the alternating G1F base/candidate runs of the default FP16 DFlash2
measured +0.9% at 32,768 and +1.2% at 131,072). The quantized-KV DFlash2 presets
rise by 0.5–1.1% at 32,768 but move by -0.8% to -1.6% at 131,072 against the
previous G2 series. That series was measured in another session, so the
comparison is not matched; the 131,072 movement is below the 5% and 3 ms
significance band and is recorded as unconfirmed, not as a regression. The
matched G1F AR pair at 131,072 shows the same sign (-0.8%), also below the band.
Later judgments keep the same points at ≥ 0.97× of these G1R values; the pp
values stay at ≥ 0.97× of the G1 values above.

Reproduce (candidate release build, one series at a time; three series each):

```sh
python3 tools/qwen27b/g1.py run --foreground --binary build/release-fresh/gufo \
  --depths 32768,131072 --out artifacts/g1r-ar-r<N>
python3 tools/qwen27b/g1.py run --foreground --binary build/release-fresh/gufo \
  --depths 32768,131072 --out artifacts/g1r-g2-r<N> \
  --extra-args "--speculative dflash2 --dflash-model <DFlash2-Q8_0>"
python3 tools/qwen27b/g1_d19.py --artifacts artifacts --series-prefix g1r-ar-r \
  --json artifacts/g1r-ar-summary.json
python3 tools/qwen27b/g1_d19.py --artifacts artifacts --series-prefix g1r-g2-r \
  --json artifacts/g1r-g2-summary.json
```

`g1_d19.py` also checks the depth 0–4K tg floor, which these series do not
measure, so its exit status reports those points as FAIL while the 32K/128K
table it prints is the one recorded here.

Measured 2026-10-03 with the host toolchain (ROCm 7.14, host glibc 2.43): the
`nix develop` build links ROCm's raw `hipClang` without the nixpkgs compiler
wrapper and mixes host glibc 2.43 with the store's glibc 2.42, so that binary
cannot start here. Reproduce with:
`for r in 1 2 3; do python3 tools/qwen27b/g1.py run --binary build/release/gufo
--out artifacts/g1d19-r$r; python3 tools/qwen27b/g1.py wait --out
artifacts/g1d19-r$r; done && python3 tools/qwen27b/g1_d19.py --artifacts
artifacts` (logs under `artifacts/g1d19-r{1,2,3}`, summary
`artifacts/g1d19-summary.json`).
<!-- /bench -->

DFlash2 (speculative decoding), re-measured 2026-10-05 after the
verification-store fix (tasks 7.8, designs D18/D21), same series method,
`gufo bench --speculative dflash2 --dflash-model <DFlash2-Q8_0>`: FP16 tg
23.48 / 17.97 / 38.36 / 7.56 t/s at d0 / 4K / 32K / 128K; the KVarN
DFlash2 tg FP16 ratios are d0 0.994–0.995, 4K 0.769–0.971, 32K
0.579–0.628, 128K 0.347–0.807. The D19 depth-0 DFlash2 floor (0.95) is
met for every preset and tail, so **G2 passes under D19**. The pp re-check
passes: every G2 pp ratio is ≥ 0.998× of the G1 recorded reference (floor
0.97×). Depth 4K DFlash2 tg is record-only (user decision D24); depth
32K+ tg is the reference value G2 records, and later judgments (G4/T-09)
must keep the same 32K+ point at ≥ 0.97× of it.

Effect of the store fix against the pre-fix (invalid) runs: depth-0 tg
0.587–0.588 → 0.994–0.995 (the defect collapsed the depth-0 acceptance to
0.277 against FP16's 0.460, D20, while the per-cycle time was already
equivalent at 158.3 against 155.6 ms, so the gap was the acceptance rate),
4K 0.728–0.866 → 0.769–0.971 (5 of 6 improved; `kvarn-k4v4` tail 2048
0.800 → 0.769), 32K unchanged (0.574–0.627 → 0.579–0.628), 128K
0.367–0.439 → 0.667–0.807 for 5 of 6 (`kvarn-k6v5` tail unset
0.383 → 0.347; at that point the now-real ring rows are written and
sealed every cycle, which the pre-fix runs never did). The pre-fix values,
cycle counts and acceptances stay recorded as invalid in
[experiments](EXPERIMENTS.md). The AR tg and pp numbers are unaffected
(they use the single-row decode store). Reproduce:

```sh
for r in 1 2 3; do
  python3 tools/qwen27b/g1.py run --binary build/release/gufo \
    --out artifacts/g2fix-r$r \
    --extra-args "--speculative dflash2 --dflash-model <DFlash2-Q8_0>"
  python3 tools/qwen27b/g1.py wait --out artifacts/g2fix-r$r
done && python3 tools/qwen27b/g1_d19.py --artifacts artifacts \
  --series-prefix g2fix-r --json artifacts/g2fix-summary.json
```
(logs under `artifacts/g2fix-r{1,2,3}`, summary
`artifacts/g2fix-summary.json`).

Measured scope: the difference to the FP16 point of the same depth is the KVarN
read-path cost (the model's projections, the recurrent layers and the KV writes
are otherwise identical). With the retained per-key read (`kKeyBlock = 1`), that
cost is 16.0 ms of a 146.4 ms token for `kvarn-k4v4` tail unset at d32K and
66.1 ms of a 196.5 ms token at d128K; FP16's read of the same keys costs 9.0 ms
and 35.7 ms. FP16 is bandwidth-bound (8.6 GB of KV per token at ~240 GB/s) while
the `kvarn-k4v4` body is 2.4 GB, so the memory saving is real, but the per-key
dequantization compute dominates: measured cache-resident it is 53.9 ms per 128K
token for `kvarn-k4v4` (121.5 ms for `kvarn-k6v5`) against FP16's 33.9 ms. The
key-block variants (2/4/8 keys) all measured slower than per-key. See
[experiments](EXPERIMENTS.md) for the sweep, the compute-only measurement, and
the memory-side motivation (`proposal.md`).

### T-09 reduced re-measurement (2026-10-08, tasks 6.3, design D23)

Release build (host ROCm 7.14), Q8_0 weights, one series with the FP16
brackets measured in the same series (pp2048 median of three, tg128 once
per point). Reused from the earlier judgments: `artifacts/g1d19-r{1,2,3}`
(G1: AR tg and pp for every preset, tail and depth 0–128K) and
`artifacts/g2fix-r{1,2,3}` (G2: DFlash2 tg). The main configuration's AR
tg and pp were re-measured here because the verification-store fix
(tasks 7.1) came after G1; the reference column is the G1 ratio for the
same point (design D19: keep the point at ≥ 0.97× of it).

| Point | t/s | FP16 t/s | ratio | floor / reference | verdict |
| --- | ---: | ---: | ---: | --- | --- |
| `kvarn-k5v4` tail 2048, AR tg128 d0 | 7.66 | 7.69 | 0.996 | ≥ 0.95 (floor) | pass |
| `kvarn-k5v4` tail 2048, pp2048 d0 | 323.72 | 328.13 | 0.987 | ≥ 0.97 × 0.985 (G1) | pass |
| `kvarn-k5v4` tail 2048, AR tg128 d4096 | 7.51 | 7.63 | 0.984 | ≥ 0.95 (floor) | pass |
| `kvarn-k5v4` tail 2048, pp2048 d4096 | 295.24 | 319.53 | 0.924 | ≥ 0.97 × 0.921 (G1) | pass |
| `kvarn-k5v4` tail 2048, AR tg128 d32768 | 6.54 | 7.20 | 0.908 | ≥ 0.97 × 0.910 (G1) | pass |
| `kvarn-k5v4` tail 2048, pp2048 d32768 | 252.83 | 284.38 | 0.889 | ≥ 0.97 × 0.894 (G1) | pass |

Startup to the first successful `/ready` (spawn → ready, warm page cache,
3 runs each): `kvarn-k5v4` tail 2048 1.25–1.26 s, FP16 1.25–1.26 s → worsening 0% (criterion ≤ 5%).

Token-to-token latency during a 128-token generation after a 2,564-token
prompt (which crosses the first seal at tail 2048), 3 repetitions each;
the server reports the per-request mean ITL, so the percentiles are across
requests. Criterion: KVarN's p99/p50 ≤ 1.05× FP16's (design D19).

| Config | ITL p50 ms | ITL p99 ms | p99/p50 | peak GiB |
| --- | ---: | ---: | ---: | ---: |
| `kvarn-k5v4` tail 2048 | 133.30 | 133.50 | 1.0015 | 22.01 |
| `kvarn-k6v5` tail 2048 | 133.83 | 133.88 | 1.0004 | 22.08 |
| FP16 | 131.30 | 131.79 | 1.0037 | 22.10 |

HIP peak allocation at the full 262,144-token KV capacity (device-global
`hipMemGetInfo` peak, the counter the memory table above uses; the KV
reservation dominates the difference): `kvarn-k5v4` tail 2048 25.75 GiB, `kvarn-k6v5` tail 2048 26.43 GiB, FP16 36.49 GiB. At the 4,096-token context used for the latency runs all three peak at
~22.0 GiB, which is the device-global baseline rather than the session
footprint.

C8 aggregate AR throughput (context 4096 per user, eight concurrent ~3.4K
prompts, 3 repetitions; criterion ≥ 0.95× FP16):

| Config | aggregate output t/s | per-request whole t/s | peak GiB |
| --- | ---: | ---: | ---: |
| `kvarn-k5v4` tail 2048 | 8.678 | 31.14 | 48.12 |
| FP16 | 9.269 | 33.10 | 47.99 |

The C8 criterion is **missed**: the aggregate output ratio is 0.936 against the 0.95 floor. Per-request medians: decode 5.553 vs 6.078 t/s (ratio 0.914), whole request 31.14 vs 33.10 t/s; both runs queued ~11 s per request with physical execution width 8. At concurrency 1 the same configuration's decode ratio is 0.985 (T6.3D4), so the deficit appears when eight sessions share the GPU. The user decision on 2026-10-08 accepts this as-is: the task is completed with the miss recorded (no criterion change, no further optimization, no pull request).

(a) `CheckKvarnBatchDecodeSingleVsTwo` (dedicated multi-session suite,
design D21, `artifacts/test-ledger.jsonl` run `T6.3A`): the 2-session batch
decode matches the C1 reference for `kvarn-k4v4` and `kvarn-k5v4` at tails
unset and 2048 (64 continuation steps, boundaries 265 and 2304).

## Memory occupation

C1, context capacity 262144, AR. Peak memory reported by HIP.

<!-- bench:memory-q4 -->
| Qwen27B Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 37.63 | 36.19 | -3.8% |
| 16K prefix, pp4096 + tg128 | 39.88 | 36.62 | -8.2% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q4.svg)

---

<!-- bench:memory-q8 -->
| Qwen27B Q8 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 50.58 | 48.60 | -3.9% |
| 16K prefix, pp4096 + tg128 | 52.83 | 49.02 | -7.2% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory-q8.svg)
