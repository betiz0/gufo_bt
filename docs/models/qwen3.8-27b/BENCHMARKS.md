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

AMD Strix Halo `gfx1151`, Qwen27B Q8_0, `gufo bench --n-prompt 2048 --n-gen 128`,
one warmed measurement per point, ratio against the mean of the two FP16 runs
that bracket each depth (design D16). Contract: AR tg ≥ 0.95× FP16 at depth
0–4K, ≥ 1.00× at depth 32K and above, and at 128K for the memory-first preset
`kvarn-k4v4` R ≥ 1 + f·G* with f = 0.5, where G* is the ideal gain computed
from the same measurement series (design D18).

<!-- bench:kvarn-ar -->
| Qwen27B Q8 AR tg128 (tok/s) | d0 | d4,096 | d32,768 | d131,072 |
| --- | ---: | ---: | ---: | ---: |
| fp16 (reference) | 7.69 | 7.62 | 7.19 | 6.04 |
| kvarn-k4v4, tail unset | 7.66 | 7.50 | 6.65 | 4.75 |
| kvarn-k4v4, tail 2048 | 7.66 | 7.52 | 6.60 | 4.71 |
| kvarn-k5v4, tail unset | 7.66 | 7.47 | 6.45 | 4.34 |
| kvarn-k5v4, tail 2048 | 7.66 | 7.49 | 6.38 | 4.32 |
| kvarn-k6v5, tail unset | 7.66 | 7.42 | 6.07 | 3.66 |
| kvarn-k6v5, tail 2048 | 7.66 | 7.44 | 6.03 | 3.65 |

Ratios to the FP16 reference: 0.996 at d0, 0.974–0.987 at d4K, 0.839–0.926 at
d32K, and 0.605–0.787 at d128K. Only the d0 and d4K floors (≥ 0.95) are met; the
d32K and d128K floors (≥ 1.00, and R ≥ 1 + 0.5·G* = 1.092/1.090 for
`kvarn-k4v4` at d128K, design D18) are not, so G1 fails and the performance
work stops for a user decision (D18).

Measured 2026-10-03 with the host toolchain (ROCm 7.14, host glibc 2.43; Q8_0,
`artifacts/g1sys`): the `nix develop` build links ROCm's raw `hipClang` without
the nixpkgs compiler wrapper and mixes host glibc 2.43 with the store's glibc
2.42, so that binary cannot start here. `python3 tools/qwen27b/g1.py evaluate
--artifacts artifacts/g1sys` reproduces the table.
<!-- /bench -->

Measured scope: one `gufo bench --n-prompt 2048 --n-gen 128` run per point, so
the difference to the FP16 point of the same depth is the KVarN read-path cost
(the model's projections, the recurrent layers and the KV writes are otherwise
identical). At d32K that cost is 17.6 ms of a 148.6 ms token (was 41.4 ms with
the per-key decoder of the phase-1 baseline) and at d128K 73 ms of a 204 ms
token (was 166 ms); FP16's read of the same keys costs 8 ms and 35 ms.

That is the whole gap: FP16's read is bandwidth-bound at the machine's
~245 GB/s, moving 8.6 GB of FP16 KV per token, while the `kvarn-k4v4` body is
2.1 GB. The contract's 1.15× at d128K therefore allows the KVarN read about
14 ms in total, only ~5 ms above its own memory floor. The per-key online
softmax alone (the 256-key dot product, the value accumulation, the five-step
lane reduction and three exponentials, for 24 heads x 131072 keys x 16
full-attention layers) issues roughly 370 warp instructions per query head and
key at about 62% issue efficiency, which is more than that budget; the tile
staging that removes two thirds of those instructions instead costs occupancy
(74 -> 107 VGPRs; see [experiments](EXPERIMENTS.md)) and measured slower.

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
