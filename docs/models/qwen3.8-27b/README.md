# Qwen3.8 27B

Dense text/image model on gfx1151. Production target GGUFs are **UD-Q4_K_XL**
(16.35 GiB weights) and **UD-Q8_K_XL** (29.30 GiB); request state and an optional
draft/projector need additional memory. BF16 targets are reference-only.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix develop -c hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q4_K_XL.gguf mmproj-BF16.gguf \
  --local-dir models/qwen3.8-27b
nix develop -c hf download z-lab/Qwen3.8-27B-DFlash2-GGUF \
  Qwen3.8-27B-DFlash2-Q4_K_M.gguf --local-dir models/qwen3.8-27b
nix build
MODEL=models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf
DRAFT=models/qwen3.8-27b/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --speculative dflash2 \
  --dflash-model "$DRAFT" --sessions 2 --context 32768
```

Substitute `Qwen3.8-27B-UD-Q8_K_XL.gguf` for Q8. Targets and drafts need not
have matching precision. Use the Q4_K_M DFlash2 draft with either target.

| Mode | Selection | Behavior |
| --- | --- | --- |
| AR | No speculative option | Target-only generation. |
| DFlash2 | `--speculative dflash2 --dflash-model PATH` | Adaptive default; `--draft-tokens` caps proposals, `--draft-policy fixed` is a comparison mode. |

Greedy speculation must match AR. Sampled modes preserve the target policy,
but AR and speculation need not share the same-seed sequence. See the
[quality contract](QUALITY.md). Thinking/template controls and HTTP
sampling defaults are described in [the server guide](../../SERVER.md#reasoning-controls).

## KV cache quantization (phase 0)

The KVarN KV-cache quantization change
(`openspec/changes/add-kvarn-kv-cache-quantization`) has passed its phase-0
quality gate (G0, design D17). Presets are selected with `--kv-cache` and the
full-precision tail behind each KVarN query with `--kv-cache-tail`. The fitting
tails are `kvarn-k4v4` 256 (memory-first) and `kvarn-k6v5` 2048 (quality-first);
`kvarn-k5v4` fits the memory-first budget with no tail. Depth 128K is **not
re-judged** in this change: the production-path re-judgment (fresh
`t08-reduced` windows, design D23) covers depths 4K and 32K, so the 128K
figures here remain G0 and are provisional, and statistical task evaluation is
deferred to a follow-up change. The 32K re-judgment rests on the four windows
that survive the pre-registered teacher-NLL rule (the rule drops four of the
eight; the spec's six-window target is a documented deviation in
[quality](QUALITY.md)). KVarN with image tokens
and M-RoPE positions is **unverified**: the vision check needs the
`mmproj-BF16.gguf` sidecar, which the measurement host does not have yet
(design D20, task 5.4), so image requests should keep the default `fp16` KV.
Sizes are design D5's formula at the 262,144-token capacity (16 attention
layers, 4 KV heads), where the session KV capacity in GiB equals the total bpv.
FP16 KV is 16.00 GiB.
"none" means no tail tokens were added (L = W).

| Preset | Tail | Body bpv | Total bpv | KV/session (GiB) |
| --- | ---: | ---: | ---: | ---: |
| `kvarn-k4v4` | none | 4.4375 | 4.452 | 4.45 |
| `kvarn-k4v4` | 256 | 4.4375 | 4.467 | 4.47 |
| `kvarn-k4v4` | 1024 | 4.4375 | 4.514 | 4.51 |
| `kvarn-k4v4` | 2048 | 4.4375 | 4.577 | 4.58 |
| `kvarn-k5v4` | none | 4.9375 | 4.952 | 4.95 |
| `kvarn-k5v4` | 256 | 4.9375 | 4.967 | 4.97 |
| `kvarn-k5v4` | 1024 | 4.9375 | 5.014 | 5.01 |
| `kvarn-k5v4` | 2048 | 4.9375 | 5.076 | 5.08 |
| `kvarn-k5v5` | none | 5.4375 | 5.452 | 5.45 |
| `kvarn-k5v5` | 256 | 5.4375 | 5.467 | 5.47 |
| `kvarn-k5v5` | 1024 | 5.4375 | 5.514 | 5.51 |
| `kvarn-k5v5` | 2048 | 5.4375 | 5.576 | 5.58 |
| `kvarn-k6v5` | none | 5.9375 | 5.951 | 5.95 |
| `kvarn-k6v5` | 256 | 5.9375 | 5.966 | 5.97 |
| `kvarn-k6v5` | 1024 | 5.9375 | 6.013 | 6.01 |
| `kvarn-k6v5` | 2048 | 5.9375 | 6.076 | 6.08 |

Speed (design D19). KVarN trades speed for KV memory: it passes the shallow
floors (AR tg ≥ 0.95× FP16 at depth 0–4K) and is slower than FP16 KV in
long-context generation. The table is the G1 measurement (2026-10-03, Q8_0,
three series, `pp` median of three; see [benchmarks](BENCHMARKS.md)). The depth
32K/128K tg ratios and the all-depth pp ratios are the recorded **reference
values**: later judgments must keep the same point at ≥ 0.97× of them.

| Preset | Tail | d0 | 4K | 32K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| `kvarn-k4v4` | none | 0.996 / 0.925 | 0.986 / 0.922 | 0.949 / 0.890 | 0.843 / 0.826 |
| `kvarn-k4v4` | 2048 | 0.996 / 0.985 | 0.987 / 0.922 | 0.943 / 0.894 | 0.838 / 0.840 |
| `kvarn-k5v4` | none | 0.996 / 0.924 | 0.982 / 0.921 | 0.914 / 0.889 | 0.753 / 0.822 |
| `kvarn-k5v4` | 2048 | 0.996 / 0.987\* | 0.984 / 0.924\* | 0.908 / 0.889\* | 0.752 / 0.835 |

\* Main configuration re-measured in the same series on 2026-10-08 (task 6.3, design D23): AR tg / pp2048 ratios against FP16 at d0 0.996/0.987, d4K 0.984/0.924, d32K 0.908/0.889; depth 128K keeps the G1 value. The floors and the reference checks (≥ 0.95 at d0–4K, ≥ 0.97× the G1 point at 32K) pass.
| `kvarn-k6v5` | none | 0.996 / 0.924 | 0.974 / 0.921 | 0.860 / 0.887 | 0.632 / 0.816 |
| `kvarn-k6v5` | 2048 | 0.996 / 0.985 | 0.979 / 0.920 | 0.854 / 0.891 | 0.631 / 0.830 |

Cells are `tg ratio / pp ratio` against FP16 KV at the same depth. The benefit
is the KV memory of many long-context sessions (16.0 GiB → 4.45–6.08 GiB per
262K session), not single-session long-context speed.

Teacher-forced comparison against the FP16-KV teacher: three corpora, the 256
positions after each 4K/32K/128K prefix, one tokenization and one history
(design D14/D17; corpus 1 / 32K excluded by the pre-registered teacher-NLL
rule). KL is KL(teacher ‖ candidate) in nats, top-1 is teacher-argmax agreement,
ΔNLL is the teacher-token NLL difference (candidate − teacher). No non-finite
logits occurred. The probes are phase-0 approximations (design D13); phase 1
implements the production read path.

| Preset | Tail | Depth | Mean KL | Median KL | P99.9 KL | Top-1 | ΔNLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `kvarn-k4v4` | none | 4K | 0.24927 | 0.000261 | 15.475 | 0.9479 | +0.01343 |
| `kvarn-k4v4` | none | 32K | 0.35534 | 0.000842 | 19.428 | 0.9277 | +0.09796 |
| `kvarn-k4v4` | none | 128K | 0.44308 | 0.000368 | 21.991 | 0.9154 | +0.08163 |
| `kvarn-k4v4` | 256 | 4K\* | 0.02181 | 0.000171 | 1.614 | 0.9834 | -0.00286 |
| `kvarn-k4v4` | 256 | 32K\* | 0.14711 | 0.000278 | 13.003 | 0.9658 | -0.08478 |
| `kvarn-k4v4` | 256 | 128K | 0.29218 | 0.000185 | 17.803 | 0.9375 | +0.04018 |
| `kvarn-k4v4` | 1024 | 4K | 0.09675 | 0.000060 | 10.346 | 0.9727 | -0.00972 |
| `kvarn-k4v4` | 1024 | 32K | 0.21262 | 0.000211 | 17.179 | 0.9473 | -0.06917 |
| `kvarn-k4v4` | 1024 | 128K | 0.27590 | 0.000180 | 17.832 | 0.9427 | +0.12797 |
| `kvarn-k4v4` | 2048 | 4K | 0.04373 | 0.000043 | 6.880 | 0.9792 | +0.00564 |
| `kvarn-k4v4` | 2048 | 32K | 0.16413 | 0.000191 | 13.909 | 0.9473 | -0.01222 |
| `kvarn-k4v4` | 2048 | 128K | 0.16229 | 0.000073 | 15.127 | 0.9518 | +0.01625 |
| `kvarn-k5v4` | none | 4K | 0.23700 | 0.000163 | 13.857 | 0.9583 | -0.05671 |
| `kvarn-k5v4` | none | 32K | 0.17164 | 0.000582 | 16.873 | 0.9570 | -0.00279 |
| `kvarn-k5v4` | none | 128K | 0.21613 | 0.000100 | 18.941 | 0.9466 | -0.05385 |
| `kvarn-k5v4` | 256 | 4K | 0.14301 | 0.000095 | 13.172 | 0.9661 | -0.03229 |
| `kvarn-k5v4` | 256 | 32K | 0.13826 | 0.000214 | 13.153 | 0.9551 | -0.05032 |
| `kvarn-k5v4` | 256 | 128K | 0.20961 | 0.000080 | 16.935 | 0.9570 | +0.06301 |
| `kvarn-k5v4` | 1024 | 4K | 0.06006 | 0.000048 | 8.950 | 0.9714 | +0.01540 |
| `kvarn-k5v4` | 1024 | 32K | 0.14627 | 0.000179 | 15.231 | 0.9512 | +0.02877 |
| `kvarn-k5v4` | 1024 | 128K | 0.13445 | 0.000068 | 16.345 | 0.9596 | +0.05124 |
| `kvarn-k5v4` | 2048 | 4K\* | 0.00799 | 0.000048 | 2.898 | 0.9907 | +0.00631 |
| `kvarn-k5v4` | 2048 | 32K\* | 0.06486 | 0.000068 | 12.713 | 0.9746 | +0.03960 |
| `kvarn-k5v4` | 2048 | 128K | 0.09981 | 0.000049 | 14.053 | 0.9661 | -0.02620 |
| `kvarn-k5v5` | none | 4K | 0.13607 | 0.000078 | 13.494 | 0.9674 | -0.00943 |
| `kvarn-k5v5` | none | 32K | 0.26603 | 0.000223 | 22.323 | 0.9512 | +0.06888 |
| `kvarn-k5v5` | none | 128K | 0.16526 | 0.000057 | 18.579 | 0.9583 | -0.03585 |
| `kvarn-k5v5` | 256 | 4K | 0.06705 | 0.000050 | 9.302 | 0.9753 | +0.04035 |
| `kvarn-k5v5` | 256 | 32K | 0.21720 | 0.000228 | 21.116 | 0.9512 | +0.10954 |
| `kvarn-k5v5` | 256 | 128K | 0.14140 | 0.000055 | 17.297 | 0.9596 | +0.00150 |
| `kvarn-k5v5` | 1024 | 4K | 0.05964 | 0.000034 | 7.945 | 0.9792 | +0.01916 |
| `kvarn-k5v5` | 1024 | 32K | 0.16733 | 0.000118 | 21.738 | 0.9629 | +0.01690 |
| `kvarn-k5v5` | 1024 | 128K | 0.16189 | 0.000043 | 16.857 | 0.9557 | +0.01494 |
| `kvarn-k5v5` | 2048 | 4K | 0.02698 | 0.000020 | 3.645 | 0.9805 | +0.00985 |
| `kvarn-k5v5` | 2048 | 32K | 0.08016 | 0.000112 | 7.796 | 0.9668 | +0.04783 |
| `kvarn-k5v5` | 2048 | 128K | 0.10445 | 0.000036 | 13.538 | 0.9635 | +0.02374 |
| `kvarn-k6v5` | none | 4K | 0.09385 | 0.000053 | 14.071 | 0.9753 | +0.03815 |
| `kvarn-k6v5` | none | 32K | 0.19179 | 0.000138 | 20.771 | 0.9590 | +0.09377 |
| `kvarn-k6v5` | none | 128K | 0.11192 | 0.000046 | 9.961 | 0.9570 | +0.01205 |
| `kvarn-k6v5` | 256 | 4K | 0.11230 | 0.000036 | 16.445 | 0.9727 | +0.05246 |
| `kvarn-k6v5` | 256 | 32K | 0.10684 | 0.000120 | 13.021 | 0.9609 | +0.05789 |
| `kvarn-k6v5` | 256 | 128K | 0.13560 | 0.000044 | 16.691 | 0.9661 | -0.00048 |
| `kvarn-k6v5` | 1024 | 4K | 0.02867 | 0.000025 | 2.966 | 0.9714 | +0.00301 |
| `kvarn-k6v5` | 1024 | 32K | 0.15900 | 0.000101 | 20.632 | 0.9648 | +0.03745 |
| `kvarn-k6v5` | 1024 | 128K | 0.09695 | 0.000028 | 14.456 | 0.9661 | +0.03620 |
| `kvarn-k6v5` | 2048 | 4K\* | 0.00389 | 0.000034 | 0.413 | 0.9897 | +0.00618 |
| `kvarn-k6v5` | 2048 | 32K\* | 0.06358 | 0.000037 | 10.909 | 0.9834 | -0.02267 |
| `kvarn-k6v5` | 2048 | 128K | 0.10306 | 0.000023 | 14.848 | 0.9661 | +0.06929 |

\* Production read path on the fresh `t08-reduced` windows (eight
corpora, seeds 11–18; the same 256 positions after each 4K/32K prefix; task 6.1
under design D23), scored with the q8→q4 positions of design D17 as amended on
2026-10-08 (D-27): the E_q8/E_q4 ruler comes from the G0 windows above, because
the baseline implementation was removed with task 3.10. Windows excluded by the
pre-registered teacher-mean-NLL rule (design D14): corpora 2, 3, 4 and 7 at 32K.
Rows without the marker, and depth 128K for every preset, are the G0 phase-0
probe values and have not been re-judged.


Gate (design D17): each judged metric (mean KL, median KL, large-deviation rate,
top-1 disagreement rate) is mapped to its q8→q4 interpolated position p per
depth, and the worst p must be ≤ 0.15 for the quality-first preset and ≤ 1 for
the memory-first preset. `kvarn-k6v5` tail 2048 passes at worst p 0.143 (128K
large-deviation rate); `kvarn-k4v4` tail 256 passes at worst p 0.971 (128K
large-deviation rate). P99.9 KL and ΔNLL are recorded but not judged.

Reproduce (design D16):

```sh
python3 tools/qwen27b/kvarn_quality.py report --label g0-tail --gate
```

**Re-measured (2026-10-05, tasks 7.8):** with the verification-store fix
the same corpus reports 0/11 AR/DFlash2 mismatches at depth 0 and ~3.9K,
the depth-0 acceptance returns to the FP16 value 0.844, and the re-measured
G2 DFlash2 tg ratios meet the D19 depth-0 floor (0.95× FP16) for every
preset and tail (0.994–0.995×); see [benchmarks](BENCHMARKS.md),
"KVarN KV read path".

DFlash2 and long-context cost (design D20, tasks 4.8). KVarN KV changes where
the target's greedy decisions land: at depth 0 the FP16 and `kvarn-k5v4`
tail-2048 DFlash2 runs emit the same first two tokens (identical sha256,
acceptance and decode steps) and separate at the third, so the divergence is
the target's rotated-space rounding, not the draft wiring. Over the 11-case
corpus at 32 generated tokens the pre-fix acceptance
difference was not significant (depth 0: 0.876 against 0.814, paired
bootstrap 95% CI [-0.198, +0.056]; depth 4096: 0.765 against 0.695,
[-0.169, +0.009]); with the store fixed the depth-0 acceptance returns to
parity (0.844 against 0.844).
At depth 128K the `kvarn: seal` cost is the depth preparation (35 prefill
chunks × 16 layers = 560 seals, 33.63 s at n=1 and 33.65 s at n=128), not
the generation cycles; the generation-side AR cost is the split-K read
(11.65 s per 128 tokens). The pre-fix DFlash2 verification read (30.98 s,
defective store) made AR the better KVarN choice in very long contexts;
with the store fixed, the re-measured 128K DFlash2 tg is faster than AR
for every preset and tail except `kvarn-k6v5` tail unset
(`kvarn-k5v4` tail 2048: 5.57 against 4.54 t/s; `kvarn-k6v5` tail
unset: AR 3.82 against DFlash2 2.62 t/s), so DFlash2 is the better
KVarN choice at every depth for those configurations.
**Outcome (2026-10-05, G2 re-measured):** the DFlash2 depth-0 tg floor
(0.95× FP16) is met — 0.994–0.995× for all six presets and tails — so G2
passes under D19 and no user decision is pending; the depth 4K DFlash2
ratio in [benchmarks](BENCHMARKS.md) is record-only (D24) and the depth
32K+ ratios are the recorded reference values for later judgments
(G4/T-09). The D19 reference values for AR stay
unchanged. Reproduce:

```sh
# (1) short-length agreement, depth 0
GUFO_DISPATCH_TELEMETRY=1 build/release/gufo bench -m MODEL -p 2048 -n 3 -d 0 \
  --kv-cache kvarn-k5v4 --kv-cache-tail 2048 \
  --speculative dflash2 --dflash-model DFLASH --verbose
# (2) corpus acceptance; KVarN needs a --binary wrapper that inserts the flags
python3 tools/bench/speculative-corpus.py --binary build/release/gufo \
  --model MODEL --draft-model DFLASH --backend dflash2 --json OUT.json
# (3) 128K seal split
python3 tools/prof/prof.py run --stages qwen --tag tg -o OUT -- \
  build/release/gufo bench -m MODEL -p 2048 -n 128 -d 131072 \
  --kv-cache kvarn-k5v4 --kv-cache-tail 2048 \
  --speculative dflash2 --dflash-model DFLASH
```

## Images

Use the matching `mmproj-BF16.gguf` beside the target or pass `--mmproj PATH`.
Vision weights upload lazily. Repeat `--image` in `prompt` or `chat`; chat
attaches them to the first user turn. Images count toward context capacity.

```sh
./result/bin/gufo prompt --model "$MODEL" --image photo.jpg \
  --prompt "Describe this image." --temperature 0
```

Send an image to the server:

```python
import base64, json, urllib.request
image = base64.b64encode(open("photo.jpg", "rb").read()).decode()
body = {"model": "vision-test", "temperature": 0, "max_tokens": 64,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}},
            {"type": "text", "text": "Describe this image."}]}]}
request = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(request).read().decode())
```

Start that server with `--served-model-name vision-test`. Ordered text/image
parts, multiple images and later turns work with AR/DFlash2 and caches.
PNG/JPEG data URLs and public HTTPS are supported (`detail: auto`); requests
share a 20 MiB encoded-byte, 16-image and 15-second download budget. Each image
is capped at 32 megapixels; private/loopback/link-local destinations are rejected.
Official dynamic resizing allows 64–16384 merged image tokens.

Image numbering is off by default; `--add-vision-id` or
`chat_template_kwargs: {"add_vision_id": true}` adds `Picture N:` prefixes.
Cache identity includes processed pixels, placement, preprocessing version and
projector contents. See [vision checks and limits](QUALITY.md#vision).
