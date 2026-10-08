# Qwen3.8 Flash-Next

Hybrid recurrent/QSA mixture-of-experts text/image model on gfx1151.
Supported target: `unsloth/Qwen3.8-Flash-Next-GGUF`, **UD-Q4_K_XL** (four shards).
Optional shared-Q8 MTP predictor; optional BF16 vision projector.
Original unquantized-model and GGUF-conversion parity remain unqualified.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix develop -c hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf" \
  --local-dir models/qwen3.8-flash-next
nix build
MODEL=/path/to/first-target-shard.gguf
MTP=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 2 --context 32768
```

The loader discovers the remaining shards. Omit the speculative options for AR;
AR sessions allocate no predictor state even if a shared model has MTP loaded.
Adaptive MTP is default, with `--draft-tokens` capping 1–7 proposals. Sampled
requests use deterministic acceptance/cost control for seeded replay; all-greedy
C>1 batches may use measured cycle costs. Each request keeps private caches,
rollback and RNG. See [MTP qualification](QUALITY.md).

The official template defaults to thinking on, `xhigh` effort and preserving
prior reasoning. Use the [reasoning controls](../../SERVER.md#reasoning-controls)
for explicit effort/thinking overrides. Native context is 262144; YaRN extension
is unsupported. Memory grows with used context and selected rollback depth;
admission reserves the configured capacity before creating sessions.

## Agention Precision AP-Q4_K_XL tier

**Not a qualified tier.** This mix loads and executes, but it is not listed as
supported: its teacher-forced agreement with llama.cpp is worse than the
unsloth `UD-Q4_K_XL` baseline's on one of the two retained prompts, so the
qualification gate fails. Keep `UD-Q4_K_XL` as the supported target; the
measurements and the failed gate are in [QUALITY.md](QUALITY.md#agention-ap-q4kxl-tier).

`agentionai/Signal-3.8-Flash-Next-GGUF`, **AP-Q4_K_XL**: a single 94.2 GiB file
whose 1224 tensors are a per-tensor mix of Q6_K 520 (dense projections,
Mixer/SSM/embedding side), IQ4_NL 51 (routed down-experts, output Mixer, PLE
table), Q4_K 96 (routed gate/up), Q8_0 144 (shared-expert down, Mixer up),
BF16 24 (indexer), F16 1 (`ple_conv1d`) and F32 388 (norms). Same 48-block
geometry, tensor names and config as the unsloth target; only the per-tensor
storage formats differ. Stored encodings execute as they are; ~0.63 GiB of
Mixer/norm-side tensors are decoded to F16/F32 at load and the loader reports
that total.

```sh
nix develop -c hf download agentionai/Signal-3.8-Flash-Next-GGUF \
  --include "AP-Q4_K_XL/*" --local-dir models/qwen3.8-flash-next
AP=models/qwen3.8-flash-next/AP-Q4_K_XL/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf
python3 tools/gguf_formats.py "$AP" --sha256   # identity, format and class histogram
./result/bin/gufo prompt --model "$AP" -p "The capital of France is"
```

The measurements and the measured context ceiling are in
[QUALITY.md](QUALITY.md#agention-ap-q4kxl-tier); the file hashes, the per-format
histogram and the per-class histogram are in
[artifacts/model-identities.json](artifacts/model-identities.json).

## Swift-1.5 Q4_K_M tier

**Not a qualified tier.** This mix loads and executes, but it is not listed as
supported: at two of the six (prompt, prefill batch configuration) pairs the
specification's paired teacher-forced gate has the Swift tier's mean KL
divergence above the unsloth `UD-Q4_K_XL` baseline's at the same configuration,
so the gate fails as a whole. Keep `UD-Q4_K_XL` as the supported target; the
measurements and the failed pairs are in
[QUALITY.md](QUALITY.md#swift-15-q4km-tier).

`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF`, **Q4_K_M**: three shards,
119,599,029,952 bytes, 1224 tensors, `general.name` "Swift 1.5 Flash
Miroslav2.0", architecture `qwen4exp` — the same 48-block geometry, tensor
names and config as the unsloth target, with a per-tensor mix of Q4_K 255
(dense `token_embd`/`attn_gate`/`attn_q`/`ssm_out`/`ple_*`, the Mixer down
projections and routed gate/up), Q5_0 121 (Mixer up projections, half the
routed down-experts and the PLE table), Q8_0 162, Q6_K 91, F32 460, BF16 122,
Q5_K 12 (`attn_output`) and F16 1 (`ple_conv1d`). Stored encodings execute as
they are: 84,248,815,104 device weight bytes with 17,858,560 bytes decoded at
load across 37 tensors, and the 35.2 GB Q5_0 `per_layer_token_embd` table stays
on disk (the n-gram reader streams its rows). The six (class, format) pairs
this mix adds to the weight-class table and the wide-Mixer-route guard its
Mixer pair needs are recorded in [EXPERIMENTS.md](EXPERIMENTS.md).

```sh
nix develop -c hf download ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF \
  --revision 0142554c963a8653912c0c8283f05f13f63653f3 \
  --include "Q4_K_M/*" --local-dir models/qwen3.8-flash-next
SWIFT=models/qwen3.8-flash-next/Q4_K_M/Swift-1.5-Qwen3.8-Flash-Next-Q4_K_M-00001-of-00003.gguf
python3 tools/gguf_formats.py "$SWIFT" --sha256   # identity, format and class histogram
./result/bin/gufo prompt --model "$SWIFT" -p "The capital of France is"
```

The per-configuration teacher-forced comparison, the measured context ceiling
and its peak memory, the route pinning and the MTP smoke check are in
[QUALITY.md](QUALITY.md#swift-15-q4km-tier); the shard hashes, the
per-format and per-class histograms and the source revision are in
[artifacts/model-identities.json](artifacts/model-identities.json).

## Images

Use this model's `mmproj-BF16.gguf`, discovered beside the target or selected
with `--mmproj`. PNG/JPEG CLI and HTTP requests use the
[same image interface](../qwen3.8-27b/README.md#images). Image state participates
in prefill, decoding, verification, multi-turn reuse and disk cache identity.
The predictor embeds shifted text IDs; visual information comes from target
hidden states and mRoPE. Image snapshots require matching prompt attachment.

## Tools and artifacts

Model tests are in `tests/models/qwen38_flash_next`, focused microbenchmarks in
`tools/qwen-flash`. [Quality](QUALITY.md) summarizes qualification and focused checks.
Build a microbenchmark with
`nix develop -c tools/bench/build.sh tools/qwen-flash/projection_plans.hip`;
`dense_blaslt_sweep.hip` times hipBLASLt on the dense prefill shapes.
No historical logit dump is required. New retained result summaries belong in
`artifacts/`; generated traces stay in the ignored top-level `artifacts/` tree.
