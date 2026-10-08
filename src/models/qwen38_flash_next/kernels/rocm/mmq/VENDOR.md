# Quantized HIP kernels

Adapted from `ggml-org/llama.cpp` at
`5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` (MIT; see `LICENSE`).
The upstream source directory is `ggml/src/ggml-cuda`; Gufo builds these
sources with HIP for AMD Strix Halo (`gfx1151`).

`qfn_mmq.h` is the public adapter. It supplies Q4_K, Q5_K, Q5_0, Q8_0, Q6_K and
IQ4_NL dense projections (tiled MMQ over F32 activations, plus the decode GEMV
over pre-quantized Q8_1 activations) and Q4_K, Q5_K, Q5_1, Q5_0, Q8_0 and
IQ4_NL routed expert projections, including the paired Q4_K gate/up path.
IQ4_NL and Q5_0 routed entries serve the down projection only, which is the one
ungated projection the Agention AP-Q4_K_XL and Swift-1.5 Q4_K_M tiers store in
those formats. The Q4_K/Q5_K/Q5_0 dense entries and the Q5_0 routed entry were
added for the Swift-1.5 mix and instantiate the same `qfn_mmq_dense_impl`,
`qfn_dense_mmvq_kernel` and `qfn_mmq_moe_impl` templates as the existing
entries. `qfn_ggml_stubs.*` supplies the minimal context and scratch pool
without the ggml graph runtime. Gufo owns scheduling and HIP graphs.

Local additions to the vendored sources: `mmvq.hip.cpp` lists IQ4_NL and Q5_0
in the MMVQ vector-dot table, its VDR table and the routed dispatch, which is
the upstream capability set the adapter's routed vector entry needs (Q6_K stays
out of the routed tables: no routed kernel executes it). The dense decode GEMV
and the load-time `DecodeWeights` kernel live in `qfn_mmq.hip.cpp` and
`../kernels.hip.cpp`; no vendored kernel body was changed.

The kernels use native HIP runtime, hipBLAS and HIP numeric types. The
upstream runtime aliases, NVIDIA capability dispatch and PTX branches have
been removed; retained matrix fragments use AMD WMMA/MFMA. Headers use the
`.hpp` suffix. Gufo owns scheduling and HIP graphs.

When importing an upstream fix, preserve the Gufo namespace, adapter and
HIP-only dispatch, update the pin, and run the Flash-Next operator tests
under `tests/models/qwen38_flash_next/` plus the AR/MTP session replay.
The routed GEMM test compares supported formats against a floating-point
reference and checks the paired adapter against two separate projections.
Build and validate through Nix as described in `AGENTS.md`.

Validation for the Q6_K/IQ4_NL entries (2026-09-26, gfx1151):
- `qwen38_flash_next.mmq_ops`: Q6_K and IQ4_NL dense MMQ and decode GEMV
  against the FP32 reference, the IQ4_NL routed vector projection against the
  reference, and the paired Q8_0 dense entries unchanged.
- `qwen38_flash_next.projection_ops`: Q6_K dense MMQ and GEMV over the AP
  trunk's shapes (10240x2560, 12288x2560, 2560x6144, 248320x2560) against the
  FP32 reference, with per-shape timings recorded in the model's EXPERIMENTS.md.
- `qwen38_flash_next.routed_wmma_ops`: IQ4_NL down-experts on the tiled WMMA
  route at 16- and 64-token tiles against the FP32 reference, with the adapter's
  vector route asserted to agree on the same input.
- `qwen38_flash_next.embed_ops`: Q6_K embedding rows against the CPU
  dequantization.
- AR session replay: `qwen38_flash_next.session_test` on the unsloth
  UD-Q4_K_XL target with the shared Q8_0 MTP sidecar (retained outputs
  unchanged); the Agention AP-Q4_K_XL trunk loads and generates through the
  same path.

Validation for the Q4_K/Q5_K/Q5_0 entries (2026-09-29, gfx1151):
- `qwen38_flash_next.mmq_ops`: Q4_K dense MMQ and decode GEMV at
  10240x2560, 12288x2560, 2560x6144 and 248320x2560, Q5_K at 2560x6144, and
  Q4_K/Q5_0 at the Mixer shapes 320x10240 and 10240x320, all against the FP32
  reference; the Q5_0 routed vector projection against the reference; the
  Q6_K/IQ4_NL entries and the paired Q8_0 entries unchanged.
- `qwen38_flash_next.routed_wmma_ops`: Q5_0 down-experts on the tiled WMMA
  route at 16- and 64-token tiles against the FP32 reference, with the
  adapter's vector route asserted to agree on the same input, and the
  existing formats unchanged.
- `qwen38_flash_next.embed_ops`: Q4_K and Q5_K embedding rows against the CPU
  dequantization (the Swift trunk's `token_embd` is Q4_K).
- `qwen38_flash_next.hc_mix_ops`: the stored-encoding Mixer chain
  (Q4_K down, Q5_0 up, epilogue) at the tiled and the decode-GEMV entry
  against F64 references.
- `qwen38_flash_next.projection_ops` / `attention_ops`: Q4_K and Q5_K dense
  and attention projections at the Swift trunk's shapes against the FP32
  reference.
