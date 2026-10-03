#!/usr/bin/env python3
"""G1 AR token-generation sweep and D18 evaluation (openspec change
add-kvarn-kv-cache-quantization, task 3.17).

Replaces the out-of-repo `run_g1.sh` / `evaluate_g1.py`. `run` drives the
`gufo bench` sweep of design D16 (detached or foreground, per-config logs and
completion markers, resumable) and `evaluate` summarizes the artifacts against
the performance contract of the spec plus the ideal-gain quantities of design
D18.

    g1.py run [--binary B] [--model M] [--out DIR] [--depths ..] [--foreground]
    g1.py status [--out DIR]
    g1.py wait [--out DIR] [--timeout SECONDS]
    g1.py evaluate [--artifacts DIR] [--model M] [--json PATH]

`run` measures pp2048 / tg128 at depths 0 / 4096 / 32768 / 131072, bracketing
each depth's KVarN candidates with an FP16 run before and after (tags `fp16`
and `fp16b`), for every KVarN preset with the tail unset and 2048. It records
the target model path and its SHA-256 next to the artifacts.

`evaluate` prints the per-depth ratio table (throughput relative to the FP16
bracket mean, matching `docs/models/qwen3.8-27b/G1-REPORT.md`) and the D18
table: n(d), B_FP16(d), B_KVarN(d) by the D5 size formula, BW, G*, R and the
`1 + f*G*` threshold. The summary JSON includes the model SHA-256.

The D18 size formula mirrors the C++ helpers `KvarnTilePayloadBytes` and
`ComputeKvarnRegions` of src/models/qwen/kvarn_reference.{hpp,cpp} and the
geometry constants of src/models/qwen/hip/execution_policy.hpp; keep them in
sync when those change. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = ROOT / "build/release/gufo"
DEFAULT_MODEL = Path("/home/betiz/models/Qwen3.8-27B-Q8_0.gguf")
DEFAULT_OUT = ROOT / "artifacts/g1"
MODEL_IDENTITY_FILE = "model.json"
SUMMARY_FILE = "d18-summary.json"
RUN_LOG = "run.log"
RUN_MARKER_SUFFIX = ".done"

DEPTHS = (0, 4096, 32768, 131072)
# Pass-through bench arguments set by `run --extra-args` (e.g. the DFlash2
# speculation flags used by the G2 judgment).
EXTRA_ARGS: list[str] = []
PROMPT_TOKENS = 2048
GEN_TOKENS = 128
DEFAULT_RUN_TIMEOUT = 3600

# Resolved KVarN presets of design D1, in sweep order. `tail` is the explicit
# --kv-cache-tail value or None for the structural minimum W.
KVarN_CONFIGS = (
    ("k4v4-none", "kvarn-k4v4", None),
    ("k4v4-t2048", "kvarn-k4v4", 2048),
    ("k5v4-none", "kvarn-k5v4", None),
    ("k5v4-t2048", "kvarn-k5v4", 2048),
    ("k6v5-none", "kvarn-k6v5", None),
    ("k6v5-t2048", "kvarn-k6v5", 2048),
)
CANDIDATES = tuple(name for name, _, _ in KVarN_CONFIGS)
PRESET_BITS = {"kvarn-k4v4": (4, 4), "kvarn-k5v4": (5, 4), "kvarn-k6v5": (6, 5)}
PRESET_OF_TAG = {name: preset for name, preset, _ in KVarN_CONFIGS}
TAIL_OF_TAG = {name: tail for name, _, tail in KVarN_CONFIGS}
FP16_TAGS = ("fp16", "fp16b")
# The memory-priority preset of design D1/D18, matched by suffix.
MEMORY_FIRST_SUFFIX = "k4v4"
# Depth of the D18 threshold judgment (design D18 replaces the 1.15x depth-128K
# criterion; the 32K+ 1.00x floor still applies to every preset).
D18_REFERENCE_DEPTH = 131072
# f of D18 (user decision, fixed before the next measurement).
F = 0.5

# Ratio floors of the spec performance contract (unchanged by D18).
FLOOR = {0: 0.95, 4096: 0.95, 32768: 1.00, 131072: 1.00}

# D18 context term: the tg128 measurement averages over the generation window.
CONTEXT_OFFSET = GEN_TOKENS // 2

# KVarN geometry and D5 size accounting, mirroring
# src/models/qwen/kvarn_reference.hpp and hip/execution_policy.hpp.
ATTENTION_LAYERS = 16  # FullAttentionLayerCount of Qwen3.8 27B
KV_HEADS = 4
HEAD_DIM = 256
PLANES = 2
FP16_BYTES = 2
SINK_ROWS = 128
# W = max(verification rows 8, rollback rows 9).
VERIFY_ROWS = 9
TILE_TOKENS = 128
TILE_CHANNELS = 256
TILE_ELEMENTS = TILE_TOKENS * TILE_CHANNELS
LOW_PLANE_BYTES = TILE_ELEMENTS // 2
BIT_PLANE_BYTES = TILE_ELEMENTS // 8
# 256 RTN scales + 256 zero points + 256 channel + 128 token, all FP16.
PAYLOAD_METADATA_BYTES = (
    2 * (TILE_CHANNELS + TILE_CHANNELS + TILE_CHANNELS) + 2 * TILE_TOKENS
)

ROW = re.compile(
    r"^\|(?:[^|]*\|){3}\s*ROCm \(HIP\)\s*\|\s*(?P<test>[^|]+?)\s*\|\s*"
    r"(?P<mean>[-+0-9.]+)"
)


def parse_depths(text) -> list[int]:
    if isinstance(text, (list, tuple)):
        parts = [str(p) for t in text for p in str(t).replace(",", " ").split()]
    else:
        parts = str(text).replace(",", " ").split()
    return [int(p) for p in parts]


# --------------------------------------------------------------------------
# D18 arithmetic
# --------------------------------------------------------------------------


def tile_payload_bytes(bits: int) -> int:
    """Bytes of one sealed tile payload for a code width (design D5), mirroring
    `KvarnTilePayloadBytes`: metadata + low plane + extra bit planes."""
    return (
        PAYLOAD_METADATA_BYTES
        + LOW_PLANE_BYTES
        + (BIT_PLANE_BYTES if bits > 4 else 0)
        + (BIT_PLANE_BYTES if bits > 5 else 0)
    )


def n_of_depth(depth: int) -> int:
    """n(d) = d + 64: mean context length during the 128 generated tokens."""
    return depth + CONTEXT_OFFSET


def b_fp16(tokens: int) -> int:
    """B_FP16(d): FP16 KV bytes live at `tokens` tokens (design D18)."""
    return tokens * ATTENTION_LAYERS * KV_HEADS * HEAD_DIM * PLANES * FP16_BYTES


def tail_rows(tail_length: int | None) -> int:
    """F16 rows behind each query: L, the explicit tail or W when unset."""
    return VERIFY_ROWS if not tail_length else tail_length


def ring_rows(tail_length: int | None) -> int:
    """C = L + 128 + W: capacity of the F16 tail ring (design D1/D5)."""
    return tail_rows(tail_length) + SINK_ROWS + VERIFY_ROWS


def sealed_tiles(tokens: int, tail_length: int | None) -> int:
    """Tiles sealed at `tokens` tokens. A tile seals when its last position
    `e_t` satisfies `e_t <= q_max - L - 1` (design D4), with `q_max` the last
    query position `tokens - 1`; positions 0..127 are the sink."""
    body = tokens - tail_rows(tail_length) - SINK_ROWS - 1
    return max(0, (max(0, body) + TILE_TOKENS - 1) // TILE_TOKENS)


def kvarn_used_bytes(tokens: int, k_bits: int, v_bits: int,
                     tail_length: int | None) -> dict:
    """B_KVarN(d): GByte region in use at `tokens` tokens by the D5 size
    formula (sink, full tail capacity C, sealed tiles for the rest)."""
    tiles = sealed_tiles(tokens, tail_length)
    tile_bytes = tile_payload_bytes(k_bits) + tile_payload_bytes(v_bits)
    fp16_rows = (SINK_ROWS + ring_rows(tail_length)) * HEAD_DIM * PLANES * FP16_BYTES
    per_head = fp16_rows + tiles * tile_bytes
    return {
        "tiles": tiles,
        "tail_length": tail_rows(tail_length),
        "tail_capacity": ring_rows(tail_length),
        "sink_bytes": SINK_ROWS * HEAD_DIM * PLANES * FP16_BYTES
        * ATTENTION_LAYERS * KV_HEADS,
        "body_bytes": tiles * tile_bytes * ATTENTION_LAYERS * KV_HEADS,
        "tail_bytes": ring_rows(tail_length) * HEAD_DIM * PLANES * FP16_BYTES
        * ATTENTION_LAYERS * KV_HEADS,
        "total_bytes": per_head * ATTENTION_LAYERS * KV_HEADS,
    }


def d18_quantities(depth: int, k_bits: int, v_bits: int,
                   tail_length: int | None, t_fp16_s: float,
                   t_fp16_zero_s: float) -> dict | None:
    """B_FP16, B_KVarN, BW, G* for a depth (design D18). None at depth 0, where
    the FP16 increment is zero and BW is undefined."""
    delta = t_fp16_s - t_fp16_zero_s
    if depth <= 0 or delta <= 0:
        return None
    tokens = n_of_depth(depth)
    b_fp16_d = b_fp16(tokens)
    kvarn = kvarn_used_bytes(tokens, k_bits, v_bits, tail_length)
    bw = b_fp16_d / delta
    g_star = t_fp16_s / (t_fp16_zero_s + kvarn["total_bytes"] / bw) - 1.0
    return {
        "n": tokens,
        "B_FP16": b_fp16_d,
        "B_KVarN": kvarn["total_bytes"],
        "tiles": kvarn["tiles"],
        "tail_capacity": kvarn["tail_capacity"],
        "BW": bw,
        "G_star": g_star,
        "threshold": 1.0 + F * g_star,
    }


# --------------------------------------------------------------------------
# Artifact parsing
# --------------------------------------------------------------------------


def bench_command(binary: Path, model: Path, depth: int, kv_cache: str,
                  tail: int | None) -> list[str]:
    command = [str(binary), "bench", "-m", str(model),
               "--n-prompt", str(PROMPT_TOKENS), "--n-gen", str(GEN_TOKENS),
               "--n-depth", str(depth), "--kv-cache", kv_cache]
    if tail:
        command += ["--kv-cache-tail", str(tail)]
    return command + EXTRA_ARGS


def read_metric(out: Path, tag: str, depth: int, metric: str) -> float | None:
    log = out / f"{tag}_d{depth}.log"
    marker = out / f"{tag}_d{depth}.done"
    if not log.is_file() or not marker.is_file():
        return None
    want = metric if depth == 0 else f"{metric} @ d{depth}"
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ROW.match(line)
        if m and m.group("test").strip() == want:
            return float(m.group("mean"))
    return None


def read_tg(out: Path, tag: str, depth: int) -> float | None:
    return read_metric(out, tag, depth, "tg128")


def fp16_bracket(out: Path, depth: int) -> float | None:
    values = [v for v in (read_tg(out, tag, depth) for tag in FP16_TAGS)
              if v is not None]
    return sum(values) / len(values) if values else None


def model_identity(model: Path, cache: Path) -> dict:
    """Model path, size, mtime and SHA-256, cached by size+mtime."""
    model = model.resolve()
    stat = model.stat()
    if cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        if (data.get("path") == str(model)
                and data.get("size_bytes") == stat.st_size
                and data.get("mtime_ns") == stat.st_mtime_ns
                and data.get("sha256")):
            return data
    digest = hashlib.sha256()
    with model.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    data = {"path": str(model), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}
    cache.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return data


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def log_line(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def run_one(out: Path, log_path: Path, binary: Path, model: Path, tag: str,
            kv_cache: str, tail: int | None, depth: int, timeout: int) -> None:
    log = out / f"{tag}_d{depth}.log"
    marker = out / f"{tag}_d{depth}.done"
    if marker.is_file():
        log_line(log_path, f"skip (done): {tag} d{depth}")
        return
    command = bench_command(binary, model, depth, kv_cache, tail)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        handle.write("+ " + shlex.join(command) + "\n")
        handle.flush()
        try:
            proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                  timeout=timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = 124
    marker.write_text(f"{code}\n", encoding="utf-8")
    elapsed = int(time.monotonic() - started)
    log_line(log_path, f"done: {tag} d{depth} rc={code} elapsed={elapsed}s")


def cmd_run(args) -> int:
    global EXTRA_ARGS
    EXTRA_ARGS = shlex.split(getattr(args, "extra_args", "") or "")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    binary = Path(args.binary)
    model = Path(args.model)
    if not binary.is_file():
        raise SystemExit(f"missing bench binary: {binary} (build the release preset)")
    if not model.is_file():
        raise SystemExit(f"missing model: {model}")
    depths = parse_depths(args.depths)

    if not args.foreground:
        # D16: start detached with a completion marker; the same command
        # resumes from the per-config markers.
        log_path = out / RUN_LOG
        driver_path = out / f"{RUN_LOG}.driver"
        marker = Path(str(log_path) + RUN_MARKER_SUFFIX)
        marker.unlink(missing_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "run",
                   "--foreground", "--binary", str(binary), "--model", str(model),
                   "--out", str(out), "--depths",
                   ",".join(str(d) for d in depths), "--timeout", str(args.timeout)]
        if getattr(args, "extra_args", ""):
            command += ["--extra-args", args.extra_args]
        # The child appends its own progress to run.log, so its stdout goes to a
        # separate driver log (tracebacks included) to avoid duplicating lines.
        script = (f"{shlex.join(command)} >> {shlex.quote(str(driver_path))} 2>&1; "
                  f"echo $? > {shlex.quote(str(marker))}")
        subprocess.Popen(["setsid", "nohup", "sh", "-c", script],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        print(f"started detached; log {log_path} (driver {driver_path})\n"
              f"completion marker {marker}\n"
              f"resume by running the same command again")
        return 0

    log_path = out / RUN_LOG
    identity = model_identity(model, out / MODEL_IDENTITY_FILE)
    if not (out / ".started").is_file():
        (out / ".started").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n",
                                      encoding="utf-8")
        log_line(log_path, f"g1 sweep: model={model} sha256={identity['sha256']} "
                           f"binary={binary}")
    jobs = (len(FP16_TAGS) + len(KVarN_CONFIGS)) * len(depths)
    log_line(log_path, f"jobs={jobs} depths={' '.join(str(d) for d in depths)} "
                       f"start={time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    # FP16 brackets each depth's candidates so the denominator comes from the
    # same measurement series (design D18).
    for depth in depths:
        run_one(out, log_path, binary, model, "fp16", "fp16", None, depth,
                args.timeout)
        for tag, kv_cache, tail in KVarN_CONFIGS:
            run_one(out, log_path, binary, model, tag, kv_cache, tail, depth,
                    args.timeout)
        run_one(out, log_path, binary, model, "fp16b", "fp16", None, depth,
                args.timeout)
    log_line(log_path, f"sweep complete at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    return 0


def cmd_status(args) -> int:
    out = Path(args.out)
    logs = sorted(out.glob("*_d*.log"))
    dones = sorted(out.glob("*_d*.done"))
    print(f"config logs: {len(logs)}, completion markers: {len(dones)}")
    print(f"model identity: {'yes' if (out / MODEL_IDENTITY_FILE).is_file() else 'no'}")
    bad = [p.name for p in dones if p.read_text().strip() not in ("0",)]
    if bad:
        print(f"non-zero exits: {', '.join(bad)}")
    marker = out / (RUN_LOG + RUN_MARKER_SUFFIX)
    if marker.is_file():
        print(f"finished, exit code {marker.read_text().strip()}")
    else:
        print("no completion marker (running, or never started)")
    log_path = out / RUN_LOG
    if log_path.is_file():
        tail = log_path.read_text(encoding="utf-8").splitlines()[-8:]
        print("--- log tail")
        print("\n".join(tail))
    return 0


def cmd_wait(args) -> int:
    marker = Path(args.out) / (RUN_LOG + RUN_MARKER_SUFFIX)
    deadline = time.monotonic() + args.timeout
    while not marker.is_file():
        if time.monotonic() >= deadline:
            print(f"still running after {args.timeout} s")
            return 3
        time.sleep(30)
    code = int(marker.read_text(encoding="utf-8").strip() or "1")
    print(f"finished, exit code {code}")
    return code


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


def ratio_floor(depth: int) -> float:
    return FLOOR.get(depth, 1.00)


def cmd_evaluate(args) -> int:
    out = Path(args.artifacts)
    if not out.is_dir():
        raise SystemExit(f"missing artifact directory: {out}")

    depths = [d for d in DEPTHS if fp16_bracket(out, d) is not None]
    if not depths:
        raise SystemExit(f"no FP16 bracket measurements under {out}")
    fp16 = {d: fp16_bracket(out, d) for d in depths}
    t_fp16 = {d: 1.0 / v for d, v in fp16.items()}
    zero = min(depths)
    if zero != 0 or t_fp16.get(0) is None:
        print(f"note: no FP16 depth-0 bracket; D18 needs T_FP16(0)",
              file=sys.stderr)

    identity_path = out / MODEL_IDENTITY_FILE
    if identity_path.is_file():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    elif Path(args.model).is_file():
        identity = model_identity(Path(args.model), identity_path)
    else:
        identity = {"path": str(args.model), "sha256": None}

    # 1. Contract ratio table.
    print(f"{'candidate':<14}{'tail':>6}{'depth':>8}{'tg t/s':>12}"
          f"{'fp16 t/s':>12}{'ratio':>9}{'floor':>8}{'d18 thr':>9}{'':>6}")
    records = []
    worst = None
    for depth in depths:
        fp16_tps = fp16[depth]
        for tag in CANDIDATES:
            value = read_tg(out, tag, depth)
            if value is None:
                print(f"{tag:<14}{_tail_label(tag):>6}{depth:>8}{'--':>12}"
                      f"{fp16_tps:>12.2f}{'--':>9}")
                continue
            ratio = value / fp16_tps
            floor = ratio_floor(depth)
            record = {
                "candidate": PRESET_OF_TAG[tag],
                "tag": tag,
                "tail": TAIL_OF_TAG[tag],
                "depth": depth,
                "tps": value,
                "tps_fp16": fp16_tps,
                "ratio": ratio,
                "floor": floor,
            }
            k_bits, v_bits = PRESET_BITS[PRESET_OF_TAG[tag]]
            quantities = None
            if depth in t_fp16 and 0 in t_fp16:
                quantities = d18_quantities(depth, k_bits, v_bits,
                                            TAIL_OF_TAG[tag], t_fp16[depth],
                                            t_fp16[0])
            threshold = None
            if (quantities and depth == D18_REFERENCE_DEPTH
                    and MEMORY_FIRST_SUFFIX in tag):
                threshold = quantities["threshold"]
                record["threshold"] = threshold
                record["d18_ok"] = ratio >= threshold
            if quantities:
                record.update(quantities)
            record["ok"] = ratio >= floor
            records.append(record)
            passed = ratio >= floor
            margin = ratio - floor
            if worst is None or margin < worst[0]:
                worst = (margin, tag, depth, ratio, floor, passed)
            thr_text = f"{threshold:9.3f}" if threshold is not None else f"{'--':>9}"
            print(f"{tag:<14}{_tail_label(tag):>6}{depth:>8}{value:>12.2f}"
                  f"{fp16_tps:>12.2f}{ratio:>9.3f}{floor:>8.2f}{thr_text}"
                  f"{_mark(passed, threshold is not None and ratio < threshold)}")
    if worst:
        _, tag, depth, ratio, floor, passed = worst
        print(f"\nworst margin: {tag} d{depth} ratio={ratio:.3f} "
              f"floor={floor:.2f} margin={ratio - floor:+.3f} "
              f"{'PASS' if passed else 'FAIL'}")
        all_pass = all(r["ok"] and r.get("d18_ok", True) for r in records)
        print("G1 thresholds:", "PASS" if all_pass else "FAIL")

    # 2. D18 ideal-gain table.
    print(f"\nD18 (f={F}, n(d)=d+{CONTEXT_OFFSET}, "
          f"B_KVarN per D5 size formula):")
    print(f"{'candidate':<14}{'tail':>6}{'depth':>8}{'n':>9}{'B_FP16 GB':>12}"
          f"{'B_KVarN GB':>12}{'tiles':>7}{'BW GB/s':>10}{'G*':>8}{'R':>8}"
          f"{'1+f*G*':>9}")
    for record in records:
        if "G_star" not in record:
            continue
        print(f"{record['tag']:<14}{_tail_label(record['tag']):>6}"
              f"{record['depth']:>8}{record['n']:>9}"
              f"{record['B_FP16'] / 1e9:>12.3f}"
              f"{record['B_KVarN'] / 1e9:>12.3f}{record['tiles']:>7}"
              f"{record['BW'] / 1e9:>10.1f}{record['G_star']:>8.3f}"
              f"{record['ratio']:>8.3f}", end="")
        if "threshold" in record:
            print(f"{record['threshold']:>9.3f}")
        else:
            print(f"{'--':>9}")

    print(f"\nmodel: {identity.get('path')}")
    print(f"sha256: {identity.get('sha256')}")

    summary = {
        "schema_version": 1,
        "change": "add-kvarn-kv-cache-quantization",
        "task": "3.17",
        "f": F,
        "context_offset": CONTEXT_OFFSET,
        "threshold_expression": "1 + f*G*",
        "floor": {str(k): v for k, v in FLOOR.items()},
        "model": identity,
        "geometry": {
            "attention_layers": ATTENTION_LAYERS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "planes": PLANES,
            "fp16_bytes": FP16_BYTES,
            "sink_rows": SINK_ROWS,
            "verify_rows": VERIFY_ROWS,
            "tile_tokens": TILE_TOKENS,
            "payload_metadata_bytes": PAYLOAD_METADATA_BYTES,
            "low_plane_bytes": LOW_PLANE_BYTES,
            "bit_plane_bytes": BIT_PLANE_BYTES,
            "tile_payload_bytes": {str(b): tile_payload_bytes(b) for b in (4, 5, 6)},
        },
        "fp16_tps": {str(d): fp16[d] for d in depths},
        "t_fp16_s": {str(d): t_fp16[d] for d in depths},
        "records": records,
    }
    json_path = Path(args.json) if args.json else out / SUMMARY_FILE
    json_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    return 0


def _tail_label(tag: str) -> str:
    tail = TAIL_OF_TAG.get(tag)
    return "none" if tail is None else str(tail)


def _mark(passed: bool, d18_failed: bool) -> str:
    if d18_failed:
        return "  D18 FAIL"
    return "  OK" if passed else "  FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="drive the G1 sweep (resumable)")
    run.add_argument("--binary", default=str(DEFAULT_BINARY))
    run.add_argument("--model", default=str(DEFAULT_MODEL))
    run.add_argument("--out", default=str(DEFAULT_OUT))
    run.add_argument("--depths", default=",".join(str(d) for d in DEPTHS))
    run.add_argument("--timeout", type=int, default=DEFAULT_RUN_TIMEOUT,
                     help="per-invocation timeout in seconds")
    run.add_argument("--foreground", action="store_true",
                     help="run in this process instead of detaching")
    run.add_argument("--extra-args", default="",
                     help="extra arguments appended to every gufo bench call "
                          "(for example \"--speculative dflash2 "
                          "--dflash-model /path\")")
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status")
    status.add_argument("--out", default=str(DEFAULT_OUT))
    status.set_defaults(func=cmd_status)

    wait = sub.add_parser("wait")
    wait.add_argument("--out", default=str(DEFAULT_OUT))
    wait.add_argument("--timeout", type=int, default=7200)
    wait.set_defaults(func=cmd_wait)

    evaluate = sub.add_parser("evaluate", help="summarize artifacts (D16/D18)")
    evaluate.add_argument("--artifacts", default=str(DEFAULT_OUT))
    evaluate.add_argument("--model", default=str(DEFAULT_MODEL),
                          help="used to compute the SHA-256 when not cached")
    evaluate.add_argument("--json", default=None,
                          help=f"summary path (default <artifacts>/{SUMMARY_FILE})")
    evaluate.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
