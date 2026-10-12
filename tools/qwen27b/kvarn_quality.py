#!/usr/bin/env python3
"""KV quantization quality tool (openspec add-kvarn-kv-cache-quantization,
re-judged by improve-kvarn-k6v5-quality-and-speed).

Drives the gpu-test `qwen_kvarn_quality_eval` harness, which compares the FP16
KV teacher with probe KV formats on identical token histories, and summarizes
its artifacts against the quality contract. Artifacts and corpora live outside
Git, by default under ~/.cache/gufo/kvarn-quality.

    kvarn_quality.py corpus [--out DIR]
    kvarn_quality.py run --model M --candidates a,b,c [--tails none,256,1024,2048]
                         [--label L] [--depths ..] [--foreground]
    kvarn_quality.py status [--label L]
    kvarn_quality.py wait [--label L] [--timeout SECONDS]
    kvarn_quality.py report [--label L] [--gate]

`run` follows design D16: the harness starts detached (setsid nohup) with its
output in `<out>/<label>/run.log`, prints the job count and estimated time,
and writes its exit code to `<out>/<label>/run.log.done` when it ends. Each
(format, tail, corpus, depth) unit is a separate artifact and teacher logits
are cached, so running the same command again resumes where it stopped. Poll
with `wait`, which blocks up to the timeout and returns 3 while still running.

Build the harness first (inside `nix develop`, or with hipcc as documented):
    cmake --preset gpu-test
    cmake --build --preset gpu-test --target qwen_kvarn_quality_eval
"""

import argparse
import json
import random
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = Path.home() / ".cache" / "gufo" / "kvarn-quality"
HARNESS = ROOT / "build/gpu-test/tests/models/qwen/qwen_kvarn_quality_eval"
# Whole documents are the unit: cutting text mid-document creates abrupt
# high-surprise positions whose KL dominates the tail statistics.
MAX_FILE_CHARS = 60_000
# Rough characters per token, used only to size corpora; the harness rejects a
# corpus that tokenizes short.
CHARS_PER_TOKEN = {"prose": 4.2, "code": 3.0, "other": 1.5}
TARGET_TOKENS = 215_000
SCHEMA_VERSION = 2
DEFAULT_LABEL = "kvarn-quality"
DEFAULT_TAILS = "none,256,1024,2048"
# Poll interval while `wait` blocks for the completion marker.
WAIT_POLL_SECONDS = 30
DEFAULT_WAIT_SECONDS = 7200
EXIT_STILL_RUNNING = 3


def read_texts(patterns, root=ROOT):
    files = []
    for pattern in patterns:
        files += sorted(p for p in Path(root).glob(pattern) if p.is_file())
    texts = []
    for path in files:
        try:
            texts.append((str(path), path.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue
    return texts


def documents(texts, rng):
    """Whole files (long ones truncated at a line) in a seeded shuffled order."""
    pieces = []
    for name, text in texts:
        if len(text.strip()) < 2000:
            continue
        if len(text) > MAX_FILE_CHARS:
            text = text[:text.rfind("\n", 0, MAX_FILE_CHARS)]
        pieces.append(f"\n\n# {name}\n{text}")
    rng.shuffle(pieces)
    return pieces


def domain_sources():
    home = Path.home()
    return {
        "prose": read_texts(["docs/**/*.md", "*.md"]) +
        read_texts(["docs/**/*.md", "*.md"], home / "llama.cpp"),
        "code": read_texts(["src/**/*.cpp", "src/**/*.hpp", "src/**/*.hip",
                            "tests/**/*.cpp", "tools/**/*.py"]),
        # Non-English: Japanese design notes from this repository plus public
        # multilingual template catalogs installed with ComfyUI.
        "other": read_texts(["openspec/**/*.md", ".agents/**/*.md"]) +
        read_texts(["venv/lib/python3.12/site-packages/"
                    "comfyui_workflow_templates_media_other/templates/"
                    "index.*.json"], home / "comfyui"),
    }


def build_corpus(sources, seed):
    """Interleaves the three domains, one whole document at a time, until the
    estimated size is reached.

    Sources are consumed without repetition, so long-range copying cannot
    inflate agreement; a corpus that cannot reach the target is an error.
    """
    rng = random.Random(seed)
    pools = {name: documents(texts, rng) for name, texts in sources.items()}
    out, estimate = [], 0.0
    order = list(pools)
    while estimate < TARGET_TOKENS:
        progressed = False
        for name in order:
            if pools[name]:
                piece = pools[name].pop()
                out.append(piece)
                estimate += len(piece) / CHARS_PER_TOKEN[name]
                progressed = True
        if not progressed:
            raise SystemExit(
                f"sources exhausted at ~{estimate:.0f} tokens "
                f"(< {TARGET_TOKENS}); add more distinct text")
    return "".join(out), estimate


def cmd_corpus(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = domain_sources()
    for name, texts in sources.items():
        print(f"{name}: {len(texts)} files, "
              f"{sum(len(t) for _, t in texts)} chars")
    # Corpora with different document orders, so a window at a fixed depth
    # lands in different content between them. The seeds are recorded so a
    # later judgment can build corpora earlier ones never used (design D17).
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for index, seed in enumerate(seeds):
        text, estimate = build_corpus(sources, seed)
        path = out / f"corpus-{index}.txt"
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path} (seed {seed}, {len(text)} chars, "
              f"~{estimate:.0f} tokens)")


def label_dir(args):
    return Path(args.out) / args.label


def log_path(args):
    return label_dir(args) / "run.log"


def cmd_run(args):
    out = Path(args.out)
    corpora = args.corpus or sorted(str(p) for p in out.glob("corpus-*.txt"))
    if not corpora:
        raise SystemExit("no corpus; run `kvarn_quality.py corpus` first")
    if not HARNESS.is_file():
        raise SystemExit(f"missing {HARNESS}; build qwen_kvarn_quality_eval")
    command = [str(HARNESS), "--model", args.model, "--corpus",
               ",".join(corpora), "--candidates", args.candidates,
               "--tails", args.tails, "--output", str(out),
               "--label", args.label, "--depths", args.depths,
               "--window", str(args.window), "--chunk", str(args.chunk)]
    print("+", " ".join(command), flush=True)
    if args.foreground:
        sys.exit(subprocess.call(command))
    log = log_path(args)
    log.parent.mkdir(parents=True, exist_ok=True)
    marker = Path(str(log) + ".done")
    marker.unlink(missing_ok=True)
    script = (f"{shlex.join(command)} >> {shlex.quote(str(log))} 2>&1; "
              f"echo $? > {shlex.quote(str(marker))}")
    subprocess.Popen(["setsid", "nohup", "sh", "-c", script],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    print(f"started detached; log {log}\ncompletion marker {marker}\n"
          f"resume by running the same command again")


def unit_files(args):
    return sorted((label_dir(args) / "units").glob("*.json"))


def cmd_status(args):
    log = log_path(args)
    marker = Path(str(log) + ".done")
    print(f"units written: {len(unit_files(args))}, teacher files: "
          f"{len(list((label_dir(args) / 'teacher').glob('*.logits')))}")
    if marker.is_file():
        print(f"finished, exit code {marker.read_text().strip()}")
    else:
        print("no completion marker (running, or never started)")
    if log.is_file():
        print("--- log tail")
        print("".join(log.read_text().splitlines(keepends=True)[-8:]), end="")


def cmd_wait(args):
    marker = Path(str(log_path(args)) + ".done")
    deadline = time.monotonic() + args.timeout
    while not marker.is_file():
        if time.monotonic() >= deadline:
            print(f"still running after {args.timeout} s; "
                  f"{len(unit_files(args))} units written")
            sys.exit(EXIT_STILL_RUNNING)
        time.sleep(WAIT_POLL_SECONDS)
    code = int(marker.read_text().strip() or "1")
    print(f"finished, exit code {code}")
    sys.exit(code)


def pct(values, q):
    values = sorted(values)
    rank = q * (len(values) - 1)
    lo, hi = int(rank), min(int(rank) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


# A window whose teacher already finds the text nearly impossible (mean NLL
# this high) is not a quality signal: every distribution there is unstable.
# Pre-registered in design D14 before the re-measurement; the same windows are
# excluded for every candidate and baseline.
MAX_TEACHER_NLL = 5.0

# Design D17: a position whose KL against the teacher exceeds this many nats is
# a large deviation. The constant is declared here, the tool of record.
LARGE_DEVIATION_NAT = 1.0
# Budgets on the worst q8->q4 position over depths and metrics. Quality-first
# must stay near E_q8, memory-first only has to beat E_q4. The stretch goal is
# recorded, never judged. All three were fixed before the re-judgment and must
# not move afterwards (design D2/D3).
QUALITY_FIRST_BUDGET = 0.13
QUALITY_STRETCH_BUDGET = 0.10
MEMORY_FIRST_BUDGET = 1.0
# The presets the gate judges and the budget each one must hold (design D3).
# Every other measured preset and tail is reported without a verdict.
JUDGED_PRESETS = (("kvarn-k6v5", QUALITY_FIRST_BUDGET),
                  ("kvarn-k4v4", MEMORY_FIRST_BUDGET))

# Judged metrics of design D17, as (display label, pooled-table key).
GATE_METRICS = (
    ("mean KL", "kl_mean"),
    ("median KL", "kl_median"),
    ("large-deviation rate", "large_dev_rate"),
    ("top-1 disagreement rate", "top1_disagree"),
)


def summarize(results, max_teacher_nll=MAX_TEACHER_NLL):
    """Groups per-position artifacts into (kv_cache, tail, depth) cells pooled
    over corpora, then recomputes the contract metrics on the pooled
    positions. Returns the table and the excluded windows."""
    cells = {}
    excluded = {}
    for r in results:
        teacher_nll = statistics.fmean(r["nll_teacher"])
        if teacher_nll > max_teacher_nll:
            excluded[(r["corpus"], r["depth"])] = teacher_nll
            continue
        cell = cells.setdefault((r["kv_cache"], r["tail"], r["depth"]), {
            "kl": [], "nll_diff": [], "top1": 0, "n": 0, "non_finite": 0,
            "large_dev": 0})
        cell["kl"] += [x if x is not None else float("inf") for x in r["kl"]]
        cell["nll_diff"] += [
            (c if c is not None else float("inf")) - t
            for c, t in zip(r["nll_candidate"], r["nll_teacher"])]
        cell["top1"] += round(r["top1_agreement"] * r["positions"])
        cell["n"] += r["positions"]
        cell["non_finite"] += r["non_finite_positions"]
        cell["large_dev"] += sum(1 for x in r["kl"]
                                 if x is not None and x > LARGE_DEVIATION_NAT)
    table = {}
    for key, c in cells.items():
        table[key] = {
            "kl_mean": statistics.fmean(c["kl"]),
            "kl_median": statistics.median(c["kl"]),
            "kl_p999": pct(c["kl"], 0.999),
            "top1": c["top1"] / c["n"],
            "nll_diff": statistics.fmean(c["nll_diff"]),
            "large_dev_rate": c["large_dev"] / c["n"],
            "top1_disagree": 1.0 - c["top1"] / c["n"],
            "non_finite": c["non_finite"], "n": c["n"],
        }
    windows = [{"corpus": corpus, "depth": depth, "teacher_mean_nll": nll}
               for (corpus, depth), nll in sorted(excluded.items())]
    return table, windows


def tail_key(tail):
    return -1 if tail is None else tail


def interpolated_position(candidate, eq8, eq4):
    """p = (M_candidate - M_q8) / (M_q4 - M_q8). None when the q4 baseline does
    not lie above q8 on this metric, which makes the position undefined."""
    denominator = eq4 - eq8
    if denominator <= 0:
        return None
    return (candidate - eq8) / denominator


def config_positions(table, name, tail, depths, eq8_name, eq4_name):
    """p of every judged metric at every depth. Returns the positions keyed by
    (depth, label) and the (depth, label) pairs whose denominator is not
    positive."""
    positions = {}
    undefined = []
    for depth in depths:
        candidate = table.get((name, tail, depth))
        eq8 = table.get((eq8_name, None, depth))
        eq4 = table.get((eq4_name, None, depth))
        if candidate is None or eq8 is None or eq4 is None:
            undefined.append((depth, "missing cell"))
            continue
        for label, key in GATE_METRICS:
            p = interpolated_position(candidate[key], eq8[key], eq4[key])
            if p is None:
                undefined.append((depth, label))
            else:
                positions[(depth, label)] = p
    return positions, undefined


def gate(table, depths):
    """Design D3 gate: each judged preset at every tail this label measured,
    against its own budget, over all judged metrics and depths. A metric whose
    denominator is not positive cannot be judged and fails the config (spec
    "品質契約"). Returns the failures, the per-config records and the stretch
    goal of the quality-first preset."""
    failures = []
    records = []
    stretch = {}
    for name, budget in JUDGED_PRESETS:
        tails = sorted({t for k, t, _ in table if k == name}, key=tail_key)
        if not tails:
            failures.append(f"{name} has no measured cell")
            print(f"gate {name}: no measured cell")
            continue
        for tail in tails:
            label = "unset" if tail is None else str(tail)
            positions, undefined = config_positions(
                table, name, tail, depths, "baseline-q8", "baseline-q4")
            if undefined:
                reason = ", ".join(f"{d}/{m}" for d, m in undefined)
                print(f"gate {name} tail {label:>4s}: INDETERMINATE "
                      f"(p undefined at {reason})")
                failures.append(f"{name} tail {label}: p undefined at {reason}")
                records.append({"kv_cache": name, "tail": tail,
                                "worst_p": None,
                                "limiting": "indeterminate",
                                "undefined": [f"{d}/{m}"
                                              for d, m in undefined]})
                continue
            (worst_depth, worst_metric), worst_p = max(
                positions.items(), key=lambda kv: kv[1])
            ok = worst_p <= budget
            if name == "kvarn-k6v5":
                stretch[label] = worst_p <= QUALITY_STRETCH_BUDGET
            print(f"gate {name} tail {label:>4s}: worst p {worst_p:.3f} "
                  f"({worst_depth} {worst_metric}) budget {budget:g} -> "
                  f"{'PASS' if ok else 'FAIL'}")
            records.append({"kv_cache": name, "tail": tail,
                            "worst_p": worst_p, "budget": budget,
                            "limiting": f"{worst_depth} {worst_metric}",
                            "pass": ok})
            if not ok:
                failures.append(f"{name} tail {label}: worst p {worst_p:.3f} "
                                f"({worst_depth} {worst_metric}) > "
                                f"{budget:g}")
    for label, met in sorted(stretch.items()):
        print(f"stretch goal {label:>4s}: worst p <= "
              f"{QUALITY_STRETCH_BUDGET:g} -> {'met' if met else 'not met'} "
              f"(recorded, not judged)")
    for failure in failures:
        print(f"gate FAIL: {failure}")
    return failures, records, stretch


def load_label_at(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise SystemExit(f"{root} has schema {manifest['schema_version']}, "
                         f"expected {SCHEMA_VERSION}")
    results = []
    for path in sorted((root / "units").glob("*.json")):
        unit = json.loads(path.read_text())
        assert unit["schema_version"] == SCHEMA_VERSION, path
        results.append(unit)
    if not results:
        raise SystemExit(f"no units under {root / 'units'}")
    return manifest, results


def load_label(args):
    return load_label_at(label_dir(args))


def canonical_candidate(name):
    """Canonical preset name for gate identity. Phase 0 measured the gpu-test
    pseudo path under `pseudo-kvarn-*` value names (design D13); the gate must
    judge those artifacts as the `kvarn-*` preset they implement."""
    prefix = "pseudo-"
    return name[len(prefix):] if name.startswith(prefix) else name


BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261005


def window_cells(results, max_teacher_nll):
    """Per (kv_cache, tail, depth, corpus) aggregates for the judged metrics;
    D14-excluded windows are dropped exactly as `summarize` drops them."""
    cells = {}
    for r in results:
        if statistics.fmean(r["nll_teacher"]) > max_teacher_nll:
            continue
        cell = cells.setdefault(
            (r["kv_cache"], r["tail"], r["depth"], r["corpus"]),
            {"kl": [], "top1": 0, "large_dev": 0, "n": 0})
        cell["kl"] += [x if x is not None else float("inf") for x in r["kl"]]
        cell["top1"] += round(r["top1_agreement"] * r["positions"])
        cell["large_dev"] += sum(1 for x in r["kl"]
                                 if x is not None and x > LARGE_DEVIATION_NAT)
        cell["n"] += r["positions"]
    return cells


def merged_cells(parts):
    total = {"kl": [], "top1": 0, "large_dev": 0, "n": 0}
    for part in parts:
        total["kl"] += part["kl"]
        total["top1"] += part["top1"]
        total["large_dev"] += part["large_dev"]
        total["n"] += part["n"]
    return total


def judged_metrics(cell):
    return {
        "kl_mean": statistics.fmean(cell["kl"]),
        "kl_median": statistics.median(cell["kl"]),
        "large_dev_rate": cell["large_dev"] / cell["n"],
        "top1_disagree": 1.0 - cell["top1"] / cell["n"],
    }


def bootstrap_positions(results, max_teacher_nll, resamples, seed):
    """The q8->q4 position p of every judged metric at every depth with a
    window-wise bootstrap 95% interval (design D17 position). The resampling
    unit is the evaluation window (one surviving (corpus, depth) unit),
    resampled paired across the candidate and both baselines.

    E_q8 and E_q4 are candidates of the same label and the same evaluation
    windows as the candidate; a baseline measured on another window is never
    the ruler (spec "品質契約").
    """
    cells = window_cells(results, max_teacher_nll)
    rng = random.Random(seed)
    rows = []
    anchors = (("baseline-q8", None), ("baseline-q4", None))
    configs = sorted({(name, tail) for name, tail, _, _ in cells
                      if canonical_candidate(name).startswith("kvarn-")},
                     key=lambda config: (config[0], tail_key(config[1])))
    for name, tail in configs:
        for depth in sorted({d for _, _, d, _ in cells}):
            pairs = (name, tail), *anchors
            corpora = [c for c in sorted({w for _, _, d, w in cells
                                          if d == depth})
                       if all((n, t, depth, c) in cells
                              for n, t in pairs)]
            if not corpora:
                continue

            def pooled(n, t, picks):
                return merged_cells([cells[(n, t, depth, corpora[p])]
                                     for p in picks])

            point = {}
            draws = {label: [] for label, _ in GATE_METRICS}
            for draw in range(resamples + 1):
                picks = (list(range(len(corpora))) if draw == 0 else
                         [rng.randrange(len(corpora)) for _ in corpora])
                candidate = judged_metrics(pooled(name, tail, picks))
                eq8 = judged_metrics(pooled(*anchors[0], picks))
                eq4 = judged_metrics(pooled(*anchors[1], picks))
                for label, key in GATE_METRICS:
                    denominator = eq4[key] - eq8[key]
                    if denominator <= 0:
                        continue
                    position = (candidate[key] - eq8[key]) / denominator
                    if draw == 0:
                        point[label] = position
                    else:
                        draws[label].append(position)
            for label, _ in GATE_METRICS:
                values = draws[label]
                rows.append({
                    "kv_cache": canonical_candidate(name), "tail": tail,
                    "depth": depth, "metric": label,
                    "p": point.get(label),
                    "lo": pct(values, 0.025) if values else None,
                    "hi": pct(values, 0.975) if values else None,
                    "windows": len(corpora), "defined": len(values),
                    "resamples": resamples})
    return rows


def cmd_report(args):
    manifest, results = load_label(args)
    cfg = manifest["config"]
    print(f"model sha256 {manifest['model']['sha256'][:16]}… "
          f"depths {cfg['depths']} window {cfg['window']} "
          f"chunk {cfg['chunk']} units {len(results)}")
    table, excluded = summarize(results, args.max_teacher_nll)
    table = {(canonical_candidate(name), tail, depth): m
             for (name, tail, depth), m in table.items()}
    # Judge the depths this label measured, with the baselines of this label.
    depths = sorted({depth for _, _, depth in table})
    for w in excluded:
        print(f"excluded corpus {w['corpus']} depth {w['depth']}: teacher mean "
              f"NLL {w['teacher_mean_nll']:.1f} > {args.max_teacher_nll}")
    memory = {(canonical_candidate(f["kv_cache"]), f["tail"]): f
              for f in manifest["memory"]["formats"]}
    print(f"{'kv-cache':20s} {'tail':>5s} {'depth':>7s} {'meanKL':>9s} "
          f"{'medKL':>9s} {'p99.9KL':>9s} {'top1':>7s} {'ldr':>7s} {'dNLL':>8s} "
          f"{'nonfin':>6s} {'totbpv':>7s} {'GiB':>6s}")
    for (name, tail, depth), m in sorted(
            table.items(), key=lambda kv: (kv[0][0], tail_key(kv[0][1]),
                                           kv[0][2])):
        mem = memory.get((name, tail), {})
        print(f"{name:20s} {'none' if tail is None else tail:>5} {depth:7d} "
              f"{m['kl_mean']:9.5f} {m['kl_median']:9.5f} "
              f"{m['kl_p999']:9.5f} {m['top1']:7.4f} "
              f"{m['large_dev_rate']:7.4f} {m['nll_diff']:8.5f} "
              f"{m['non_finite']:6d} {mem.get('total_bpv', float('nan')):7.3f} "
              f"{mem.get('session_kv_gib', float('nan')):6.2f}")
    failures, records, stretch = gate(table, depths)
    intervals = bootstrap_positions(results, args.max_teacher_nll,
                                    args.bootstrap, args.bootstrap_seed)
    print(f"\np (q8->q4 position) with window bootstrap 95% CI "
          f"({args.bootstrap} resamples, seed {args.bootstrap_seed}; the gate "
          f"budgets judge the pooled p)")
    print(f"{'kv-cache':20s} {'tail':>5s} {'depth':>7s} {'metric':22s} "
          f"{'p':>8s} {'95% CI':>18s} {'win':>4s}")
    for row in intervals:
        interval = (f"[{row['lo']:.3f}, {row['hi']:.3f}]"
                    if row["lo"] is not None else "undefined")
        point = f"{row['p']:.3f}" if row["p"] is not None else "n/a"
        tail = "none" if row["tail"] is None else str(row["tail"])
        print(f"{row['kv_cache']:20s} {tail:>5s} {row['depth']:7d} "
              f"{row['metric']:22s} {point:>8s} {interval:>18s} "
              f"{row['windows']:4d}")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "excluded_windows": excluded,
        "max_teacher_nll": args.max_teacher_nll,
        "large_deviation_nat": LARGE_DEVIATION_NAT,
        "budgets": {"quality_first": QUALITY_FIRST_BUDGET,
                    "quality_stretch": QUALITY_STRETCH_BUDGET,
                    "memory_first": MEMORY_FIRST_BUDGET},
        "memory": manifest["memory"],
        "gate_failures": failures,
        "gate": records,
        "stretch_goal": stretch,
        "p_intervals": intervals,
        "cells": [{"kv_cache": name, "tail": tail, "depth": depth, **m}
                  for (name, tail, depth), m in sorted(
                      table.items(),
                      key=lambda kv: (kv[0][0], tail_key(kv[0][1]), kv[0][2]))],
    }
    (label_dir(args) / "summary.json").write_text(
        json.dumps(summary, indent=1) + "\n")
    print(f"wrote {label_dir(args) / 'summary.json'}")
    if args.gate:
        sys.exit(1 if failures else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    corpus = sub.add_parser("corpus")
    corpus.add_argument("--out", default=str(DEFAULT_DIR))
    corpus.add_argument("--seeds", default="1,2,3",
                        help="comma-separated document-order seeds; one "
                             "corpus file per seed (default: 1,2,3)")
    corpus.set_defaults(func=cmd_corpus)

    def common(p):
        p.add_argument("--out", default=str(DEFAULT_DIR))
        p.add_argument("--label", default=DEFAULT_LABEL)

    run = sub.add_parser("run")
    run.add_argument("--model", required=True)
    run.add_argument("--candidates", required=True)
    run.add_argument("--tails", default=DEFAULT_TAILS,
                     help="tail lengths for KVarN candidates (default: "
                          f"{DEFAULT_TAILS}); baselines have none")
    run.add_argument("--depths", default="4096,32768,131072")
    run.add_argument("--window", type=int, default=256)
    run.add_argument("--chunk", type=int, default=256)
    run.add_argument("--corpus", action="append")
    run.add_argument("--foreground", action="store_true",
                     help="run in this process (short runs only)")
    common(run)
    run.set_defaults(func=cmd_run)
    status = sub.add_parser("status")
    common(status)
    status.set_defaults(func=cmd_status)
    wait = sub.add_parser("wait")
    wait.add_argument("--timeout", type=int, default=DEFAULT_WAIT_SECONDS)
    common(wait)
    wait.set_defaults(func=cmd_wait)
    report = sub.add_parser("report")
    report.add_argument("--gate", action="store_true")
    report.add_argument("--max-teacher-nll", type=float,
                        default=MAX_TEACHER_NLL)
    report.add_argument("--bootstrap", type=int,
                        default=BOOTSTRAP_RESAMPLES,
                        help="window bootstrap resamples for the p intervals")
    report.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    common(report)
    report.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
