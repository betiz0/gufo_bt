#!/usr/bin/env python3
"""Test ledger wrapper (user decision 2026-10-08, designs D16/D22).

Runs one test at a time. The `run` subcommand writes an IN record to
`artifacts/test-ledger.jsonl`, forks a detached monitor that executes the
command (design D16: no interactive wait), appends a PROGRESS record every
five minutes (marking the entries that pass the estimate so the agent can
report and wait for the user's decision instead of stopping the test), and
writes an OUT record when the command ends. `judge` lets the agent record the
pass/fail decision and the observed values, so the ledger keeps the objective
facts and the judgment apart.

    ledger_run.py run --id T2 --task 6.1 --purpose "..." \
        --estimate-seconds 600 --estimate-basis "G0 teacher job 47 s x 2" \
        --pass-criteria "exit 0 and both jobs complete" \
        --units-glob 'artifacts/t08-smoke/*/units/*.json' \
        --artifacts-dir /home/betiz/.cache/gufo/kvarn-quality/t08 -- \
        <command> [args...]
    ledger_run.py status --id T2
    ledger_run.py wait --id T2 --timeout 3600
    ledger_run.py judge --id T2 --state pass --value "mean KL 0.1" \
        --note "one sentence"
    ledger_run.py records            # print the whole ledger
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "artifacts" / "test-ledger.jsonl"
RUNS = ROOT / "artifacts" / "test-runs"
SHA_CACHE = ROOT / "artifacts" / ".sha256-cache.json"
PROGRESS_SECONDS = 300
EXIT_ABORTED = 143


def append(record: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    record["logged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def records() -> list[dict]:
    if not LEDGER.is_file():
        return []
    out = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    cache = {}
    if SHA_CACHE.is_file():
        try:
            cache = json.loads(SHA_CACHE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cache = {}
    stat = path.stat()
    key = str(path)
    entry = cache.get(key)
    if entry and entry.get("size") == stat.st_size \
            and entry.get("mtime_ns") == stat.st_mtime_ns:
        return entry["sha256"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    cache[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                  "sha256": digest.hexdigest()}
    SHA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SHA_CACHE.write_text(json.dumps(cache, indent=1) + "\n", encoding="utf-8")
    return digest.hexdigest()


def git_facts() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(("git", "-C", str(ROOT), *args),
                                  capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def dmesg_amdgpu(lines: int = 40) -> list[str]:
    """Best-effort kernel log excerpt mentioning amdgpu, newest last."""
    for command in (["dmesg"], ["journalctl", "-k", "--no-pager"]):
        try:
            out = subprocess.run(command, capture_output=True, text=True,
                                 timeout=60)
        except Exception:
            continue
        if out.returncode == 0 and out.stdout:
            hits = [l for l in out.stdout.splitlines() if "amdgpu" in l.lower()]
            return hits[-lines:]
    return []


def settings_from_pairs(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        out[key.strip()] = value.strip()
    return out


def inputs_from_paths(paths: list[str]) -> list[dict]:
    out = []
    for raw in paths or []:
        path = Path(raw)
        if not path.is_file():
            out.append({"path": raw, "exists": False})
            continue
        text = None
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
        out.append({"path": str(path), "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "chars": len(text) if text is not None else None,
                    "tokens_estimate": round(len(text) / 3.9) if text else None})
    return out


def monitor(args, run_dir: Path, log_path: Path) -> None:
    os.setsid()
    environment = os.environ.copy()
    started = time.time()
    with log_path.open("wb") as log:
        child = subprocess.Popen(args.command, cwd=str(ROOT), stdout=log,
                                 stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL,
                                 env=environment, start_new_session=False)
        next_progress = started + PROGRESS_SECONDS
        code = None
        while True:
            code = child.poll()
            if code is not None:
                break
            now = time.time()
            if now >= next_progress:
                next_progress = now + PROGRESS_SECONDS
                elapsed = now - started
                units = len(glob.glob(str(ROOT / args.units_glob))) \
                    if args.units_glob else None
                sizes = {}
                for root in (Path(args.artifacts_dir) if args.artifacts_dir
                             else run_dir,):
                    if root.exists():
                        sizes[str(root)] = sum(
                            p.stat().st_size for p in root.rglob("*")
                            if p.is_file())
                append({"type": "PROGRESS", "id": args.id, "task": args.task,
                        "elapsed_seconds": round(elapsed, 1),
                        "estimate_seconds": args.estimate_seconds,
                        "over_estimate": elapsed > args.estimate_seconds,
                        "units_done": units,
                        "artifact_bytes": sizes,
                        "log_bytes": log_path.stat().st_size
                        if log_path.exists() else 0})
                print(f"PROGRESS {args.id} elapsed {elapsed:.0f}s "
                      f"units={units} over_estimate="
                      f"{elapsed > args.estimate_seconds}", flush=True)
            time.sleep(1)
    ended = time.time()
    tail = ""
    if log_path.is_file():
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            tail = "".join(handle.readlines()[-25:])
    artifacts = []
    if args.artifacts_dir:
        for path in sorted(Path(args.artifacts_dir).rglob("*")):
            if path.is_file() and path.stat().st_size < (1 << 24):
                artifacts.append({"path": str(path),
                                  "bytes": path.stat().st_size,
                                  "sha256": sha256_file(path)})
    state = "completed" if code == 0 else (
        "aborted" if code in (EXIT_ABORTED, -signal.SIGTERM) else "failed")
    record = {
        "type": "OUT", "id": args.id, "task": args.task,
        "end": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(ended - started, 1),
        "estimate_seconds": args.estimate_seconds,
        "estimate_delta_seconds": round(ended - started
                                        - args.estimate_seconds, 1),
        "exit_code": code, "state": state,
        "log_path": str(log_path), "log_tail": tail,
        "artifacts": artifacts,
    }
    if code != 0:
        record["dmesg_amdgpu"] = dmesg_amdgpu()
    append(record)
    print(f"OUT {args.id} state={state} exit={code} "
          f"elapsed={ended - started:.0f}s", flush=True)


def cmd_run(args) -> int:
    run_dir = RUNS / args.id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    in_record = {
        "type": "IN", "id": args.id, "task": args.task,
        "purpose": args.purpose,
        "start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "estimate_seconds": args.estimate_seconds,
        "estimate_basis": args.estimate_basis,
        "pass_criteria": args.pass_criteria,
        "command": " ".join(args.command),
        "cwd": str(ROOT),
        "git": git_facts(),
        "binary_sha256": sha256_file(ROOT / args.command[0])
        if args.command else None,
        "model_sha256": sha256_file(Path(args.model)) if args.model else None,
        "environment_guof": {k: v for k, v in os.environ.items()
                             if k.startswith(("GUFO_", "HIP_", "ROCR_", "HSA_"))},
        "settings": settings_from_pairs(args.setting),
        "inputs": inputs_from_paths(args.input),
        "units_glob": args.units_glob,
        "artifacts_dir": args.artifacts_dir,
        "run_dir": str(run_dir),
    }
    append(in_record)
    print(f"IN {args.id} command: {' '.join(args.command)}")
    if os.fork() != 0:
        return 0
    monitor(args, run_dir, log_path)
    os._exit(0)


def _latest(kind: str, ident: str) -> dict | None:
    found = None
    for record in records():
        if record.get("type") == kind and record.get("id") == ident:
            found = record
    return found


def cmd_status(args) -> int:
    in_record = _latest("IN", args.id)
    out_record = _latest("OUT", args.id)
    progress = [r for r in records()
                if r.get("type") == "PROGRESS" and r.get("id") == args.id]
    if in_record:
        print(f"[{args.id}] task {in_record['task']}: {in_record['purpose']}")
        print(f"  started {in_record['start']} estimate "
              f"{in_record['estimate_seconds']}s "
              f"({in_record['estimate_basis']})")
        print(f"  command: {in_record['command']}")
    if progress:
        last = progress[-1]
        print(f"  last PROGRESS: elapsed {last['elapsed_seconds']}s units="
              f"{last.get('units_done')} over_estimate="
              f"{last.get('over_estimate')}")
    if out_record:
        print(f"  OUT: state {out_record['state']} exit "
              f"{out_record['exit_code']} elapsed "
              f"{out_record['elapsed_seconds']}s (delta "
              f"{out_record['estimate_delta_seconds']}s)")
        if out_record.get("dmesg_amdgpu"):
            print("  dmesg(amdgpu):")
            for line in out_record["dmesg_amdgpu"][-5:]:
                print("   ", line)
        for artifact in out_record.get("artifacts", [])[:6]:
            print(f"  artifact {artifact['path']} "
                  f"({artifact['bytes']} B)")
    else:
        print("  running (no OUT record yet)")
    judged = _latest("JUDGE", args.id)
    if judged:
        print(f"  judge: {judged['state']} values={judged.get('values')}")
    return 0 if out_record else 3


def cmd_wait(args) -> int:
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        out_record = _latest("OUT", args.id)
        if out_record:
            print(f"{args.id} finished: state {out_record['state']} exit "
                  f"{out_record['exit_code']}")
            return 0 if out_record["state"] == "completed" else 1
        time.sleep(15)
    print(f"{args.id} still running after {args.timeout}s")
    return 3


def cmd_judge(args) -> int:
    append({"type": "JUDGE", "id": args.id, "state": args.state,
            "values": args.value, "note": args.note})
    print(f"JUDGE {args.id} {args.state}")
    return 0


def cmd_records(args) -> int:
    for record in records():
        summary = {k: v for k, v in record.items()
                   if k in ("type", "id", "task", "state", "exit_code",
                            "elapsed_seconds", "estimate_seconds",
                            "over_estimate", "units_done", "logged_at")}
        print(json.dumps(summary, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--id", required=True)
    run.add_argument("--task", required=True)
    run.add_argument("--purpose", required=True)
    run.add_argument("--estimate-seconds", type=float, required=True)
    run.add_argument("--estimate-basis", required=True)
    run.add_argument("--pass-criteria", required=True)
    run.add_argument("--model")
    run.add_argument("--setting", action="append", default=[])
    run.add_argument("--input", action="append", default=[])
    run.add_argument("--units-glob", default="")
    run.add_argument("--artifacts-dir", default="")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status")
    status.add_argument("--id", required=True)
    status.set_defaults(func=cmd_status)

    wait = sub.add_parser("wait")
    wait.add_argument("--id", required=True)
    wait.add_argument("--timeout", type=int, default=7200)
    wait.set_defaults(func=cmd_wait)

    judge = sub.add_parser("judge")
    judge.add_argument("--id", required=True)
    judge.add_argument("--state", required=True,
                       choices=("pass", "fail", "aborted-by-user", "pending"))
    judge.add_argument("--value", action="append", default=[])
    judge.add_argument("--note", default="")
    judge.set_defaults(func=cmd_judge)

    show = sub.add_parser("records")
    show.set_defaults(func=cmd_records)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if getattr(args, "command", None) and isinstance(args.command, list) \
            and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
