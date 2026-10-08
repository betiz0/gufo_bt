#!/usr/bin/env python3
"""Test ledger wrapper (user decision 2026-10-08, designs D16/D22/D23).

Runs one test at a time. The `run` subcommand spawns the command in its own
session, writes an IN record (including the harness pid, design D23) to
`artifacts/test-ledger.jsonl`, forks a detached monitor (design D16: no
interactive wait), appends a PROGRESS record every five minutes, and writes an
OUT record when the command ends. `judge` lets the agent record the pass/fail
decision and the observed values, so the ledger keeps the objective facts and
the judgment apart.

Time rules (design D23): an estimate above one hour is rejected, and when the
elapsed time reaches twice the estimate the monitor sends SIGTERM to the
harness process group recorded in the IN record - never a `pkill -f` pattern -
and keeps itself alive to write the OUT record with state "auto-stopped", the
last PROGRESS values, and the journalctl/udev excerpts. SIGKILL follows after
`AUTOSTOP_GRACE_SECONDS` if the harness ignores SIGTERM, so the OUT record is
always written.

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
MAX_ESTIMATE_SECONDS = 3600
AUTOSTOP_FACTOR = 2
AUTOSTOP_GRACE_SECONDS = 60
EXIT_ABORTED = 143
STATE_AUTO_STOPPED = "auto-stopped"


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


def journalctl_amdgpu(since: str, lines: int = 40) -> list[str]:
    """amdgpu/drm lines from the kernel log since `since`.

    `dmesg` needs root here (Operation not permitted), so the kernel-log
    excerpt in the ledger uses journalctl plus the udev monitor output
    (user decision 2026-10-08, step 2).
    """
    try:
        out = subprocess.run(["journalctl", "-k", "--no-pager", "--since",
                              since], capture_output=True, text=True,
                             timeout=90)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    hits = [line for line in out.stdout.splitlines()
            if "amdgpu" in line.lower() or "drm" in line.lower()]
    return hits[-lines:]


def start_udev_monitor(run_dir: Path):
    """Background `udevadm monitor` for DRM kernel events."""
    path = run_dir / "udev.log"
    try:
        handle = path.open("wb")
        proc = subprocess.Popen(
            ["udevadm", "monitor", "--kernel", "--property",
             "--subsystem-match=drm"], stdout=handle,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    except OSError:
        return None, None
    return proc, path


def tail_text(path, lines: int = 40) -> str:
    if not path or not Path(path).is_file():
        return ""
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        return "".join(handle.readlines()[-lines:])


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


def progress_snapshot(args, run_dir: Path, log_path: Path, started: float,
                      now: float) -> dict:
    units = len(glob.glob(str(ROOT / args.units_glob))) \
        if args.units_glob else None
    sizes = {}
    for root in (Path(args.artifacts_dir) if args.artifacts_dir else run_dir,):
        if root.exists():
            sizes[str(root)] = sum(p.stat().st_size for p in root.rglob("*")
                                   if p.is_file())
    return {"elapsed_seconds": round(now - started, 1),
            "estimate_seconds": args.estimate_seconds,
            "over_estimate": now - started > args.estimate_seconds,
            "units_done": units,
            "artifact_bytes": sizes,
            "log_bytes": log_path.stat().st_size
            if log_path.exists() else 0}


def signal_harness(harness: subprocess.Popen, sig: int) -> bool:
    """Signal the harness process group only (design D23).

    `start_new_session=True` in `cmd_run` makes the harness a process-group
    leader, so `killpg` reaches it and its children without matching any other
    process by name (`pkill -f` also killed this monitor in the 6.1 runs).
    """
    try:
        os.killpg(harness.pid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def monitor(args, run_dir: Path, log_path: Path, ready_fd: int) -> None:
    """Run the harness as this process's child and write the ledger records.

    The monitor must be the harness's parent: `Popen.poll` in a sibling
    process gets ECHILD and reports a fake exit code 0. The harness pid goes
    into the IN record before the `run` subcommand returns, so the caller can
    always stop the harness by pid and its children only (design D23).
    """
    os.setsid()
    started = time.time()
    udev_proc, udev_path = start_udev_monitor(run_dir)
    log = log_path.open("wb")
    try:
        harness = subprocess.Popen(args.command, cwd=str(ROOT), stdout=log,
                                   stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    except OSError as error:
        log.close()
        with os.fdopen(ready_fd, "wb") as ready:
            ready.write(f"error {error}\n".encode())
        return
    log.close()
    append({
        "type": "IN", "id": args.id, "task": args.task,
        "purpose": args.purpose,
        "start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "estimate_seconds": args.estimate_seconds,
        "estimate_basis": args.estimate_basis,
        "auto_stop_seconds": round(args.estimate_seconds * AUTOSTOP_FACTOR, 1),
        "pass_criteria": args.pass_criteria,
        "command": " ".join(args.command),
        "cwd": str(ROOT),
        "harness_pid": harness.pid,
        "harness_pgid": harness.pid,
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
    })
    with os.fdopen(ready_fd, "wb") as ready:
        ready.write(f"{harness.pid}\n".encode())
    auto_stop_seconds = args.estimate_seconds * AUTOSTOP_FACTOR
    auto_stop_at = started + auto_stop_seconds
    next_progress = started + PROGRESS_SECONDS
    stop = {"sent": False, "kill_at": None, "last_progress": None}
    code = None
    while True:
        code = harness.poll()
        if code is not None:
            break
        now = time.time()
        if not stop["sent"] and now >= auto_stop_at:
            stop["last_progress"] = progress_snapshot(args, run_dir, log_path,
                                                      started, now)
            stop["sent"] = signal_harness(harness, signal.SIGTERM)
            stop["kill_at"] = now + AUTOSTOP_GRACE_SECONDS
            append({"type": "AUTOSTOP", "id": args.id, "task": args.task,
                    "elapsed_seconds": round(now - started, 1),
                    "estimate_seconds": args.estimate_seconds,
                    "auto_stop_seconds": round(auto_stop_seconds, 1),
                    "signal": "SIGTERM", "harness_pid": harness.pid,
                    "signalled": stop["sent"],
                    "last_progress": stop["last_progress"]})
            print(f"AUTOSTOP {args.id} SIGTERM at {now - started:.0f}s "
                  f"(2 x estimate {auto_stop_seconds:.0f}s)", flush=True)
        elif stop["kill_at"] is not None and now >= stop["kill_at"]:
            signalled = signal_harness(harness, signal.SIGKILL)
            stop["kill_at"] = None
            append({"type": "AUTOSTOP", "id": args.id, "task": args.task,
                    "elapsed_seconds": round(now - started, 1),
                    "signal": "SIGKILL", "harness_pid": harness.pid,
                    "signalled": signalled})
            print(f"AUTOSTOP {args.id} SIGKILL at {now - started:.0f}s",
                  flush=True)
        elif now >= next_progress:
            next_progress = now + PROGRESS_SECONDS
            snapshot = progress_snapshot(args, run_dir, log_path, started, now)
            append({"type": "PROGRESS", "id": args.id, "task": args.task,
                    **snapshot})
            print(f"PROGRESS {args.id} elapsed "
                  f"{snapshot['elapsed_seconds']:.0f}s "
                  f"units={snapshot['units_done']} over_estimate="
                  f"{snapshot['over_estimate']}", flush=True)
        time.sleep(1)
    ended = time.time()
    start_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started))
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
    if code == 0:
        state = "completed"
    elif stop["sent"]:
        state = STATE_AUTO_STOPPED
    elif code in (EXIT_ABORTED, -signal.SIGTERM):
        state = "aborted"
    else:
        state = "failed"
    record = {
        "type": "OUT", "id": args.id, "task": args.task,
        "end": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(ended - started, 1),
        "estimate_seconds": args.estimate_seconds,
        "estimate_delta_seconds": round(ended - started
                                        - args.estimate_seconds, 1),
        "auto_stop_seconds": round(auto_stop_seconds, 1),
        "auto_stopped": stop["sent"],
        "last_progress": stop["last_progress"],
        "exit_code": code, "state": state,
        "log_path": str(log_path), "log_tail": tail,
        "artifacts": artifacts,
    }
    if udev_proc is not None:
        udev_proc.terminate()
        try:
            udev_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            udev_proc.kill()
    record["kernel_log_excerpt"] = {
        "udev_drm_tail": tail_text(udev_path, 40),
        "journalctl_k_amdgpu": journalctl_amdgpu(start_stamp),
        "note": "dmesg needs root here; replaced by the udev DRM monitor and "
                "journalctl (user decision 2026-10-08, step 2)",
    }
    append(record)
    print(f"OUT {args.id} state={state} exit={code} "
          f"elapsed={ended - started:.0f}s", flush=True)


def cmd_run(args) -> int:
    if args.estimate_seconds > MAX_ESTIMATE_SECONDS:
        print(f"rejected: estimate {args.estimate_seconds:.0f}s exceeds the "
              f"{MAX_ESTIMATE_SECONDS}s limit of design D23; split the test "
              f"or shrink its range", file=sys.stderr)
        return 2
    run_dir = RUNS / args.id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    ready_read, ready_write = os.pipe()
    if os.fork() == 0:
        os.close(ready_read)
        monitor(args, run_dir, log_path, ready_write)
        os._exit(0)
    os.close(ready_write)
    with os.fdopen(ready_read, "rb") as ready:
        payload = ready.readline().decode("utf-8", "replace").strip()
    if not payload or payload.startswith("error"):
        print(f"failed to start the harness: {payload or 'no handshake'}",
              file=sys.stderr)
        return 2
    print(f"IN {args.id} harness pid {payload} "
          f"command: {' '.join(args.command)}")
    print(f"  estimate {args.estimate_seconds:.0f}s, auto stop at "
          f"{args.estimate_seconds * AUTOSTOP_FACTOR:.0f}s (design D23)")
    return 0


def _latest(kind: str, ident: str) -> dict | None:
    found = None
    for record in records():
        if record.get("type") == kind and record.get("id") == ident:
            found = record
    return found


def cmd_status(args) -> int:
    in_record = _latest("IN", args.id)
    out_record = current_out(args.id)
    progress = [r for r in records()
                if r.get("type") == "PROGRESS" and r.get("id") == args.id]
    if in_record:
        print(f"[{args.id}] task {in_record['task']}: {in_record['purpose']}")
        print(f"  started {in_record['start']} estimate "
              f"{in_record['estimate_seconds']}s "
              f"({in_record['estimate_basis']})")
        if in_record.get("harness_pid"):
            print(f"  harness pid {in_record['harness_pid']} (auto stop at "
                  f"{in_record.get('auto_stop_seconds')}s, design D23)")
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
        excerpt = out_record.get("kernel_log_excerpt") or {}
        if excerpt.get("udev_drm_tail"):
            print("  udev DRM events (tail):")
            for line in excerpt["udev_drm_tail"].splitlines()[-5:]:
                print("   ", line)
        if excerpt.get("journalctl_k_amdgpu"):
            print("  journalctl -k (amdgpu, tail):")
            for line in excerpt["journalctl_k_amdgpu"][-5:]:
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


def current_out(ident: str) -> dict | None:
    """OUT of the current IN, or None while that run is still going.

    `_latest` alone would return an older OUT after the same id is run again
    (D16 resume: "resume by running the same command again"), and a chained
    `wait` would then fire immediately.
    """
    in_record = _latest("IN", ident)
    out_record = _latest("OUT", ident)
    if in_record is None or out_record is None:
        return None
    if out_record.get("end", "") < in_record.get("start", ""):
        return None
    return out_record


def cmd_wait(args) -> int:
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        out_record = current_out(args.id)
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
