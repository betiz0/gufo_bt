#!/usr/bin/env python3
"""Startup time and HIP peak-allocation driver for `gufo serve llm` (task 6.3).

Starts one `serve llm` per run with a chosen KV format, measures the time from
spawn to the first successful `/ready`, samples device-global HIP memory
(`hipMemGetInfo`, the counter the benchmark memory table uses) while an
optional workload command runs against the server, and stops the server. The
workload command may be `tools/gufo/serving_bench.py` with `{base_url}`
substituted, so the token-to-token latency percentiles and the C8 aggregate
throughput come from the established tool while this driver supplies the
startup time and the peak.

    serve_startup_peak.py --binary build/release/gufo --model M \
        --config kvarn-k5v4 --tail 2048 --runs 3 --out DIR --label NAME
    serve_startup_peak.py ... --runs 1 \
        --workload-cmd 'python3 tools/gufo/serving_bench.py --base-url
        {base_url} --concurrency 1 --prompt-file P --max-tokens 128 --json'

Each run appends one JSON record to `<out>/<label>/runs.jsonl` and writes the
server log and the workload output next to it, so a series is resumable and
auditable. Run under `tools/qwen27b/ledger_run.py` (design D22/D23).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "gufo"))

from model_bench.servers import HipMemory  # noqa: E402

LISTEN_TIMEOUT = 600.0
SAMPLE_SECONDS = 0.25


def wait_ready(process: subprocess.Popen, base_url: str,
               timeout: float) -> float | None:
    """Seconds until `/ready` answers 200, or None when the server died."""
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            return None
        try:
            with urllib.request.urlopen(base_url + "/ready", timeout=5) as r:
                if r.status == 200:
                    return time.monotonic() - started
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.25)
    return None


def port_from_log(path: Path) -> int | None:
    """The address the server prints: `event=listening address=http://h:p`."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "listening" in line and "address=http://" in line:
            tail = line.rsplit("address=http://", 1)[1]
            host_port = tail.split()[0]
            if ":" in host_port:
                try:
                    return int(host_port.rsplit(":", 1)[1])
                except ValueError:
                    return None
    return None


class PeakSampler:
    def __init__(self, memory: HipMemory | None):
        self.memory = memory
        self.peak: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self.memory is not None:
                used = self.memory.used_gib()
                if used is not None and (self.peak is None or used > self.peak):
                    self.peak = used
            self._stop.wait(SAMPLE_SECONDS)

    def __enter__(self) -> "PeakSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def server_command(args: argparse.Namespace, log_path: Path) -> list[str]:
    del log_path
    command = [
        str(args.binary), "serve", "--host", args.host, "--port", "0",
        "llm", "--served-model-name", args.served_model_name,
        "--model", str(args.model), "--context", str(args.context),
        "--sessions", str(args.sessions),
    ]
    if args.config != "fp16":
        command += ["--kv-cache", args.config]
        if args.tail:
            command += ["--kv-cache-tail", str(args.tail)]
    command += shlex.split(args.extra_args or "")
    return command


def run_once(args: argparse.Namespace, memory: HipMemory | None,
             index: int) -> dict:
    out = Path(args.out) / args.label
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.config}-{args.tail or 'none'}-r{index}"
    log_path = out / f"{stem}-server.log"
    command = server_command(args, log_path)
    spawn_time = time.monotonic()
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, stdout=log,
                                   stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL,
                                   start_new_session=True)
    base_url = ""
    startup = None
    workload_log = out / f"{stem}-workload.log"
    workload_code = None
    with PeakSampler(memory) as sampler:
        port = None
        started = time.monotonic()
        while time.monotonic() - started < LISTEN_TIMEOUT:
            if process.poll() is not None:
                break
            port = port_from_log(log_path)
            if port is not None:
                break
            time.sleep(0.25)
        if port is not None:
            base_url = f"http://{args.host}:{port}"
            # Startup is spawn -> first successful /ready: the listen line
            # appears after the model is loaded, and wait_ready measures only
            # its own poll loop, so time the whole window here.
            if wait_ready(process, base_url, LISTEN_TIMEOUT) is not None:
                startup = time.monotonic() - spawn_time
        if startup is not None and args.workload_cmd:
            workload = args.workload_cmd.replace("{base_url}", base_url)
            with workload_log.open("wb") as log:
                workload_code = subprocess.call(
                    workload, shell=True, stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        peak = sampler.peak
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    record = {
        "config": args.config, "tail": args.tail, "sessions": args.sessions,
        "context": args.context, "run": index,
        "command": command,
        "startup_seconds": startup,
        "peak_used_gib": peak,
        "base_url": base_url,
        "workload_cmd": args.workload_cmd or None,
        "workload_exit_code": workload_code,
        "workload_log": str(workload_log) if args.workload_cmd else None,
        "server_log": str(log_path),
        "exit_code": process.returncode,
        "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with (out / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{stem}: startup={startup if startup is None else f'{startup:.2f} s'} "
          f"peak={peak if peak is None else f'{peak:.2f} GiB'} "
          f"workload={workload_code}", flush=True)
    if startup is None or (workload_code not in (None, 0)):
        raise SystemExit(f"run {index} failed: startup={startup} "
                         f"workload={workload_code}")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", required=True,
                        help="fp16 or a KVarN preset name")
    parser.add_argument("--tail", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--served-model-name", default="qwen3.8-27b")
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--workload-cmd", default="",
                        help="shell command run while the peak is sampled; "
                             "{base_url} is substituted")
    parser.add_argument("--out", type=Path,
                        default=Path.home() / ".cache/gufo/kvarn-serve")
    parser.add_argument("--label", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    memory = HipMemory.for_binary(args.binary)
    if memory is None:
        print("warning: libamdhip64 not found; no peak sampling",
              file=sys.stderr)
    for index in range(1, args.runs + 1):
        run_once(args, memory, index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
