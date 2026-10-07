#!/usr/bin/env python3
"""Resumable DFlash2-vs-AR greedy consistency suite for KVarN (tasks 4.4, D16).

The suite drives `gufo serve llm --kv-cache <preset> [--kv-cache-tail N]` over
HTTP and replays the categorized speculative corpus
(`docs/models/qwen3.8-27b/artifacts/speculative-corpus.json`) at depths
0..128K. Each case is sent twice on the same prompt: once to a server without
speculation (greedy AR) and once to a server with `--speculative dflash2`; the
two greedy completions must be byte-identical. Because the KV layout depends
on the preset and the tail length, both servers are started per configuration
and results are cached per configuration.

Sealing makes a tile readable in quantized form at query position
`q == e_t + L + 1`, i.e. `q % 128 == L % 128` (design D2/D4). A depth is
"seal-boundary" when some position in the first verification window
`[prompt_tokens, prompt_tokens + draft_tokens]` satisfies that residue. The
tokenizer is not available to stdlib Python, so the harness calibrates the
prefix character length from the token count the server reports and nudges it
until the window straddles a boundary (or gives up after `--boundary-attempts`
and records the miss for `report --gate`).

Layout under `--out`/`--label` (D16: one artifact per (configuration, case,
depth), reused on resume):

    calibration.json            prefix chars per (case, depth), tokenizer only
    <preset>-tail-<tail>/
        run.json                configuration manifest and server KV report
        server-ar.log           AR server stdout/stderr
        server-spec.log         DFlash2 server stdout/stderr
        ar/<case>__d<depth>.json    greedy AR response
        spec/<case>__d<depth>.json  greedy DFlash2 response
        units/<case>__d<depth>.json merged comparison
    run.log, run.log.done       detached log and completion marker

Commands mirror `tools/qwen27b/check.py` and `kvarn_quality.py`: `run` detaches
by default, `status` prints progress, `wait` blocks on the completion marker
(bounded), and `report [--gate]` aggregates the units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = Path.home() / ".cache" / "gufo" / "kvarn-consistency"
DEFAULT_SUITE = ROOT / "docs/models/qwen3.8-27b/artifacts/speculative-corpus.json"
DEFAULT_PRESETS = "kvarn-k4v4,kvarn-k5v4,kvarn-k6v5"
DEFAULT_TAILS = "unset,2048"
DEFAULT_DEPTHS = "0,1024,4096,8192,16384,32768,65536,131072"
KVARN_PRESETS = ("kvarn-k4v4", "kvarn-k5v4", "kvarn-k6v5")
# Geometry of design D1 / execution_policy.hpp (QwenKvarnPreset).
KVARN_TILE_TOKENS = 128
KVARN_W = 9
KVARN_TAIL_MIN = 256
KVARN_TAIL_MAX = 2048
SCHEMA_VERSION = 1
# 2: the compared text is `reasoning_content` + `content` on the chat endpoint
# (1 compared `content` alone, which is empty when a short generation stays
# inside the reasoning channel and made those units vacuously equal).
RESPONSE_TEXT_VERSION = 2
WAIT_POLL_SECONDS = 30
DEFAULT_WAIT_SECONDS = 7200
EXIT_STILL_RUNNING = 3
LISTEN_RE = re.compile(
    r"event=listening address=http://(?P<host>[0-9.]+):(?P<port>\d+)")
DEPTH_RE = re.compile(r"__d(\d+)\.json$")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_identity(path: Path) -> dict:
    resolved = Path(path).resolve(strict=False)
    if not resolved.is_file():
        return {"path": str(resolved), "missing": True}
    stat = resolved.stat()
    return {"path": str(resolved), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def load_prompts(path: Path, quick: bool, limit: int,
                 case_ids: list[str]) -> list[dict[str, str]]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = document.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise SystemExit(f"{path}: suite must contain a non-empty prompts array")
    selected = prompts
    if quick:
        selected = [case for case in selected if case.get("quick") is True]
    if case_ids:
        requested = set(case_ids)
        selected = [case for case in selected if case.get("id") in requested]
        missing = requested.difference(case.get("id") for case in selected)
        if missing:
            raise SystemExit(
                f"{path}: unknown suite case(s): {', '.join(sorted(missing))}")
    if limit > 0:
        selected = selected[:limit]
    if not selected:
        raise SystemExit(f"{path}: suite selection contains no prompts")
    for case in selected:
        if not all(isinstance(case.get(field), str) and case[field]
                   for field in ("id", "category", "text")):
            raise SystemExit(f"{path}: every prompt needs non-empty "
                             "id/category/text")
    return selected


def load_filler(args: argparse.Namespace) -> str:
    if args.filler is not None:
        text = Path(args.filler).read_text(encoding="utf-8")
        if not text.strip():
            raise SystemExit(f"{args.filler}: filler is empty")
        return text
    pieces = []
    for pattern in ("docs/**/*.md", "*.md"):
        for path in sorted(Path(ROOT).glob(pattern)):
            if not path.is_file():
                continue
            try:
                pieces.append(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, OSError):
                continue
    if not pieces:
        raise SystemExit("no filler text found; pass --filler FILE")
    return "\n\n".join(pieces)


def prefix_text(filler: str, chars: int) -> str:
    if chars <= 0:
        return ""
    repeats = chars // len(filler) + 1
    return (filler * repeats)[:chars]


def request_prompt(args: argparse.Namespace, case: dict[str, str], depth: int,
                   chars: int, filler: str) -> dict:
    """Build the HTTP body shared by the AR and DFlash2 requests.

    `depth == 0` sends the case text alone; deeper requests prepend the filler
    so the context reaches the requested depth. The same body must be used on
    both servers, so only the deterministic (depth, chars, filler) tuple feeds
    it.
    """
    prefix = prefix_text(filler, chars) if depth > 0 else ""
    sampling = {
        "temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
        "min_keep": 0, "repeat_penalty": 1.0, "repeat_last_n": 64,
        "frequency_penalty": 0.0, "presence_penalty": 0.0,
    }
    if args.endpoint == "chat":
        messages = []
        if prefix:
            messages.append({"role": "system", "content": prefix})
        messages.append({"role": "user", "content": case["text"]})
        body = {"model": args.served_model_name, "messages": messages}
    else:
        text = f"{prefix}\n\n{case['text']}" if prefix else case["text"]
        body = {"model": args.served_model_name, "prompt": text}
    body["max_tokens"] = args.max_tokens
    body["stream"] = False
    body["cache_prompt"] = False
    body.update(sampling)
    return body


def http_json(url: str, payload: dict | None, timeout: float) -> dict:
    data = None if payload is None else json.dumps(
        payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"cannot reach {url}: {error}") from error


def run_request(args: argparse.Namespace, base_url: str,
                case: dict[str, str], depth: int, chars: int,
                filler: str) -> dict:
    body = request_prompt(args, case, depth, chars, filler)
    endpoint = ("/v1/chat/completions" if args.endpoint == "chat"
                else "/v1/completions")
    started = time.perf_counter()
    response = http_json(base_url.rstrip("/") + endpoint, body, args.timeout)
    elapsed = time.perf_counter() - started
    try:
        choice = response["choices"][0]
        if args.endpoint == "chat":
            # The model separates its reasoning from the answer, and a short
            # generation can live entirely in `reasoning_content`. Comparing
            # the answer alone made such units vacuously equal (observed on
            # 2026-10-06: 32 generated tokens, content empty); compare the
            # whole generation in order.
            message = choice["message"]
            completion = ((message.get("reasoning_content") or "")
                          + (message.get("content") or ""))
        else:
            completion = choice.get("text") or ""
        usage = response["usage"]
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError(f"unexpected response shape: {response}") from error
    if completion_tokens > 0 and not completion:
        raise RuntimeError(
            f"response reported {completion_tokens} tokens but no text: "
            f"{json.dumps(response)[:400]}")
    return {
        "completion": completion,
        "completion_sha256": sha256_text(completion),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "seconds": elapsed,
        "usage": usage,
        "timings": response.get("timings"),
        "request_sha256": sha256_text(json.dumps(body, sort_keys=True)),
    }


class ServerError(RuntimeError):
    pass


class Server:
    """A detached-not `gufo serve llm` child bound to an ephemeral port."""

    def __init__(self, command: list[str], log_path: Path, host: str,
                 timeout: float):
        self.command = command
        self.log_path = Path(log_path)
        self.host = host
        self.timeout = timeout
        self.process: subprocess.Popen | None = None
        self.base_url = ""

    def __enter__(self) -> "Server":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("w", encoding="utf-8")
        log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                  f"{shlex.join(self.command)}\n")
        log.flush()
        self.process = subprocess.Popen(
            self.command, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL)
        deadline = time.monotonic() + self.timeout
        address = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise ServerError(
                    f"server exited with {self.process.returncode}:\n"
                    + self._tail())
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
            match = None
            for match in LISTEN_RE.finditer(text):
                pass
            if match is not None:
                address = (match.group("host"), int(match.group("port")))
                break
            time.sleep(1.0)
        if address is None:
            self.stop()
            raise ServerError(f"server did not listen within {self.timeout:.0f} s:\n"
                              + self._tail())
        self.base_url = f"http://{address[0]}:{address[1]}"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise ServerError(
                    f"server exited with {self.process.returncode}:\n"
                    + self._tail())
            try:
                http_json(self.base_url + "/ready", None, 10.0)
                return self
            except RuntimeError:
                time.sleep(1.0)
        self.stop()
        raise ServerError(f"server never became ready:\n{self._tail()}")

    def _tail(self, lines: int = 20) -> str:
        if not self.log_path.is_file():
            return ""
        content = self.log_path.read_text(encoding="utf-8",
                                          errors="replace").splitlines()
        return "\n".join(content[-lines:])

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None

    def __exit__(self, *exc_info) -> None:
        self.stop()


def server_command(args: argparse.Namespace, config: dict,
                   speculative: bool) -> list[str]:
    command = [
        str(args.binary), "serve", "--host", args.host, "--port", "0",
        "llm", "--served-model-name", args.served_model_name,
        "--model", str(args.model),
        "--context", str(args.context), "--sessions", str(args.sessions),
    ]
    if config["kv_cache"]:
        command += ["--kv-cache", config["kv_cache"]]
        if config["tail_tokens"]:
            command += ["--kv-cache-tail", str(config["tail_tokens"])]
    if speculative:
        command += [
            "--speculative", "dflash2",
            "--dflash-model", str(args.dflash_model),
            "--draft-tokens", str(args.draft_tokens),
            "--min-draft-tokens", "1",
        ]
        if args.draft_policy:
            command += ["--draft-policy", args.draft_policy]
    return command


def label_dir(args: argparse.Namespace) -> Path:
    return Path(args.out) / args.label


def log_path(args: argparse.Namespace) -> Path:
    return label_dir(args) / "run.log"


def config_slug(config: dict) -> str:
    return f"{config['kv_cache'] or 'fp16'}-tail-{config['tail']}"


def config_kv_lines(config_dir: Path) -> list[str]:
    """KV format/capacity lines the servers reported (D12/D16 evidence)."""
    lines: list[str] = []
    for name in ("server-ar.log", "server-spec.log"):
        path = config_dir / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8",
                                   errors="replace").splitlines():
            if "kv_cache" in line or "kvarn" in line.lower():
                lines.append(line)
    return lines


def depths_list(args: argparse.Namespace) -> list[int]:
    values = sorted({int(part) for part in args.depths.split(",") if part})
    if not values or values[0] < 0:
        raise SystemExit("--depths must be non-negative, comma-separated")
    return values


def configs_list(args: argparse.Namespace) -> list[dict]:
    presets = [part for part in args.presets.split(",") if part]
    for preset in presets:
        if preset not in KVARN_PRESETS:
            raise SystemExit(f"unknown preset {preset!r}; "
                             f"choose from {', '.join(KVARN_PRESETS)}")
    tails = []
    for part in args.tails.split(","):
        part = part.strip()
        if part in ("unset", "none", ""):
            tails.append(("unset", 0))
        else:
            value = int(part)
            if not KVARN_TAIL_MIN <= value <= KVARN_TAIL_MAX:
                raise SystemExit(f"--tails {value} outside "
                                 f"{KVARN_TAIL_MIN}..{KVARN_TAIL_MAX}")
            tails.append((str(value), value))
    configs = []
    for preset in presets:
        for tail, tail_tokens in tails:
            configs.append({
                "kv_cache": preset, "tail": tail,
                "tail_tokens": tail_tokens,
                "tail_rows": KVARN_W if tail_tokens == 0 else tail_tokens,
                "slug": config_slug({"kv_cache": preset, "tail": tail}),
            })
    return configs


def unit_name(case_id: str, depth: int) -> str:
    return f"{case_id}__d{depth}.json"


def ar_path(args: argparse.Namespace, config: dict, case_id: str,
            depth: int) -> Path:
    return label_dir(args) / config["slug"] / "ar" / unit_name(case_id, depth)


def spec_path(args: argparse.Namespace, config: dict, case_id: str,
              depth: int) -> Path:
    return label_dir(args) / config["slug"] / "spec" / unit_name(case_id, depth)


def unit_path(args: argparse.Namespace, config: dict, case_id: str,
              depth: int) -> Path:
    return label_dir(args) / config["slug"] / "units" / unit_name(case_id, depth)


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")


def boundary_residue(tail_rows: int) -> int:
    return tail_rows % KVARN_TILE_TOKENS


def boundary_offset(prompt_tokens: int, tail_rows: int) -> int:
    """Tokens between the first verification position and the next seal
    boundary (design D2: q % 128 == L % 128)."""
    return (tail_rows - prompt_tokens) % KVARN_TILE_TOKENS


def boundary_in_window(prompt_tokens: int, tail_rows: int,
                       window: int) -> bool:
    return boundary_offset(prompt_tokens, tail_rows) <= window


def calibration_path(args: argparse.Namespace) -> Path:
    return label_dir(args) / "calibration.json"


def load_calibration(args: argparse.Namespace) -> dict:
    document = read_json(calibration_path(args))
    if document is None or \
            document.get("schema_version") != SCHEMA_VERSION or \
            not isinstance(document.get("entries"), dict):
        return {"schema_version": SCHEMA_VERSION, "entries": {},
                "tokenizer": None}
    return document


def store_calibration(args: argparse.Namespace, document: dict) -> None:
    write_json(calibration_path(args), document)


def calibration_key(endpoint: str, case_id: str, depth: int,
                    residue: int) -> str:
    return f"{endpoint}|{case_id}|{depth}|r{residue}"


def estimate_chars(args: argparse.Namespace, depth: int) -> int:
    return max(0, round(depth * args.chars_per_token))


def calibrate(args: argparse.Namespace, server: Server, calibration: dict,
              case: dict[str, str], depth: int, filler: str, config: dict,
              window: int) -> tuple[int, dict]:
    """Return (prefix chars, AR response) for one (case, depth) unit.

    Requests are graded by the tokenizer-independent fact that tokenization is
    identical for every KV preset, so the chosen prefix length is cached once
    and reused by the DFlash2 server and by every later configuration. A depth
    that already straddles a boundary is kept; otherwise the length is nudged
    forward so the first verification window reaches the next boundary.
    """
    key = calibration_key(args.endpoint, case["id"], depth,
                          boundary_residue(config["tail_rows"]))
    entry = calibration["entries"].get(key)
    chars = entry["chars"] if entry else estimate_chars(args, depth)
    attempts = 1 if args.no_boundary_target else args.boundary_attempts
    result = None
    for attempt in range(max(1, attempts)):
        result = run_request(args, server.base_url, case, depth, chars, filler)
        prompt_tokens = result["prompt_tokens"]
        covered = boundary_in_window(prompt_tokens, config["tail_rows"], window)
        if depth == 0 or args.no_boundary_target or covered:
            break
        if attempt + 1 >= attempts:
            break
        need = boundary_offset(prompt_tokens, config["tail_rows"]) - window // 2
        if need <= 0:
            break
        slope = chars / max(1, prompt_tokens)
        chars += max(1, round(need * slope))
    calibration["entries"][key] = {
        "chars": chars, "prompt_tokens": result["prompt_tokens"],
        "boundary": boundary_in_window(result["prompt_tokens"],
                                       config["tail_rows"], window),
    }
    return chars, result


def ar_identity(args: argparse.Namespace, config: dict, case: dict,
                depth: int, chars: int) -> dict:
    return {
        "binary": file_identity(args.binary),
        "model": file_identity(args.model),
        "kv_cache": config["kv_cache"], "tail": config["tail"],
        "endpoint": args.endpoint, "served_model_name": args.served_model_name,
        "case": case["id"], "text_sha256": sha256_text(case["text"]),
        "depth": depth, "chars": chars, "max_tokens": args.max_tokens,
        "response_text": RESPONSE_TEXT_VERSION,
    }


def spec_identity(args: argparse.Namespace, config: dict, case: dict,
                  depth: int, chars: int) -> dict:
    identity = ar_identity(args, config, case, depth, chars)
    identity.update({
        "dflash": file_identity(args.dflash_model),
        "draft_tokens": args.draft_tokens,
        "draft_policy": args.draft_policy or "default",
    })
    return identity


def cached(path: Path, identity: dict) -> dict | None:
    document = read_json(path)
    if document is None or document.get("schema_version") != SCHEMA_VERSION:
        return None
    if document.get("identity") != identity:
        return None
    return document


def execute(args: argparse.Namespace) -> int:
    label = label_dir(args)
    label.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(Path(args.suite), args.quick, args.limit,
                           args.case)
    depths = depths_list(args)
    configs = configs_list(args)
    filler = load_filler(args)
    window = args.draft_tokens
    jobs = [(config, case, depth) for config in configs
            for case in prompts for depth in depths]
    started = time.monotonic()
    print(f"suite: {len(configs)} configurations x {len(prompts)} cases x "
          f"{len(depths)} depths = {len(jobs)} units "
          f"({2 * len(jobs)} requests)", flush=True)
    print(f"estimated {args.seconds_per_unit * len(jobs) / 3600.0:.2f} h at "
          f"{args.seconds_per_unit:.0f} s/unit; resuming skips finished units",
          flush=True)
    print(f"filler {len(filler)} chars sha256="
          f"{sha256_text(filler)[:16]}…", flush=True)

    calibration = load_calibration(args)
    calibration["tokenizer"] = file_identity(args.model)
    failures = 0
    for config in configs:
        config_dir = label / config["slug"]
        residue = boundary_residue(config["tail_rows"])
        ar_pending = [job for job in jobs if job[0] is config and
                      cached(ar_path(args, config, job[1]["id"], job[2]),
                             ar_identity(
                                 args, config, job[1], job[2],
                                 expected_chars(calibration, args, job[1],
                                                job[2],
                                                residue))) is None]
        spec_pending = [job for job in jobs if job[0] is config and
                        cached(spec_path(args, config, job[1]["id"], job[2]),
                               spec_identity(
                                   args, config, job[1], job[2],
                                   expected_chars(calibration, args, job[1],
                                                  job[2],
                                                  residue))) is None]
        print(f"\n[{config['slug']}] AR pending {len(ar_pending)}, "
              f"DFlash2 pending {len(spec_pending)}", flush=True)
        if ar_pending:
            with Server(server_command(args, config, False),
                        config_dir / "server-ar.log", args.host,
                        args.server_timeout) as server:
                for case, depth in [(j[1], j[2]) for j in ar_pending]:
                    chars, result = calibrate(
                        args, server, calibration, case, depth, filler,
                        config, window)
                    write_json(ar_path(args, config, case["id"], depth), {
                        "schema_version": SCHEMA_VERSION,
                        "identity": ar_identity(args, config, case, depth,
                                               chars),
                        "chars": chars, **result,
                    })
                    store_calibration(args, calibration)
                    done = time.monotonic() - started
                    print(f"  AR   {case['id']} depth {depth} "
                          f"prompt {result['prompt_tokens']} tok "
                          f"{result['seconds']:.1f}s ({done / 60:.1f} min "
                          f"elapsed)", flush=True)
        if spec_pending:
            with Server(server_command(args, config, True),
                        config_dir / "server-spec.log", args.host,
                        args.server_timeout) as server:
                for case, depth in [(j[1], j[2]) for j in spec_pending]:
                    chars = expected_chars(calibration, args, case, depth,
                                           residue)
                    result = run_request(args, server.base_url, case, depth,
                                         chars, filler)
                    write_json(spec_path(args, config, case["id"], depth), {
                        "schema_version": SCHEMA_VERSION,
                        "identity": spec_identity(args, config, case, depth,
                                                  chars),
                        "chars": chars, **result,
                    })
                    done = time.monotonic() - started
                    print(f"  DFL  {case['id']} depth {depth} "
                          f"prompt {result['prompt_tokens']} tok "
                          f"{result['seconds']:.1f}s ({done / 60:.1f} min "
                          f"elapsed)", flush=True)
        write_json(config_dir / "run.json", {
            "schema_version": SCHEMA_VERSION,
            "config": {k: v for k, v in config.items() if k != "slug"},
            "suite": {"path": str(args.suite),
                      "sha256": sha256_text(
                          Path(args.suite).read_text(encoding="utf-8"))},
            "filler": {"chars": len(filler),
                       "sha256": sha256_text(filler)},
            "model": file_identity(args.model),
            "dflash_model": file_identity(args.dflash_model),
            "binary": file_identity(args.binary),
            "settings": {
                "endpoint": args.endpoint, "max_tokens": args.max_tokens,
                "depths": depths, "draft_tokens": args.draft_tokens,
                "draft_policy": args.draft_policy or "default",
                "context": args.context, "sessions": args.sessions,
            },
            "server_kv_report": config_kv_lines(config_dir),
        })
        failures += merge_config(args, config, prompts, depths)
        store_calibration(args, calibration)
    elapsed = time.monotonic() - started
    print(f"\ncompleted {len(jobs)} units in {elapsed / 60.0:.1f} min; "
          f"mismatches {failures}", flush=True)
    return 1 if failures else 0


def expected_chars(calibration: dict, args: argparse.Namespace,
                   case: dict, depth: int, residue: int) -> int:
    entry = calibration.get("entries", {}).get(
        calibration_key(args.endpoint, case["id"], depth, residue))
    return entry["chars"] if entry else estimate_chars(args, depth)


def merge_config(args: argparse.Namespace, config: dict, prompts: list,
                 depths: list[int]) -> int:
    """Merge cached AR/DFlash2 responses into comparison units. Returns the
    number of mismatches in this configuration."""
    mismatches = 0
    for case in prompts:
        for depth in depths:
            ar = read_json(ar_path(args, config, case["id"], depth))
            spec = read_json(spec_path(args, config, case["id"], depth))
            if ar is None or spec is None:
                continue
            chars = ar.get("chars", 0)
            same_prompt = (ar["prompt_tokens"] == spec["prompt_tokens"]
                           and ar.get("chars") == spec.get("chars")
                           and ar["request_sha256"] == spec["request_sha256"])
            match = (same_prompt and not ar.get("errors")
                     and not spec.get("errors")
                     and ar["completion"] == spec["completion"])
            covered = boundary_in_window(ar["prompt_tokens"],
                                         config["tail_rows"],
                                         args.draft_tokens)
            status = "match" if match else (
                "mismatch" if same_prompt else "prompt-mismatch")
            write_json(unit_path(args, config, case["id"], depth), {
                "schema_version": SCHEMA_VERSION,
                "config": {k: v for k, v in config.items() if k != "slug"},
                "case": case["id"], "category": case["category"],
                "depth": depth, "chars": chars,
                "prompt_tokens": ar["prompt_tokens"],
                "spec_prompt_tokens": spec["prompt_tokens"],
                "seal_boundary": {
                    "residue": boundary_residue(config["tail_rows"]),
                    "offset": boundary_offset(ar["prompt_tokens"],
                                              config["tail_rows"]),
                    "window": args.draft_tokens,
                    "in_window": covered,
                },
                "ar": ar, "spec": spec, "match": match, "status": status,
            })
            if not match:
                mismatches += 1
                print(f"  {status.upper()} {config['slug']} {case['id']} "
                      f"depth {depth}", flush=True)
    return mismatches


def unit_files(args: argparse.Namespace) -> list[Path]:
    return sorted(label_dir(args).glob("*/units/*.json"))


def cmd_status(args: argparse.Namespace) -> None:
    label = label_dir(args)
    marker = Path(str(log_path(args)) + ".done")
    units = unit_files(args)
    configs = sorted(label.glob("*/run.json"))
    print(f"units written: {len(units)} in {len(configs)} configurations")
    if marker.is_file():
        print(f"finished, exit code {marker.read_text().strip()}")
    else:
        print("no completion marker (running, or never started)")
    log = log_path(args)
    if log.is_file():
        print("--- log tail")
        print("".join(log.read_text().splitlines(keepends=True)[-8:]), end="")


def cmd_wait(args: argparse.Namespace) -> None:
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


def load_units(args: argparse.Namespace) -> list[dict]:
    units = []
    for path in unit_files(args):
        document = read_json(path)
        if document is not None and \
                document.get("schema_version") == SCHEMA_VERSION:
            units.append(document)
    return units


def cmd_report(args: argparse.Namespace) -> None:
    label = label_dir(args)
    units = load_units(args)
    configs = {}
    for path in sorted(label.glob("*/run.json")):
        document = read_json(path)
        if document is not None:
            configs[config_slug(document["config"])] = document
    cells: dict[str, dict] = {}
    for unit in units:
        key = config_slug(unit["config"])
        cell = cells.setdefault(key, {
            "units": 0, "match": 0, "mismatch": 0, "boundary": 0,
            "errors": 0, "depths": set()})
        cell["units"] += 1
        cell["match" if unit["match"] else "mismatch"] += 1
        cell["boundary"] += 1 if unit["seal_boundary"]["in_window"] else 0
        cell["errors"] += 1 if (unit["ar"].get("errors")
                                or unit["spec"].get("errors")) else 0
        cell["depths"].add(unit["depth"])
    print(f"{'configuration':26s} {'units':>5s} {'match':>5s} {'mism':>5s} "
          f"{'err':>4s} {'boundary':>8s} depths")
    failures = []
    for key in sorted(cells):
        cell = cells[key]
        depths = ",".join(str(d) for d in sorted(cell["depths"]))
        print(f"{key:26s} {cell['units']:5d} {cell['match']:5d} "
              f"{cell['mismatch']:5d} {cell['errors']:4d} "
              f"{cell['boundary']:8d} {depths}")
        if cell["mismatch"] or cell["errors"]:
            failures.append(f"{key}: {cell['mismatch']} mismatches, "
                            f"{cell['errors']} errors")
        if cell["boundary"] == 0:
            failures.append(f"{key}: no unit with a seal boundary in the "
                            "verification window")
    for config in sorted(configs):
        if config not in cells:
            failures.append(f"{config}: no units written")
    for failure in failures:
        print(f"FAIL: {failure}")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "configs": {key: {**{k: v for k, v in cell.items() if k != "depths"},
                          "depths": sorted(cell["depths"])}
                    for key, cell in sorted(cells.items())},
        "failures": failures,
    }
    write_json(label / "summary.json", summary)
    print(f"wrote {label / 'summary.json'}")
    if args.gate:
        sys.exit(1 if failures else 0)


def cmd_run(args: argparse.Namespace) -> None:
    if args.foreground:
        sys.exit(execute(args))
    label = label_dir(args)
    label.mkdir(parents=True, exist_ok=True)
    log = log_path(args)
    marker = Path(str(log) + ".done")
    marker.unlink(missing_ok=True)
    child = [sys.executable, str(Path(__file__).resolve()),
             *sys.argv[1:], "--foreground"]
    script = (f"{shlex.join(child)} >> {shlex.quote(str(log))} 2>&1; "
              f"echo $? > {shlex.quote(str(marker))}")
    subprocess.Popen(["setsid", "nohup", "sh", "-c", script],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    print(f"started detached; log {log}\ncompletion marker {marker}\n"
          f"resume by running the same command again")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--out", default=str(DEFAULT_DIR),
                            help="artifact root (Git-external)")
        target.add_argument("--label", default="kvarn-consistency",
                            help="run label under --out")

    run = sub.add_parser("run")
    common(run)
    run.add_argument("--binary", type=Path, required=True,
                     help="gufo executable that provides `serve llm`")
    run.add_argument("--model", type=Path, required=True,
                     help="Qwen3.8 27B target GGUF")
    run.add_argument("--dflash-model", type=Path, required=True,
                     help="DFlash2 draft GGUF")
    run.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    run.add_argument("--presets", default=DEFAULT_PRESETS,
                     help="comma-separated KVarN presets")
    run.add_argument("--tails", default=DEFAULT_TAILS,
                     help="`unset` and/or tail lengths 256..2048")
    run.add_argument("--depths", default=DEFAULT_DEPTHS,
                     help="comma-separated context depths in tokens")
    run.add_argument("--endpoint", choices=("chat", "completions"),
                     default="chat")
    run.add_argument("--served-model-name", default="qwen27b")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--context", type=int, default=262144)
    run.add_argument("--sessions", type=int, default=1)
    run.add_argument("--max-tokens", type=int, default=32)
    run.add_argument("--draft-tokens", type=int, default=7)
    run.add_argument("--draft-policy")
    run.add_argument("--timeout", type=float, default=900.0,
                     help="per-request HTTP timeout (seconds)")
    run.add_argument("--server-timeout", type=float, default=1800.0,
                     help="model load and readiness wait (seconds)")
    run.add_argument("--chars-per-token", type=float, default=4.0,
                     help="starting estimate before token-count calibration")
    run.add_argument("--boundary-attempts", type=int, default=4,
                     help="prefix adjustments to place a seal boundary in "
                          "the first verification window")
    run.add_argument("--no-boundary-target", action="store_true",
                     help="record boundary coverage without adjusting depth")
    run.add_argument("--filler", type=Path,
                     help="text repeated for the context prefix")
    run.add_argument("--quick", action="store_true")
    run.add_argument("--case", action="append", default=[])
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--seconds-per-unit", type=float, default=60.0,
                     help="initial time estimate per (case, depth) unit")
    run.add_argument("--foreground", action="store_true",
                     help="run in this process (short runs only)")
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status")
    common(status)
    status.set_defaults(func=cmd_status)
    suite = sub.add_parser(
        "suite", help="build a request suite from corpus document slices")
    suite.add_argument("--corpus", type=Path, required=True,
                       help="directory holding corpus-*.txt files")
    suite.add_argument("--include", type=Path,
                       help="curated suite JSON whose cases come first")
    suite.add_argument("--count", type=int, default=160)
    suite.add_argument("--slice-chars", type=int, default=1200)
    suite.add_argument("--out", type=Path, required=True,
                       help="suite JSON to write")
    suite.set_defaults(func=cmd_suite)
    wait = sub.add_parser("wait")
    common(wait)
    wait.add_argument("--timeout", type=int, default=DEFAULT_WAIT_SECONDS)
    wait.set_defaults(func=cmd_wait)
    report = sub.add_parser("report")
    common(report)
    report.add_argument("--gate", action="store_true",
                        help="exit non-zero on a mismatch or a missing "
                             "seal-boundary unit")
    report.set_defaults(func=cmd_report)
    return parser


def cmd_suite(args) -> None:
    """Build a consistency suite from corpus document slices.

    The curated `--include` cases come first; the remainder are document heads
    sliced in corpus file and document order, so the suite is reproducible from
    the corpus files alone. Categories follow the document's source path."""
    cases: list[dict] = []
    if args.include:
        included = json.loads(args.include.read_text(encoding="utf-8"))
        cases = [{key: case[key] for key in ("id", "category", "text")}
                 for case in included.get("prompts", []) if case.get("text")]
        cases = cases[: args.count]
    corpora = sorted(args.corpus.glob("corpus-*.txt"))
    if not corpora:
        raise SystemExit(f"no corpus-*.txt under {args.corpus}")
    for path in corpora:
        if len(cases) >= args.count:
            break
        blocks = path.read_text(encoding="utf-8").split("\n\n# ")
        for index, block in enumerate(blocks):
            if len(cases) >= args.count:
                break
            block = block.strip()
            if len(block) < 400:
                continue
            name = block.split("\n", 1)[0].strip()
            if name.endswith((".cpp", ".hpp", ".hip", ".py")):
                category = "code"
            elif name.endswith(".json") or "venv/" in name:
                category = "other"
            else:
                category = "prose"
            cases.append({"id": f"{path.stem}-d{index:04d}",
                          "category": category,
                          "text": block[: args.slice_chars].strip()})
    if len(cases) < args.count:
        raise SystemExit(f"only {len(cases)} cases sliced (< {args.count})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema": "gufo.consistency-suite.v1",
        "generator": {"corpus_dir": str(args.corpus),
                      "include": str(args.include or ""),
                      "count": args.count,
                      "slice_chars": args.slice_chars},
        "prompts": cases}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print(f"wrote {args.out}: {len(cases)} cases")


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
