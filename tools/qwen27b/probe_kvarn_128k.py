#!/usr/bin/env python3
"""Tokenizer-counted seal-boundary placement for the 128K units (user
decision 2026-10-06, tasks 4.4).

Starts a temporary server, asks the model's tokenizer for the prompt token
count of short probe requests (no calibration loop is run during the
measurement), and computes, for each case, the filler character count whose
prompt puts the first verification window across a seal boundary at depth
131072 (design D2: the window covers query positions
[prompt_tokens, prompt_tokens + draft_tokens] and a seal boundary sits at
q % 128 == tail % 128). The chosen lengths are written to the harness
calibration file; the measurement then runs with `--no-boundary-target`
(one request per unit).

The probe uses a custom --filler block, so the prefix the harness builds is
(block repeated)[:chars]. With p1 = tokens(one block + case), E = tokens(two
blocks + case) - p1 (the marginal cost of one more block, junction included)
and probe(r) = tokens(r characters + case), the count for k full blocks plus
a remainder of r characters is

    tokens(k * L + r) = probe(r) + k * E

because the case framing and the first block cancel: both sides count the
case, and the k-block prefix contributes k marginal blocks on top of the
remainder's own prefix. The calibration stores the FULL character count
(k * L + r), which is what the harness passes to prefix_text; only short
probes are needed (the validation probe checks the model at several blocks).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/qwen27b"))
import kvarn_consistency as kc  # noqa: E402

OUT = Path("/home/betiz/.cache/gufo/kvarn-consistency")
LABEL = "44-reduced"
MODEL = "/home/betiz/models/Qwen3.8-27B-Q8_0.gguf"
DEPTH = 131072
TAIL = 2048
WINDOW = 7          # draft_tokens; the first verification window width
BLOCK = 16384       # custom filler length in characters
COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TARGET_LO = DEPTH
TARGET_HI = DEPTH + 4096
SUITE = ROOT / "artifacts/consistency-160.json"
FILLER_FILE = OUT / "filler-128k.txt"
RESIDUE = TAIL % kc.KVARN_TILE_TOKENS
SERVER_LOG = Path("/tmp/probe-128k-server.log")
CHARS_PER_TOKEN = 3.62   # measured: 16384 chars -> 4530 marginal tokens


def covered(tokens: int) -> bool:
    return (TAIL - tokens) % kc.KVARN_TILE_TOKENS <= WINDOW


def build_block() -> str:
    pieces = []
    for pattern in ("docs/**/*.md", "*.md"):
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                try:
                    pieces.append(path.read_text(encoding="utf-8"))
                except (UnicodeDecodeError, OSError):
                    continue
    if not pieces:
        raise SystemExit("no filler text found")
    return "\n\n".join(pieces)[:BLOCK]


def probe(base_url: str, block: str, case: dict, chars: int) -> tuple[int, float]:
    prefix = ((block * (chars // len(block) + 2))[:chars]) if chars > 0 else ""
    messages = []
    if prefix:
        messages.append({"role": "system", "content": prefix})
    messages.append({"role": "user", "content": case["text"]})
    body = {
        "model": "qwen27b", "messages": messages, "max_tokens": 1,
        "stream": False, "cache_prompt": False, "temperature": 0.0,
        "top_p": 1.0, "top_k": 0, "min_p": 0.0, "min_keep": 0,
        "repeat_penalty": 1.0, "repeat_last_n": 64,
        "frequency_penalty": 0.0, "presence_penalty": 0.0,
    }
    started = time.monotonic()
    document = kc.http_json(f"{base_url}/v1/chat/completions", body, 3600)
    return int(document["usage"]["prompt_tokens"]), time.monotonic() - started


def place(base_url: str, block: str, case: dict, e: int) -> dict | None:
    """Return the full character count whose prompt lands in the target range
    with the seal boundary inside the first verification window."""
    size = len(block)
    r = size // 2
    tokens_r, _ = probe(base_url, block, case, r)
    for _ in range(8):
        k = max(1, round((TARGET_LO + 2048 - tokens_r) / e))
        total = tokens_r + k * e
        if not TARGET_LO <= total <= TARGET_HI:
            k = max(1, k + (1 if total < TARGET_LO else -1))
            total = tokens_r + k * e
        if not TARGET_LO <= total <= TARGET_HI:
            return None
        if covered(total):
            return {"chars": k * size + r, "periods": k, "remainder": r,
                    "prompt_tokens": total}
        # shift the remainder so the residue lands in the middle of the window
        need = (124 - total) % kc.KVARN_TILE_TOKENS
        if need > kc.KVARN_TILE_TOKENS // 2:
            need -= kc.KVARN_TILE_TOKENS
        r = max(1, min(size - 1, r + round(need * CHARS_PER_TOKEN) or (5 if need > 0 else -5)))
        tokens_r, _ = probe(base_url, block, case, r)
    return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    block = build_block()
    FILLER_FILE.write_text(block, encoding="utf-8")
    print(f"filler block {len(block)} chars -> {FILLER_FILE}", flush=True)
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    cases = suite["prompts"][:COUNT]
    command = [
        str(ROOT / "build/release/gufo"), "serve", "--host", "127.0.0.1",
        "--port", "0", "llm", "--served-model-name", "qwen27b",
        "--model", MODEL, "--context", "262144", "--sessions", "1",
        "--kv-cache", "kvarn-k5v4", "--kv-cache-tail", str(TAIL),
    ]
    cal_path = OUT / LABEL / "calibration.json"
    calibration = kc.read_json(cal_path) or {
        "schema_version": kc.SCHEMA_VERSION, "entries": {}, "tokenizer": None}
    results = []
    with kc.Server(command, SERVER_LOG, "127.0.0.1", 1800) as server:
        for case in cases:
            p1, _ = probe(server.base_url, block, case, BLOCK)
            p2, _ = probe(server.base_url, block, case, 2 * BLOCK)
            e = p2 - p1
            placed = place(server.base_url, block, case, e)
            entry = None
            if placed:
                entry = {"chars": placed["chars"], "prompt_tokens":
                         placed["prompt_tokens"], "boundary": True}
                calibration["entries"][f"chat|{case['id']}|{DEPTH}|r{RESIDUE}"] = entry
                calibration["schema_version"] = kc.SCHEMA_VERSION
                kc.write_json(cal_path, calibration)
            residue = placed["prompt_tokens"] % kc.KVARN_TILE_TOKENS if placed else None
            print(f"{case['id']:<24} p1={p1} E={e} -> "
                  f"{'placed chars=' + str(placed['chars']) + ' tokens=' + str(placed['prompt_tokens']) + ' residue=' + str(residue) if placed else 'NO PLACEMENT'}",
                  flush=True)
            results.append({"case": case["id"], "placed": bool(placed),
                            **(placed or {})})
    kc.write_json(cal_path, calibration)
    placed = [r for r in results if r["placed"]]
    summary = {"depth": DEPTH, "count": len(cases), "placed": len(placed),
               "covered": len(placed), "block": BLOCK, "results": results}
    kc.write_json(OUT / "PROBE-128K.json", summary)
    (OUT / "PROBE-DONE.marker").write_text(
        f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} placed={len(placed)} "
        f"covered={len(placed)}\n", encoding="utf-8")
    print(f"placed {len(placed)}/{len(cases)} with the seal boundary inside "
          f"the first verification window", flush=True)
    return 0 if len(placed) >= 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
