#!/usr/bin/env python3
"""Single-session KVarN DFlash2 regression (design D20/D21, tasks 7.2).

Runs the production entry (`gufo prompt`) one session at a time and requires the
DFlash2 greedy continuation to match the AR continuation:

* every case of the speculative corpus at depth 0 and at ~3.9K, and
* a run that crosses the first tile seal boundary with the default tail
  (256+L = 512) over 400 generated tokens.

Several sessions are never run at once here; the multi-session suite owns those
checks (design D21).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "docs/models/qwen3.8-27b/artifacts/speculative-corpus.json"
TRACE = re.compile(r"^\[TokenTrace\]: count=(\d+) sha256=([0-9a-f]{64})$",
                   re.MULTILINE)
# `gufo prompt` uses a fixed 4096-token CLI context (kDefaultContext), so the
# deep prompt is the largest docs prefix that still leaves room for the case
# text and the continuation (~3.9K tokens for 13824 characters).
DEEP_PREFIX_CHARS = 13824
BOUNDARY_PREFIX_CHARS = 1200
BOUNDARY_TOKENS = 400
CASE_TOKENS = 32


def build_prefix(characters: int) -> str:
    pieces: list[str] = []
    for pattern in ("docs/**/*.md", "*.md"):
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                try:
                    pieces.append(path.read_text(encoding="utf-8"))
                except (UnicodeDecodeError, OSError):
                    continue
    text = "\n\n".join(pieces)
    if not text:
        raise SystemExit("no docs text available for the prefix")
    return (text * (characters // len(text) + 1))[:characters]


def run(binary: Path, model: Path, draft: Path, text: str, tokens: int,
        kv_cache: str, tail: int | None, system: str | None) -> tuple[int, str]:
    command = [str(binary), "prompt", "--verbose", "--model", str(model),
               "--max-tokens", str(tokens), "--temperature", "0",
               "--speculative", "dflash2", "--draft-tokens", "7",
               "--min-draft-tokens", "1", "--dflash-model", str(draft),
               "--kv-cache", kv_cache]
    if tail is not None:
        command += ["--kv-cache-tail", str(tail)]
    if system is not None:
        command += ["--system", system]
    command.append(text)
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=1800)
    if result.returncode != 0:
        raise SystemExit(f"prompt failed ({result.returncode}): "
                         f"{result.stderr.strip()[-400:]}")
    match = TRACE.search(result.stderr) or TRACE.search(result.stdout)
    if match is None:
        raise SystemExit("missing emitted-token trace")
    return int(match.group(1)), match.group(2)


def compare(binary: Path, model: Path, draft: Path, label: str, text: str,
            tokens: int, kv_cache: str, tail: int | None,
            system: str | None) -> list[str]:
    ar_count, ar_sha = run(binary, model, draft, text, tokens, kv_cache, tail,
                           system)
    df_count, df_sha = run(binary, model, draft, text, tokens, kv_cache, tail,
                           system)
    if ar_count != df_count or ar_sha != df_sha:
        return [f"{label}: AR count={ar_count} sha={ar_sha[:16]} vs DFlash2 "
                f"count={df_count} sha={df_sha[:16]}"]
    print(f"  {label}: {ar_count} tokens exact ({ar_sha[:16]})")
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gufo", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dflash-model", type=Path)
    args = parser.parse_args()

    model = args.model or os.environ.get("GUFO_QWEN27B_MODEL")
    draft = args.dflash_model or os.environ.get("GUFO_QWEN27B_DFLASH_MODEL")
    if not model or not Path(model).is_file() or not draft or not Path(
            draft).is_file():
        print("qwen27b.dflash-kvarn: skipped (set GUFO_QWEN27B_MODEL and "
              "GUFO_QWEN27B_DFLASH_MODEL)")
        return 77
    binary = args.gufo.resolve()
    model_path = Path(model).resolve()
    draft_path = Path(draft).resolve()
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))["prompts"]
    deep = build_prefix(DEEP_PREFIX_CHARS)

    failures: list[str] = []
    for case in corpus:
        failures += compare(binary, model_path, draft_path, f"{case['id']} d0",
                            case["text"], CASE_TOKENS, "kvarn-k5v4", 2048, None)
        failures += compare(binary, model_path, draft_path,
                            f"{case['id']} ~3.9K", case["text"], CASE_TOKENS,
                            "kvarn-k5v4", 2048, deep)
    # Default tail (L = W = 256): the boundary is 256 + L = 512, so a ~1200
    # character prefix plus 400 generated tokens crosses it during generation.
    repetition = next(case for case in corpus
                      if case["id"] == "repetition_sequence")
    boundary_failures: list[str] = []
    ar_count, ar_sha = run(binary, model_path, draft_path, repetition["text"],
                           BOUNDARY_TOKENS, "kvarn-k4v4", None,
                           build_prefix(BOUNDARY_PREFIX_CHARS))
    df_count, df_sha = run(binary, model_path, draft_path, repetition["text"],
                           BOUNDARY_TOKENS, "kvarn-k4v4", None,
                           build_prefix(BOUNDARY_PREFIX_CHARS))
    if ar_count < BOUNDARY_TOKENS or ar_count != df_count or ar_sha != df_sha:
        boundary_failures.append(
            f"boundary tail=unset: AR {ar_count} tokens sha={ar_sha[:16]} vs "
            f"DFlash2 {df_count} tokens sha={df_sha[:16]} "
            f"(expected >= {BOUNDARY_TOKENS} identical tokens)")
    else:
        print(f"  boundary tail=unset: {ar_count} tokens exact ({ar_sha[:16]})")
    failures += boundary_failures
    if failures:
        print("KVarN DFlash2 single-session regression FAILED:")
        for line in failures:
            print(" ", line)
        return 1
    print("KVarN DFlash2 single-session regression passed ("
          f"{len(corpus)} cases x 2 depths + boundary run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
