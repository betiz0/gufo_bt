#!/usr/bin/env python3
"""Teacher-forced Gufo / llama.cpp agreement over one fixed token sequence.

For one artifact this runs:

1. `logit_dump` (built from tools/qwen-flash/logit_dump.cpp against the local
   llama.cpp Vulkan build) over a deterministic pp2048-class prompt: it scores
   the prompt, greedily continues it, and records every position's full logits.
2. `gpu_probe --dump` on Gufo over that same sequence, chunked to its 64-row
   dump limit, rebuilding each prefix from the recorded token pieces so both
   engines score byte-identical token sequences. Only the positions that
   predict the forced continuation are scored: the row after the last prompt
   token through the row that predicts the last generated token. Prompt
   positions are never scored, though a batch-aligned dump may cover some.
3. Per-position top-1 agreement and KL divergence of the next-token
   distributions, plus the first position where Gufo's greedy token differs
   from llama.cpp's (the free-running divergence point), with both engines'
   top-1/top-2 logits there.

The summary (not the raw logits) is written as JSON.

Usage:
  teacher_forced.py --logit-dump /tmp/logit_dump --gufo-probe probe \\
    --model MODEL.gguf --tag ap --out summary.json \\
    [--words 1600] [--continuation 128] [--context 4096] [--batch 64] \\
    [--llama-batch 2048] [--llama-ubatch 1024]

`--batch` is Gufo's prefill chunk width (`gpu_probe --batch`); `--llama-batch`
and `--llama-ubatch` are llama.cpp's logical and physical batch sizes. Every
summary records the requested values, the effective `n_batch`/`n_ubatch`
llama.cpp reports and the Gufo chunk, because a comparison is only meaningful
at a stated configuration.
"""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gufo.model_bench.llm import synthetic_text  # noqa: E402

# gpu_probe dumps at most 64 rows of its last prefill chunk.
PROBE_DUMP_ROWS = 64


def run(command, **kwargs):
    result = subprocess.run(command, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, **kwargs)
    if result.returncode != 0:
        raise SystemExit(f"{command[0]} failed ({result.returncode}):\n"
                         f"{result.stdout[-2000:]}")
    return result.stdout


def read_dump(path, vocab=None):
    """Reads a logit dump. logit_dump writes an (positions, vocab) header;
    gpu_probe --dump writes raw float rows for a vocabulary the caller knows."""
    with open(path, "rb") as f:
        if vocab is None:
            positions, vocab = struct.unpack("<ii", f.read(8))
        else:
            header = f.read(0)
            del header
            positions = None
        data = f.read()
    if positions is None:
        positions = len(data) // (vocab * 4)
    rows = [struct.unpack_from(f"<{vocab}f", data, i * vocab * 4)
            for i in range(positions)]
    return rows


def log_softmax(row):
    top = max(row)
    total = sum(math.exp(v - top) for v in row)
    log_total = top + math.log(total)
    return [v - log_total for v in row]


def kl_divergence(log_p, log_q):
    """KL(P || Q) from log-probabilities."""
    total = 0.0
    for p, q in zip(log_p, log_q):
        if p > -745.0:
            total += math.exp(p) * (p - q)
    return total


def top_two(row):
    best = max(range(len(row)), key=lambda i: (row[i], -i))
    second = max((i for i in range(len(row)) if i != best),
                 key=lambda i: (row[i], -i))
    return (best, row[best]), (second, row[second])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logit-dump", required=True)
    parser.add_argument("--gufo-probe", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--words", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--continuation", type=int, default=128)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=64,
                        help="Gufo prefill chunk width (gpu_probe --batch)")
    parser.add_argument("--llama-batch", type=int, default=2048,
                        help="llama.cpp logical batch size")
    parser.add_argument("--llama-ubatch", type=int, default=0,
                        help="llama.cpp physical batch size; 0 keeps its "
                             "default")
    args = parser.parse_args()

    prompt = synthetic_text(args.seed, args.words)
    with tempfile.TemporaryDirectory(prefix="qfn-teacher-") as work:
        prompt_file = os.path.join(work, "prompt.txt")
        Path(prompt_file).write_text(prompt)
        dump_file = os.path.join(work, "llama.bin")
        pieces_file = os.path.join(work, "pieces.json")
        tokens_file = os.path.join(work, "tokens.txt")
        output = run([args.logit_dump, "--model", args.model,
                      "--prompt-file", prompt_file, "--out", dump_file,
                      "--pieces-out", pieces_file,
                      "--tokens-out", tokens_file,
                      "--continuation", str(args.continuation),
                      "--ctx", str(args.context),
                      "--batch", str(args.llama_batch),
                      "--n-ubatch", str(args.llama_ubatch)])
        header = {}
        for line in output.splitlines():
            if line.startswith("prompt_tokens"):
                _, tokens, _, vocab = line.split()
                header["prompt_tokens"] = int(tokens)
                header["vocab"] = int(vocab)
            elif line.startswith("n_batch"):
                _, n_batch, _, n_ubatch = line.split()
                header["llama_n_batch"] = int(n_batch)
                header["llama_n_ubatch"] = int(n_ubatch)
            elif line.startswith("scored_positions"):
                _, positions, _, generated = line.split()
                header["positions"] = int(positions)
                header["generated"] = int(generated)
        if "llama_n_batch" not in header:
            raise SystemExit("logit_dump reported no effective batch sizes")
        pieces = json.loads(Path(pieces_file).read_text())
        vocab = header["vocab"]
        llama_rows = read_dump(dump_file)
        if len(llama_rows) != header["positions"]:
            raise SystemExit("logit dump size disagrees with its header")
        prompt_tokens = header["prompt_tokens"]

        # The scored positions are exactly those that predict the forced
        # continuation: row prompt_tokens - 1 (after the last prompt token)
        # through row prompt_tokens + generated - 2 (predicting the last
        # generated token). Prompt rows are never scored.
        scored_start = prompt_tokens - 1
        scored_end = prompt_tokens + header["generated"] - 2
        if scored_end < scored_start:
            raise SystemExit("no continuation tokens to score")

        # Gufo's probe dumps the last rows of the prefix's last chunk, at most
        # 64 of them, so walk the covered range backwards one aligned window at
        # a time. A window may straddle scored_start (its earlier rows are
        # prompt positions and are filtered out below); the walk stops after
        # the first such window.
        gufo_rows = {}
        prefix_end = header["positions"]
        while prefix_end > scored_start:
            chunk_rows = ((prefix_end - 1) % args.batch) + 1
            rows_expected = min(chunk_rows, PROBE_DUMP_ROWS)
            window = prefix_end - rows_expected
            prefix = "".join(pieces[:prefix_end])
            dump = os.path.join(work, f"gufo-{prefix_end}.bin")
            probe = run([args.gufo_probe, "--model", args.model,
                         "--prompt", prefix, "--batch", str(args.batch),
                         "--context", str(args.context), "--dump", dump])
            for line in probe.splitlines():
                if line.startswith("prompt tokens"):
                    reported = int(line.split()[2].strip("()"))
                    if reported != prefix_end:
                        raise SystemExit(
                            f"gpu_probe tokenized prefix {prefix_end} as "
                            f"{reported} tokens")
            rows = read_dump(dump, vocab)
            if len(rows) != rows_expected:
                raise SystemExit(
                    f"gpu_probe dumped {len(rows)} rows for prefix "
                    f"{prefix_end}, expected {rows_expected}")
            for offset, row in enumerate(rows):
                gufo_rows[window + offset] = row
            if window <= scored_start:
                break
            prefix_end = window

        covered = sorted(p for p in gufo_rows
                         if scored_start <= p <= scored_end)
        if not covered:
            raise SystemExit("gpu_probe covered no continuation positions")
        ids = [int(line) for line in
               Path(tokens_file).read_text().split()] if tokens_file else []
        agreements = 0
        kl_gufo_llama = 0.0
        kl_llama_gufo = 0.0
        worst_kl = 0.0
        worst_kl_position = None
        divergence = None
        for position in covered:
            gufo = gufo_rows[position]
            llama = llama_rows[position]
            if len(gufo) != len(llama):
                raise SystemExit(f"vocab mismatch at position {position}")
            gufo_top1 = max(range(len(gufo)), key=lambda i: (gufo[i], -i))
            llama_top1 = max(range(len(llama)), key=lambda i: (llama[i], -i))
            agreements += gufo_top1 == llama_top1
            log_g = log_softmax(gufo)
            log_l = log_softmax(llama)
            kgl = kl_divergence(log_g, log_l)
            kl_gufo_llama += kgl
            kl_llama_gufo += kl_divergence(log_l, log_g)
            if kgl > worst_kl:
                worst_kl = kgl
                worst_kl_position = position
            if (divergence is None and position + 1 < len(ids) and
                    position + 1 >= prompt_tokens):
                # llama.cpp's greedy token for this position is the scored
                # token; Gufo's argmax differing from it is the first
                # free-running greedy divergence.
                if gufo_top1 != ids[position + 1]:
                    gufo_first, gufo_second = top_two(gufo)
                    llama_first, llama_second = top_two(llama)
                    divergence = {
                        "position": position,
                        "forced_token": ids[position + 1],
                        "gufo_top1": [gufo_first[0], gufo_first[1]],
                        "gufo_top2": [gufo_second[0], gufo_second[1]],
                        "llama_top1": [llama_first[0], llama_first[1]],
                        "llama_top2": [llama_second[0], llama_second[1]],
                    }
        count = len(covered)
        summary = {
            "tag": args.tag,
            "model": os.path.basename(args.model),
            "prompt_tokens": prompt_tokens,
            "words": args.words,
            "seed": args.seed,
            "continuation_tokens": header["generated"],
            "scored_positions": header["positions"],
            "covered_positions": count,
            "covered_first": covered[0] if covered else None,
            "covered_last": covered[-1] if covered else None,
            "top1_agreement": agreements / count if count else 0.0,
            "mean_kl_gufo_llama": kl_gufo_llama / count if count else 0.0,
            "mean_kl_llama_gufo": kl_llama_gufo / count if count else 0.0,
            "worst_kl_gufo_llama": worst_kl,
            "worst_kl_position": worst_kl_position,
            "context": args.context,
            "llama_batch_requested": args.llama_batch,
            "llama_ubatch_requested": args.llama_ubatch,
            "llama_n_batch": header["llama_n_batch"],
            "llama_n_ubatch": header["llama_n_ubatch"],
            "gufo_prefill_chunk": args.batch,
            "divergence": divergence,
        }
        Path(args.out).write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
