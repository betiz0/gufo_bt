#!/usr/bin/env python3
"""Long-distance reference check for the KVarN quality gate (task 6.2, D23).

One configuration per run. `gufo serve llm` with a fixed KV format serves one
32K context that holds 64 invented facts after the sink (first 128 tokens) and
far more than 2,304 tokens before the questions. All questions share the
prefix, so the server reuses the cached prefix KV and each configuration
prefills the 32K context once (design D23). Chat requests use
`reasoning_effort: low` and the answers are graded by exact match of the
4-digit code, so the check is a short, fixed problem set with no pass/fail
criterion (design D17/D23).

    longref_check.py --config kvarn-k5v4 --tail 2048 \
        --problems docs/models/qwen3.8-27b/artifacts/longref-32k.json \
        --output ~/.cache/gufo/longref --label t62-k5v4-2048 \
        --binary result/bin/gufo --model .../Qwen3.8-27B-Q8_0.gguf

Layout under `<output>/<label>/`:

    server.log        serve llm stdout/stderr (KV report line)
    calibration.json  chars-per-token probe, fact span, prefix tokens
    units/q<NN>.json  one question: settings, timings, answer text, grade
    summary.json      correct count, settings, hashes, per-question grades

`--dry-run` prints the assembled plan and hashes without starting a server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kvarn_consistency import Server, file_identity, http_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBLEMS = ROOT / "docs/models/qwen3.8-27b/artifacts/longref-32k.json"
DEFAULT_OUT = Path.home() / ".cache" / "gufo" / "longref"
SCHEMA_VERSION = 1
CHAT_ENDPOINT = "/v1/chat/completions"
RATIO_PROBE_CHARS = 4000
SERVER_READY_TIMEOUT = 900.0
REQUEST_TIMEOUT = 1800.0
DIGITS_RE = re.compile(r"[^0-9]")
KV_REPORT_RE = re.compile(r"kv_cache=\S+(?:[^\n]*)")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_atomically(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def fill_to(text: str, chars: int) -> str:
    if chars <= 0:
        return ""
    return (text * (chars // len(text) + 1))[:chars]


def load_problems(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    filler = document.get("filler")
    facts = document.get("facts")
    questions = document.get("questions")
    if not isinstance(filler, str) or not filler.strip():
        raise SystemExit(f"{path}: 'filler' must be non-empty text")
    if not isinstance(facts, list) or not facts:
        raise SystemExit(f"{path}: 'facts' must be a non-empty list")
    if not isinstance(questions, list) or not questions:
        raise SystemExit(f"{path}: 'questions' must be a non-empty list")
    for fact in facts:
        if not isinstance(fact.get("text"), str) or not fact["text"]:
            raise SystemExit(f"{path}: every fact needs non-empty 'text'")
    answers = []
    for question in questions:
        if not isinstance(question.get("text"), str) or not question["text"]:
            raise SystemExit(f"{path}: every question needs non-empty 'text'")
        if not isinstance(question.get("answer"), str) \
                or not question["answer"]:
            raise SystemExit(f"{path}: every question needs 'answer'")
        answers.append(question["answer"])
    if len(set(answers)) != len(answers):
        raise SystemExit(f"{path}: answers must be unique for exact grading")
    return document


def facts_text(document: dict) -> str:
    return "\n".join(fact["text"] for fact in document["facts"])


def build_prefix(document: dict, chars_a: int, chars_b: int) -> str:
    filler = document["filler"]
    return (fill_to(filler, chars_a) + "\n\n" + facts_text(document) + "\n\n"
            + fill_to(filler, chars_b))


def chat_body(args: argparse.Namespace, messages: list[dict],
              cache_prompt: bool, max_tokens: int) -> dict:
    body = {
        "model": args.served_model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": cache_prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "repeat_last_n": 64,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }
    if args.reasoning_effort != "off":
        body["reasoning_effort"] = args.reasoning_effort
    return body


def parse_chat(response: dict) -> dict:
    try:
        choice = response["choices"][0]
        message = choice["message"]
        usage = response["usage"]
        out = {
            "content": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content") or "",
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": int(usage["prompt_tokens"]),
            "completion_tokens": int(usage["completion_tokens"]),
            "cached_tokens": int(usage.get("cached_tokens") or 0),
            "timings": response.get("timings") or {},
        }
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"unexpected response shape: {json.dumps(response)[:400]}") \
            from error
    if out["completion_tokens"] > 0 and not (out["content"]
                                             or out["reasoning_content"]):
        raise RuntimeError(
            f"response reported {out['completion_tokens']} tokens but no "
            f"text: {json.dumps(response)[:400]}")
    return out


def probe_tokens(base_url: str, args: argparse.Namespace, text: str) -> int:
    """Prompt tokens the tokenizer assigns to `text` (filler + template)."""
    body = chat_body(args, [{"role": "user", "content": text}],
                     cache_prompt=False, max_tokens=1)
    response = http_json(base_url + CHAT_ENDPOINT, body, args.timeout)
    return parse_chat(response)["prompt_tokens"]


SINGLE_REQUEST_INSTRUCTION = (
    "以下に 64 問を並べます。各問いについて、"
    "「番号: 数字 4 桁」の形式で 1 行ずつ答えてください。"
    "説明や他の数字は書かないでください。\n"
)


def single_request_question(questions: list[dict]) -> str:
    lines = [SINGLE_REQUEST_INSTRUCTION]
    for question in questions:
        lines.append(f"{question['id']}: {question['text']}")
    return "\n".join(lines)


def parse_single_response(content: str) -> dict:
    """Codes keyed by question id, from lines like `01: 1137`."""
    codes = {}
    pattern = re.compile(r"(\d{1,2})\s*[:：.)、]\s*(\d{4})")
    for line in content.splitlines():
        for match in pattern.finditer(line):
            codes.setdefault(f"{int(match.group(1)):02d}", match.group(2))
    return codes


def single_settings(args: argparse.Namespace) -> dict:
    return {"reasoning_effort": args.reasoning_effort,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k, "cache_prompt": True,
            "single_request": True}


def run_single_request(server, args: argparse.Namespace, prefix: str,
                       questions: list[dict], calibration: dict,
                       root: Path) -> tuple[list[dict], dict]:
    """All questions in one request, so the 32K context is prefilled once per
    configuration (design D23). The per-question codes are read from the
    numbered answer lines; the raw generation is kept as the unit."""
    path = root / "units" / "single.json"
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        recorded_ids = [q.get("id") for q in raw.get("questions", [])]
        wanted_ids = [q["id"] for q in questions]
        if raw.get("prefix_sha256") != calibration["prefix_sha256"] or \
                (raw.get("settings") or {}) != single_settings(args) or \
                recorded_ids != wanted_ids:
            raise RuntimeError(
                f"{path} was measured against another prefix, settings or "
                f"question set ({len(recorded_ids)} vs {len(wanted_ids)} "
                f"questions); use a new --label")
    else:
        response = ask_question(server.base_url, args, prefix, [],
                                single_request_question(questions))
        raw = {
            "schema_version": SCHEMA_VERSION,
            "prefix_sha256": calibration["prefix_sha256"],
            "instruction": SINGLE_REQUEST_INSTRUCTION,
            "questions": [{"id": q["id"], "text": q["text"]} for q in questions],
            "content": response["content"],
            "reasoning_content": response["reasoning_content"],
            "finish_reason": response["finish_reason"],
            "prompt_tokens": response["prompt_tokens"],
            "completion_tokens": response["completion_tokens"],
            "cached_tokens": response["cached_tokens"],
            "timings": response["timings"],
            "seconds": response["seconds"],
            "settings": single_settings(args),
        }
        write_atomically(path, json.dumps(raw, ensure_ascii=False, indent=1)
                         + "\n")
    codes = parse_single_response(raw["content"])
    results = []
    for question in questions:
        code = codes.get(question["id"])
        results.append({
            "id": question["id"],
            "question": question["text"],
            "expected": question["answer"],
            "content": code or "",
            "correct": code == question["answer"],
            "empty_content": code is None,
            "cached_tokens": raw["cached_tokens"],
            "prompt_tokens": raw["prompt_tokens"],
            "completion_tokens": raw["completion_tokens"],
            "timings": raw["timings"],
            "seconds": raw["seconds"],
            "settings": raw["settings"],
        })
    return results, raw


def ask_question(base_url: str, args: argparse.Namespace, prefix: str,
                 history: list[dict], question: str) -> dict:
    """One question, either standalone or appended to the running history.

    `--sequential` keeps one conversation: each request extends the previous
    prompt by the last answer plus the new question, so the runner reuses the
    resident prefix and the 32K context is prefilled once per configuration
    (design D23). Standalone mode sends the shared system prefix again with a
    fresh user turn, which the continuation cache only reuses when it holds a
    matching checkpoint.
    """
    if args.sequential:
        messages = [{"role": "system", "content": prefix}, *history,
                    {"role": "user", "content": question}]
    else:
        messages = [{"role": "system", "content": prefix},
                    {"role": "user", "content": question}]
    body = chat_body(args, messages, cache_prompt=True,
                     max_tokens=args.max_tokens)
    started = time.perf_counter()
    response = http_json(base_url + CHAT_ENDPOINT, body, args.timeout)
    out = parse_chat(response)
    out["seconds"] = time.perf_counter() - started
    return out


def grade(content: str, answer: str) -> bool:
    return DIGITS_RE.sub("", content) == answer


def calibrate(server: Server, args: argparse.Namespace,
              document: dict) -> dict:
    """Probe the tokenizer, place the facts and size the prefix to 32K."""
    filler = document["filler"]
    ratio_tokens = probe_tokens(server.base_url, args,
                                fill_to(filler, RATIO_PROBE_CHARS))
    if ratio_tokens <= 0:
        raise RuntimeError("chars-per-token probe returned no prompt tokens")
    chars_per_token = RATIO_PROBE_CHARS / ratio_tokens

    chars_a = round(args.filler_a_tokens * chars_per_token)
    filler_a = fill_to(filler, chars_a)
    facts_start = probe_tokens(server.base_url, args, filler_a)
    facts_end = probe_tokens(server.base_url, args, filler_a + "\n\n"
                             + facts_text(document))

    prefix_budget = args.context_tokens - args.question_reserve_tokens
    filler_b_tokens = prefix_budget - facts_end
    if filler_b_tokens <= 0:
        raise RuntimeError(
            f"prefix budget {prefix_budget} tokens is below the fact block "
            f"end at {facts_end} tokens")
    chars_b = round(filler_b_tokens * chars_per_token)
    prefix = build_prefix(document, chars_a, chars_b)
    prefix_tokens = probe_tokens(server.base_url, args, prefix)

    distance = prefix_tokens - facts_end
    if facts_start < args.min_sink_tokens:
        raise RuntimeError(
            f"the facts start at token {facts_start}, inside the "
            f"{args.min_sink_tokens}-token sink")
    if distance < args.min_distance_tokens:
        raise RuntimeError(
            f"the facts end {distance} tokens before the question, less than "
            f"the required {args.min_distance_tokens}")
    return {
        "chars_per_token": chars_per_token,
        "chars_filler_a": chars_a,
        "chars_filler_b": chars_b,
        "facts_start_token": facts_start,
        "facts_end_token": facts_end,
        "prefix_tokens": prefix_tokens,
        "fact_to_question_distance": distance,
        "prefix_sha256": sha256_text(prefix),
    }, prefix


def cache_disk_dir(args: argparse.Namespace) -> Path:
    return Path(args.output) / "cache" / args.label


def configuration(args: argparse.Namespace) -> dict:
    return {"kv_cache": args.config, "tail": args.tail,
            "context": args.context_tokens, "sessions": args.sessions,
            "cache_disk": str(cache_disk_dir(args)) if args.cache_disk else None}


def server_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.binary), "serve", "--host", args.host, "--port", "0",
        "llm", "--served-model-name", args.served_model_name,
        "--model", str(args.model), "--context", str(args.server_context),
        "--sessions", str(args.sessions),
    ]
    if args.cache_disk:
        # Cross-request prefix reuse needs the opt-in continuation cache: the
        # runner stores a snapshot at the shared-prefix boundary during the
        # first request and later questions restore it, so each configuration
        # prefills the 32K context once (design D23). Its identity includes the
        # KV format (task 5.2), so configurations never share entries.
        command += ["--cache-disk", str(cache_disk_dir(args))]
        if args.cache_disk_bytes:
            command += ["--cache-disk-bytes", str(args.cache_disk_bytes)]
        if args.cache_disk_staging_bytes:
            command += ["--cache-disk-staging-bytes",
                        str(args.cache_disk_staging_bytes)]
    if args.config != "fp16":
        command += ["--kv-cache", args.config]
        if args.tail:
            command += ["--kv-cache-tail", str(args.tail)]
    return command


def run(args: argparse.Namespace) -> int:
    if args.max_tokens is None:
        args.max_tokens = 2048 if args.single_request else 160
    document = load_problems(args.problems)
    questions = document["questions"]
    if args.limit > 0:
        questions = questions[:args.limit]
    root = Path(args.output) / args.label
    root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        plan = {
            "problems": str(args.problems),
            "problems_sha256": hashlib.sha256(
                Path(args.problems).read_bytes()).hexdigest(),
            "facts": len(document["facts"]),
            "questions": len(questions),
            "filler_chars_estimate": round(
                (args.context_tokens - args.question_reserve_tokens)
                * args.chars_per_token_estimate),
            "configuration": configuration(args),
        }
        print(json.dumps(plan, ensure_ascii=False, indent=1))
        return 0

    command = server_command(args)
    with Server(command, root / "server.log", args.host,
                SERVER_READY_TIMEOUT) as server:
        started = time.perf_counter()
        calibration, prefix = calibrate(server, args, document)
        write_atomically(root / "calibration.json", json.dumps({
            "schema_version": SCHEMA_VERSION,
            "problems": str(args.problems),
            "configuration": configuration(args),
            "server_command": command,
            **calibration,
        }, ensure_ascii=False, indent=1) + "\n")
        print(f"calibration: prefix {calibration['prefix_tokens']} tokens, "
              f"facts at tokens "
              f"{calibration['facts_start_token']}.."
              f"{calibration['facts_end_token']}, distance "
              f"{calibration['fact_to_question_distance']}", flush=True)

        results = []
        correct = 0
        history: list[dict] = []
        if args.single_request:
            results, raw = run_single_request(server, args, prefix, questions,
                                              calibration, root)
            correct = sum(1 for unit in results if unit["correct"])
            empty = sum(1 for unit in results if unit["empty_content"])
            print(f"single request: {correct}/{len(results)} correct, {empty} "
                  f"unparsed, prompt_n={raw['prompt_tokens']} "
                  f"completion_n={raw['completion_tokens']} in "
                  f"{raw['seconds']:.1f} s", flush=True)
            questions = []
        for index, question in enumerate(questions, start=1):
            path = root / "units" / f"q{question['id']}.json"
            if path.is_file():
                unit = json.loads(path.read_text(encoding="utf-8"))
                if unit.get("prefix_sha256") != calibration["prefix_sha256"]:
                    raise RuntimeError(
                        f"{path} was measured against a different prefix "
                        f"({unit.get('prefix_sha256')} vs "
                        f"{calibration['prefix_sha256']}); use a new --label")
                recorded = unit.get("settings") or {}
                current = {"reasoning_effort": args.reasoning_effort,
                           "max_tokens": args.max_tokens,
                           "temperature": args.temperature,
                           "top_p": args.top_p, "top_k": args.top_k,
                           "cache_prompt": True,
                           "sequential": args.sequential}
                if recorded != current:
                    raise RuntimeError(
                        f"{path} was measured with {recorded} but this run "
                        f"uses {current}; use a new --label")
                if args.sequential and unit.get("content"):
                    history.append({"role": "assistant",
                                    "content": unit["content"]})
            else:
                response = ask_question(server.base_url, args, prefix,
                                        history, question["text"])
                if args.sequential and response["content"]:
                    history.append({"role": "assistant",
                                    "content": response["content"]})
                unit = {
                    "schema_version": SCHEMA_VERSION,
                    "id": question["id"],
                    "prefix_sha256": calibration["prefix_sha256"],
                    "question": question["text"],
                    "expected": question["answer"],
                    "content": response["content"],
                    "reasoning_content": response["reasoning_content"],
                    "finish_reason": response["finish_reason"],
                    "correct": grade(response["content"], question["answer"]),
                    "empty_content": not response["content"].strip(),
                    "prompt_tokens": response["prompt_tokens"],
                    "completion_tokens": response["completion_tokens"],
                    "cached_tokens": response["cached_tokens"],
                    "timings": response["timings"],
                    "seconds": response["seconds"],
                    "settings": {
                        "reasoning_effort": args.reasoning_effort,
                        "max_tokens": args.max_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "cache_prompt": True,
                        "sequential": args.sequential,
                    },
                }
                write_atomically(path, json.dumps(unit, ensure_ascii=False,
                                                  indent=1) + "\n")
            results.append(unit)
            correct += 1 if unit["correct"] else 0
            empty = sum(1 for u in results if u["empty_content"])
            print(f"Q {index}/{len(questions)} id={unit['id']} "
                  f"correct={correct} empty={empty} "
                  f"prompt_n={unit['prompt_tokens']} "
                  f"cache_n={unit['cached_tokens']} "
                  f"cache_prompt_n={unit['timings'].get('prompt_n')} "
                  f"in={unit['seconds']:.1f} s t="
                  f"{time.perf_counter() - started:.1f} s", flush=True)

        empty = sum(1 for unit in results if unit["empty_content"])
        summary = {
            "schema_version": SCHEMA_VERSION,
            "label": args.label,
            "problems": str(args.problems),
            "problems_sha256": hashlib.sha256(
                Path(args.problems).read_bytes()).hexdigest(),
            "model": file_identity(args.model),
            "binary": file_identity(args.binary),
            "configuration": configuration(args),
            "server_command": command,
            "calibration": calibration,
            "settings": {
                "reasoning_effort": args.reasoning_effort,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "cache_prompt": True,
                "sequential": args.sequential,
            },
            "questions": len(results),
            "correct": correct,
            "empty_content": empty,
            "seconds": time.perf_counter() - started,
            "prefix_reuse": {
                "cached_tokens_last": results[-1]["cached_tokens"] if results
                else 0,
                "prompt_n_first": results[0]["timings"].get("prompt_n")
                if results else None,
                "prompt_n_last": results[-1]["timings"].get("prompt_n")
                if results else None,
            },
            "per_question": [
                {"id": unit["id"], "correct": unit["correct"],
                 "expected": unit["expected"], "content": unit["content"],
                 "cached_tokens": unit["cached_tokens"]}
                for unit in results
            ],
        }
        log = (root / "server.log").read_text(encoding="utf-8",
                                              errors="replace")
        match = KV_REPORT_RE.search(log)
        summary["server_kv_report"] = match.group(0) if match else None
        write_atomically(root / "summary.json",
                         json.dumps(summary, ensure_ascii=False, indent=1)
                         + "\n")
        print(f"summary: {args.config} tail {args.tail or 'none'}: "
              f"{correct}/{len(results)} correct, {empty} empty, "
              f"{summary['seconds'] / 60.0:.1f} min", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True,
                        help="KV format: fp16 or a KVarN preset name")
    parser.add_argument("--tail", type=int, default=0,
                        help="--kv-cache-tail value, 0 = not given")
    parser.add_argument("--problems", type=Path, default=DEFAULT_PROBLEMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--label", required=True)
    parser.add_argument("--binary", type=Path, required=True,
                        help="gufo executable that provides `serve llm`")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--served-model-name", default="qwen3.8-27b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--context", dest="context_tokens", type=int,
                        default=32768,
                        help="target prompt tokens of the shared prefix")
    parser.add_argument("--server-context", type=int, default=40960,
                        help="serve llm --context: KV capacity, kept above the "
                             "prefix plus question and answer")
    parser.add_argument("--single-request", dest="single_request",
                        action="store_true",
                        help="ask all questions in one request: the 32K "
                             "context is prefilled exactly once per "
                             "configuration and the numbered answer lines are "
                             "parsed per question (used by task 6.2)")
    parser.add_argument("--sequential", dest="sequential",
                        action="store_true", default=True,
                        help="ask the 64 questions in one growing "
                             "conversation so each request extends the "
                             "previous prompt and the runner reuses the "
                             "resident prefix (default)")
    parser.add_argument("--standalone-questions", dest="sequential",
                        action="store_false",
                        help="send every question as a fresh user turn on the "
                             "shared system prefix (needs a matching "
                             "continuation-cache checkpoint)")
    parser.add_argument("--no-cache-disk", dest="cache_disk",
                        action="store_false",
                        help="do not pass --cache-disk; without the "
                             "continuation cache every question re-prefills "
                             "the whole prefix")
    parser.add_argument("--cache-disk-bytes", type=int, default=0,
                        help="--cache-disk-bytes (0 = server default)")
    parser.add_argument("--cache-disk-staging-bytes", type=int, default=0,
                        help="--cache-disk-staging-bytes (0 = server default)")
    parser.set_defaults(cache_disk=True)
    parser.add_argument("--question-reserve-tokens", type=int, default=256,
                        help="headroom below --context for question+answer")
    parser.add_argument("--filler-a-tokens", type=int, default=2048,
                        help="filler tokens before the fact block")
    parser.add_argument("--min-sink-tokens", type=int, default=128,
                        help="the facts must start after this many tokens")
    parser.add_argument("--min-distance-tokens", type=int, default=2304,
                        help="minimum distance from the facts to a question")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="answer budget (default: 160 per question, 2048 "
                             "for --single-request); `reasoning_effort: low` "
                             "spends part of it on the reasoning channel, so a "
                             "value near 80 truncates the answer")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0,
                        help="run only the first N questions (smoke tests)")
    parser.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT)
    parser.add_argument("--chars-per-token-estimate", type=float, default=3.9,
                        help="dry-run only: chars per token for the plan")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without starting a server")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except (RuntimeError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
