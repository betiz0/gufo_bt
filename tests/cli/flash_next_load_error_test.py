#!/usr/bin/env python3
"""CLI load-error check for an unsupported qwen4exp tensor format.

Builds a synthetic `qwen4exp` GGUF whose token embedding is stored as IQ3_S
(a format the weight-class table refuses for that class) and checks that
`gufo prompt`, `gufo chat` and `gufo serve llm` all exit with a failure status
and print the offending tensor and format.

Usage: flash_next_load_error_test.py <gufo> <fixture-emitter>
"""

import os
import subprocess
import sys
import tempfile

TENSOR = "token_embd.weight"
FORMAT = "IQ3_S"
TIMEOUT_SECONDS = 120


def run(command, fixture):
    return subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=TIMEOUT_SECONDS,
    )


def check(name, command, fixture):
    result = run(command, fixture)
    output = result.stdout.decode("utf-8", "replace")
    if result.returncode == 0:
        raise SystemExit(f"{name}: expected a failure status, got success\n{output}")
    if TENSOR not in output or FORMAT not in output:
        raise SystemExit(
            f"{name}: load error does not name {TENSOR} and {FORMAT}\n{output}"
        )
    print(f"{name}: exit {result.returncode}, message names {TENSOR} and {FORMAT}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <gufo> <fixture-emitter>")
    gufo, emitter = sys.argv[1], sys.argv[2]

    with tempfile.TemporaryDirectory(prefix="qfn-cli-load-") as directory:
        fixture = os.path.join(directory, "unsupported-format.gguf")
        emit = subprocess.run(
            [emitter, "--emit", "cli", fixture],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=TIMEOUT_SECONDS,
        )
        if emit.returncode != 0 or not os.path.exists(fixture):
            raise SystemExit(
                "fixture generation failed: "
                + emit.stdout.decode("utf-8", "replace")
            )
        check("prompt", [gufo, "prompt", "-m", fixture, "-p", "hi", "-n", "1"], fixture)
        check("chat", [gufo, "chat", "-m", fixture], fixture)
        check("serve", [gufo, "serve", "llm", "-m", fixture], fixture)

    print("Flash-Next CLI load-error checks passed.")


if __name__ == "__main__":
    main()
