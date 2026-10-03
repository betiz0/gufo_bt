#!/usr/bin/env python3
"""`--kv-cache` and `--kv-cache-tail` option checks that need no model weights.

Covers the canonical options on `serve llm`, `chat`, `prompt` and `bench`:
acceptance and help text, rejection of unknown values and of a tail outside
256..2048 (or without a KVarN format) before any model load, and the startup
rejection of an explicit value (`fp16` included) on models that do not offer
the options. The phase-0 probe values (`pseudo-kvarn-*`, `baseline-*`) no
longer exist in any build and must be rejected as unknown values.

Usage: kv_cache_test.py <gufo>
"""

import os
import socket
import struct
import subprocess
import sys
import tempfile
import time

COMMANDS = (["serve", "llm"], ["chat"], ["prompt"], ["bench"])
TIMEOUT_SECONDS = 30
# Loading a full model before its startup log is a separate, longer budget.
MODEL_LOAD_SECONDS = 300
TAIL_MIN = 256
TAIL_MAX = 2048
GGUF_STRING = 8
GGUF_ALIGNMENT = 32
# The production KVarN presets (design D1), registered in every build.
PRODUCTION_PRESETS = ("kvarn-k4v4", "kvarn-k5v4", "kvarn-k6v5")
# Phase-0 probe values, removed with the pseudo-quantization path (task 3.10).
REMOVED_PROBES = ("pseudo-kvarn-k4v4", "pseudo-kvarn-k5v4",
                  "pseudo-kvarn-k3v4", "pseudo-kvarn-k4v4-l0",
                  "pseudo-kvarn-k5v4-l1024", "baseline-q8", "baseline-q4",
                  "baseline-q4-l256", "baseline-q5")


def run(binary, args):
    result = subprocess.run([binary, *args], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=TIMEOUT_SECONDS)
    return result.returncode, result.stdout.decode("utf-8", "replace")


def write_architecture_only_gguf(path, architecture):
    """Writes a GGUF v3 file whose only metadata is general.architecture."""
    key = b"general.architecture"
    value = architecture.encode()
    with open(path, "wb") as out:
        out.write(b"GGUF" + struct.pack("<IQQ", 3, 0, 1))
        out.write(struct.pack("<Q", len(key)) + key)
        out.write(struct.pack("<IQ", GGUF_STRING, len(value)) + value)
        out.write(b"\0" * (-out.tell() % GGUF_ALIGNMENT))


def expect(name, condition, output):
    if not condition:
        raise SystemExit(f"{name}\n{output}")


def check_tail(binary):
    """`--kv-cache-tail`: help, range, and the KVarN-only rule. Every error
    exits 2 before the model is opened, so the missing model is never named."""
    for command in COMMANDS:
        label = " ".join(command)
        status, output = run(binary, [*command, "--help"])
        expect(f"{label}: --help lacks --kv-cache-tail and its range",
               status == 0 and "--kv-cache-tail" in output
               and str(TAIL_MIN) in output and str(TAIL_MAX) in output, output)

        rejected = [
            (["--kv-cache", "kvarn-k4v4", "--kv-cache-tail", value],
             f"tail {value!r}")
            for value in (str(TAIL_MIN - 1), str(TAIL_MAX + 1), "0", "-1",
                          "abc", "512.5", "5e2", "")]
        rejected += [
            (["--kv-cache", "fp16", "--kv-cache-tail", "512"],
             "tail with fp16"),
            (["--kv-cache-tail", "512"], "tail without --kv-cache")]
        for arguments, what in rejected:
            status, output = run(binary, [*command, *arguments,
                                          "--model", "missing.gguf"])
            expect(f"{label}: {what} must exit 2 before loading",
                   status == 2 and "missing.gguf" not in output, output)

        for value in (str(TAIL_MIN), "1024", str(TAIL_MAX)):
            status, output = run(binary, [
                *command, "--kv-cache", "kvarn-k4v4",
                "--kv-cache-tail", value, "--model", "missing.gguf"])
            expect(f"{label}: KVarN tail {value} must be accepted",
                   status != 2 and "--kv-cache" not in output, output)


def check_production_presets(binary):
    """The production KVarN presets are accepted by every parser, listed in
    --help, and combine with --kv-cache-tail. A tail outside the range still
    exits 2 before the model is opened."""
    for command in COMMANDS:
        label = " ".join(command)
        status, output = run(binary, [*command, "--help"])
        for preset in PRODUCTION_PRESETS:
            expect(f"{label}: --help lacks {preset}", preset in output, output)
        for preset in PRODUCTION_PRESETS:
            status, output = run(binary, [*command, "--kv-cache", preset,
                                          "--model", "missing.gguf"])
            expect(f"{label}: {preset} must be accepted by the parser",
                   status != 2 and "--kv-cache value" not in output, output)
            status, output = run(binary, [*command, "--kv-cache", preset,
                                          "--kv-cache-tail", str(TAIL_MAX),
                                          "--model", "missing.gguf"])
            expect(f"{label}: {preset} with a tail must be accepted",
                   status != 2 and "--kv-cache value" not in output, output)
        status, output = run(binary, [*command, "--kv-cache", "kvarn-k4v4",
                                      "--kv-cache-tail", str(TAIL_MIN - 1),
                                      "--model", "missing.gguf"])
        expect(f"{label}: a tail below the range must exit 2",
               status == 2 and "missing.gguf" not in output, output)


def check_model_report(binary):
    """A model that does not offer `--kv-cache` still reports its own KV format
    in the startup log. Only DeepSeek V4 Flash has a format distinct from F16
    (`fp8`), so its artifact is what proves the report is model-specific."""
    model = os.environ.get("GUFO_DEEPSEEK_V4_FLASH_MODEL")
    if not model:
        print("SKIP: GUFO_DEEPSEEK_V4_FLASH_MODEL is not set")
        return
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [binary, "serve", "llm", "--model", model, "--context", "2048",
         "--host", "127.0.0.1", "--port", str(port)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    output = ""
    deadline = time.monotonic() + MODEL_LOAD_SECONDS
    try:
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            output += line
            if "kv_cache=" in line:
                break
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
    expect("DeepSeek V4 Flash: startup log must report kv_cache=fp8",
           "kv_cache=fp8" in output, output)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <gufo>")
    binary = sys.argv[1]

    for command in COMMANDS:
        label = " ".join(command)
        status, output = run(binary, [*command, "--help"])
        expect(f"{label}: --help lacks --kv-cache",
               status == 0 and "--kv-cache" in output and "fp16" in output,
               output)

        status, output = run(binary, [*command, "--kv-cache", "bogus",
                                      "--model", "missing.gguf"])
        expect(f"{label}: unknown value must exit 2 before loading",
               status == 2 and "--kv-cache" in output
               and "missing.gguf" not in output, output)

        status, output = run(binary, [*command, "--kv-cache", "fp16",
                                      "--model", "missing.gguf"])
        expect(f"{label}: fp16 must be accepted by the parser",
               status != 2 and "--kv-cache value" not in output, output)

    for command in COMMANDS:
        label = " ".join(command)
        # Removed phase-0 probes are unknown values in every build now.
        for value in REMOVED_PROBES:
            status, output = run(binary, [*command, "--kv-cache", value,
                                          "--model", "missing.gguf"])
            expect(f"{label}: removed probe {value} must be rejected",
                   status == 2 and "--kv-cache" in output, output)
        # The tail is an option, so a name suffix is malformed.
        for value in ("kvarn-k4v4-l256", "kvarn-k7v4", "kvarn-k4v5"):
            status, output = run(binary, [*command, "--kv-cache", value,
                                          "--model", "missing.gguf"])
            expect(f"{label}: malformed {value} must be rejected",
                   status == 2 and "--kv-cache" in output, output)

    check_tail(binary)
    check_production_presets(binary)
    check_model_report(binary)

    with tempfile.TemporaryDirectory(prefix="kv-cache-cli-") as directory:
        for architecture, model in (("deepseek4", "DeepSeek V4 Flash"),
                                    ("qwen4exp", "Qwen3.8-Flash-Next")):
            fixture = os.path.join(directory, architecture + ".gguf")
            write_architecture_only_gguf(fixture, architecture)
            for command in COMMANDS:
                for value in ("fp16", "kvarn-k4v4"):
                    label = f"{' '.join(command)} {architecture} {value}"
                    # serve resolves the native context before loading.
                    extra = ["--context", "4096"] if command[0] == "serve" else []
                    status, output = run(binary, [*command, "--model",
                                                  fixture, *extra,
                                                  "--kv-cache", value])
                    expect(f"{label}: must exit non-zero",
                           status != 0, output)
                    expect(f"{label}: must name the unsupported option",
                           "--kv-cache" in output, output)
                for arguments in (["--kv-cache-tail", "512"],
                                  ["--kv-cache", "kvarn-k4v4",
                                   "--kv-cache-tail", "512"]):
                    label = f"{' '.join(command)} {architecture} {arguments}"
                    status, output = run(binary, [*command, "--model",
                                                  fixture, *extra, *arguments])
                    expect(f"{label}: must exit non-zero", status != 0, output)
                    expect(f"{label}: must name the unsupported option",
                           "--kv-cache" in output, output)

    print("--kv-cache CLI checks passed.")


if __name__ == "__main__":
    main()
