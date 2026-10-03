#!/usr/bin/env python3
"""D19 G1 aggregation: median over three measurement series of pp and tg per
preset/tail/depth, the FP16 ratios, the depth 0-4K tg floor and the recorded
reference values (depth 32K+ tg, all-depth pp).

    python3 tools/qwen27b/g1_d19.py [--artifacts artifacts] [--json out.json]

Expects `artifacts/g1d19-r{1,2,3}/` written by three `g1.py run` series (see
docs/models/qwen3.8-27b/BENCHMARKS.md). Exit status is non-zero when the tg
floor fails.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

ROW = re.compile(
    r"^\|(?:[^|]*\|){3}\s*ROCm \(HIP\)\s*\|\s*(?P<test>[^|]+?)\s*\|\s*(?P<mean>[-+0-9.]+)"
)
DEPTHS = [0, 4096, 32768, 131072]
KVARN = ["k4v4-none", "k4v4-t2048", "k5v4-none", "k5v4-t2048",
         "k6v5-none", "k6v5-t2048"]
SERIES = [1, 2, 3]
TG_FLOOR_DEPTHS = (0, 4096)
TG_FLOOR = 0.95


def read(path: Path, depth: int, metric: str) -> float | None:
    if not path.exists():
        return None
    want = metric if depth == 0 else f"{metric} @ d{depth}"
    for line in path.read_text(errors="replace").splitlines():
        match = ROW.match(line)
        if match and match.group("test").strip() == want:
            return float(match.group("mean"))
    return None


def median(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    root = Path(args.artifacts)
    out = Path(args.json) if args.json else root / "g1d19-summary.json"

    fp16: dict[int, dict[str, float | None]] = {}
    kvarn: dict[tuple[str, int], dict[str, float | None]] = {}
    for depth in DEPTHS:
        fp16[depth] = {
            metric: median(
                [read(root / f"g1d19-r{r}" / f"fp16_d{depth}.log", depth, metric)
                 for r in SERIES])
            for metric in ("pp2048", "tg128")
        }
        for tag in KVARN:
            kvarn[(tag, depth)] = {
                metric: median(
                    [read(root / f"g1d19-r{r}" / f"{tag}_d{depth}.log", depth,
                          metric) for r in SERIES])
                for metric in ("pp2048", "tg128")
            }

    summary: dict = {"model_sha256": None, "fp16": {}, "presets": {},
                     "gate": {}, "reference": {}}
    for r in SERIES:
        identity = root / f"g1d19-r{r}" / "model.json"
        if identity.exists():
            summary["model_sha256"] = json.loads(identity.read_text()).get("sha256")
            break

    print("depth      fp16_pp   fp16_tg")
    for depth in DEPTHS:
        print(f"d{depth:<8} {fp16[depth]['pp2048']!s:>9} {fp16[depth]['tg128']!s:>9}")
        summary["fp16"][str(depth)] = dict(fp16[depth])

    failures: list[str] = []
    print("\npreset            depth      tg     tg/fp16      pp     pp/fp16")
    for tag in KVARN:
        summary["presets"][tag] = {}
        for depth in DEPTHS:
            entry = kvarn[(tag, depth)]
            tg, pp = entry["tg128"], entry["pp2048"]
            tg_ratio = (tg / fp16[depth]["tg128"]
                        if tg and fp16[depth]["tg128"] else None)
            pp_ratio = (pp / fp16[depth]["pp2048"]
                        if pp and fp16[depth]["pp2048"] else None)
            summary["presets"][tag][str(depth)] = {
                "tg128": tg, "pp2048": pp, "tg_ratio": tg_ratio,
                "pp_ratio": pp_ratio}
            print(f"{tag:<17} d{depth:<7} {tg!s:>6} {tg_ratio!s:>8} "
                  f"{pp!s:>7} {pp_ratio!s:>8}")
            if depth in TG_FLOOR_DEPTHS:
                ok = tg_ratio is not None and tg_ratio >= TG_FLOOR
                summary["gate"][f"{tag}:d{depth}"] = {"tg_ratio": tg_ratio,
                                                      "pass": ok}
                if not ok:
                    failures.append(f"{tag} d{depth} tg ratio {tg_ratio}")
            else:
                summary["reference"][f"{tag}:d{depth}:tg"] = tg_ratio
            summary["reference"][f"{tag}:d{depth}:pp"] = pp_ratio

    print("\nG1/D19 tg floor (depth 0-4K >= 0.95):",
          "PASS" if not failures else "FAIL")
    for failure in failures:
        print("  FAIL:", failure)
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print("wrote", out)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
