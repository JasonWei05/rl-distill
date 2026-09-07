#!/usr/bin/env python3
"""Generic pass@k curves from math-eval trace files.

Each ``--trace LABEL=PATH`` adds one curve; traces are grouped into panels by dataset (parsed from the
file name ``<tag>__<dataset>.jsonl``). pass@k is the unbiased estimator (Chen et al. 2021) for
k = 1..n where n = samples per question in that file, so 32-sample files give k = 1..32.

    python rl-distill-scripts/plot_passk_from_traces.py --out figures/passk_e4b_base_val32.png \
        --trace "E4B base=/tmp/gemma4_e4b_val32/id_medium/traces/base_e4b__id_medium.jsonl" \
        --trace "E4B base=/tmp/gemma4_e4b_val32/id_hard/traces/base_e4b__id_hard.jsonl"
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pass_at_k_curve(trace_path: Path) -> tuple[list[int], list[float], int]:
    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            total[row["uid"]] += 1
            correct[row["uid"]] += int(bool(row["acc"]))
    n = max(total.values())
    if min(total.values()) != n:
        raise ValueError(f"{trace_path}: uneven samples per question")
    ks = list(range(1, n + 1))
    curve = [100.0 * sum(1.0 - comb(n - c, k) / comb(n, k) for c in correct.values()) / len(correct) for k in ks]
    return ks, curve, len(correct)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True, help='"LABEL=PATH" (repeatable)')
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="pass@k (unbiased estimator); verifier = RL reward (strict last \\boxed{})")
    args = parser.parse_args()

    panels: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for item in args.trace:
        label, path = item.split("=", 1)
        dataset = Path(path).stem.split("__")[-1]
        panels[dataset].append((label, Path(path)))
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    print(f"{'dataset':<10}{'label':<28}{'q':>5}{'n':>4}{'pass@1':>8}{'pass@4':>8}{'pass@16':>9}{'pass@n':>8}")
    for ax, (dataset, series) in zip(axes[0], sorted(panels.items())):
        for label, path in series:
            ks, curve, questions = pass_at_k_curve(path)
            ax.plot(ks, curve, marker="o", ms=3, label=f"{label} (n={ks[-1]})")
            g = lambda k: curve[k - 1] if k <= len(curve) else float("nan")
            print(f"{dataset:<10}{label:<28}{questions:>5}{ks[-1]:>4}{g(1):>8.1f}{g(4):>8.1f}{g(16):>9.1f}{curve[-1]:>8.1f}")
        ax.set_xscale("log", base=2)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("k"); ax.set_ylabel("pass@k (%)"); ax.set_title(dataset); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(args.title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
