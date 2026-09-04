#!/usr/bin/env python3
"""pass@k curves for one student architecture of the distillation study.

Reads the per-sample trace files the math eval writes
(``<results>/<tag>/<tag>/math/traces/<tag>__<dataset>.jsonl``, one line per (question, sample) with
``acc``) and plots the unbiased pass@k estimator (Chen et al. 2021: mean over questions of
1 - C(n-c, k)/C(n, k)) for k = 1..n. One row per band the trained models were trained on (easy /
medium / hard); columns = that band, MATH500, GSM8K; curves = untrained base, RL, and the distilled
students from the requested teachers (self-distillation excluded by default).

    python rl-distill-scripts/plot_distill_study_passk.py --student e4b --teachers 12b 26b \
        --out rl-distill-scripts/figures/passk_e4b.png
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANDS = ("easy", "medium", "hard")
COLUMNS = [("id_{band}", "in-distribution ({band} band, 300 q)"), ("math500", "MATH500"), ("gsm8k", "GSM8K")]
STYLE = {"base": dict(color="0.45", ls="--"), "rl": dict(color="black", ls="-"), "12b": dict(color="tab:blue", ls="-"), "26b": dict(color="tab:red", ls="-"), "e4b": dict(color="tab:green", ls="-"), "e2b": dict(color="tab:orange", ls="-")}


def pass_at_k_curve(trace_path: Path) -> tuple[list[int], list[float]]:
    correct: dict[str, int] = {}
    total: dict[str, int] = {}
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            total[row["uid"]] = total.get(row["uid"], 0) + 1
            correct[row["uid"]] = correct.get(row["uid"], 0) + int(bool(row["acc"]))
    n = max(total.values())
    if min(total.values()) != n:
        raise ValueError(f"{trace_path}: uneven samples per question")
    ks = list(range(1, n + 1))
    curve = []
    for k in ks:
        vals = [1.0 - comb(n - c, k) / comb(n, k) for c in correct.values()]
        curve.append(100.0 * sum(vals) / len(vals))
    return ks, curve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="e4b", choices=("e4b", "e2b"))
    parser.add_argument("--teachers", nargs="+", default=["12b", "26b"], help="distillation teachers to plot (RL of this size -> student)")
    parser.add_argument("--results-base", type=Path, default=Path("/tmp/gemma4_distill_study_eval/results"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or Path(__file__).resolve().parent / "figures" / f"passk_{args.student}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    def trace(tag: str, dataset: str) -> Path | None:
        path = args.results_base / tag / tag / "math" / "traces" / f"{tag}__{dataset}.jsonl"
        return path if path.exists() else None

    fig, axes = plt.subplots(len(BANDS), len(COLUMNS), figsize=(14, 11), sharey=False)
    summary = []
    for r, band in enumerate(BANDS):
        series = [("base", f"{args.student.upper()} base", f"base_{args.student}"), ("rl", f"{args.student.upper()} RL ({band})", f"rl_{args.student}_{band}")]
        series += [(t, f"{t} RL ({band}) → {args.student.upper()}", f"distill_{t}_{band}_to_{args.student}") for t in args.teachers]
        for c, (dataset_tpl, title_tpl) in enumerate(COLUMNS):
            dataset = dataset_tpl.format(band=band)
            ax = axes[r][c]
            for key, label, tag in series:
                path = trace(tag, dataset)
                if path is None:
                    ax.plot([], [], label=f"{label} (missing)", **STYLE[key])
                    continue
                ks, curve = pass_at_k_curve(path)
                ax.plot(ks, curve, marker="o", ms=3, label=label, **STYLE[key])
                summary.append((band, dataset, label, curve[0], curve[min(3, len(curve) - 1)], curve[-1]))
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8, 16] if dataset != "gsm8k" else [1, 2, 4, 8])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel("k")
            ax.set_ylabel("pass@k (%)")
            ax.set_title(title_tpl.format(band=band), fontsize=10)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(
        f"Gemma 4 {args.student.upper()} student — pass@k (unbiased estimator; 16 samples, GSM8K 8). "
        f"Rows: band the RL / distilled models were trained on. Verifier = RL reward (strict last \\boxed{{}}).",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    print(f"{'band':<7}{'dataset':<10}{'model':<26}{'pass@1':>8}{'pass@4':>8}{'pass@max':>9}")
    for band, dataset, label, p1, p4, pmax in summary:
        print(f"{band:<7}{dataset:<10}{label:<26}{p1:>8.1f}{p4:>8.1f}{pmax:>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
