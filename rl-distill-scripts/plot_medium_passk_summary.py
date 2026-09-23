#!/usr/bin/env python3
"""One-panel id_medium pass@k summary of the E4B distillation study (§9.0o wrap-up).

Curves (unbiased pass@k, Chen et al. 2021, from the per-sample math-eval traces, 300 questions × 16 samples):
  E4B RL (bs 64, seeds 42/43/44) and E4B RL 4× batch (bs 256, seeds 45/46/47) as seed means with each seed as a faint line; and the four 1000-step 2e-6 → 2e-7 distilled students — teacher = untrained-12B RL, off-policy-distilled-12B RL,
  untrained-26B RL (warm-started run), off-policy-distilled-26B RL.

    python rl-distill-scripts/plot_medium_passk_summary.py [--out figures/passk_id_medium_summary.png]
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("/tmp/gemma4_distill_study_eval/results")
DATASET = "id_medium"
SINGLES = [  # (tag, label, color)
    ("distill_12br_lr2e6_medium_to_e4b_step1000", "12B Base RL → E4B", "tab:blue"),
    ("distill_12bd_medium_to_e4b_step1000", "(E4B → 12B) RL → E4B", "tab:cyan"),
    ("distill_26bw_lr2e6_medium_to_e4b_step1000_n8", "26B Base RL → E4B", "tab:red"),
    ("distill_26bd_lr2e6_medium_to_e4b_step1000", "(E4B → 26B) RL → E4B", "tab:orange"),
]
GROUPS = [  # (tags, label, color)
    (["rl_e4b_medium", "rl_e4b_medium_s43", "rl_e4b_medium_s44"], "E4B RL, bs 64 (mean of 3 seeds; faint = each seed)", "black"),
    (["rl_e4b_medium_bs256_s45", "rl_e4b_medium_bs256_s46", "rl_e4b_medium_bs256_s47"], "E4B RL, bs 256 (mean of 3 seeds; faint = each seed)", "tab:green"),
]


def pass_at_k_curve(tag: str) -> np.ndarray:
    path = RESULTS / tag / tag / "math" / "traces" / f"{tag}__{DATASET}.jsonl"
    correct: dict[str, int] = {}
    total: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            total[row["uid"]] = total.get(row["uid"], 0) + 1
            correct[row["uid"]] = correct.get(row["uid"], 0) + int(bool(row["acc"]))
    n = max(total.values())
    if min(total.values()) != n:
        raise ValueError(f"{path}: uneven samples per question")
    counts = np.array(list(correct.values()))
    return np.array([100.0 * np.mean([1.0 - comb(n - c, k) / comb(n, k) for c in counts]) for k in range(1, n + 1)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures" / "passk_id_medium_summary.png")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ks = None
    for tags, label, color in GROUPS:
        curves = np.stack([pass_at_k_curve(t) for t in tags])
        ks = np.arange(1, curves.shape[1] + 1)
        mean = curves.mean(axis=0)
        for curve in curves:
            ax.plot(ks, curve, color=color, lw=1, alpha=0.3)
        ax.plot(ks, mean, color=color, lw=2, marker="o", ms=3, label=label)
        print(f"{label}: pass@1 {mean[0]:.1f} (seeds {', '.join(f'{c[0]:.1f}' for c in curves)})  pass@16 {mean[-1]:.1f} (seeds {', '.join(f'{c[-1]:.1f}' for c in curves)})")
    for tag, label, color in SINGLES:
        curve = pass_at_k_curve(tag)
        ks = np.arange(1, len(curve) + 1)
        ax.plot(ks, curve, color=color, lw=2, marker="o", ms=3, label=label)
        print(f"{label}: pass@1 {curve[0]:.1f}  pass@16 {curve[-1]:.1f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k (samples per question)")
    ax.set_ylabel("pass@k (%)")
    ax.set_title("E4B models on the medium band (300 held-out questions, 16 samples)\nall distilled models: 1000 steps, lr 2e-6 → 2e-7, top-128 KL on 3000×16 teacher traces", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
