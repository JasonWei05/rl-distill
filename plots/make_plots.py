#!/usr/bin/env python3
"""Regenerate the seed-sweep CI plots.

Style/method matched to the original 1B panels: per training step, aggregate across seeds as
mean +/- 95% CI where CI = t_{0.975, df=n-1} * (sample_std / sqrt(n)) (scipy t), step 0 dropped,
seaborn-whitegrid, markers on the mean line. Accuracy bands clipped at 0 (scores are >= 0).

Outputs (all -> rl-distill/plots/):
  4B (2 seeds, run ids below): per-metric panels + a 3-panel seed_sweep, each drawing the TWO
    individual seed curves + the mean + the 95% CI band.
  Overlay: 1B (3 seeds) vs 4B (2 seeds) val accuracy on one axis, each as mean + 95% CI band.

Run:  set -a && source .env; set +a ; .venv/bin/python plots/make_plots.py
"""
import warnings

import matplotlib
import numpy as np
import pandas as pd
import wandb
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanstd on all-nan slices
plt.style.use("seaborn-v0_8-whitegrid")

PLOTS = "/mnt/efs/jasonwei/rl-distill/plots"
ENTITY_PROJ = "rl-distill/DAPO"
api = wandb.Api()

# run ids -> seed label
RUNS_4B = {"ul9vfhdw": "seed 42", "mo2ftnik": "seed 43"}
RUNS_1B = {"rdb4on62": "seed 42", "rwmx0jg0": "seed 43", "gjjze9j5": "seed 44"}

# smooth: rolling-mean window (in logged points) for dense per-step metrics; None = raw.
METRICS = [
    ("val-core/math/acc/mean@16", "val math acc  mean@16", "#3b7dd8", "val-core_math_acc_meanat16", None),
    ("val-core/math/acc/best@16/mean", "val math acc  best@16/mean", "#e8613c", "val-core_math_acc_bestat16_mean", None),
    ("train/score_mean", "train score_mean (smoothed)", "#3aa03a", "train_score_mean", 15),
]


def series(rid, metric, smooth=None):
    """Full per-step history for one run/metric -> DataFrame[step, value], step>0.

    train/score_mean is logged every step and is very noisy; `smooth` applies a centered
    rolling mean over that many logged points (standard for RL reward curves)."""
    h = api.run(f"{ENTITY_PROJ}/runs/{rid}").history(keys=[metric], samples=100000, pandas=True)
    h = h[["_step", metric]].dropna().rename(columns={"_step": "step", metric: "value"})
    h = h[h["step"] > 0].drop_duplicates("step").sort_values("step").reset_index(drop=True)
    if smooth and len(h) > smooth:
        h["value"] = h["value"].rolling(smooth, center=True, min_periods=1).mean()
    return h


def aggregate(run_ids, metric, smooth=None):
    """Outer-join seeds on step; return (steps, mean, ci_halfwidth, n_per_step, {rid: df})."""
    per_run = {rid: series(rid, metric, smooth) for rid in run_ids}
    wide = None
    for rid, df in per_run.items():
        col = df.rename(columns={"value": rid})[["step", rid]]
        wide = col if wide is None else wide.merge(col, on="step", how="outer")
    wide = wide.sort_values("step").reset_index(drop=True)
    steps = wide["step"].to_numpy()
    mat = wide[list(run_ids)].to_numpy(dtype=float)  # [n_steps, n_seeds], NaN where missing
    n = np.sum(~np.isnan(mat), axis=1)
    mean = np.nanmean(mat, axis=1)
    half = np.zeros_like(mean)
    for i in range(len(mean)):
        if n[i] >= 2:
            sd = np.nanstd(mat[i], ddof=1)
            half[i] = stats.t.ppf(0.975, n[i] - 1) * sd / np.sqrt(n[i])
    return steps, mean, half, n, per_run


def draw_panel(ax, run_ids, metric, title, color, smooth=None, clip0=True, show_individual=True):
    steps, mean, half, n, per_run = aggregate(run_ids, metric, smooth)
    lo, hi = mean - half, mean + half
    if clip0:
        lo = np.clip(lo, 0, None)
    ax.fill_between(steps, lo, hi, alpha=0.18, color=color, lw=0)
    if show_individual:
        for rid, df in per_run.items():
            ax.plot(df["step"], df["value"], "-", color=color, alpha=0.4, lw=1.1)
            ax.plot([], [], "-", color=color, alpha=0.5, lw=1.1, label=run_ids[rid])  # legend proxy
    # thin markers on dense (per-step) lines so they don't blob into a solid band
    me = max(1, len(steps) // 45) if len(steps) > 80 else 1
    ax.plot(steps, mean, "-o", color=color, ms=3.5, lw=1.8, markevery=me, label="mean")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("training step")
    ax.set_ylabel("accuracy" if metric.startswith("val") else "score")
    return steps.max() if len(steps) else 0


def main():
    # ---- 4B: individual per-metric panels (two seed curves + mean + 95% CI) ----
    for metric, title, color, fname, smooth in METRICS:
        fig, ax = plt.subplots(figsize=(7.4, 5.1))
        draw_panel(ax, RUNS_4B, metric, f"{title}   (mean & 95% CI, 2 seeds)", color, smooth)
        ax.legend(frameon=False, fontsize=9, loc="best")
        fig.tight_layout()
        out = f"{PLOTS}/panel_4b_{fname}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print("wrote", out)

    # ---- 4B: 3-panel seed sweep ----
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.4))
    order = ["val-core/math/acc/mean@16", "train/score_mean", "val-core/math/acc/best@16/mean"]
    lut = {m[0]: m for m in METRICS}
    for ax, m in zip(axes, order):
        _, title, color, _, smooth = lut[m]
        draw_panel(ax, RUNS_4B, m, title, color, smooth)
        ax.legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("Gemma3-4B-PT few-shot math RL  —  mean & 95% CI over 2 seeds (42/43)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"{PLOTS}/seed_sweep_metrics_4b.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("wrote", out)

    # ---- Overlay: 1B vs 4B val accuracy (mean + 95% CI band, no individual curves) ----
    for metric, mlabel, fname in [
        ("val-core/math/acc/mean@16", "mean@16", "meanat16"),
        ("val-core/math/acc/best@16/mean", "best@16/mean", "bestat16"),
    ]:
        fig, ax = plt.subplots(figsize=(8.2, 5.4))
        for run_ids, cohort, color in [(RUNS_1B, "1B (3 seeds)", "#3b7dd8"),
                                       (RUNS_4B, "4B (2 seeds)", "#d1442f")]:
            steps, mean, half, n, _ = aggregate(run_ids, metric)
            lo = np.clip(mean - half, 0, None)
            ax.fill_between(steps, lo, mean + half, alpha=0.16, color=color, lw=0)
            ax.plot(steps, mean, "-o", color=color, ms=3.5, lw=1.9, label=cohort)
        ax.set_title(f"Gemma3-PT few-shot math RL: val math acc {mlabel}\n1B vs 4B (mean ± 95% CI)",
                     fontweight="bold")
        ax.set_xlabel("training step")
        ax.set_ylabel("accuracy")
        ax.legend(frameon=False, fontsize=10, loc="best")
        fig.tight_layout()
        out = f"{PLOTS}/overlay_1b_vs_4b_val_{fname}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print("wrote", out)

    # ---- report step extents ----
    print("\nstep extents:")
    for cohort, runs in [("1B", RUNS_1B), ("4B", RUNS_4B)]:
        for rid, lab in runs.items():
            s = series(rid, "val-core/math/acc/mean@16")
            print(f"  {cohort} {lab} ({rid}): {len(s)} val pts, steps {int(s['step'].min())}-{int(s['step'].max())}")


if __name__ == "__main__":
    main()
