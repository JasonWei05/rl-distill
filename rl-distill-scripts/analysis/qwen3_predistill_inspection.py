#!/usr/bin/env python3
"""Inspection report for the base->base on-policy pre-distillations (pre-training vs RL ablation, PROGRESS_LOG 2026-10-25).

For each direction (1.7B base -> 4B base = 4B', 4B base -> 1.7B base = 1.7B') and band, compares the student by step (every
10 = every HF export) with the SOURCE base it is meant to imitate and the base it started from, on the band's val300 x16:
mean@16, best@16 (verl bootstrap, ~pass@10), response length, training NLL/token (entropy estimate of its own samples) and
k1 to the teacher. References come from the same runs: step 0 of a direction is its student base; the source base's NLL is the
opposite direction's steps 1-5 (where that base is the student). Writes a markdown table and a figure.
  python rl-distill-scripts/analysis/qwen3_predistill_inspection.py
"""
import json, os

import numpy as np
import wandb

V, BEST, LEN = "val-core/math/acc/mean@16", "val-core/math/acc/best@16/mean", "val-aux/math/response_length/mean@16"
NLL, K1 = "rollout_corr/training_log_ppl", "actor/distillation/k1"
BANDS = ["easy", "medium", "hard"]
DIRS = {"4B' (4B base <- 1.7B base)": ("qb1p7b{b}-to-q4b-base", "qb4b{b}-to-q1p7b-base"),
        "1.7B' (1.7B base <- 4B base)": ("qb4b{b}-to-q1p7b-base", "qb1p7b{b}-to-q4b-base")}
OUT = "rl-distill-scripts/figures/qwen3_predistill_inspection"

api = wandb.Api(timeout=120)
cache = {}


def run(spec):
    if spec not in cache:
        r = api.run(f"rl-distill/DAPO/q3-opd-{spec}-pgk1-pre-v1-s42")
        val = r.history(keys=[V, BEST, LEN], samples=5000, pandas=True).sort_values("_step")
        train = r.history(keys=[NLL, K1], samples=5000, pandas=True).sort_values("_step")
        cache[spec] = (r.state, val, train)
    return cache[spec]


rows, data = [], {}
for direction, (spec_t, other_t) in DIRS.items():
    for band in BANDS:
        b = band[0]
        state, val, train = run(spec_t.format(b=b))
        _, oval, otrain = run(other_t.format(b=b))
        src = dict(mean=100 * oval[V].iloc[0], best=100 * oval[BEST].iloc[0], len=oval[LEN].iloc[0],
                   nll=otrain[otrain._step <= 5][NLL].mean())
        start = dict(mean=100 * val[V].iloc[0], best=100 * val[BEST].iloc[0], len=val[LEN].iloc[0],
                     nll=train[train._step <= 5][NLL].mean())
        steps = [int(s) for s in val._step if s % 10 == 0]
        table = []
        for s in steps:
            vr = val[val._step == s].iloc[0]
            tr = train[(train._step > s - 5) & (train._step <= s + 5)]   # NLL / k1 averaged over a +-5-step window
            table.append(dict(step=s, mean=100 * vr[V], best=100 * vr[BEST], len=vr[LEN],
                              nll=tr[NLL].mean() if len(tr) else np.nan, k1=tr[K1].mean() if len(tr) else np.nan))
        # closeness to the source base: |d mean@16| in points + |d NLL| scaled so 0.05 nats ~ 1 point (s >= 50 only)
        cand = [t for t in table if t["step"] >= 50 and not np.isnan(t["nll"])]
        closest = min(cand, key=lambda t: abs(t["mean"] - src["mean"]) + 20 * abs(t["nll"] - src["nll"])) if cand else None
        data[(direction, band)] = dict(state=state, src=src, start=start, table=table, closest=closest)

lines = ["# Pre-distillation inspection (base -> base on-policy distillation)", "",
         "Per band: the student by step vs the SOURCE base it imitates (target) and the base it started from. mean@16 / best@16 on",
         "the band's val300 x16 (best@16 = verl bootstrap, ~pass@10); len = mean response tokens; NLL = training NLL/token of its own",
         "samples (entropy estimate, +-5-step window); k1 = mean per-token log-ratio to the teacher on its samples.",
         "`closest` = step >= 50 minimizing |d mean@16| + 20 |d NLL| to the source base (a suggestion, not a choice).", ""]
for (direction, band), d in data.items():
    s, st = d["src"], d["start"]
    lines += [f"## {direction}, {band} ({d['state']}, last step {d['table'][-1]['step']})", "",
              "| step | mean@16 | best@16 | len | NLL/tok | k1 |", "|---|---|---|---|---|---|",
              f"| **target: source base** | **{s['mean']:.1f}** | **{s['best']:.1f}** | **{s['len']:.0f}** | **{s['nll']:.3f}** | 0 |",
              f"| start: student base | {st['mean']:.1f} | {st['best']:.1f} | {st['len']:.0f} | {st['nll']:.3f} | |"]
    for t in d["table"]:
        if t["step"] == 0 or t["step"] % 50 and t is not d["closest"]:
            continue
        mark = " (closest)" if t is d["closest"] else ""
        lines.append(f"| {t['step']}{mark} | {t['mean']:.1f} | {t['best']:.1f} | {t['len']:.0f} | {t['nll']:.3f} | {t['k1']:.3f} |")
    lines.append("")
open(OUT + ".md", "w").write("\n".join(lines) + "\n")
json.dump({f"{k[0]}|{k[1]}": v for k, v in data.items()}, open(OUT + ".json", "w"), indent=1, default=float)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, direction in enumerate(DIRS):
    for j, band in enumerate(BANDS):
        d = data[(direction, band)]; ax = axes[i][j]; ax2 = ax.twinx()
        st = [t["step"] for t in d["table"]]
        ax.plot(st, [t["mean"] for t in d["table"]], color="tab:blue", label="student mean@16")
        ax.axhline(d["src"]["mean"], color="tab:blue", ls="--", label="source base mean@16 (target)")
        ax.axhline(d["start"]["mean"], color="tab:blue", ls=":", alpha=0.6, label="start base mean@16")
        ax2.plot(st, [t["nll"] for t in d["table"]], color="tab:orange", label="student NLL/tok")
        ax2.axhline(d["src"]["nll"], color="tab:orange", ls="--", label="source base NLL (target)")
        ax.set_title(f"{direction}, {band}", fontsize=10); ax.set_xlabel("pre-distillation step")
        ax.set_ylabel("val mean@16 (%)", color="tab:blue"); ax2.set_ylabel("NLL/token", color="tab:orange")
        if i == 0 and j == 0:
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")
fig.tight_layout(); fig.savefig(OUT + ".png", dpi=130)
print("wrote", OUT + ".md", OUT + ".png")
