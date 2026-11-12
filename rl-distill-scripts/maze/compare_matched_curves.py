#!/usr/bin/env python3
"""Compare the goal_reached pass@k curves of models matched at the same starting level (text-maze study).

Reads screen_sft_ckpts.py outputs (per-maze success counts out of k samples on the same dev mazes) and, per level,
compares every model against the level's first (reference) model:
  pass@k for k = 1, 2, 4, ..., up to the samples per maze (unbiased estimator, Chen et al. 2021)
  delta = model - reference with a paired bootstrap 95% CI over mazes
  W1    Wasserstein-1 distance between the per-maze success-rate distributions
  mixed16, the share of 16-rollout groups with a nonzero GRPO advantage
and flags each delta against the tolerance |delta| <= --tol-rel * min(p, 1 - p) (p = the two models' mean), every k <= --max-k.

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/compare_matched_curves.py --screens <screen.json> ... \
        --level p1=medium_p1,large_p1 --level p01=medium_p01,large_p01 --out <compare.json>
"""

import argparse
import json

import numpy as np
from scipy.special import gammaln
from scipy.stats import wasserstein_distance


def pass_at_k(counts, n, k):
    """Unbiased pass@k per maze: 1 - C(n - c, k) / C(n, k), vectorized."""
    c = np.asarray(counts, dtype=np.float64)
    out = np.ones_like(c)
    ok = n - c >= k
    out[ok] = 1.0 - np.exp(gammaln(n - c[ok] + 1) - gammaln(n - c[ok] - k + 1) - gammaln(n + 1) + gammaln(n - k + 1))
    return out


def mixed16(counts, n):
    p = np.asarray(counts, dtype=np.float64) / n
    return float(np.mean(1.0 - p**16 - (1.0 - p) ** 16))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screens", nargs="+", required=True, help="screen_sft_ckpts.py output JSONs (labels must be unique)")
    ap.add_argument("--level", action="append", required=True, help="<name>=<ref label>,<label>,... (first = reference)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tol-rel", type=float, default=0.02, help="relative to min(p, 1 - p): success rate when sparse, miss rate near 1")
    ap.add_argument("--max-k", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    screens = {}
    for path in args.screens:
        screens.update(json.load(open(path)))
    rng = np.random.default_rng(0)
    report = {"tolerance": {"rel_of_min_p_1mp": args.tol_rel, "max_k": args.max_k}, "levels": {}}

    for spec in args.level:
        name, labels = spec.split("=")
        labels = labels.split(",")
        runs = {lab: screens[lab] for lab in labels}
        n = min(r["k"] for r in runs.values())
        assert all(r["k"] == n for r in runs.values()), f"{name}: models screened at different samples per maze"
        counts = {lab: np.asarray(r["per_maze"]["goal_reached"]) for lab, r in runs.items()}
        n_mazes = len(next(iter(counts.values())))
        ks = [2**i for i in range(int(np.log2(n)) + 1)]
        boot_idx = rng.integers(0, n_mazes, size=(args.n_boot, n_mazes))
        ref = labels[0]
        per_k_ref = {k: pass_at_k(counts[ref], n, k) for k in ks}
        level = {"samples_per_maze": n, "mazes": n_mazes, "reference": ref, "models": {}}
        for lab in labels:
            m = {"ckpt": runs[lab]["ckpt"], "mixed16": mixed16(counts[lab], n), "pass": {}, "delta": {}}
            for k in ks:
                pk = pass_at_k(counts[lab], n, k)
                m["pass"][k] = float(pk.mean())
                if lab != ref:
                    d = pk - per_k_ref[k]
                    lo, hi = np.percentile(d[boot_idx].mean(axis=1), [2.5, 97.5])
                    pbar = 0.5 * (float(per_k_ref[k].mean()) + float(pk.mean()))
                    scale = max(min(pbar, max(1.0 - pbar, 1.0 / n_mazes)), 1e-12)  # miss side floored at one maze
                    within = abs(d.mean()) <= args.tol_rel * scale
                    m["delta"][k] = {"delta": float(d.mean()), "ci95": [float(lo), float(hi)],
                                     "rel": float(d.mean() / scale), "within_tol": bool(within)}
            if lab != ref:
                m["w1_success_rate"] = float(wasserstein_distance(counts[lab] / n, counts[ref] / n))
                m["all_within_tol"] = all(v["within_tol"] for k, v in m["delta"].items() if k <= args.max_k)
            level["models"][lab] = m
        report["levels"][name] = level

        print(f"\n== {name}: {n_mazes} mazes x {n} samples, reference {ref}")
        print(f"{'k':>6} " + " ".join(f"{lab:>14}" for lab in labels) + "   " +
              " ".join(f"{'d(' + lab + ')':>30}" for lab in labels[1:]))
        for k in ks:
            row = f"{k:>6} " + " ".join(f"{level['models'][lab]['pass'][k]:14.5f}" for lab in labels) + "   "
            for lab in labels[1:]:
                v = level["models"][lab]["delta"][k]
                flag = "" if (k > args.max_k or v["within_tol"]) else " !"
                row += f"{v['delta']:+10.5f} [{v['ci95'][0]:+.5f},{v['ci95'][1]:+.5f}]{flag:2}"
            print(row)
        print("mixed16 " + " ".join(f"{level['models'][lab]['mixed16']:.5f}" for lab in labels) +
              "   W1 " + " ".join(f"{level['models'][lab]['w1_success_rate']:.5f}" for lab in labels[1:]) +
              "   all within tol: " + " ".join(str(level["models"][lab]["all_within_tol"]) for lab in labels[1:]))

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
