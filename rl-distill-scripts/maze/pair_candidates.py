#!/usr/bin/env python3
"""Pair S and L candidates with the same starting capability (text-maze study, step 3).

Reads <run>/select_<metric>.json from every run, takes the candidates measured at --k samples per
maze (the k=1024 finalists by default), and for every (S candidate, L candidate) pair reports
  d_p1      |pass@1_S - pass@1_L| on the matching metric
  d_curve   max over k in {1,4,16,64,256,512,1024} of |pass@k_S - pass@k_L| (goal_reached)
  d_short   |shortest-path pass@1_S - pass@1_L|
  W1        Wasserstein-1 distance between the per-maze success-rate histograms (goal_reached)
  mixed16   fraction of mazes whose 16-rollout group would be mixed, for each side
Pairs are ranked by d_p1, then W1. The matching rule is d_p1 <= --tol (0.002 = 0.2 points).

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/pair_candidates.py --runs-root ~/tailrl_maze_runs/pretrain20k --k 1024
"""

import argparse
import glob
import json
import os

import numpy as np


def load_candidates(runs_root, metric, k):
    cands = []
    for p in sorted(glob.glob(os.path.join(runs_root, "*_ds0_*", f"select_{metric}.json"))):
        run = os.path.basename(os.path.dirname(p))
        d = json.load(open(p))
        for key, r in d.items():
            if key.startswith("_") or r.get("k") != k:
                continue
            cands.append((run, key, r))
    return cands


def w1(a, b):
    """Wasserstein-1 between two empirical distributions on [0, 1] (per-maze success rates)."""
    a, b = np.sort(a), np.sort(b)
    q = np.linspace(0, 1, 1001)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=os.path.expanduser("~/tailrl_maze_runs/pretrain20k"))
    ap.add_argument("--metric", default="goal_reached")
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--tol", type=float, default=0.002)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    cands = load_candidates(args.runs_root, args.metric, args.k)
    S = [c for c in cands if c[0].startswith("S_")]
    L = [c for c in cands if c[0].startswith("L_")]
    print(f"{len(S)} S candidates, {len(L)} L candidates at k={args.k}\n")
    print(f"{'candidate':>34} {'goal p@1':>9} {'short p@1':>9} {'goal p@16':>9} {'goal p@64':>9} {'short p@256':>11} {'goal mix16':>10}")
    for run, key, r in S + L:
        g, s = r["goal_reached"], r["is_shortest"]
        print(f"{run + '/' + key.split('@')[0]:>34} {g['pass@1']:9.4f} {s['pass@1']:9.4f} {g['pass@16']:9.4f} {g['pass@64']:9.4f} {s.get('pass@256', float('nan')):11.4f} {g['mixed16']:10.3f}")

    ks = [k for k in (1, 4, 16, 64, 256, 1024) if k <= args.k]
    rows = []
    for rs, ks_, r_s in S:
        for rl, kl, r_l in L:
            gs, gl = r_s["goal_reached"], r_l["goal_reached"]
            d_p1 = abs(gs["pass@1"] - gl["pass@1"])
            d_curve = max(abs(gs[f"pass@{k}"] - gl[f"pass@{k}"]) for k in ks)
            d_short = abs(r_s["is_shortest"]["pass@1"] - r_l["is_shortest"]["pass@1"])
            ps = np.array(r_s["per_maze"]["goal_reached"]) / args.k
            pl = np.array(r_l["per_maze"]["goal_reached"]) / args.k
            rows.append((d_p1, w1(ps, pl), d_curve, d_short, gs["mixed16"], gl["mixed16"], f"{rs}/{ks_.split('@')[0]}", f"{rl}/{kl.split('@')[0]}"))
    rows.sort()
    print(f"\nbest pairs (rule: d_p1 <= {args.tol}):")
    print(f"{'S candidate':>30} {'L candidate':>30} {'d_p1':>7} {'W1':>7} {'d_curve':>8} {'d_short':>8} {'mix16 S/L':>12} ok")
    for d_p1, W, d_curve, d_short, ms, ml, s, l in rows[: args.top]:
        print(f"{s:>30} {l:>30} {d_p1:7.4f} {W:7.4f} {d_curve:8.4f} {d_short:8.4f} {ms:5.3f}/{ml:5.3f} {'yes' if d_p1 <= args.tol else 'no'}")


if __name__ == "__main__":
    main()
