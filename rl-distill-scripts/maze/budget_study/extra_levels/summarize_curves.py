#!/usr/bin/env python3
"""Per-curve end points of rl_curve_family.py extract output (README tables), plus dev at fixed PF budgets.
    summarize_curves.py <rl_curves.json ...> [--model L0] [--budgets 25 50 100 200 250]
A later file replaces a whole (pair, model, seed) curve, as rl_curve_family.py fit does."""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--model", nargs="+", default=["S0", "L0"])
    ap.add_argument("--budgets", type=float, nargs="+", default=[25, 50, 100, 200, 250])
    args = ap.parse_args()
    curves, starts = {}, {}
    for p in args.data:
        d = json.load(open(p))
        starts.update(d["starts"])
        new = {}
        for c in d["curves"]:
            new.setdefault((c["pair"], c["model"], c["seed"]), []).append(c)
        curves.update(new)
    for model in args.model:
        keys = sorted((k for k in curves if k[1] == model), key=lambda k: (-starts[k[0]][model]["pass@16"], k[2]))
        print(f"== {model}: pair seed | last step  PF  final dev (mean of last 3) | dev at " + " ".join(f"{b:g}" for b in args.budgets) + " PF")
        for k in keys:
            pts = sorted(curves[k], key=lambda c: (c["step"], c["pflops"]))
            pf = np.array([c["pflops"] for c in pts]); dev = np.array([c["dev"] for c in pts])
            at = [f"{np.interp(b, pf, dev):.3f}" if b <= pf.max() else "  -  " for b in args.budgets]
            print(f"{k[0]:>4} {k[2]} | {pts[-1]['step']:5d} {pf.max():6.1f} {dev[-1]:.3f} ({dev[-3:].mean():.3f}) | " + " ".join(at))


if __name__ == "__main__":
    main()
