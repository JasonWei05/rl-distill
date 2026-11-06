#!/usr/bin/env python3
"""Per-curve C_mid with the shared PAPER-form A and B fixed (the runbook's "what to look at": does log C_mid against
log pass@16 stay a straight line over the six starts, and how far do the two 0.3% seeds disagree on C_mid).
Same preprocessing as rl_curve_family.py fit --form PAPER: drop steps below 30, R_ref = mean dev over steps 30-50.
    cmid_per_curve.py <data.json ...> --fit rl_curve_family_S0.json --model S0 [--json-out out.json]
"""
import argparse
import json

import numpy as np
from scipy.optimize import least_squares


def load(paths, model):
    starts, curves = {}, {}
    for p in paths:
        d = json.load(open(p))
        starts.update(d["starts"])
        new = {}
        for c in d["curves"]:
            if c["model"] == model:
                new.setdefault((c["pair"], c["seed"]), []).append(c)
        curves.update(new)  # a later file replaces a whole curve, as rl_curve_family.py does
    return starts, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--fit", required=True)
    ap.add_argument("--model", default="S0")
    ap.add_argument("--json-out")
    args = ap.parse_args()
    fit = json.load(open(args.fit))
    th = (fit.get(args.model) or fit).get("params", fit)
    A, B = float(th["A"]), float(th["B"])
    starts, curves = load(args.data, args.model)
    rows = []
    for (pair, seed), pts in sorted(curves.items(), key=lambda kv: -starts[kv[0][0]][args.model]["pass@16"]):
        pts = sorted(pts, key=lambda c: c["step"])
        ref = [c["dev"] for c in pts if 30 <= c["step"] <= 50]
        keep = [c for c in pts if c["step"] >= 30 and c["pflops"] > 0]
        if not ref or len(keep) < 5:
            continue
        r = float(np.mean(ref))
        C = np.array([c["pflops"] for c in keep]); y = np.array([c["dev"] for c in keep])
        f = lambda lc: r + (A - r) / (1 + (10 ** lc[0] / C) ** B) - y
        sol = least_squares(f, x0=[1.0], bounds=([-3], [6]))
        J = sol.jac; cov = np.linalg.pinv(J.T @ J) * (sol.fun @ sol.fun) / max(1, len(y) - 1)
        rows.append(dict(pair=pair, seed=seed, p16=starts[pair][args.model]["pass@16"], R_ref=round(r, 4),
                         Cmid_pf=round(10 ** sol.x[0], 2), log10_Cmid_se=round(float(np.sqrt(cov[0, 0])), 3),
                         rmse=round(float(np.sqrt(np.mean(sol.fun ** 2))), 4), last_pf=round(float(C.max()), 1)))
    x = np.log10([r["p16"] for r in rows]); yv = np.log10([r["Cmid_pf"] for r in rows])
    slope, icpt = np.polyfit(x, yv, 1)
    for r, xi, yi in zip(rows, x, yv):
        r["resid_log10"] = round(float(yi - (icpt + slope * xi)), 3)
    print(f"{args.model}: A {A:.3f}, B {B:.3f} fixed; per-curve C_mid (PF); line log10 Cmid = {icpt:.3f} + {slope:.3f} log10 p16"
          f" (gamma {-slope:.2f})")
    print(f"{'pair':>5} {'seed':>4} {'p16':>6} {'R_ref':>6} {'Cmid':>9} {'se(log10)':>9} {'rmse':>6} {'last PF':>7} {'resid':>6}")
    for r in rows:
        print(f"{r['pair']:>5} {r['seed']:>4} {r['p16']:6.3f} {r['R_ref']:6.3f} {r['Cmid_pf']:9.1f} {r['log10_Cmid_se']:9.3f} "
              f"{r['rmse']:6.3f} {r['last_pf']:7.1f} {r['resid_log10']:+6.3f}")
    if args.json_out:
        json.dump({"A": A, "B": B, "line": {"intercept": icpt, "slope": slope}, "curves": rows}, open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
