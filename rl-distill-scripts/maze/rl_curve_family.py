#!/usr/bin/env python3
"""RL-curve family per model size for the text-maze study: a(C; start) with the start given by the starting
checkpoint's (pass@1, pass@16, pass@64) and C the cumulative RL compute in PFLOPs.

Two subcommands:

  extract   dev curves + start profiles of finished pair RL runs (launch_pair_rl.sh) into one JSON
      ~/TailRL/.venv/bin/python rl-distill-scripts/maze/rl_curve_family.py extract --pairs p3 p03 \
          --runs ~/tailrl_maze_runs --out ~/tailrl_maze_runs/rl_curves_extra.json
      Reads <runs>/rl/<pair>_{S0,L0}_rl.log (dev metric every TEST_FREQ steps, tflops/cumulative_total every step; no
      checkpoint dirs needed, so pruned runs still extract) and <runs>/matched/manifest.json (start profiles at the
      final k of make_pair.py). Replicates: --replicates p10s1=p10:1 maps log tag p10s1 to pair p10, seed 1.

  fit       joint fit of one size's curves over all its starts, with leave-one-start-out validation
      ~/TailRL/.venv/bin/python rl-distill-scripts/maze/rl_curve_family.py fit --model S0 \
          --data rl-distill-scripts/maze/data/rl_curves_v1.json ~/tailrl_maze_runs/rl_curves_extra.json --loo
      Form SIG (default):  a = p1 + (R - p1)(1 - exp(-C/tau)) + (A - R) / (1 + (Cmid/C)^beta),
                           R = p16 + kappa (p64 - p16),  log10 Cmid = m0 + m1 log10 p16        (6 params per size)
      Form LIN:            a = min(A, p1 + (R - p1)(1 - exp(-C/tau)) + s log10(1 + C/(3 tau))),
                           s = smax sigmoid((log10 p16 - x*)/w)                                  (6 params per size)
      Needs scipy (not in the TailRL venv by default: `uv pip install --python ~/TailRL/.venv/bin/python scipy`).

Data JSON: {"starts": {pair: {"S0": {"k", "pass@1", "pass@16", "pass@64", ...}, "L0": {...}}},
            "curves": [{"pair", "model", "seed", "step", "pflops", "dev"}, ...]}
"""
import argparse
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plan_budgets import read_val_curve  # noqa: E402

MODELS = ("S0", "L0")
MANIFEST_ALIAS = {"p10": "primary"}  # the 10% pair is stored under its original name


# ----------------------------------------------------------------------------------------------------- extract
def read_cumulative_tflops(log):
    """{step: cumulative total TFLOPs} from the per-step metric lines of a verl console log."""
    out = {}
    step_re = re.compile(r"step:(\d+) - ")
    cum_re = re.compile(r"tflops/cumulative_total:([0-9.eE+-]+)")
    for line in open(log, errors="replace"):
        m = step_re.search(line)
        c = cum_re.search(line)
        if m and c:
            out[int(m.group(1))] = float(c.group(1))
    return out


def start_profile(manifest, pair, model):
    entry = manifest["pairs"].get(pair) or manifest["pairs"].get(MANIFEST_ALIAS.get(pair, ""))
    if entry is None:
        raise SystemExit(f"pair {pair} not in the matched manifest (keys: {list(manifest['pairs'])})")
    side = entry[model]
    mk = [k for k in side if k.startswith("metrics_k")]
    if not mk:
        raise SystemExit(f"{pair} {model}: no metrics_k* block in the manifest")
    g = side[mk[0]]["goal_reached"]
    prof = {"k": int(mk[0].split("_k")[1]), "candidate": side.get("candidate"), "run": side.get("run")}
    prof.update({k: g[k] for k in g if k.startswith("pass@") or k == "mixed16"})
    return prof


def cmd_extract(args):
    runs = os.path.expanduser(args.runs)
    manifest = json.load(open(os.path.join(runs, "matched", "manifest.json")))
    key = f"val-aux/maze_17/goal_reached/mean@{args.val_n}"
    jobs = [(p, p, 0) for p in args.pairs]
    for spec in args.replicates:
        tag, rest = spec.split("=")
        pair, seed = rest.split(":")
        jobs.append((tag, pair, int(seed)))
    starts, curves = {}, []
    for tag, pair, seed in jobs:
        if pair not in starts:
            starts[pair] = {m: start_profile(manifest, pair, m) for m in MODELS}
        for model in MODELS:
            log = os.path.join(runs, "rl", f"{tag}_{model}_rl.log")
            if not os.path.exists(log):
                print(f"[skip] {log} missing")
                continue
            cum = read_cumulative_tflops(log)
            vals = read_val_curve(log, key)
            n = 0
            for step, dev in vals:
                if step == 0:
                    pf = 0.0
                elif step in cum:
                    pf = cum[step] / 1000.0
                else:
                    continue
                curves.append(dict(pair=pair, model=model, seed=seed, step=step, pflops=round(pf, 4), dev=round(dev, 4)))
                n += 1
            last = max((s for s, _ in vals), default=None)
            print(f"{tag} {model} seed {seed}: {n} dev points (last step {last}), start p1 {starts[pair][model]['pass@1']:.4f} "
                  f"p16 {starts[pair][model]['pass@16']:.3f} p64 {starts[pair][model]['pass@64']:.3f}")
    out = {"note": f"extracted by rl_curve_family.py from {runs}", "val_key": key, "starts": starts, "curves": curves}
    with open(os.path.expanduser(args.out), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {args.out}: {len(curves)} curve points, starts for {sorted(starts)}")


# --------------------------------------------------------------------------------------------------------- fit
def load_data(paths):
    starts, curves = {}, []
    for p in paths:
        d = json.load(open(os.path.expanduser(p)))
        for pair, v in d["starts"].items():
            starts.setdefault(pair, {}).update(v)
        curves.extend(d["curves"])
    return starts, curves


def model_fn(form, th, C, start):
    import numpy as np
    p1, p16, p64 = start
    C = np.asarray(C, float)
    if form == "SIG":
        kappa, tau, A, beta, m0, m1 = th
        r = p16 + kappa * (p64 - p16)
        fast = p1 + (r - p1) * (1 - np.exp(-C / tau))
        cmid = 10 ** (m0 + m1 * np.log10(p16))
        return fast + np.maximum(A - r, 0) / (1 + (cmid / C) ** beta)
    kappa, tau, A, smax, xstar, w = th
    r = p16 + kappa * (p64 - p16)
    fast = p1 + (r - p1) * (1 - np.exp(-C / tau))
    s = smax / (1 + np.exp(-(np.log10(p16) - xstar) / w))
    return np.minimum(A, fast + s * np.log10(1 + C / (3 * tau)))


BOUNDS = {"SIG": ([0, 0.02, 0.3, 0.2, -2, -4], [1.5, 30, 1.0, 5, 5, 1]),
          "LIN": ([0, 0.02, 0.3, 0, -3, 0.1], [1.5, 30, 1.0, 1.5, 0.5, 2])}
X0 = {"SIG": [0.5, 1.0, 0.9, 1.0, 1.0, -0.8], "LIN": [0.5, 1.0, 0.9, 0.3, -1.5, 0.4]}
NAMES = {"SIG": ["kappa", "tau_pf", "A", "beta", "m0", "m1"], "LIN": ["kappa", "tau_pf", "A", "smax", "xstar", "w"]}


def fit_family(form, keys, curves, starts, model, restarts=12, seed=0):
    import numpy as np
    from scipy.optimize import least_squares

    rng = np.random.default_rng(seed)
    series = {}
    for k in keys:
        pts = sorted((c["pflops"], c["dev"]) for c in curves[k] if c["pflops"] > 0)
        series[k] = (np.array([a for a, _ in pts]), np.array([b for _, b in pts]))

    def resid(th):
        r = []
        for k in keys:
            x, y = series[k]
            st = starts[k[0]][model]
            r.append((model_fn(form, th, x, (st["pass@1"], st["pass@16"], st["pass@64"])) - y) / math.sqrt(len(x)))
        return np.concatenate(r)

    best = None
    inits = [X0[form]] + [[rng.uniform(lo, hi) for lo, hi in zip(*BOUNDS[form])] for _ in range(restarts)]
    for x0 in inits:
        try:
            sol = least_squares(resid, x0, bounds=BOUNDS[form], max_nfev=4000)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    return best.x


def curve_rmse(form, th, x, y, start):
    import numpy as np
    return float(math.sqrt(np.mean((model_fn(form, th, x, start) - y) ** 2)))


def cmd_fit(args):
    import numpy as np

    starts, rows = load_data(args.data)
    order = sorted(starts, key=lambda p: -starts[p]["S0"]["pass@1"])  # easiest start first
    results = {}
    for model in args.model:
        curves = {}
        for c in rows:
            if c["model"] == model:
                curves.setdefault((c["pair"], c["seed"]), []).append(c)
        keys = sorted(curves, key=lambda k: (order.index(k[0]), k[1]))
        pairs = sorted({k[0] for k in keys}, key=order.index)
        print(f"\n==== {model}: {len(keys)} curves over starts {pairs} (form {args.form})")
        th = fit_family(args.form, keys, curves, starts, model, args.restarts)
        print("params: " + ", ".join(f"{n} {v:.3f}" for n, v in zip(NAMES[args.form], th)))
        res = {"params": dict(zip(NAMES[args.form], map(float, th))), "in_sample_rmse": {}, "per_start": {}, "loo": {}}
        for k in keys:
            x = np.array([c["pflops"] for c in curves[k] if c["pflops"] > 0]); y = np.array([c["dev"] for c in curves[k] if c["pflops"] > 0])
            st = starts[k[0]][model]
            r = curve_rmse(args.form, th, x, y, (st["pass@1"], st["pass@16"], st["pass@64"]))
            res["in_sample_rmse"][f"{k[0]}:s{k[1]}"] = r
        print("in-sample RMSE: " + "  ".join(f"{k} {v:.3f}" for k, v in res["in_sample_rmse"].items()))
        for p in pairs:
            st = starts[p][model]; p16, p64 = st["pass@16"], st["pass@64"]
            if args.form == "SIG":
                d = {"R_ref": p16 + th[0] * (p64 - p16), "Cmid_pf": 10 ** (th[4] + th[5] * np.log10(p16))}
            else:
                d = {"R_ref": p16 + th[0] * (p64 - p16), "slope_per_decade": th[3] / (1 + np.exp(-(np.log10(p16) - th[4]) / th[5]))}
            res["per_start"][p] = {k: float(v) for k, v in d.items()}
            print(f"  {p}: " + ", ".join(f"{k} {v:.3f}" for k, v in d.items()))
        if args.loo and len(pairs) >= 3:
            print("leave-one-start-out (fit on the other starts, predict the held-out seed-0 curve):")
            for held in pairs:
                tr = [k for k in keys if k[0] != held]
                th2 = fit_family(args.form, tr, curves, starts, model, args.restarts)
                k = (held, 0)
                if k not in curves:
                    continue
                x = np.array([c["pflops"] for c in curves[k] if c["pflops"] > 0]); y = np.array([c["dev"] for c in curves[k] if c["pflops"] > 0])
                st = starts[held][model]
                r = curve_rmse(args.form, th2, x, y, (st["pass@1"], st["pass@16"], st["pass@64"]))
                res["loo"][held] = {"rmse": r, "params": dict(zip(NAMES[args.form], map(float, th2)))}
                print(f"  held out {held}: RMSE {r:.3f}")
        results[model] = res
    if args.json_out:
        with open(os.path.expanduser(args.json_out), "w") as f:
            json.dump(results, f, indent=1)
        print("wrote", args.json_out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--pairs", nargs="+", required=True, help="pair tags whose <pair>_{S0,L0}_rl.log exist (seed 0)")
    e.add_argument("--replicates", nargs="*", default=[], help="extra logs: <logtag>=<pair>:<seed>, e.g. p10s1=p10:1")
    e.add_argument("--runs", default="~/tailrl_maze_runs")
    e.add_argument("--val-n", type=int, default=16, help="VAL_N of the runs (the dev metric is mean@VAL_N)")
    e.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--data", nargs="+", required=True)
    f.add_argument("--model", nargs="+", default=["S0", "L0"], choices=MODELS)
    f.add_argument("--form", default="SIG", choices=["SIG", "LIN"])
    f.add_argument("--loo", action="store_true")
    f.add_argument("--restarts", type=int, default=12)
    f.add_argument("--json-out", default=None)
    args = ap.parse_args()
    {"extract": cmd_extract, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
