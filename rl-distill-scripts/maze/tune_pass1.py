#!/usr/bin/env python3
"""Hit a target goal_reached (or --metric is_shortest) pass@1 by weight interpolation between two checkpoints of one training run (text-maze study).

theta(alpha) = (1 - alpha) theta_A + alpha theta_B with A below and B above the target, alpha first from log-linear
interpolation of the two endpoints' pass@1, then secant steps on (alpha, log pass@1). The search screens the dev mazes at
--search-k samples per maze (default --k) and stops when |pass@1 - target| <= --stop-sigma x its sampling SE or after
--max-iters. With --search-k < --k, the closest model is re-measured at --k and corrected by Newton steps on log pass@1
(slope from the search points, then from the --k points) until it is within --tol (relative) or --stop-sigma SE of the
target, for at most --fine-iters steps. The model closest to the target at --k is then re-screened with a fresh sampling
seed (--confirm-seed), and that independent measurement is what gets reported.
With --fine-mode confirm the --k correction stage is skipped: the closest model (a cached --k correction point if any, else the
closest search point) is confirmed directly at --k with a fresh seed; only when that confirmation misses the target by more
than --tol (or --stop-sigma SE) is a Newton step taken from it and the new model confirmed with the next fresh seed, for at
most --fine-iters steps. The last confirmation is what gets reported (one 2048-sample run per level instead of >= 2 when the
search lands close). Endpoint pass@1 values can be passed in (--pa / --pb) when already measured
at --search-k on these mazes; otherwise they are screened first. Writes <out>/tune.json (every screen with per-maze counts) and
keeps the chosen model at <out>/best.

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/tune_pass1.py --a <run>/avg-5480-w60 --b <run>/avg-5500-w60 \
        --target 0.01 --out <run>/tune-p1
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import screen_sft_ckpts as ss  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="checkpoint below the target")
    ap.add_argument("--b", required=True, help="checkpoint above the target")
    ap.add_argument("--pa", type=float, default=None)
    ap.add_argument("--pb", type=float, default=None)
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--search-k", type=int, default=None, help="samples per maze for the search (default --k)")
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--fine-iters", type=int, default=3, help="Newton corrections at --k after a cheaper search")
    ap.add_argument("--tol", type=float, default=0.01, help="relative tolerance of the --k correction stage")
    ap.add_argument("--fine-mode", default="fine", choices=["fine", "confirm"],
                    help="fine: Newton corrections at --k, then a fresh-seed confirm; confirm: correct from fresh-seed confirms only")
    ap.add_argument("--stop-sigma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confirm-seed", type=int, default=1)
    ap.add_argument("--tailrl-root", default=ss.DEFAULT_TAILRL)
    ap.add_argument("--metric", default="goal_reached", choices=["goal_reached", "is_shortest"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "tune.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else None
    if log is not None and (log["a"], log["b"]) != (args.a, args.b):   # new endpoints: the cached screens do not apply
        os.replace(log_path, log_path + ".prev")
        for d in os.listdir(args.out):
            if d.startswith("interp-a"):
                shutil.rmtree(os.path.join(args.out, d))
        log = None
    log = log or {"target": args.target, "a": args.a, "b": args.b, "screens": {}}

    judge = ss.load_judge(args.tailrl_root)
    prompts, gts = ss.load_prompts(ss.dev_parquet(args.tailrl_root), args.n_prompts)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sk = args.search_k or args.k

    def screen(path, key, seed, k):
        if key in log["screens"]:
            return log["screens"][key]
        torch.manual_seed(seed)
        r = ss.screen_one(path, prompts, gts, judge, SimpleNamespace(k=k, batch_prompts=128, max_new_tokens=ss.max_new_tokens()), device)
        r["seed"] = seed
        log["screens"][key] = r
        with open(log_path + ".tmp", "w") as f:
            json.dump(log, f)
        os.replace(log_path + ".tmp", log_path)
        g = r[args.metric]
        print(f"{key:>22} pass@1 {g['pass@1']:.6f} ({100 * (g['pass@1'] - args.target) / args.target:+.2f}% vs target) "
              f"pass@16 {g.get('pass@16', float('nan')):.4f} ({r['seconds']}s)", flush=True)
        return r

    def p1(r):
        return r[args.metric]["pass@1"]

    def se(p, k):  # sampling SE of pass@1 on this maze set, binomial over all samples
        return math.sqrt(max(p, 1e-9) * (1 - p) / (k * len(prompts)))

    def log_slope(q1, q2):  # d log pass@1 / d alpha through two points
        return (math.log(max(q2[1], 1e-9)) - math.log(max(q1[1], 1e-9))) / (q2[0] - q1[0])

    def interp(alpha):
        path = os.path.join(args.out, f"interp-a{alpha:.4f}")
        if not os.path.isfile(os.path.join(path, "model.safetensors")):
            subprocess.run([sys.executable, os.path.join(HERE, "interpolate_ckpts.py"), "--a", args.a, "--b", args.b,
                            "--alpha", f"{alpha:.4f}", "--out", path], check=True, stdout=subprocess.DEVNULL)
        return path

    pts = []  # (alpha, pass@1, path)
    pa = args.pa if args.pa is not None else p1(screen(args.a, "endpoint-a", args.seed, sk))
    pb = args.pb if args.pb is not None else p1(screen(args.b, "endpoint-b", args.seed, sk))
    pts += [(0.0, pa, args.a), (1.0, pb, args.b)]
    if not pa < args.target < pb:
        print(f"WARNING: endpoints do not bracket the target ({pa:.6f}, {pb:.6f} vs {args.target}); extrapolating", flush=True)
    t = math.log(args.target)
    for it in range(args.max_iters):
        best = min(pts, key=lambda q: abs(q[1] - args.target))
        if abs(best[1] - args.target) <= args.stop_sigma * se(args.target, sk) and it > 0:
            break
        # secant on the two points nearest the target in log pass@1, one on each side when available
        lo = max((q for q in pts if q[1] <= args.target), key=lambda q: q[1], default=None)
        hi = min((q for q in pts if q[1] > args.target), key=lambda q: q[1], default=None)
        q1, q2 = (lo, hi) if lo and hi else sorted(pts, key=lambda q: abs(q[1] - args.target))[:2]
        x1, y1, x2, y2 = q1[0], math.log(max(q1[1], 1e-9)), q2[0], math.log(max(q2[1], 1e-9))
        alpha = x1 + (t - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else 0.5 * (x1 + x2)
        alpha = min(max(alpha, -0.5), 1.5)
        path = interp(alpha)
        pts.append((alpha, p1(screen(path, f"alpha-{alpha:.4f}", args.seed, sk)), path))

    alpha, p, path = min(pts, key=lambda q: abs(q[1] - args.target))
    if sk < args.k and args.fine_mode == "fine":
        lo = max((q for q in pts if q[1] <= args.target), key=lambda q: q[1], default=None)
        hi = min((q for q in pts if q[1] > args.target), key=lambda q: q[1], default=None)
        slope = log_slope(lo, hi) if lo and hi else log_slope(*sorted(pts, key=lambda q: abs(q[1] - args.target))[:2])
        tol = max(args.tol * args.target, args.stop_sigma * se(args.target, args.k))
        fine = []
        for it in range(args.fine_iters + 1):
            fine.append((alpha, p1(screen(path, f"fine-a{alpha:.4f}-k{args.k}", args.seed, args.k)), path))
            if abs(fine[-1][1] - args.target) <= tol or it == args.fine_iters:
                break
            if len(fine) >= 2 and fine[-1][1] != fine[-2][1] and log_slope(fine[-2], fine[-1]) > 0:
                slope = log_slope(fine[-2], fine[-1])
            if slope <= 0:   # noisy search points: fall back to the endpoints
                slope = log_slope((0.0, pa), (1.0, pb))
            alpha = min(max(alpha + (t - math.log(max(fine[-1][1], 1e-9))) / slope, -0.5), 1.5)
            path = interp(alpha)
        alpha, p, path = min(fine, key=lambda q: abs(q[1] - args.target))
    if args.fine_mode == "fine" or sk >= args.k:
        conf = screen(path, f"confirm-a{alpha:.4f}-seed{args.confirm_seed}", args.confirm_seed, args.k)
    else:
        cached = [(float(key.split("-a")[1].split("-k")[0]), r[args.metric]["pass@1"]) for key, r in log["screens"].items()
                  if key.startswith("fine-a") and key.endswith(f"-k{args.k}")]
        if cached:   # --k points from an earlier fine-mode attempt are more precise than the search points
            alpha, p = min(cached, key=lambda q: abs(q[1] - args.target))
            path = interp(alpha) if alpha not in (0.0, 1.0) else (args.a if alpha == 0.0 else args.b)
        lo = max((q for q in pts if q[1] <= args.target), key=lambda q: q[1], default=None)
        hi = min((q for q in pts if q[1] > args.target), key=lambda q: q[1], default=None)
        slope = log_slope(lo, hi) if lo and hi else log_slope(*sorted(pts, key=lambda q: abs(q[1] - args.target))[:2])
        if slope <= 0:
            slope = log_slope((0.0, pa), (1.0, pb))
        tol = max(args.tol * args.target, args.stop_sigma * se(args.target, args.k))
        confs = []
        for it in range(args.fine_iters + 1):
            seed = args.confirm_seed + it
            conf = screen(path, f"confirm-a{alpha:.4f}-seed{seed}", seed, args.k)
            confs.append((alpha, p1(conf)))
            if abs(p1(conf) - args.target) <= tol or it == args.fine_iters:
                break
            if len(confs) >= 2 and confs[-1][1] != confs[-2][1] and log_slope(confs[-2], confs[-1]) > 0:
                slope = log_slope(confs[-2], confs[-1])
            alpha = min(max(alpha + (t - math.log(max(p1(conf), 1e-9))) / slope, -0.5), 1.5)
            path = interp(alpha)
    best_dir = os.path.join(args.out, "best")
    if os.path.lexists(best_dir):
        shutil.rmtree(best_dir) if os.path.isdir(best_dir) and not os.path.islink(best_dir) else os.remove(best_dir)
    shutil.copytree(path, best_dir)
    log["best"] = {"alpha": alpha, "path": path, "pass1_search": p, "pass1_confirm": p1(conf),
                   "confirm_rel_vs_target": (p1(conf) - args.target) / args.target, "se_rel": se(args.target, args.k) / args.target}
    with open(log_path, "w") as f:
        json.dump(log, f)
    print(f"BEST alpha {alpha:.4f}: search pass@1 {p:.6f}, confirm (seed {conf['seed']}) {p1(conf):.6f} "
          f"({100 * (p1(conf) - args.target) / args.target:+.2f}% vs target, 1 sigma {100 * se(args.target, args.k) / args.target:.2f}%)")


if __name__ == "__main__":
    main()
