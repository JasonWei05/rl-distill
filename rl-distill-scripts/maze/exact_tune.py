#!/usr/bin/env python3
"""Tune one pretraining run to a shortest-path pass@1 level with exact (sampling-free) pass@1 (exact_pass.py).

Each candidate is scored exactly in seconds, so there is no sampling noise to average over:
  1. the run's stride screens (k=16) give the approximate crossing step;
  2. SWA-averaged checkpoints (window +-15 over stride 5, as level_task.py) are scored exactly at centers 25 steps apart,
     walking from the crossing until an adjacent pair brackets the target;
  3. weight interpolation between that pair, secant steps on log pass@1, until within --tol (relative) of the target.
The chosen model is copied to <out>/best and <out>/tune.json records every exact evaluation plus a "best" entry in the
tune_pass1.py format (pass1_confirm = the exact pass@1, method "exact"; per-maze probabilities kept, so pass@k is exact).

Works on a local mirror of the run: only the checkpoints it needs are fetched from --run-s3.

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/exact_tune.py --run-s3 s3://.../pt_dfs_eps05_v1/L/L_is0_a11082334 \
        --target 0.01 --level p1 --work /tmp/exact_tune --parquet <dev test.parquet>
"""
import argparse
import glob
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exact_pass as ep  # noqa: E402
import screen_sft_ckpts as ss  # noqa: E402


def sh(*cmd):
    subprocess.run(list(cmd), check=True, stdout=subprocess.DEVNULL)


def aws(*cmd, **kw):
    """aws CLI call with retries (concurrent syncs into the same S3 prefix fail transiently)."""
    for i in range(4):
        env = {**os.environ, "AWS_PROFILE": os.environ.get("EXACT_TUNE_AWS_PROFILE", "ml-worker")}   # can write s3://scale-ml
        if subprocess.run(["aws", *cmd], env=env, **kw).returncode == 0:
            return
        time.sleep(10 * (i + 1))
    raise SystemExit(f"aws {' '.join(cmd[:3])} failed 4 times")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-s3", required=True)
    ap.add_argument("--level", required=True)
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=0.0025, help="relative tolerance on the exact pass@1")
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--center-step", type=int, default=25)
    ap.add_argument("--half-window", type=int, default=15)
    ap.add_argument("--max-paths", type=int, default=2500)
    ap.add_argument("--upload", action="store_true", help="write <run-s3>/<out-name>/{best,tune.json,COMPLETE}")
    ap.add_argument("--ckpt-stride", type=int, default=5, help="spacing of the averaged checkpoints (5 = dense; 100 = every-100 runs)")
    ap.add_argument("--batch-tokens", type=int, default=262144, help="exact scorer tokens per batch (lower on a shared GPU)")
    ap.add_argument("--out-name", default=None, help="output dir name (default tune-<level>)")
    args = ap.parse_args()
    t_start = time.time()
    run = os.path.join(args.work, os.path.basename(args.run_s3.rstrip("/")))
    out_name = args.out_name or f"tune-{args.level}"
    out = os.path.join(run, out_name)
    os.makedirs(out, exist_ok=True)
    aws("s3", "sync", "--only-show-errors", args.run_s3, run, "--exclude", "*", "--include", "screen_*.json")
    prompts, gts = ss.load_prompts(args.parquet, args.n_prompts)
    log = {"target": args.target, "method": "exact", "screens": {}}

    def evaluate(path, key):
        r = ep.exact_screen(path, prompts, gts, max_paths=args.max_paths, batch_tokens=args.batch_tokens)
        log["screens"][key] = r
        p = r["pass@1_lo"]
        print(f"{key:>28} exact pass@1 {p:.6f} ({100 * (p - args.target) / args.target:+.2f}%) [{r['seconds']}s]", flush=True)
        return p

    # 1. approximate crossing from the stride screens
    sc = {}
    for f in glob.glob(os.path.join(run, "screen_*.json")):
        try:   # a screen file mirrored mid-write is truncated; the others cover the curve
            sc.update(json.load(open(f)))
        except (json.JSONDecodeError, OSError):
            print(f"skipping unreadable {os.path.basename(f)}", flush=True)
    pts = sorted((int(k.split("-")[1]), v["is_shortest"]["pass@1"]) for k, v in sc.items() if k.startswith("ckpt-"))
    smooth = [(s, statistics.median([q for _, q in pts[max(0, i - 2):i + 3]])) for i, (s, _) in enumerate(pts)]
    cross = next((s for s, q in smooth if q >= args.target), pts[-1][0])
    hw, cs, ks = args.half_window, args.center_step, args.ckpt_stride
    print(f"{os.path.basename(run)} {args.level}: screened crossing ~{cross}", flush=True)

    cache = {}

    def center_p(c):
        if c in cache:
            return cache[c]
        inc = []
        for s in range(c - hw, c + hw + 1, ks):
            inc += ["--include", f"ckpt-{s}/*"]
        aws("s3", "sync", "--only-show-errors", args.run_s3, run, "--exclude", "*", *inc)
        missing = [s for s in range(c - hw, c + hw + 1, ks) if not os.path.isfile(os.path.join(run, f"ckpt-{s}", "model.safetensors"))]
        if missing:
            raise SystemExit(f"checkpoints {missing} not on S3 for center {c}")
        path = os.path.join(run, f"avg-{c}-w{hw}")
        if not os.path.isfile(os.path.join(path, "model.safetensors")):
            sh(sys.executable, os.path.join(HERE, "average_ckpts.py"), "--run", run, "--center", str(c), "--half-window", str(hw),
               "--stride", str(ks), "--out", path)
        cache[c] = (evaluate(path, f"avg-{c}-w{hw}"), path)
        return cache[c]

    # 2. walk 25-step centers from the crossing to an adjacent bracketing pair
    c = max(hw, int(round(cross / cs)) * cs)
    p, _ = center_p(c)
    step = cs if p < args.target else -cs
    for _ in range(200):
        c2 = c + step
        if c2 - hw < ks:
            break
        p2, _ = center_p(c2)
        if (p2 >= args.target) != (p >= args.target):
            break
        c, p = c2, p2
    if c + step not in cache:
        raise SystemExit(f"no bracketing pair found walking from the crossing (last center {c}: {p:.6f})")
    lo_c, hi_c = sorted((c, c + step), key=lambda x: cache[x][0])
    (pa, a), (pb, b) = cache[lo_c], cache[hi_c]
    if not pa < args.target <= pb:
        raise SystemExit(f"no bracketing pair found (last {lo_c}:{pa:.6f}, {hi_c}:{pb:.6f})")

    # 3. interpolate: keep a bracket [lo, hi] in alpha with pass@1(lo) < target <= pass@1(hi) (pass@1 need not be monotone in
    # alpha: an interpolated model can beat both endpoints), secant on log pass@1 inside it, bisection if the secant leaves it
    t = math.log(args.target)
    pts_i = [(0.0, pa, a), (1.0, pb, b)]
    lo, hi = pts_i
    for _ in range(args.max_iters):
        best = min(pts_i, key=lambda q: abs(q[1] - args.target))
        if abs(best[1] - args.target) <= args.tol * args.target:
            break
        y1, y2 = math.log(max(lo[1], 1e-12)), math.log(max(hi[1], 1e-12))
        alpha = lo[0] + (t - y1) * (hi[0] - lo[0]) / (y2 - y1) if y2 != y1 else 0.5 * (lo[0] + hi[0])
        span = abs(hi[0] - lo[0])
        if not min(lo[0], hi[0]) + 0.05 * span < alpha < max(lo[0], hi[0]) - 0.05 * span:
            alpha = 0.5 * (lo[0] + hi[0])
        path = os.path.join(out, f"interp-a{alpha:.5f}")
        if not os.path.isfile(os.path.join(path, "model.safetensors")):
            sh(sys.executable, os.path.join(HERE, "interpolate_ckpts.py"), "--a", a, "--b", b, "--alpha", f"{alpha:.5f}", "--out", path)
        q = (alpha, evaluate(path, f"interp-a{alpha:.5f}"), path)
        pts_i.append(q)
        if q[1] < args.target:
            lo = q
        else:
            hi = q
    alpha, p, path = min(pts_i, key=lambda q: abs(q[1] - args.target))
    best_dir = os.path.join(out, "best")
    shutil.rmtree(best_dir, ignore_errors=True)
    shutil.copytree(path, best_dir)
    key = os.path.basename(path)
    r = log["screens"][key]
    log.update(a=a, b=b, best={"alpha": alpha, "path": path, "pass1_search": p, "pass1_confirm": p,
                               "confirm_rel_vs_target": (p - args.target) / args.target, "se_rel": 0.0, "method": "exact",
                               "pass1_hi": r["pass@1_hi"], "passk": r["pass@k_lo"], "key": key},
               seconds=round(time.time() - t_start, 1))
    with open(os.path.join(out, "tune.json"), "w") as f:
        json.dump(log, f)
    print(f"BEST {os.path.basename(run)} {args.level}: alpha {alpha:.4f} between {os.path.basename(a)} and {os.path.basename(b)}, "
          f"exact pass@1 {p:.6f} ({100 * (p - args.target) / args.target:+.3f}%) in {log['seconds']}s", flush=True)
    if args.upload:
        dst = f"{args.run_s3.rstrip('/')}/{out_name}"
        aws("s3", "sync", "--only-show-errors", best_dir, f"{dst}/best")
        aws("s3", "cp", "--only-show-errors", os.path.join(out, "tune.json"), f"{dst}/tune.json")
        aws("s3", "cp", "-", f"{dst}/COMPLETE", input=b"")
        print(f"UPLOADED {dst}", flush=True)


if __name__ == "__main__":
    main()
