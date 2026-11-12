#!/usr/bin/env python3
"""Status table of the small-model maze RL grid from S3: per run (level x prompts per step x seed) the state (done with
reason / running / queued), last validated step, best validation goal_reached (mean@16 on the 500-maze dev set) and the
step of the best checkpoint (best_hf); --json also keeps each run's validation history (for report_matched_curves.py
--rl-status), and with --wandb-run-suffix each run's actor/entropy curve from W&B (run id <suite>-<suffix>).
Reads <root>/runs/<suite>/{COMPLETE.json, state/best.json}.

    python3 rl-distill-scripts/maze/rl_grid_status.py --root s3://scale-ml/genai/rl-distill/maze/rl_small_v1 [--prefix sml] [--json out.json]
"""
import argparse
import json

import boto3

LEVELS = ["p01", "p03", "p1", "p3", "p10"]
BATCHES = [4, 16, 64, 256]
SEEDS = [0, 1, 2]


def read_json(s3, bucket, key):
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--prefix", default="sml", help="run-name prefix (sml for grid v1, sml2 for v2)")
    ap.add_argument("--json", help="also write the rows here")
    ap.add_argument("--wandb-run-suffix", default=None, help="fetch actor/entropy from W&B run <suite>-<suffix> (needs WANDB_API_KEY)")
    ap.add_argument("--wandb-path", default="rl-distill/tailrl-text-maze", help="<entity>/<project>")
    ap.add_argument("--entropy-samples", type=int, default=400)
    ap.add_argument("--batches", type=int, nargs="+", default=BATCHES, help="prompts per step in the grid")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="RL seeds in the grid")
    ap.add_argument("--levels", nargs="+", default=LEVELS, help="start levels in the grid")
    args = ap.parse_args()
    bucket, prefix = args.root[len("s3://"):].rstrip("/").split("/", 1)
    s3 = boto3.Session(profile_name="ml-worker").client("s3")
    rows = []
    for lvl in args.levels:
        for b in args.batches:
            for seed in args.seeds:
                suite = f"{args.prefix}-{lvl}-b{b}-s{seed}"
                done = read_json(s3, bucket, f"{prefix}/runs/{suite}/COMPLETE.json")
                best = read_json(s3, bucket, f"{prefix}/runs/{suite}/state/best.json")
                hist = (best or {}).get("history") or []
                rows.append({
                    "suite": suite, "level": lvl, "batch": b, "seed": seed,
                    "state": done["reason"] if done else ("running" if best else "queued"),
                    "last_val_step": hist[-1][0] if hist else None,
                    "best_step": (done or best or {}).get("best_step"),
                    "best_value": (done or best or {}).get("best_value"),
                    "last_step": (done or {}).get("last_step"),
                    "history": hist, "wandb_suffix": args.wandb_run_suffix,
                })
    if args.wandb_run_suffix:
        import wandb
        api = wandb.Api(timeout=60)
        for r in rows:
            if r["state"] == "queued":
                continue
            try:
                hist = api.run(f"{args.wandb_path}/{r['suite']}-{args.wandb_run_suffix}").history(
                    keys=["actor/entropy"], samples=args.entropy_samples, pandas=False)
            except wandb.errors.CommError:  # run not created yet
                continue
            r["entropy"] = sorted((h["_step"], h["actor/entropy"]) for h in hist)
    fmt = lambda v: "-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    print(f"{'run':>16} {'state':>10} {'last val':>9} {'best step':>9} {'best':>7}")
    for r in rows:
        print(f"{r['suite']:>16} {r['state']:>10} {fmt(r['last_val_step']):>9} {fmt(r['best_step']):>9} {fmt(r['best_value']):>7}")
    n_done = sum(r["state"] not in ("running", "queued") for r in rows)
    print(f"{n_done}/{len(rows)} done, {sum(r['state'] == 'running' for r in rows)} running")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
