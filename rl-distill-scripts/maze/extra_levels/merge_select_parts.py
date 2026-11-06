#!/usr/bin/env python3
"""Merge a selection that was split over several GPUs back into the one file make_pair.py reads (federer, 2026-11-03).

The L 0.3% selection (select_matched_init.py --k 256 over the bracket [1550, 2000]) crashed on an OOM caused by another
user's job, so it was re-run as three --bracket parts with --n-final 0 (select_goal_reached_0p3pct.part{A,B,C}.json),
then one --bracket c c --k 1024 --n-final 1 job per finalist (select_goal_reached_0p3pct.final_<c>.json). Every candidate
is screened with torch.manual_seed(0) first, so the merged file holds what the single job would have written.
    merge_select_parts.py <run dir> <out name> <target> <lo> <hi> [--top 3]   # grid parts -> out, prints the finalists
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out_name")
    ap.add_argument("target", type=float)
    ap.add_argument("lo", type=int)
    ap.add_argument("hi", type=int)
    ap.add_argument("--metric", default="goal_reached")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()
    stem = args.out_name[: -len(".json")]
    merged, sources = {}, []
    for p in sorted(glob.glob(os.path.join(args.run, stem + ".part*.json")) + glob.glob(os.path.join(args.run, stem + ".final_*.json"))):
        d = json.load(open(p))
        merged.update({k: v for k, v in d.items() if not k.startswith("_")})
        sources.append(os.path.basename(p))
    grid = {k.rsplit("@k", 1)[0]: v[args.metric]["pass@1"] for k, v in merged.items() if not k.endswith("@k1024")}
    ranked = sorted(grid, key=lambda n: abs(grid[n] - args.target))
    finals = {k.rsplit("@k", 1)[0]: v[args.metric]["pass@1"] for k, v in merged.items() if k.endswith("@k1024")}
    merged["_selection"] = {"metric": args.metric, "target": args.target, "bracket": [args.lo, args.hi],
                            "ranked_centers": [int(n.split("-")[1]) for n in ranked],
                            "note": "merged by merge_select_parts.py from " + ", ".join(sources)}
    out = os.path.join(args.run, args.out_name)
    with open(out + ".tmp", "w") as f:
        json.dump(merged, f)
    os.replace(out + ".tmp", out)
    print(f"{len(grid)} grid candidates, {len(finals)} finals -> {out}")
    print("closest at the grid k:", ", ".join(f"{n} {grid[n]:.4f}" for n in ranked[: args.top + 2]))
    print("finalists:", " ".join(str(int(n.split("-")[1])) for n in ranked[: args.top]))
    if finals:
        print("finals k=1024:", ", ".join(f"{n} {p:.5f}" for n, p in sorted(finals.items(), key=lambda x: abs(x[1] - args.target))))


if __name__ == "__main__":
    main()
