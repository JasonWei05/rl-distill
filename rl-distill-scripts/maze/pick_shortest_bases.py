#!/usr/bin/env python3
"""Choose the RL bases of the shortest-path maze study: per pass@1 level, the seed whose fresh-seed confirmation is closest
to the target (relative), among the runs listed in <pt>/<size>/status.json (the current attempt of maze_pt_pipeline.py).

Copies each chosen tune-<level>/best to <out>/bases/<level>_s<seed>/, writes <out>/bases.json and prints the
BASES_SPEC line for launch_rl_small_grid.sh (GRID=sp1). Needs every level of every listed run confirmed unless
--allow-missing.

    python3 pick_shortest_bases.py --pt s3://scale-ml/genai/rl-distill/maze/pt_shortest_v1 --size XS \
        --out s3://scale-ml/genai/rl-distill/maze/rl_small_shortest_v1
"""
import argparse
import json
import subprocess

LEVELS = {"p01": 0.001, "p03": 0.003, "p1": 0.01, "p3": 0.03, "p10": 0.10}


def s3_json(uri):
    p = subprocess.run(["aws", "s3", "cp", "--quiet", uri, "-"], capture_output=True, text=True)
    return json.loads(p.stdout) if p.returncode == 0 and p.stdout else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", required=True, help="pretraining output prefix (maze_pt_pipeline.py root on S3)")
    ap.add_argument("--size", default="XS")
    ap.add_argument("--out", required=True, help="RL grid prefix; bases go to <out>/bases/")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    status = s3_json(f"{args.pt}/{args.size}/status.json")
    chosen, table = {}, {}
    for lvl, tgt in LEVELS.items():
        cands = []
        for run, st in status["runs"].items():
            tune = s3_json(f"{args.pt}/{args.size}/{run}/tune-{lvl}/tune.json")
            if tune and "best" in tune:
                cands.append((abs(tune["best"]["confirm_rel_vs_target"]), st["seed"], run, tune["best"]))
        table[lvl] = {f"s{s}": b["pass1_confirm"] for _, s, _, b in cands}
        if len(cands) < len(status["runs"]) and not args.allow_missing:
            raise SystemExit(f"{lvl}: {len(cands)} of {len(status['runs'])} runs confirmed")
        _, seed, run, best = min(cands)
        chosen[lvl] = {"seed": seed, "run": run, "pass1_confirm": best["pass1_confirm"], "target": tgt,
                       "rel": best["confirm_rel_vs_target"], "src": f"{args.pt}/{args.size}/{run}/tune-{lvl}/best",
                       "base": f"{args.out}/bases/{lvl}_s{seed}"}
        print(f"{lvl} target {tgt:g}: seed {seed} {best['pass1_confirm']:.6f} ({100 * best['confirm_rel_vs_target']:+.2f}%)"
              f"  all seeds {table[lvl]}", flush=True)
    if not args.dry_run:
        for c in chosen.values():
            subprocess.run(["aws", "s3", "sync", "--only-show-errors", c["src"], c["base"]], check=True)
        subprocess.run(["aws", "s3", "cp", "-", f"{args.out}/bases.json"], input=json.dumps(
            {"size": args.size, "pt": args.pt, "levels": chosen, "all_seeds": table}, indent=1), text=True, check=True)
    print("BASES_SPEC=" + " ".join(f"{lvl}.s{c['seed']}" for lvl, c in chosen.items()))


if __name__ == "__main__":
    main()
