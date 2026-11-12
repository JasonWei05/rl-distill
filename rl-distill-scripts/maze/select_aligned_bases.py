#!/usr/bin/env python3
"""Per level, pick the base of one size whose exact shortest-path pass@k curve (k = 1..1024) is closest to a reference base
(max_k |pass@k - ref|), among every confirmed candidate of that size: tune-<level>/best of each run (pass@1-matched) and
align-<level>/best (align_passk.py, curve-matched). All curves are scored exactly (exact_pass.py) on the same dev mazes.
Copies each chosen model to <out>/bases/<level>_s<seed>/, writes <out>/bases.json and prints BASES_SPEC.

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/select_aligned_bases.py --pt s3://.../pt_dfs_eps05_v1 --size S \
        --ref-bases s3://.../rl_small_dfs_eps05_v1 --out s3://.../rl_medium_dfs_eps05_v1 --parquet <dev test.parquet>
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exact_pass as ep  # noqa: E402
import screen_sft_ckpts as ss  # noqa: E402

LEVELS = {"p01": 0.001, "p03": 0.003, "p1": 0.01, "p3": 0.03, "p10": 0.10}


def aws(*a, **kw):
    return subprocess.run(["aws", *a], env={**os.environ, "AWS_PROFILE": "ml-worker"}, capture_output=True, text=True, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--ref-bases", required=True, help="RL grid prefix whose bases.json / bases/<level>_s<seed> are the references")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--work", default="/tmp/jw_maze/select_bases")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    prompts, gts = ss.load_prompts(args.parquet, 1000)
    ref_spec = json.loads(aws("s3", "cp", "--quiet", f"{args.ref_bases}/bases.json", "-").stdout)["levels"]
    runs = [l.split()[-1].rstrip("/") for l in aws("s3", "ls", f"{args.pt}/{args.size}/").stdout.splitlines()
            if l.strip().startswith("PRE") and f"{args.size}_is" in l]
    chosen, table = {}, {}
    for lvl, tgt in LEVELS.items():
        ref_dir = os.path.join(args.work, f"ref_{lvl}")
        aws("s3", "sync", "--only-show-errors", f"{args.ref_bases}/bases/{lvl}_s{ref_spec[lvl]['seed']}", ref_dir)
        ref = ep.exact_screen(ref_dir, prompts, gts, max_paths=2500)["pass@k_lo"]
        cands = []
        for run in runs:
            seed = int(run.split("_is")[1].split("_")[0])
            for kind in ("tune", "align"):
                src = f"{args.pt}/{args.size}/{run}/{kind}-{lvl}"
                if aws("s3", "ls", f"{src}/COMPLETE").returncode != 0:
                    continue
                d = os.path.join(args.work, f"{args.size}_{lvl}_s{seed}_{kind}")
                aws("s3", "sync", "--only-show-errors", f"{src}/best", d)
                c = ep.exact_screen(d, prompts, gts, max_paths=2500)["pass@k_lo"]
                diff = max(abs(c[k] - ref[k]) for k in ref)
                cands.append((diff, seed, kind, src, c))
                print(f"{lvl} s{seed} {kind:5s}: pass@1 {c[1]:.6f} ({100 * (c[1] - tgt) / tgt:+.2f}%), max |pass@k - ref| {diff:.4f}", flush=True)
        diff, seed, kind, src, c = min(cands)
        table[lvl] = [{"seed": s, "kind": k, "max_abs_diff": d} for d, s, k, _, _ in cands]
        chosen[lvl] = {"seed": seed, "kind": kind, "src": f"{src}/best", "base": f"{args.out}/bases/{lvl}_s{seed}", "target": tgt,
                       "pass1_confirm": c[1], "rel": (c[1] - tgt) / tgt, "max_abs_diff_vs_ref": diff, "passk": c, "ref_passk": ref}
        print(f"CHOSEN {lvl}: s{seed} {kind} (max |pass@k - ref| {diff:.4f}, pass@1 {100 * (c[1] - tgt) / tgt:+.2f}%)", flush=True)
    if not args.dry_run:
        for c in chosen.values():
            aws("s3", "sync", "--only-show-errors", c["src"], c["base"], check=True)
        aws("s3", "cp", "-", f"{args.out}/bases.json", input=json.dumps(
            {"size": args.size, "pt": args.pt, "ref": args.ref_bases, "levels": chosen, "candidates": table}, indent=1), check=True)
    print("BASES_SPEC=" + " ".join(f"{lvl}.s{c['seed']}" for lvl, c in chosen.items()))


if __name__ == "__main__":
    main()
