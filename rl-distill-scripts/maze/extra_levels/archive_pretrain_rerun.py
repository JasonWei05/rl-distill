#!/usr/bin/env python3
"""Archive what Step 1-2 of the extra-starts run keeps from the pretraining re-run, then let the raw checkpoints go.

Mirrors the original archive's pretrain/ layout into hi-todayis-jh/tailrl-text-maze-extra-levels: per run, the averaged
candidates the new matched pairs use (matched/<pair>_{S0,L0} -> avg-<c>-w60 or interp-*), their raw member checkpoints
(AVERAGED_FROM.txt; for an interpolated candidate, both ends and their members), init_model/, train/eval metrics, the
k = 16 screening curves and every selection table/log; plus pretrain/manifest.json (which pair uses which directory).
    archive_pretrain_rerun.py [--pairs p03 p3 p5] [--dry-run]
"""
import argparse
import datetime as dt
import json
import os
import re

from huggingface_hub import HfApi

REPO = "hi-todayis-jh/tailrl-text-maze-extra-levels"
HOME = os.path.expanduser("~")
R = os.path.join(HOME, "tailrl_maze_runs/pretrain20k_rerun")
MATCHED = os.path.join(HOME, "tailrl_maze_runs/matched")
NOTE = ("Re-run of the budget study's two pretraining trajectories on federer (A800 40 GB), runbook 'Extra start levels for "
        "the maze RL-curve law': S_ds0_is4 (S, init_seed 4) and L (init_seed 0), data_seed 0, a checkpoint every 20 steps, "
        "init_model identical to the originals. L_ds0_is0_rep was stopped at ~8k steps after its 0.3% bracket had been "
        "pruned by mistake; L_ds0_is0_rep2 repeats it to step 5,000 and reproduced it bit for bit (identical k = 16 "
        "screening), so every new L base comes from rep2. The re-run L crosses ~1,000 steps earlier than the original "
        "(which had a loss spike near step 2,000); the re-run S ~500-1,800 steps later. Raw checkpoints other than the "
        "used candidates' averaging windows were deleted.")


def members_of(cand_dir):
    """(raw checkpoint names, candidate dir names) behind a candidate: avg-* -> its AVERAGED_FROM.txt members; interp-*
    -> both ends of INTERPOLATED_FROM.txt, recursively (an end can itself be an interp-*; its AVERAGED_FROM.txt is only
    a copy of end a's)."""
    name = os.path.basename(cand_dir)
    interp = os.path.join(cand_dir, "INTERPOLATED_FROM.txt")
    if os.path.exists(interp):
        ends = dict(line.split("=", 1) for line in open(interp).read().split("\n") if "=" in line)
        ckpts, cands = set(), {name}
        for end in (ends["a"], ends["b"]):
            k, n = members_of(end)
            ckpts |= k
            cands |= n
        return ckpts, cands
    txt = open(os.path.join(cand_dir, "AVERAGED_FROM.txt")).read()
    m = re.search(r"members=\[([^\]]*)\]", txt)
    return {f"ckpt-{int(s)}" for s in m.group(1).split(",") if s.strip()}, {name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["p03", "p3", "p5"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    keep = {}  # run -> set of relative paths (dirs or files)
    manifest = {"note": NOTE, "created": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z", "pairs": {}, "runs": {}}
    for pair in args.pairs:
        for side in ("S0", "L0"):
            target = os.path.realpath(os.path.join(MATCHED, f"{pair}_{side}"))
            run, cand = os.path.dirname(target), os.path.basename(target)
            ckpts, cands = members_of(target)
            keep.setdefault(os.path.basename(run), set()).update(ckpts | cands)
            manifest["pairs"].setdefault(pair, {})[side] = {"run": os.path.basename(run), "candidate": cand,
                                                              "raw_members": sorted(ckpts, key=lambda s: int(s.split("-")[1]))}
    api = HfApi()
    for run in ("S_ds0_is4", "L_ds0_is0_rep2", "L_ds0_is0_rep"):
        rdir = os.path.join(R, run)
        files = sorted(f for f in os.listdir(rdir) if os.path.isfile(os.path.join(rdir, f)))
        dirs = sorted(keep.get(run, set())) + (["init_model"] if os.path.isdir(os.path.join(rdir, "init_model")) else [])
        manifest["runs"][run] = {"files": files, "dirs": dirs}
        allow = files + [f"{d}/*" for d in dirs]
        size = sum(os.path.getsize(os.path.join(rdir, f)) for f in files) + sum(
            os.path.getsize(os.path.join(dp, f)) for d in dirs for dp, _, fs in os.walk(os.path.join(rdir, d)) for f in fs)
        print(f"{run}: {len(files)} files + {len(dirs)} dirs, {size / 1e9:.2f} GB")
        if not args.dry_run:
            api.upload_folder(folder_path=rdir, path_in_repo=f"pretrain/{run}", repo_id=REPO, repo_type="dataset",
                              allow_patterns=allow, commit_message=f"pretrain/{run}: re-run archive (extra starts)")
    out = os.path.join(R, "pretrain_rerun_manifest.json")
    json.dump(manifest, open(out, "w"), indent=1)
    if not args.dry_run:
        api.upload_file(path_or_fileobj=out, path_in_repo="pretrain/manifest.json", repo_id=REPO, repo_type="dataset",
                        commit_message="pretrain/manifest.json")
    print(json.dumps(manifest["pairs"], indent=1)[:2000])


if __name__ == "__main__":
    main()
