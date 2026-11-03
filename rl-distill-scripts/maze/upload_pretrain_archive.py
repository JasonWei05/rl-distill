#!/usr/bin/env python3
"""Upload what remains of the text-maze pretraining runs to the study's Hub dataset repo.

After `prune_pretrain_ckpts.sh` (2026-10-27), `~/tailrl_maze_runs/pretrain20k/` holds ~1.3 GB: per run the SWA-averaged
candidates (`avg-<center>-w60/`), the raw checkpoints inside each used candidate's averaging window plus the same-step
S/L control checkpoints, `init_model/`, and every metrics / screening / selection file. This script mirrors that tree to
`pretrain/<run>/...` in the Hub repo and writes `meta/pretrain_manifest.json` describing every kept directory (role,
averaging members, model config, which matched pair uses it).

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/upload_pretrain_archive.py --dry-run   # manifest + counts only
    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/upload_pretrain_archive.py             # upload (re-runnable)

Uses the cached Hub token (~/.cache/huggingface/token). Re-running re-uploads nothing that is already there (LFS blobs are
deduplicated by sha256) and rewrites the manifest.
"""
import argparse
import datetime as dt
import json
import os
import re
import sys

HOME = os.path.expanduser("~")
REPO = "hi-todayis-jh/tailrl-text-maze-budget-study"
PRIMARY = {"S": "S_ds0_is4", "L": "L_ds0_is0_rep"}  # the runs every used starting point comes from

RECIPE = {
    "launcher": "rl-distill-scripts/maze/launch_pretrain_matrix.sh with RUNS_ROOT=pretrain20k MAX_STEPS=20000 SAVE_STEPS=20",
    "trainer": "TailRL experiments/text_maze/scripts/sft_pretrain.py (+ --data_seed / --init_seed from tailrl_text_maze.patch)",
    "data": "sft_data/train_random.json = the random-path SFT split of max-rl/maze_17x17_diverse_1.3m (TEST_PROMPTS=1000)",
    "learning_rate": 5e-4, "batch_size": 32, "micro_batch_size": 8, "optimizer": "AdamW betas (0.9, 0.95), weight_decay 0.01",
    "lr_scheduler": "cosine without warmup over the trainer's default 10-epoch horizon (~406k steps), stopped at max_steps: "
                    "effectively constant, 5.00e-4 at step 10 -> 4.97e-4 at step 20000 (train_metrics.jsonl)",
    "max_steps": 20000, "save_steps": 20, "eval_steps": 50, "data_seed": 0,
    "note": "data_seed 0 for every run: S and L see identical batches in identical order; only init_seed differs. "
            "`_rep` runs repeat init_seed 0 bit-for-bit in configuration (reproducibility check).",
}
KEPT = ("Per matched pair (p10 / p1 / p01, plus the unused alternates alt1 / alt2): the averaged candidate the pair's S0 / L0 "
        "symlink points to (avg-<center>-w60 = SWA of 7 checkpoints, center ± 60 steps, stride 20); for the six used "
        "candidates also their 7 raw member checkpoints and the same-step S/L control checkpoints (S_ds0_is4 at the L0 "
        "centers, L_ds0_is0_rep at the S0 centers); every run's init_model (step 0), train/eval metrics, dev screening "
        "curves (screen_stride50_k16*.json, every 100 steps, k = 16 on eval1000) and selection tables "
        "(select_goal_reached*.json). All other raw checkpoints of the 20k-step trajectories were deleted.")


def read_json(p):
    with open(p) as f:
        return json.load(f)


def averaged_from(d):
    """Parse avg-*/AVERAGED_FROM.txt -> dict(run, center, half_window, stride, members)."""
    out = {}
    p = os.path.join(d, "AVERAGED_FROM.txt")
    if not os.path.exists(p):
        return out
    for line in open(p):
        if "=" not in line:
            continue
        k, v = line.strip().split("=", 1)
        if k == "members":
            out[k] = [int(x) for x in re.findall(r"\d+", v)]
        elif k in ("center", "half_window", "stride"):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def dir_stats(d):
    n, b = 0, 0
    for root, _, files in os.walk(d):
        for f in files:
            n += 1
            b += os.path.getsize(os.path.join(root, f))
    return n, b


def pair_uses(matched_manifest):
    """{(run, candidate): ['p10 S0', ...]} from matched/manifest.json (primary == p10)."""
    uses = {}
    for pair, v in matched_manifest.get("pairs", {}).items():
        name = "p10" if pair == "primary" else pair
        for side in ("S0", "L0"):
            if side in v and "run" in v[side]:
                uses.setdefault((v[side]["run"], v[side]["candidate"]), []).append(f"{name} {side}")
    return uses


def build_manifest(src, matched):
    uses = pair_uses(matched)
    centers = {}  # (run, step) -> avg name, for the same-step controls
    runs = {}
    for run in sorted(os.listdir(src)):
        rd = os.path.join(src, run)
        if not os.path.isdir(rd):
            continue
        m = re.match(r"([SL])_ds(\d+)_is(\d+)(_rep)?$", run)
        if not m:
            continue
        size, data_seed, init_seed, rep = m.group(1), int(m.group(2)), int(m.group(3)), bool(m.group(4))
        avgs = {}
        for a in sorted(x for x in os.listdir(rd) if x.startswith("avg-")):
            info = averaged_from(os.path.join(rd, a))
            members = info.get("members", [])
            present = all(os.path.isdir(os.path.join(rd, f"ckpt-{s}")) for s in members) if members else False
            avgs[a] = dict(center=info.get("center"), half_window=info.get("half_window"), stride=info.get("stride"),
                           members=members, members_present=present, used_as=uses.get((run, a), []))
            if info.get("center") is not None:
                centers[(run, info["center"])] = a
        runs[run] = dict(size=size, data_seed=data_seed, init_seed=init_seed, repeat_of_init_seed_0=rep, avg=avgs,
                         raw_ckpts={}, init_model=os.path.isdir(os.path.join(rd, "init_model")),
                         metrics=sorted(x for x in os.listdir(rd) if x.endswith((".json", ".jsonl"))),
                         logs=sorted(x for x in os.listdir(rd) if x.endswith(".log")))
        n, b = dir_stats(rd)
        runs[run]["files"], runs[run]["bytes"] = n, b
        cfg = os.path.join(rd, "init_model", "config.json")
        if os.path.exists(cfg):
            c = read_json(cfg)
            runs[run]["config"] = {k: c.get(k) for k in ("hidden_size", "intermediate_size", "num_hidden_layers",
                                                          "num_attention_heads", "num_key_value_heads", "vocab_size")}
    # roles of the raw checkpoints: window member of an avg in the same run, or same-step control of the other primary run
    other = {PRIMARY["S"]: PRIMARY["L"], PRIMARY["L"]: PRIMARY["S"]}
    for run, r in runs.items():
        rd = os.path.join(src, run)
        for c in sorted((x for x in os.listdir(rd) if x.startswith("ckpt-")), key=lambda x: int(x.split("-")[1])):
            step = int(c.split("-")[1])
            roles = [f"member of {a}" for a, v in r["avg"].items() if step in v["members"]]
            o = other.get(run)
            if o and (o, step) in centers:
                roles.append(f"same-step control for {o}/{centers[(o, step)]} ({', '.join(runs[o]['avg'][centers[(o, step)]]['used_as']) or 'unused'})")
            r["raw_ckpts"][c] = dict(step=step, roles=roles or ["other"])
    pairs = {}
    for (run, cand), names in uses.items():
        for nm in names:
            pair, side = nm.split()
            pairs.setdefault(pair, {})[side] = f"pretrain/{run}/{cand}"
    return dict(created=dt.date.today().isoformat(), source=f"{src} on aiai-rtx-2, after prune_pretrain_ckpts.sh (2026-10-27)",
                recipe=RECIPE, kept=KEPT, primary_runs=PRIMARY, pairs=pairs,
                matching_rule=matched.get("rule"), dev_set=matched.get("dev_set"), runs=runs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(HOME, "tailrl_maze_runs/pretrain20k"))
    ap.add_argument("--matched", default=os.path.join(HOME, "tailrl_maze_runs/matched/manifest.json"))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--path-in-repo", default="pretrain")
    ap.add_argument("--manifest-out", default=None, help="also write the manifest here (default: <src>/../pretrain_manifest.json)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = build_manifest(args.src, read_json(args.matched))
    total_files = sum(r["files"] for r in manifest["runs"].values())
    total_bytes = sum(r["bytes"] for r in manifest["runs"].values())
    top = [x for x in os.listdir(args.src) if os.path.isfile(os.path.join(args.src, x))]
    print(f"{len(manifest['runs'])} runs, {total_files} files in run dirs + {len(top)} top-level files, "
          f"{total_bytes / 1e9:.2f} GB; pairs: {json.dumps(manifest['pairs'])}")
    for run, r in manifest["runs"].items():
        print(f"  {run:14s} {r['bytes'] / 1e6:7.1f} MB  avg {list(r['avg'])}  raw ckpts {len(r['raw_ckpts'])}  init_model {r['init_model']}")
    out = args.manifest_out or os.path.join(os.path.dirname(os.path.abspath(args.src)), "pretrain_manifest.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=1)
    print("manifest ->", out)
    if args.dry_run:
        return 0

    from huggingface_hub import HfApi  # noqa: E402

    api = HfApi()
    print(f"uploading {args.src} -> {args.repo}:{args.path_in_repo}/ ...", flush=True)
    r = api.upload_folder(folder_path=args.src, path_in_repo=args.path_in_repo, repo_id=args.repo, repo_type="dataset",
                          commit_message=f"pretrain/: what remains of the 12 pretraining runs ({total_bytes / 1e9:.2f} GB)")
    print("folder commit:", getattr(r, "commit_url", r), flush=True)
    r = api.upload_file(path_or_fileobj=out, path_in_repo="meta/pretrain_manifest.json", repo_id=args.repo,
                        repo_type="dataset", commit_message="meta/pretrain_manifest.json")
    print("manifest commit:", getattr(r, "commit_url", r))
    print("UPLOAD_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
