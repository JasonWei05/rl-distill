#!/usr/bin/env python3
"""Keep the S RL runs' checkpoint trees off /work2 while they train (federer, 2026-11-03).

Per run dir under ~/tailrl_maze_runs/checkpoints/tailrl-text-maze/ whose name matches --runs-regex:
  * the two newest global_step_N/ stay complete (resume_mode=auto needs the newest one's optimizer state);
  * older steps are slimmed to what the Hub archive of the budget study keeps (actor/huggingface/ + flops_ledger.json):
    the FSDP model/optimizer/extra-state shards and data.pt are deleted;
  * every --chunk slimmed steps are packed into one tar, uploaded to the private repo as
    rl/<run>/global_step_<first>-<last>.tar, checked by size on the Hub, and only then deleted locally.
    The archived actor/huggingface/model.safetensors is cast fp32 -> bf16 (7.9 MB instead of 15.8 MB): evaluation loads
    bf16 anyway (screen_sft_ckpts.screen_one), and the account's private quota (100 GB, 57 GB used) cannot hold up to
    ~3,000 fp32 exports if the 0.3% runs are extended to 10,000 steps. The final full checkpoint stays fp32.
A run with a FINALIZE file in its dir is packed down to its newest step (whatever the chunk fill) and its newest
step is also uploaded whole (with optimizer state, for a later extension) as rl/<run>/final_full_global_step_<N>.tar.
Uploads stop (and everything stays local) once the repo would pass --max-repo-gb, the account's private quota margin.
    python rl_ckpt_janitor.py [--loop 300] [--dry-run]
"""
import argparse
import json
import os
import re
import shutil
import tarfile
import time
import traceback

from huggingface_hub import HfApi

REPO = "hi-todayis-jh/tailrl-text-maze-extra-levels"
ROOT = os.path.expanduser("~/tailrl_maze_runs/checkpoints/tailrl-text-maze")
STAGE = "/work2/jiahaoz4/tmp/maze/rl_stage"
STATE = "/work2/jiahaoz4/maze_extra/rl_janitor_state.json"
HEAVY = re.compile(r"^(model|optim|extra_state)_world_size_\d+_rank_\d+\.pt$")


def log(msg):
    print(f"{time.strftime('%m-%d %H:%M:%S')} {msg}", flush=True)


def dir_bytes(path):
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)


def steps_of(run_dir):
    return sorted(int(d.split("_")[-1]) for d in os.listdir(run_dir) if re.fullmatch(r"global_step_\d+", d))


def latest_of(run_dir):
    try:
        return int(open(os.path.join(run_dir, "latest_checkpointed_iteration.txt")).read().strip())
    except (OSError, ValueError):
        return None


def slim(step_dir, dry):
    """Delete the FSDP shards and dataloader state of an old step; keep actor/huggingface and flops_ledger.json."""
    freed = 0
    actor = os.path.join(step_dir, "actor")
    for f in (os.listdir(actor) if os.path.isdir(actor) else []):
        if HEAVY.match(f):
            p = os.path.join(actor, f)
            freed += os.path.getsize(p)
            if not dry:
                os.remove(p)
    p = os.path.join(step_dir, "data.pt")
    if os.path.exists(p):
        freed += os.path.getsize(p)
        if not dry:
            os.remove(p)
    return freed


def is_slim(step_dir):
    actor = os.path.join(step_dir, "actor")
    return not os.path.exists(os.path.join(step_dir, "data.pt")) and not any(
        HEAVY.match(f) for f in (os.listdir(actor) if os.path.isdir(actor) else []))


def bf16_copy(src, dst):
    import torch
    from safetensors.torch import load_file, save_file
    sd = load_file(src)
    save_file({k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}, dst, metadata={"format": "pt"})


def pack(run_dir, steps, out, whole=False):
    """Tar global_step_N/{actor/huggingface (bf16 weights), flops_ledger.json} (or the whole step dir, as is) with paths
    relative to run_dir."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".bf16.safetensors"
    with tarfile.open(out + ".part", "w") as tf:
        for s in steps:
            sd = f"global_step_{s}"
            if whole:
                tf.add(os.path.join(run_dir, sd), arcname=sd)
                continue
            hf = os.path.join(sd, "actor", "huggingface")
            for f in sorted(os.listdir(os.path.join(run_dir, hf))):
                src = os.path.join(run_dir, hf, f)
                if f == "model.safetensors":
                    bf16_copy(src, tmp)
                    tf.add(tmp, arcname=os.path.join(hf, f))
                else:
                    tf.add(src, arcname=os.path.join(hf, f))
            if os.path.exists(os.path.join(run_dir, sd, "flops_ledger.json")):
                tf.add(os.path.join(run_dir, sd, "flops_ledger.json"), arcname=os.path.join(sd, "flops_ledger.json"))
    if os.path.exists(tmp):
        os.remove(tmp)
    os.replace(out + ".part", out)
    return os.path.getsize(out)


def upload_verified(api, path, path_in_repo, msg):
    size = os.path.getsize(path)
    api.upload_file(path_or_fileobj=path, path_in_repo=path_in_repo, repo_id=REPO, repo_type="dataset", commit_message=msg)
    info = api.get_paths_info(REPO, [path_in_repo], repo_type="dataset")
    if not info or getattr(info[0], "size", None) != size:
        raise RuntimeError(f"Hub size mismatch for {path_in_repo}: local {size}, hub {info[0].size if info else None}")
    return size


def one_pass(api, args, state):
    if not os.path.isdir(args.root):
        return
    for run in sorted(os.listdir(args.root)):
        run_dir = os.path.join(args.root, run)
        # a symlink is a resumed run's alias under another GPU id (the run name embeds it): handled via its target
        if os.path.islink(run_dir) or not os.path.isdir(run_dir) or not re.search(args.runs_regex, run):
            continue
        st = state.setdefault(run, {"archived": [], "final_full": None})
        if not isinstance(st, dict):
            continue
        latest = latest_of(run_dir)
        steps = steps_of(run_dir)
        if latest is None or not steps:
            continue
        finalize = os.path.exists(os.path.join(run_dir, "FINALIZE"))
        keep_full = set(s for s in steps if s >= latest) | set(sorted(s for s in steps if s <= latest)[-2:])
        if finalize:
            keep_full = {latest}
        freed = 0
        for s in steps:
            if s not in keep_full and s < latest:
                freed += slim(os.path.join(run_dir, f"global_step_{s}"), args.dry_run)
        if freed:
            log(f"{run}: slimmed old steps, freed {freed / 1e9:.2f} GB")
        pending = [s for s in steps if s not in keep_full and s < latest and s not in st["archived"]
                   and is_slim(os.path.join(run_dir, f"global_step_{s}"))]
        if finalize and latest not in st["archived"]:
            pending.append(latest)  # the newest step's HF export + ledger go into the last chunk too
        while pending and (len(pending) >= args.chunk or finalize):
            chunk = pending[: args.chunk]
            pending = pending[args.chunk:]
            name = f"global_step_{chunk[0]}-{chunk[-1]}.tar"
            out = os.path.join(STAGE, run, name)
            if args.dry_run:
                log(f"{run}: would pack and upload {len(chunk)} steps as {name}")
                continue
            size = pack(run_dir, chunk, out)
            if state.get("_uploaded_bytes", 0) + size > args.max_repo_gb * 1e9:
                log(f"{run}: QUOTA GUARD: not uploading {name} ({size / 1e9:.2f} GB); keeping it local")
                os.remove(out)
                break
            upload_verified(api, out, f"rl/{run}/{name}", f"{run}: {name}")
            state["_uploaded_bytes"] = state.get("_uploaded_bytes", 0) + size
            for s in chunk:
                if s == latest and finalize:
                    continue  # the newest step stays local until its full upload below
                shutil.rmtree(os.path.join(run_dir, f"global_step_{s}"), ignore_errors=True)
            st["archived"].extend(chunk)
            os.remove(out)
            save_state(state)
            log(f"{run}: uploaded {name} ({size / 1e6:.0f} MB, repo total {state['_uploaded_bytes'] / 1e9:.2f} GB), deleted {len(chunk)} local steps")
        if finalize and st["final_full"] != latest and not args.dry_run:
            name = f"final_full_global_step_{latest}.tar"
            out = os.path.join(STAGE, run, name)
            size = pack(run_dir, [latest], out, whole=True)
            if state.get("_uploaded_bytes", 0) + size <= args.max_repo_gb * 1e9:
                upload_verified(api, out, f"rl/{run}/{name}", f"{run}: {name}")
                state["_uploaded_bytes"] = state.get("_uploaded_bytes", 0) + size
                st["final_full"] = latest
                save_state(state)
                log(f"{run}: uploaded {name} ({size / 1e6:.0f} MB); newest step kept locally as well")
            else:
                log(f"{run}: QUOTA GUARD: final full checkpoint stays local only")
            os.remove(out)


def save_state(state):
    tmp = state["_path"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, state["_path"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-regex", default=r"^textmaze_(p3|p3s1|p03|p03s1|p5|p5s1|p1s1|p01s1)-(s0|l0)_")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--max-repo-gb", type=float, default=38.0, help="stop uploading past this (private quota: 100 GB, 57 GB used)")
    ap.add_argument("--loop", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--state", default=STATE)
    args = ap.parse_args()
    api = HfApi()
    state = json.load(open(args.state)) if os.path.exists(args.state) else {}
    state["_path"] = args.state
    while True:
        try:
            one_pass(api, args, state)
        except Exception:  # network or Hub hiccup: nothing was deleted for the failed chunk; retry next pass
            log("pass failed:\n" + traceback.format_exc())
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
