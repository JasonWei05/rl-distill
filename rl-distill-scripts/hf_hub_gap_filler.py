"""Keep the Hub in sync with local checkpoints: upload any of the newest KEEP completed local
checkpoint exports that are missing on the Hub, then prune Hub step folders older than the newest
KEEP local steps. Independent of the trainer's own pusher (which can lose a step to a transient
upload failure). Safe to run repeatedly."""
import glob
import os
import time

from huggingface_hub import HfApi, delete_folder, upload_folder

# Env knobs: GAP_FILLER_BANDS="medium hard", GAP_FILLER_ROOT, GAP_FILLER_KEEP, GAP_FILLER_NAMESPACE,
# GAP_FILLER_SUFFIX (repo name suffix, default local2gpu), GAP_FILLER_SEED.
KEEP = int(os.environ.get("GAP_FILLER_KEEP", "5"))
ROOT = os.path.expanduser(os.environ.get("GAP_FILLER_ROOT", "~/gemma4-e2b-difficulty-s42"))
_NS = os.environ.get("GAP_FILLER_NAMESPACE", "JWei05")
_SEED = os.environ.get("GAP_FILLER_SEED", "42")
_SUFFIX = os.environ.get("GAP_FILLER_SUFFIX", "local2gpu")
REPOS = {
    b: f"{_NS}/DAPO-gemma4-e2b-PT-DeepScaleR-gemma26b-{b}-seed{_SEED}-{_SUFFIX}"
    for b in os.environ.get("GAP_FILLER_BANDS", "medium hard").split()
}
api = HfApi()
def completed_steps(band):
    ck = f"{ROOT}/{band}/ckpts"
    try: latest = int(open(f"{ck}/latest_checkpointed_iteration.txt").read().strip())
    except Exception: return []
    steps = []
    for d in glob.glob(f"{ck}/global_step_*"):
        n = int(d.rsplit("_", 1)[1]); hf = f"{d}/actor/huggingface/model.safetensors"
        # only fully written exports: step recorded as checkpointed and export older than 10 min
        if n <= latest and os.path.exists(hf) and time.time() - os.path.getmtime(hf) > 600:
            steps.append(n)
    return sorted(steps)
for band, repo in REPOS.items():
    steps = completed_steps(band)
    want = steps[-KEEP:]
    try: have = sorted({f.split("/")[0] for f in api.list_repo_files(repo) if f.startswith("step_")})
    except Exception as e: print(f"[{band}] list failed: {e}", flush=True); continue
    have_steps = sorted(int(h.split("_")[1]) for h in have)
    missing = [s for s in want if s not in have_steps]
    print(f"[{band}] local completed={steps} hub={have_steps} missing(newest {KEEP})={missing}", flush=True)
    for s in missing:
        src = f"{ROOT}/{band}/ckpts/global_step_{s}/actor/huggingface"
        for attempt in range(1, 6):
            try:
                upload_folder(folder_path=src, path_in_repo=f"step_{s:06d}", repo_id=repo, repo_type="model", commit_message=f"step {s} (gap filler)")
                print(f"[{band}] uploaded step {s}", flush=True); break
            except Exception as e:
                print(f"[{band}] step {s} attempt {attempt} failed: {e}", flush=True); time.sleep(min(2**attempt, 120))
    # prune: anything on the Hub older than the newest KEEP local steps (only when there are more than KEEP)
    if len(want) == KEEP:
        for h in have_steps:
            if h < want[0]:
                try:
                    delete_folder(path_in_repo=f"step_{h:06d}", repo_id=repo, repo_type="model", commit_message=f"prune step_{h:06d} (gap filler)")
                    print(f"[{band}] pruned step {h}", flush=True)
                except Exception as e: print(f"[{band}] prune {h} failed: {e}", flush=True)
