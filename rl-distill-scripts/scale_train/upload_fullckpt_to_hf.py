#!/usr/bin/env python3
"""Stage a full RL checkpoint from S3 and push it to the Hub as a resumable
artifact (model + Adam optimizer + RNG/scheduler extra-state + dataloader cursor),
so the run can be resumed on a different cluster.

Generic over run/step/repo. Example (26B-A4B medium, permanent step 80):

  python upload_fullckpt_to_hf.py \
    --checkpoint-s3-uri s3://scale-ml/genai/rl-distill/gemma4-difficulty-s42-20260819-full-checkpoints/26b-a4b-medium/global_step_80 \
    --repo-id JWei05/gemma-4-26B-A4B-DeepScaleR-medium-s42-fullckpt-step80 \
    --base-model google/gemma-4-26B-A4B --band medium --step 80 \
    --best-score 0.6492 --best-step 80 --world-size 8

Resume contract (see CLAUDE.md): the checkpoint is FSDP2-sharded, so resume on the
same GPU count / FSDP layout via
`RESUME_MODE=resume_path trainer.resume_from_path=<dir>/global_step_<step>`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _load_hf_token() -> str:
    for line in Path("/mnt/efs/jasonwei/rl-distill/.env").read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("HF_TOKEN not found in .env")


def _readme(base_model: str, band: str, step: int, world: int, best_score, best_step) -> str:
    best_line = (
        f"- Best mean@16 so far: **{best_score}** at step {best_step}."
        if best_score is not None
        else ""
    )
    return f"""---
license: gemma
base_model: {base_model}
tags:
  - verl
  - dapo
  - grpo
  - full-checkpoint
  - resumable
---

# {base_model} — DeepScaleR {band} band — seed 42 — full resumable checkpoint (global step {step})

DAPO/GRPO RL full checkpoint (verl FSDP2 fork). Not inference-only: it holds
everything needed to resume training on another cluster.

- `actor/model_world_size_{world}_rank_*.pt` — FSDP2-sharded model weights.
- `actor/optim_world_size_{world}_rank_*.pt` — Adam optimizer state shards.
- `actor/extra_state_world_size_{world}_rank_*.pt` — RNG and LR-scheduler state.
- `actor/huggingface/` — tokenizer, config, and a consolidated `model.safetensors`.
- `data.pt` — StatefulDataLoader cursor (exact dataset position).
- `validation_early_stopping.json` — early-stopping state.
{best_line}

## Resume

Sharded at `world_size={world}`, so resume on **{world} GPUs** with the same FSDP
layout, keeping data paths, shuffle seed, batch size, and response count unchanged:

```
RESUME_MODE=resume_path trainer.resume_from_path=<local_dir>/global_step_{step}
```
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-s3-uri", required=True, help="S3 URI of the global_step_N directory")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--base-model", default="google/gemma-4-26B-A4B")
    ap.add_argument("--band", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--best-score", type=float, default=None)
    ap.add_argument("--best-step", type=int, default=None)
    ap.add_argument("--stage-dir", default=None)
    args = ap.parse_args()

    token = _load_hf_token()
    stage = Path(args.stage_dir or f"/tmp/hf-fullckpt-{args.repo_id.split('/')[-1]}") / f"global_step_{args.step}"
    stage.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(AWS_PROFILE="ml-worker", AWS_DEFAULT_REGION="us-west-2", AWS_REGION="us-west-2")

    print(f"[1/3] downloading {args.checkpoint_s3_uri} -> {stage}", flush=True)
    subprocess.run(
        ["aws", "s3", "cp", "--recursive", "--only-show-errors", args.checkpoint_s3_uri.rstrip("/"), str(stage)],
        check=True,
        env=env,
    )
    (stage / "README.md").write_text(
        _readme(args.base_model, args.band, args.step, args.world_size, args.best_score, args.best_step)
    )
    total = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file())
    print(f"    staged {total/1e9:.1f} GB", flush=True)

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"[2/3] creating/ensuring public repo {args.repo_id}", flush=True)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=False, exist_ok=True)
    api.update_repo_settings(repo_id=args.repo_id, repo_type="model", private=False)

    print(f"[3/3] uploading {stage} -> {args.repo_id}", flush=True)
    api.upload_large_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(stage), print_report=True)
    print("DONE: upload complete", flush=True)


if __name__ == "__main__":
    main()
