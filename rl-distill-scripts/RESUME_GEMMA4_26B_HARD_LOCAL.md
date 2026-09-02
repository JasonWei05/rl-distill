# Resume Gemma 4 26B-A4B (hard band) locally on another cluster

This resumes the seed-42 DeepScaleR **hard**-band DAPO/GRPO run from the published full
checkpoint, restoring **model weights, Adam optimizer states, RNG + LR scheduler, the exact
dataset position, and the early-stopping state**. It runs directly with a local Ray (no
ScaleTrain, no S3), so it works on any 8-GPU node.

- **Checkpoint (public):** `JWei05/gemma-4-26B-A4B-DeepScaleR-hard-s42-fullckpt-step47`
- **Resumes from:** global step 47 (best mean@16 so far `0.211` at step 40)
- **Hard requirement:** the checkpoint is FSDP2-sharded at `world_size=8`, so it **must** resume
  on **exactly 8 GPUs** with the same FSDP layout. 80 GB cards (H100/B200) are assumed; the memory
  knobs below target that.

The code path is `run_gemma4_pt_deepscaler_4of4strict_rl.sh` → `gemma3_pt_fewshot_math_rl.sh` →
`python3 -m dapo.main_dapo`. The wrapper prepares the hard dataset itself and, because
`trainer.resume_mode=auto` reads `trainer.default_local_dir` (`CKPTS_DIR`), it resumes from
whatever `global_step_N` you place there.

## 0. Prerequisites

- A single node with **8** GPUs (80 GB each), CUDA, git, and `git-lfs`.
- Clone this repo and create a repo-root `.env` (untracked) with `HF_TOKEN=...` and, optionally,
  `WANDB_API_KEY=...`.

```bash
git clone git@github.com:JasonWei05/rl-distill.git
cd rl-distill
printf 'HF_TOKEN=hf_xxx\nWANDB_API_KEY=xxx\n' > .env   # WANDB optional
```

## 1. Build the Gemma 4 environment

This builds `.venv-gemma4` (torch/vLLM/transformers pinned for Gemma 4) and applies the vLLM R3
router-replay patch. One-time per node; takes a while.

```bash
bash rl-distill-scripts/setup_env_gemma4.sh
```

## 2. Download the checkpoint into the local checkpoint directory

Place the checkpoint at `<CKPTS_DIR>/global_step_47/` and write the tracker file that
`resume_mode=auto` reads. Use a **persistent** path (not `/tmp`) so resume state survives a reboot.

```bash
export CKPTS_DIR="$HOME/gemma4-26b-hard-s42/ckpts"
mkdir -p "$CKPTS_DIR/global_step_47"

# All of actor/ (model + optim + extra_state shards, huggingface/), data.pt,
# and validation_early_stopping.json land under global_step_47/.
.venv-gemma4/bin/huggingface-cli download \
  JWei05/gemma-4-26B-A4B-DeepScaleR-hard-s42-fullckpt-step47 \
  --local-dir "$CKPTS_DIR/global_step_47"

# Tell verl this is the latest step to resume from.
echo 47 > "$CKPTS_DIR/latest_checkpointed_iteration.txt"
```

Sanity-check the layout (8 model shards, 8 optim shards, `data.pt`):

```bash
ls "$CKPTS_DIR/global_step_47"                      # actor/  data.pt  validation_early_stopping.json
ls "$CKPTS_DIR/global_step_47/actor" | grep -c model_world_size_8_rank   # -> 8
ls "$CKPTS_DIR/global_step_47/actor" | grep -c optim_world_size_8_rank   # -> 8
```

## 3. Resume training

Paste this block. It reproduces the exact hard-band training config and resumes locally. It sets
**no** S3 or HF-push targets, so nothing is uploaded; new checkpoints are written under `CKPTS_DIR`.

```bash
export DATA_DIR="$HOME/gemma4-26b-hard-s42/data"     # wrapper writes the prepared hard dataset here
mkdir -p "$DATA_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
RAY_ADDRESS=local \
CKPTS_DIR="$CKPTS_DIR" \
DATA_DIR="$DATA_DIR" \
RESUME_MODE=auto \
GEMMA4_MODEL=google/gemma-4-26B-A4B \
GEMMA4_MODEL_REVISION=24548b62aa021d562695c04aaf7758a1ea47990b \
DIFFICULTY_DATASET_SOURCE=gemma4_26b_bands \
DIFFICULTY_DATASET=hard \
DIFFICULTY_DATASET_REPO=JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k \
DIFFICULTY_DATASET_REVISION=a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db \
DATA_SEED=42 \
TOTAL_TRAINING_STEPS=400 \
TRAIN_PROMPT_BSZ=64 GEN_PROMPT_BSZ=64 N_RESP_PER_PROMPT=16 TRAIN_PROMPT_MINI_BSZ=32 \
ACTOR_LR=1e-6 ACTOR_LR_WARMUP_STEPS=20 \
MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=8192 MAX_MODEL_LEN=12288 \
OVERLONG_BUFFER_LEN=2048 ENABLE_OVERLONG_BUFFER=True OVERLONG_PENALTY_FACTOR=1.0 \
MICRO_BATCH_SIZE_PER_GPU=1 MAX_PADDED_TOKENS_PER_MICROBATCH=4096 \
SP_SIZE=1 GEN_TP=1 ACTOR_FSDP_SIZE=-1 \
FSDP_CPU_OFFLOAD_POLICY=True OFFLOAD=False \
ROUTER_REPLAY_MODE=R3 ROUTER_Z_LOSS_COEF=0.0 VERL_SKIP_VLLM_MM_WEIGHT_RELOAD=1 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.10 VLLM_KV_CACHE_MEMORY_BYTES=3221225472 \
ROLLOUT_ENFORCE_EAGER=True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TEST_FREQ=10 SAVE_FREQ=10 \
ROLLING_CHECKPOINT_ENABLED=False HF_PUSH_ENABLE=False HF_PUSH_REQUIRED=False \
EARLY_STOPPING_ENABLED=True EARLY_STOPPING_METRIC='val-core/math/acc/mean@16' \
EARLY_STOPPING_MODE=max EARLY_STOPPING_PATIENCE=2 EARLY_STOPPING_MIN_DELTA=0.0 \
EARLY_STOPPING_INCLUDE_INITIAL_VALIDATION=True \
LOG_VAL_GENERATIONS=100 LOG_TRAIN_GENERATIONS=100 \
VENV="$PWD/.venv-gemma4" \
  bash rl-distill-scripts/scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh
```

### Weights & Biases (optional)

- **Continue the original run's curve:** add `WANDB_RUN_ID=g4ds26b-26b-a4b-hard-s42-v1
  WANDB_RESUME=allow` to the block. (Note: W&B enforces monotonic steps, so any step already logged
  by the old run will be rejected — the curve continues forward from step 47.)
- **Fresh W&B run:** set `EXP_NAME=<your-name>` and omit `WANDB_RUN_ID`.
- **No W&B:** leave `WANDB_API_KEY` out of `.env`.

## 4. Confirm the resume actually restored optimizer + dataset position

In the first minutes of the log you should see (not a cold start):

- `global_step` set to `47` (training continues at step 48, not step 1).
- all 8 ranks loading model / optimizer / RNG / LR-scheduler state.
- the StatefulDataLoader restored to its saved cursor (no dataset reshuffle from the top).
- initial validation **skipped on resume**, and early-stopping restored to best `0.211` at step 40.

If instead you see step 0/1 and a full initial validation, the checkpoint was not found — recheck
that `latest_checkpointed_iteration.txt` contains `47` and that `global_step_47/actor` holds the 8
model and 8 optim shards.

## 5. What resume restores (and what it does not)

| Restored on resume | Source |
|---|---|
| Model weights | `global_step_47/actor/model_world_size_8_rank_*.pt` |
| Adam optimizer state | `global_step_47/actor/optim_world_size_8_rank_*.pt` |
| RNG + LR scheduler | `global_step_47/actor/extra_state_world_size_8_rank_*.pt` |
| Exact dataset position | `global_step_47/data.pt` (StatefulDataLoader cursor) |
| Early-stopping state | `global_step_47/validation_early_stopping.json` |

Keep the training recipe identical across the move — same dataset repo/revision, `DATA_SEED=42`,
batch sizes, `N_RESP_PER_PROMPT`, and **8-GPU** layout. Changing any of these invalidates the
dataloader cursor and the optimizer sharding.

## Notes and gotchas

- **8 GPUs, no more, no less.** The shards are keyed `world_size_8`; a different world size cannot
  load them.
- **New checkpoints** are written locally under `CKPTS_DIR/global_step_{50,60,...}` every
  `SAVE_FREQ=10` steps. Bump `MAX_ACTOR_CKPT_TO_KEEP` (default keeps only the newest) if you want to
  retain more than one.
- **Different GPU model / memory:** the `ROLLOUT_GPU_MEMORY_UTILIZATION`, `VLLM_KV_CACHE_MEMORY_BYTES`,
  and `MAX_PADDED_TOKENS_PER_MICROBATCH` values target 80 GB cards; lower them for smaller cards.
- **R3 router replay** needs the patched vLLM from `setup_env_gemma4.sh` (step 1); do not skip it.
- The checkpoint also lives in S3 at
  `s3://scale-ml/genai/rl-distill/gemma4-difficulty-s42-20260819-full-checkpoints/26b-a4b-hard/rolling/global_step_47`.
  If the new cluster can reach that bucket with AWS credentials, you can point
  `FULL_CHECKPOINT_S3_URI` at `.../26b-a4b-hard` instead of steps 2's manual download and the wrapper
  will pull it automatically.
