# Run Gemma 4 E2B difficulty RL (easy / medium / hard) locally on another cluster

Trains `google/gemma-4-E2B` from scratch with DAPO/GRPO on each of the three
DeepScaleR difficulty bands, one run per band, **4 GPUs each**, directly with a
local Ray (no ScaleTrain). This reproduces the seed-42 E2B difficulty sweep and
keeps a **permanent checkpoint every 10 steps** so the best-step weights are
retained.

- **Model:** `google/gemma-4-E2B` (dense), revision `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`
- **Data:** `JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k`, bands easy/medium/hard
  (3,000 train + 300 validation questions each)
- **Per band:** 4 GPUs (FSDP2 DP-4, TP-1), 80 GB cards assumed
- The three bands are independent — run them one at a time on a 4-GPU node, or all
  three at once if you have 12 GPUs (see "Running all three").

The code path is `run_gemma4_pt_deepscaler_4of4strict_rl.sh` → `gemma3_pt_fewshot_math_rl.sh`
→ `python3 -m dapo.main_dapo`. The wrapper downloads the base model, prepares the band
dataset, and trains.

## 0. Prerequisites

- A node with **4** GPUs (80 GB each), CUDA, git, `git-lfs`.
- Clone this repo and add a repo-root `.env` (untracked) with `HF_TOKEN=...` and,
  optionally, `WANDB_API_KEY=...`.

```bash
git clone git@github.com:JasonWei05/rl-distill.git
cd rl-distill
printf 'HF_TOKEN=hf_xxx\nWANDB_API_KEY=xxx\n' > .env   # WANDB optional
```

## 1. Build the Gemma 4 environment

Builds `.venv-gemma4` (torch/vLLM/transformers pinned for Gemma 4). One-time per node.

```bash
bash rl-distill-scripts/setup_env_gemma4.sh
```

## 2. Train one band

Pick a band and paste the block. It trains from the base model, saves a full
checkpoint (model + Adam + dataloader cursor) to a local directory every 10 steps,
validates every 10 steps, and early-stops after 5 non-improving validations (max 400
steps). Nothing is uploaded.

```bash
export BAND=easy                                    # easy | medium | hard
export CKPTS_DIR="$HOME/gemma4-e2b-${BAND}-s42/ckpts"
export DATA_DIR="$HOME/gemma4-e2b-${BAND}-s42/data"
mkdir -p "$CKPTS_DIR" "$DATA_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
RAY_ADDRESS=local \
CKPTS_DIR="$CKPTS_DIR" \
DATA_DIR="$DATA_DIR" \
GEMMA4_MODEL=google/gemma-4-E2B \
GEMMA4_MODEL_REVISION=d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f \
DIFFICULTY_DATASET_SOURCE=gemma4_26b_bands \
DIFFICULTY_DATASET="$BAND" \
DIFFICULTY_DATASET_REPO=JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k \
DIFFICULTY_DATASET_REVISION=a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db \
DATA_SEED=42 \
TOTAL_TRAINING_STEPS=400 \
TRAIN_PROMPT_BSZ=64 GEN_PROMPT_BSZ=64 N_RESP_PER_PROMPT=16 TRAIN_PROMPT_MINI_BSZ=32 \
ACTOR_LR=1e-6 ACTOR_LR_WARMUP_STEPS=20 \
MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=8192 MAX_MODEL_LEN=12288 \
OVERLONG_BUFFER_LEN=2048 ENABLE_OVERLONG_BUFFER=True OVERLONG_PENALTY_FACTOR=1.0 \
MICRO_BATCH_SIZE_PER_GPU=8 MAX_PADDED_TOKENS_PER_MICROBATCH=12288 \
SP_SIZE=1 GEN_TP=1 ACTOR_FSDP_SIZE=-1 \
FSDP_CPU_OFFLOAD_POLICY=True OFFLOAD=False \
ROUTER_REPLAY_MODE=disabled ROUTER_Z_LOSS_COEF=0.0 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.25 VLLM_KV_CACHE_MEMORY_BYTES=536870912 \
ROLLOUT_ENFORCE_EAGER=False VLLM_DISABLE_COMPILE_CACHE=0 \
TEST_FREQ=10 SAVE_FREQ=10 MAX_ACTOR_CKPT_TO_KEEP=100 \
ROLLING_CHECKPOINT_ENABLED=False HF_PUSH_ENABLE=False HF_PUSH_REQUIRED=False \
EARLY_STOPPING_ENABLED=True EARLY_STOPPING_METRIC='val-core/math/acc/mean@16' \
EARLY_STOPPING_MODE=max EARLY_STOPPING_PATIENCE=5 EARLY_STOPPING_MIN_DELTA=0.0 \
EARLY_STOPPING_INCLUDE_INITIAL_VALIDATION=True \
LOG_VAL_GENERATIONS=100 LOG_TRAIN_GENERATIONS=100 \
VENV="$PWD/.venv-gemma4" \
  bash rl-distill-scripts/scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh
```

Run the block again with `BAND=medium` and `BAND=hard` for the other two.

### Weights & Biases (optional)

Add `EXP_NAME=g4-e2b-${BAND}-s42-local` to name the run, or leave `WANDB_API_KEY`
out of `.env` to disable W&B.

## 3. Checkpoints

- A full checkpoint lands at `$CKPTS_DIR/global_step_{10,20,30,...}/` every 10 steps:
  `actor/` (FSDP2 model + Adam shards + a consolidated `huggingface/model.safetensors`
  inference export) and `data.pt` (dataset cursor).
- `MAX_ACTOR_CKPT_TO_KEEP=100` retains every 10-step checkpoint (40 for a 400-step run).
  Each E2B full checkpoint is ~68 GB, so 40 of them is ~2.7 TB. Lower it if disk is
  tight; the per-checkpoint `actor/huggingface/model.safetensors` (~10 GB) is the
  weight-only inference model if you only need weights, not resume state.
- The best step is the one whose validation `val-core/math/acc/mean@16` is highest;
  it is printed at each validation and recorded in `validation_early_stopping.json`.

## 4. Running all three at once (12 GPUs)

If the node has 12 GPUs, run three copies of the block concurrently, each pinned to a
disjoint 4-GPU slice and its own directories and vLLM port base:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3   VERL_VLLM_PORT_BASE=52000 ... BAND=easy   ... &
CUDA_VISIBLE_DEVICES=4,5,6,7   VERL_VLLM_PORT_BASE=54000 ... BAND=medium ... &
CUDA_VISIBLE_DEVICES=8,9,10,11 VERL_VLLM_PORT_BASE=56000 ... BAND=hard   ... &
```

Give each its own `CKPTS_DIR`, `DATA_DIR`, `RAY_TEMP_DIR`, and `VERL_VLLM_PORT_BASE`
so they don't collide.

## 5. Optional: durable checkpoints to S3 (the full rolling infra)

If the cluster can reach `s3://scale-ml` with AWS credentials and you want the
production infra — a rolling "latest step always" checkpoint plus the permanent
every-10 checkpoints mirrored to S3, with seamless resume after a crash — set these
instead of the local-only checkpoint knobs above:

```bash
  FULL_CHECKPOINT_S3_URI=s3://<your-bucket>/gemma4-e2b-${BAND}-s42/full-checkpoints \
  RUN_ARTIFACT_S3_URI=s3://<your-bucket>/gemma4-e2b-${BAND}-s42/artifacts \
  WANDB_RUN_ID=g4-e2b-${BAND}-s42-local \
  ROLLING_CHECKPOINT_ENABLED=True ROLLING_CHECKPOINT_FREQ=1 MAX_ACTOR_CKPT_TO_KEEP=1 \
```

With `RESUME_MODE=auto` (the default) the wrapper then restores the latest S3
checkpoint automatically on restart, resuming model, Adam, and dataset position.

## Notes

- **4 GPUs, dense model.** FSDP2 shards E2B across the 4 GPUs (`ACTOR_FSDP_SIZE=-1`),
  with CPU offload on. `ROUTER_REPLAY_MODE=disabled` because E2B is not a MoE.
- Keep the recipe identical to reproduce the sweep: same dataset repo/revision,
  `DATA_SEED=42`, batch sizes, `N_RESP_PER_PROMPT=16`, 4-GPU layout.
- Different GPU memory: lower `ROLLOUT_GPU_MEMORY_UTILIZATION` /
  `VLLM_KV_CACHE_MEMORY_BYTES` / `MAX_PADDED_TOKENS_PER_MICROBATCH` for smaller cards.
