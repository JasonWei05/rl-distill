# Resume Gemma 4 26B-A4B difficulty runs locally on another cluster

Resumes a seed-42 DeepScaleR 26B-A4B DAPO/GRPO run from its published full checkpoint,
restoring **model weights, Adam optimizer states, RNG + LR scheduler, the exact dataset
position, and the early-stopping state**. Runs directly with a local Ray (no ScaleTrain,
no S3), so it works on any 8-GPU node.

## Published resumable checkpoints

| Band | HuggingFace repo (public) | Step | Best mean@16 (at upload) |
|---|---|---:|---|
| hard | `JWei05/gemma-4-26B-A4B-DeepScaleR-hard-s42-fullckpt-step47` | 47 | 0.211 @ 40 |
| medium | `JWei05/gemma-4-26B-A4B-DeepScaleR-medium-s42-fullckpt-step90` | 90 | 0.656 @ 90 |

Both are FSDP2-sharded at `world_size=8`, so they **must** resume on **exactly 8 GPUs**
(80 GB cards assumed; the memory knobs below target that). The `medium` checkpoint is a
mid-training snapshot — the ScaleTrain run may have progressed further; resuming from it
just replays the steps after `STEP`.

The code path is `run_gemma4_pt_deepscaler_4of4strict_rl.sh` → `gemma3_pt_fewshot_math_rl.sh`
→ `python3 -m dapo.main_dapo`. The wrapper prepares the band dataset itself and, because
`trainer.resume_mode=auto` reads `trainer.default_local_dir` (`CKPTS_DIR`), it resumes from
whatever `global_step_N` you place there.

## Pick a band

Set these three once; every command below uses them.

```bash
export BAND=medium                                                            # medium | hard
export STEP=90                                                                # 90 for medium, 47 for hard
export HF_REPO=JWei05/gemma-4-26B-A4B-DeepScaleR-${BAND}-s42-fullckpt-step${STEP}
```

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

Builds `.venv-gemma4` (torch/vLLM/transformers pinned for Gemma 4) and applies the vLLM R3
router-replay patch. One-time per node; takes a while.

```bash
bash rl-distill-scripts/setup_env_gemma4.sh
```

## 2. Download the checkpoint into the local checkpoint directory

Place it at `<CKPTS_DIR>/global_step_<STEP>/` and write the tracker file `resume_mode=auto`
reads. Use a **persistent** path (not `/tmp`) so resume state survives a reboot.

```bash
export CKPTS_DIR="$HOME/gemma4-26b-${BAND}-s42/ckpts"
mkdir -p "$CKPTS_DIR/global_step_${STEP}"

# All of actor/ (model + optim + extra_state shards, huggingface/), data.pt,
# and validation_early_stopping.json land under global_step_${STEP}/.
.venv-gemma4/bin/huggingface-cli download "$HF_REPO" --local-dir "$CKPTS_DIR/global_step_${STEP}"

# Tell verl this is the latest step to resume from.
echo "$STEP" > "$CKPTS_DIR/latest_checkpointed_iteration.txt"
```

Sanity-check the layout (8 model shards, 8 optim shards, `data.pt`):

```bash
ls "$CKPTS_DIR/global_step_${STEP}"                                   # actor/  data.pt  validation_early_stopping.json
ls "$CKPTS_DIR/global_step_${STEP}/actor" | grep -c model_world_size_8_rank   # -> 8
ls "$CKPTS_DIR/global_step_${STEP}/actor" | grep -c optim_world_size_8_rank   # -> 8
```

## 3. Resume training

Paste this block. It reproduces the exact band training config and resumes locally. It sets
**no** S3 or HF-push targets, so nothing is uploaded; new checkpoints are written under `CKPTS_DIR`.

```bash
export DATA_DIR="$HOME/gemma4-26b-${BAND}-s42/data"   # wrapper writes the prepared band dataset here
mkdir -p "$DATA_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
RAY_ADDRESS=local \
CKPTS_DIR="$CKPTS_DIR" \
DATA_DIR="$DATA_DIR" \
RESUME_MODE=auto \
GEMMA4_MODEL=google/gemma-4-26B-A4B \
GEMMA4_MODEL_REVISION=24548b62aa021d562695c04aaf7758a1ea47990b \
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

- **Continue the original run's curve:** add `WANDB_RUN_ID=g4ds26b-26b-a4b-${BAND}-s42-v1
  WANDB_RESUME=allow`. (W&B enforces monotonic steps, so any step already logged by the old run is
  rejected — the curve continues forward from `STEP`.) Check first that the original ScaleTrain job is
  no longer alive (W&B run state / heartbeat): two live writers on one run id interleave their metrics.
- **Fresh W&B run:** set `EXP_NAME=<your-name>` and omit `WANDB_RUN_ID`.
- **No W&B:** leave `WANDB_API_KEY` out of `.env`.

## 3b. Faster update phase (measured) and local checkpoint cadence

The block above is the original ScaleTrain recipe. On a single 8xH100 node it spends 75% of each step
in `update_actor`, because the FSDP2 CPU offload policy moves parameters and gradients over PCIe for
every one of the 128 single-sequence micro-batches per GPU. Measured on the same step-51 batch from the
step-50 checkpoint (2026-09-03):

| Config | gen | old log-prob | update_actor | step |
|---|---|---|---|---|
| recipe: mbsz 1, 4096 padded cap, `FSDP_CPU_OFFLOAD_POLICY=True`, vLLM resident | 410 s | 109 s | 1625 s | 2157 s |
| mbsz 4, 8192 cap, offload policy on | 416 s | 47 s | 553 s | 1025 s |
| **mbsz 4, 8192 cap, `FSDP_CPU_OFFLOAD_POLICY=False OFFLOAD=True`, vLLM sleep mode** | 267 s | 42 s | **110 s** | **439 s** |

Gradient norms and the vLLM/trainer importance ratio were unchanged across the three. The micro-batch
size is gradient-neutral because the token-mean loss is normalized by the global token count. The
winning row keeps the fp32 master/Adam state on GPU only during the update (one bulk copy per phase)
and puts the 52 GB bf16 vLLM copy to sleep in host RAM during it (`VLLM_SLEEP_MODE=True`, level 1);
peak update memory was 76.8 GB/GPU, bounded by `MAX_PADDED_TOKENS_PER_MICROBATCH`. Fully removing
offload is not possible on 80 GB cards: fp32 master + grads + Adam is 51.6 GB/GPU and a TP=1 vLLM copy
another ~56 GB; sharding vLLM with `GEN_TP=8` made it worse (one engine holding all 1024 sequences
replicates full-vocab logits on every rank, ~49 GB/GPU during generation).

`rl-distill-scripts/resume_gemma4_26b_a4b_local.sh` wraps the recipe with these defaults plus a local
checkpoint cadence (`SAVE_FREQ=2`, `MAX_ACTOR_CKPT_TO_KEEP=2`, validation and HF push every 10 steps). It takes
`BAND=medium|hard`; `resume_gemma4_26b_{medium,hard}_local.sh` are one-line wrappers that set it. Before launching
it checks the venv and the vLLM R3 patch, and refuses to start unless `CUDA_VISIBLE_DEVICES` lists exactly 8 GPUs
and `global_step_<STEP>/` holds 8 model + 8 optim + 8 extra-state shards, `data.pt`, and the early-stopping state.
With `HF_PUSH_ENABLE=True` (the default) it also verifies the `HF_TOKEN` has write access and creates the push repo
before training starts, since run-time pushes are non-fatal and a bad token would otherwise only show up in the log.

```bash
CKPTS_DIR="$HOME/gemma4-26b-${BAND}-s42/ckpts" bash rl-distill-scripts/resume_gemma4_26b_${BAND}_local.sh

# Preflight only: every check above plus model/dataset download and full Hydra config composition (--cfg job),
# without starting Ray or touching a GPU.
DRY_RUN=1 CKPTS_DIR="$HOME/gemma4-26b-${BAND}-s42/ckpts" bash rl-distill-scripts/resume_gemma4_26b_${BAND}_local.sh
```

Per band it reproduces the original ScaleTrain run's experiment name (`RUN_NAME_SUFFIX` is empty for hard and
`26b-bands-es5` for medium), so the W&B display name and the default HF push repo
(`JWei05/DAPO-gemma4-26b-a4b-PT-DeepScaleR-gemma26b-<band>-seed42[-26b-bands-es5]`) match the original run, and
`WANDB_RUN_ID` defaults to `g4ds26b-26b-a4b-<band>-s42-v1` with `WANDB_RESUME=allow`.

Every knob is `${VAR:-default}`; `VLLM_SLEEP_MODE=False FSDP_CPU_OFFLOAD_POLICY=True OFFLOAD=False
MICRO_BATCH_SIZE_PER_GPU=1 MAX_PADDED_TOKENS_PER_MICROBATCH=4096` reproduces the original recipe.

### Continuing after early stopping with a longer patience

A run that stopped on patience `P` leaves a terminal checkpoint whose `validation_early_stopping.json` says
`triggered`. To keep training under a longer patience `Q` (e.g. "stop after three non-improving validations in a
row" instead of two), resume from that checkpoint with both knobs; the saved best score and miss count are kept and
only the trigger flag is recomputed, so the next validation is miss `P+1` of `Q`:

```bash
EARLY_STOPPING_PATIENCE=3 EARLY_STOPPING_MIGRATE_PATIENCE_FROM=2 \
  CKPTS_DIR="$HOME/gemma4-26b-${BAND}-s42/ckpts" bash rl-distill-scripts/resume_gemma4_26b_${BAND}_local.sh
```

The log prints `EARLY_STOPPING_PATIENCE_MIGRATED ... triggered=False` on success. Without the migrate knob the
trainer refuses the mismatched early-stopping config; with it but with the wrong old value it also refuses.

Two operational notes:

- verl prunes only checkpoints written by the current process, so `global_step_*` directories left by
  earlier resumes (~334 GB each) must be deleted by hand.
- HF snapshot pushes are bounded to the newest `HF_PUSH_MAX_TO_KEEP` (default 3): the pusher prunes the
  repo before each upload and squashes history, since deleting LFS pointers alone does not free Hub quota.
  On a free-tier account the Hub rejects commits at the account cap
  (`You have exceeded your public storage space`), and freed space is reflected with delay; pushes are
  non-fatal (`HF_PUSH_REQUIRED=False`) so this cannot fail a finished run. Keep >= 3 so the
  early-stopping best is always still on the Hub (the default keeps `EARLY_STOPPING_PATIENCE + 1` snapshots).

## 4. Confirm the resume actually restored optimizer + dataset position

In the first minutes of the log you should see (not a cold start):

- `global_step` set to `STEP` (training continues at `STEP+1`, not step 1).
- all 8 ranks loading model / optimizer / RNG / LR-scheduler state.
- the StatefulDataLoader restored to its saved cursor (no dataset reshuffle from the top).
- initial validation **skipped on resume**, and early-stopping restored to the checkpoint's best.

If instead you see step 0/1 and a full initial validation, the checkpoint was not found — recheck
that `latest_checkpointed_iteration.txt` contains `STEP` and that `global_step_<STEP>/actor` holds
the 8 model and 8 optim shards.

## 5. What resume restores

| Restored on resume | Source |
|---|---|
| Model weights | `global_step_<STEP>/actor/model_world_size_8_rank_*.pt` |
| Adam optimizer state | `global_step_<STEP>/actor/optim_world_size_8_rank_*.pt` |
| RNG + LR scheduler | `global_step_<STEP>/actor/extra_state_world_size_8_rank_*.pt` |
| Exact dataset position | `global_step_<STEP>/data.pt` (StatefulDataLoader cursor) |
| Early-stopping state | `global_step_<STEP>/validation_early_stopping.json` |

Keep the training recipe identical across the move — same dataset repo/revision, `DATA_SEED=42`,
batch sizes, `N_RESP_PER_PROMPT`, and **8-GPU** layout. Changing any of these invalidates the
dataloader cursor and the optimizer sharding.

## Notes and gotchas

- **8 GPUs, no more, no less.** The shards are keyed `world_size_8`; a different world size cannot
  load them.
- **New checkpoints** are written locally under `CKPTS_DIR/global_step_N`: every `SAVE_FREQ=10` steps with the
  section-3 block, every `SAVE_FREQ=2` steps (newest 2 kept) with the section-3b launcher. Bump
  `MAX_ACTOR_CKPT_TO_KEEP` to retain more.
- **Different GPU model / memory:** the `ROLLOUT_GPU_MEMORY_UTILIZATION`, `VLLM_KV_CACHE_MEMORY_BYTES`,
  and `MAX_PADDED_TOKENS_PER_MICROBATCH` values target 80 GB cards; lower them for smaller cards.
- **R3 router replay** needs the patched vLLM from `setup_env_gemma4.sh` (step 1); do not skip it.
- Each checkpoint also lives in S3 at
  `s3://scale-ml/genai/rl-distill/gemma4-difficulty-s42-20260819-full-checkpoints/26b-a4b-${BAND}/global_step_${STEP}`.
  If the new cluster can reach that bucket with AWS credentials, point `FULL_CHECKPOINT_S3_URI` at
  `.../26b-a4b-${BAND}` instead of step 2's manual download and the wrapper pulls it automatically.
