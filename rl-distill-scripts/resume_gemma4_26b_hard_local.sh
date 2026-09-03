#!/usr/bin/env bash
# Resume the Gemma 4 26B-A4B hard-band DAPO/GRPO run (seed 42) from the latest local checkpoint on a
# single 8x80GB node, with the update-phase configuration measured fastest on 8xH100 (2026-09-03).
#
# Companion to RESUME_GEMMA4_26B_HARD_LOCAL.md (which documents the checkpoint download and what resume
# restores). Everything below marked OVERRIDABLE is `${VAR:-default}`; the rest is the pinned recipe.
#
#   CKPTS_DIR=$HOME/gemma4-26b-hard-s42/ckpts bash rl-distill-scripts/resume_gemma4_26b_hard_local.sh
#
# Measured per-step timings on the same step-51 batch (8xH100 80GB):
#   recipe  (mbsz 1, 4096 padded cap, FSDP2 CPU offload policy, vLLM resident): gen 410 s, update 1625 s, step 2157 s
#   this    (mbsz 4, 8192 padded cap, phase-level OFFLOAD=True, vLLM sleep mode): gen 267 s, update  110 s, step  439 s
# Peak update-phase memory 76.8 GB/GPU; activations are bounded by MAX_PADDED_TOKENS_PER_MICROBATCH so the peak
# is stable across batches. token-mean loss is normalized by the GLOBAL token count, so the micro-batch size does
# not change the gradient (only bf16 summation order).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
export PATH="${HOME}/.local/bin:${PATH}"
if [ -f .env ]; then set -a; source .env; set +a; fi

# --- node-local paths (OVERRIDABLE) -------------------------------------------------------------------------
export CKPTS_DIR="${CKPTS_DIR:-${HOME}/gemma4-26b-hard-s42/ckpts}"   # holds global_step_N/ + latest_checkpointed_iteration.txt
export DATA_DIR="${DATA_DIR:-${HOME}/gemma4-26b-hard-s42/data}"
export HF_HOME="${HF_HOME:-${HOME}/hf_cache}"                          # persistent (the wrapper defaults to /tmp)
export RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_26b_hard_s42}"
export VENV="${VENV:-${REPO}/.venv-gemma4}"
mkdir -p "${DATA_DIR}"
test -s "${CKPTS_DIR}/latest_checkpointed_iteration.txt" || { echo "FATAL: no tracker file in ${CKPTS_DIR}" >&2; exit 2; }
echo "RESUME_FROM_STEP=$(cat "${CKPTS_DIR}/latest_checkpointed_iteration.txt") CKPTS_DIR=${CKPTS_DIR}"

# Single node: loopback NCCL/GLOO (the wrapper defaults to eth0/AF_INET6 for ScaleTrain pods).
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}" NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

# --- W&B: the trainer hard-fails at wandb.init without a key. Continue the original curve when a key is present;
# otherwise (or with WANDB_DISABLE=1) log to the console only.
EXTRA_OVERRIDES=()
if [ -n "${WANDB_API_KEY:-}" ] && [ "${WANDB_DISABLE:-0}" != 1 ]; then
  export WANDB_RUN_ID="${WANDB_RUN_ID:-g4ds26b-26b-a4b-hard-s42-v1}" WANDB_RESUME="${WANDB_RESUME:-allow}"
else
  echo "NOTE: wandb disabled or no WANDB_API_KEY -> console logger only"
  EXTRA_OVERRIDES+=("trainer.logger=[\"console\"]")
fi

# --- pinned recipe ---------------------------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" RAY_ADDRESS=local RESUME_MODE=auto
export GEMMA4_MODEL=google/gemma-4-26B-A4B
export GEMMA4_MODEL_REVISION=24548b62aa021d562695c04aaf7758a1ea47990b
export DIFFICULTY_DATASET_SOURCE=gemma4_26b_bands DIFFICULTY_DATASET=hard
export DIFFICULTY_DATASET_REPO=JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k
export DIFFICULTY_DATASET_REVISION=a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db
export DATA_SEED=42
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}"
export TRAIN_PROMPT_BSZ=64 GEN_PROMPT_BSZ=64 N_RESP_PER_PROMPT=16 TRAIN_PROMPT_MINI_BSZ=32
export ACTOR_LR=1e-6 ACTOR_LR_WARMUP_STEPS=20
export MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=8192 MAX_MODEL_LEN=12288
export OVERLONG_BUFFER_LEN=2048 ENABLE_OVERLONG_BUFFER=True OVERLONG_PENALTY_FACTOR=1.0
export SP_SIZE=1 GEN_TP="${GEN_TP:-1}" ACTOR_FSDP_SIZE=-1
export ROUTER_REPLAY_MODE=R3 ROUTER_Z_LOSS_COEF=0.0 VERL_SKIP_VLLM_MM_WEIGHT_RELOAD=1
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.10 VLLM_KV_CACHE_MEMORY_BYTES=3221225472
export ROLLOUT_ENFORCE_EAGER=True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EARLY_STOPPING_ENABLED="${EARLY_STOPPING_ENABLED:-True}" EARLY_STOPPING_METRIC='val-core/math/acc/mean@16'
export EARLY_STOPPING_MODE=max EARLY_STOPPING_PATIENCE=2 EARLY_STOPPING_MIN_DELTA=0.0
export EARLY_STOPPING_INCLUDE_INITIAL_VALIDATION=True
export LOG_VAL_GENERATIONS=100 LOG_TRAIN_GENERATIONS=100

# --- update-phase throughput (OVERRIDABLE; defaults = the measured-fastest configuration above) --------------
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}"
export MAX_PADDED_TOKENS_PER_MICROBATCH="${MAX_PADDED_TOKENS_PER_MICROBATCH:-8192}"
export FSDP_CPU_OFFLOAD_POLICY="${FSDP_CPU_OFFLOAD_POLICY:-False}"   # per-layer FSDP2 CPU offload: every microbatch pays a full param round trip
export OFFLOAD="${OFFLOAD:-True}"                                    # verl phase-level offload: one bulk copy per phase instead
export VLLM_SLEEP_MODE="${VLLM_SLEEP_MODE:-True}"                    # park the 52 GB bf16 vLLM copy in host RAM during the update

# --- checkpoint cadence (OVERRIDABLE) --------------------------------------------------------------------------
# Full resumable checkpoint every 2 steps (~334 GB, ~4 min), newest 2 kept; validation + HF snapshot push every
# 10 steps. verl only prunes checkpoints written by the current process: directories left by earlier resumes
# must be removed by hand. HF_PUSH_FREQ must be a multiple of SAVE_FREQ.
export TEST_FREQ="${TEST_FREQ:-10}" SAVE_FREQ="${SAVE_FREQ:-2}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
export ROLLING_CHECKPOINT_ENABLED=False
# HF Hub snapshots (weights-only, ~53 GB each) go to <repo>/step_0000NN. The pusher prunes the repo to
# HF_PUSH_MAX_TO_KEEP-1 before each upload and squashes history so quota is really released (deleting LFS pointers
# alone frees nothing). Keep >= 3: early stopping (patience 2 validations) guarantees the best snapshot is among
# the newest 3. HF_PUSH_REQUIRED=False so a Hub storage-quota failure cannot fail a finished training run.
export HF_PUSH_ENABLE="${HF_PUSH_ENABLE:-True}" HF_PUSH_FREQ="${HF_PUSH_FREQ:-10}"
export HF_PUSH_MAX_TO_KEEP="${HF_PUSH_MAX_TO_KEEP:-3}"
export HF_PUSH_REQUIRED="${HF_PUSH_REQUIRED:-False}" HF_PUSH_DELETE_LOCAL_AFTER=False
export HF_PUSH_REPO="${HF_PUSH_REPO:-JWei05/DAPO-gemma4-26b-a4b-PT-DeepScaleR-gemma26b-hard-seed42}"
if [ $((HF_PUSH_FREQ % SAVE_FREQ)) -ne 0 ]; then
  echo "FATAL: HF_PUSH_FREQ=${HF_PUSH_FREQ} must be a multiple of SAVE_FREQ=${SAVE_FREQ}" >&2
  exit 2
fi

echo "KNOBS sleep_mode=${VLLM_SLEEP_MODE} gen_tp=${GEN_TP} mbsz=${MICRO_BATCH_SIZE_PER_GPU} padded_cap=${MAX_PADDED_TOKENS_PER_MICROBATCH} offload_policy=${FSDP_CPU_OFFLOAD_POLICY} offload=${OFFLOAD} save_freq=${SAVE_FREQ} keep=${MAX_ACTOR_CKPT_TO_KEEP} hf_push=${HF_PUSH_ENABLE}/${HF_PUSH_FREQ}/keep${HF_PUSH_MAX_TO_KEEP}"
exec bash rl-distill-scripts/scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh "${EXTRA_OVERRIDES[@]}" "$@"
