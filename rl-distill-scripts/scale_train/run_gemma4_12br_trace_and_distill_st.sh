#!/usr/bin/env bash
# ScaleTrain pod entry for §9.0j(a): the RESUMED untrained-12B medium RL teacher (best step 140, 4-rank S3 prefix) -> E4B base.
# Stage 1: top-128 trace collection (spec 12br-medium: 3000 train prompts x 16 + 300 val x 1) with every visible GPU as a
#          DP worker (TP=1), mirrored per shard to S3 (resumable; skipped entirely once the S3 bundle has COMPLETE.json).
# Stage 2: run_gemma4_distill_one.sh with the §9 recipe (bs 128, lr 2e-6 -> 2e-7 linear, 100 warm-up, 1000 steps, val every 10),
#          S3-only checkpoints (permanent every 250, rolling every 50 with the HF export kept under hf_exports/), no Hub pushes.
# Preemption: both stages resume (shards on S3; the trainer restores the newest complete S3 checkpoint). Launched by
# launch_gemma4_12br_trace_distill_st.sh (4 GPUs, borrowing, priority high) via launch_st_with_code.sh (git archive HEAD).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${PROJECT_ROOT}"
export PATH="${PROJECT_ROOT}/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
echo "ST_12BR_START $(date -u +%FT%TZ) host=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null || echo baked)"
if [ -n "${CODE_S3_URI:-}" ]; then
  echo "### refreshing repo code from ${CODE_S3_URI}"
  AWS_BIN="$(command -v aws || echo "${PROJECT_ROOT}/.venv/bin/aws")"
  "${AWS_BIN}" s3 cp --only-show-errors "${CODE_S3_URI}" /tmp/rl-distill-code.tar.gz
  tar -xzf /tmp/rl-distill-code.tar.gz --unlink-first --recursive-unlink -C "${PROJECT_ROOT}"
  echo "CODE_REFRESHED $(sha256sum /tmp/rl-distill-code.tar.gz | cut -c1-16) files=$(tar -tzf /tmp/rl-distill-code.tar.gz | wc -l)"
fi
TEACHER_SPEC="${TEACHER_SPEC:-12br-medium}"; STUDENT="${STUDENT:-e4b}"
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${HF_TOKEN:?HF_TOKEN missing (forward it with --dotenv-keys)}"
: "${WANDB_API_KEY:?WANDB_API_KEY missing (forward it with --dotenv-keys)}"
# --- gemma-4 venv on local disk (same recipe as run_gemma4_pt_deepscaler_4of4strict_rl.sh) -----------------
export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
export VENV="${VENV:-/tmp/.venv-gemma4}"
export GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu129}"   # p5 fleet has a CUDA 12.8 driver
if [ ! -x "${VENV}/bin/python" ]; then
  echo "### building ${VENV} via setup_env_gemma4.sh (${GEMMA4_CUDA_VARIANT})"
  VENV="${VENV}" bash rl-distill-scripts/setup_env_gemma4.sh
fi
export PATH="${VENV}/bin:${PATH}"; source "${VENV}/bin/activate"
if [ -x /usr/local/cuda/bin/nvcc ]; then export CUDA_HOME="${CUDA_HOME:-$(readlink -f /usr/local/cuda)}"; elif [ -x /usr/local/cuda-12.9/bin/nvcc ]; then export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"; fi
if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then export PATH="${CUDA_HOME}/bin:${PATH}"; fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
python -c "import torch, transformers, verl" || { echo "FATAL: venv import check failed" >&2; exit 2; }
command -v aws >/dev/null || { echo "FATAL: aws CLI missing (trace bundle / mirrors use S3)" >&2; exit 2; }
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"; export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"; export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET6}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"
n_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
GPU_IDS_CSV="$(seq -s, 0 $((n_gpus - 1)))"
(( n_gpus >= 4 )) || { echo "FATAL: the E4B student needs >= 4 GPUs (fp32 master + Adam sharded), got ${n_gpus}" >&2; exit 2; }
unset CUDA_VISIBLE_DEVICES
echo "ST_12BR config spec=${TEACHER_SPEC} student=${STUDENT} gpus=${GPU_IDS_CSV} venv=${VENV} disk=$(df -h /tmp | tail -1 | awk '{print $4}')"

# --- stage 1: trace collection (DP = all GPUs, TP 1); skipped once the S3 bundle is complete -----------------------------------
TRACE_S3_BASE="${TRACE_S3_BASE:-s3://scale-ml/genai/rl-distill/gemma4-bestckpt-traces-topk128-v2}"
if aws s3 ls "${TRACE_S3_BASE}/${TEACHER_SPEC}/COMPLETE.json" >/dev/null 2>&1; then
  echo "ST_12BR stage1: S3 bundle ${TRACE_S3_BASE}/${TEACHER_SPEC} already complete -> skipping collection"
else
  echo "ST_12BR stage1: trace collection start $(date -u +%FT%TZ)"
  TRACE_SPEC="${TEACHER_SPEC}" TEACHER_SOURCE=s3 TRACE_GPU_IDS="${GPU_IDS_CSV}" TENSOR_PARALLEL_SIZE=1 \
    TRAIN_SAMPLES_PER_QUESTION="${TRAIN_SAMPLES_PER_QUESTION:-16}" VALIDATION_SAMPLES_PER_QUESTION=1 TRACE_S3_MIRROR_ENABLE=true \
    bash rl-distill-scripts/scale_train/run_gemma4_bestckpt_trace_collection.sh
  echo "ST_12BR stage1: trace collection done $(date -u +%FT%TZ)"
fi

# --- stage 2: distillation into the E4B base (§9 recipe; local bundle from stage 1 or re-fetched from S3) ---------------------
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}" LR="${LR:-2e-6}" TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}" TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}" LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}" MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}" TEST_FREQ="${TEST_FREQ:-10}" SAVE_FREQ="${SAVE_FREQ:-250}"
export TRAIN_SAMPLES_PER_QUESTION="${TRAIN_SAMPLES_PER_QUESTION:-16}"
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}" MAX_PADDED_TOKENS_PER_MICROBATCH="${MAX_PADDED_TOKENS_PER_MICROBATCH:-8192}"
export FULL_VOCAB_KL_CHUNK_SIZE="${FULL_VOCAB_KL_CHUNK_SIZE:-4096}" VERL_FAIL_ON_NONFINITE_GRAD="${VERL_FAIL_ON_NONFINITE_GRAD:-0}"
export CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-[\"model\",\"optimizer\",\"extra\",\"hf_model\"]}" MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-1}"
export REMOTE_CHECKPOINT_ENABLE=true ROLLING_CHECKPOINT_FREQ="${ROLLING_CHECKPOINT_FREQ:-50}" ROLLING_HF_EXPORT=true ROLLING_HF_EXPORT_S3=true
export REMOTE_CHECKPOINT_S3_URI="${REMOTE_CHECKPOINT_S3_URI:-s3://scale-ml/genai/rl-distill/gemma4-12br-distill-ckpts-v1/${TEACHER_SPEC}-to-${STUDENT}-bs${TRAIN_BATCH_SIZE}-s${TOTAL_TRAINING_STEPS}-lr${LR}}"
export HF_PUSH_ENABLE=false HF_PUSH_DELETE_LOCAL=false
export EXP_NAME="${EXP_NAME:-${TEACHER_SPEC}-to-${STUDENT}-base-bs${TRAIN_BATCH_SIZE}-s${TOTAL_TRAINING_STEPS}-lr${LR}-st4}"
export PROJECT_NAME="${PROJECT_NAME:-gemma4-12br-distill-v1}" ALLOW_UNDERSIZED_STUDENT_LAYOUT="${ALLOW_UNDERSIZED_STUDENT_LAYOUT:-true}" FSDP_OFFLOAD="${FSDP_OFFLOAD:-false}"
REQUIRE_CKPT_DISK_GB="${REQUIRE_CKPT_DISK_GB:-350}"   # two E4B full checkpoints (~130 GB each) + bundle + venv
free_gb="$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)"
if (( REQUIRE_CKPT_DISK_GB > 0 && free_gb < REQUIRE_CKPT_DISK_GB )); then echo "FATAL: /tmp has ${free_gb} GB free; need >= ${REQUIRE_CKPT_DISK_GB} GB" >&2; exit 2; fi
echo "ST_12BR stage2: distill start $(date -u +%FT%TZ) s3=${REMOTE_CHECKPOINT_S3_URI} exp=${EXP_NAME} free_disk_gb=${free_gb}"
export TEACHER_SPEC STUDENT DISTILL_GPU_IDS="${GPU_IDS_CSV}"
bash rl-distill-scripts/scale_train/run_gemma4_distill_one.sh
status=$?
# NOTE (2026-09-17 11:57Z): the distill trainer already writes its own run_complete.json (fields: checkpoint_step, remote_checkpoint_s3_uri,
# hf_push_enabled, …) when it finishes; running full_checkpoint_s3.py complete-run afterwards raised "run completion receipt mismatch" (RL
# protocol fields) and turned a finished 1000-step run into a FAILED job. Do not add an RL-style receipt here; for the supervisor treat a
# COMPLETED job as finished (no --completion-s3-uri) and verify global_step_${TOTAL_TRAINING_STEPS}/_REMOTE_COMPLETE.json by hand.
echo "ST_12BR_DONE spec=${TEACHER_SPEC} student=${STUDENT} exit=${status} $(date -u +%FT%TZ)"
exit "${status}"
