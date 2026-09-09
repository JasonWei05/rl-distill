#!/usr/bin/env bash
# ScaleTrain pod entry for the pre-training-control distillations (E4B *base* traces -> 12B / 26B-A4B students).
# Launched by launch_st_job.py; the repo is baked into the image at /workspace/rl-distill. Builds the gemma-4
# venv into /tmp once per pod (as the Gemma 4 RL run-file does), then runs run_gemma4_distill_one.sh with the
# v2 recipe: batch 128, lr 2e-6 (100 warmup, linear -> 2e-7), 1000 steps, validate every 10; every 50 steps a resumable
# checkpoint (model + Adam + LR/RNG + dataloader position) AND an HF export pushed to the Hub (pass@k eval point);
# every 250 steps the checkpoint is kept permanently in S3.
# Borrowed pods get preempted (the job goes back to QUEUED and the run-file starts again): the permanent saves (every
# SAVE_FREQ, with the HF export + push) and the rolling saves (every ROLLING_CHECKPOINT_FREQ, single S3 slot) are both full
# FSDP checkpoints mirrored to S3, and the trainer restores the newest complete one at startup and resumes. A relaunch of the same direction + recipe therefore continues the earlier
# attempt; to start over, pass a fresh REMOTE_CHECKPOINT_S3_URI (or delete the prefix).
#
#   --env-vars "TEACHER_SPEC=e4b-base-medium,STUDENT=12b"   (4 GPUs)   /   "...,STUDENT=26b"   (8 GPUs)
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
# The pod's login shell drops the image PATH: put the baked FSDP2 venv back (it holds the aws CLI the distill
# runner needs for the S3 trace bundle) plus the usual system dirs; the gemma-4 venv is prepended below.
export PATH="${PROJECT_ROOT}/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
echo "ST_DISTILL_START $(date -u +%FT%TZ) host=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null || echo baked)"

# Optional code refresh: when the job runs on a pre-built image (--image ...), CODE_S3_URI points at a `git archive`
# tarball of the commit to run; it is unpacked over the baked /workspace/rl-distill so the pod runs current code.
if [ -n "${CODE_S3_URI:-}" ]; then
  echo "### refreshing repo code from ${CODE_S3_URI}"
  AWS_BIN="$(command -v aws || echo "${PROJECT_ROOT}/.venv/bin/aws")"
  "${AWS_BIN}" s3 cp --only-show-errors "${CODE_S3_URI}" /tmp/rl-distill-code.tar.gz
  tar -xzf /tmp/rl-distill-code.tar.gz --unlink-first --recursive-unlink -C "${PROJECT_ROOT}"   # baked tree may have symlink/dir type clashes
  echo "CODE_REFRESHED $(sha256sum /tmp/rl-distill-code.tar.gz | cut -c1-16) files=$(tar -tzf /tmp/rl-distill-code.tar.gz | wc -l)"
fi

TEACHER_SPEC="${TEACHER_SPEC:-e4b-base-medium}"
STUDENT="${STUDENT:?12b or 26b}"
case "${TEACHER_SPEC}" in e4b-base-medium|e4b-base-hard) ;; *) echo "FATAL: TEACHER_SPEC must be e4b-base-medium|e4b-base-hard" >&2; exit 2 ;; esac
case "${STUDENT}" in 12b|26b) ;; *) echo "FATAL: STUDENT must be 12b or 26b" >&2; exit 2 ;; esac

# HF_TOKEN / WANDB_API_KEY arrive as forwarded env vars (launch_st_job --dotenv-keys); .env may be absent.
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
command -v aws >/dev/null || { echo "FATAL: aws CLI missing (trace bundle comes from S3)" >&2; exit 2; }
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"; export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"; export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
# ScaleTrain pods on ml-gpu-batch are IPv6-only (pod IP 2602:fb33:...), interface eth0.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET6}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"

# --- GPUs: all visible devices; 12B needs 4, 26B-A4B 8 --------------------------------------------------------
n_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
DISTILL_GPU_IDS="$(seq -s, 0 $((n_gpus - 1)))"
case "${STUDENT}" in
  12b) (( n_gpus >= 4 )) || { echo "FATAL: 12B needs >= 4 GPUs, got ${n_gpus}" >&2; exit 2; } ;;
  26b) (( n_gpus >= 8 )) || { echo "FATAL: 26B-A4B needs 8 GPUs (or FSDP_OFFLOAD=true), got ${n_gpus}" >&2; exit 2; } ;;
esac
echo "ST_DISTILL config spec=${TEACHER_SPEC} student=${STUDENT} gpus=${DISTILL_GPU_IDS} venv=${VENV}"
echo "ST_DISTILL disk: $(df -h /tmp | tail -1)"
unset CUDA_VISIBLE_DEVICES   # the runner sets it from DISTILL_GPU_IDS

# --- v2 recipe (overridable) ---------------------------------------------------------------------------------
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}" LR="${LR:-2e-6}" TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}" MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}" TEST_FREQ="${TEST_FREQ:-10}" SAVE_FREQ="${SAVE_FREQ:-250}"

# --- preemption-safe checkpoints ---------------------------------------------------------------------------------
# Permanent checkpoint every SAVE_FREQ steps (S3 history, ~170 GB per 12B save / ~370 GB per 26B save) plus a rolling
# resumable checkpoint every ROLLING_CHECKPOINT_FREQ steps (one S3 slot, uploaded in the background, retired when the next
# permanent save lands). Both write the HF export and push it to the Hub -> one evaluable step_* every 50 steps. One checkpoint kept on local disk. The pusher must not delete
# the local hf export: the S3 upload enumerates the whole step directory after the (async) push starts.
export CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-[\"model\",\"optimizer\",\"extra\",\"hf_model\"]}"
export MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-1}" HF_PUSH_DELETE_LOCAL="${HF_PUSH_DELETE_LOCAL:-false}" HF_PUSH_MAX_TO_KEEP="${HF_PUSH_MAX_TO_KEEP:-24}"   # keep all 20 step_* exports
export REMOTE_CHECKPOINT_ENABLE="${REMOTE_CHECKPOINT_ENABLE:-true}" ROLLING_CHECKPOINT_FREQ="${ROLLING_CHECKPOINT_FREQ:-50}" ROLLING_HF_EXPORT="${ROLLING_HF_EXPORT:-true}"
export REMOTE_CHECKPOINT_S3_URI="${REMOTE_CHECKPOINT_S3_URI:-s3://scale-ml/genai/rl-distill/gemma4-e4b-base-distill-ckpts-v1/${TEACHER_SPEC}-to-${STUDENT}-bs${TRAIN_BATCH_SIZE}-s${TOTAL_TRAINING_STEPS}-lr${LR}}"
# Local disk must hold two full checkpoints (the new one is written before the old one is pruned) plus the student
# snapshot, venv and trace bundle. REQUIRE_CKPT_DISK_GB=0 disables the check.
case "${STUDENT}" in 12b) ckpt_disk_default=450 ;; *) ckpt_disk_default=900 ;; esac
REQUIRE_CKPT_DISK_GB="${REQUIRE_CKPT_DISK_GB:-${ckpt_disk_default}}"
free_gb="$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)"
if (( REQUIRE_CKPT_DISK_GB > 0 && free_gb < REQUIRE_CKPT_DISK_GB )); then
  echo "FATAL: /tmp has ${free_gb} GB free; full checkpoints for STUDENT=${STUDENT} need >= ${REQUIRE_CKPT_DISK_GB} GB (REQUIRE_CKPT_DISK_GB)" >&2; exit 2
fi
echo "ST_DISTILL checkpoints: save_freq=${SAVE_FREQ} rolling_freq=${ROLLING_CHECKPOINT_FREQ} contents=${CHECKPOINT_SAVE_CONTENTS} s3=${REMOTE_CHECKPOINT_S3_URI} free_disk_gb=${free_gb}"
export PROJECT_NAME="${PROJECT_NAME:-gemma4-e4b-base-distill-v1}" ALLOW_UNDERSIZED_STUDENT_LAYOUT="${ALLOW_UNDERSIZED_STUDENT_LAYOUT:-true}"
export TRACE_S3_MIRROR_ENABLE="${TRACE_S3_MIRROR_ENABLE:-false}"   # HF-only artifacts; the bundle itself is read from S3
export TEACHER_SPEC STUDENT DISTILL_GPU_IDS
bash rl-distill-scripts/scale_train/run_gemma4_distill_one.sh
status=$?
echo "ST_DISTILL_DONE spec=${TEACHER_SPEC} student=${STUDENT} exit=${status} $(date -u +%FT%TZ)"
exit "${status}"
