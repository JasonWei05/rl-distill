#!/usr/bin/env bash
# ScaleTrain pod entry for the pre-training-control distillations (E4B *base* traces -> 12B / 26B-A4B students).
# Launched by launch_st_job.py; the repo is baked into the image at /workspace/rl-distill. Builds the gemma-4
# venv into /tmp once per pod (as the Gemma 4 RL run-file does), then runs run_gemma4_distill_one.sh with the
# v2 recipe: batch 128, lr 2e-6 (100 warmup, linear -> 2e-7), 1000 steps, validate every 10, save + push every 250.
#
#   --env-vars "TEACHER_SPEC=e4b-base-medium,STUDENT=12b"   (4 GPUs)   /   "...,STUDENT=26b"   (8 GPUs)
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
echo "ST_DISTILL_START $(date -u +%FT%TZ) host=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null || echo baked)"

# Optional code refresh: when the job runs on a pre-built image (--image ...), CODE_S3_URI points at a `git archive`
# tarball of the commit to run; it is unpacked over the baked /workspace/rl-distill so the pod runs current code.
if [ -n "${CODE_S3_URI:-}" ]; then
  echo "### refreshing repo code from ${CODE_S3_URI}"
  aws s3 cp --only-show-errors "${CODE_S3_URI}" /tmp/rl-distill-code.tar.gz
  tar -xzf /tmp/rl-distill-code.tar.gz -C "${PROJECT_ROOT}"
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
unset CUDA_VISIBLE_DEVICES   # the runner sets it from DISTILL_GPU_IDS

# --- v2 recipe (overridable) ---------------------------------------------------------------------------------
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}" LR="${LR:-2e-6}" TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}" MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}" TEST_FREQ="${TEST_FREQ:-10}" SAVE_FREQ="${SAVE_FREQ:-250}"
export PROJECT_NAME="${PROJECT_NAME:-gemma4-e4b-base-distill-v1}" ALLOW_UNDERSIZED_STUDENT_LAYOUT="${ALLOW_UNDERSIZED_STUDENT_LAYOUT:-true}"
export TRACE_S3_MIRROR_ENABLE="${TRACE_S3_MIRROR_ENABLE:-false}"   # HF-only artifacts; the bundle itself is read from S3
export TEACHER_SPEC STUDENT DISTILL_GPU_IDS
bash rl-distill-scripts/scale_train/run_gemma4_distill_one.sh
status=$?
echo "ST_DISTILL_DONE spec=${TEACHER_SPEC} student=${STUDENT} exit=${status} $(date -u +%FT%TZ)"
exit "${status}"
