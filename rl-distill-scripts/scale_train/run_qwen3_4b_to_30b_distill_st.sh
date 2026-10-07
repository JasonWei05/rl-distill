#!/usr/bin/env bash
# ScaleTrain pod entry: off-policy top-128 forward-KL distillation of the untrained Qwen3-4B-Base teacher's traces
# (q4b-base-<band> bundles, 3000 train q x 16 + 300 val q x 1) into Qwen3-30B-A3B-Base -- the Qwen analogue of the
# E4B-base -> 26B-A4B pre-training control (DISTILLATION_EXPERIMENTS.md §9, same recipe): batch 128, lr 2e-6 (100 warmup,
# linear -> 2e-7), 1000 steps (~2.7 epochs), validation KL on 128 teacher val generations every 10 steps, one sequence per
# micro-batch (4096 padded-token ceiling), KL over 1024-token vocab chunks, fp32 master + Adam, bf16 param views, grad ckpt.
# Layout: FSDP2 DP8 on one node, no offload (30.5B x 16 B = 61 GB/GPU of params/grads/Adam; 4 GPUs would need CPU offload).
# S3-only checkpoints: permanent every 250 steps only (user, 2026-10-07: no rolling saves; ROLLING_CHECKPOINT_FREQ=50 restores
# them), each with its HF export; no Hub pushes. Preemption: the trainer restores the newest complete S3 checkpoint at startup
# (<= 250 steps lost).
#   launched by launch_qwen3_4b_to_30b_distill_st.sh (8 GPUs, borrowing, priority high) via launch_st_with_code.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${PROJECT_ROOT}"
export PATH="${PROJECT_ROOT}/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
echo "Q30B_DISTILL_START $(date -u +%FT%TZ) host=$(hostname)"
if [ -n "${CODE_S3_URI:-}" ]; then
  AWS_BIN="$(command -v aws || echo "${PROJECT_ROOT}/.venv/bin/aws")"
  "${AWS_BIN}" s3 cp --only-show-errors "${CODE_S3_URI}" /tmp/rl-distill-code.tar.gz
  tar -xzf /tmp/rl-distill-code.tar.gz --unlink-first --recursive-unlink -C "${PROJECT_ROOT}"
  echo "CODE_REFRESHED $(sha256sum /tmp/rl-distill-code.tar.gz | cut -c1-16)"
fi
TEACHER_SPEC="${TEACHER_SPEC:-q4b-base-medium}"; STUDENT=q30b
case "${TEACHER_SPEC}" in q4b-base-medium|q4b-base-hard) ;; *) echo "FATAL: TEACHER_SPEC must be q4b-base-medium|q4b-base-hard" >&2; exit 2 ;; esac
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${HF_TOKEN:?HF_TOKEN missing (forward it with --dotenv-keys)}"
: "${WANDB_API_KEY:?WANDB_API_KEY missing (forward it with --dotenv-keys)}"
# --- venv on local disk (same as the Qwen RL / Gemma 4 runs) ---------------------------------------------------
export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
export VENV="${VENV:-/tmp/.venv-gemma4}" GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu129}"   # p5 fleet: CUDA 12.8 driver
[ -x "${VENV}/bin/python" ] || VENV="${VENV}" bash rl-distill-scripts/setup_env_gemma4.sh
export PATH="${VENV}/bin:${PATH}"; source "${VENV}/bin/activate"
if [ -x /usr/local/cuda/bin/nvcc ]; then export CUDA_HOME="${CUDA_HOME:-$(readlink -f /usr/local/cuda)}"; export PATH="${CUDA_HOME}/bin:${PATH}"; fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
python -c "import torch, transformers, verl" || { echo "FATAL: venv import check failed" >&2; exit 2; }
command -v aws >/dev/null || { echo "FATAL: aws CLI missing (trace bundle + checkpoints use S3)" >&2; exit 2; }
export HF_HOME="${HF_HOME:-/tmp/hf_cache}" HF_HUB_CACHE="${HF_HUB_CACHE:-/tmp/hf_cache/hub}" WANDB_DIR="${WANDB_DIR:-/tmp/wandb}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
# ScaleTrain pods on ml-gpu-batch are IPv6-only, interface eth0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}" NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET6}" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"
n_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( n_gpus >= 8 )) || [ "${FSDP_OFFLOAD:-false}" = true ] || { echo "FATAL: the Qwen3-30B-A3B student needs 8 GPUs (or FSDP_OFFLOAD=true), got ${n_gpus}" >&2; exit 2; }
unset CUDA_VISIBLE_DEVICES
# --- §9 recipe (the E4B-base -> 26B-A4B student runs) ---------------------------------------------------------------
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}" LR="${LR:-2e-6}" TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}" TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}" LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}" MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}" TEST_FREQ="${TEST_FREQ:-10}" SAVE_FREQ="${SAVE_FREQ:-250}"
export TRAIN_SAMPLES_PER_QUESTION="${TRAIN_SAMPLES_PER_QUESTION:-16}" FSDP_OFFLOAD="${FSDP_OFFLOAD:-false}" ALLOW_UNDERSIZED_STUDENT_LAYOUT="${ALLOW_UNDERSIZED_STUDENT_LAYOUT:-true}"
# --- S3-only checkpoints --------------------------------------------------------------------------------------------
export CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-[\"model\",\"optimizer\",\"extra\",\"hf_model\"]}" MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-1}"
export REMOTE_CHECKPOINT_ENABLE=true ROLLING_CHECKPOINT_FREQ="${ROLLING_CHECKPOINT_FREQ:-0}" ROLLING_HF_EXPORT=true ROLLING_HF_EXPORT_S3=true
export REMOTE_CHECKPOINT_S3_URI="${REMOTE_CHECKPOINT_S3_URI:-s3://scale-ml/genai/rl-distill/qwen3-4b-base-distill-ckpts-v1/${TEACHER_SPEC}-to-${STUDENT}-bs${TRAIN_BATCH_SIZE}-s${TOTAL_TRAINING_STEPS}-lr${LR}}"
export HF_PUSH_ENABLE=false HF_PUSH_DELETE_LOCAL=false PROJECT_NAME="${PROJECT_NAME:-qwen3-4b-base-distill-v1}"
export EXP_NAME="${EXP_NAME:-${TEACHER_SPEC}-to-${STUDENT}-base-bs${TRAIN_BATCH_SIZE}-s${TOTAL_TRAINING_STEPS}-lr${LR}-g${n_gpus}}"
# two full checkpoints (~430 GB each with the HF export; the new one is written before the old is pruned) + model + bundle + venv
REQUIRE_CKPT_DISK_GB="${REQUIRE_CKPT_DISK_GB:-1000}"
free_gb="$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)"
if (( REQUIRE_CKPT_DISK_GB > 0 && free_gb < REQUIRE_CKPT_DISK_GB )); then echo "FATAL: /tmp has ${free_gb} GB free; need >= ${REQUIRE_CKPT_DISK_GB} GB" >&2; exit 2; fi
echo "Q30B_DISTILL config spec=${TEACHER_SPEC} gpus=${n_gpus} offload=${FSDP_OFFLOAD} s3=${REMOTE_CHECKPOINT_S3_URI} exp=${EXP_NAME} free_disk_gb=${free_gb}"
export TEACHER_SPEC STUDENT DISTILL_GPU_IDS="$(seq -s, 0 $((n_gpus - 1)))"
status=0; bash rl-distill-scripts/scale_train/run_gemma4_distill_one.sh || status=$?
# The distill trainer writes its own run_complete.json; no RL-style receipt (see run_gemma4_12br_trace_and_distill_st.sh).
echo "Q30B_DISTILL_DONE spec=${TEACHER_SPEC} exit=${status} $(date -u +%FT%TZ)"
exit "${status}"
