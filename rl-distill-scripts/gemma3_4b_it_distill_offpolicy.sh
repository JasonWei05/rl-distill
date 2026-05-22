#!/usr/bin/env bash
set -xeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

export PROJECT_NAME="distill"
export EXP_NAME="${EXP_NAME:-offpolicy-gemma3-4b-$(date +%Y%m%d-%H%M)}"

RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl}"

# ---- Data: train/val split of JWei05/DAPO-Gemma3-27B-IT-RL-SFT-Data ----
# 16,000 prompt_ids * 2 responses = 32,000 rows train.
# remaining ~1,398 prompt_ids * 1 response    = ~1,398 rows val.
# Both splits are produced deterministically with seed=42.
export TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/teacher_27b_step40_sft_train.parquet}"
export VAL_FILE="${VAL_FILE:-${RAY_DATA_HOME}/data/teacher_27b_step40_sft_val.parquet}"

# ---- Checkpoints: route to /tmp (NVMe, ~918 GB) — rootfs only has ~18 GB. ----
CKPT_BASE="${CKPT_BASE:-/tmp/verl/ckpts}"
export CKPTS_DIR="${CKPT_BASE}/${PROJECT_NAME}/${EXP_NAME}"

MODEL_PATH="${MODEL_PATH:-${RAY_DATA_HOME}/models/gemma-3-4b-it}"

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=lo
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ---- Cadence + HF push ----
# SAVE_FREQ=50  -> save every 50 SFT steps and push that checkpoint to HF.
# TEST_FREQ=2   -> compute val loss (forward KL) on the DAPO held-out split
#                  every 2 SFT steps.
# HF_PUSH_MAX_TO_KEEP=10 -> HF ring buffer: 11th save evicts the oldest.
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-2}"
export MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-2}"
export HF_PUSH_ENABLE="${HF_PUSH_ENABLE:-true}"
export HF_PUSH_REPO="${HF_PUSH_REPO:-JWei05/gemma3-4b-it-off-policy-distilled-from-dapo27b}"
export HF_PUSH_PRIVATE="${HF_PUSH_PRIVATE:-false}"
export HF_PUSH_MAX_TO_KEEP="${HF_PUSH_MAX_TO_KEEP:-10}"
# delete_local_after reclaims /tmp after each successful push (no local resume)
export HF_PUSH_DELETE_LOCAL="${HF_PUSH_DELETE_LOCAL:-false}"

# ---- Ensure split parquets exist (auto-split on first run) ----
if [ ! -f "${TRAIN_FILE}" ] || [ ! -f "${VAL_FILE}" ]; then
    echo "[distill] missing split parquet(s); running data/split_sft_dataset.py"
    python3 "$(dirname "${BASH_SOURCE[0]}")/data/split_sft_dataset.py" \
        --output_dir "${RAY_DATA_HOME}/data" \
        --seed 42
fi

cd "$(dirname "${BASH_SOURCE[0]}")"

# 1 node × 8 × B200 — SFTTrainer needs torch.distributed initialized.
torchrun --standalone --nnodes=1 --nproc_per_node=8 main_distill_offpolicy.py \
    model.path="${MODEL_PATH}" \
    model.enable_gradient_checkpointing=True \
    engine.strategy=fsdp2 \
    engine.fsdp_size=-1 \
    '+engine.wrap_policy.transformer_layer_cls_to_wrap=["Gemma3DecoderLayer"]' \
    optim.lr=1e-5 \
    optim.lr_warmup_steps=200 \
    optim.lr_scheduler_type=constant \
    optim.total_training_steps=400 \
    optim.min_lr_ratio=0.1 \
    optim.weight_decay=0.1 \
    optim.betas=[0.9,0.98] \
    optim.clip_grad=1.0 \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=16 \
    data.max_length=18432 \
    trainer.total_epochs=3 \
    trainer.total_training_steps=250 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.seed=42 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    "$@"
