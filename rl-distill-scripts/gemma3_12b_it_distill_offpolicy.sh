#!/usr/bin/env bash
set -xeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

export PROJECT_NAME="distill"
export EXP_NAME="${EXP_NAME:-offpolicy-gemma3-12b-$(date +%Y%m%d-%H%M)}"

RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl}"

export TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/teacher_27b_step40_sft_train.parquet}"
export VAL_FILE="${VAL_FILE:-${RAY_DATA_HOME}/data/teacher_27b_step40_sft_val.parquet}"

CKPT_BASE="${CKPT_BASE:-/tmp/verl/ckpts}"
export CKPTS_DIR="${CKPT_BASE}/${PROJECT_NAME}/${EXP_NAME}"

MODEL_PATH="${MODEL_PATH:-${RAY_DATA_HOME}/models/gemma-3-12b-it}"

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=lo
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export SAVE_FREQ="${SAVE_FREQ:-50}"
export TEST_FREQ="${TEST_FREQ:-2}"
export MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-2}"
export HF_PUSH_ENABLE="${HF_PUSH_ENABLE:-true}"
export HF_PUSH_REPO="${HF_PUSH_REPO:-JWei05/gemma3-12b-it-off-policy-distilled-from-dapo27b}"
export HF_PUSH_PRIVATE="${HF_PUSH_PRIVATE:-false}"
export HF_PUSH_MAX_TO_KEEP="${HF_PUSH_MAX_TO_KEEP:-10}"
export HF_PUSH_DELETE_LOCAL="${HF_PUSH_DELETE_LOCAL:-false}"

if [ ! -f "${TRAIN_FILE}" ] || [ ! -f "${VAL_FILE}" ]; then
    echo "[distill-12b] missing split parquet(s); running data/split_sft_dataset.py"
    python3 "$(dirname "${BASH_SOURCE[0]}")/data/split_sft_dataset.py" \
        --output_dir "${RAY_DATA_HOME}/data" \
        --seed 42
fi

# Auto-download Gemma 3 12B IT if missing (first run only).
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "[distill-12b] ${MODEL_PATH} missing — downloading Gemma 3 12B IT"
    python3 "${PROJECT_ROOT}/dapo/setup_model.py" --size 12b --variant it \
        --output-dir "${MODEL_PATH}"
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
    optim.lr_warmup_steps=20 \
    optim.lr_scheduler_type=cosine \
    optim.total_training_steps=200 \
    optim.min_lr_ratio=0.1 \
    optim.weight_decay=0.1 \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=16 \
    data.max_length=22528 \
    trainer.total_epochs=3 \
    trainer.total_training_steps=200 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.seed=42 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    "$@"
