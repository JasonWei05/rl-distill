#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_FORCE_PLATFORM="${VLLM_FORCE_PLATFORM:-cuda}"
mkdir -p "${HF_HOME}"

SHARD_ID="${SHARD_ID:?set SHARD_ID}"
TOTAL_SHARDS="${TOTAL_SHARDS:?set TOTAL_SHARDS}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}"

TP="${TP:-1}"
DP="${DP:-1}"
N="${N:-1}"
MAX_TOKENS="${MAX_TOKENS:-20480}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-22528}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TEACHER_MODEL="${TEACHER_MODEL:-google/gemma-3-4b-pt}"
REVISION="${REVISION:-main}"
INPUT="${INPUT:?set INPUT}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"
MM_ENCODER_ATTN_BACKEND="${MM_ENCODER_ATTN_BACKEND:-}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

EXTRA_ARGS=()
if [ -n "${MM_ENCODER_ATTN_BACKEND}" ]; then
    EXTRA_ARGS+=(--mm_encoder_attn_backend "${MM_ENCODER_ATTN_BACKEND}")
fi
if [ "${ENFORCE_EAGER}" = "true" ]; then
    EXTRA_ARGS+=(--enforce_eager)
fi
if [ -n "${CHAT_TEMPLATE}" ]; then
    EXTRA_ARGS+=(--chat_template "${CHAT_TEMPLATE}")
fi

echo "Launching shard ${SHARD_ID}/${TOTAL_SHARDS} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    nohup "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_teacher_data.py" \
        --teacher_model "${TEACHER_MODEL}" \
        --revision "${REVISION}" \
        --input_parquet "${INPUT}" \
        --output_dir "${OUTPUT_DIR}" \
        --shard_id "${SHARD_ID}" \
        --num_shards "${TOTAL_SHARDS}" \
        --max_samples "${MAX_SAMPLES}" \
        --tp "${TP}" \
        --dp "${DP}" \
        --n "${N}" \
        --max_tokens "${MAX_TOKENS}" \
        --max_model_len "${MAX_MODEL_LEN}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --top_k "${TOP_K}" \
        --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
        "${EXTRA_ARGS[@]}" \
        > "${LOG_DIR}/shard_${SHARD_ID}.log" 2>&1 &

echo "PID: $!"
