#!/usr/bin/env bash
set -euo pipefail

# Launch data-parallel teacher generation on one node.
# 8 B200 GPUs, TP=2 per instance → 4 shards per node.
#
# For 2 nodes:
#   Node 0: bash launch_teacher_gen.sh 0   (shards 0-3)
#   Node 1: bash launch_teacher_gen.sh 1   (shards 4-7)
#
# After both finish, merge:
#   python3 merge_teacher_shards.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source .env for HF_TOKEN
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_FORCE_PLATFORM="${VLLM_FORCE_PLATFORM:-cuda}"
mkdir -p "${HF_HOME}"

NODE_ID="${1:-0}"          # 0 or 1
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TP="${TP:-2}"
DP="${DP:-1}"
INSTANCE_GPUS=$((TP * DP))
SHARDS_PER_NODE=$((GPUS_PER_NODE / INSTANCE_GPUS))
TOTAL_SHARDS="${TOTAL_SHARDS:-8}"         # 8 for 2 nodes
N="${N:-4}"                                # responses per prompt
MAX_TOKENS="${MAX_TOKENS:-20480}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-22528}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MM_ENCODER_ATTN_BACKEND="${MM_ENCODER_ATTN_BACKEND:-}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
WAIT="${WAIT:-false}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"

TEACHER_MODEL="${TEACHER_MODEL:-JWei05/dapo-gemma3-27b-it}"
REVISION="${REVISION:-step_000040}"
INPUT="${INPUT:-${HOME}/verl/data/dapo-math-17k.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/verl/data/teacher_gen}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

echo "=== Node ${NODE_ID}: launching ${SHARDS_PER_NODE} shards (TP=${TP}, DP=${DP}) ==="

PIDS=()
for LOCAL_SHARD in $(seq 0 $((SHARDS_PER_NODE - 1))); do
    SHARD_ID=$((NODE_ID * SHARDS_PER_NODE + LOCAL_SHARD))
    GPU_START=$((LOCAL_SHARD * INSTANCE_GPUS))
    GPU_END=$((GPU_START + INSTANCE_GPUS - 1))
    GPUS=$(seq -s, ${GPU_START} ${GPU_END})

    echo "  Shard ${SHARD_ID}/${TOTAL_SHARDS} on GPUs ${GPUS}"

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

    ENV_PREFIX=(env "CUDA_VISIBLE_DEVICES=${GPUS}" "VLLM_ENABLE_V1_MULTIPROCESSING=0")
    if [ -n "${VLLM_USE_V1:-}" ]; then
        ENV_PREFIX+=("VLLM_USE_V1=${VLLM_USE_V1}")
    fi

    "${ENV_PREFIX[@]}" nohup "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_teacher_data.py" \
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

    PID="$!"
    PIDS+=("${PID}")
    echo "    PID: ${PID}"
    # Stagger — launching all 4 vLLM instances in parallel caused silent
    # startup failures for the last one (suspected /dev/shm or FD contention
    # during simultaneous shared-memory setup). 30s between launches gives
    # each one time to finish its shm allocation before the next starts.
    sleep 30
done

echo ""
echo "All shards launched. Monitor with:"
echo "  tail -f ${LOG_DIR}/shard_*.log"
echo ""
echo "After all nodes finish, merge with:"
echo "  python3 ${SCRIPT_DIR}/merge_teacher_shards.py --input_dir ${OUTPUT_DIR}"

if [ "${WAIT}" = "true" ]; then
    echo ""
    echo "Waiting for shard PIDs: ${PIDS[*]}"
    wait "${PIDS[@]}"
    echo "All local shard processes finished."
fi
