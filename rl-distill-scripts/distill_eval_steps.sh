#!/usr/bin/env bash
# Evaluate selected SFT checkpoints from a pushed HF repo on the 8 RL val sets.
#
# Required:
#   HF_PUSH_REPO       model repo containing step_000250/, step_000500/, ...
#   STUDENT_HF_REPO    base model repo, e.g. google/gemma-3-4b-pt
# Optional:
#   EXP_NAME           output subdir name
#   EVAL_STEPS         space-separated numeric steps, default "250 500"
#   EVAL_DATA_DIR      directory containing the *_compat.parquet val files
#   RAY_DATA_HOME      output root, default /tmp/verl
#   WANDB_PROXY_CPU_IP / WANDB_PROXY_CPU_PORT for B200 HF connectivity
set -xeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
set -a; source "${PROJECT_ROOT}/.env"; set +a

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}"

if [ -f /proc/driver/nvidia/version ]; then
    NVIDIA_KERNEL_VERSION="$(sed -n 's/.*  \([0-9][0-9.]*\)  .*/\1/p' /proc/driver/nvidia/version | head -1 || true)"
    if [ -n "${NVIDIA_KERNEL_VERSION}" ] && [ -f "/usr/lib/x86_64-linux-gnu/libcuda.so.${NVIDIA_KERNEL_VERSION}" ]; then
        DRIVER_LIB_DIR="/tmp/nvidia-driver-libs-${NVIDIA_KERNEL_VERSION}"
        mkdir -p "${DRIVER_LIB_DIR}"
        ln -sf "/usr/lib/x86_64-linux-gnu/libcuda.so.${NVIDIA_KERNEL_VERSION}" "${DRIVER_LIB_DIR}/libcuda.so"
        ln -sf "/usr/lib/x86_64-linux-gnu/libcuda.so.${NVIDIA_KERNEL_VERSION}" "${DRIVER_LIB_DIR}/libcuda.so.1"
        if [ -f "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.${NVIDIA_KERNEL_VERSION}" ]; then
            ln -sf "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.${NVIDIA_KERNEL_VERSION}" "${DRIVER_LIB_DIR}/libnvidia-ml.so.1"
        fi
        export LD_LIBRARY_PATH="${DRIVER_LIB_DIR}:${LD_LIBRARY_PATH:-}"
        echo "[env] using NVIDIA user libraries from ${DRIVER_LIB_DIR}"
    fi
fi

: "${HF_PUSH_REPO:?HF_PUSH_REPO is required}"
: "${STUDENT_HF_REPO:?STUDENT_HF_REPO is required}"

TS="$(date +%Y%m%d-%H%M)"
export EXP_NAME="${EXP_NAME:-eval-${HF_PUSH_REPO##*/}-${TS}}"
RAY_DATA_HOME="${RAY_DATA_HOME:-/tmp/verl}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${HOME}/verl/data}"
EVAL_OUT="${EVAL_OUT:-${RAY_DATA_HOME}/data/eval_results/${EXP_NAME}}"
mkdir -p "${EVAL_OUT}"

PROXY_PID=""
if [ -n "${WANDB_PROXY_CPU_IP:-}" ] && [ -n "${WANDB_PROXY_CPU_PORT:-}" ]; then
    WANDB_PROXY_LOCAL_PORT="${WANDB_PROXY_LOCAL_PORT:-18080}"
    echo "[env] starting SSH SOCKS proxy for HF via ${WANDB_PROXY_CPU_IP}:${WANDB_PROXY_CPU_PORT} on 127.0.0.1:${WANDB_PROXY_LOCAL_PORT}"
    ssh -N -D "127.0.0.1:${WANDB_PROXY_LOCAL_PORT}" \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -p "${WANDB_PROXY_CPU_PORT}" \
        "tiger@${WANDB_PROXY_CPU_IP}" &
    PROXY_PID="$!"
    trap 'if [ -n "${PROXY_PID}" ]; then kill "${PROXY_PID}" >/dev/null 2>&1 || true; fi' EXIT
    sleep 3
    export HTTP_PROXY="socks5h://127.0.0.1:${WANDB_PROXY_LOCAL_PORT}"
    export HTTPS_PROXY="${HTTP_PROXY}"
    export http_proxy="${HTTP_PROXY}"
    export https_proxy="${HTTPS_PROXY}"
fi

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

VAL_FILES=(
    "${EVAL_DATA_DIR}/dapo_openmath2_mix_val_compat.parquet"
    "${EVAL_DATA_DIR}/math__aime2024_repeated_32x_960_compat.parquet"
    "${EVAL_DATA_DIR}/math__aime2025_repeated_32x_960_compat.parquet"
    "${EVAL_DATA_DIR}/math__aime2026_repeated_32x_960_compat.parquet"
    "${EVAL_DATA_DIR}/math__math_500_repeated_2x_1000_compat.parquet"
    "${EVAL_DATA_DIR}/math__olympiadbench_repeated_2x_compat.parquet"
    "${EVAL_DATA_DIR}/math__minervamath_repeated_4x_compat.parquet"
    "${EVAL_DATA_DIR}/math__gsm8k_test_compat.parquet"
)

EVAL_STEPS="${EVAL_STEPS:-250 500}"
STEPS=()
for s in ${EVAL_STEPS}; do
    STEPS+=("$(printf 'step_%06d' "${s}")")
done

EVAL_EXTRA_ARGS=()
if [ "${STUDENT_TAG:-}" = "1b" ]; then
    EVAL_EXTRA_ARGS+=(--attention_backend FLASH_ATTN)
fi
if [ -n "${EVAL_ATTENTION_BACKEND:-}" ]; then
    EVAL_EXTRA_ARGS+=(--attention_backend "${EVAL_ATTENTION_BACKEND}")
fi
if [ -n "${MM_ENCODER_ATTN_BACKEND:-}" ]; then
    EVAL_EXTRA_ARGS+=(--mm_encoder_attn_backend "${MM_ENCODER_ATTN_BACKEND}")
fi
if [ -n "${EVAL_BLOCK_SIZE:-}" ]; then
    EVAL_EXTRA_ARGS+=(--block_size "${EVAL_BLOCK_SIZE}")
fi
if [ "${EVAL_ENFORCE_EAGER:-0}" = "1" ]; then
    EVAL_EXTRA_ARGS+=(--enforce_eager)
fi

EVAL_TP="${EVAL_TP:-1}"
EVAL_PARALLEL="${EVAL_PARALLEL:-4}"
EVAL_GPU_OFFSET="${EVAL_GPU_OFFSET:-0}"
idx=0
echo "[eval] repo=${HF_PUSH_REPO} base=${STUDENT_HF_REPO} steps=${STEPS[*]} out=${EVAL_OUT}"
while [ $idx -lt ${#STEPS[@]} ]; do
    BATCH_PIDS=()
    for slot in $(seq 0 $((EVAL_PARALLEL - 1))); do
        [ $idx -ge ${#STEPS[@]} ] && break
        SUB="${STEPS[$idx]}"
        GPU_START=$((EVAL_GPU_OFFSET + slot * EVAL_TP))
        GPU_END=$((GPU_START + EVAL_TP - 1))
        GPUS="$(seq -s, ${GPU_START} ${GPU_END})"
        LOG="${EVAL_OUT}/eval_${SUB}.log"
        echo "[eval] batch=$((idx / EVAL_PARALLEL)) slot=${slot} ${SUB} GPUs=${GPUS} -> ${LOG}"
        CUDA_VISIBLE_DEVICES="${GPUS}" \
        nohup python3 "${PROJECT_ROOT}/dapo/_eval_model_on_math.py" \
            --repo_id "${HF_PUSH_REPO}" \
            --subfolder "${SUB}" \
            --base_hf_model "${STUDENT_HF_REPO}" \
            --val_files "${VAL_FILES[@]}" \
            --output_dir "${EVAL_OUT}" \
            --tp "${EVAL_TP}" \
            --temperature 1.0 \
            --top_p 0.7 \
            --top_k -1 \
            --max_tokens 20480 \
            "${EVAL_EXTRA_ARGS[@]}" \
            > "${LOG}" 2>&1 &
        BATCH_PIDS+=("$!")
        idx=$((idx + 1))
        sleep 30
    done
    echo "[eval] waiting on batch PIDs: ${BATCH_PIDS[*]}"
    wait "${BATCH_PIDS[@]}" || true
    echo "[eval] batch done"
done

python3 - <<PYEOF
import glob, json, os
out = "${EVAL_OUT}"
files = sorted(glob.glob(f"{out}/*__summary.json"))
benches = [
    ("dapo_val", "dapo_openmath2_mix_val_compat"),
    ("aime2024", "math__aime2024_repeated_32x_960_compat"),
    ("aime2025", "math__aime2025_repeated_32x_960_compat"),
    ("aime2026", "math__aime2026_repeated_32x_960_compat"),
    ("math500",  "math__math_500_repeated_2x_1000_compat"),
    ("olympiad", "math__olympiadbench_repeated_2x_compat"),
    ("minerva",  "math__minervamath_repeated_4x_compat"),
    ("gsm8k",    "math__gsm8k_test_compat"),
]
print("\\n=== ${EXP_NAME} eval summary (acc) ===")
print(f"{'step':>14s} | " + " | ".join(f"{b[0]:>9s}" for b in benches))
for f in files:
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f"{os.path.basename(f)}: parse error {e}")
        continue
    step = d.get("subfolder", "?")
    row = []
    for _, key in benches:
        v = d.get("per_dataset", {}).get(key, {}).get("acc")
        row.append(f"{v:>9.4f}" if isinstance(v, (int, float)) else f"{'?':>9s}")
    print(f"{step:>14s} | " + " | ".join(row))
PYEOF

python3 "${SCRIPT_DIR}/update_eval_results_md.py" \
    --results-dir "${EVAL_OUT}" \
    --run-name "${EXP_NAME}" || true

echo "[done] ${EXP_NAME}"
