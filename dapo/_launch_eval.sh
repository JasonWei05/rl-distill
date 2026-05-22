#!/usr/bin/env bash
# Launch 4 parallel vLLM eval instances on this node — one per checkpoint (step 50/100/150/200),
# TP=2 each, using 2 GPUs per instance. Each instance runs all 7 val sets.
#
# Usage: bash _launch_eval.sh <SIZE>    where SIZE is 4b or 12b
set -euo pipefail
SIZE="${1:?SIZE 4b or 12b}"

cd /mlx_devbox/users/jason.wei/playground/rl-distill
source .venv/bin/activate
set -a; source .env; set +a

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_SOCKET_IFNAME=lo
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=lo

case "${SIZE}" in
  4b)
    REPO=JWei05/gemma3-4b-it-off-policy-distilled-from-dapo27b
    BASE=google/gemma-3-4b-it
    ;;
  12b)
    REPO=JWei05/gemma3-12b-it-off-policy-distilled-from-dapo27b
    BASE=google/gemma-3-12b-it
    ;;
  *) echo "unknown SIZE=${SIZE}"; exit 1 ;;
esac

RAY_DATA="${HOME}/verl/data"
OUT="${RAY_DATA}/eval_results"
LOG_DIR="${OUT}/logs"
mkdir -p "${OUT}" "${LOG_DIR}"

VAL=(
  "${RAY_DATA}/math__aime2024_repeated_32x_960.parquet"
  "${RAY_DATA}/math__aime2025_repeated_32x_960.parquet"
  "${RAY_DATA}/math__aime2026_repeated_32x_960.parquet"
  "${RAY_DATA}/math__math_500_repeated_2x_1000.parquet"
  "${RAY_DATA}/math__olympiadbench_repeated_2x.parquet"
  "${RAY_DATA}/math__minervamath_repeated_4x.parquet"
  "${RAY_DATA}/math__gsm8k_test.parquet"
)

# Kill any stale eval / distill / teacher-gen processes
pkill -f "_eval_model_on_math|generate_teacher_data|main_distill_offpolicy" 2>/dev/null || true
sleep 2
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' | sort -u || true)
if [ -n "${PIDS}" ]; then
    echo "killing leftover GPU pids: ${PIDS}"
    echo "${PIDS}" | xargs -r kill -9 2>/dev/null || true
    sleep 2
fi

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo ""
echo "=== launching 4 eval instances for ${SIZE} (TP=2 each) ==="
for i in 0 1 2 3; do
    STEP=$((50 * (i + 1)))
    SUBFOLDER=$(printf "step_%06d" ${STEP})
    GPU_START=$((i * 2))
    GPU_END=$((GPU_START + 1))
    GPUS="${GPU_START},${GPU_END}"

    LOG="${LOG_DIR}/eval_${SIZE}_${SUBFOLDER}.log"
    echo "  ${SIZE} ${SUBFOLDER} on GPUs ${GPUS} -> ${LOG}"

    CUDA_VISIBLE_DEVICES="${GPUS}" nohup python3 "dapo/_eval_model_on_math.py" \
        --repo_id "${REPO}" \
        --subfolder "${SUBFOLDER}" \
        --base_hf_model "${BASE}" \
        --val_files "${VAL[@]}" \
        --output_dir "${OUT}" \
        --tp 2 \
        --temperature 1.0 \
        --top_p 0.7 \
        --max_tokens 20480 \
        > "${LOG}" 2>&1 &

    echo "    PID: $!"
    sleep 20  # stagger the spawn to avoid simultaneous shm contention
done

echo ""
echo "all launched; monitor with:"
echo "  tail -f ${LOG_DIR}/eval_${SIZE}_step_*.log"
