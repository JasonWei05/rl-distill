#!/usr/bin/env bash
# Evaluate the original Gemma 3 IT base models (no distillation) on the same
# 7 math val sets, using the same sampling params as the distilled eval so we
# get an apples-to-apples baseline.
#
# Layout: 1 node × 8 GPUs. 2 vLLM instances, TP=2 each:
#   - google/gemma-3-4b-it  on GPUs 0,1
#   - google/gemma-3-12b-it on GPUs 2,3
# (4 GPUs idle — not enough demand to fill them.)
set -euo pipefail

cd /mlx_devbox/users/jason.wei/playground/rl-distill
source .venv/bin/activate
set -a; source .env; set +a

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_SOCKET_IFNAME=lo
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=lo

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

# Reuse same sampling params we used for the distilled eval (T=0.7, top_p=0.95).
# (Match these to the distilled launch — kept consistent for direct comparison.)
TEMP=0.7
TOPP=0.95
MAX_TOK=20480

# Clean any stale eval procs / GPU pids
pkill -f "_eval_model_on_math" 2>/dev/null || true
sleep 2
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' | sort -u || true)
[ -n "${PIDS}" ] && echo "${PIDS}" | xargs -r kill -9 2>/dev/null || true
sleep 2

echo "=== launching base-IT eval (2 instances, TP=2) ==="
for spec in "0,1:google/gemma-3-4b-it:base_4b_it" "2,3:google/gemma-3-12b-it:base_12b_it"; do
    GPUS=${spec%%:*}
    rest=${spec#*:}
    REPO=${rest%%:*}
    TAG=${rest##*:}
    LOG="${LOG_DIR}/eval_${TAG}.log"
    echo "  ${REPO} on GPUs ${GPUS} -> ${LOG}"

    CUDA_VISIBLE_DEVICES="${GPUS}" nohup python3 dapo/_eval_model_on_math.py \
        --repo_id "${REPO}" \
        --base_hf_model "${REPO}" \
        --val_files "${VAL[@]}" \
        --output_dir "${OUT}" \
        --tp 2 \
        --temperature "${TEMP}" \
        --top_p "${TOPP}" \
        --max_tokens "${MAX_TOK}" \
        > "${LOG}" 2>&1 &
    echo "    PID: $!"
    sleep 20  # stagger
done
echo ""
echo "logs at ${LOG_DIR}/eval_base_*.log"
