#!/usr/bin/env bash
set -euo pipefail

NODE_ID="${1:?usage: launch_teacher_dp8_torchrun_node.sh NODE_ID}"

cd /mlx_devbox/users/jason.wei/playground/rl-distill
source .venv/bin/activate
set -a
source .env
set +a

export VLLM_FORCE_PLATFORM="${VLLM_FORCE_PLATFORM:-cuda}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

OUTPUT_DIR="${OUTPUT_DIR:-/home/tiger/verl/data/dapo_gemma3_4b_pt_teacher_v6_torchrun_dp8/train_n16}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

nohup torchrun --standalone --nproc-per-node=8 \
  rl-distill-scripts/data/generate_teacher_data.py \
  --teacher_model "${TEACHER_MODEL:-google/gemma-3-4b-pt}" \
  --revision "${REVISION:-main}" \
  --input_parquet "${INPUT:-/home/tiger/verl/data/dapo_17_4k_train.parquet}" \
  --output_dir "${OUTPUT_DIR}" \
  --shard_id "${NODE_ID}" \
  --num_shards "${TOTAL_SHARDS:-2}" \
  --tp 1 \
  --dp 8 \
  --distributed_executor_backend external_launcher \
  --n "${N:-16}" \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --chat_template "${CHAT_TEMPLATE:-/mlx_devbox/users/jason.wei/playground/rl-distill/rl-distill-scripts/data/gemma3_it_chat_template.jinja}" \
  > "${LOG_DIR}/node${NODE_ID}_torchrun.log" 2>&1 &

echo "PID:$!"
echo "LOG:${LOG_DIR}/node${NODE_ID}_torchrun.log"
