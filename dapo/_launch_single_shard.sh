#!/usr/bin/env bash
# Launch ONE shard on specified GPUs. For recovering the 4th shard on each
# node that silently died during the 4-way parallel startup.
# Usage: bash _launch_single_shard.sh <SHARD_ID> <GPU_PAIR>
#   e.g. bash _launch_single_shard.sh 3 "6,7"
set -euo pipefail

SHARD_ID="${1:?SHARD_ID required}"
GPUS="${2:?GPUS (e.g. '6,7') required}"

cd /mlx_devbox/users/jason.wei/playground/rl-distill
source .venv/bin/activate 2>/dev/null || true
set -a; source .env; set +a

export TEACHER_MODEL=/tmp/teacher_27b_step40
export REVISION=none
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_SOCKET_IFNAME=lo
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=lo

LOG_DIR="${HOME}/verl/data/teacher_gen/logs"
mkdir -p "${LOG_DIR}"

SCRIPT="/mlx_devbox/users/jason.wei/playground/rl-distill/rl-distill-scripts/data/generate_teacher_data.py"
INPUT="${HOME}/verl/data/dapo-math-17k.parquet"
OUTPUT_DIR="${HOME}/verl/data/teacher_gen"

# kill any stale (dead) python proc for this shard_id, just in case
pkill -f "shard_id ${SHARD_ID} " 2>/dev/null || true
sleep 1

echo "=== launching shard ${SHARD_ID} on GPUs ${GPUS} ==="
setsid bash -c "CUDA_VISIBLE_DEVICES='${GPUS}' nohup python3 '${SCRIPT}' \
    --teacher_model '${TEACHER_MODEL}' \
    --revision '${REVISION}' \
    --input_parquet '${INPUT}' \
    --output_dir '${OUTPUT_DIR}' \
    --shard_id '${SHARD_ID}' \
    --num_shards 8 \
    --tp 2 \
    --n 4 \
    > '${LOG_DIR}/shard_${SHARD_ID}.log' 2>&1 &"
sleep 3
pgrep -af "shard_id ${SHARD_ID} " | head -3
