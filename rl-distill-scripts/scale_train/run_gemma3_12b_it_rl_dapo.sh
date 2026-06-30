#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/rl-distill}"
cd "${PROJECT_ROOT}"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export NNODES="${NNODES:-${NUM_INSTANCES:-${SCALE_TRAIN_NUM_NODES:-2}}}"
export NODE_RANK="${NODE_RANK:-${JOB_COMPLETION_INDEX:-0}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export HEAD_ADDR="${MASTER_ADDR:-${LEADER_ADDR:-}}"
export RAY_PORT="${RAY_PORT:-6379}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
export RAY_DATA_HOME="${RAY_DATA_HOME:-/tmp/verl}"
export DATA_DIR="${RAY_DATA_HOME}/data"
export DAPO_TEST_SIZE="${DAPO_TEST_SIZE:-100}"
export DAPO_EVAL_REPEAT="${DAPO_EVAL_REPEAT:-16}"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export MODEL_REPO="${MODEL_REPO:-JWei05/DAPO-Gemma3-12B-PT-TopK128Distill-From-Gemma3-4B-PT-DAPO-17.4k-LR2e6-linear500-1node}"
export MODEL_SUBFOLDER="${MODEL_SUBFOLDER:-step_000500}"
export BASE_MODEL_REPO="${BASE_MODEL_REPO:-google/gemma-3-12b-it}"
export MODEL_LOCAL_DIR="${MODEL_LOCAL_DIR:-/tmp/hf_models/gemma3-12b-topk128-distill-step000500}"
export MODEL_PATH="${MODEL_PATH:-${MODEL_LOCAL_DIR}}"
export PREPARE_MODEL=0
export HF_PUSH_ENABLE="${HF_PUSH_ENABLE:-True}"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"

if [ "${NNODES}" != "1" ] && [ -z "${HEAD_ADDR}" ]; then
    echo "MASTER_ADDR/LEADER_ADDR is required for NNODES=${NNODES}" >&2
    exit 2
fi

mkdir -p "${DATA_DIR}" "$(dirname "${MODEL_LOCAL_DIR}")"

python3 rl-distill-scripts/data/download_hf_subfolder.py \
    --repo-id "${MODEL_REPO}" \
    --subfolder "${MODEL_SUBFOLDER}" \
    --metadata-repo "${BASE_MODEL_REPO}" \
    --output-dir "${MODEL_LOCAL_DIR}"

ray stop --grace-period 30 >/dev/null 2>&1 || true

if [ "${NODE_RANK}" = "0" ]; then
    NODE_IP="${NODE_IP:-$(hostname -I | awk '{print $1}')}"
    HEAD_ADDR="${HEAD_ADDR:-${NODE_IP}}"
    echo "[ray] starting head on ${NODE_IP}:${RAY_PORT}"
    ray start --head \
        --node-ip-address="${NODE_IP}" \
        --port="${RAY_PORT}" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT}" \
        --num-gpus="${NPROC_PER_NODE}" \
        --disable-usage-stats

    echo "[data] preparing DAPO-17.4k split in ${DATA_DIR}"
    DATA_DIR="${DATA_DIR}" bash rl-distill-scripts/data/prepare_dapo_17k_split.sh

    echo "[ray] waiting for ${NNODES} nodes / $((NNODES * NPROC_PER_NODE)) GPUs"
    python3 - <<'PY'
import os
import time

import ray

want_nodes = int(os.environ["NNODES"])
want_gpus = want_nodes * int(os.environ["NPROC_PER_NODE"])
ray.init(address="auto", ignore_reinit_error=True)
for i in range(180):
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    gpus = int(ray.cluster_resources().get("GPU", 0))
    print(f"[ray] alive_nodes={len(nodes)}/{want_nodes} gpus={gpus}/{want_gpus}", flush=True)
    if len(nodes) >= want_nodes and gpus >= want_gpus:
        break
    time.sleep(10)
else:
    raise SystemExit("timed out waiting for Ray workers")
PY

    export RAY_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
    status=0
    bash rl-distill-scripts/gemma3_12b_it_fsdp2_20k.sh "$@" || status=$?
    ray stop --grace-period 30 >/dev/null 2>&1 || true
    exit "${status}"
fi

echo "[ray] worker ${NODE_RANK} joining ${HEAD_ADDR}:${RAY_PORT}"
for _ in $(seq 1 120); do
    if ray start --address="${HEAD_ADDR}:${RAY_PORT}" --num-gpus="${NPROC_PER_NODE}" --disable-usage-stats; then
        break
    fi
    sleep 5
done

while ray status --address="${HEAD_ADDR}:${RAY_PORT}" >/dev/null 2>&1; do
    sleep 60
done
ray stop --grace-period 30 >/dev/null 2>&1 || true
