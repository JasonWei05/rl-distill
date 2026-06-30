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

export PREPARE_INPUTS="${PREPARE_INPUTS:-1}"
export NNODES="${NNODES:-${NUM_INSTANCES:-${SCALE_TRAIN_NUM_NODES:-1}}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NODE_RANK="${NODE_RANK:-${JOB_COMPLETION_INDEX:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-${LEADER_ADDR:-}}"
export MASTER_PORT="${MASTER_PORT:-${LEADER_PORT:-29571}}"
export TEACHER_TOP_K="${TEACHER_TOP_K:-128}"
export TRAIN_LOGGER="${TRAIN_LOGGER:-[\"console\",\"wandb\"]}"
export HF_PUSH_ENABLE="${HF_PUSH_ENABLE:-true}"

if [ "${NNODES}" != "1" ]; then
    if [ -z "${MASTER_ADDR}" ]; then
        echo "MASTER_ADDR/LEADER_ADDR is required for NNODES=${NNODES}" >&2
        exit 2
    fi
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"
fi

exec bash rl-distill-scripts/gemma3_12b_pt_topk128_distill_from_4b_pt_fsdp2.sh "$@"
