#!/usr/bin/env bash
# Launch teacher-gen with the pre-downloaded LOCAL teacher path baked in.
# Bypasses env-var propagation issues through the mlx-login -> nohup chain.
# Usage: bash _launch_teacher_gen_local.sh <NODE_ID>   where NODE_ID is 0 or 1
set -euo pipefail
NODE_ID="${1:?NODE_ID (0 or 1) required}"

# Baked-in, non-overridable
export TEACHER_MODEL=/tmp/teacher_27b_step40
export REVISION=none
# vLLM v1 multiproc workers fail to initialize CUDA if parent has already
# touched torch.cuda and the spawn method is "fork". Force spawn.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Force NCCL to loopback (IPv4) for the per-shard TP=2 group.
# Without this, NCCL auto-picks bond0 (IPv6) and fails intra-node init.
export NCCL_SOCKET_IFNAME=lo
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=lo

cd /mlx_devbox/users/jason.wei/playground/rl-distill
source .venv/bin/activate 2>/dev/null || true

# cleanup
ray stop --grace-period 20 2>/dev/null | tail -2 || true
sleep 2
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' | sort -u || true)
if [ -n "${PIDS}" ]; then
    echo "killing GPU pids: ${PIDS}"
    echo "${PIDS}" | xargs -r kill -9 2>/dev/null || true
    sleep 2
fi
# also kill any leftover generate_teacher_data.py procs from prior crashes
pkill -f generate_teacher_data.py 2>/dev/null || true
sleep 1

echo "=== GPU before launch ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo ""
echo "=== launching teacher-gen, NODE_ID=${NODE_ID}, TEACHER_MODEL=${TEACHER_MODEL} ==="
# Sanity-check local path before forking workers
if [ ! -f "${TEACHER_MODEL}/config.json" ]; then
    echo "ERROR: ${TEACHER_MODEL}/config.json not found; refusing to launch."
    exit 1
fi

bash rl-distill-scripts/data/launch_teacher_gen.sh "${NODE_ID}"
