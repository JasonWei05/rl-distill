#!/usr/bin/env bash
# ScaleTrain entrypoint: Gemma 3 1B PT few-shot math DAPO RL (single node, 2 GPUs).
#
# Launch:
#   python rl-distill-scripts/scale_train/launch_st_job.py \
#     --cluster eks --n-instances 1 --gpus-per-instance 2 --priority high --allow-borrowing \
#     --build-config-key train-rl-distill --team egp --product train.enterprise_rlvr \
#     --job-name gemma3-1b-pt-fewshot-math \
#     --run-file rl-distill-scripts/scale_train/run_gemma3_1b_pt_fewshot_math_rl.sh
set -euxo pipefail
cd /workspace/rl-distill

VENV="${VENV:-/workspace/rl-distill/.venv}"           # FSDP2 stack (torch 2.9 / vllm 0.15.1)
export PATH="${VENV}/bin:${PATH}"; source "${VENV}/bin/activate"
if [ -f .env ]; then set -a; source .env; set +a; fi

export RAY_DATA_HOME="${RAY_DATA_HOME:-/tmp/verl}"; export DATA_DIR="${RAY_DATA_HOME}/data"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"; export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"; export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"

n_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((n_gpus - 1)))"
mkdir -p "${DATA_DIR}"

# model (gated; HF_TOKEN from .env) + data (from HF, idempotent)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-3-1b-pt')"
DATA_DIR="${DATA_DIR}" bash rl-distill-scripts/data/prepare_fewshot_math_rl_data.sh

# single-node ray
ray stop --grace-period 30 >/dev/null 2>&1 || true
NODE_IP="$(hostname -I | awk '{print $1}')"
ray start --head --node-ip-address="${NODE_IP}" --port=6379 \
    --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus="${n_gpus}" --disable-usage-stats
export RAY_ADDRESS="http://127.0.0.1:8265"

status=0
MODEL_TAG=gemma3-1b-pt \
MODEL_REPO=google/gemma-3-1b-pt \
MODEL_PATH=google/gemma-3-1b-pt \
HF_PUSH_REPO="${HF_PUSH_REPO:-JWei05/DAPO-Gemma3-1B-PT-FewShotMath}" \
N_GPUS_PER_NODE="${n_gpus}" NNODES=1 OFFLOAD="${OFFLOAD:-False}" \
DATA_DIR="${DATA_DIR}" CKPTS_DIR="${RAY_DATA_HOME}/ckpts/gemma3-1b-pt-fewshot-math" \
    bash rl-distill-scripts/gemma3_pt_fewshot_math_rl.sh "$@" || status=$?
ray stop --grace-period 30 >/dev/null 2>&1 || true
exit "${status}"
