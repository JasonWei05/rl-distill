#!/usr/bin/env bash
# ScaleTrain entrypoint: Gemma 3 4B PT DeepScaleR DAPO RL (single node, full 8-GPU H100 node).
# Identical recipe to run_gemma3_4b_pt_fewshot_math_rl.sh (unified 12-shot prompt, 20k response +
# 4k overlong buffer, val=train sampling, SAVE_FREQ=25 + HF push /25, wandb) via the shared core
# gemma3_pt_fewshot_math_rl.sh — only the dataset changes: trains on the DeepScaleR train split
# (40.1k) with a 200-question x16 val (pass@16/maj@16/mean@16). Data pulled from JWei05/DeepScaleR-RL.
#
# Launch:
#   python rl-distill-scripts/scale_train/launch_st_job.py \
#     --cluster eks --n-instances 1 --gpus-per-instance 8 --priority high --allow-borrowing \
#     --build-config-key train-rl-distill --team egp --product train.enterprise_rlvr \
#     --job-name gemma3-4b-pt-deepscaler \
#     --run-file run_gemma3_4b_pt_deepscaler_rl.sh
set -euxo pipefail
cd /workspace/rl-distill

VENV="${VENV:-/workspace/rl-distill/.venv}"           # FSDP2 stack (torch 2.9 / vllm 0.15.1)
export PATH="${VENV}/bin:${PATH}"; source "${VENV}/bin/activate"
if [ -f .env ]; then set -a; source .env; set +a; fi

export RAY_DATA_HOME="${RAY_DATA_HOME:-/tmp/verl}"; export DATA_DIR="${RAY_DATA_HOME}/data"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"; export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"; export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
# ScaleTrain pods on ml-gpu-batch are IPv6-only (pod IP e.g. 2602:fb33:...), interface eth0.
# Forcing AF_INET (IPv4) makes NCCL filter out the IPv6-only eth0 -> "Bootstrap : no socket
# interface found". Use eth0 + AF_INET6 (matches CLAUDE.md's multi-node = bond0/AF_INET6).
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET6}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"

n_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((n_gpus - 1)))"
mkdir -p "${DATA_DIR}"

# model (gated; HF_TOKEN from .env) + data (from JWei05/DeepScaleR-RL, idempotent)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-3-4b-pt')"
DATA_DIR="${DATA_DIR}" bash rl-distill-scripts/data/prepare_deepscaler_rl_data.sh

status=0
MODEL_TAG=gemma3-4b-pt MODEL_REPO=google/gemma-3-4b-pt MODEL_PATH=google/gemma-3-4b-pt \
HF_PUSH_REPO="${HF_PUSH_REPO:-JWei05/DAPO-Gemma3-4B-PT-DeepScaleR}" \
EXP_NAME="${EXP_NAME:-DAPO-gemma3-4b-pt-DeepScaleR-$(date +%Y%m%d-%H%M)}" \
TRAIN_FILE="${DATA_DIR}/deepscaler_rl_train.parquet" \
VAL_FILES="['${DATA_DIR}/deepscaler_rl_val200_x16.parquet']" \
GEMMA3_CHAT_TEMPLATE_FILE="${GEMMA3_CHAT_TEMPLATE_FILE:-${PWD}/rl-distill-scripts/data/gemma3_it_fewshot_math.jinja}" \
N_GPUS_PER_NODE="${n_gpus}" NNODES=1 OFFLOAD="${OFFLOAD:-True}" SAVE_FREQ="${SAVE_FREQ:-25}" \
RAY_ADDRESS=local VERL_VLLM_PORT_BASE="${VERL_VLLM_PORT_BASE:-52000}" \
DATA_DIR="${DATA_DIR}" CKPTS_DIR="${RAY_DATA_HOME}/ckpts/gemma3-4b-pt-deepscaler" \
    bash rl-distill-scripts/gemma3_pt_fewshot_math_rl.sh \
      +ray_kwargs.ray_init._temp_dir=/tmp/ray_4b_deepscaler \
      +ray_kwargs.ray_init.include_dashboard=False \
      "$@" || status=$?
echo "RUN_DONE rc=${status}"
exit "${status}"
