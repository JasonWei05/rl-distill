#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

source .venv/bin/activate
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

export DAPO_LOCAL_TASK_RUNNER=1
export HF_HOME=${HF_HOME:-"/tmp/hf_cache"}
export HF_HUB_CACHE=${HF_HUB_CACHE:-"${HF_HOME}/hub"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"${HF_HOME}"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"/tmp/.cache"}
export WANDB_DIR=${WANDB_DIR:-"/tmp/wandb"}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-"/tmp/wandb/cache"}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-"/tmp/wandb/config"}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-"/tmp/vllm_cache"}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-"/tmp/torchinductor_cache"}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"/tmp/triton_cache"}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-"/tmp/cuda_cache"}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-"/tmp/flashinfer"}
export FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_DIR:-"/tmp/flashinfer_cubins"}
export MODEL_PATH=${MODEL_PATH:-"/tmp/hf_models/gemma-3-27b-pt"}
export EXP_NAME=${EXP_NAME:-"DAPO-Gemma3-27B-PT-FSDP2-$(date +%Y%m%d-%H%M)"}
export HF_PUSH_REPO=${HF_PUSH_REPO:-"JWei05/dapo-gemma3-27b-pt"}
export GEMMA3_CHAT_TEMPLATE_FILE=${GEMMA3_CHAT_TEMPLATE_FILE:-"${PROJECT_ROOT}/rl-distill-scripts/data/gemma3_it_chat_template.jinja"}
export NNODES=${NNODES:-2}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET6}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}

bash rl-distill-scripts/gemma3_27b_it_fsdp2_20k.sh \
    +ray_kwargs.ray_init.address="'${RAY_ADDRESS}'" \
    actor_rollout_ref.model.custom_chat_template="@${GEMMA3_CHAT_TEMPLATE_FILE}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.save_freq=5 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    ++trainer.hf_push.freq=20 \
    "$@"
