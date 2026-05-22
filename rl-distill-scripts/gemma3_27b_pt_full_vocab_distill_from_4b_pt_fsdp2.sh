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

export HF_HOME=${HF_HOME:-"/tmp/hf_cache"}
export HF_HUB_CACHE=${HF_HUB_CACHE:-"${HF_HOME}/hub"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"${HF_HOME}"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"/tmp/.cache"}
export WANDB_DIR=${WANDB_DIR:-"/tmp/wandb"}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-"/tmp/wandb/cache"}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-"/tmp/wandb/config"}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-"/tmp/torchinductor_cache"}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"/tmp/triton_cache"}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-"/tmp/cuda_cache"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

export CUDA_HOME=${CUDA_HOME:-"/tmp/cuda-no-recursion-real"}
export CUDA_PATH=${CUDA_PATH:-"${CUDA_HOME}"}
export CUDA_INC_PATH=${CUDA_INC_PATH:-"${CUDA_HOME}/include"}
export CUDACXX=${CUDACXX:-"${CUDA_HOME}/bin/nvcc"}

NVML_SHIM_DIR=${NVML_SHIM_DIR:-}
if [ -z "${NVML_SHIM_DIR}" ]; then
    if [ -e /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.129.03 ]; then
        mkdir -p /tmp/nvidia-nvml-535
        ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.129.03 /tmp/nvidia-nvml-535/libnvidia-ml.so.1
    fi
    if [ -e /tmp/nvidia-nvml-535/libnvidia-ml.so.1 ]; then
        NVML_SHIM_DIR=/tmp/nvidia-nvml-535
    elif [ -e /tmp/nvidia-nvml-570/libnvidia-ml.so.1 ]; then
        NVML_SHIM_DIR=/tmp/nvidia-nvml-570
    elif [ -e /tmp/nvidia-nvml-580/libnvidia-ml.so.1 ]; then
        NVML_SHIM_DIR=/tmp/nvidia-nvml-580
    fi
fi
if [ -n "${NVML_SHIM_DIR}" ]; then
    export NVML_SHIM_DIR
fi

NVIDIA_LIB_ROOT="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages/nvidia"
CUDA_LIBRARY_PATHS=(
    "${NVIDIA_LIB_ROOT}/nccl/lib"
    "${NVIDIA_LIB_ROOT}/cudnn/lib"
    "${NVIDIA_LIB_ROOT}/cuda_nvrtc/lib"
    "${NVIDIA_LIB_ROOT}/curand/lib"
    "${NVIDIA_LIB_ROOT}/cublas/lib"
    "${NVIDIA_LIB_ROOT}/cuda_runtime/lib"
    "${NVIDIA_LIB_ROOT}/nvshmem/lib"
    /usr/lib/x86_64-linux-gnu/openmpi
    /usr/lib
    /lib64
    /usr/local/lib
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda/lib64
)
if [ -n "${NVML_SHIM_DIR}" ]; then
    CUDA_LIBRARY_PATHS=("${NVML_SHIM_DIR}" "${CUDA_LIBRARY_PATHS[@]}")
fi
export LD_LIBRARY_PATH="$(IFS=:; echo "${CUDA_LIBRARY_PATHS[*]}"):${LD_LIBRARY_PATH:-}"

export TRAIN_FILE=${TRAIN_FILE:-"/home/tiger/verl/data/dapo_gemma3_4b_pt_teacher_v7_independent/train.parquet"}
export VAL_FILE=${VAL_FILE:-"/home/tiger/verl/data/dapo_gemma3_4b_pt_teacher_v7_independent/validation.parquet"}
export MODEL_PATH=${MODEL_PATH:-"/tmp/hf_models/gemma-3-27b-pt"}
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"/tmp/rl_fresh_moe_models/gemma-3-4b-pt-dense"}
export PROJECT_NAME=${PROJECT_NAME:-"full-vocab-distill"}
export EXP_NAME=${EXP_NAME:-"Gemma3-27B-PT-FullVocabDistill-From-4B-PT-DAPO17k-$(date +%Y%m%d-%H%M)"}
export CKPTS_DIR=${CKPTS_DIR:-"/tmp/verl/ckpts/${PROJECT_NAME}/${EXP_NAME}"}
export HF_PUSH_REPO=${HF_PUSH_REPO:-"JWei05/DAPO-Gemma3-27B-PT-FullVocabDistill-From-Gemma3-4B-PT-DAPO-17.4k"}

export NNODES=${NNODES:-1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_PORT=${MASTER_PORT:-29571}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET6}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE:-0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

cd "${PROJECT_ROOT}/rl-distill-scripts"

if [ "${NNODES}" = "1" ]; then
    exec torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NPROC_PER_NODE}" \
        main_full_vocab_distill_fsdp2.py \
        "$@"
fi

if [ -z "${MASTER_ADDR:-}" ]; then
    echo "MASTER_ADDR must be set for NNODES=${NNODES}" >&2
    exit 2
fi

exec torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    main_full_vocab_distill_fsdp2.py \
    "$@"
