#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
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

if [ -d /usr/local/cuda ]; then
    export CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda"}
else
    export CUDA_HOME=${CUDA_HOME:-"/tmp/cuda-no-recursion-real"}
fi
export CUDA_PATH=${CUDA_PATH:-"${CUDA_HOME}"}
export CUDA_INC_PATH=${CUDA_INC_PATH:-"${CUDA_HOME}/include"}
export CUDACXX=${CUDACXX:-"${CUDA_HOME}/bin/nvcc"}

NVML_SHIM_DIR=${NVML_SHIM_DIR:-}
if [ -z "${NVML_SHIM_DIR}" ]; then
    for shim in /tmp/nvidia-nvml-580 /tmp/nvidia-nvml-570 /tmp/nvidia-nvml-535; do
        if [ -e "${shim}/libnvidia-ml.so.1" ]; then
            NVML_SHIM_DIR="${shim}"
            break
        fi
    done
fi
if [ -n "${NVML_SHIM_DIR}" ]; then
    export NVML_SHIM_DIR
fi

PY_SITE="$(python3 - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
NVIDIA_LIB_ROOT="${PY_SITE}/nvidia"
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

export DATASET_REPO=${DATASET_REPO:-"JWei05/DAPO-Gemma3-4B-PT-DAPO-17.4k"}
export DATA_SPLIT_SEED=${DATA_SPLIT_SEED:-42}
export VAL_ROW_COUNT=${VAL_ROW_COUNT:-${VAL_PROMPT_COUNT:-500}}
export DATA_DIR=${DATA_DIR:-"/tmp/verl/data/dapo_gemma3_4b_pt_teacher_row_split_seed${DATA_SPLIT_SEED}_eval${VAL_ROW_COUNT}"}
export SOURCE_DATA_FILE=${SOURCE_DATA_FILE:-"${DATA_DIR}/source_train.parquet"}
export TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/train.parquet"}
export VAL_FILE=${VAL_FILE:-"${DATA_DIR}/validation.parquet"}
export HELDOUT_ROW_IDX_JSON=${HELDOUT_ROW_IDX_JSON:-"${DATA_DIR}/heldout_source_row_idx_seed${DATA_SPLIT_SEED}_n${VAL_ROW_COUNT}.json"}
export HELDOUT_ROW_IDX_TXT=${HELDOUT_ROW_IDX_TXT:-"${DATA_DIR}/heldout_source_row_idx_seed${DATA_SPLIT_SEED}_n${VAL_ROW_COUNT}.txt"}
export MODEL_PATH=${MODEL_PATH:-"/tmp/hf_models/gemma-3-12b-pt"}
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"/tmp/hf_models/gemma-3-4b-pt"}
export PROJECT_NAME=${PROJECT_NAME:-"topk-distill"}
export EXP_NAME=${EXP_NAME:-"Gemma3-12B-PT-TopK128Distill-From-4B-PT-DAPO17k-$(date +%Y%m%d-%H%M)"}
export CKPTS_DIR=${CKPTS_DIR:-"/tmp/verl/ckpts/${PROJECT_NAME}/${EXP_NAME}"}
export HF_PUSH_REPO=${HF_PUSH_REPO:-"JWei05/DAPO-Gemma3-12B-PT-TopK128Distill-From-Gemma3-4B-PT-DAPO-17.4k"}

export TEACHER_TOP_K=${TEACHER_TOP_K:-128}
export FULL_VOCAB_KL_CHUNK_SIZE=${FULL_VOCAB_KL_CHUNK_SIZE:-16}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
export MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-20480}
export MAX_LENGTH=${MAX_LENGTH:-20480}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-4000}
export LR=${LR:-1e-5}
export LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-100}
export LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
export MIN_LR_RATIO=${MIN_LR_RATIO:-0.0}
export ENGINE_FSDP_SIZE=${ENGINE_FSDP_SIZE:-8}
export ENGINE_MODEL_DTYPE=${ENGINE_MODEL_DTYPE:-bfloat16}
export ULYSSES_SEQUENCE_PARALLEL_SIZE=${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}
export SAVE_FREQ=${SAVE_FREQ:-500}
export TEST_FREQ=${TEST_FREQ:-5}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
export MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-1}
export HF_PUSH_ENABLE=${HF_PUSH_ENABLE:-true}
export HF_PUSH_MAX_TO_KEEP=${HF_PUSH_MAX_TO_KEEP:-8}

export NNODES=${NNODES:-1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_PORT=${MASTER_PORT:-29571}
if [ "${NNODES}" = "1" ]; then
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
    export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
else
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
    export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET6}
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
fi
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE:-0}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

if [ "${PREPARE_INPUTS:-1}" = "1" ]; then
    mkdir -p "${DATA_DIR}" "$(dirname "${MODEL_PATH}")" "$(dirname "${TEACHER_MODEL_PATH}")"
    if [ ! -f "${TRAIN_FILE}" ] || [ ! -f "${VAL_FILE}" ] || [ ! -f "${HELDOUT_ROW_IDX_JSON}" ]; then
        python3 rl-distill-scripts/data/split_dapo_gemma3_4b_pt_by_prompt_idx.py \
            --repo-id "${DATASET_REPO}" \
            --filename data/train.parquet \
            --input-parquet "${SOURCE_DATA_FILE}" \
            --output-dir "${DATA_DIR}" \
            --train-output "${TRAIN_FILE}" \
            --val-output "${VAL_FILE}" \
            --heldout-output-json "${HELDOUT_ROW_IDX_JSON}" \
            --heldout-output-txt "${HELDOUT_ROW_IDX_TXT}" \
            --num-val-rows "${VAL_ROW_COUNT}" \
            --seed "${DATA_SPLIT_SEED}"
    fi
    if [ ! -f "${MODEL_PATH}/config.json" ]; then
        python3 rl-distill-scripts/setup_model.py --size 12b --variant pt --output-dir "${MODEL_PATH}"
    fi
    if [ ! -f "${TEACHER_MODEL_PATH}/config.json" ]; then
        python3 rl-distill-scripts/setup_model.py --size 4b --variant pt --output-dir "${TEACHER_MODEL_PATH}"
    fi
fi

cd "${PROJECT_ROOT}/rl-distill-scripts"

COMMON_OVERRIDES=(
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}"
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}"
    data.max_length="${MAX_LENGTH}"
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.val_before_train="${VAL_BEFORE_TRAIN}"
    trainer.max_ckpt_to_keep="${MAX_CKPT_TO_KEEP}"
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NPROC_PER_NODE}"
    engine.fsdp_size="${ENGINE_FSDP_SIZE}"
    engine.model_dtype="${ENGINE_MODEL_DTYPE}"
    engine.ulysses_sequence_parallel_size="${ULYSSES_SEQUENCE_PARALLEL_SIZE}"
    optim.lr="${LR}"
    optim.lr_warmup_steps="${LR_WARMUP_STEPS}"
    optim.total_training_steps="${TOTAL_TRAINING_STEPS}"
    optim.lr_scheduler_type="${LR_SCHEDULER_TYPE}"
    optim.min_lr_ratio="${MIN_LR_RATIO}"
)

if [ "${NNODES}" = "1" ]; then
    exec torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node="${NPROC_PER_NODE}" \
        main_full_vocab_distill_fsdp2.py \
        "${COMMON_OVERRIDES[@]}" \
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
    "${COMMON_OVERRIDES[@]}" \
    "$@"
