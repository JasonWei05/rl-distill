#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

source "${PROJECT_ROOT}/.venv/bin/activate"

NUM_EXPERTS="${NUM_EXPERTS:-2}"
if [ "${NUM_EXPERTS}" != "2" ] && [ "${NUM_EXPERTS}" != "4" ]; then
    echo "NUM_EXPERTS must be 2 or 4; got ${NUM_EXPERTS}" >&2
    exit 2
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
EP_SIZE="${EP_SIZE:-${NUM_EXPERTS}}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"

MODEL_DIR="${MODEL_DIR:-/tmp/verl/models/gemma-3-4b-pt}"
DATA_DIR="${DATA_DIR:-/tmp/sft_data_nemotron_cascade2_16k}"
PROJECT_NAME="${PROJECT_NAME:-gemma3-moe-sft}"
EXP_NAME="${EXP_NAME:-Gemma3-4B-PT-MoE-${NUM_EXPERTS}E-top1-SFT-16k-$(date +%Y%m%d-%H%M%S)}"
CKPTS_DIR="${CKPTS_DIR:-/tmp/verl/ckpts/${PROJECT_NAME}/${EXP_NAME}}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"

export TOKENIZERS_PARALLELISM=true
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME="${FORCE_NCCL_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_FAMILY="${FORCE_NCCL_SOCKET_FAMILY:-AF_INET}"
export GLOO_SOCKET_IFNAME="${FORCE_GLOO_SOCKET_IFNAME:-lo}"

SITE_PACKAGES="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages"
export CUDNN_HOME="${CUDNN_HOME:-${SITE_PACKAGES}/nvidia/cudnn}"
export NVRTC_HOME="${NVRTC_HOME:-${SITE_PACKAGES}/nvidia/cuda_nvrtc}"
export CURAND_HOME="${CURAND_HOME:-${SITE_PACKAGES}/nvidia/curand}"
export NCCL_HOME="${NCCL_HOME:-${SITE_PACKAGES}/nvidia/nccl}"
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${CUDNN_HOME}/lib:${NVRTC_HOME}/lib:${CURAND_HOME}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [ ! -f "${MODEL_DIR}/config.json" ]; then
    mkdir -p "$(dirname "${MODEL_DIR}")"
    python3 rl-distill-scripts/setup_model.py --size 4b --variant pt --output-dir "${MODEL_DIR}"
fi

if [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/val.parquet" ]; then
    python3 rl-distill-scripts/data/prep_sft_dataset.py \
        --repo JWei05/Nemotron-Cascade-2-SFT-Data-16k-subset \
        --out_dir "${DATA_DIR}"
fi

hydra_args=(
    engine=megatron
    optim=megatron
    "model.path=${MODEL_DIR}"
    model.trust_remote_code=True
    model.enable_gradient_checkpointing=True
    "model.enable_activation_offload=${ENABLE_ACTIVATION_OFFLOAD:-False}"
    model.use_remove_padding=True
    "data.train_files=${DATA_DIR}/train.parquet"
    "data.val_files=${DATA_DIR}/val.parquet"
    data.train_batch_size=64
    "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU:-1}"
    "data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU:-16384}"
    "data.use_dynamic_bsz=${USE_DYNAMIC_BSZ:-False}"
    "data.max_length=${MAX_LENGTH:-16384}"
    data.truncation=right
    data.pad_mode=no_padding
    data.messages_key=messages
    data.train_max_samples=16000
    data.val_max_samples=500
    "data.num_workers=${DATA_NUM_WORKERS:-4}"
    data.ignore_input_ids_mismatch=True
    engine.vanilla_mbridge=False
    "engine.tensor_model_parallel_size=${TP_SIZE}"
    "engine.pipeline_model_parallel_size=${PP_SIZE}"
    "engine.context_parallel_size=${CP_SIZE}"
    "engine.expert_model_parallel_size=${EP_SIZE}"
    engine.expert_tensor_parallel_size=1
    "engine.sequence_parallel=${SEQUENCE_PARALLEL:-False}"
    engine.use_distributed_optimizer=True
    "engine.param_offload=${PARAM_OFFLOAD:-False}"
    "engine.grad_offload=${GRAD_OFFLOAD:-False}"
    "engine.optimizer_offload=${OPTIMIZER_OFFLOAD:-False}"
    engine.override_transformer_config.recompute_granularity=full
    engine.override_transformer_config.recompute_method=uniform
    engine.override_transformer_config.recompute_num_layers=1
    +engine.override_transformer_config.apply_rope_fusion=False
    +engine.override_transformer_config.gradient_accumulation_fusion=False
    "+engine.override_transformer_config.gemma3_moe_num_experts=${NUM_EXPERTS}"
    "+engine.override_transformer_config.gemma3_moe_aux_loss_coeff=${MOE_AUX_LOSS_COEFF:-1e-2}"
    optim.optimizer=adam
    optim.lr=1e-5
    optim.lr_warmup_steps=200
    optim.lr_warmup_init=0.0
    optim.lr_decay_style=constant
    optim.total_training_steps=250
    "optim.betas=[0.9,0.98]"
    +optim.override_optimizer_config.adam_beta1=0.9
    +optim.override_optimizer_config.adam_beta2=0.98
    optim.weight_decay=0.1
    "optim.clip_grad=${CLIP_GRAD:-1.0}"
    trainer.total_epochs=100
    trainer.total_training_steps=250
    "trainer.logger=${TRAINER_LOGGER}"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXP_NAME}"
    "trainer.default_local_dir=${CKPTS_DIR}"
    "trainer.resume_mode=${RESUME_MODE:-disable}"
    "trainer.n_gpus_per_node=${NPROC_PER_NODE}"
    trainer.nnodes=1
    "trainer.test_freq=${TEST_FREQ:-250}"
    "trainer.save_freq=${SAVE_FREQ:-250}"
    "trainer.max_ckpt_to_keep=${MAX_CKPT_TO_KEEP:-1}"
    'checkpoint.save_contents=["model","optimizer","extra"]'
)

python3 -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m verl.trainer.sft_trainer \
    "${hydra_args[@]}" \
    "$@"
