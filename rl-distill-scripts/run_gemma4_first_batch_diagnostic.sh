#!/usr/bin/env bash
set -euo pipefail

# Reproduce the first E2B-overlay -> E4B production batch without W&B, HF
# mutation, validation, or checkpoint output.  The deliberately tiny gradient
# threshold stops after backward and before the optimizer/save path.  This is
# diagnostic-only: production remains gated by the signed FSDP2 audit receipt.

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"${PROJECT_ROOT}/.venv-gemma4/bin/python"}
TORCHRUN_BIN=${TORCHRUN_BIN:-"${PROJECT_ROOT}/.venv-gemma4/bin/torchrun"}
MODEL_PATH=${MODEL_PATH:-"/home/ubuntu/.cache/huggingface/models--google--gemma-4-E4B/snapshots/411aa17b749aa952df1359d2dcea73917a544d9a"}
DATASET_INDEX=${DATASET_INDEX:-"/tmp/verl/datasets/gemma4-e2b-base-topk128-hf-overlay-v128-seed42/dataset_index.json"}
SOURCE_DATASET_INDEX=${SOURCE_DATASET_INDEX:-"/tmp/verl/datasets/gemma4-e2b-base-topk128-traces-e32aaa02681a-val128-seed42/dataset_index.json"}
PREFLIGHT_RECEIPT_CACHE=${PREFLIGHT_RECEIPT_CACHE:-"$(dirname "${DATASET_INDEX}")/training_preflight_receipt.json"}

FSDP_PARAM_DTYPE=${FSDP_PARAM_DTYPE:-bf16}
FSDP_CAST_FORWARD_INPUTS=${FSDP_CAST_FORWARD_INPUTS:-true}
MODEL_GRADIENT_CHECKPOINTING=${MODEL_GRADIENT_CHECKPOINTING:-true}
DISTILL_CHUNK_CHECKPOINTING=${DISTILL_CHUNK_CHECKPOINTING:-true}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-2}
GEMMA4_CUDNN_SDPA=${GEMMA4_CUDNN_SDPA:-1}
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-false}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-12288}
MAX_PADDED_TOKENS_PER_MICROBATCH=${MAX_PADDED_TOKENS_PER_MICROBATCH:-0}
DIAGNOSTIC_LABEL=${DIAGNOSTIC_LABEL:-"param-${FSDP_PARAM_DTYPE}-inputcast-${FSDP_CAST_FORWARD_INPUTS}-modelckpt-${MODEL_GRADIENT_CHECKPOINTING}-chunkckpt-${DISTILL_CHUNK_CHECKPOINTING}-cudnn${GEMMA4_CUDNN_SDPA}-dynamic${USE_DYNAMIC_BSZ}-tok${MAX_TOKEN_LEN_PER_GPU}-paddedtok${MAX_PADDED_TOKENS_PER_MICROBATCH}-mb${MICRO_BATCH_SIZE_PER_GPU}"}
DIAGNOSTIC_ROOT=${DIAGNOSTIC_ROOT:-"/tmp/verl/first-batch-diagnostics"}
OUTPUT_DIR="${DIAGNOSTIC_ROOT}/${DIAGNOSTIC_LABEL}"
LOG_PATH="${OUTPUT_DIR}/train.log"
GRAD_PATH="${OUTPUT_DIR}/grad.json"
MICROBATCH_GRAD_PATH="${OUTPUT_DIR}/microbatch_grads.jsonl"

for value_name in FSDP_CAST_FORWARD_INPUTS MODEL_GRADIENT_CHECKPOINTING DISTILL_CHUNK_CHECKPOINTING USE_DYNAMIC_BSZ; do
    case "${!value_name}" in
        true|false) ;;
        *) echo "${value_name} must be true or false" >&2; exit 2 ;;
    esac
done
case "${FSDP_PARAM_DTYPE}" in
    fp32|bf16) ;;
    *) echo "FSDP_PARAM_DTYPE must be fp32 or bf16" >&2; exit 2 ;;
esac
if ! [[ "${MICRO_BATCH_SIZE_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MICRO_BATCH_SIZE_PER_GPU must be a positive integer" >&2
    exit 2
fi
if ! [[ "${MAX_TOKEN_LEN_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_TOKEN_LEN_PER_GPU must be a positive integer" >&2
    exit 2
fi
if ! [[ "${MAX_PADDED_TOKENS_PER_MICROBATCH}" =~ ^[0-9]+$ ]]; then
    echo "MAX_PADDED_TOKENS_PER_MICROBATCH must be a non-negative integer" >&2
    exit 2
fi
case "${GEMMA4_CUDNN_SDPA}" in
    0|1) ;;
    *) echo "GEMMA4_CUDNN_SDPA must be 0 or 1" >&2; exit 2 ;;
esac
if [ -e "${OUTPUT_DIR}" ]; then
    echo "Diagnostic output already exists: ${OUTPUT_DIR}" >&2
    exit 2
fi
mkdir -p "${OUTPUT_DIR}"

PREFLIGHT_OUTPUT=$("${PYTHON_BIN}" "${PROJECT_ROOT}/rl-distill-scripts/data/preflight_gemma4_training_topk_overlay.py" \
    --dataset-index "${DATASET_INDEX}" \
    --source-dataset-index "${SOURCE_DATASET_INDEX}" \
    --student-model "${MODEL_PATH}" \
    --expected-direction e2b_base_to_e4b \
    --expected-teacher-identity-sha256 2d48d343709dcae087d6ff2def9f09d2950ca66dc2183a8bee38850c4ddbbb36 \
    --expected-student-identity-sha256 acdc0d2bcb8f676593b5387807da1cd1b84a9e26fa279db4a86f54a211055b2d \
    --expected-train-questions 9723 \
    --expected-validation-questions 128 \
    --expected-train-samples-per-question 5 \
    --expected-validation-samples-per-question 1 \
    --receipt-cache "${PREFLIGHT_RECEIPT_CACHE}" \
    --local-files-only)

TRAIN_FILES_HYDRA=
VAL_FILES_HYDRA=
while IFS='=' read -r key value; do
    case "${key}" in
        TRAIN_FILES_HYDRA) TRAIN_FILES_HYDRA=${value} ;;
        VAL_FILES_HYDRA) VAL_FILES_HYDRA=${value} ;;
    esac
done <<< "${PREFLIGHT_OUTPUT}"
if [ -z "${TRAIN_FILES_HYDRA}" ] || [ -z "${VAL_FILES_HYDRA}" ]; then
    echo "Preflight did not return train and validation file lists" >&2
    exit 2
fi

export HF_HOME=${HF_HOME:-/tmp/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-"${HF_HOME}/hub"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/.cache}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VERL_FSDP2_LOCAL_LOAD=${VERL_FSDP2_LOCAL_LOAD:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE:-0}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export VERL_FAIL_ON_NONFINITE_LOSS=1
export VERL_FAIL_ON_NONFINITE_GRAD=1
export VERL_MAX_PRECLIP_GRAD_NORM=0.000001
export VERL_FSDP2_GRAD_DIAGNOSTICS=1
export VERL_FSDP2_GRAD_DIAGNOSTICS_TOPK=30
export VERL_FSDP2_GRAD_DIAGNOSTICS_PATH="${GRAD_PATH}"
export VERL_FSDP2_GRAD_DIAGNOSTICS_EACH_MICROBATCH=${VERL_FSDP2_GRAD_DIAGNOSTICS_EACH_MICROBATCH:-0}
export VERL_FSDP2_MICROBATCH_GRAD_DIAGNOSTICS_PATH="${MICROBATCH_GRAD_PATH}"
export VERL_GEMMA4_CUDNN_SDPA="${GEMMA4_CUDNN_SDPA}"

set +e
cd "${PROJECT_ROOT}/rl-distill-scripts"
"${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node=8 \
    main_full_vocab_distill_fsdp2.py \
    teacher_model.path=null \
    teacher_model.precomputed_topk=true \
    teacher_model.top_k=128 \
    teacher_model.chunk_size=4096 \
    teacher_model.temperature=1.0 \
    teacher_model.clamp_min_kl=false \
    teacher_model.checkpoint_student_chunks="${DISTILL_CHUNK_CHECKPOINTING}" \
    data.train_files="${TRAIN_FILES_HYDRA}" \
    data.val_files="${VAL_FILES_HYDRA}" \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    data.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    data.max_padded_tokens_per_microbatch="${MAX_PADDED_TOKENS_PER_MICROBATCH}" \
    data.max_length=12288 \
    data.truncation=error \
    data.use_precomputed_topk=true \
    data.teacher_topk_width=128 \
    data.teacher_topk_validation_tolerance=0.0025 \
    data.require_exact_val_coverage=true \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=false \
    model.enable_gradient_checkpointing="${MODEL_GRADIENT_CHECKPOINTING}" \
    model.override_config.attn_implementation=sdpa \
    engine.fsdp_size=-1 \
    engine.model_dtype=fp32 \
    engine.use_torch_compile=false \
    'engine.wrap_policy.transformer_layer_cls_to_wrap=["Gemma4TextDecoderLayer"]' \
    optim.lr=2e-6 \
    optim.lr_warmup_steps=100 \
    optim.total_training_steps=1 \
    optim.lr_scheduler_type=linear \
    optim.min_lr_ratio=0.1 \
    optim.weight_decay=0.1 \
    'optim.betas=[0.9,0.98]' \
    trainer.project_name=gemma4-first-batch-diagnostic \
    trainer.experiment_name="${DIAGNOSTIC_LABEL}" \
    trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
    trainer.total_epochs=2 \
    trainer.total_training_steps=1 \
    'trainer.logger=["console"]' \
    trainer.save_freq=999 \
    trainer.test_freq=999 \
    trainer.val_before_train=false \
    trainer.resume_mode=disable \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.hf_push.enable=false \
    '+engine.mixed_precision={param_dtype:'"${FSDP_PARAM_DTYPE}"',reduce_dtype:fp32,buffer_dtype:fp32,cast_forward_inputs:'"${FSDP_CAST_FORWARD_INPUTS}"'}' \
    >"${LOG_PATH}" 2>&1
STATUS=$?
set -e

if [ ! -s "${GRAD_PATH}" ]; then
    echo "Diagnostic did not reach post-backward gradient collection; status=${STATUS}, log=${LOG_PATH}" >&2
    exit "${STATUS}"
fi
"${PYTHON_BIN}" -c 'import json,sys; value=json.load(open(sys.argv[1])); print(json.dumps({"torch_total_norm": value["torch_total_norm"], "manual_total_norm": value["manual_total_norm"], "top": value["top"][:10]}, indent=2))' "${GRAD_PATH}"
echo "Expected diagnostic stop status=${STATUS}; log=${LOG_PATH}; gradients=${GRAD_PATH}"
