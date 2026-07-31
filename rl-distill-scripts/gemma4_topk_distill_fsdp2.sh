#!/usr/bin/env bash
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Precomputed top-128 off-policy distillation for Gemma 4 students.
#
# Production inputs are accepted only through a validated dataset index:
#   MODEL_PATH                         - immutable local E2B/E4B student snapshot
#   DATASET_INDEX                      - validated vLLM index or finalized unsharded-HF overlay index
#   SOURCE_DATASET_INDEX               - exact vLLM source index (required only for an overlay)
#   DISTILL_DIRECTION                  - e4b_rl100_to_e2b or e2b_base_to_e4b
#   EXPECTED_TEACHER_IDENTITY_SHA256   - hash_json() of the pinned teacher identity
#   EXPECTED_STUDENT_IDENTITY_SHA256   - content-bound student identity from preflight
#   PREFLIGHT_RECEIPT_CACHE             - optional overlay receipt path; defaults beside its index
#   HF_PUSH_REPO                       - required only when HF_PUSH_ENABLE=true
#
# Direct TRAIN_FILE/VAL_FILE inputs bypass the dataset-wide provenance checks
# and are therefore available only for <=2-step smoke tests after explicitly
# setting SMOKE_ONLY_ALLOW_DIRECT_FILES=true.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

VENV=${VENV:-"${PROJECT_ROOT}/.venv-gemma4"}
if [ -f "${VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi
LOAD_DOTENV=${LOAD_DOTENV:-true}
if [ "${LOAD_DOTENV,,}" = "true" ] && [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

: "${MODEL_PATH:?Set MODEL_PATH to the Gemma 4 student initialization}"

PYTHON_BIN=${PYTHON_BIN:-python}
SMOKE_ONLY_ALLOW_DIRECT_FILES=${SMOKE_ONLY_ALLOW_DIRECT_FILES:-false}
ALLOW_QUESTION_OVERLAP=${ALLOW_QUESTION_OVERLAP:-false}
PREFLIGHT_LOCAL_FILES_ONLY=${PREFLIGHT_LOCAL_FILES_ONLY:-true}
PREFLIGHT_RECEIPT_CACHE=${PREFLIGHT_RECEIPT_CACHE:-}
REFRESH_PREFLIGHT_RECEIPT=${REFRESH_PREFLIGHT_RECEIPT:-false}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-750}
EXPECTED_TRAIN_QUESTIONS=${EXPECTED_TRAIN_QUESTIONS:-9723}
EXPECTED_VALIDATION_QUESTIONS=${EXPECTED_VALIDATION_QUESTIONS:-200}
EXPECTED_TRAIN_SAMPLES_PER_QUESTION=${EXPECTED_TRAIN_SAMPLES_PER_QUESTION:-5}
EXPECTED_VALIDATION_SAMPLES_PER_QUESTION=${EXPECTED_VALIDATION_SAMPLES_PER_QUESTION:-5}

if [ "$#" -ne 0 ]; then
    echo "Positional Hydra overrides are disabled; use the launcher's validated environment variables" >&2
    exit 2
fi

case "${SMOKE_ONLY_ALLOW_DIRECT_FILES,,}" in
    true|false) ;;
    *) echo "SMOKE_ONLY_ALLOW_DIRECT_FILES must be true or false" >&2; exit 2 ;;
esac
case "${ALLOW_QUESTION_OVERLAP,,}" in
    true|false) ;;
    *) echo "ALLOW_QUESTION_OVERLAP must be true or false" >&2; exit 2 ;;
esac
case "${PREFLIGHT_LOCAL_FILES_ONLY,,}" in
    true|false) ;;
    *) echo "PREFLIGHT_LOCAL_FILES_ONLY must be true or false" >&2; exit 2 ;;
esac
case "${REFRESH_PREFLIGHT_RECEIPT,,}" in
    true|false) ;;
    *) echo "REFRESH_PREFLIGHT_RECEIPT must be true or false" >&2; exit 2 ;;
esac
for count_name in EXPECTED_TRAIN_QUESTIONS EXPECTED_VALIDATION_QUESTIONS \
    EXPECTED_TRAIN_SAMPLES_PER_QUESTION EXPECTED_VALIDATION_SAMPLES_PER_QUESTION; do
    if ! [[ ${!count_name} =~ ^[1-9][0-9]*$ ]]; then
        echo "${count_name} must be a positive integer" >&2
        exit 2
    fi
done

TRAIN_FILES_HYDRA=""
VAL_FILES_HYDRA=""
PREFLIGHT_TOPK_WIDTH=""
PREFLIGHT_TOPK_TOLERANCE=""

if [ -n "${DATASET_INDEX:-}" ]; then
    if [ "${SMOKE_ONLY_ALLOW_DIRECT_FILES,,}" = "true" ]; then
        echo "DATASET_INDEX and SMOKE_ONLY_ALLOW_DIRECT_FILES are mutually exclusive" >&2
        exit 2
    fi
    if [ -n "${TRAIN_FILE:-}" ] || [ -n "${VAL_FILE:-}" ]; then
        echo "Do not set TRAIN_FILE or VAL_FILE when DATASET_INDEX is used" >&2
        exit 2
    fi
    if [ ! -d "${MODEL_PATH}" ]; then
        echo "Production distillation requires MODEL_PATH to be an immutable local HF snapshot directory" >&2
        exit 2
    fi
    : "${DISTILL_DIRECTION:?Set DISTILL_DIRECTION to the intended experimental direction}"
    : "${EXPECTED_TEACHER_IDENTITY_SHA256:?Set the pinned teacher identity SHA256}"
    : "${EXPECTED_STUDENT_IDENTITY_SHA256:?Set the pinned student identity SHA256}"

    if ! DATASET_SCHEMA_VERSION=$("${PYTHON_BIN}" -c '
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
schema = value.get("schema_version") if isinstance(value, dict) else None
if not isinstance(schema, str) or not schema:
    raise SystemExit("dataset index schema_version must be a non-empty string")
print(schema)
' "${DATASET_INDEX}"); then
        echo "Could not read DATASET_INDEX schema_version" >&2
        exit 2
    fi
    case "${DATASET_SCHEMA_VERSION}" in
        gemma4-distill-topk-v1)
            if [ -n "${SOURCE_DATASET_INDEX:-}" ]; then
                echo "SOURCE_DATASET_INDEX is valid only for an unsharded-HF overlay" >&2
                exit 2
            fi
            if [ -n "${PREFLIGHT_RECEIPT_CACHE}" ]; then
                echo "PREFLIGHT_RECEIPT_CACHE is currently supported only for an unsharded-HF overlay" >&2
                exit 2
            fi
            PREFLIGHT_SCRIPT="${PROJECT_ROOT}/rl-distill-scripts/data/preflight_gemma4_topk_distill.py"
            ;;
        gemma4-hf-bf16-sdpa-topk-overlay-v1)
            if [ -z "${SOURCE_DATASET_INDEX:-}" ]; then
                echo "Set SOURCE_DATASET_INDEX to the immutable vLLM source index for the overlay" >&2
                exit 2
            fi
            PREFLIGHT_SCRIPT="${PROJECT_ROOT}/rl-distill-scripts/data/preflight_gemma4_training_topk_overlay.py"
            if [ -z "${PREFLIGHT_RECEIPT_CACHE}" ]; then
                PREFLIGHT_RECEIPT_CACHE="$(dirname -- "${DATASET_INDEX}")/training_preflight_receipt.json"
            fi
            ;;
        *)
            echo "Unsupported DATASET_INDEX schema_version: ${DATASET_SCHEMA_VERSION}" >&2
            exit 2
            ;;
    esac

    PREFLIGHT_ARGS=(
        "${PREFLIGHT_SCRIPT}"
        --dataset-index "${DATASET_INDEX}"
        --student-model "${MODEL_PATH}"
        --expected-direction "${DISTILL_DIRECTION}"
        --expected-teacher-identity-sha256 "${EXPECTED_TEACHER_IDENTITY_SHA256}"
        --expected-student-identity-sha256 "${EXPECTED_STUDENT_IDENTITY_SHA256}"
        --expected-train-questions "${EXPECTED_TRAIN_QUESTIONS}"
        --expected-validation-questions "${EXPECTED_VALIDATION_QUESTIONS}"
        --expected-train-samples-per-question "${EXPECTED_TRAIN_SAMPLES_PER_QUESTION}"
        --expected-validation-samples-per-question "${EXPECTED_VALIDATION_SAMPLES_PER_QUESTION}"
    )
    if [ "${DATASET_SCHEMA_VERSION}" = "gemma4-hf-bf16-sdpa-topk-overlay-v1" ]; then
        PREFLIGHT_ARGS+=(
            --source-dataset-index "${SOURCE_DATASET_INDEX}"
            --receipt-cache "${PREFLIGHT_RECEIPT_CACHE}"
        )
        if [ "${REFRESH_PREFLIGHT_RECEIPT,,}" = "true" ]; then
            PREFLIGHT_ARGS+=(--refresh-receipt)
        fi
    fi
    if [ "${PREFLIGHT_LOCAL_FILES_ONLY,,}" = "true" ]; then
        PREFLIGHT_ARGS+=(--local-files-only)
    else
        PREFLIGHT_ARGS+=(--no-local-files-only)
    fi
    if [ "${ALLOW_QUESTION_OVERLAP,,}" = "true" ]; then
        echo "WARNING: explicitly allowing indexed train/validation question-text overlap" >&2
        PREFLIGHT_ARGS+=(--allow-question-overlap)
    fi

    PREFLIGHT_OUTPUT=$("${PYTHON_BIN}" "${PREFLIGHT_ARGS[@]}")
    declare -A PREFLIGHT_KEYS=()
    while IFS= read -r line; do
        [ -n "${line}" ] || continue
        key=${line%%=*}
        value=${line#*=}
        if [ "${key}" = "${line}" ] || [ -n "${PREFLIGHT_KEYS[${key}]:-}" ]; then
            echo "Malformed or duplicate preflight output key: ${key}" >&2
            exit 2
        fi
        PREFLIGHT_KEYS["${key}"]=1
        case "${key}" in
            TRAIN_FILES_HYDRA) TRAIN_FILES_HYDRA=${value} ;;
            VAL_FILES_HYDRA) VAL_FILES_HYDRA=${value} ;;
            TOPK_WIDTH) PREFLIGHT_TOPK_WIDTH=${value} ;;
            TOPK_VALIDATION_TOLERANCE) PREFLIGHT_TOPK_TOLERANCE=${value} ;;
            DATASET_INDEX_SHA256|GENERATION_EXPERIMENT_SHA256|STUDENT_TOKENIZER_SHA256) ;;
            DIRECTION)
                [ "${value}" = "${DISTILL_DIRECTION}" ] || {
                    echo "Preflight returned an unexpected direction: ${value}" >&2
                    exit 2
                }
                ;;
            TEACHER_IDENTITY_SHA256)
                [ "${value}" = "${EXPECTED_TEACHER_IDENTITY_SHA256}" ] || {
                    echo "Preflight returned an unexpected teacher identity" >&2
                    exit 2
                }
                ;;
            STUDENT_IDENTITY_SHA256)
                [ "${value}" = "${EXPECTED_STUDENT_IDENTITY_SHA256}" ] || {
                    echo "Preflight returned an unexpected student identity" >&2
                    exit 2
                }
                ;;
            *) echo "Unknown preflight output key: ${key}" >&2; exit 2 ;;
        esac
    done <<< "${PREFLIGHT_OUTPUT}"
    for required_key in \
        TRAIN_FILES_HYDRA VAL_FILES_HYDRA TOPK_WIDTH TOPK_VALIDATION_TOLERANCE \
        DATASET_INDEX_SHA256 GENERATION_EXPERIMENT_SHA256 DIRECTION \
        TEACHER_IDENTITY_SHA256 STUDENT_IDENTITY_SHA256 STUDENT_TOKENIZER_SHA256; do
        if [ -z "${PREFLIGHT_KEYS[${required_key}]:-}" ]; then
            echo "Preflight output is missing ${required_key}" >&2
            exit 2
        fi
    done
    if [ -z "${TRAIN_FILES_HYDRA}" ] || [ -z "${VAL_FILES_HYDRA}" ] || \
       [ -z "${PREFLIGHT_TOPK_WIDTH}" ] || [ -z "${PREFLIGHT_TOPK_TOLERANCE}" ]; then
        echo "Preflight returned an empty trainer input" >&2
        exit 2
    fi
    if [ -n "${TEACHER_TOP_K:-}" ] && [ "${TEACHER_TOP_K}" != "${PREFLIGHT_TOPK_WIDTH}" ]; then
        echo "TEACHER_TOP_K conflicts with the validated dataset index" >&2
        exit 2
    fi
    if [ -n "${TEACHER_TOPK_VALIDATION_TOLERANCE:-}" ] && \
       [ "${TEACHER_TOPK_VALIDATION_TOLERANCE}" != "${PREFLIGHT_TOPK_TOLERANCE}" ]; then
        echo "TEACHER_TOPK_VALIDATION_TOLERANCE conflicts with the validated dataset index" >&2
        exit 2
    fi
    export TEACHER_TOP_K=${PREFLIGHT_TOPK_WIDTH}
    export TEACHER_TOPK_VALIDATION_TOLERANCE=${PREFLIGHT_TOPK_TOLERANCE}
else
    if [ -n "${SOURCE_DATASET_INDEX:-}" ]; then
        echo "SOURCE_DATASET_INDEX requires DATASET_INDEX" >&2
        exit 2
    fi
    if [ "${SMOKE_ONLY_ALLOW_DIRECT_FILES,,}" != "true" ]; then
        echo "Set DATASET_INDEX for production, or explicitly enable the direct-file smoke-only path" >&2
        exit 2
    fi
    : "${TRAIN_FILE:?Set TRAIN_FILE for the smoke-only direct-file path}"
    : "${VAL_FILE:?Set VAL_FILE for the smoke-only direct-file path}"
    if ! [[ "${TOTAL_TRAINING_STEPS}" =~ ^[0-9]+$ ]] || [ "${TOTAL_TRAINING_STEPS}" -lt 1 ] || \
       [ "${TOTAL_TRAINING_STEPS}" -gt 2 ]; then
        echo "The direct-file smoke-only path requires TOTAL_TRAINING_STEPS between 1 and 2" >&2
        exit 2
    fi
    TRAIN_FILES_HYDRA=${TRAIN_FILE}
    VAL_FILES_HYDRA=${VAL_FILE}
fi

HF_PUSH_ENABLE=${HF_PUSH_ENABLE:-false}
if [ "${HF_PUSH_ENABLE,,}" = "true" ]; then
    : "${HF_PUSH_REPO:?Set HF_PUSH_REPO when HF_PUSH_ENABLE=true}"
fi
if [ "${SMOKE_ONLY_ALLOW_DIRECT_FILES,,}" = "true" ] && [ "${HF_PUSH_ENABLE,,}" = "true" ]; then
    echo "HF checkpoint upload is disabled for the direct-file smoke-only path" >&2
    exit 2
fi
export HF_PUSH_PRIVATE=${HF_PUSH_PRIVATE:-true}

export HF_HOME=${HF_HOME:-/tmp/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-"${HF_HOME}/hub"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/.cache}
export WANDB_DIR=${WANDB_DIR:-/tmp/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/tmp/wandb/cache}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-/tmp/wandb/config}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Gemma 4 has rank-dependent non-persistent buffers in some Transformers
# versions. Local per-rank loading avoids the unsafe rank-0 broadcast path in
# this fork; release/v0.8.0's sorted-buffer fix should remain the default after
# the planned forward-port.
export VERL_FSDP2_LOCAL_LOAD=${VERL_FSDP2_LOCAL_LOAD:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE:-0}
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

export TEACHER_TOP_K=${TEACHER_TOP_K:-128}
export TEACHER_TOPK_VALIDATION_TOLERANCE=${TEACHER_TOPK_VALIDATION_TOLERANCE:-0.0025}
export FULL_VOCAB_KL_CHUNK_SIZE=${FULL_VOCAB_KL_CHUNK_SIZE:-4096}
# The historical Gemma 3 top-k objective is a truncated contribution to the
# full-vocabulary forward KL and can legitimately be negative when the omitted
# tail would supply the compensating positive term. Clamping would silently
# zero its gradient, so preserve the established objective by default.
export CLAMP_MIN_TOPK_KL=${CLAMP_MIN_TOPK_KL:-false}
export CHECKPOINT_DISTILL_CHUNKS=${CHECKPOINT_DISTILL_CHUNKS:-true}
export MODEL_DTYPE=${MODEL_DTYPE:-fp32}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-2}
if ! [[ "${MICRO_BATCH_SIZE_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MICRO_BATCH_SIZE_PER_GPU must be a positive integer" >&2
    exit 2
fi
export MAX_LENGTH=${MAX_LENGTH:-12288}
export MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-${MAX_LENGTH}}
export TOTAL_TRAINING_STEPS
export LR=${LR:-5e-6}
export LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-100}
export LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
case "${LR_SCHEDULER_TYPE}" in
    constant|cosine|linear) ;;
    *) echo "LR_SCHEDULER_TYPE must be constant, cosine, or linear" >&2; exit 2 ;;
esac
export MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
export SAVE_FREQ=${SAVE_FREQ:-250}
export TEST_FREQ=${TEST_FREQ:-10}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
if ! [[ "${TOTAL_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TOTAL_EPOCHS must be a positive integer" >&2
    exit 2
fi
export REQUIRE_EXACT_VAL_COVERAGE=${REQUIRE_EXACT_VAL_COVERAGE:-true}
export PARQUET_ROW_GROUP_CACHE_SIZE=${PARQUET_ROW_GROUP_CACHE_SIZE:-2}
export PARQUET_ROW_GROUP_CACHE_MAX_BYTES=${PARQUET_ROW_GROUP_CACHE_MAX_BYTES:-1073741824}
export PARQUET_MAX_ROW_GROUP_BYTES=${PARQUET_MAX_ROW_GROUP_BYTES:-536870912}
export PARQUET_OVERSIZED_ROW_GROUP_POLICY=${PARQUET_OVERSIZED_ROW_GROUP_POLICY:-error}
export PROJECT_NAME=${PROJECT_NAME:-gemma4-distill-vs-rl}
export EXP_NAME=${EXP_NAME:-"gemma4-topk128-distill-$(date +%Y%m%d-%H%M%S)"}
export CKPTS_DIR=${CKPTS_DIR:-"/tmp/verl/ckpts/${PROJECT_NAME}/${EXP_NAME}"}
export TRAIN_LOGGER=${TRAIN_LOGGER:-'["console","wandb"]'}

export NNODES=${NNODES:-1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_PORT=${MASTER_PORT:-29571}
if [ "${NNODES}" = "1" ]; then
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
else
    : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
fi

COMMON_OVERRIDES=(
    teacher_model.path=null
    teacher_model.precomputed_topk=true
    teacher_model.top_k="${TEACHER_TOP_K}"
    teacher_model.chunk_size="${FULL_VOCAB_KL_CHUNK_SIZE}"
    teacher_model.temperature=1.0
    teacher_model.clamp_min_kl="${CLAMP_MIN_TOPK_KL}"
    teacher_model.checkpoint_student_chunks="${CHECKPOINT_DISTILL_CHUNKS}"
    data.train_files="${TRAIN_FILES_HYDRA}"
    data.val_files="${VAL_FILES_HYDRA}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}"
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}"
    data.use_dynamic_bsz=false
    data.max_length="${MAX_LENGTH}"
    data.truncation=error
    data.use_precomputed_topk=true
    data.teacher_topk_width="${TEACHER_TOP_K}"
    data.teacher_topk_validation_tolerance="${TEACHER_TOPK_VALIDATION_TOLERANCE}"
    data.require_exact_val_coverage="${REQUIRE_EXACT_VAL_COVERAGE}"
    model.path="${MODEL_PATH}"
    model.use_remove_padding=false
    model.enable_gradient_checkpointing=true
    model.override_config.attn_implementation=sdpa
    engine.fsdp_size=-1
    engine.model_dtype="${MODEL_DTYPE}"
    engine.use_torch_compile=false
    'engine.wrap_policy.transformer_layer_cls_to_wrap=["Gemma4TextDecoderLayer"]'
    optim.lr="${LR}"
    optim.lr_warmup_steps="${LR_WARMUP_STEPS}"
    optim.total_training_steps="${TOTAL_TRAINING_STEPS}"
    optim.lr_scheduler_type="${LR_SCHEDULER_TYPE}"
    optim.min_lr_ratio="${MIN_LR_RATIO}"
    optim.weight_decay=0.1
    'optim.betas=[0.9,0.98]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXP_NAME}"
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
    trainer.logger="${TRAIN_LOGGER}"
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.val_before_train="${VAL_BEFORE_TRAIN}"
    trainer.resume_mode=auto
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NPROC_PER_NODE}"
    trainer.hf_push.enable="${HF_PUSH_ENABLE}"
    trainer.hf_push.repo_id="${HF_PUSH_REPO:-unused}"
    trainer.hf_push.private="${HF_PUSH_PRIVATE}"
    trainer.hf_push.delete_local_after=false
    'checkpoint.save_contents=["model","optimizer","extra","hf_model"]'
    'checkpoint.load_contents=["model","optimizer","extra"]'
)

cd "${PROJECT_ROOT}/rl-distill-scripts"
if [ "${NNODES}" = "1" ]; then
    exec torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
        main_full_vocab_distill_fsdp2.py "${COMMON_OVERRIDES[@]}"
fi

exec torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    main_full_vocab_distill_fsdp2.py "${COMMON_OVERRIDES[@]}"
