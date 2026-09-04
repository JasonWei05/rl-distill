#!/usr/bin/env bash
# Collect off-policy top-128 teacher traces from a COMPLETED difficulty-band RL
# checkpoint, over the exact training/validation data that checkpoint trained on.
#
# Per checkpoint: 8 responses per training question + 1 response per validation
# question, at training sampling (temp 1.0 / top-p 1.0 / top-k -1), rendered with
# the RL few-shot template, capturing token ids + top-128 teacher logprobs.
#
# Teacher weights come from S3 (the full checkpoint's consolidated actor/huggingface
# HF model), not the Hub. Source data comes from the same 26B-band datasets the RL
# runs used. Reuses generate_gemma4_distill_traces.py and the RL data-prep unchanged.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

TRACE_SPEC="${TRACE_SPEC:?TRACE_SPEC is required}"
# run_key = full-checkpoint subdir; band = dataset band; best_step = checkpoint step.
TEACHER_HF_REPO=""  # set for teachers whose best export lives on the Hub instead of S3
case "${TRACE_SPEC}" in
  # Every teacher is fetched from its Hub export (step_NNNNNN/ subdir), so a node only needs HF
  # access. The e4b/12b/26b-easy exports are the S3 full checkpoints' actor/huggingface dirs
  # re-uploaded by scale_train/upload_fullckpt_to_hf.py; set TEACHER_SOURCE=s3 to pull those
  # from S3 instead. Best steps = W&B val mean@16 peaks.
  e4b-easy)   RUN_KEY=e4b-easy;       BAND=easy;   BEST_STEP=100; DIRECTION=e4b_easy_to_e2b;   TEACHER_HF_REPO=JWei05/DAPO-gemma4-e4b-PT-DeepScaleR-gemma26b-easy-seed42-26b-bands-es5 ;;
  e4b-medium) RUN_KEY=e4b-medium;     BAND=medium; BEST_STEP=90;  DIRECTION=e4b_medium_to_e2b; TEACHER_HF_REPO=JWei05/DAPO-gemma4-e4b-PT-DeepScaleR-gemma26b-medium-seed42-26b-bands-es5 ;;
  e4b-hard)   RUN_KEY=e4b-hard;       BAND=hard;   BEST_STEP=120; DIRECTION=e4b_hard_to_e2b;   TEACHER_HF_REPO=JWei05/DAPO-gemma4-e4b-PT-DeepScaleR-gemma26b-hard-seed42-26b-bands-es5 ;;
  12b-easy)   RUN_KEY=12b-easy;       BAND=easy;   BEST_STEP=70;  DIRECTION=12b_easy_to_e2b;   TEACHER_HF_REPO=JWei05/DAPO-gemma4-12b-PT-DeepScaleR-gemma26b-easy-seed42-26b-bands-es5 ;;
  12b-medium) RUN_KEY=12b-medium;     BAND=medium; BEST_STEP=120; DIRECTION=12b_medium_to_e2b; TEACHER_HF_REPO=JWei05/DAPO-gemma4-12b-PT-DeepScaleR-gemma26b-medium-seed42-26b-bands-es5 ;;
  12b-hard)   RUN_KEY=12b-hard;       BAND=hard;   BEST_STEP=140; DIRECTION=12b_hard_to_e2b;   TEACHER_HF_REPO=JWei05/DAPO-gemma4-12b-PT-DeepScaleR-gemma26b-hard-seed42-26b-bands-es5 ;;
  26b-easy)   RUN_KEY=26b-a4b-easy;   BAND=easy;   BEST_STEP=80;  DIRECTION=26b_easy_to_e2b;   TEACHER_HF_REPO=JWei05/DAPO-gemma4-26b-a4b-PT-DeepScaleR-gemma26b-easy-seed42-26b-bands-es5 ;;
  # Runs continued off-cluster / rerun locally (Hub exports only).
  e2b-easy)   RUN_KEY=e2b-easy;       BAND=easy;   BEST_STEP=130; DIRECTION=e2b_easy_to_e2b;   TEACHER_HF_REPO=JWei05/DAPO-gemma4-e2b-PT-DeepScaleR-gemma26b-easy-seed42-local2gpu ;;
  e2b-medium) RUN_KEY=e2b-medium;     BAND=medium; BEST_STEP=190; DIRECTION=e2b_medium_to_e2b; TEACHER_HF_REPO=JWei05/DAPO-gemma4-e2b-PT-DeepScaleR-gemma26b-medium-seed42-local2gpu ;;
  e2b-hard)   RUN_KEY=e2b-hard;       BAND=hard;   BEST_STEP=190; DIRECTION=e2b_hard_to_e2b;   TEACHER_HF_REPO=JWei05/DAPO-gemma4-e2b-PT-DeepScaleR-gemma26b-hard-seed42-local2gpu ;;
  26b-medium) RUN_KEY=26b-a4b-medium; BAND=medium; BEST_STEP=120; DIRECTION=26b_medium_to_e2b; TEACHER_HF_REPO=JWei05/DAPO-gemma4-26b-a4b-PT-DeepScaleR-gemma26b-medium-seed42-26b-bands-es5 ;;
  26b-hard)   RUN_KEY=26b-a4b-hard;   BAND=hard;   BEST_STEP=160; DIRECTION=26b_hard_to_e2b;   TEACHER_HF_REPO=JWei05/DAPO-gemma4-26b-a4b-PT-DeepScaleR-gemma26b-hard-seed42 ;;
  *)
    echo "FATAL: unsupported TRACE_SPEC=${TRACE_SPEC}" >&2
    exit 2
    ;;
esac

# TEACHER_SOURCE=s3 pulls the e4b/12b/26b-easy teachers from the S3 full checkpoints instead
# of their Hub re-uploads (same weights, same content hash; only for boxes with S3 access).
if [[ ${TEACHER_SOURCE:-hf} == s3 ]]; then
  case "${RUN_KEY}" in e4b-*|12b-*|26b-a4b-easy) TEACHER_HF_REPO="" ;; esac
fi
FULL_CHECKPOINT_S3_BASE="${FULL_CHECKPOINT_S3_BASE:-s3://scale-ml/genai/rl-distill/gemma4-difficulty-s42-20260819-full-checkpoints}"
TEACHER_S3_URI="${FULL_CHECKPOINT_S3_BASE}/${RUN_KEY}/global_step_${BEST_STEP}/actor/huggingface"
DIFFICULTY_DATASET_REPO="${DIFFICULTY_DATASET_REPO:-JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k}"
DIFFICULTY_DATASET_REVISION="${DIFFICULTY_DATASET_REVISION:-a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db}"

export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-west-2}"
export AWS_REGION="${AWS_REGION:-us-west-2}"
# Off ScaleTrain (i.e. on a dev box) the bare EC2 instance role (ec2-ml-worker)
# can read but NOT s3:PutObject to the trace bucket; the ml-worker profile assumes
# a role that can. Adopt it only when it exists and nothing is already set, so
# ScaleTrain pods (which have their own creds and no such profile) are unaffected.
if [[ -z "${AWS_PROFILE:-}" ]] && aws configure list-profiles 2>/dev/null | grep -qx ml-worker; then
  export AWS_PROFILE=ml-worker
fi

if [[ -f .env ]]; then set -a; source .env; set +a; fi

export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
export VENV="${VENV:-/tmp/.venv-gemma4}"
export GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu129}"
if [[ ! -x ${VENV}/bin/python ]]; then
  VENV="${VENV}" GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT}" bash rl-distill-scripts/setup_env_gemma4.sh
fi
source "${VENV}/bin/activate"
export PATH="${VENV}/bin:${PATH}"
# Keep vLLM/FlashInfer JIT + autotune caches on local disk: on a loaded shared box
# the default (~/.cache/vllm on EFS) stalls in NFS RPC waits during kernel warmup.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm-cache-${USER}}"
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDA_HOME="${CUDA_HOME:-$(readlink -f /usr/local/cuda)}"
elif [[ -x /usr/local/cuda-12.9/bin/nvcc ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
fi
if [[ -n ${CUDA_HOME:-} && -x ${CUDA_HOME}/bin/nvcc ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/rl-distill-scripts/data:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/tmp/hf_cache_${TRACE_SPEC}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache_${TRACE_SPEC}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${TRACE_SPEC}}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# --- Teacher: the completed checkpoint's consolidated HF model, from S3 ---
MODEL_DIR="${MODEL_DIR:-/tmp/gemma4_trace_models/${TRACE_SPEC}}"
mkdir -p "${MODEL_DIR}"
if [[ -n ${TEACHER_HF_REPO} ]]; then
  TEACHER_HF_SUBDIR="$(printf 'step_%06d' "${BEST_STEP}")"
  TEACHER_SOURCE_URI="hf://${TEACHER_HF_REPO}/${TEACHER_HF_SUBDIR}"
  TEACHER_SOURCE_SUBFOLDER="${TEACHER_HF_SUBDIR}"
  echo "TEACHER_DOWNLOAD spec=${TRACE_SPEC} hf=${TEACHER_SOURCE_URI}"
  # Pull only the best-step subdir, onto local disk (not the EFS HF cache), then flatten it
  # into MODEL_DIR so the rest of the pipeline is source-agnostic.
  "${VENV}/bin/python" - "${TEACHER_HF_REPO}" "${TEACHER_HF_SUBDIR}" "${MODEL_DIR}" <<'PY'
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo, subdir, model_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
staged = Path(
    snapshot_download(repo, allow_patterns=[f"{subdir}/*"], local_dir=str(model_dir.with_name(model_dir.name + "_hf")))
)
for path in (staged / subdir).iterdir():
    shutil.copy2(path, model_dir / path.name)
PY
else
  TEACHER_SOURCE_URI="${TEACHER_S3_URI}"
  TEACHER_SOURCE_SUBFOLDER="actor/huggingface"
  echo "TEACHER_DOWNLOAD spec=${TRACE_SPEC} s3=${TEACHER_S3_URI}"
  aws s3 cp --recursive --only-show-errors "${TEACHER_S3_URI}" "${MODEL_DIR}"
fi
if [[ ! -f ${MODEL_DIR}/config.json ]]; then
  echo "FATAL: teacher HF model missing config.json after S3 download (${MODEL_DIR})" >&2
  exit 2
fi

# Some actor exports (the 12b runs) omit processor_config.json. vLLM needs it to build the
# unified (text+vision+audio) Gemma 4 model, which is exactly how the RL rollout served these
# checkpoints. Never fall back to a text-only architecture override: it mis-maps the
# multimodal rows of the LM head, inflating their logits (stealing softmax mass from the real
# tokens, i.e. corrupting the top-k logprobs) and leaking <image|> into most responses. The
# base model's processor_config is byte-compatible (identical multimodal token ids). Added
# before the content hash, which covers weights only, so the teacher identity is unchanged.
if [[ ! -f ${MODEL_DIR}/processor_config.json ]]; then
  case "${RUN_KEY}" in
    e2b-*)     BASE_MODEL_REPO=google/gemma-4-E2B ;;
    e4b-*)     BASE_MODEL_REPO=google/gemma-4-E4B ;;
    12b-*)     BASE_MODEL_REPO=google/gemma-4-12B ;;
    26b-a4b-*) BASE_MODEL_REPO=google/gemma-4-26B-A4B ;;
    *) echo "FATAL: no base-model repo mapping for RUN_KEY=${RUN_KEY}" >&2; exit 2 ;;
  esac
  echo "PROCESSOR_CONFIG_PROVISION spec=${TRACE_SPEC} from=${BASE_MODEL_REPO}"
  "${VENV}/bin/python" - "${BASE_MODEL_REPO}" "${MODEL_DIR}" <<'PY'
import shutil
import sys

from huggingface_hub import hf_hub_download

repo, model_dir = sys.argv[1], sys.argv[2]
shutil.copy(hf_hub_download(repo, "processor_config.json"), f"{model_dir}/processor_config.json")
PY
fi
TEACHER_CONTENT_SHA256="$(${VENV}/bin/python - "${MODEL_DIR}" <<'PY'
import sys
from pathlib import Path

from gemma4_model_identity import inspect_local_hf_model

print(inspect_local_hf_model(Path(sys.argv[1])).weight_content_sha256)
PY
)"
if [[ ! ${TEACHER_CONTENT_SHA256} =~ ^[0-9a-f]{64}$ ]]; then
  echo "FATAL: invalid teacher content hash: ${TEACHER_CONTENT_SHA256}" >&2
  exit 2
fi

# --- Source: the band's training + validation data (same as RL) ---
GLOBAL_SEED="${GLOBAL_SEED:-42}"
TRAIN_SAMPLES_PER_QUESTION="${TRAIN_SAMPLES_PER_QUESTION:-8}"
VALIDATION_SAMPLES_PER_QUESTION="${VALIDATION_SAMPLES_PER_QUESTION:-1}"
PROMPTS_PER_SHARD="${PROMPTS_PER_SHARD:-8}"
ROW_GROUP_ROWS="${ROW_GROUP_ROWS:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
# Workers = GPUs / TP. 26B-A4B (~52 GB weights) defaults to TP2 so one engine spans its
# 2-GPU slice with ample KV cache; smaller teachers default to TP1 (DP across the slice).
case "${RUN_KEY}" in
  26b-a4b-*) TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}" ;;
  *)         TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}" ;;
esac
MAX_WORKER_ATTEMPTS="${MAX_WORKER_ATTEMPTS:-5}"
# Separate local root per trace version: the generator refuses to write into a directory
# holding a different generation configuration, so v2 must not share v1's local dirs
# (their stale v1 run_config.json would block every spec that ran under v1).
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/gemma4_bestckpt_traces_v2/${TRACE_SPEC}}"
# v2: v1 was generated through a text-only architecture override that corrupted the
# top-k logprobs and leaked <image|> into ~70% of responses; v1 must not be used.
TRACE_OUTPUT_S3_URI="${TRACE_OUTPUT_S3_URI:-s3://scale-ml/genai/rl-distill/gemma4-bestckpt-traces-topk128-v2/${TRACE_SPEC}}"
TRACE_S3_MIRROR_ENABLE="${TRACE_S3_MIRROR_ENABLE:-true}"
DATA_DIR="${DATA_DIR:-${OUTPUT_ROOT}/source}"
TRAIN_DIR="${OUTPUT_ROOT}/train"
VALIDATION_DIR="${OUTPUT_ROOT}/validation"
mkdir -p "${DATA_DIR}" "${TRAIN_DIR}" "${VALIDATION_DIR}" "${OUTPUT_ROOT}/logs"

# validation-repeats 1 => the 300 unique validation questions (1 response each).
"${VENV}/bin/python" rl-distill-scripts/data/prepare_deepscaler_gemma4_26b_difficulty_rl_data.py \
  --data-dir "${DATA_DIR}" --band "${BAND}" --validation-repeats 1
TRAIN_PARQUET="${DATA_DIR}/deepscaler_gemma4_26b_${BAND}_train.parquet"
VALIDATION_PARQUET="${DATA_DIR}/deepscaler_gemma4_26b_${BAND}_val300_x16.parquet"
if [[ ! -f ${TRAIN_PARQUET} || ! -f ${VALIDATION_PARQUET} ]]; then
  echo "FATAL: band source parquets missing (${TRAIN_PARQUET}, ${VALIDATION_PARQUET})" >&2
  exit 2
fi

count_unique_uids() {
  "${VENV}/bin/python" - "$1" <<'PY'
import sys
import pyarrow.parquet as pq
print(pq.read_table(sys.argv[1], columns=["uid"])["uid"].to_pandas().nunique())
PY
}
EXPECTED_TRAIN_QUESTIONS="$(count_unique_uids "${TRAIN_PARQUET}")"
EXPECTED_VALIDATION_QUESTIONS="$(count_unique_uids "${VALIDATION_PARQUET}")"
SOURCE_DATASET="${DIFFICULTY_DATASET_REPO}@${DIFFICULTY_DATASET_REVISION}:${BAND}-seed${GLOBAL_SEED}"

# TRACE_GPU_IDS pins this collection to explicit physical GPU indices (used by the
# async queue orchestrator, which does NOT set a parent CUDA_VISIBLE_DEVICES mask so
# each worker can re-select its absolute device). Falls back to CUDA_VISIBLE_DEVICES,
# then all visible GPUs.
if [[ -n ${TRACE_GPU_IDS:-} ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${TRACE_GPU_IDS}"
elif [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
fi
if ((${#GPU_IDS[@]} < 1)); then
  echo "FATAL: no visible GPU" >&2
  exit 2
fi
# TP consumes GPUs inside one engine; the rest are DP replicas (shard_id % NUM_WORKERS).
if (( ${#GPU_IDS[@]} % TENSOR_PARALLEL_SIZE != 0 )); then
  echo "FATAL: visible GPUs (${#GPU_IDS[@]}) not divisible by TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}" >&2
  exit 2
fi
NUM_WORKERS=$(( ${#GPU_IDS[@]} / TENSOR_PARALLEL_SIZE ))
echo "TRACE_COLLECTION spec=${TRACE_SPEC} teacher_step=${BEST_STEP} gpus=${#GPU_IDS[@]} tp=${TENSOR_PARALLEL_SIZE} workers=${NUM_WORKERS} s3=${TRACE_OUTPUT_S3_URI}"

child_pids=()
cleanup_children() {
  local pid
  trap - INT TERM
  for pid in "${child_pids[@]:-}"; do
    kill -0 "${pid}" 2>/dev/null && kill "${pid}" 2>/dev/null || true
  done
  for pid in "${child_pids[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
  exit 143
}
trap cleanup_children INT TERM

# One vLLM worker; DP shard = worker_id, pinned to its slice of GPUs.
run_worker() {
  local worker_id=$1 gpu_slice=$2 split=$3 samples=$4 input_parquet=$5 output_dir=$6 s3_subdir=$7
  local attempt status worker_pid log_path
  log_path="${OUTPUT_ROOT}/logs/${split}-worker-${worker_id}.log"
  worker_pid=

  terminate_worker() {
    trap - INT TERM
    if [[ -n ${worker_pid} ]] && kill -0 "${worker_pid}" 2>/dev/null; then
      kill -TERM -- "-${worker_pid}" 2>/dev/null || kill -TERM "${worker_pid}" 2>/dev/null || true
      wait "${worker_pid}" 2>/dev/null || true
    fi
    exit 143
  }
  trap terminate_worker INT TERM

  for ((attempt = 1; attempt <= MAX_WORKER_ATTEMPTS; attempt++)); do
    local generator_args=(
      rl-distill-scripts/data/generate_gemma4_distill_traces.py
      --teacher-model "${MODEL_DIR}"
      --teacher-content-sha256 "${TEACHER_CONTENT_SHA256}"
      --teacher-source-repo "${TEACHER_SOURCE_URI}"
      --teacher-source-revision "${TEACHER_CONTENT_SHA256}"
      --teacher-source-subfolder "${TEACHER_SOURCE_SUBFOLDER}"
      --input-parquet "${input_parquet}"
      --source-dataset "${SOURCE_DATASET}"
      --output-dir "${output_dir}"
      --direction "${DIRECTION}"
      --split "${split}"
      --chat-template "${PROJECT_ROOT}/rl-distill-scripts/data/gemma3_it_fewshot_math.jinja"
      --samples-per-question "${samples}"
      --global-seed "${GLOBAL_SEED}"
      --temperature 1.0
      --top-p 1.0
      --sampling-top-k -1
      --max-prompt-tokens 4096
      --max-response-tokens 8192
      --max-model-len 12288
      --prompts-per-shard "${PROMPTS_PER_SHARD}"
      --row-group-rows "${ROW_GROUP_ROWS}"
      --worker-id "${worker_id}"
      --num-workers "${NUM_WORKERS}"
      --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
      --max-num-seqs "${MAX_NUM_SEQS}"
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
      # TRACE_MAX_SHARDS>0 = smoke run through the exact production path (default: all).
      --max-shards "${TRACE_MAX_SHARDS:--1}"
    )
    if [[ ${TRACE_S3_MIRROR_ENABLE,,} == true ]]; then
      generator_args+=(--s3-mirror-uri "${TRACE_OUTPUT_S3_URI}/${s3_subdir}")
    fi
    echo "[supervisor] split=${split} worker=${worker_id}/${NUM_WORKERS} gpus=${gpu_slice} attempt=${attempt}" | tee -a "${log_path}"
    setsid env CUDA_VISIBLE_DEVICES="${gpu_slice}" PYTHONDONTWRITEBYTECODE=1 \
      "${VENV}/bin/python" "${generator_args[@]}" >>"${log_path}" 2>&1 &
    worker_pid=$!
    # Capture wait's status directly: `status=$?` after `if wait ...; fi` reads the if
    # statement's own status (always 0), which hid every real exit code and defeated the
    # no-retry check below.
    wait "${worker_pid}"; status=$?
    if ((status == 0)); then
      worker_pid=; echo "[supervisor] split=${split} worker=${worker_id} complete" | tee -a "${log_path}"; return 0
    fi
    worker_pid=
    echo "[supervisor] split=${split} worker=${worker_id} failed status=${status}" | tee -a "${log_path}" >&2
    if ((status == 3)); then
      echo "[supervisor] deterministic validation failure; not retrying" | tee -a "${log_path}" >&2
      return "${status}"
    fi
    ((attempt < MAX_WORKER_ATTEMPTS)) && sleep 10
  done
  return 1
}

# Launch NUM_WORKERS DP workers for one split, each owning TENSOR_PARALLEL_SIZE GPUs.
run_split() {
  local split=$1 samples=$2 input_parquet=$3 output_dir=$4 s3_subdir=$5
  local worker_id gpu_slice status=0
  child_pids=()
  for ((worker_id = 0; worker_id < NUM_WORKERS; worker_id++)); do
    gpu_slice="$(IFS=,; echo "${GPU_IDS[*]:worker_id*TENSOR_PARALLEL_SIZE:TENSOR_PARALLEL_SIZE}")"
    run_worker "${worker_id}" "${gpu_slice}" "${split}" "${samples}" "${input_parquet}" "${output_dir}" "${s3_subdir}" &
    child_pids+=("$!")
  done
  for pid in "${child_pids[@]}"; do wait "${pid}" || status=1; done
  child_pids=()
  if ((status != 0)); then echo "FATAL: ${split} trace worker failed" >&2; exit 1; fi
}

run_split train      "${TRAIN_SAMPLES_PER_QUESTION}"      "${TRAIN_PARQUET}"      "${TRAIN_DIR}"      train
run_split validation "${VALIDATION_SAMPLES_PER_QUESTION}" "${VALIDATION_PARQUET}" "${VALIDATION_DIR}" validation

"${VENV}/bin/python" rl-distill-scripts/data/validate_gemma4_distill_traces.py \
  --split-dir "train=${TRAIN_DIR}" \
  --split-dir "validation=${VALIDATION_DIR}" \
  --output-index "${OUTPUT_ROOT}/dataset_index.json" \
  --tokenizer-model "${MODEL_DIR}" \
  --local-files-only \
  --expected-train-questions "${EXPECTED_TRAIN_QUESTIONS}" \
  --expected-validation-questions "${EXPECTED_VALIDATION_QUESTIONS}" \
  --expected-train-samples-per-question "${TRAIN_SAMPLES_PER_QUESTION}" \
  --expected-validation-samples-per-question "${VALIDATION_SAMPLES_PER_QUESTION}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/final-validation.log"

"${VENV}/bin/python" - "${OUTPUT_ROOT}/dataset_index.json" "${TRACE_SPEC}" "${TRACE_OUTPUT_S3_URI}" <<'PY'
import json, sys
from datetime import UTC, datetime
from pathlib import Path

index_path = Path(sys.argv[1])
index = json.loads(index_path.read_text(encoding="utf-8"))
Path(index_path.with_name("COMPLETE.json")).write_text(
    json.dumps(
        {
            "schema_version": 1,
            "trace_spec": sys.argv[2],
            "s3_uri": sys.argv[3],
            "completed_at": datetime.now(UTC).isoformat(),
            "dataset_index_sha256": index["dataset_index_sha256"],
            "total_rows": index["total_rows"],
            "total_response_tokens": index["total_response_tokens"],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

if [[ ${TRACE_S3_MIRROR_ENABLE,,} == true ]]; then
  "${VENV}/bin/python" rl-distill-scripts/data/gemma4_trace_s3.py upload \
    --s3-uri "${TRACE_OUTPUT_S3_URI}" \
    --root "${OUTPUT_ROOT}" \
    "${TRAIN_PARQUET}" \
    "${VALIDATION_PARQUET}" \
    "${OUTPUT_ROOT}/dataset_index.json" \
    "${OUTPUT_ROOT}/logs/final-validation.log" \
    "${OUTPUT_ROOT}/COMPLETE.json"
else
  echo "TRACE_S3_MIRROR_DISABLED output=${OUTPUT_ROOT}"
fi

echo "TRACE_COLLECTION_COMPLETE spec=${TRACE_SPEC} s3=${TRACE_OUTPUT_S3_URI}"
