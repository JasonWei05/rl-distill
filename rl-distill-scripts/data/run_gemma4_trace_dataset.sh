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

# Generate, validate, and upload one complete Gemma 4 top-128 trace dataset.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

PYTHON_BIN=${PYTHON_BIN:-/tmp/.venv-gemma4-e2e/bin/python}
TRAIN_PARQUET=${TRAIN_PARQUET:-/tmp/verl/data/deepscaler_4of4strict_rl_train.parquet}
VALIDATION_PARQUET=${VALIDATION_PARQUET:-/tmp/verl/data/deepscaler_4of4strict_rl_val200_x16.parquet}
GLOBAL_SEED=${GLOBAL_SEED:-42}
PROMPTS_PER_SHARD=${PROMPTS_PER_SHARD:-8}
ROW_GROUP_ROWS=${ROW_GROUP_ROWS:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_WORKER_ATTEMPTS=${MAX_WORKER_ATTEMPTS:-20}
UPLOAD_REVISION=${UPLOAD_REVISION:-main}

teacher_model=
teacher_content_sha256=
direction=
output_root=
hf_repo_id=
gpu_csv=
allow_question_overlap=false
skip_upload=false

usage() {
    cat <<'EOF'
Usage: run_gemma4_trace_dataset.sh \
  --teacher-model PATH \
  --teacher-content-sha256 SHA256 \
  --direction {e4b_rl100_to_e2b|e2b_base_to_e4b} \
  --output-root PATH \
  --gpus GPU[,GPU...] \
  [--hf-repo-id NAMESPACE/NAME] \
  [--allow-question-overlap] [--skip-upload]

The scientific generation contract is fixed to five samples/question,
temperature 1.0, top-p 1.0, sampling top-k disabled, an 8,192-token response
cap, and stored rank-1-through-128 full-vocabulary-normalized log probabilities.

The command is resumable: completed shards are verified and skipped. Each GPU
runs one independent worker. After both splits finish, the script regenerates a
complete dataset index and uploads the exact validated bundle to a private HF
dataset repository. Unless --skip-upload is used, --hf-repo-id and an HF_TOKEN
supplied through the environment are required.
EOF
}

while (($#)); do
    case "$1" in
        --teacher-model)
            teacher_model=${2:?missing value for --teacher-model}
            shift 2
            ;;
        --teacher-content-sha256)
            teacher_content_sha256=${2:?missing value for --teacher-content-sha256}
            shift 2
            ;;
        --direction)
            direction=${2:?missing value for --direction}
            shift 2
            ;;
        --output-root)
            output_root=${2:?missing value for --output-root}
            shift 2
            ;;
        --hf-repo-id)
            hf_repo_id=${2:?missing value for --hf-repo-id}
            shift 2
            ;;
        --gpus)
            gpu_csv=${2:?missing value for --gpus}
            shift 2
            ;;
        --allow-question-overlap)
            allow_question_overlap=true
            shift
            ;;
        --skip-upload)
            skip_upload=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for required_name in teacher_model teacher_content_sha256 direction output_root gpu_csv; do
    if [[ -z ${!required_name} ]]; then
        echo "Missing required argument: ${required_name}" >&2
        usage >&2
        exit 2
    fi
done
if [[ $skip_upload != true && -z $hf_repo_id ]]; then
    echo "Missing required argument for upload: hf_repo_id" >&2
    usage >&2
    exit 2
fi

if [[ $direction != e4b_rl100_to_e2b && $direction != e2b_base_to_e4b ]]; then
    echo "Unsupported direction: ${direction}" >&2
    exit 2
fi
if [[ ! $teacher_content_sha256 =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "--teacher-content-sha256 must be a 64-character hexadecimal digest" >&2
    exit 2
fi
if [[ -n $hf_repo_id && ($hf_repo_id != */* || $hf_repo_id == */*/*) ]]; then
    echo "--hf-repo-id must be an explicit namespace/name" >&2
    exit 2
fi
if [[ ! -d $teacher_model ]]; then
    echo "Local teacher model does not exist: ${teacher_model}" >&2
    exit 2
fi
if [[ ! -f $TRAIN_PARQUET || ! -f $VALIDATION_PARQUET ]]; then
    echo "Required DeepScaleR train/validation parquet files are missing" >&2
    exit 2
fi
if [[ ! -x $PYTHON_BIN ]]; then
    echo "Python executable is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ $skip_upload != true && -z ${HF_TOKEN:-} ]]; then
    echo "HF_TOKEN must be exported for the final private dataset upload" >&2
    exit 2
fi
if [[ -n $(git -C "$REPO_ROOT" status --porcelain) ]]; then
    echo "Refusing production generation from a dirty repository: ${REPO_ROOT}" >&2
    exit 2
fi
resolved_output_root=$(realpath -m -- "$output_root")
case "${resolved_output_root}/" in
    "${REPO_ROOT}/"*)
        echo "--output-root must be outside the source repository so generated shards cannot dirty provenance" >&2
        exit 2
        ;;
esac

IFS=',' read -r -a gpu_ids <<< "$gpu_csv"
if ((${#gpu_ids[@]} == 0)); then
    echo "At least one GPU is required" >&2
    exit 2
fi
declare -A seen_gpu_ids=()
for gpu_id in "${gpu_ids[@]}"; do
    if [[ ! $gpu_id =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU identifier: ${gpu_id}" >&2
        exit 2
    fi
    if [[ -n ${seen_gpu_ids[$gpu_id]:-} ]]; then
        echo "GPU identifiers must be unique: ${gpu_id}" >&2
        exit 2
    fi
    seen_gpu_ids[$gpu_id]=1
    if ! nvidia-smi --id="$gpu_id" --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
        echo "GPU identifier is not available on this host: ${gpu_id}" >&2
        exit 2
    fi
done

python_bin_dir=$(dirname -- "$PYTHON_BIN")
export PATH="${python_bin_dir}:${PATH}"
if ! command -v ninja >/dev/null 2>&1; then
    echo "ninja is required in PATH for FlashInfer kernel compilation" >&2
    exit 2
fi
if ! command -v setsid >/dev/null 2>&1; then
    echo "setsid is required for reliable worker process-group cleanup" >&2
    exit 2
fi

mkdir -p "$resolved_output_root/train" "$resolved_output_root/validation" "$resolved_output_root/logs"
output_root=$resolved_output_root
output_root=$(cd -- "$output_root" && pwd)
num_workers=${#gpu_ids[@]}
generator="${SCRIPT_DIR}/generate_gemma4_distill_traces.py"
validator="${SCRIPT_DIR}/validate_gemma4_distill_traces.py"
uploader="${SCRIPT_DIR}/upload_gemma4_distill_dataset.py"

child_pids=()
cleanup_children() {
    local pid
    trap - INT TERM
    for pid in "${child_pids[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${child_pids[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit 143
}
trap cleanup_children INT TERM

run_worker() {
    local split=$1
    local worker_id=$2
    local gpu_id=$3
    local input_parquet source_dataset log_path attempt worker_pid worker_status

    if [[ $split == train ]]; then
        input_parquet=$TRAIN_PARQUET
        source_dataset=deepscaler_4of4strict_rl_train
    else
        input_parquet=$VALIDATION_PARQUET
        source_dataset=deepscaler_4of4strict_rl_val200_x16
    fi
    log_path="${output_root}/logs/${split}-worker-${worker_id}.log"

    worker_pid=
    terminate_worker() {
        trap - INT TERM
        if [[ -n $worker_pid ]] && kill -0 "$worker_pid" 2>/dev/null; then
            kill -TERM -- "-${worker_pid}" 2>/dev/null || kill -TERM "$worker_pid" 2>/dev/null || true
            wait "$worker_pid" 2>/dev/null || true
        fi
        exit 143
    }
    trap terminate_worker INT TERM

    for ((attempt = 1; attempt <= MAX_WORKER_ATTEMPTS; attempt++)); do
        {
            echo "[supervisor] split=${split} worker=${worker_id}/${num_workers} gpu=${gpu_id} attempt=${attempt}"
            echo "[supervisor] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        } >> "$log_path"
        setsid env CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONPATH="$REPO_ROOT" PYTHONDONTWRITEBYTECODE=1 \
            "$PYTHON_BIN" "$generator" \
            --teacher-model "$teacher_model" \
            --teacher-content-sha256 "$teacher_content_sha256" \
            --input-parquet "$input_parquet" \
            --source-dataset "$source_dataset" \
            --output-dir "${output_root}/${split}" \
            --direction "$direction" \
            --split "$split" \
            --samples-per-question 5 \
            --global-seed "$GLOBAL_SEED" \
            --temperature 1.0 \
            --top-p 1.0 \
            --sampling-top-k -1 \
            --max-prompt-tokens 4096 \
            --max-response-tokens 8192 \
            --max-model-len 12288 \
            --prompts-per-shard "$PROMPTS_PER_SHARD" \
            --row-group-rows "$ROW_GROUP_ROWS" \
            --worker-id "$worker_id" \
            --num-workers "$num_workers" \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            >> "$log_path" 2>&1 &
        worker_pid=$!
        if wait "$worker_pid"; then
            worker_pid=
            echo "[supervisor] split=${split} worker=${worker_id} complete" | tee -a "$log_path"
            return 0
        else
            worker_status=$?
        fi
        worker_pid=
        echo "[supervisor] split=${split} worker=${worker_id} failed attempt=${attempt} status=${worker_status}" \
            | tee -a "$log_path" >&2
        if ((attempt < MAX_WORKER_ATTEMPTS)); then
            sleep 5
        fi
    done
    echo "[supervisor] split=${split} worker=${worker_id} exhausted retries" | tee -a "$log_path" >&2
    return 1
}

run_split() {
    local split=$1 worker_id status pid
    child_pids=()
    for worker_id in "${!gpu_ids[@]}"; do
        run_worker "$split" "$worker_id" "${gpu_ids[$worker_id]}" &
        child_pids+=("$!")
    done
    status=0
    for pid in "${child_pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done
    child_pids=()
    if ((status != 0)); then
        echo "At least one ${split} worker failed" >&2
        return 1
    fi
}

echo "[supervisor] generation_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
echo "[supervisor] direction=${direction} workers=${num_workers} output_root=${output_root}"
run_split train
run_split validation

validator_args=(
    "$validator"
    --split-dir "train=${output_root}/train"
    --split-dir "validation=${output_root}/validation"
    --output-index "${output_root}/dataset_index.json"
    --expected-train-questions 9723
    --expected-validation-questions 200
    --expected-samples-per-question 5
    --local-files-only
)
if [[ $allow_question_overlap != true ]]; then
    validator_args+=(--fail-on-question-overlap)
fi
PYTHONPATH=$REPO_ROOT PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" "${validator_args[@]}" \
    2>&1 | tee "${output_root}/logs/final-validation.log"

if [[ $skip_upload == true ]]; then
    echo "[supervisor] upload skipped; validated dataset is at ${output_root}"
    exit 0
fi

uploader_args=(
    "$uploader"
    --dataset-path "$output_root"
    --repo-id "$hf_repo_id"
    --revision "$UPLOAD_REVISION"
)
if [[ $allow_question_overlap == true ]]; then
    uploader_args+=(--allow-question-overlap)
fi
PYTHONPATH=$REPO_ROOT PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" "${uploader_args[@]}" \
    2>&1 | tee "${output_root}/logs/hf-upload-result.txt"

echo "[supervisor] complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
