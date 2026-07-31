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

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DATASET_INDEX=${SOURCE_DATASET_INDEX:-/lambda/nfs/Jason-scale/rl-distill-traces/gemma4-e4b-rl100-topk128/dataset_index.json}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-/tmp/verl/models/nemorl-gemma4-e4b-step100-vllm}
OVERLAY_ROOT=${OVERLAY_ROOT:-/lambda/nfs/Jason-scale/rl-distill-traces/gemma4-e4b-rl100-hf-bf16-sdpa-topk128-overlay}
PYTHON_BIN=${PYTHON_BIN:-/tmp/.venv-gemma4-e2e/bin/python}
RESCORER_SCRIPT=${RESCORER_SCRIPT:-${PROJECT_ROOT}/rl-distill-scripts/data/rescore_gemma4_training_topk.py}
GPU_IDS=(0 1 2 3)
LM_HEAD_CHUNK_TOKENS=8192
EXPECTED_SOURCE_DATASET_INDEX_SHA256=8b5712e0f5dea3388340a9bc91a6ceee40ff2ff990e66b7238090e240daeda6c

usage() {
    cat <<'EOF'
Usage: prepare_gemma4_e4b_rl_to_e2b_topk_overlay_4gpu.sh MODE

MODE is one of:
  status    Show whether each required overlay artifact exists (default).
  inspect   Validate source/model identities and write rescore_config.json.
  parity    Run the mandatory native-forward parity gate on GPU 0.
  score     Resume bulk scoring with one worker on each of GPUs 0,1,2,3.
  finalize  Validate all shards and write the overlay dataset_index.json.
  all       Run inspect, parity, score, and finalize in order.

parity, score, and all require:
  GEMMA4_E4B_RL_TO_E2B_RESCORE_AUTHORIZED=YES
EOF
}

if [ "$#" -gt 1 ]; then
    usage >&2
    exit 2
fi
MODE=${1:-status}
case "${MODE}" in
    status|inspect|parity|score|finalize|all) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

if [ "${MODE}" != "status" ]; then
    if [ ! -f "${SOURCE_DATASET_INDEX}" ]; then
        echo "Source dataset index does not exist: ${SOURCE_DATASET_INDEX}" >&2
        exit 2
    fi
    if [ ! -d "${TEACHER_MODEL_PATH}" ]; then
        echo "Teacher model directory does not exist: ${TEACHER_MODEL_PATH}" >&2
        exit 2
    fi
    if [ ! -x "${PYTHON_BIN}" ]; then
        echo "Python executable does not exist: ${PYTHON_BIN}" >&2
        exit 2
    fi
    if [ ! -f "${RESCORER_SCRIPT}" ]; then
        echo "Rescorer does not exist: ${RESCORER_SCRIPT}" >&2
        exit 2
    fi
    if ! SOURCE_DATASET_INDEX_SHA256=$(python3 -c '
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(value.get("dataset_index_sha256", ""))
' "${SOURCE_DATASET_INDEX}"); then
        echo "Could not read source dataset identity: ${SOURCE_DATASET_INDEX}" >&2
        exit 2
    fi
    if [ "${SOURCE_DATASET_INDEX_SHA256}" != "${EXPECTED_SOURCE_DATASET_INDEX_SHA256}" ]; then
        echo "Source dataset identity is not the pinned E4B-RL bundle: ${SOURCE_DATASET_INDEX_SHA256}" >&2
        exit 2
    fi
fi

export HF_HOME=${HF_HOME:-/tmp/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/.cache}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

COMMON_ARGS=(
    --repo-root "${PROJECT_ROOT}"
    --source-dataset-index "${SOURCE_DATASET_INDEX}"
    --model-path "${TEACHER_MODEL_PATH}"
    --output-root "${OVERLAY_ROOT}"
    --lm-head-chunk-tokens "${LM_HEAD_CHUNK_TOKENS}"
)

require_gpu_authorization() {
    if [ "${GEMMA4_E4B_RL_TO_E2B_RESCORE_AUTHORIZED:-NO}" != "YES" ]; then
        echo "Set GEMMA4_E4B_RL_TO_E2B_RESCORE_AUTHORIZED=YES before ${MODE}" >&2
        exit 2
    fi
}

run_inspect() {
    "${PYTHON_BIN}" "${RESCORER_SCRIPT}" inspect "${COMMON_ARGS[@]}"
}

run_parity() {
    require_gpu_authorization
    CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${RESCORER_SCRIPT}" parity \
        "${COMMON_ARGS[@]}" --parity-rows 8 --parity-max-response-tokens 512
}

run_score() {
    require_gpu_authorization
    mkdir -p "${OVERLAY_ROOT}/logs"
    local pids=()
    local worker_id gpu log_path status=0
    for worker_id in "${!GPU_IDS[@]}"; do
        gpu=${GPU_IDS[${worker_id}]}
        log_path="${OVERLAY_ROOT}/logs/score-worker-${worker_id}.log"
        echo "Starting overlay worker ${worker_id} on GPU ${gpu}; log=${log_path}"
        (
            CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${RESCORER_SCRIPT}" score \
                "${COMMON_ARGS[@]}" --worker-id "${worker_id}" --num-workers "${#GPU_IDS[@]}"
        ) >"${log_path}" 2>&1 &
        pids+=("$!")
    done
    trap 'for pid in "${pids[@]}"; do kill "${pid}" 2>/dev/null || true; done; exit 130' INT TERM
    for worker_id in "${!pids[@]}"; do
        if ! wait "${pids[${worker_id}]}"; then
            echo "Overlay worker ${worker_id} failed; see ${OVERLAY_ROOT}/logs/score-worker-${worker_id}.log" >&2
            status=1
        fi
    done
    trap - INT TERM
    return "${status}"
}

run_finalize() {
    "${PYTHON_BIN}" "${RESCORER_SCRIPT}" finalize "${COMMON_ARGS[@]}"
}

show_status() {
    local path
    printf 'source_index=%s\nteacher_model=%s\noverlay_root=%s\n' \
        "${SOURCE_DATASET_INDEX}" "${TEACHER_MODEL_PATH}" "${OVERLAY_ROOT}"
    for path in rescore_config.json parity_receipt.json dataset_index.json; do
        if [ -f "${OVERLAY_ROOT}/${path}" ]; then
            printf '%s=present\n' "${path}"
        else
            printf '%s=missing\n' "${path}"
        fi
    done
}

case "${MODE}" in
    status) show_status ;;
    inspect) run_inspect ;;
    parity) run_parity ;;
    score) run_score ;;
    finalize) run_finalize ;;
    all)
        require_gpu_authorization
        run_inspect
        run_parity
        run_score
        run_finalize
        ;;
esac
