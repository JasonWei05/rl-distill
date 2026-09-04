#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
fi

export VENV="${VENV:-/tmp/.venv-gemma4}"
export GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu129}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  VENV="${VENV}" GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT}" \
    bash rl-distill-scripts/setup_env_gemma4.sh
fi
export PATH="${VENV}/bin:${PATH}"
source "${VENV}/bin/activate"
PYTHON_BIN="${VENV}/bin/python"

if [[ ! -x "${VENV}/bin/aws" || ! -x "${VENV}/bin/lm_eval" ]]; then
  uv pip install --python "${PYTHON_BIN}" awscli -e ./lm-evaluation-harness
fi
command -v aws >/dev/null
command -v lm_eval >/dev/null
aws --version
"${PYTHON_BIN}" -c "import lm_eval, math_verify, pandas, transformers, vllm; print('GEMMA4_EVAL_ENV_OK', transformers.__version__, vllm.__version__)"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"

RESULT_S3_ROOT="${RESULT_S3_ROOT:-s3://scale-ml/genai/rl-distill/gemma4-distill-study-evals-v1}"
PACKED_WORK_ROOT="${PACKED_WORK_ROOT:-/tmp/gemma4-rl-distill-eval-packed}"
SHARED_DATA_ROOT="${PACKED_WORK_ROOT}/shared/data"
SHARED_MMMLU_ROOT="${PACKED_WORK_ROOT}/shared/mmmlu14k_tasks"
PACKED_MAX_ATTEMPTS="${PACKED_MAX_ATTEMPTS:-3}"
PACKED_START_STAGGER_SECONDS="${PACKED_START_STAGGER_SECONDS:-20}"
PACKED_RETRY_DELAY_SECONDS="${PACKED_RETRY_DELAY_SECONDS:-60}"

mkdir -p "${SHARED_DATA_ROOT}" "${SHARED_MMMLU_ROOT}"

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 8 ]]; then
  echo "FATAL: packed evaluation requires exactly 8 visible GPUs, found ${GPU_COUNT}" >&2
  exit 2
fi

echo "PACKED_PREPARE_SHARED_ASSETS start=$(date -u +%FT%TZ)"
"${PYTHON_BIN}" rl-distill-scripts/data/prepare_gemma4_rl_distill_eval_data.py \
  --output-dir "${SHARED_DATA_ROOT}" --overwrite

"${PYTHON_BIN}" rl-distill-scripts/data/prepare_gemma4_mmmlu14k.py \
  --output-dir "${SHARED_MMMLU_ROOT}" \
  --harness-dir /workspace/rl-distill/lm-evaluation-harness \
  --skip-harness-git-check --overwrite

test -s "${SHARED_DATA_ROOT}/math_eval_manifest.json"
test -s "${SHARED_MMMLU_ROOT}/manifest.json"
echo "PACKED_PREPARE_SHARED_ASSETS done=$(date -u +%FT%TZ)"

export SHARED_DATA_ROOT SHARED_MMMLU_ROOT RESULT_S3_ROOT
exec "${PYTHON_BIN}" rl-distill-scripts/scale_train/run_gemma4_rl_distill_eval_packed.py \
  --gpu-ids 0 1 2 3 4 5 6 7 \
  --work-root "${PACKED_WORK_ROOT}" \
  --result-s3-root "${RESULT_S3_ROOT}" \
  --max-attempts "${PACKED_MAX_ATTEMPTS}" \
  --start-stagger-seconds "${PACKED_START_STAGGER_SECONDS}" \
  --retry-delay-seconds "${PACKED_RETRY_DELAY_SECONDS}"
