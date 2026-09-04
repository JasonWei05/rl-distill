#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
VENV="${VENV:-/tmp/.venv-gemma4}"
PYTHON_BIN="${VENV}/bin/python"
export PATH="${VENV}/bin:${PATH}"

if [[ "${EVAL_S3_ENABLE:-true}" == "true" ]]; then command -v aws >/dev/null; fi
test -x "${PYTHON_BIN}"

MODEL_TAG="${MODEL_TAG:?MODEL_TAG is required}"
GPU_COUNT="${GPU_COUNT:?GPU_COUNT is required}"
RESULT_S3_ROOT="${RESULT_S3_ROOT:-s3://scale-ml/genai/rl-distill/gemma4-distill-study-evals-v1}"
# EVAL_S3_ENABLE=false runs fully locally (no result sync) on nodes without S3 access.
EVAL_S3_ENABLE="${EVAL_S3_ENABLE:-true}"
SOURCE_REGISTRY="${SOURCE_REGISTRY:-rl-distill-scripts/config/gemma4_distill_study_eval_sources.json}"
# Answer verifier = exactly the RL reward: verl math_verify, strict last-\boxed{} grading, 30 s verify
# timeout, 5 s SymPy fallback (the RL launcher runs these defaults; pin them so a node env cannot drift).
export VERL_MATH_VERIFY_STRICT_BOXED="${VERL_MATH_VERIFY_STRICT_BOXED:-1}"
export VERL_MATH_VERIFY_TIMEOUT="${VERL_MATH_VERIFY_TIMEOUT:-30}"
export VERL_MATH_SYMPY_TIMEOUT="${VERL_MATH_SYMPY_TIMEOUT:-5.0}"
WORK_ROOT="${MODEL_WORK_ROOT:-/tmp/gemma4-rl-distill-eval/${MODEL_TAG}}"
DATA_ROOT="${SHARED_DATA_ROOT:-${WORK_ROOT}/data}"
MODEL_ROOT="${MODEL_ROOT_OVERRIDE:-${WORK_ROOT}/models}"
RESULT_ROOT="${WORK_ROOT}/results"
REMOTE_ROOT="${RESULT_S3_ROOT%/}/${MODEL_TAG}/"
MMMLU_ROOT="${SHARED_MMMLU_ROOT:-${DATA_ROOT}/mmmlu14k_tasks}"
PREPARE_SHARED_ASSETS="${PREPARE_SHARED_ASSETS:-true}"

case "${GPU_COUNT}" in
  1|2) ;;
  *) echo "FATAL: GPU_COUNT must be 1 or 2" >&2; exit 2 ;;
esac

if [[ -n "${PACKED_PHYSICAL_GPU_IDS:-}" ]]; then
  IFS=',' read -r -a GPU_ARGS <<< "${PACKED_PHYSICAL_GPU_IDS}"
  if [[ "${#GPU_ARGS[@]}" -ne "${GPU_COUNT}" ]]; then
    echo "FATAL: PACKED_PHYSICAL_GPU_IDS count does not match GPU_COUNT" >&2
    exit 2
  fi
  declare -A SEEN_GPU_IDS=()
  for gpu_id in "${GPU_ARGS[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-7]$ ]] || [[ -n "${SEEN_GPU_IDS[${gpu_id}]:-}" ]]; then
      echo "FATAL: PACKED_PHYSICAL_GPU_IDS must contain unique physical IDs in 0..7" >&2
      exit 2
    fi
    SEEN_GPU_IDS["${gpu_id}"]=1
  done
else
  case "${GPU_COUNT}" in
    1) GPU_ARGS=(0) ;;
    2) GPU_ARGS=(0 1) ;;
  esac
fi

mkdir -p "${DATA_ROOT}" "${MODEL_ROOT}" "${RESULT_ROOT}"
if [[ "${EVAL_S3_ENABLE}" == "true" ]]; then aws s3 sync "${REMOTE_ROOT}" "${RESULT_ROOT}" --only-show-errors || true; fi

upload_results() {
  if [[ "${EVAL_S3_ENABLE}" == "true" ]]; then aws s3 sync "${RESULT_ROOT}" "${REMOTE_ROOT}" --only-show-errors || true; fi
}
trap upload_results EXIT

if [[ "${PREPARE_SHARED_ASSETS}" == "true" ]]; then
  "${PYTHON_BIN}" rl-distill-scripts/data/prepare_gemma4_rl_distill_eval_data.py \
    --output-dir "${DATA_ROOT}" --overwrite
  "${PYTHON_BIN}" rl-distill-scripts/data/prepare_gemma4_mmmlu14k.py \
    --output-dir "${MMMLU_ROOT}" \
    --harness-dir "${PROJECT_ROOT}/lm-evaluation-harness" \
    --skip-harness-git-check --overwrite
elif [[ "${PREPARE_SHARED_ASSETS}" == "false" ]]; then
  test -s "${DATA_ROOT}/math_eval_manifest.json"
  test -s "${MMMLU_ROOT}/manifest.json"
else
  echo "FATAL: PREPARE_SHARED_ASSETS must be true or false" >&2
  exit 2
fi

"${PYTHON_BIN}" rl-distill-scripts/data/materialize_gemma4_eval_models.py \
  --source-registry "${SOURCE_REGISTRY}" \
  --output-root "${MODEL_ROOT}" \
  --models "${MODEL_TAG}" --execute

RESOLVED_REGISTRY="${MODEL_ROOT}/resolved_model_registry.json"

"${PYTHON_BIN}" rl-distill-scripts/run_gemma4_math_4gpu.py \
  --gpus "${GPU_ARGS[@]}" \
  --models "${MODEL_TAG}" \
  --resolved-model-registry "${RESOLVED_REGISTRY}" \
  --data-manifest "${DATA_ROOT}/math_eval_manifest.json" \
  --output-root "${RESULT_ROOT}" \
  --request-batch-size 8 --execute

upload_results

"${PYTHON_BIN}" rl-distill-scripts/run_gemma4_ood_4gpu.py \
  --gpus "${GPU_ARGS[@]}" \
  --models "${MODEL_TAG}" \
  --resolved-model-registry "${RESOLVED_REGISTRY}" \
  --output-root "${RESULT_ROOT}" \
  --lm-eval-executable "${VENV}/bin/lm_eval" \
  --mmmlu-task-dir "${MMMLU_ROOT}" \
  --mmmlu-manifest "${MMMLU_ROOT}/manifest.json" \
  --skip-harness-git-check --execute

"${PYTHON_BIN}" - "${MODEL_TAG}" "${RESULT_ROOT}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

tag, root = sys.argv[1], Path(sys.argv[2])
files = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name.endswith(".partial"):
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    files.append({"path": str(path.relative_to(root)), "size": path.stat().st_size, "sha256": digest})
(root / "RUN_COMPLETE.json").write_text(json.dumps({
    "schema_version": 1,
    "protocol": "gemma4_rl_distill_base_evals_v2",
    "model_tag": tag,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "files": files,
}, indent=2, sort_keys=True) + "\n")
PY

upload_results
echo "GEMMA4_EVAL_RUN_DONE model=${MODEL_TAG} remote=${REMOTE_ROOT}"
