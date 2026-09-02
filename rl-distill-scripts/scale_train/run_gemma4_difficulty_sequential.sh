#!/usr/bin/env bash
# Run a configurable difficulty sequence back to back on one node.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

GEMMA4_MODEL="${GEMMA4_MODEL:?GEMMA4_MODEL must be set}"
: "${RUN_ARTIFACT_S3_BASE:?RUN_ARTIFACT_S3_BASE must be set}"
FULL_CHECKPOINT_S3_BASE="${FULL_CHECKPOINT_S3_BASE:-${RUN_ARTIFACT_S3_BASE%/}-full-checkpoints}"
case "${GEMMA4_MODEL}" in
  *E4B*)
    MODEL_TAG=gemma4-e4b
    ROUTER_REPLAY_MODE_DEFAULT=disabled
    ;;
  *12B*)
    MODEL_TAG=gemma4-12b
    ROUTER_REPLAY_MODE_DEFAULT=disabled
    ;;
  *26B-A4B*)
    MODEL_TAG=gemma4-26b-a4b
    ROUTER_REPLAY_MODE_DEFAULT=R3
    ;;
  *)
    echo "FATAL: difficulty wrapper supports only Gemma 4 E4B, 12B, or 26B-A4B; got ${GEMMA4_MODEL}" >&2
    exit 2
    ;;
esac

if [ -f .env ]; then set -a; source .env; set +a; fi
export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || {
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
}
export VENV="${VENV:-/tmp/.venv-gemma4}"
export GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu129}"
if [ ! -x "${VENV}/bin/python" ]; then
  VENV="${VENV}" GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT}" \
    bash rl-distill-scripts/setup_env_gemma4.sh
fi
source "${VENV}/bin/activate"

export DATA_DIR="${DATA_DIR:-/tmp/verl-shared/data}"
mkdir -p "${DATA_DIR}"
SEED="${DATA_SEED:-42}"
CHILD_LAUNCHER=rl-distill-scripts/scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh
LOG_DIR="${SEQUENTIAL_LOG_DIR:-/tmp/${MODEL_TAG}-difficulty-sequential-logs}"
mkdir -p "${LOG_DIR}"
read -r -a DIFFICULTY_RUNS <<<"${DIFFICULTY_SEQUENCE:-easy medium hard}"
if [ "${#DIFFICULTY_RUNS[@]}" -eq 0 ]; then
  echo "FATAL: DIFFICULTY_SEQUENCE selected no runs" >&2
  exit 2
fi
for difficulty in "${DIFFICULTY_RUNS[@]}"; do
  case "${difficulty}" in
    easy|medium|hard) ;;
    *)
      echo "FATAL: DIFFICULTY_SEQUENCE contains unsupported band ${difficulty}" >&2
      exit 2
      ;;
  esac
  "${VENV}/bin/python" rl-distill-scripts/data/prepare_deepscaler_gemma4_26b_difficulty_rl_data.py \
    --data-dir "${DATA_DIR}" --band "${difficulty}" --validation-repeats 16
done

run_child() {
  local difficulty="$1"
  local port_base="$2"
  local log_path="${LOG_DIR}/${difficulty}.log"
  local artifact_uri="${RUN_ARTIFACT_S3_BASE%/}/${MODEL_TAG}-${difficulty}"
  local checkpoint_key="${MODEL_TAG#gemma4-}-${difficulty}"
  local checkpoint_uri="${FULL_CHECKPOINT_S3_BASE%/}/${checkpoint_key}"
  local wandb_run_id="g4ds26b-${checkpoint_key}-s${SEED}-${WANDB_RUN_SUFFIX:-v1}"
  local hf_repo="JWei05/DAPO-${MODEL_TAG}-PT-DeepScaleR-gemma26b-${difficulty}-seed${SEED}-${RUN_NAME_SUFFIX:-26b-bands}"
  local legacy_early_stopping_state_b64=""
  if [ "${EARLY_STOPPING_LEGACY_RUN_KEY:-}" = "${checkpoint_key}" ]; then
    legacy_early_stopping_state_b64="${EARLY_STOPPING_LEGACY_STATE_B64:?legacy migration state is required for ${checkpoint_key}}"
  fi

  set +e
  "${VENV}/bin/python" rl-distill-scripts/full_checkpoint_s3.py check-completion-max \
    --s3-uri "${checkpoint_uri}" \
    --max-step "${TOTAL_TRAINING_STEPS}" \
    --expected-world-size "$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}")" \
    --model "${GEMMA4_MODEL}" \
    --difficulty "${difficulty}" \
    --seed "${SEED}" \
    --wandb-run-id "${wandb_run_id}" \
    --hf-repo "${hf_repo}" >/dev/null 2>&1
  local receipt_rc=$?
  "${VENV}/bin/python" rl-distill-scripts/full_checkpoint_s3.py check-best-hf \
    --s3-uri "${artifact_uri}" >/dev/null 2>&1
  local best_hf_rc=$?
  set -e
  if [ "${receipt_rc}" -eq 0 ] && [ "${best_hf_rc}" -eq 0 ]; then
    echo "SEQUENTIAL_SKIP_COMPLETE model=${GEMMA4_MODEL} difficulty=${difficulty} s3=${artifact_uri}"
    return 0
  fi
  if { [ "${receipt_rc}" -ne 0 ] && [ "${receipt_rc}" -ne 3 ]; } || \
     { [ "${best_hf_rc}" -ne 0 ] && [ "${best_hf_rc}" -ne 3 ]; }; then
    echo "SEQUENTIAL_COMPLETION_CHECK_FAILED model=${GEMMA4_MODEL} difficulty=${difficulty} receipt_rc=${receipt_rc} best_hf_rc=${best_hf_rc}" >&2
    return 1
  fi

  echo "SEQUENTIAL_LAUNCH model=${GEMMA4_MODEL} difficulty=${difficulty} port_base=${port_base} log=${log_path} s3=${artifact_uri}"
  set +e
  DIFFICULTY_DATASET_SOURCE=gemma4_26b_bands DIFFICULTY_DATASET="${difficulty}" \
  ROUTER_REPLAY_MODE="${ROUTER_REPLAY_MODE:-${ROUTER_REPLAY_MODE_DEFAULT}}" \
  ROUTER_Z_LOSS_COEF="${ROUTER_Z_LOSS_COEF:-0.0}" \
  DATA_SEED="${SEED}" RUN_SLOT="${MODEL_TAG}-${difficulty}" \
  RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-26b-bands}" \
  DATA_DIR="${DATA_DIR}" RAY_DATA_HOME="/tmp/verl-${MODEL_TAG}-${difficulty}" \
  RAY_TEMP_DIR="/tmp/ray_${MODEL_TAG}_${difficulty}" \
  WANDB_DIR="/tmp/wandb-${MODEL_TAG}-${difficulty}" \
  VLLM_CACHE_ROOT="/tmp/vllm-cache-${MODEL_TAG}-${difficulty}" \
  TRITON_CACHE_DIR="/tmp/triton-cache-${MODEL_TAG}-${difficulty}" \
  TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor-cache-${MODEL_TAG}-${difficulty}" \
  RUN_ARTIFACT_S3_URI="${artifact_uri}" \
  FULL_CHECKPOINT_S3_URI="${checkpoint_uri}" \
  EARLY_STOPPING_LEGACY_STATE_B64="${legacy_early_stopping_state_b64}" \
  WANDB_RUN_ID="${wandb_run_id}" WANDB_RESUME=allow \
  VERL_VLLM_PORT_BASE="${port_base}" \
    bash "${CHILD_LAUNCHER}" 2>&1 | tee "${log_path}"
  local child_rc="${PIPESTATUS[0]}"
  set -e

  if [ "${child_rc}" -ne 0 ] || ! grep -q 'RUN_DONE rc=0' "${log_path}"; then
    echo "SEQUENTIAL_CHILD_FAILED model=${GEMMA4_MODEL} difficulty=${difficulty} rc=${child_rc}" >&2
    return 1
  fi
  echo "SEQUENTIAL_CHILD_DONE model=${GEMMA4_MODEL} difficulty=${difficulty}"
}

cleanup_ray() {
  if [ -x "${VENV}/bin/ray" ]; then
    "${VENV}/bin/ray" stop --force >/dev/null 2>&1 || true
  fi
  sleep "${SEQUENTIAL_COOLDOWN_SECONDS:-30}"
}

for run_index in "${!DIFFICULTY_RUNS[@]}"; do
  difficulty="${DIFFICULTY_RUNS[run_index]}"
  case "${difficulty}" in
    easy) port_base=52000 ;;
    medium) port_base=54000 ;;
    hard) port_base=56000 ;;
  esac
  run_child "${difficulty}" "${port_base}"
  if [ "${run_index}" -lt "$(( ${#DIFFICULTY_RUNS[@]} - 1 ))" ]; then
    cleanup_ray
  fi
done
echo "SEQUENTIAL_GEMMA4_DIFFICULTY_RUNS_DONE model=${GEMMA4_MODEL}"
