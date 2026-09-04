#!/usr/bin/env bash
# GPU-pool queue for the distillation-study evaluation suite: one model per GPU, the model runs its
# whole suite (math: 3 bands + MATH500 + GSM8K, then OOD: MMLU-Pro / GPQA-Diamond / MMLU-14k) via
# run_gemma4_rl_distill_eval_one_model.sh, and the next model starts the moment a GPU frees up.
#
#   EVAL_QUEUE_GPUS=4,5,6,7 bash rl-distill-scripts/scale_train/run_gemma4_distill_study_eval_queue.sh
#
# The roster is the study registry (config/gemma4_distill_study_eval_sources.json). Every cycle the
# registry is regenerated (data/build_gemma4_distill_study_eval_registry.py), so distilled students
# that finish on the nodes are appended and evaluated without restarting. Models whose
# RUN_COMPLETE.json already exists under the results root are skipped (resume). After each model
# completes, DISTILLATION_EXPERIMENTS.md §8 is regenerated from the result files and committed.
#
# Shared assets (math manifest + MMMLU task tree) are prepared once here, not per model.

set -uo pipefail

# Whole file is parsed before anything runs (bash reads scripts incrementally; a hours-long queue
# must not be affected by later edits to this file).
main() {

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
if [[ -f .env ]]; then set -a; source .env; set +a; fi

export VENV="${VENV:-/tmp/.venv-gemma4}"
PY="${VENV}/bin/python"
RUNNER="rl-distill-scripts/scale_train/run_gemma4_rl_distill_eval_one_model.sh"
REGISTRY="${SOURCE_REGISTRY:-rl-distill-scripts/config/gemma4_distill_study_eval_sources.json}"
STUDY_ROOT="${STUDY_ROOT:-/tmp/gemma4_distill_study_eval}"
export SHARED_DATA_ROOT="${SHARED_DATA_ROOT:-${STUDY_ROOT}/data}"
export SHARED_MMMLU_ROOT="${SHARED_MMMLU_ROOT:-${STUDY_ROOT}/mmmlu14k_tasks}"
export RESULT_ROOT_OVERRIDE="${RESULT_ROOT_OVERRIDE:-${STUDY_ROOT}/results}"
export SOURCE_REGISTRY="${REGISTRY}"
export EVAL_S3_ENABLE="${EVAL_S3_ENABLE:-true}"
LOG_ROOT="${STUDY_ROOT}/queue_logs"; mkdir -p "${LOG_ROOT}" "${RESULT_ROOT_OVERRIDE}"
POLL_SECONDS="${EVAL_QUEUE_POLL_SECONDS:-60}"
REFRESH_EVERY="${EVAL_QUEUE_REFRESH_EVERY:-10}"   # registry refresh every N polls
COMMIT_DOC="${EVAL_QUEUE_COMMIT_DOC:-true}"

if [[ -n ${EVAL_QUEUE_GPUS:-} ]]; then IFS=',' read -r -a FREE_GPUS <<< "${EVAL_QUEUE_GPUS}"; else mapfile -t FREE_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' '); fi

# --- shared assets, once -----------------------------------------------------------------------
if [[ ! -s ${SHARED_DATA_ROOT}/math_eval_manifest.json ]]; then
  echo "EVAL_QUEUE preparing math eval data -> ${SHARED_DATA_ROOT}"
  "${PY}" rl-distill-scripts/data/prepare_gemma4_rl_distill_eval_data.py --output-dir "${SHARED_DATA_ROOT}" --overwrite || { echo "FATAL: math data prep failed" >&2; exit 2; }
fi
if [[ ! -s ${SHARED_MMMLU_ROOT}/manifest.json ]]; then
  echo "EVAL_QUEUE preparing MMMLU-14k task tree -> ${SHARED_MMMLU_ROOT}"
  "${PY}" rl-distill-scripts/data/prepare_gemma4_mmmlu14k.py --output-dir "${SHARED_MMMLU_ROOT}" \
    --harness-dir "${PROJECT_ROOT}/lm-evaluation-harness" --skip-harness-git-check --overwrite || { echo "FATAL: MMMLU prep failed" >&2; exit 2; }
fi
test -x "${VENV}/bin/lm_eval" || { echo "FATAL: ${VENV}/bin/lm_eval missing (uv pip install -e ./lm-evaluation-harness)" >&2; exit 2; }

roster() {  # tags in registry order: bases, rl, distilled
  "${PY}" - "${REGISTRY}" <<'PY'
import json, sys
models = json.load(open(sys.argv[1]))["models"]
order = {"base": 0, "rl": 1, "distilled": 2}
for m in sorted(models, key=lambda m: (order.get(m["category"], 9), m.get("architecture", ""), m.get("trained_on") or "", m["tag"])):
    print(m["tag"])
PY
}
refresh_registry() {
  "${PY}" rl-distill-scripts/data/build_gemma4_distill_study_eval_registry.py --output "${REGISTRY}" >"${LOG_ROOT}/registry_refresh.log" 2>&1 || echo "EVAL_QUEUE warning: registry refresh failed (keeping previous roster)" >&2
}
is_complete() { [[ -s ${RESULT_ROOT_OVERRIDE}/$1/RUN_COMPLETE.json ]]; }

update_doc() {
  "${PY}" rl-distill-scripts/update_distill_study_results_doc.py --results-root "${RESULT_ROOT_OVERRIDE}" --registry "${REGISTRY}" \
    && if [[ ${COMMIT_DOC} == true ]]; then
         git add rl-distill-scripts/DISTILLATION_EXPERIMENTS.md "${REGISTRY}" \
           && git commit --no-verify -q -m "Distill study evals: results update ($1 complete)" \
           && { git fetch -q origin main; git merge-base --is-ancestor origin/main HEAD && git push -q origin HEAD:main || echo "EVAL_QUEUE note: doc committed locally, push deferred (origin moved)"; }
       fi
}

declare -A PID_TAG=() PID_GPU=() TAG_STATUS=() LAUNCHED=()
launch() {
  local tag="$1" gpu="${FREE_GPUS[0]}"; FREE_GPUS=("${FREE_GPUS[@]:1}")
  echo "EVAL_QUEUE launch model=${tag} gpu=${gpu} $(date -u +%FT%TZ)"
  env -u CUDA_VISIBLE_DEVICES MODEL_TAG="${tag}" GPU_COUNT=1 PACKED_PHYSICAL_GPU_IDS="${gpu}" PREPARE_SHARED_ASSETS=false \
    MODEL_WORK_ROOT="${STUDY_ROOT}/work/${tag}" bash "${RUNNER}" >"${LOG_ROOT}/${tag}.log" 2>&1 &
  PID_TAG[$!]="${tag}"; PID_GPU[$!]="${gpu}"; LAUNCHED[$tag]=1
}
reap() {
  local pid tag status progressed=1
  for pid in "${!PID_TAG[@]}"; do
    kill -0 "${pid}" 2>/dev/null && continue
    wait "${pid}"; status=$?; tag="${PID_TAG[$pid]}"
    FREE_GPUS+=("${PID_GPU[$pid]}"); unset 'PID_TAG[$pid]' 'PID_GPU[$pid]'; TAG_STATUS[$tag]=${status}
    if ((status == 0)) && is_complete "${tag}"; then echo "EVAL_QUEUE done model=${tag} $(date -u +%FT%TZ)"; update_doc "${tag}"
    else echo "EVAL_QUEUE FAILED model=${tag} status=${status} (log ${LOG_ROOT}/${tag}.log) $(date -u +%FT%TZ)" >&2; fi
    progressed=0
  done
  return "${progressed}"
}

echo "EVAL_QUEUE start gpus=[${FREE_GPUS[*]}] registry=${REGISTRY} results=${RESULT_ROOT_OVERRIDE}"
poll=0
while true; do
  if ((poll % REFRESH_EVERY == 0)); then refresh_registry; fi
  mapfile -t TAGS < <(roster)
  pending=0
  for tag in "${TAGS[@]}"; do
    [[ -n ${LAUNCHED[$tag]:-} ]] && continue
    if is_complete "${tag}"; then LAUNCHED[$tag]=1; TAG_STATUS[$tag]=0; echo "EVAL_QUEUE skip model=${tag} (already complete)"; continue; fi
    if ((${#FREE_GPUS[@]} > 0)); then launch "${tag}"; else pending=$((pending + 1)); fi
  done
  # Finished when nothing runs and nothing is pending; distilled models still training on the
  # nodes will appear in later registry refreshes, so keep polling while EVAL_QUEUE_WAIT_FOR_NEW=true.
  if ((${#PID_TAG[@]} == 0 && pending == 0)) && [[ ${EVAL_QUEUE_WAIT_FOR_NEW:-true} != true ]]; then break; fi
  if ((${#PID_TAG[@]} == 0 && pending == 0)); then echo "EVAL_QUEUE idle: roster complete; waiting for new distilled models $(date -u +%FT%TZ)"; fi
  reap || sleep "${POLL_SECONDS}"
  poll=$((poll + 1))
done

echo "EVAL_QUEUE summary:"; for tag in "${!TAG_STATUS[@]}"; do echo "  ${tag}: exit=${TAG_STATUS[$tag]}"; done | sort
echo "EVAL_QUEUE_COMPLETE"
}

main "$@"
