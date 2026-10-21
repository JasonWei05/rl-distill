#!/usr/bin/env bash
# Start the Qwen math suite (all 8 sets, SINGLE_PASS, 1 GPU, borrowing off, priority high) on each on-policy-distilled student
# as soon as its run is done: run_outcome.json present and best_hf = the best-validation step (_RUN_OUTCOME.json, written by
# stop_qwen3_opd_at_step.sh / publish_qwen3_opd_best_hf.sh; a run-file final-step publish is re-published first). Models: `suite_models opd`
# (qwen3_math_suite_models.sh); results under s3://scale-ml/genai/rl-distill/qwen3-math-suite-v1/<spec>-opd120/<set>/.
#   tmux new -d -s q3-opd-math-watch "bash rl-distill-scripts/scale_train/run_qwen3_opd_math_suite_watcher.sh"   # stop: touch $STATE/STOP
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source rl-distill-scripts/scale_train/qwen3_math_suite_models.sh
STATE=".scale_train_supervisors/q3-opd-math-watch"; mkdir -p "${STATE}"
export AWS_PROFILE=ml-worker
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "${STATE}/watch.log"; }
entries="$(suite_models opd)"
while [ ! -e "${STATE}/STOP" ]; do
  pending=0
  for e in ${entries}; do
    name="${e%%=*}"; src="${e#*=}"; uri="${src#s3:}"; spec="${name%-opd120}"
    [ -e "${STATE}/started.${spec}" ] && continue
    pending=$((pending + 1))
    files="$(aws s3 ls "${uri}/" 2>/dev/null | awk '{print $4}' | tr '\n' ' ' || true)"   # not-yet-published prefix: aws exits 1
    [[ " ${files} " == *" config.json "* ]] || continue
    aws s3 ls "${uri%/best_hf}/run_outcome.json" >/dev/null 2>&1 || continue
    if [[ " ${files} " != *" _RUN_OUTCOME.json "* ]]; then
      # finished on its own: the run-file published the FINAL step (early stopping off) -> re-publish the best-val step first
      [[ " ${files} " == *" _REMOTE_COMPLETE.json "* ]] || continue
      SPEC="${spec}" bash rl-distill-scripts/scale_train/publish_qwen3_opd_best_hf.sh >> "${STATE}/watch.log" 2>&1 || { log "PUBLISH_FAILED ${spec}"; continue; }
    fi
    log "START math suite ${name} <- ${uri}"
    MODEL_NAME="${name}" MODEL_SRC="${src}" JOB_TAG="opd-${spec}" GPUS=1 EXTRA_ENV_VARS=SINGLE_PASS=1 ALLOW_BORROWING=false \
      bash rl-distill-scripts/scale_train/start_qwen3_math_suite_supervisor.sh >> "${STATE}/watch.log" 2>&1
    touch "${STATE}/started.${spec}"
  done
  (( pending > 0 )) || { log "WATCH_DONE all 9 suites started"; break; }
  sleep 300
done
