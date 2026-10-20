#!/usr/bin/env bash
# Stop a running Qwen on-policy distillation run (launched with more steps) at STOP_STEP: wait until the pod log has the
# step-STOP_STEP validation and S3 has the complete permanent global_step_STOP_STEP checkpoint, then stop the supervisor +
# janitor, cancel the ScaleTrain job (by ID), and publish best_hf = the HF export of the best-validation step <= STOP_STEP
# (val mean@16; earliest on ties; step 0 = the init and is not eligible) plus a run_outcome.json, as the run-file would.
#   SPEC=q30bm-to-q1p7b STOP_STEP=120 bash stop_qwen3_opd_at_step.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
SPEC="${SPEC:?SPEC required}"; STOP_STEP="${STOP_STEP:-120}"; RUN_TAG="${RUN_TAG:-pgk1-v1}"
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/qwen3-opd}"
CKPT="${S3_BASE}-full-checkpoints/${SPEC}-${RUN_TAG}"; ART="${S3_BASE}/${SPEC}-${RUN_TAG}"
name=q3-opd-${SPEC}; sup=".scale_train_supervisors/${name}-${RUN_TAG}"
export AWS_PROFILE=ml-worker AWS_REGION=us-west-2
log() { echo "$(date -u +%FT%TZ) ${SPEC} $*" | tee -a "${sup}/stop-at-step.log"; }
log "waiting for step ${STOP_STEP} validation + checkpoint"
until cat "${sup}"/pod-logs/*.log 2>/dev/null | grep -a " step:${STOP_STEP} - " | grep -q "val-core/math/acc/mean@16" \
  && aws s3 ls "${CKPT}/global_step_${STOP_STEP}/_REMOTE_COMPLETE.json" >/dev/null 2>&1; do sleep 30; done
job="$(grep -oE 'job_d[a-z0-9]+' "${sup}/state.json" | head -1)"
touch "${sup}/STOP"
tmux kill-session -t "rl-${name}" 2>/dev/null || true
tmux kill-session -t "ckpt-janitor-${name}" 2>/dev/null || true
env -u AWS_PROFILE scale-train cancel job "${job}" >/dev/null 2>&1 || true
log "STOPPED at step ${STOP_STEP}: supervisor + janitor killed, job ${job} cancel requested ($(env -u AWS_PROFILE scale-train get job "${job}" 2>&1 | grep -o 'status: [A-Z_]*'))"
read -r best_step best_score < <(cat "${sup}"/pod-logs/*.log | grep -a " step:[0-9]* - " | python3 -c "
import re, sys
val = {}
for line in sys.stdin:
    m = re.search(r' step:(\d+) - ', line); v = re.search(r'val-core/math/acc/mean@16:(?:np\.float64\()?([0-9.e-]+)', line)
    if m and v and 0 < int(m.group(1)) <= ${STOP_STEP}: val.setdefault(int(m.group(1)), float(v.group(1)))
step = max(sorted(val), key=lambda s: val[s]); print(step, val[step])")
aws s3 sync --only-show-errors "${CKPT}/global_step_${best_step}/actor/huggingface/" "${ART}/best_hf/"
python3 - "${SPEC}" "${RUN_TAG}" "${STOP_STEP}" "${best_step}" "${best_score}" > /tmp/opd_outcome_${SPEC}.json <<'PY'
import json, sys
from datetime import datetime, timezone
spec, tag, stop, best, score = sys.argv[1:6]
print(json.dumps({"spec": spec, "run_tag": tag, "status": "complete", "stop_reason": f"manual_stop_at_{stop}", "final_step": int(stop),
                  "best_step": int(best), "best_score": float(score), "early_stopping_metric": "val-core/math/acc/mean@16",
                  "best_hf_source": f"global_step_{best}/actor/huggingface", "written_at": datetime.now(timezone.utc).isoformat()}, indent=2))
PY
aws s3 cp --only-show-errors "/tmp/opd_outcome_${SPEC}.json" "${ART}/run_outcome.json"
aws s3 cp --only-show-errors "/tmp/opd_outcome_${SPEC}.json" "${ART}/best_hf/_RUN_OUTCOME.json"
log "BEST_HF_PUBLISHED best_step=${best_step} val=${best_score} -> ${ART}/best_hf"
