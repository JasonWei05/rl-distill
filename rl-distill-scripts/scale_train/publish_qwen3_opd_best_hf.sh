#!/usr/bin/env bash
# Re-publish best_hf of a finished Qwen on-policy distillation run as the best-VALIDATION step <= MAX_STEP (val mean@16 from
# the pod logs; earliest on ties; step 0 = the init is not eligible), the selection stop_qwen3_opd_at_step.sh uses. Needed for
# runs that finished on their own: with early stopping off the run-file publishes the final step as best_hf (best_score null).
# The run-file's receipt is kept as run_outcome.runfile.json; run_outcome.json and best_hf/_RUN_OUTCOME.json are rewritten.
#   SPEC=q30bm-to-q4b bash publish_qwen3_opd_best_hf.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
SPEC="${SPEC:?SPEC required}"; MAX_STEP="${MAX_STEP:-120}"; RUN_TAG="${RUN_TAG:-pgk1-v1}"
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/qwen3-opd}"
CKPT="${S3_BASE}-full-checkpoints/${SPEC}-${RUN_TAG}"; ART="${S3_BASE}/${SPEC}-${RUN_TAG}"
sup=".scale_train_supervisors/q3-opd-${SPEC}-${RUN_TAG}"
export AWS_PROFILE=ml-worker AWS_REGION=us-west-2
read -r best_step best_score n_val < <(cat "${sup}"/pod-logs/*.log | grep -a " step:[0-9]* - " | python3 -c "
import re, sys
val = {}
for line in sys.stdin:
    m = re.search(r' step:(\d+) - ', line); v = re.search(r'val-core/math/acc/mean@16:(?:np\.float64\()?([0-9.e-]+)', line)
    if m and v and 0 < int(m.group(1)) <= ${MAX_STEP}: val.setdefault(int(m.group(1)), float(v.group(1)))
step = max(sorted(val), key=lambda s: val[s]); print(step, val[step], len(val))")
(( n_val == MAX_STEP / 10 )) || { echo "FATAL: ${SPEC}: found ${n_val} validations in (0, ${MAX_STEP}], expected $((MAX_STEP / 10))" >&2; exit 2; }
aws s3 ls "${CKPT}/global_step_${best_step}/_REMOTE_COMPLETE.json" >/dev/null || { echo "FATAL: no complete global_step_${best_step}" >&2; exit 2; }
if ! aws s3 ls "${ART}/run_outcome.runfile.json" >/dev/null 2>&1; then
  aws s3 cp --only-show-errors "${ART}/run_outcome.json" "${ART}/run_outcome.runfile.json"
fi
# cp, not sync: sync skips same-size files whose destination is newer, i.e. it would keep the run-file's final-step weights
aws s3 rm --only-show-errors --recursive "${ART}/best_hf/" --exclude "_REMOTE_COMPLETE.json"
aws s3 cp --only-show-errors --recursive "${CKPT}/global_step_${best_step}/actor/huggingface/" "${ART}/best_hf/"
python3 - "${SPEC}" "${RUN_TAG}" "${MAX_STEP}" "${best_step}" "${best_score}" > "/tmp/opd_outcome_${SPEC}.json" <<'PY'
import json, sys
from datetime import datetime, timezone
spec, tag, final, best, score = sys.argv[1:6]
print(json.dumps({"spec": spec, "run_tag": tag, "status": "complete", "stop_reason": "max_steps", "final_step": int(final),
                  "best_step": int(best), "best_score": float(score), "early_stopping_metric": "val-core/math/acc/mean@16",
                  "best_hf_source": f"global_step_{best}/actor/huggingface", "best_selection": "best val step <= final (republished)",
                  "written_at": datetime.now(timezone.utc).isoformat()}, indent=2))
PY
aws s3 cp --only-show-errors "/tmp/opd_outcome_${SPEC}.json" "${ART}/run_outcome.json"
aws s3 cp --only-show-errors "/tmp/opd_outcome_${SPEC}.json" "${ART}/best_hf/_RUN_OUTCOME.json"
echo "$(date -u +%FT%TZ) ${SPEC} BEST_HF_REPUBLISHED best_step=${best_step} val=${best_score} -> ${ART}/best_hf" | tee -a "${sup}/stop-at-step.log"
