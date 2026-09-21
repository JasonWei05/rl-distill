#!/usr/bin/env bash
# §9.0o checkpoint-eval controller: for each (run, step) below, wait for the distill checkpoint's completion marker on S3, then
# submit ONE 2-GPU ScaleTrain eval job (dedicated capacity, priority high; run-file run_gemma4_eval_one_model_st.sh = §7 math
# suite, data-parallel over the datasets on the 2 GPUs). Tracks submissions in $STATE so a restart never double-submits; when a
# job's results appear under RESULT_S3_ROOT/<tag>/RUN_COMPLETE.json it syncs them into the local study root and rebuilds §8.
# Runs until every tag is evaluated. ALLOW_BORROWING=true launches a duplicate set on borrowing (job suffix -brw, own state file); both
# controllers share the S3 result prefix per tag, so whichever finishes first supplies the row and the other stops submitting for that tag.
#   setsid bash eval_ckpt_controller.sh   (tmux eval-ckpt-controller)  |  ALLOW_BORROWING=true ... (tmux eval-ckpt-controller-brw)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export AWS_PROFILE=ml-worker
TAGS="${TAGS:-distill_26bd_lr2e6_medium_to_e4b_step500 distill_26bd_lr2e6_medium_to_e4b_step750 distill_26bd_lr2e6_medium_to_e4b_step1000 distill_26bw_lr2e6_medium_to_e4b_step500 distill_26bw_lr2e6_medium_to_e4b_step750 distill_26bw_lr2e6_medium_to_e4b_step1000_n8}"
RESULT_S3_ROOT="${RESULT_S3_ROOT:-s3://scale-ml/genai/rl-distill/gemma4-distill-study-evals-v1}"
ALLOW_BORROWING="${ALLOW_BORROWING:-false}"; BORROW_FLAG=""; SUFFIX=""; [ "$ALLOW_BORROWING" = true ] && { BORROW_FLAG="--allow-borrowing"; SUFFIX="-brw"; }
STUDY="${STUDY:-/tmp/gemma4_distill_study_eval}"; STATE="${STATE:-${STUDY}/ckpt_eval_controller${SUFFIX}.state}"; mkdir -p "${STUDY}/results" "$(dirname "$STATE")"; touch "$STATE"
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
PY=/tmp/.venv-gemma4/bin/python
REG=rl-distill-scripts/config/gemma4_distill_study_eval_sources.json
completion_uri() { "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(next(m["source"]["completion_uri"] for m in d["models"] if m["tag"]==sys.argv[2]))' "$REG" "$1"; }
ts() { date -u +%FT%TZ; }
submit() { local tag="$1" name; name="g4ev-$(echo "$tag" | sed -E 's/distill_//; s/_lr2e6_medium_to_e4b_/-/; s/_/-/g' | cut -c1-36)${SUFFIX}"
  echo "$(ts) SUBMIT $tag as $name"
  out="$(cd rl-distill-scripts/scale_train && env -u AWS_PROFILE bash launch_st_with_code.sh --gpus-per-instance 2 --priority high ${BORROW_FLAG} --active-deadline-hours 12 \
        --run-file run_gemma4_eval_one_model_st.sh --job-name "$name" --env-vars "MODEL_TAG=${tag},EVAL_PHASES=math" --image "$IMG" 2>&1)"
  jid="$(grep -oE 'job_[a-z0-9]{16,}' <<<"$out" | head -1)"
  if [ -n "$jid" ]; then echo "$tag SUBMITTED $jid $(ts)" >> "$STATE"; echo "$(ts) SUBMITTED $tag job=$jid"; else echo "$(ts) SUBMIT_FAILED $tag: $(tail -3 <<<"$out" | tr '\n' ' ' | cut -c1-300)"; fi; }
while :; do
  pending=0
  for tag in $TAGS; do
    grep -q "^$tag DONE" "$STATE" && continue
    pending=$((pending+1))
    if grep -q "^$tag SUBMITTED" "$STATE"; then
      # job submitted: check for results; if the job died without results, allow one resubmission
      if aws s3 ls "${RESULT_S3_ROOT}/${tag}/RUN_COMPLETE.json" >/dev/null 2>&1; then
        aws s3 sync --only-show-errors "${RESULT_S3_ROOT}/${tag}/" "${STUDY}/results/${tag}/" && echo "$tag DONE $(ts)" >> "$STATE"
        echo "$(ts) EVAL_DONE $tag -> rebuilding §8"; "$PY" rl-distill-scripts/update_distill_study_results_doc.py --fallback-from-doc >/dev/null 2>&1 || echo "$(ts) doc rebuild failed"
        grep -E "\`$tag\` \| distilled" rl-distill-scripts/DISTILLATION_EXPERIMENTS.md | cut -c1-200
      else
        jid="$(grep "^$tag SUBMITTED" "$STATE" | tail -1 | awk '{print $3}')"; st="$(env -u AWS_PROFILE timeout 120 scale-train get job "$jid" 2>/dev/null | grep -oE "status: [A-Z_]+" | head -1 | awk '{print $2}')"
        case "$st" in FAILED|CANCELED|CANCELLED|ERROR) n=$(grep -c "^$tag SUBMITTED" "$STATE"); if [ "$n" -lt 3 ]; then echo "$(ts) JOB_${st} $tag ($jid) -> resubmitting ($n/3)"; submit "$tag"; else echo "$(ts) GIVEUP $tag after $n attempts"; echo "$tag DONE giveup $(ts)" >> "$STATE"; fi ;; esac
      fi
    else
      if aws s3 ls "${RESULT_S3_ROOT}/${tag}/RUN_COMPLETE.json" >/dev/null 2>&1; then echo "$tag DONE (other controller) $(ts)" >> "$STATE"; echo "$(ts) EVAL_DONE $tag (results already on S3)"; continue; fi
      cu="$(completion_uri "$tag")"
      if aws s3 ls "$cu" >/dev/null 2>&1; then submit "$tag"; fi
    fi
  done
  [ "$pending" -eq 0 ] && { echo "$(ts) ALL_CKPT_EVALS_DONE"; exit 0; }
  sleep 300
done
