#!/usr/bin/env bash
# §9.0j step 3: the §7 math suite (id_easy/id_medium/id_hard x16, MATH500 x16, GSM8K x8; RL verifier) for the final step_001000 exports of the
# §9.0j distillations, locally, through the study's single-model runner so the rows land in the §8 tables. Registry tags (s3_hf_export):
#   distill_12bd_lr1e6_medium_to_e4b_step1000   (b) 12bd -> E4B, peak 1e-6
#   distill_12br_medium_to_e4b_step1000         (a) 12br -> E4B, peak 1e-6
#   distill_12br_lr2e6_medium_to_e4b_step1000   (c) 12br -> E4B, peak 2e-6  -- evaluated once its S3 completion marker exists
# Waits for idle GPUs (the box is shared; a failed attempt — typically a co-tenant grabbing the GPUs during vLLM start-up — is retried), prepares the shared math data if the devbox restart wiped it, runs one model at a time on
# GPUS_PER_MODEL GPUs, mirrors results to the study's S3 root and rebuilds §8 after each model.
#
#   bash rl-distill-scripts/local_jobs/eval_9_0j_math.sh                       # all three (c waits for completion)
#   TAGS="distill_12br_medium_to_e4b_step1000" GPUS_PER_MODEL=2 bash rl-distill-scripts/local_jobs/eval_9_0j_math.sh
set -euo pipefail
cd /mnt/efs/jasonwei/rl-distill; set -a; source .env; set +a
export VENV=/tmp/.venv-gemma4 AWS_PROFILE=ml-worker PY=/tmp/.venv-gemma4/bin/python
TAGS="${TAGS:-distill_12bd_lr1e6_medium_to_e4b_step1000 distill_12br_medium_to_e4b_step1000 distill_12br_lr2e6_medium_to_e4b_step1000}"
GPUS_PER_MODEL="${GPUS_PER_MODEL:-2}"
STUDY=/tmp/gemma4_distill_study_eval; mkdir -p "$STUDY/queue_logs" "$STUDY/data"
export SHARED_DATA_ROOT="$STUDY/data" SHARED_MMMLU_ROOT="$STUDY/mmmlu14k_tasks"
if [[ ! -s $SHARED_DATA_ROOT/math_eval_manifest.json ]]; then
  echo "$(date -u +%FT%TZ) preparing shared math eval data -> $SHARED_DATA_ROOT"
  "$PY" rl-distill-scripts/data/prepare_gemma4_rl_distill_eval_data.py --output-dir "$SHARED_DATA_ROOT" --overwrite
fi
pick_free_gpus() { local want="$1" idx uuid used free=() busy; busy="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u || true)"
  while IFS=', ' read -r idx uuid used; do [ "${used}" -lt 2048 ] && ! grep -qx "${uuid}" <<<"${busy}" && free+=("${idx}"); done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
  local IFS=,; echo "${free[*]:0:${want}}"; }
completion_uri() { "$PY" - "$1" <<'PY'
import json, sys
d = json.load(open("rl-distill-scripts/config/gemma4_distill_study_eval_sources.json"))
print(next(m["source"]["completion_uri"] for m in d["models"] if m["tag"] == sys.argv[1]))
PY
}
wait_for_gpus() { local want="$1" tag="$2" gpus="" again=""
  while :; do gpus="$(pick_free_gpus "$want")"
    if [ -n "$gpus" ] && [ "$(awk -F, '{print NF}' <<<"$gpus")" -eq "$want" ]; then
      sleep 30; again="$(pick_free_gpus "$want")"   # co-tenants grab idle GPUs within minutes; require two consistent readings
      [ "$again" = "$gpus" ] && { echo "$gpus"; return; }
    fi
    echo "$(date -u +%FT%TZ) $tag: GPU_WAIT idle='$gpus'" >&2; sleep 120; done; }
for tag in $TAGS; do
  cu="$(completion_uri "$tag")"
  until aws s3 ls "$cu" >/dev/null 2>&1; do echo "$(date -u +%FT%TZ) $tag: waiting for $cu"; sleep 600; done
  attempt=0
  while :; do
    attempt=$((attempt+1)); gpus="$(wait_for_gpus "$GPUS_PER_MODEL" "$tag")"
    echo "$(date -u +%FT%TZ) EVAL $tag phases=math gpus=$gpus attempt=$attempt"
    if env -u CUDA_VISIBLE_DEVICES MODEL_TAG="$tag" EVAL_PHASES=math GPU_COUNT="$GPUS_PER_MODEL" PACKED_PHYSICAL_GPU_IDS="$gpus" PREPARE_SHARED_ASSETS=false \
        RESULT_ROOT_OVERRIDE="$STUDY/results/$tag" MODEL_WORK_ROOT="$STUDY/work/$tag" \
        EVAL_KV_CACHE_GIB=16 EVAL_GPU_MEMORY_UTILIZATION=0.40 EVAL_PREDICTIVE_TOPK_WIDTH=0 \
        MATH_QUESTIONS_PER_BATCH=64 MATH_REQUEST_BATCH_SIZE=1024 MATH_RESUME_TRACES=1 EVAL_S3_ENABLE=true \
        bash rl-distill-scripts/scale_train/run_gemma4_rl_distill_eval_one_model.sh 2>&1 | tee -a "$STUDY/queue_logs/$tag.log"; then break; fi
    echo "$(date -u +%FT%TZ) EVAL_FAILED $tag attempt=$attempt (usually a co-tenant took the GPUs during vLLM start-up) -> waiting for GPUs and retrying"
    [ "$attempt" -ge "${MAX_ATTEMPTS:-8}" ] && { echo "$(date -u +%FT%TZ) EVAL_GIVEUP $tag"; continue 2; }
    sleep 300
  done
  echo "$(date -u +%FT%TZ) rebuilding §8"; "$PY" rl-distill-scripts/update_distill_study_results_doc.py --fallback-from-doc || true
  echo "$(date -u +%FT%TZ) EVAL_DONE $tag"
done
echo "$(date -u +%FT%TZ) ALL_EVALS_DONE"
