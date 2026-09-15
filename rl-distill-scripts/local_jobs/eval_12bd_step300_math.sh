#!/usr/bin/env bash
# §9.0g step 3a: the §7 math suite (id_easy/id_medium/id_hard x16, MATH500 x16, GSM8K x8; RL verifier) for the step-300 export of
# the 12bd-medium -> E4B-base distillation, through the study's single-model runner so the result lands in the §8 tables next to
# the §4 control (distill_12b_medium_to_e4b) and base_e4b. Registry entry: distill_12bd_medium_to_e4b_step300 (s3_hf_export source).
#
#   GPUS=0,1 bash rl-distill-scripts/local_jobs/eval_12bd_step300_math.sh        # after the distillation frees GPUs 0-3
#   EVAL_PHASES=ood GPUS=0 bash ... (MMLU-Pro / GPQA-Diamond / MMLU-14k via lm-eval; needs lm-evaluation-harness installed in the venv)
set -euo pipefail
cd /mnt/efs/jasonwei/rl-distill; set -a; source .env; set +a
GPUS="${GPUS:-0,1}"; IFS=',' read -r -a G <<< "$GPUS"
export VENV=/tmp/.venv-gemma4 AWS_PROFILE=ml-worker
export MODEL_TAG="${MODEL_TAG:-distill_12bd_medium_to_e4b_step300}" EVAL_PHASES="${EVAL_PHASES:-math}"
export GPU_COUNT="${#G[@]}" PACKED_PHYSICAL_GPU_IDS="$GPUS"
# same roots/knobs as the study queue (§7): shared prepared data, one results root for all models, no per-token logprobs,
# fixed 16 GiB KV per vLLM instance, cross-question batching, per-dataset resume, results mirrored to the study's S3 root.
export SHARED_DATA_ROOT=/tmp/gemma4_distill_study_eval/data PREPARE_SHARED_ASSETS=false
# queue layout: one results root PER MODEL (<base>/<tag>/<tag>/math/...); a shared root here would make the runner mirror every
# model's results to this tag's S3 prefix and hide the row from update_distill_study_results_doc.py (2026-09-15 lesson).
export RESULT_ROOT_OVERRIDE="/tmp/gemma4_distill_study_eval/results/${MODEL_TAG}" MODEL_WORK_ROOT="/tmp/gemma4_distill_study_eval/work/${MODEL_TAG}"
export SHARED_MMMLU_ROOT=/tmp/gemma4_distill_study_eval/mmmlu14k_tasks
export EVAL_KV_CACHE_GIB=16 EVAL_GPU_MEMORY_UTILIZATION=0.40 EVAL_PREDICTIVE_TOPK_WIDTH=0
export MATH_QUESTIONS_PER_BATCH=64 MATH_REQUEST_BATCH_SIZE=1024 MATH_RESUME_TRACES=1 EVAL_S3_ENABLE=true
mkdir -p /tmp/gemma4_distill_study_eval/queue_logs
echo "$(date -u +%FT%TZ) EVAL ${MODEL_TAG} phases=${EVAL_PHASES} gpus=${GPUS}"
bash rl-distill-scripts/scale_train/run_gemma4_rl_distill_eval_one_model.sh 2>&1 | tee -a "/tmp/gemma4_distill_study_eval/queue_logs/${MODEL_TAG}.log"
echo "$(date -u +%FT%TZ) rebuilding §8"
/tmp/.venv-gemma4/bin/python rl-distill-scripts/update_distill_study_results_doc.py --fallback-from-doc
echo "$(date -u +%FT%TZ) EVAL_DONE ${MODEL_TAG}"
