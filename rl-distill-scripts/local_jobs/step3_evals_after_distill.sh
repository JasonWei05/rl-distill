#!/usr/bin/env bash
# §9.0g step 3 orchestrator: once the 12bd-medium -> E4B distillation has written its final step-1000 export, clear GPUs 0-3 and run
#   (a) the §7 math suite on the final export (GPUs 0,1; local_jobs/eval_12bd_step300_math.sh with MODEL_TAG=..._step1000) and
#   (b) pass@k x32 for every 50-step export + the 12bd teacher + the §4 control, then reverse/forward KL vs the teacher (GPUs 2,3;
#       local_jobs/eval_12bd_medium_to_e4b.sh)
# in parallel. Guards keep the box's free-GPU schedulers off GPUs 0-3 while the vLLM instances start.
set -euo pipefail
cd /mnt/efs/jasonwei/rl-distill
S3=s3://scale-ml/genai/rl-distill/gemma4-12bd-distill-ckpts-v1/12bd-medium-to-e4b-bs128-s1000-lr2e-6
LOG=/tmp/gemma4_12bd_distill/distill.log
log() { echo "$(date -u +%FT%TZ) $*"; }
log "waiting for the step-1000 export on S3"
until AWS_PROFILE=ml-worker aws s3 ls "$S3/hf_exports/global_step_1000/_REMOTE_COMPLETE.json" >/dev/null 2>&1 && grep -q "DISTILL_SHELL_EXIT" <(tail -n 5 "$LOG"); do sleep 60; done
log "distillation finished; clearing GPUs 0-3"
clear_gpus() { for g in 0 1 2 3; do u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$g"); for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$u" '$1==u{print $2}'); do [[ $(ps -o user= -p "$p" 2>/dev/null || true) == jasonwei ]] && continue; cid=$(grep -oE 'docker[-/][0-9a-f]{64}' /proc/"$p"/cgroup 2>/dev/null | head -1 | grep -oE '[0-9a-f]{64}' || true); [ -n "$cid" ] && { sudo -n docker update --restart=no "$cid" >/dev/null 2>&1; sudo -n docker stop -t 2 "$cid" >/dev/null 2>&1 && log "stopped container ${cid:0:12} on GPU $g"; }; sudo kill -9 "$p" 2>/dev/null && log "cleared pid $p from GPU $g"; done; done; }
clear_gpus; sleep 3
# guard for the eval start-ups (15 min): keep clearing intruders until our processes hold >= 30 GB on each GPU
( start=$(date +%s); declare -A ok; while [ $(( $(date +%s) - start )) -lt 900 ]; do for g in 0 1 2 3; do [ -n "${ok[$g]:-}" ] && continue; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$g"); ours=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | awk -F', ' -v u="$u" '$1==u{print $2, $3}' | while read -r p mm; do [[ $(ps -o user= -p "$p" 2>/dev/null || true) == jasonwei ]] && echo "${mm%% *}"; done | sort -n | tail -1); [ "${ours:-0}" -ge 30000 ] && ok[$g]=1; done; [ "${#ok[@]}" -ge 4 ] && { log "guard: all four eval instances hold their GPUs"; exit 0; }; clear_gpus >/dev/null; sleep 3; done; log "guard: timeout" ) &
mkdir -p /tmp/gemma4_12bd_evals /tmp/gemma4_distill_study_eval/queue_logs
log "launching (a) §7 math suite on the step-1000 export (GPUs 0,1)"
tmux new-session -d -s eval-12bd-step1000 "MODEL_TAG=distill_12bd_medium_to_e4b_step1000 GPUS=0,1 EVAL_PHASES=math EVAL_KV_CACHE_GIB=48 EVAL_GPU_MEMORY_UTILIZATION=0.85 MATH_REQUEST_BATCH_SIZE=2048 MATH_QUESTIONS_PER_BATCH=128 bash rl-distill-scripts/local_jobs/eval_12bd_step300_math.sh 2>&1 | tee -a /tmp/gemma4_distill_study_eval/queue_logs/eval_12bd_step1000_driver.log; echo EVAL_SHELL_EXIT=\${PIPESTATUS[0]} | tee -a /tmp/gemma4_distill_study_eval/queue_logs/eval_12bd_step1000_driver.log"
log "launching (b) pass@k x32 per export + teacher + control, then KL (GPUs 2,3)"
tmux new-session -d -s eval-12bd-passk "GPUS='2 3' bash rl-distill-scripts/local_jobs/eval_12bd_medium_to_e4b.sh 2>&1 | tee -a /tmp/gemma4_12bd_evals/passk_driver.log; echo PASSK_SHELL_EXIT=\${PIPESTATUS[0]} | tee -a /tmp/gemma4_12bd_evals/passk_driver.log"
log "STEP3_LAUNCHED"
