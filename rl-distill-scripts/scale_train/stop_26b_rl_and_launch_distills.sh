#!/usr/bin/env bash
# §9.0o (2026-09-21): at the 5-h mark stop both 26B-A4B medium RL runs, take each run's best-validation checkpoint as a teacher, and launch
# one 4-GPU ScaleTrain trace+distill job per teacher (3000x16 top-128 traces -> E4B base; bs 128, lr 2e-6 -> 2e-7, 1000 steps, 100 warmup),
# then queue the §7 math suite on the final exports. Idempotent per stage; DRY_RUN=1 prints the plan only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export AWS_PROFILE=ml-worker
declare -A SUP=( [26bd]=g4-26b-distill-rl-med-nb-20260921 [26bw]=g4-26b-med-ws140-nb-20260921-r2 )   # ws140 relaunched 08:3xZ after its weight-sync OOM
declare -A TMUX=( [26bd]=rl-g4-26b-distill-rl-med-nb [26bw]=rl-g4-26b-med-ws140-nb )
declare -A CKPT=( [26bd]=s3://scale-ml/genai/rl-distill/gemma4-26b-from-e4bbase-distill-rl-full-checkpoints/26b-a4b-medium-from-e4bbase-distill-es5
                  [26bw]=s3://scale-ml/genai/rl-distill/gemma4-difficulty-s42-20260819-full-checkpoints/26b-a4b-medium-ws140 )
declare -A BEST
for k in 26bd 26bw; do
  # newest COMPLETE permanent checkpoint carries the run's early-stopping state (best_step); best steps are multiples of 10 = permanent saves
  steps="$(aws s3 ls "${CKPT[$k]}/" | grep -oE "global_step_[0-9]+" | grep -oE "[0-9]+" | sort -n)"
  latest=""; for s in $(echo "$steps" | sort -rn); do aws s3 ls "${CKPT[$k]}/global_step_${s}/_REMOTE_COMPLETE.json" >/dev/null 2>&1 && { latest=$s; break; }; done
  [ -n "$latest" ] || { echo "FATAL: no complete permanent checkpoint under ${CKPT[$k]}" >&2; exit 2; }
  best="$(aws s3 cp "${CKPT[$k]}/global_step_${latest}/validation_early_stopping.json" - | python3 -c 'import json,sys; d=json.load(sys.stdin)["state"]; print(d["best_step"], d["best_score"])')"
  BEST[$k]="${best%% *}"; score="${best##* }"
  aws s3 ls "${CKPT[$k]}/global_step_${BEST[$k]}/actor/huggingface/config.json" >/dev/null 2>&1 || { echo "FATAL: ${k} best step ${BEST[$k]} has no actor/huggingface export" >&2; exit 2; }
  echo "TEACHER ${k}: latest_complete=${latest} best_step=${BEST[$k]} best_val=${score} hf=${CKPT[$k]}/global_step_${BEST[$k]}/actor/huggingface"
done
[ "${DRY_RUN:-0}" = 1 ] && { echo "DRY_RUN: would stop ${TMUX[*]} and launch 26bd-medium (BEST_STEP_OVERRIDE=${BEST[26bd]}) + 26bw-medium (BEST_STEP_OVERRIDE=${BEST[26bw]})"; exit 0; }
# --- stop the RL runs: STOP file, kill supervisor, cancel job, wait for pods to go
for k in 26bd 26bw; do
  d=".scale_train_supervisors/${SUP[$k]}"; touch "$d/STOP"; tmux kill-session -t "=${TMUX[$k]}" 2>/dev/null || true
  for j in $(cat .scale_train_supervisors/${SUP[$k]%-r2}*/supervisor.out 2>/dev/null | grep -aoE 'job_id=job_[a-z0-9]+' | cut -d= -f2 | sort -u); do
    st="$(env -u AWS_PROFILE timeout 120 scale-train get job "$j" 2>/dev/null | grep -oE "status: [A-Z_]+" | head -1)"
    case "$st" in *QUEUED*|*IN_PROGRESS*|*PENDING*|*RUNNING*) env -u AWS_PROFILE timeout 120 scale-train cancel job "$j" 2>&1 | grep -iE "cancel" | head -1 ;; *) echo "  $j already $st" ;; esac
  done
  echo "STOPPED ${k}: supervisor ${TMUX[$k]}"
done
# --- launch the two distillations (each its own 4-GPU job, dedicated capacity, priority high; recipe fixed in the ST run-file)
for k in 26bd 26bw; do
  TEACHER_SPEC="${k}-medium" LR=2e-6 ALLOW_BORROWING=false BEST_STEP_OVERRIDE="${BEST[$k]}" bash rl-distill-scripts/scale_train/start_gemma4_12br_trace_distill_supervisor.sh
  sleep 20
done
# --- math evals wait for each student's completion marker (registry tags distill_26b{d,w}_lr2e6_medium_to_e4b_step1000)
mkdir -p /tmp/gemma4_distill_study_eval/queue_logs
tmux new-session -d -s eval-26b-students "TAGS='distill_26bd_lr2e6_medium_to_e4b_step1000 distill_26bw_lr2e6_medium_to_e4b_step1000' GPUS_PER_MODEL=2 setsid bash rl-distill-scripts/local_jobs/eval_9_0j_math.sh 2>&1 | tee -a /tmp/gemma4_distill_study_eval/queue_logs/eval_26b_students_driver.log; sleep 999999"
echo "LAUNCHED distills 26bd(best ${BEST[26bd]}) 26bw(best ${BEST[26bw]}); eval driver tmux eval-26b-students"
