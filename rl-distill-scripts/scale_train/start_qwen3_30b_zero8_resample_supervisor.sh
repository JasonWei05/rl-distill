#!/usr/bin/env bash
# Resample the 9,182 DeepScaleR questions Qwen3-30B-A3B-Base solved 0/8 (n=8 run, 2026-10-05) 16 more times with the identical
# protocol (0-shot ChatML + boxed instruction, temp 1, top-p 1, 8192 tokens, same stops) but a NEW seed (1042), on 2 GPUs (DP 2 x TP 1;
# the 61 GB bf16 30B fits one H100 with ~10 GB KV -> 64 seqs/engine), borrowing on. Supervised: relaunch after preemption, every
# finished chunk is restored from S3. Done = ${OUT}/RUN_COMPLETE.json; grade with data/grade_band_eval_samples.py-style strict math_verify.
#   bash start_qwen3_30b_zero8_resample_supervisor.sh      # stop: touch <state>/STOP; tmux kill-session -t rl-q3-30b-zero8-n16
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
name=q3-30b-zero8-n16
OUT=s3://scale-ml/genai/rl-distill/qwen3-deepscaler-samples/qwen3-30b-a3b-base-0shot-qwenmath-zero8-n16-seed1042-8k
ENV_VARS="MODEL_REPO=Qwen/Qwen3-30B-A3B-Base,MODEL_REVISION=1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9,OUT_S3=${OUT},INPUT_S3=s3://scale-ml/genai/rl-distill/data/deepscaler_it_gen_dedup_v2_q30b_zero_of_8.parquet"
ENV_VARS+=",N=16,TP=1,SEED=1042,MAX_NUM_SEQS=64,GPU_MEMORY_UTILIZATION=0.92"
BORROW_FLAG="--allow-borrowing"; [ "${ALLOW_BORROWING:-true}" = false ] && BORROW_FLAG=""
sup=".scale_train_supervisors/${name}-$(date -u +%Y%m%d)${SUP_TAG:-}"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
tmux kill-session -t "=rl-${name}" 2>/dev/null || true
tmux new-session -d -s "rl-${name}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${name} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches ${MAX_RELAUNCHES:-200} --quick-cancel-seconds 300 --max-quick-cancels 3 --failure-backoff-seconds ${FAILURE_BACKOFF_SECONDS:-3600} --max-failure-backoff-seconds ${MAX_FAILURE_BACKOFF_SECONDS:-3600} \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
  --completion-marker-s3-uri ${OUT}/RUN_COMPLETE.json \
  -- bash rl-distill-scripts/scale_train/launch_st_with_code.sh --gpus-per-instance 2 --priority high ${BORROW_FLAG} --active-deadline-hours 48 \
  --run-file run_deepscaler_samples_st.sh --job-name ${name} --env-vars ${ENV_VARS} --image ${IMG} 2>&1 | tee -a ${sup}/supervisor.out"
echo "started supervisor rl-${name} (state ${sup}, out ${OUT})"
