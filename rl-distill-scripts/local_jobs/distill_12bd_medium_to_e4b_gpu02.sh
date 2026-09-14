#!/usr/bin/env bash
# §9.0g step 2: top-128 forward-KL distillation of the RL'd distilled 12B (§9.0 best, step 190; trace spec 12bd-medium)
# into the untrained E4B base with the §9 recipe, on local GPUs 0 and 2 only.
#
#   bs 128 / 1000 steps / lr 2e-6 peak, 100 warmup, linear -> 2e-7 / 1 seq per micro-batch under the 4096 padded-token
#   ceiling / fp32 master + Adam / val top-128 KL every 10 steps on 128 teacher validation generations.
#   E4B normally wants 4 GPUs (fp32 master + Adam ~56 GB/GPU on two); on 2 GPUs we offload params + Adam to the 2 TB host
#   (FSDP_OFFLOAD=true) and allow the undersized layout.
#   Checkpoints are S3-only (no Hub pushes): permanent full checkpoint every 250 steps, resumable rolling checkpoint every
#   50 steps (single slot), and every rolling save's weight-only HF export kept under <S3>/hf_exports/global_step_N/
#   (ROLLING_HF_EXPORT_S3, commit c3e82067). The trainer restores the newest complete S3 checkpoint at startup, so a
#   relaunch of this script resumes.
#
# Waits for the local trace bundle's COMPLETE.json (run_gemma4_bestckpt_trace_collection.sh, tmux trace-12bd-medium)
# before starting, so it can be chained right after the collection on the same GPUs.
set -euo pipefail
cd /mnt/efs/jasonwei/rl-distill
BUNDLE=/tmp/gemma4_bestckpt_traces_v2/12bd-medium
LOG_DIR=/tmp/gemma4_12bd_distill; mkdir -p "${LOG_DIR}"
echo "$(date -u +%FT%TZ) waiting for ${BUNDLE}/COMPLETE.json"
until [[ -f ${BUNDLE}/COMPLETE.json && -f ${BUNDLE}/dataset_index.json ]]; do sleep 60; done
echo "$(date -u +%FT%TZ) bundle complete; waiting for GPUs 0,2 to be released by the trace workers"
until [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0) -lt 2000 && $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2) -lt 2000 ]]; do sleep 30; done
echo "$(date -u +%FT%TZ) launching distillation"
export TEACHER_SPEC=12bd-medium STUDENT=e4b DISTILL_GPU_IDS=0,2
export ALLOW_UNDERSIZED_STUDENT_LAYOUT=true FSDP_OFFLOAD="${FSDP_OFFLOAD:-true}"
export TRAIN_SAMPLES_PER_QUESTION=16 TRAIN_BATCH_SIZE=128 TOTAL_TRAINING_STEPS=1000 TOTAL_EPOCHS=100
export LR=2e-6 LR_WARMUP_STEPS=100 LR_SCHEDULER_TYPE=linear MIN_LR_RATIO=0.1 TEST_FREQ=10
export FULL_VOCAB_KL_CHUNK_SIZE="${FULL_VOCAB_KL_CHUNK_SIZE:-4096}"
export SAVE_FREQ=250 ROLLING_CHECKPOINT_FREQ=50 REMOTE_CHECKPOINT_ENABLE=true ROLLING_HF_EXPORT=true ROLLING_HF_EXPORT_S3=true
export REMOTE_CHECKPOINT_S3_URI="${REMOTE_CHECKPOINT_S3_URI:-s3://scale-ml/genai/rl-distill/gemma4-12bd-distill-ckpts-v1/12bd-medium-to-e4b-bs128-s1000-lr2e-6}"
export CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra","hf_model"]' MAX_CKPT_TO_KEEP=1
export HF_PUSH_ENABLE=false HF_PUSH_DELETE_LOCAL=false
export EXP_NAME="${EXP_NAME:-12bd-medium-to-e4b-base-bs128-s1000-lr2e-6-g2-offload}"
export VENV=/tmp/.venv-gemma4 AWS_PROFILE=ml-worker
exec bash rl-distill-scripts/scale_train/run_gemma4_distill_one.sh
