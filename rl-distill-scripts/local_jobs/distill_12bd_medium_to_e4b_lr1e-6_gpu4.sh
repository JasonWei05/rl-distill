#!/usr/bin/env bash
# §9.0j: re-run of the §9.0g distillation (RL'd distilled 12B, step 190 -> E4B base; trace spec 12bd-medium, bundle on S3) with a
# LOWER learning rate: peak 1e-6 -> floor 1e-7 (linear, MIN_LR_RATIO=0.1), 100 warm-up. Motivation: in the 2e-6 run the val KL sat
# flat for a long stretch and only fell once the schedule had decayed. Everything else = §9 recipe (bs 128 / 1000 steps / 16
# samples, val every 10, S3-only checkpoints permanent every 250 + rolling every 50 with HF exports). 4 local GPUs. Resumes from S3.
#
#   GPUS=4,5,6,7 bash rl-distill-scripts/local_jobs/distill_12bd_medium_to_e4b_lr1e-6_gpu4.sh   # default: first 4 idle GPUs
set -euo pipefail
cd /mnt/efs/jasonwei/rl-distill
LOG_DIR=/tmp/gemma4_12bd_distill_lr1e-6; mkdir -p "${LOG_DIR}"
pick_free_gpus() { local want="$1" idx uuid used free=() busy; busy="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u || true)"
  while IFS=', ' read -r idx uuid used; do [ "${used}" -lt 512 ] && ! grep -qx "${uuid}" <<<"${busy}" && free+=("${idx}"); done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
  local IFS=,; echo "${free[*]:0:${want}}"; }
GPUS="${GPUS:-}"; while [ "$(awk -F, '{print NF}' <<<"${GPUS}")" -ne 4 ] || [ -z "${GPUS}" ]; do GPUS="$(pick_free_gpus 4)"; [ "$(awk -F, '{print NF}' <<<"${GPUS}")" -eq 4 ] && [ -n "${GPUS}" ] && break; echo "$(date -u +%FT%TZ) GPU_WAIT idle='${GPUS}'"; sleep 60; done
CK=/tmp/verl/ckpts/gemma4-12bd-distill-v1; for d in "$CK"/*lr1e-6*/global_step_*; do [ -d "$d" ] && { echo "purging stale local checkpoint $d"; rm -rf "$d"; }; done
echo "$(date -u +%FT%TZ) launching distillation on GPUs ${GPUS} (/tmp free: $(df -h /tmp | tail -1 | awk '{print $4}'))"
export TEACHER_SPEC=12bd-medium STUDENT=e4b DISTILL_GPU_IDS="${GPUS}"
export ALLOW_UNDERSIZED_STUDENT_LAYOUT=true FSDP_OFFLOAD="${FSDP_OFFLOAD:-false}" DISTILL_RESERVE_GPU_GB="${DISTILL_RESERVE_GPU_GB:-60}"
export TRAIN_SAMPLES_PER_QUESTION=16 TRAIN_BATCH_SIZE=128 TOTAL_TRAINING_STEPS=1000 TOTAL_EPOCHS=100
export LR="${LR:-1e-6}" LR_WARMUP_STEPS=100 LR_SCHEDULER_TYPE=linear MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}" TEST_FREQ=10   # floor = LR x MIN_LR_RATIO = 1e-7
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}" MAX_PADDED_TOKENS_PER_MICROBATCH="${MAX_PADDED_TOKENS_PER_MICROBATCH:-8192}"
export FULL_VOCAB_KL_CHUNK_SIZE="${FULL_VOCAB_KL_CHUNK_SIZE:-4096}" VERL_FAIL_ON_NONFINITE_GRAD="${VERL_FAIL_ON_NONFINITE_GRAD:-0}"
export SAVE_FREQ=250 ROLLING_CHECKPOINT_FREQ=50 REMOTE_CHECKPOINT_ENABLE=true ROLLING_HF_EXPORT=true ROLLING_HF_EXPORT_S3=true
export REMOTE_CHECKPOINT_S3_URI="${REMOTE_CHECKPOINT_S3_URI:-s3://scale-ml/genai/rl-distill/gemma4-12bd-distill-ckpts-v1/12bd-medium-to-e4b-bs128-s1000-lr1e-6}"
export CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra","hf_model"]' MAX_CKPT_TO_KEEP=1
export HF_PUSH_ENABLE=false HF_PUSH_DELETE_LOCAL=false
export EXP_NAME="${EXP_NAME:-12bd-medium-to-e4b-base-bs128-s1000-lr1e-6-g4}"
export VENV="${VENV:-/tmp/.venv-gemma4}" AWS_PROFILE="${AWS_PROFILE:-ml-worker}"
exec bash rl-distill-scripts/scale_train/run_gemma4_distill_one.sh
