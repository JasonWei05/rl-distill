#!/usr/bin/env bash
# §9.0j step 2: top-128 forward-KL distillation of the RESUMED untrained-12B medium RL teacher (step 140, trace spec 12br-medium)
# into the untrained E4B base, §9 recipe (bs 128 / 1000 steps / lr 2e-6 peak -> 2e-7 linear, 100 warm-up, 16 samples/prompt,
# val top-128 KL every 10 steps), 4 local GPUs (fp32 master + Adam sharded 4-way, no offload). S3-only checkpoints: permanent
# every 250, rolling every 50 with the weight-only HF export kept under <S3>/hf_exports/. A relaunch resumes from S3.
# Waits for the local trace bundle (local_jobs/trace_12br_medium_dp2.sh) to complete, so it can be chained.
#
#   GPUS=0,1,2,3 bash rl-distill-scripts/local_jobs/distill_12br_medium_to_e4b_gpu4.sh   # default: first 4 idle GPUs once the bundle is complete
set -euo pipefail
cd /mnt/efs/jasonwei/rl-distill
BUNDLE=/tmp/gemma4_bestckpt_traces_v2/12br-medium
LOG_DIR=/tmp/gemma4_12br_distill; mkdir -p "${LOG_DIR}"
echo "$(date -u +%FT%TZ) waiting for ${BUNDLE}/COMPLETE.json"
until [[ -f ${BUNDLE}/COMPLETE.json && -f ${BUNDLE}/dataset_index.json ]]; do sleep 60; done
echo "$(date -u +%FT%TZ) bundle complete; waiting for trace engines to exit"
until [[ $(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -c EngineCore) -eq 0 ]]; do sleep 20; done
pick_free_gpus() { local want="$1" idx uuid used free=() busy; busy="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u || true)"
  while IFS=', ' read -r idx uuid used; do [ "${used}" -lt 512 ] && ! grep -qx "${uuid}" <<<"${busy}" && free+=("${idx}"); done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
  local IFS=,; echo "${free[*]:0:${want}}"; }
GPUS="${GPUS:-}"; while [ "$(awk -F, '{print NF}' <<<"${GPUS}")" -ne 4 ] || [ -z "${GPUS}" ]; do GPUS="$(pick_free_gpus 4)"; [ "$(awk -F, '{print NF}' <<<"${GPUS}")" -eq 4 ] && [ -n "${GPUS}" ] && break; echo "$(date -u +%FT%TZ) GPU_WAIT idle='${GPUS}'"; sleep 60; done
CK=/tmp/verl/ckpts/gemma4-12br-distill-v1; for d in "$CK"/*/global_step_*; do [ -d "$d" ] && { echo "purging stale local checkpoint $d"; rm -rf "$d"; }; done
echo "$(date -u +%FT%TZ) launching distillation on GPUs ${GPUS} (/tmp free: $(df -h /tmp | tail -1 | awk '{print $4}'))"
export TEACHER_SPEC=12br-medium STUDENT=e4b DISTILL_GPU_IDS="${GPUS}"
export ALLOW_UNDERSIZED_STUDENT_LAYOUT=true FSDP_OFFLOAD="${FSDP_OFFLOAD:-false}" DISTILL_RESERVE_GPU_GB="${DISTILL_RESERVE_GPU_GB:-60}"
export TRAIN_SAMPLES_PER_QUESTION=16 TRAIN_BATCH_SIZE=128 TOTAL_TRAINING_STEPS=1000 TOTAL_EPOCHS=100
export LR="${LR:-2e-6}" LR_WARMUP_STEPS=100 LR_SCHEDULER_TYPE=linear MIN_LR_RATIO=0.1 TEST_FREQ=10
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}" MAX_PADDED_TOKENS_PER_MICROBATCH="${MAX_PADDED_TOKENS_PER_MICROBATCH:-8192}"
export FULL_VOCAB_KL_CHUNK_SIZE="${FULL_VOCAB_KL_CHUNK_SIZE:-4096}" VERL_FAIL_ON_NONFINITE_GRAD="${VERL_FAIL_ON_NONFINITE_GRAD:-0}"
export SAVE_FREQ=250 ROLLING_CHECKPOINT_FREQ=50 REMOTE_CHECKPOINT_ENABLE=true ROLLING_HF_EXPORT=true ROLLING_HF_EXPORT_S3=true
export REMOTE_CHECKPOINT_S3_URI="${REMOTE_CHECKPOINT_S3_URI:-s3://scale-ml/genai/rl-distill/gemma4-12br-distill-ckpts-v1/12br-medium-to-e4b-bs128-s1000-lr${LR}}"
export CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra","hf_model"]' MAX_CKPT_TO_KEEP=1
export HF_PUSH_ENABLE=false HF_PUSH_DELETE_LOCAL=false
export EXP_NAME="${EXP_NAME:-12br-medium-to-e4b-base-bs128-s1000-lr${LR}-g4}"
export VENV="${VENV:-/tmp/.venv-gemma4}" AWS_PROFILE="${AWS_PROFILE:-ml-worker}"
exec bash rl-distill-scripts/scale_train/run_gemma4_distill_one.sh
