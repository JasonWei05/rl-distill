#!/usr/bin/env bash
# §9.0g step 2: top-128 forward-KL distillation of the RL'd distilled 12B (§9.0 best, step 190; trace spec 12bd-medium)
# into the untrained E4B base with the §9 recipe, on local GPUs 0-3 (was 0,2 + offload until the user cleared 0-3 at 22:11Z).
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
echo "$(date -u +%FT%TZ) bundle complete; waiting for our trace engines to exit"
until [[ $(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -c EngineCore) -eq 0 || -z $(pgrep -f run_gemma4_bestckpt_trace_collection.sh | head -1) ]]; do sleep 20; done
# GPUs 0-3 are ours (user, 2026-09-14 22:11Z: "sudo kill everything on gpus 0-3"): clear whatever landed on them since.
GPUS="${DISTILL_GPU_IDS:-0,1,2,3}"
clear_gpus() {  # stop Docker containers (restart policy off) and kill bare processes that hold our GPUs
  for g in ${GPUS//,/ }; do u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$g")
    for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$u" '$1==u{print $2}'); do
      [[ $(ps -o user= -p "$p" 2>/dev/null || true) == jasonwei ]] && continue   # ours
      cid=$(grep -oE 'docker[-/][0-9a-f]{64}' /proc/"$p"/cgroup 2>/dev/null | head -1 | grep -oE '[0-9a-f]{64}' || true)   # bare process -> empty (set -e safe)
      if [[ -n $cid ]]; then sudo -n docker update --restart=no "$cid" >/dev/null 2>&1; sudo -n docker stop -t 2 "$cid" >/dev/null 2>&1 && echo "$(date -u +%FT%TZ) stopped container ${cid:0:12} ($(sudo -n docker inspect -f '{{.Name}}' "$cid" 2>/dev/null)) on GPU $g"; fi
      sudo kill -9 "$p" 2>/dev/null && echo "$(date -u +%FT%TZ) cleared pid $p from GPU $g"
    done; done
}
clear_gpus; sleep 3
# startup guard: keep clearing intruders on our GPUs until every rank holds its reservation (up to 25 min; the trainer's reservation retries meanwhile)
( start=$(date +%s); while [ $(( $(date +%s) - start )) -lt 1500 ]; do held=0; for g in ${GPUS//,/ }; do u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$g"); m=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | awk -F', ' -v u="$u" '$1==u{print $2, $3}' | while read -r p mm; do [[ $(ps -o user= -p "$p" 2>/dev/null || true) == jasonwei ]] && echo "${mm%% *}"; done | sort -n | tail -1); [ "${m:-0}" -ge 50000 ] && held=$((held+1)); done; n=$(echo "${GPUS//,/ }" | wc -w); [ "$held" -ge "$n" ] && { echo "$(date -u +%FT%TZ) guard: all $n ranks hold their GPUs"; exit 0; }; clear_gpus; sleep 3; done; echo "$(date -u +%FT%TZ) guard: timeout" ) &
# local checkpoints of a previous process are not pruned by the new trainer (max_ckpt_to_keep only tracks its own saves) and
# each is ~110 GB; every completed step is on S3, so purge them before launching (2026-09-15: a stale one helped fill /tmp).
CK=/tmp/verl/ckpts/gemma4-12bd-distill-v1; for d in "$CK"/*/global_step_*; do [ -d "$d" ] && { echo "$(date -u +%FT%TZ) purging stale local checkpoint $d"; rm -rf "$d"; }; done
echo "$(date -u +%FT%TZ) /tmp free: $(df -h /tmp | tail -1 | awk '{print $4}')"
echo "$(date -u +%FT%TZ) launching distillation on GPUs ${GPUS}"
export TEACHER_SPEC=12bd-medium STUDENT=e4b DISTILL_GPU_IDS="${GPUS}"
# 4 GPUs = the §4 E4B layout (fp32 master + Adam sharded 4-way, no offload). Reserve 60 GB per rank at startup so the box's
# free-GPU schedulers do not land jobs next to us (DISTILL_RESERVE_GPU_GB, main_full_vocab_distill_fsdp2.py).
export ALLOW_UNDERSIZED_STUDENT_LAYOUT=true FSDP_OFFLOAD="${FSDP_OFFLOAD:-false}" DISTILL_RESERVE_GPU_GB="${DISTILL_RESERVE_GPU_GB:-60}"
export TRAIN_SAMPLES_PER_QUESTION=16 TRAIN_BATCH_SIZE=128 TOTAL_TRAINING_STEPS=1000 TOTAL_EPOCHS=100
export LR=2e-6 LR_WARMUP_STEPS=100 LR_SCHEDULER_TYPE=linear MIN_LR_RATIO=0.1 TEST_FREQ=10
export FULL_VOCAB_KL_CHUNK_SIZE="${FULL_VOCAB_KL_CHUNK_SIZE:-4096}"
export SAVE_FREQ=250 ROLLING_CHECKPOINT_FREQ=50 REMOTE_CHECKPOINT_ENABLE=true ROLLING_HF_EXPORT=true ROLLING_HF_EXPORT_S3=true
export REMOTE_CHECKPOINT_S3_URI="${REMOTE_CHECKPOINT_S3_URI:-s3://scale-ml/genai/rl-distill/gemma4-12bd-distill-ckpts-v1/12bd-medium-to-e4b-bs128-s1000-lr2e-6}"
export CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra","hf_model"]' MAX_CKPT_TO_KEEP=1
export HF_PUSH_ENABLE=false HF_PUSH_DELETE_LOCAL=false
export EXP_NAME="${EXP_NAME:-12bd-medium-to-e4b-base-bs128-s1000-lr2e-6-g4}"
export VENV=/tmp/.venv-gemma4 AWS_PROFILE=ml-worker
exec bash rl-distill-scripts/scale_train/run_gemma4_distill_one.sh
