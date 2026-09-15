#!/usr/bin/env bash
# Exact local resume (weights + Adam + LR schedule + dataloader cursor + early-stopping history) of the untrained-12B
# DeepScaleR *medium* DAPO run (DAPO-gemma4-12b-pt-DeepScaleR-gemma26b-medium-seed42-26b-bands-es5) from its step-130
# checkpoint on FOUR local GPUs.  The ScaleTrain run was sharded at world_size=8; reshard_fsdp2_checkpoint.py
# rewrites the per-rank shards to world_size=4 (bit-exact, see DISTILLATION_EXPERIMENTS.md §9.0h) and this script
# drives scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh against that directory.
#
# Differences from the ScaleTrain launcher (scale_train/launch_gemma4_12b_medium_resume.sh): 4 GPUs (picked at start
# from the GPUs that are actually idle), checkpoints on EFS (the /tmp volume is full), a NEW S3 prefix (…/12b-medium-local4)
# so the 4-rank checkpoints never mix with the 8-rank history, MAX_ACTOR_CKPT_TO_KEEP=2, and a relaunch loop that resumes
# from the newest checkpoint if the run dies.  Same W&B run id (curve continues), same fast layout (mbs 4 / 8192-token
# cap / OFFLOAD + vLLM sleep), patience 4 (migrated from the original patience 1).
#
#   bash rl-distill-scripts/local_jobs/resume_gemma4_12b_medium_local4.sh            # picks 4 free GPUs
#   GPUS=0,1,2,4 bash rl-distill-scripts/local_jobs/resume_gemma4_12b_medium_local4.sh
set -euo pipefail
REPO=/mnt/efs/jasonwei/rl-distill
cd "${REPO}"
if [ -f .env ]; then set -a; source .env; set +a; fi

WORK="${WORK:-/mnt/efs/jasonwei/gemma4-12b-medium-s42-local4}"
export CKPTS_DIR="${CKPTS_DIR:-${WORK}/ckpts}"                 # holds the resharded global_step_130 + latest_checkpointed_iteration.txt
export DATA_DIR="${DATA_DIR:-/tmp/gemma4_12b_local_test/data}" # medium band parquet already materialized by the local layout test
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"                     # google/gemma-4-12B@023679ed already cached here
export RAY_DATA_HOME="${RAY_DATA_HOME:-${WORK}/verl}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_12b_medium_local4}"
export VENV="${VENV:-/tmp/.venv-gemma4}"                       # node.py patched for RAY_RAYLET_START_WAIT_TIME_S on the loaded box
LOG_DIR="${WORK}/logs"; mkdir -p "${LOG_DIR}" "${DATA_DIR}"

# ---- GPUs: take 4 that are idle right now (no compute process, < 512 MiB used) ------------------------------------
pick_free_gpus() {
  # NB: no `| head` inside the pipeline -- with pipefail, head closing the pipe early makes the function fail (exit 141)
  # and `set -e` then kills the script before it prints anything (2026-09-15, two silent launches).
  local want="$1" busy_uuids idx uuid used free=()
  busy_uuids="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u || true)"
  while IFS=', ' read -r idx uuid used; do
    if [ "${used}" -lt 512 ] && ! grep -qx "${uuid}" <<<"${busy_uuids}"; then free+=("${idx}"); fi
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
  local IFS=,; echo "${free[*]:0:${want}}"
}
if [ -z "${GPUS:-}" ]; then GPUS="$(pick_free_gpus 4)"; fi
n_gpus="$(awk -F, '{print NF}' <<<"${GPUS}")"
if [ "${n_gpus}" -ne 4 ]; then
  echo "FATAL: need exactly 4 free GPUs (checkpoint is resharded to world_size=4); free right now: '${GPUS}'" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${GPUS}"
echo "USING_GPUS=${GPUS}"

# ---- checkpoint sanity ----------------------------------------------------------------------------------------------
test -s "${CKPTS_DIR}/latest_checkpointed_iteration.txt" || { echo "FATAL: ${CKPTS_DIR}/latest_checkpointed_iteration.txt missing" >&2; exit 2; }
STEP="$(tr -d '[:space:]' < "${CKPTS_DIR}/latest_checkpointed_iteration.txt")"
STEP_DIR="${CKPTS_DIR}/global_step_${STEP}"
for kind in model optim extra_state; do
  n="$(ls "${STEP_DIR}/actor" 2>/dev/null | grep -c "^${kind}_world_size_4_rank_[0-3]\.pt$" || true)"
  [ "${n}" -eq 4 ] || { echo "FATAL: ${STEP_DIR}/actor has ${n}/4 ${kind} shards for world_size 4" >&2; exit 2; }
done
test -s "${STEP_DIR}/data.pt" || { echo "FATAL: ${STEP_DIR}/data.pt missing (dataloader cursor)" >&2; exit 2; }
test -s "${STEP_DIR}/validation_early_stopping.json" || { echo "FATAL: ${STEP_DIR}/validation_early_stopping.json missing" >&2; exit 2; }
grep -q '"world_size": 4' "${STEP_DIR}/actor/fsdp_config.json" || { echo "FATAL: fsdp_config.json is not world_size 4" >&2; exit 2; }
echo "RESUME_FROM_STEP=${STEP} CKPTS_DIR=${CKPTS_DIR} (4/4/4 shards, data.pt, early-stopping state present)"

# ---- GPU guard: the shared box has evaluator loops that grab any GPU that looks idle (vLLM sleep / OFFLOAD phases
# leave ours briefly empty).  Kill non-jasonwei compute processes that land on OUR GPUs after we start.  GUARD=0 disables.
guard_loop() {
  local uuids pids
  uuids="$(for i in ${GPUS//,/ }; do nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${i}"; done)"
  while true; do
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | while IFS=', ' read -r uuid pid; do
      grep -qx "${uuid}" <<<"${uuids}" || continue
      owner="$(ps -o user= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
      if [ -n "${owner}" ] && [ "${owner}" != "jasonwei" ]; then
        echo "$(date -u +%FT%TZ) GUARD killing pid=${pid} user=${owner} on gpu=${uuid} cmd=$(ps -o args= -p "${pid}" | cut -c1-120)" >> "${LOG_DIR}/guard.log"
        sudo kill -9 "${pid}" 2>/dev/null || kill -9 "${pid}" 2>/dev/null || true
      fi
    done
    sleep 30
  done
}
if [ "${GUARD:-1}" = 1 ]; then guard_loop & GUARD_PID=$!; trap 'kill ${GUARD_PID} 2>/dev/null || true' EXIT; fi

# ---- run contract = scale_train/launch_gemma4_12b_medium_resume.sh, local flavour ---------------------------------
export NCCL_SOCKET_IFNAME=lo NCCL_SOCKET_FAMILY=AF_INET GLOO_SOCKET_IFNAME=lo
export RAY_ADDRESS=local RAY_RAYLET_START_WAIT_TIME_S=600 RAY_gcs_server_port_wait_time_s=600
export GEMMA4_MODEL=google/gemma-4-12B GEMMA4_MODEL_REVISION=023679ed352de9bb66cc873c9009ce3482585c08
export DIFFICULTY_DATASET_SOURCE=gemma4_26b_bands DIFFICULTY_DATASET=medium
export DIFFICULTY_DATASET_REPO=JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k DIFFICULTY_DATASET_REVISION=a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db
export DATA_SEED=42 RUN_NAME_SUFFIX=26b-bands-es5 RUN_SLOT=gemma4-12b-medium-s42-local4 VERL_VLLM_PORT_BASE="${VERL_VLLM_PORT_BASE:-56000}"
export ACTOR_FSDP_SIZE=-1 ACTOR_LR=1e-6 ACTOR_LR_WARMUP_STEPS=20
export EARLY_STOPPING_ENABLED=True EARLY_STOPPING_INCLUDE_INITIAL_VALIDATION=True EARLY_STOPPING_METRIC='val-core/math/acc/mean@16'
export EARLY_STOPPING_MIN_DELTA=0.0 EARLY_STOPPING_MODE=max EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-4}"
export EARLY_STOPPING_MIGRATE_PATIENCE_FROM="${EARLY_STOPPING_MIGRATE_PATIENCE_FROM:-1}"   # original run stopped at patience 1
export ENABLE_OVERLONG_BUFFER=True GEN_PROMPT_BSZ=64 GEN_TP=1 LOG_TRAIN_GENERATIONS=100 LOG_VAL_GENERATIONS=100
export MAX_ACTOR_CKPT_TO_KEEP=2 MAX_MODEL_LEN=12288 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=8192
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}" ROUTER_REPLAY_MODE=disabled ROUTER_Z_LOSS_COEF=0.0 SP_SIZE=1
export SAVE_FREQ=10 TEST_FREQ=10 TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}"
export TRAIN_PROMPT_BSZ=64 TRAIN_PROMPT_MINI_BSZ=32 N_RESP_PER_PROMPT=16 VAL_BEFORE_TRAIN=True VAL_N=1 VLLM_DISABLE_COMPILE_CACHE=0
# fast layout validated locally on 4 GPUs (2026-09-15: 418 s/step, update 289 s, no OOM)
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}" MAX_PADDED_TOKENS_PER_MICROBATCH="${MAX_PADDED_TOKENS_PER_MICROBATCH:-8192}"
export FSDP_CPU_OFFLOAD_POLICY="${FSDP_CPU_OFFLOAD_POLICY:-False}" OFFLOAD="${OFFLOAD:-True}" VLLM_SLEEP_MODE="${VLLM_SLEEP_MODE:-True}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}" VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-10737418240}"
export VERL_SKIP_VLLM_MM_WEIGHT_RELOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# S3 only (no HF pushes); NEW prefixes for the 4-rank continuation
export HF_PUSH_ENABLE=False HF_PUSH_REQUIRED=False HF_PUSH_REPO=JWei05/DAPO-gemma4-12b-PT-DeepScaleR-gemma26b-medium-seed42-26b-bands-es5 HF_PUSH_FREQ=10 HF_PUSH_MAX_TO_KEEP=8
export ROLLING_CHECKPOINT_ENABLED=True ROLLING_CHECKPOINT_FREQ="${ROLLING_CHECKPOINT_FREQ:-5}"
export FULL_CHECKPOINT_S3_URI="${FULL_CHECKPOINT_S3_URI:-s3://scale-ml/genai/rl-distill/gemma4-difficulty-s42-20260819-full-checkpoints/12b-medium-local4}"
export RUN_ARTIFACT_S3_URI="${RUN_ARTIFACT_S3_URI:-s3://scale-ml/genai/rl-distill/gemma4-difficulty-s42-20260819/gemma4-12b-medium-local4}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-g4ds26b-12b-medium-s42-v1}" WANDB_RESUME=allow

# ---- relaunch loop: a crash resumes from the newest complete checkpoint (S3 restore-latest, else local tracker) ----
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  log="${LOG_DIR}/run_$(date -u +%Y%m%dT%H%M%SZ)_attempt${attempt}.log"
  echo "ATTEMPT ${attempt}/${MAX_ATTEMPTS} log=${log}"
  set +e
  bash rl-distill-scripts/scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh \
    "+ray_kwargs.ray_init.runtime_env.env_vars.EARLY_STOPPING_MIGRATE_PATIENCE_FROM='${EARLY_STOPPING_MIGRATE_PATIENCE_FROM}'" \
    > "${log}" 2>&1
  rc=$?
  set -e
  if grep -q "RUN_DONE rc=0" "${log}" || grep -q "RUN_ALREADY_COMPLETE" "${log}"; then echo "RUN_FINISHED attempt=${attempt}"; exit 0; fi
  echo "$(date -u +%FT%TZ) attempt ${attempt} exited rc=${rc}; last lines:"; tail -5 "${log}" | cut -c1-200
  "${VENV}/bin/ray" stop --grace-period 30 >/dev/null 2>&1 || true
  sleep 120
done
echo "RUN_GAVE_UP after ${MAX_ATTEMPTS} attempts" >&2; exit 1
