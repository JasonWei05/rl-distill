#!/usr/bin/env bash
# DAPO RL on the DeepScaleR *medium* band, initialized from the tail-bucket on-policy-distilled 12B at step 50 (the best offline
# reverse-KL checkpoint: 0.0713 nats/token to the E4B base teacher, DISTILLATION_EXPERIMENTS.md §9.0f), on FOUR local GPUs.
# Recipe = scale_train/launch_gemma4_12b_distilled_rl_medium.sh (the §9.0 distilled-12B RL run: DAPO, 64 prompts × 16, lr 1e-6,
# early stopping patience 5 on val mean@16, cap 400 steps, saves every 10 + rolling S3 every 5, no Hub pushes) with the local
# 4-GPU layout proven by local_jobs/resume_gemma4_12b_medium_local4.sh: fast update layout (mbs 4 / 8192-token cap / phase-level
# OFFLOAD + vLLM sleep), everything on local NVMe, a Ray head with fixed agent ports, GPU wait + guard, relaunch loop (resumes from
# the newest S3 checkpoint).
#
#   bash rl-distill-scripts/local_jobs/rl_12b_onpolicy_bkt50_local4.sh              # picks 4 idle GPUs (waits for them)
#   GPUS=0,1,2,3 DRY_RUN=1 bash rl-distill-scripts/local_jobs/rl_12b_onpolicy_bkt50_local4.sh   # compose config only
set -euo pipefail
REPO=/mnt/efs/jasonwei/rl-distill
cd "${REPO}"
if [ -f .env ]; then set -a; source .env; set +a; fi

INIT_PATH="${INIT_PATH:-/tmp/gemma4_onpolicy_bkt_step50/hf}"   # S3: …-onpolicy-full-checkpoints/12b-medium-onpolicy-studenttop128-bucket-from-e4bbase-distill/global_step_50/actor/huggingface/
RUN_TAG="${RUN_TAG:-from-onpolicy-bkt50-es5}"
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/gemma4-12b-from-onpolicy-bkt50-rl}"
WORK="${WORK:-/tmp/gemma4_12b_rl_${RUN_TAG}}"                    # local NVMe only -- never $HOME/EFS (assert_local_fs.sh enforces this)
export CKPTS_DIR="${CKPTS_DIR:-${WORK}/ckpts}" DATA_DIR="${DATA_DIR:-${WORK}/data}" HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export RAY_DATA_HOME="${RAY_DATA_HOME:-${WORK}/verl}" RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_12b_rl_${RUN_TAG}}"
export VENV="${VENV:-/tmp/.venv-gemma4}"                          # local copy (setup_env_gemma4.sh + rl-distill-scripts/patches/); EFS venv works but imports slowly
export AWS_PROFILE="${AWS_PROFILE:-ml-worker}"                    # S3 writes (the instance role is read-only)
LOG_DIR="${WORK}/logs"; mkdir -p "${LOG_DIR}" "${DATA_DIR}" "${HF_HOME}"

test -x "${VENV}/bin/python" || { echo "FATAL: ${VENV}/bin/python missing (VENV=/mnt/efs/jasonwei/rl-distill/.venv-gemma4 to use the EFS venv)" >&2; exit 2; }
for f in config.json processor_config.json; do test -s "${INIT_PATH}/${f}" || { echo "FATAL: ${INIT_PATH}/${f} missing (download the step-50 export from S3 and add the 12B base processor_config.json)" >&2; exit 2; }; done
ls "${INIT_PATH}"/*.safetensors >/dev/null || { echo "FATAL: no safetensors in ${INIT_PATH}" >&2; exit 2; }
export GEMMA4_INIT_MODEL_PATH="${INIT_PATH}"

# ---- GPUs: wait for 4 idle ones (shared box) -----------------------------------------------------------------------------------
pick_free_gpus() {
  local want="$1" busy_uuids idx uuid used free=()
  busy_uuids="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u || true)"
  while IFS=', ' read -r idx uuid used; do
    if [ "${used}" -lt 512 ] && ! grep -qx "${uuid}" <<<"${busy_uuids}"; then free+=("${idx}"); fi
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
  local IFS=,; echo "${free[*]:0:${want}}"
}
if [ -z "${GPUS:-}" ]; then
  GPU_WAIT_HOURS="${GPU_WAIT_HOURS:-24}"; waited=0; GPUS="$(pick_free_gpus 4)"
  while [ "$(awk -F, '{print NF}' <<<"${GPUS}")" -ne 4 ] || [ -z "${GPUS}" ]; do
    [ $((waited % 600)) -eq 0 ] && echo "GPU_WAIT $(date -u +%FT%TZ) idle='${GPUS}' waited=$((waited / 60))min"
    [ "${waited}" -ge $((GPU_WAIT_HOURS * 3600)) ] && { echo "FATAL: no 4 idle GPUs within ${GPU_WAIT_HOURS}h" >&2; exit 2; }
    sleep 60; waited=$((waited + 60)); GPUS="$(pick_free_gpus 4)"
  done
fi
n_gpus="$(awk -F, '{print NF}' <<<"${GPUS}")"; [ "${n_gpus}" -eq 4 ] || { echo "FATAL: need 4 GPUs, got '${GPUS}'" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${GPUS}"; echo "USING_GPUS=${GPUS}"

# ---- GPU guard: kill non-owner processes that land on our GPUs mid-run (co-tenant evaluators grab GPUs that look idle) ------------
guard_loop() {
  local uuids; uuids="$(for i in ${GPUS//,/ }; do nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${i}"; done)"
  while true; do
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | while IFS=', ' read -r uuid pid; do
      grep -qx "${uuid}" <<<"${uuids}" || continue
      owner="$(ps -o user= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
      if [ -n "${owner}" ] && [ "${owner}" != "jasonwei" ]; then
        echo "$(date -u +%FT%TZ) GUARD killing pid=${pid} user=${owner} on gpu=${uuid} cmd=$(ps -o args= -p "${pid}" | cut -c1-120)" >> "${LOG_DIR}/guard.log"
        sudo kill -9 "${pid}" 2>/dev/null || kill -9 "${pid}" 2>/dev/null || true
      fi
    done; sleep 30
  done
}
if [ "${GUARD:-1}" = 1 ] && [ "${DRY_RUN:-0}" != 1 ]; then guard_loop & GUARD_PID=$!; trap 'kill ${GUARD_PID} 2>/dev/null || true' EXIT; fi

# ---- Ray head with fixed agent ports (Ray 2.5x raylet otherwise waits a hardcoded 15 s for the agents' port files) ---------------
export NCCL_SOCKET_IFNAME=lo NCCL_SOCKET_FAMILY=AF_INET GLOO_SOCKET_IFNAME=lo
export RAY_RAYLET_START_WAIT_TIME_S=1200 RAY_gcs_server_port_wait_time_s=1200 RAY_worker_register_timeout_seconds=900 RAY_local_fs_capacity_threshold=0.99
RAY_PORT_BASE="${RAY_PORT_BASE:-56490}"
export RUN_RAY_ADDRESS="127.0.0.1:${RAY_PORT_BASE}"
start_ray_head() {
  "${VENV}/bin/ray" stop --grace-period 10 >/dev/null 2>&1 || true
  local s=$(date +%s)
  "${VENV}/bin/ray" start --head --port "${RAY_PORT_BASE}" --node-manager-port $((RAY_PORT_BASE + 1)) \
    --object-manager-port $((RAY_PORT_BASE + 2)) --dashboard-agent-grpc-port $((RAY_PORT_BASE + 3)) \
    --dashboard-agent-listen-port $((RAY_PORT_BASE + 4)) --runtime-env-agent-port $((RAY_PORT_BASE + 5)) \
    --metrics-export-port $((RAY_PORT_BASE + 6)) --ray-client-server-port $((RAY_PORT_BASE + 7)) \
    --include-dashboard=false --disable-usage-stats --temp-dir "${RAY_TEMP_DIR}" --num-gpus "${n_gpus}" \
    > "${LOG_DIR}/ray_head_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 || { echo "FATAL: ray start --head failed (see ${LOG_DIR}/ray_head_*.log)" >&2; return 1; }
  echo "RAY_HEAD_UP address=${RUN_RAY_ADDRESS} gpus=${n_gpus} in $(( $(date +%s) - s )) s"
}

# ---- run contract (= launch_gemma4_12b_distilled_rl_medium.sh, local 4-GPU flavour) ------------------------------------------
export GEMMA4_MODEL=google/gemma-4-12B GEMMA4_MODEL_REVISION=023679ed352de9bb66cc873c9009ce3482585c08   # architecture/metadata; weights come from INIT_PATH
export DIFFICULTY_DATASET_SOURCE=gemma4_26b_bands DIFFICULTY_DATASET=medium
export DIFFICULTY_DATASET_REPO=JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k DIFFICULTY_DATASET_REVISION=a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db
export DATA_SEED=42 RUN_NAME_SUFFIX="${RUN_TAG}" RUN_SLOT="gemma4-12b-medium-${RUN_TAG}" VERL_VLLM_PORT_BASE="${VERL_VLLM_PORT_BASE:-58000}"
export EXP_NAME="DAPO-gemma4-12b-pt-DeepScaleR-gemma26b-medium-seed42-${RUN_TAG}"
export ACTOR_FSDP_SIZE=-1 ACTOR_LR=1e-6 ACTOR_LR_WARMUP_STEPS=20
export EARLY_STOPPING_ENABLED=True EARLY_STOPPING_INCLUDE_INITIAL_VALIDATION=True EARLY_STOPPING_METRIC='val-core/math/acc/mean@16'
export EARLY_STOPPING_MIN_DELTA=0.0 EARLY_STOPPING_MODE=max EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
export ENABLE_OVERLONG_BUFFER=True OVERLONG_BUFFER_LEN=2048 OVERLONG_PENALTY_FACTOR=1.0 GEN_PROMPT_BSZ=64 GEN_TP=1
export LOG_TRAIN_GENERATIONS=100 LOG_VAL_GENERATIONS=100 MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
export MAX_MODEL_LEN=12288 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=8192 N_RESP_PER_PROMPT=16
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}" ROUTER_REPLAY_MODE=disabled ROUTER_Z_LOSS_COEF=0.0 SP_SIZE=1
export SAVE_FREQ=10 TEST_FREQ=10 TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}" TRAIN_PROMPT_BSZ=64 TRAIN_PROMPT_MINI_BSZ=32
export VAL_BEFORE_TRAIN=True VAL_N=1 VLLM_DISABLE_COMPILE_CACHE=0
# fast update layout, validated on 4 local H100s for the 12B medium resume (~370-490 s/step): the ST launcher used mbs 1 / 4096 / per-layer CPU offload
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}" MAX_PADDED_TOKENS_PER_MICROBATCH="${MAX_PADDED_TOKENS_PER_MICROBATCH:-8192}"
export FSDP_CPU_OFFLOAD_POLICY="${FSDP_CPU_OFFLOAD_POLICY:-False}" OFFLOAD="${OFFLOAD:-True}" VLLM_SLEEP_MODE="${VLLM_SLEEP_MODE:-True}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}" VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-10737418240}"
export VERL_SKIP_VLLM_MM_WEIGHT_RELOAD=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_PUSH_ENABLE=False HF_PUSH_REQUIRED=False HF_PUSH_FREQ=10 HF_PUSH_MAX_TO_KEEP=8
export ROLLING_CHECKPOINT_ENABLED=True ROLLING_CHECKPOINT_FREQ=5
export FULL_CHECKPOINT_S3_URI="${S3_BASE}-full-checkpoints/12b-medium-${RUN_TAG}" RUN_ARTIFACT_S3_URI="${S3_BASE}/gemma4-12b-medium-${RUN_TAG}"
export WANDB_RUN_ID="g4ds26b-12b-medium-${RUN_TAG}-s42-v1" WANDB_RESUME=allow
export DRY_RUN="${DRY_RUN:-0}"

if [ "${DRY_RUN}" = 1 ]; then
  bash rl-distill-scripts/scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh; exit $?
fi
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  log="${LOG_DIR}/run_$(date -u +%Y%m%dT%H%M%SZ)_attempt${attempt}.log"
  echo "ATTEMPT ${attempt}/${MAX_ATTEMPTS} log=${log}"
  start_ray_head || { sleep 120; continue; }
  set +e
  bash rl-distill-scripts/scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh "+ray_kwargs.ray_init.runtime_env.env_vars.AWS_PROFILE='${AWS_PROFILE}'" > "${log}" 2>&1
  rc=$?
  set -e
  if grep -q "RUN_DONE rc=0" "${log}" || grep -q "RUN_ALREADY_COMPLETE" "${log}"; then echo "RUN_FINISHED attempt=${attempt}"; exit 0; fi
  echo "$(date -u +%FT%TZ) attempt ${attempt} exited rc=${rc}; last lines:"; tail -5 "${log}" | cut -c1-200
  "${VENV}/bin/ray" stop --grace-period 30 >/dev/null 2>&1 || true
  sleep 120
done
echo "RUN_GAVE_UP after ${MAX_ATTEMPTS} attempts" >&2; exit 1
