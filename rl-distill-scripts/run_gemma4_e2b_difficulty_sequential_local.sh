#!/usr/bin/env bash
# Gemma 4 E2B difficulty RL, easy -> medium -> hard back to back on ONE local node (2 GPUs by
# default), no ScaleTrain, no S3. Each band:
#   * full local checkpoint every SAVE_FREQ (10) steps, HF push of the weight-only export every
#     HF_PUSH_FREQ (10) steps keeping only the newest HF_PUSH_MAX_TO_KEEP (5) step folders on the Hub;
#   * validation every TEST_FREQ (10) steps; early-stop after EARLY_STOPPING_PATIENCE (4)
#     consecutive validations that do not set a new all-time best val-core/math/acc/mean@16;
#   * the next band starts automatically when the previous one exits with RUN_DONE rc=0.
# Requires HF_TOKEN (write) in the repo-root .env or the environment. If it is missing the
# launcher WAITS for it (poll every 60s) instead of failing, so it can be started ahead of time.
#
#   nohup bash rl-distill-scripts/run_gemma4_e2b_difficulty_sequential_local.sh \
#       > ~/verl/logs/g4_e2b_sequential.log 2>&1 &
# Several chains can share a node if each gets its own CUDA_VISIBLE_DEVICES, DIFFICULTY_SEQUENCE and
# VERL_VLLM_PORT_BASE (see run_gemma4_e2b_medium_hard_parallel_local.sh).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PATH="${HOME}/.local/bin:${PATH}"

load_dotenv() { if [ -f .env ]; then set -a; source .env; set +a; fi; }
load_dotenv
until [ -n "${HF_TOKEN:-}" ]; do
  echo "[$(date +%F' '%T)] WAITING_FOR_HF_TOKEN: add HF_TOKEN=... to ${PROJECT_ROOT}/.env"
  sleep 60; load_dotenv
done
export HF_TOKEN

export VENV="${VENV:-${PROJECT_ROOT}/.venv-gemma4}"
[ -x "${VENV}/bin/python" ] || { echo "FATAL: ${VENV} missing; run rl-distill-scripts/setup_env_gemma4.sh" >&2; exit 2; }
HF_USER="$("${VENV}/bin/python" -c 'from huggingface_hub import whoami; print(whoami()["name"])')"
echo "HF_TOKEN_OK user=${HF_USER}"
HF_PUSH_NAMESPACE="${HF_PUSH_NAMESPACE:-${HF_USER}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RAY_ADDRESS=local
export HF_HOME="${HF_HOME:-${HOME}/hf_cache}"
# Single local node: loopback for NCCL/Gloo (the child wrapper defaults to ScaleTrain's eth0/IPv6).
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}" NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

SEED="${DATA_SEED:-42}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-local2gpu}"   # keeps HF repos / run names distinct from the cluster sweep
ROOT="${SEQUENTIAL_ROOT:-${HOME}/gemma4-e2b-difficulty-s${SEED}}"
LOG_DIR="${ROOT}/logs"; mkdir -p "${LOG_DIR}"
read -r -a BANDS <<<"${DIFFICULTY_SEQUENCE:-easy medium hard}"

# W&B only if a key is present; otherwise console-only (wandb.init would block on login).
LOGGER_ARGS=()
if [ -z "${WANDB_API_KEY:-}" ]; then LOGGER_ARGS=('trainer.logger=["console"]'); fi
# Ray pins every worker to OMP_NUM_THREADS=1 unless the variable is preset. With the FSDP2 CPU offload
# policy the gradient accumulation (`sharded_grad += new_grad` on 9.6 GB of fp32 per micro-batch) and the
# AdamW step run on the CPU, so a single thread made update_actor ~700 s (vs ~170 s all-GPU) on
# 2026-09-03 while the worker sat at exactly 100% CPU. Give the workers real threads (execution only,
# no effect on the optimizer trajectory).
EXTRA_ARGS=("+ray_kwargs.ray_init.runtime_env.env_vars.OMP_NUM_THREADS='${WORKER_OMP_THREADS:-16}'")
# Threads alone only cut update_actor from ~700 s to ~600 s: py-spy showed the time in FSDP2's
# blocking pageable D2H copy of each micro-batch's fp32 grad shard (~100 x 9.6 GB per step). The fork
# patch verl/utils/fsdp2_cpu_offload_pinned_patch.py routes that copy through cached pinned memory
# (same bytes, async DMA). Set FSDP2_PINNED_ACCUM=0 to run stock torch.
EXTRA_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FSDP2_CPU_OFFLOAD_PINNED_ACCUM='${FSDP2_PINNED_ACCUM:-1}'")
# vLLM stays resident (sleep mode off) and its caching allocator kept ~10 GB/GPU of generation workspace
# through the trainer's update (GPU peaks of 80.5 of 81.5 GB with both bands running). Fork hook: release
# it right after each generation round. ROLLOUT_RELEASE_CACHE_AFTER_GEN=0 disables.
EXTRA_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.VERL_ROLLOUT_RELEASE_CACHE_AFTER_GEN='${ROLLOUT_RELEASE_CACHE_AFTER_GEN:-1}'")

for band in "${BANDS[@]}"; do
  case "${band}" in easy|medium|hard) ;; *) echo "FATAL: bad band ${band}" >&2; exit 2;; esac
  ckpts="${ROOT}/${band}/ckpts"; data="${ROOT}/${band}/data"; mkdir -p "${ckpts}" "${data}"
  log="${LOG_DIR}/${band}.log"
  if [ -f "${ckpts}/run_outcome.json" ] && grep -q '"final_step"' "${ckpts}/run_outcome.json"; then
    echo "SEQUENTIAL_SKIP_COMPLETE band=${band} (${ckpts}/run_outcome.json exists)"; continue
  fi
  # One Hub repo per band; namespace defaults to the token owner's account.
  hf_repo="${HF_PUSH_NAMESPACE}/DAPO-gemma4-e2b-PT-DeepScaleR-gemma26b-${band}-seed${SEED}-${RUN_NAME_SUFFIX}"
  echo "[$(date +%F' '%T)] SEQUENTIAL_LAUNCH band=${band} hf_repo=${hf_repo} log=${log}"
  # Memory layout (2026-09-03, 2x80GB per band, vLLM resident because sleep mode is off in the wrapper):
  #   * FSDP2 CPU offload policy ON (the 4-GPU sweep's knob): fp32 params/grads/Adam live on the host
  #     and only the current layer is all-gathered to GPU. With the phase-level offload (OFFLOAD=True,
  #     policy off) the 2-way sharded fp32 state alone is ~38 GB per GPU for the 4.8B-param E2B
  #     (9.6 params + 9.6 grads + 19.2 Adam) and medium OOMed in step 2's update_actor at 47 GB
  #     allocated next to a 30 GB vLLM process. On 4 GPUs that state halves, which is why the phase-level
  #     variant only ever fit there.
  #   * 4 GiB vLLM KV cache (vs the wrapper's 512 MiB) for rollout concurrency. Neither knob changes
  #     the optimizer trajectory.
  # vLLM stays eager (the wrapper default). With ROLLOUT_ENFORCE_EAGER=False (FULL_AND_PIECEWISE CUDA
  # graphs, max_num_seqs 1024) the resident vLLM engine process grew to 46-58 GB per GPU on 2026-09-03
  # and both medium and hard OOMed at step 1 (update_actor / compute_log_prob) on 80 GB cards.
  # Micro-batches stay at 8 seqs / 12k padded tokens: 16 / 24k OOMed on 2x80GB during backward
  # (FSDP2 all-gathers the ~6 GiB root unit holding both 262k-vocab embedding tables while the
  # micro-batch's logits are live; trainer 57 GB + vLLM 17 GB).
  set +e
  CKPTS_DIR="${ckpts}" DATA_DIR="${data}" \
  RAY_TEMP_DIR="/tmp/ray_g4_e2b_${band}" \
  VLLM_CACHE_ROOT="/tmp/vllm-cache-g4-e2b-${band}" TRITON_CACHE_DIR="/tmp/triton-cache-g4-e2b-${band}" \
  TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor-cache-g4-e2b-${band}" \
  GEMMA4_MODEL=google/gemma-4-E2B \
  GEMMA4_MODEL_REVISION="${GEMMA4_MODEL_REVISION:-d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f}" \
  DIFFICULTY_DATASET_SOURCE=gemma4_26b_bands DIFFICULTY_DATASET="${band}" \
  DATA_SEED="${SEED}" RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX}" \
  TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}" \
  TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-64}" GEN_PROMPT_BSZ="${GEN_PROMPT_BSZ:-64}" \
  N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-16}" TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-32}" \
  ACTOR_LR="${ACTOR_LR:-1e-6}" ACTOR_LR_WARMUP_STEPS="${ACTOR_LR_WARMUP_STEPS:-20}" \
  MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}" \
  OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-2048}" ENABLE_OVERLONG_BUFFER=True OVERLONG_PENALTY_FACTOR=1.0 \
  MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-8}" \
  MAX_PADDED_TOKENS_PER_MICROBATCH="${MAX_PADDED_TOKENS_PER_MICROBATCH:-12288}" \
  SP_SIZE=1 GEN_TP=1 ACTOR_FSDP_SIZE=-1 \
  FSDP_CPU_OFFLOAD_POLICY="${FSDP_CPU_OFFLOAD_POLICY:-True}" OFFLOAD="${OFFLOAD:-False}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  ROUTER_REPLAY_MODE=disabled ROUTER_Z_LOSS_COEF=0.0 \
  ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.25}" \
  VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-4294967296}" \
  ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}" VLLM_DISABLE_COMPILE_CACHE=0 \
  TEST_FREQ="${TEST_FREQ:-10}" SAVE_FREQ="${SAVE_FREQ:-10}" \
  MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-100}" \
  HF_PUSH_ENABLE=True HF_PUSH_REQUIRED=True \
  HF_PUSH_FREQ="${HF_PUSH_FREQ:-10}" HF_PUSH_MAX_TO_KEEP="${HF_PUSH_MAX_TO_KEEP:-5}" \
  HF_PUSH_REPO="${hf_repo}" \
  ROLLING_CHECKPOINT_ENABLED=False \
  EARLY_STOPPING_ENABLED=True EARLY_STOPPING_METRIC='val-core/math/acc/mean@16' EARLY_STOPPING_MODE=max \
  EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-4}" EARLY_STOPPING_MIN_DELTA=0.0 \
  EARLY_STOPPING_INCLUDE_INITIAL_VALIDATION=True \
  LOG_VAL_GENERATIONS=100 LOG_TRAIN_GENERATIONS=100 \
  EXP_NAME="${EXP_NAME_PREFIX:-g4-e2b}-${band}-s${SEED}-${RUN_NAME_SUFFIX}" \
    bash rl-distill-scripts/scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh \
      "${LOGGER_ARGS[@]}" "${EXTRA_ARGS[@]}" "$@" 2>&1 | tee -a "${log}"
  rc="${PIPESTATUS[0]}"
  set -e
  if [ "${rc}" -ne 0 ] || ! grep -q 'RUN_DONE rc=0' "${log}"; then
    echo "SEQUENTIAL_CHILD_FAILED band=${band} rc=${rc} log=${log}" >&2; exit 1
  fi
  echo "[$(date +%F' '%T)] SEQUENTIAL_CHILD_DONE band=${band}"
  # Tear down only this band's Ray session (all its processes carry RAY_TEMP_DIR in argv). A node-wide
  # `ray stop` would also kill sibling chains running other bands on the same node.
  pkill -f "ray_g4_e2b_${band}" 2>/dev/null || true
  sleep "${SEQUENTIAL_COOLDOWN_SECONDS:-30}"
done
echo "SEQUENTIAL_GEMMA4_E2B_DIFFICULTY_RUNS_DONE"
