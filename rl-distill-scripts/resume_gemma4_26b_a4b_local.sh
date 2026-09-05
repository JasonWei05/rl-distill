#!/usr/bin/env bash
# Resume a Gemma 4 26B-A4B difficulty-band DAPO/GRPO run (seed 42) from the latest local checkpoint on a
# single 8x80GB node, with the update-phase configuration measured fastest on 8xH100 (2026-09-03).
#
# Companion to RESUME_GEMMA4_26B_A4B_LOCAL.md (which documents the checkpoint download and what resume
# restores). Everything below marked OVERRIDABLE is `${VAR:-default}`; the rest is the pinned recipe.
#
#   BAND=medium bash rl-distill-scripts/resume_gemma4_26b_a4b_local.sh      # or BAND=hard
#   bash rl-distill-scripts/resume_gemma4_26b_medium_local.sh               # thin per-band wrappers
#   DRY_RUN=1 BAND=medium bash rl-distill-scripts/resume_gemma4_26b_a4b_local.sh
#       -> runs every preflight (venv, R3 patch, checkpoint shards, model + dataset download) and composes
#          the full Hydra config (`--cfg job`, appended last by gemma3_pt_fewshot_math_rl.sh) without starting
#          Ray or touching a GPU.
#
# Measured per-step timings on the same step-51 batch (8xH100 80GB, hard band):
#   recipe  (mbsz 1, 4096 padded cap, FSDP2 CPU offload policy, vLLM resident): gen 410 s, update 1625 s, step 2157 s
#   this    (mbsz 4, 8192 padded cap, phase-level OFFLOAD=True, vLLM sleep mode): gen 267 s, update  110 s, step  439 s
# Peak update-phase memory 76.8 GB/GPU; activations are bounded by MAX_PADDED_TOKENS_PER_MICROBATCH so the peak
# is stable across batches. token-mean loss is normalized by the GLOBAL token count, so the micro-batch size does
# not change the gradient (only bf16 summation order).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
export PATH="${HOME}/.local/bin:${PATH}"
if [ -f .env ]; then set -a; source .env; set +a; fi

# --- band ------------------------------------------------------------------------------------------------------
# RUN_NAME_SUFFIX reproduces the original ScaleTrain run's experiment name, which the wrapper turns into the W&B
# display name and the default HF push repo (JWei05/DAPO-gemma4-26b-a4b-PT-DeepScaleR-gemma26b-<band>-seed42<-suffix>).
BAND="${BAND:?set BAND=medium or BAND=hard}"
case "${BAND}" in
  hard)   BAND_RUN_NAME_SUFFIX="" ;;               # original run: DAPO-gemma4-26b-a4b-pt-DeepScaleR-gemma26b-hard-seed42
  medium) BAND_RUN_NAME_SUFFIX="26b-bands-es5" ;;  # original run: ...-gemma26b-medium-seed42-26b-bands-es5
  *) echo "FATAL: BAND must be medium or hard (easy finished on ScaleTrain); got ${BAND}" >&2; exit 2 ;;
esac
export RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX-${BAND_RUN_NAME_SUFFIX}}"

# --- node-local paths (OVERRIDABLE) -------------------------------------------------------------------------
export CKPTS_DIR="${CKPTS_DIR:-${HOME}/gemma4-26b-${BAND}-s42/ckpts}"   # holds global_step_N/ + latest_checkpointed_iteration.txt
export DATA_DIR="${DATA_DIR:-${HOME}/gemma4-26b-${BAND}-s42/data}"
export HF_HOME="${HF_HOME:-${HOME}/hf_cache}"                            # persistent (the wrapper defaults to /tmp)
export RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_26b_${BAND}_s42}"
export VENV="${VENV:-${REPO}/.venv-gemma4}"
mkdir -p "${DATA_DIR}"

# --- preflight: the shards are keyed world_size_8, so the resume needs exactly 8 GPUs and 8+8+8 shards -----------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
n_gpus="$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
if [ "${n_gpus}" -ne 8 ]; then
  echo "FATAL: the 26B-A4B checkpoints are FSDP2-sharded at world_size=8; got ${n_gpus} GPUs (${CUDA_VISIBLE_DEVICES})" >&2
  exit 2
fi
test -x "${VENV}/bin/python" || { echo "FATAL: ${VENV}/bin/python missing; run rl-distill-scripts/setup_env_gemma4.sh" >&2; exit 2; }
"${VENV}/bin/python" rl-distill-scripts/patch_vllm_gemma4_r3.py   # idempotent; fails closed if the vLLM wheel is wrong
test -s "${CKPTS_DIR}/latest_checkpointed_iteration.txt" || { echo "FATAL: no tracker file in ${CKPTS_DIR}" >&2; exit 2; }
STEP="$(tr -d '[:space:]' < "${CKPTS_DIR}/latest_checkpointed_iteration.txt")"
STEP_DIR="${CKPTS_DIR}/global_step_${STEP}"
n_model="$(ls "${STEP_DIR}/actor" 2>/dev/null | grep -c '^model_world_size_8_rank_[0-7]\.pt$' || true)"
n_optim="$(ls "${STEP_DIR}/actor" 2>/dev/null | grep -c '^optim_world_size_8_rank_[0-7]\.pt$' || true)"
n_extra="$(ls "${STEP_DIR}/actor" 2>/dev/null | grep -c '^extra_state_world_size_8_rank_[0-7]\.pt$' || true)"
if [ "${n_model}" -ne 8 ] || [ "${n_optim}" -ne 8 ] || [ "${n_extra}" -ne 8 ] || [ ! -s "${STEP_DIR}/data.pt" ]; then
  echo "FATAL: ${STEP_DIR} is not a complete world_size_8 checkpoint: model=${n_model}/8 optim=${n_optim}/8 extra=${n_extra}/8 data.pt=$([ -s "${STEP_DIR}/data.pt" ] && echo yes || echo missing)" >&2
  exit 2
fi
if [ ! -s "${STEP_DIR}/validation_early_stopping.json" ]; then
  echo "FATAL: ${STEP_DIR}/validation_early_stopping.json missing; the trainer refuses to reset early-stopping state on resume" >&2
  exit 2
fi
echo "RESUME_FROM_STEP=${STEP} BAND=${BAND} CKPTS_DIR=${CKPTS_DIR} (model/optim/extra shards 8/8/8, data.pt + early-stopping state present)"

# Single node: loopback NCCL/GLOO (the wrapper defaults to eth0/AF_INET6 for ScaleTrain pods).
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}" NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

# --- W&B: the trainer hard-fails at wandb.init without a key. Continue the original curve when a key is present;
# otherwise (or with WANDB_DISABLE=1) log to the console only.
EXTRA_OVERRIDES=()
if [ -n "${WANDB_API_KEY:-}" ] && [ "${WANDB_DISABLE:-0}" != 1 ]; then
  export WANDB_RUN_ID="${WANDB_RUN_ID:-g4ds26b-26b-a4b-${BAND}-s42-v1}" WANDB_RESUME="${WANDB_RESUME:-allow}"
else
  echo "NOTE: wandb disabled or no WANDB_API_KEY -> console logger only"
  EXTRA_OVERRIDES+=("trainer.logger=[\"console\"]")
fi
export DRY_RUN="${DRY_RUN:-0}"   # 1 -> gemma3_pt_fewshot_math_rl.sh appends `--cfg job`: compose the config, start nothing

# --- pinned recipe ---------------------------------------------------------------------------------------------
export RAY_ADDRESS=local RESUME_MODE=auto
export GEMMA4_MODEL=google/gemma-4-26B-A4B
export GEMMA4_MODEL_REVISION=24548b62aa021d562695c04aaf7758a1ea47990b
export DIFFICULTY_DATASET_SOURCE=gemma4_26b_bands DIFFICULTY_DATASET="${BAND}"
export DIFFICULTY_DATASET_REPO=JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k
export DIFFICULTY_DATASET_REVISION=a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db
export DATA_SEED=42
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}"
export TRAIN_PROMPT_BSZ=64 GEN_PROMPT_BSZ=64 N_RESP_PER_PROMPT=16 TRAIN_PROMPT_MINI_BSZ=32
export ACTOR_LR=1e-6 ACTOR_LR_WARMUP_STEPS=20
export MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=8192 MAX_MODEL_LEN=12288
export OVERLONG_BUFFER_LEN=2048 ENABLE_OVERLONG_BUFFER=True OVERLONG_PENALTY_FACTOR=1.0
export SP_SIZE=1 GEN_TP="${GEN_TP:-1}" ACTOR_FSDP_SIZE=-1
export ROUTER_REPLAY_MODE=R3 ROUTER_Z_LOSS_COEF=0.0 VERL_SKIP_VLLM_MM_WEIGHT_RELOAD=1
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.10 VLLM_KV_CACHE_MEMORY_BYTES=3221225472
export ROLLOUT_ENFORCE_EAGER=True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EARLY_STOPPING_ENABLED="${EARLY_STOPPING_ENABLED:-True}" EARLY_STOPPING_METRIC='val-core/math/acc/mean@16'
export EARLY_STOPPING_MODE=max EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-2}" EARLY_STOPPING_MIN_DELTA=0.0
# Changing patience on resume: the trainer refuses a checkpoint whose saved early-stopping config differs from the
# active one unless the old value is named explicitly. EARLY_STOPPING_MIGRATE_PATIENCE_FROM=<old> keeps the saved
# best/miss history and recomputes the trigger flag under the new patience (dapo/validation_early_stopping.py).
# The trainer reads it from the Ray actor's os.environ; local Ray inherits the driver env, and it is also forwarded
# through runtime_env below so this does not depend on inheritance.
if [ -n "${EARLY_STOPPING_MIGRATE_PATIENCE_FROM:-}" ]; then
  export EARLY_STOPPING_MIGRATE_PATIENCE_FROM
  EXTRA_OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.EARLY_STOPPING_MIGRATE_PATIENCE_FROM='${EARLY_STOPPING_MIGRATE_PATIENCE_FROM}'")
fi
export EARLY_STOPPING_INCLUDE_INITIAL_VALIDATION=True
export LOG_VAL_GENERATIONS=100 LOG_TRAIN_GENERATIONS=100

# --- update-phase throughput (OVERRIDABLE; defaults = the measured-fastest configuration above) --------------
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}"
export MAX_PADDED_TOKENS_PER_MICROBATCH="${MAX_PADDED_TOKENS_PER_MICROBATCH:-8192}"
export FSDP_CPU_OFFLOAD_POLICY="${FSDP_CPU_OFFLOAD_POLICY:-False}"   # per-layer FSDP2 CPU offload: every microbatch pays a full param round trip
export OFFLOAD="${OFFLOAD:-True}"                                    # verl phase-level offload: one bulk copy per phase instead
export VLLM_SLEEP_MODE="${VLLM_SLEEP_MODE:-True}"                    # park the 52 GB bf16 vLLM copy in host RAM during the update

# --- checkpoint cadence (OVERRIDABLE) --------------------------------------------------------------------------
# Full resumable checkpoint every 2 steps (~334 GB, ~4 min), newest 2 kept; validation + HF snapshot push every
# 10 steps. verl only prunes checkpoints written by the current process: directories left by earlier resumes
# must be removed by hand. HF_PUSH_FREQ must be a multiple of SAVE_FREQ.
export TEST_FREQ="${TEST_FREQ:-10}" SAVE_FREQ="${SAVE_FREQ:-2}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
export ROLLING_CHECKPOINT_ENABLED=False
# HF Hub snapshots (weights-only, ~53 GB each) go to <repo>/step_0000NN. The pusher prunes the repo to
# HF_PUSH_MAX_TO_KEEP-1 before each upload and squashes history so quota is really released (deleting LFS pointers
# alone frees nothing). Keeping patience+1 snapshots guarantees the early-stopping best is among them (the run
# stops after `patience` non-improving validations, so the best is at most `patience` pushes back).
# HF_PUSH_REQUIRED=False so a Hub storage-quota failure cannot fail a finished training run.
# HF_PUSH_REPO defaults to the wrapper's per-band repo (the original run's repo, see RUN_NAME_SUFFIX above).
export HF_PUSH_ENABLE="${HF_PUSH_ENABLE:-True}" HF_PUSH_FREQ="${HF_PUSH_FREQ:-10}"
export HF_PUSH_MAX_TO_KEEP="${HF_PUSH_MAX_TO_KEEP:-$((EARLY_STOPPING_PATIENCE + 1))}"
export HF_PUSH_REQUIRED="${HF_PUSH_REQUIRED:-False}" HF_PUSH_DELETE_LOCAL_AFTER=False
if [ $((HF_PUSH_FREQ % SAVE_FREQ)) -ne 0 ]; then
  echo "FATAL: HF_PUSH_FREQ=${HF_PUSH_FREQ} must be a multiple of SAVE_FREQ=${SAVE_FREQ}" >&2
  exit 2
fi
# HF push preflight: pushes are non-fatal at run time (HF_PUSH_REQUIRED=False), so a bad token would otherwise
# only surface as "[HFPusher] ... failed" lines hours in. Verify the token can write to the account and create the
# destination repo up front (the pusher itself does create_repo(exist_ok=True, private=False) on first push).
if [ "${HF_PUSH_ENABLE,,}" = true ]; then
  : "${HF_TOKEN:?HF_PUSH_ENABLE=True requires HF_TOKEN in .env}"
  HF_PUSH_REPO="${HF_PUSH_REPO:-JWei05/DAPO-gemma4-26b-a4b-PT-DeepScaleR-gemma26b-${BAND}-seed42${RUN_NAME_SUFFIX:+-${RUN_NAME_SUFFIX}}}"
  export HF_PUSH_REPO
  "${VENV}/bin/python" - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
who = api.whoami()
tok = who.get("auth", {}).get("accessToken", {})
role = tok.get("role")
scoped = {p for s in tok.get("fineGrained", {}).get("scoped", []) for p in s.get("permissions", [])}
if role not in ("write",) and "repo.write" not in scoped:
    raise SystemExit(f"FATAL: HF token for {who['name']} lacks write access (role={role}, scoped={sorted(scoped)})")
repo = os.environ["HF_PUSH_REPO"]
api.create_repo(repo_id=repo, repo_type="model", private=False, exist_ok=True)
print(f"HF_PUSH_PREFLIGHT_OK user={who['name']} role={role} repo={repo} url=https://huggingface.co/{repo}")
PY
fi

echo "KNOBS band=${BAND} es_patience=${EARLY_STOPPING_PATIENCE}${EARLY_STOPPING_MIGRATE_PATIENCE_FROM:+(migrated from ${EARLY_STOPPING_MIGRATE_PATIENCE_FROM})} run_suffix='${RUN_NAME_SUFFIX}' wandb_run_id=${WANDB_RUN_ID:-<console>} sleep_mode=${VLLM_SLEEP_MODE} gen_tp=${GEN_TP} mbsz=${MICRO_BATCH_SIZE_PER_GPU} padded_cap=${MAX_PADDED_TOKENS_PER_MICROBATCH} offload_policy=${FSDP_CPU_OFFLOAD_POLICY} offload=${OFFLOAD} save_freq=${SAVE_FREQ} keep=${MAX_ACTOR_CKPT_TO_KEEP} hf_push=${HF_PUSH_ENABLE}/${HF_PUSH_FREQ}/keep${HF_PUSH_MAX_TO_KEEP} hf_repo=${HF_PUSH_REPO:-<disabled>}"
exec bash rl-distill-scripts/scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh "${EXTRA_OVERRIDES[@]}" "$@"
