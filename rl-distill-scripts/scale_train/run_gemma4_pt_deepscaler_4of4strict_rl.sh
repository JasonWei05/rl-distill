#!/usr/bin/env bash
# ScaleTrain entrypoint: Gemma 4 PT DeepScaleR DAPO RL (single node, full node's GPUs).
# Same recipe as the gemma-3 deepscaler-4of4-strict runs (GRPO n=16, 12-shot prompt, temp 1.0 val=train,
# SAVE_FREQ=25 + HF push /25, wandb, val_before_train, strict boxed-only grader) BUT:
#   * 20k max response length (recipe default: 20480 resp + 4096 overlong buffer, factor 1.0)
#   * the gemma-4 stack — the baked FSDP2 .venv (torch 2.9 / vllm 0.15.1) CANNOT run gemma-4, so this
#     builds .venv-gemma4 (torch 2.11 cu130 / vllm 0.25.1 / transformers 5.14.1, NO flash-attn) into /tmp
#     at job start (all wheels, no compile) and runs from it. Needs the p5 host driver to support CUDA 13.
#   * gemma-4 deltas: wrap class Gemma4TextDecoderLayer, attn_implementation=sdpa, use_remove_padding=False,
#     rollout.update_weights_bucket_megabytes=8192 (its per-layer embedding table is a single ~4.5GB tensor
#     that overflows the default 2GB FSDP->vLLM weight-sync bucket).
#
# Which model via GEMMA4_MODEL env (google/gemma-4-E2B | google/gemma-4-E4B |
# google/gemma-4-12B | google/gemma-4-26B-A4B). Per-seed via DATA_SEED.
#
# Launch (E2B, one node, borrowing OFF):
#   python rl-distill-scripts/scale_train/launch_st_job.py --cluster eks --n-instances 1 \
#     --gpus-per-instance 8 --priority high --job-name gemma4-e2b-ds4of4s \
#     --build-config-key train-rl-distill --team egp --product train.enterprise_rlvr \
#     --run-file run_gemma4_pt_deepscaler_4of4strict_rl.sh \
#     --env-vars GEMMA4_MODEL=google/gemma-4-E2B,DATA_SEED=42
#   # E4B: --job-name gemma4-e4b-ds4of4s --env-vars GEMMA4_MODEL=google/gemma-4-E4B,DATA_SEED=42
#   # Validated E2B 8k / configured microbatch-2 path (long samples split to singleton forwards):
#   # --env-vars MAX_RESPONSE_LENGTH=8192,MICRO_BATCH_SIZE_PER_GPU=2,MAX_PADDED_TOKENS_PER_MICROBATCH=4096,FSDP_CPU_OFFLOAD_POLICY=True
#   # NOTE: --run-file is resolved relative to scale_train/ (bare filename), NOT the repo root — a
#   #       repo-root-relative path doubles to .../scale_train/rl-distill-scripts/scale_train/... and 404s.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

GEMMA4_MODEL="${GEMMA4_MODEL:-google/gemma-4-E2B}"
case "${GEMMA4_MODEL}" in
  *E2B*) MODEL_TAG=gemma4-e2b; WRAP_LAYER_CLS_DEFAULT=Gemma4TextDecoderLayer ;;
  *E4B*) MODEL_TAG=gemma4-e4b; WRAP_LAYER_CLS_DEFAULT=Gemma4TextDecoderLayer ;;
  *12B*) MODEL_TAG=gemma4-12b; WRAP_LAYER_CLS_DEFAULT=Gemma4UnifiedTextDecoderLayer ;;
  *26B-A4B*) MODEL_TAG=gemma4-26b-a4b; WRAP_LAYER_CLS_DEFAULT=Gemma4TextDecoderLayer ;;
  *)
    MODEL_TAG="$(echo "${GEMMA4_MODEL}" | tr '/A-Z' '-a-z')"
    echo "FATAL: unsupported GEMMA4_MODEL=${GEMMA4_MODEL}" >&2
    exit 2
    ;;
esac

WRAP_LAYER_CLS="${WRAP_LAYER_CLS:-${WRAP_LAYER_CLS_DEFAULT}}"

DIFFICULTY_DATASET_SOURCE="${DIFFICULTY_DATASET_SOURCE:-legacy}"
DIFFICULTY_DATASET="${DIFFICULTY_DATASET:-strict4of4}"
if [ "${DIFFICULTY_DATASET_SOURCE}" = gemma4_26b_bands ]; then
  case "${DIFFICULTY_DATASET}" in
    easy|medium|hard)
      DATASET_TAG="gemma26b-${DIFFICULTY_DATASET}"
      TRAIN_BASENAME="deepscaler_gemma4_26b_${DIFFICULTY_DATASET}_train.parquet"
      IN_DIST_VAL_BASENAME="deepscaler_gemma4_26b_${DIFFICULTY_DATASET}_val300_x16.parquet"
      ;;
    *)
      echo "FATAL: gemma4_26b_bands requires DIFFICULTY_DATASET=easy, medium, or hard; got ${DIFFICULTY_DATASET}" >&2
      exit 2
      ;;
  esac
elif [ "${DIFFICULTY_DATASET_SOURCE}" = legacy ]; then
  case "${DIFFICULTY_DATASET}" in
    strict4of4)
      DATASET_TAG=4of4strict
      TRAIN_BASENAME=deepscaler_4of4strict_rl_train.parquet
      IN_DIST_VAL_BASENAME=deepscaler_4of4strict_rl_val200_x16.parquet
      ;;
    easy)
      DATASET_TAG=easy10k
      TRAIN_BASENAME=deepscaler_easy_10k_train.parquet
      IN_DIST_VAL_BASENAME=deepscaler_easy_10k_val500_x16.parquet
      ;;
    medium)
      DATASET_TAG=medium20k
      TRAIN_BASENAME=deepscaler_medium_20k_train.parquet
      IN_DIST_VAL_BASENAME=deepscaler_medium_20k_val500_x16.parquet
      ;;
    *)
      echo "FATAL: legacy DIFFICULTY_DATASET must be strict4of4, easy, or medium; got ${DIFFICULTY_DATASET}" >&2
      exit 2
      ;;
  esac
else
  echo "FATAL: unsupported DIFFICULTY_DATASET_SOURCE=${DIFFICULTY_DATASET_SOURCE}" >&2
  exit 2
fi

SEED="${DATA_SEED:-42}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-}"
NORMALIZED_RUN_SUFFIX="${RUN_NAME_SUFFIX:+-${RUN_NAME_SUFFIX}}"
DEFAULT_EXP_NAME="DAPO-${MODEL_TAG}-pt-DeepScaleR-${DATASET_TAG}-seed${SEED}${NORMALIZED_RUN_SUFFIX}"
DEFAULT_HF_PUSH_REPO="JWei05/DAPO-${MODEL_TAG}-PT-DeepScaleR-${DATASET_TAG}-seed${SEED}${NORMALIZED_RUN_SUFFIX}"

# HF_TOKEN / WANDB_API_KEY arrive as forwarded env vars (launch_st_job --dotenv-keys); .env may be absent.
if [ -f .env ]; then set -a; source .env; set +a; fi

# Build the gemma-4 venv into /tmp (local disk, writable) once per pod. All wheels -> ~10 min, no compile.
# uv is baked into the image by setup_env.sh but under root's ~/.local/bin; ensure it's on PATH (install if not).
export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
export VENV="${VENV:-/tmp/.venv-gemma4}"
# ScaleTrain p5 fleet has a CUDA 12.8 driver -> cu130 wheels fail ("driver too old, found 12080").
# gemma-4 does NOT need CUDA 13: use the cu129 variant (torch cu129 + vllm 0.25.1+cu129 GitHub wheel),
# which runs on 12.8 drivers via CUDA 12.x minor-version compatibility (the official recipe path is cu129).
export GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu129}"
if [ ! -x "${VENV}/bin/python" ]; then
  echo "### building ${VENV} via setup_env_gemma4.sh (${GEMMA4_CUDA_VARIANT})"
  VENV="${VENV}" bash rl-distill-scripts/setup_env_gemma4.sh
fi
export PATH="${VENV}/bin:${PATH}"; source "${VENV}/bin/activate"
# Compiled vLLM/torch-inductor kernels invoke `nvcc` by name.  CUDA is
# installed under /usr/local/cuda on both the local p5 devbox and ScaleTrain
# p5 images, but its bin directory is not guaranteed to be on PATH.  A cold,
# isolated compile cache therefore fails before rollout initialization even
# though the compiler itself is present.  Resolve the canonical toolkit path
# and propagate it explicitly to Ray workers below.
if [ -x /usr/local/cuda/bin/nvcc ]; then
  export CUDA_HOME="${CUDA_HOME:-$(readlink -f /usr/local/cuda)}"
elif [ -x /usr/local/cuda-12.9/bin/nvcc ]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
fi
if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
# Hard guard: the reward path imports math_verify inside worker subprocesses and converts import
# errors into silent 0.0 scores (observed: a full run "trained" with every sample scored 0 because
# the venv lacked math-verify). Fail fast instead.
python3 -c "import math_verify" || { echo "FATAL: math-verify missing from ${VENV}"; exit 1; }
echo "MATH_VERIFY_OK"
# cu130 only: nvrtc.so.13 lives in the nvidia cu13 wheel and is off the default load path (cu129's
# nvidia-*-cu12 libs are found via torch's rpaths, no extra path needed).
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"   # always set (set -u; also referenced in a hydra override)
CU13_LIB="${VENV}/lib/python3.12/site-packages/nvidia/cu13/lib"
if [ -d "${CU13_LIB}" ]; then export LD_LIBRARY_PATH="${CU13_LIB}:${LD_LIBRARY_PATH}"; fi

export RAY_DATA_HOME="${RAY_DATA_HOME:-/tmp/verl}"
export DATA_DIR="${DATA_DIR:-${RAY_DATA_HOME}/data}"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"; export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"; export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
# ScaleTrain pods on ml-gpu-batch are IPv6-only (pod IP 2602:fb33:...), interface eth0.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET6}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  n_gpus="$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
else
  n_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((n_gpus - 1)))"
fi
if [ "${n_gpus}" -lt 1 ]; then
  echo "FATAL: no visible GPUs" >&2
  exit 2
fi
mkdir -p "${DATA_DIR}"
RUN_SLOT="${RUN_SLOT:-${DATASET_TAG}}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
# With padded Gemma 4 forwards, a fixed microbatch can multiply a rare long
# sequence into an OOM-sized logits tensor.  A nonzero ceiling keeps the
# configured microbatch for short samples but emits long samples as singletons.
MAX_PADDED_TOKENS_PER_MICROBATCH="${MAX_PADDED_TOKENS_PER_MICROBATCH:-0}"
# FSDP2's native CPU offload policy keeps parameter shards, gradients, and Adam
# state on CPU between layer computations.  Unlike manual optimizer offload, it
# also avoids the first-step GPU allocation spike when Adam moments are created.
FSDP_CPU_OFFLOAD_POLICY="${FSDP_CPU_OFFLOAD_POLICY:-False}"
# Gemma 4 checkpoints advertise a 131k native context window.  vLLM sizes its
# minimum KV cache against that value unless max_model_len is explicit, even
# though this recipe only serves prompt+response tokens.  Keep the engine's KV
# requirement aligned with the actual training window.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-20480}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-536870912}"
ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
VERL_SKIP_VLLM_MM_WEIGHT_RELOAD="${VERL_SKIP_VLLM_MM_WEIGHT_RELOAD:-0}"
ROUTER_REPLAY_MODE="${ROUTER_REPLAY_MODE:-disabled}"
ROUTER_Z_LOSS_COEF="${ROUTER_Z_LOSS_COEF:-0.0}"
if [ "${ROUTER_Z_LOSS_COEF}" != 0 ] && [ "${ROUTER_Z_LOSS_COEF}" != 0.0 ]; then
  echo "FATAL: Gemma 4 FSDP2 router z-loss is not implemented in this trainer; use ROUTER_Z_LOSS_COEF=0.0" >&2
  exit 2
fi
if [ "${MODEL_TAG}" = gemma4-26b-a4b ]; then
  if [ "${ROUTER_REPLAY_MODE}" != R3 ]; then
    echo "FATAL: Gemma 4 26B-A4B requires ROUTER_REPLAY_MODE=R3 for this sweep" >&2
    exit 2
  fi
  ENABLE_ROLLOUT_ROUTING_REPLAY=True
elif [ "${ROUTER_REPLAY_MODE}" = disabled ]; then
  ENABLE_ROLLOUT_ROUTING_REPLAY=False
else
  echo "FATAL: router replay is enabled only for Gemma 4 26B-A4B; got ${GEMMA4_MODEL}" >&2
  exit 2
fi
# verl defaults VLLM_DISABLE_COMPILE_CACHE=1 in its Ray runtime environment.
# That is pathological for the Gemma 4 compiled-rollout path: every worker
# recompiles instead of reusing the cache, and the four-GPU 8k canary remained
# saturated for tens of minutes.  An explicit driver value removes verl's
# default and is inherited by Ray workers.  Keep eager runs unchanged.
if [ "${ROLLOUT_ENFORCE_EAGER,,}" = "false" ]; then
  export VLLM_DISABLE_COMPILE_CACHE=0
fi
# Use `${name-default}` (not `:-`) so an explicitly empty value can force a
# max-length rollout in memory canaries. Production keeps both stop strings.
VERL_ROLLOUT_EXTRA_STOP="${VERL_ROLLOUT_EXTRA_STOP-<end_of_turn>,<start_of_turn>}"

DEFAULT_CKPTS_DIR="${RAY_DATA_HOME}/ckpts/${MODEL_TAG}-deepscaler-${DATASET_TAG}-seed${SEED}${NORMALIZED_RUN_SUFFIX}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_${MODEL_TAG}_${DATASET_TAG}_seed${SEED}_${RUN_SLOT}}"
# Ray appends a long session/socket suffix, while Linux AF_UNIX paths are
# limited to 107 bytes. Preserve uniqueness with a hash when a descriptive
# run slot makes the root too long.
if [ "${#RAY_TEMP_DIR}" -gt 40 ]; then
  RAY_TEMP_HASH="$(printf '%s' "${RAY_TEMP_DIR}" | sha256sum | cut -c1-12)"
  RAY_TEMP_DIR="/tmp/ray_${RAY_TEMP_HASH}"
fi
echo "RAY_TEMP_DIR=${RAY_TEMP_DIR}"
CKPTS_DIR="${CKPTS_DIR:-${DEFAULT_CKPTS_DIR}}"
export ROLLING_CHECKPOINT_ENABLED="${ROLLING_CHECKPOINT_ENABLED:-False}"
export ROLLING_CHECKPOINT_FREQ="${ROLLING_CHECKPOINT_FREQ:-1}"
case "${ROLLING_CHECKPOINT_ENABLED,,}" in
  1|true|yes|on)
    if [ -z "${FULL_CHECKPOINT_S3_URI:-}" ]; then
      echo "FATAL: ROLLING_CHECKPOINT_ENABLED requires FULL_CHECKPOINT_S3_URI" >&2
      exit 2
    fi
    if ! [[ "${ROLLING_CHECKPOINT_FREQ}" =~ ^[1-9][0-9]*$ ]]; then
      echo "FATAL: ROLLING_CHECKPOINT_FREQ must be a positive integer; got ${ROLLING_CHECKPOINT_FREQ}" >&2
      exit 2
    fi
    ;;
  0|false|no|off|"") ;;
  *)
    echo "FATAL: invalid ROLLING_CHECKPOINT_ENABLED=${ROLLING_CHECKPOINT_ENABLED}" >&2
    exit 2
    ;;
esac

# A borrowing-enabled relaunch can arrive after the terminal checkpoint or one
# of the two durable completion markers was uploaded. Skip only when both the
# full-checkpoint receipt and best-HF marker are valid. Otherwise restore and
# idempotently finish publication without taking an extra optimization step.
if [ -n "${FULL_CHECKPOINT_S3_URI:-}" ]; then
  : "${TOTAL_TRAINING_STEPS:?TOTAL_TRAINING_STEPS is required with FULL_CHECKPOINT_S3_URI}"
  : "${WANDB_RUN_ID:?WANDB_RUN_ID is required with FULL_CHECKPOINT_S3_URI}"
  export WANDB_RESUME="${WANDB_RESUME:-allow}"
  set +e
  "${VENV}/bin/python" rl-distill-scripts/full_checkpoint_s3.py check-completion-max \
    --s3-uri "${FULL_CHECKPOINT_S3_URI}" \
    --max-step "${TOTAL_TRAINING_STEPS}" \
    --expected-world-size "${n_gpus}" \
    --model "${GEMMA4_MODEL}" \
    --difficulty "${DIFFICULTY_DATASET}" \
    --seed "${SEED}" \
    --wandb-run-id "${WANDB_RUN_ID}" \
    --hf-repo "${HF_PUSH_REPO:-${DEFAULT_HF_PUSH_REPO}}"
  completion_rc=$?
  best_hf_rc=3
  if [ -n "${RUN_ARTIFACT_S3_URI:-}" ]; then
    "${VENV}/bin/python" rl-distill-scripts/full_checkpoint_s3.py check-best-hf \
      --s3-uri "${RUN_ARTIFACT_S3_URI}"
    best_hf_rc=$?
  fi
  set -e
  if [ "${completion_rc}" -eq 0 ] && [ "${best_hf_rc}" -eq 0 ]; then
    echo "RUN_ALREADY_COMPLETE s3=${FULL_CHECKPOINT_S3_URI} artifact=${RUN_ARTIFACT_S3_URI}"
    echo "RUN_DONE rc=0"
    exit 0
  fi
  if { [ "${completion_rc}" -ne 0 ] && [ "${completion_rc}" -ne 3 ]; } || \
     { [ "${best_hf_rc}" -ne 0 ] && [ "${best_hf_rc}" -ne 3 ]; }; then
    echo "FATAL: remote completion preflight failed completion_rc=${completion_rc} best_hf_rc=${best_hf_rc} s3=${FULL_CHECKPOINT_S3_URI}" >&2
    preflight_rc="${completion_rc}"
    if [ "${preflight_rc}" -eq 0 ]; then
      preflight_rc="${best_hf_rc}"
    fi
    exit "${preflight_rc}"
  fi
fi

# Pin model weights when the matrix launcher supplies a revision, then train and
# serve from the resolved local snapshot so a moving Hub branch cannot change a
# run.  Continuations may keep GEMMA4_MODEL set to the architecture-bearing base
# model while initializing from a pinned HF export subfolder.
export GEMMA4_INIT_MODEL_REPO="${GEMMA4_INIT_MODEL_REPO:-${GEMMA4_MODEL}}"
export GEMMA4_INIT_MODEL_REVISION="${GEMMA4_INIT_MODEL_REVISION:-${GEMMA4_MODEL_REVISION:-}}"
export GEMMA4_INIT_MODEL_SUBFOLDER="${GEMMA4_INIT_MODEL_SUBFOLDER:-}"
MODEL_LOCAL_PATH="$(${VENV}/bin/python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = os.environ["GEMMA4_INIT_MODEL_REPO"]
revision = os.environ.get("GEMMA4_INIT_MODEL_REVISION", "")
subfolder = os.environ.get("GEMMA4_INIT_MODEL_SUBFOLDER", "").strip("/")
kwargs = {"repo_id": repo_id}
if revision:
    kwargs["revision"] = revision
if subfolder:
    kwargs["allow_patterns"] = [f"{subfolder}/*"]

snapshot_root = Path(snapshot_download(**kwargs))
model_path = snapshot_root / subfolder if subfolder else snapshot_root
if not (model_path / "config.json").is_file():
    raise FileNotFoundError(f"missing config.json in resolved model path: {model_path}")
if not any(model_path.glob("*.safetensors")):
    raise FileNotFoundError(f"missing safetensors weights in resolved model path: {model_path}")
print(model_path)
PY
)"

if [ "${DIFFICULTY_DATASET_SOURCE}" = gemma4_26b_bands ]; then
  "${VENV}/bin/python" rl-distill-scripts/data/prepare_deepscaler_gemma4_26b_difficulty_rl_data.py \
    --data-dir "${DATA_DIR}" --band "${DIFFICULTY_DATASET}" --validation-repeats 16
  DEFAULT_VAL_FILES="['${DATA_DIR}/${IN_DIST_VAL_BASENAME}']"
elif [ "${DIFFICULTY_DATASET}" = strict4of4 ]; then
  DATA_DIR="${DATA_DIR}" bash rl-distill-scripts/data/prepare_deepscaler_4of4strict_rl_data.sh
  DEFAULT_VAL_FILES="['${DATA_DIR}/${IN_DIST_VAL_BASENAME}']"
else
  "${VENV}/bin/python" rl-distill-scripts/data/prepare_deepscaler_easy_medium_rl_data.py \
    --data-dir "${DATA_DIR}" --min-validation-rows "${MIN_VALIDATION_ROWS:-8000}"
  DEFAULT_VAL_FILES="['${DATA_DIR}/${IN_DIST_VAL_BASENAME}','${DATA_DIR}/math__gsm8k_test_x7.parquet','${DATA_DIR}/math__math_500_x16.parquet']"
fi

TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/${TRAIN_BASENAME}}"
VAL_FILES="${VAL_FILES:-${DEFAULT_VAL_FILES}}"

# ScaleTrain's /tmp is an emptyDir and disappears with a preempted pod.  When a
# remote prefix is configured, recover the newest checkpoint whose completion
# manifest was uploaded last.  That checkpoint includes Adam, scheduler/RNG
# extras, and the StatefulDataLoader cursor; the ordinary HF export does not.
if [ -n "${FULL_CHECKPOINT_S3_URI:-}" ]; then
  "${VENV}/bin/python" rl-distill-scripts/full_checkpoint_s3.py restore-latest \
    --checkpoint-root "${CKPTS_DIR}" \
    --s3-uri "${FULL_CHECKPOINT_S3_URI}"
  if [ -f "${CKPTS_DIR}/latest_checkpointed_iteration.txt" ]; then
    export RESUME_MODE=auto
    export DATALOADER_SKIP_BATCHES=0
  fi
fi

status=0
MODEL_TAG="${MODEL_TAG}" MODEL_REPO="${GEMMA4_MODEL}" MODEL_PATH="${MODEL_LOCAL_PATH}" \
WRAP_LAYER_CLS="${WRAP_LAYER_CLS}" \
HF_PUSH_REPO="${HF_PUSH_REPO:-${DEFAULT_HF_PUSH_REPO}}" \
EXP_NAME="${EXP_NAME:-${DEFAULT_EXP_NAME}}" \
DATA_SEED="${SEED}" \
TRAIN_FILE="${TRAIN_FILE}" \
VAL_FILES="${VAL_FILES}" \
GEMMA3_CHAT_TEMPLATE_FILE="${GEMMA3_CHAT_TEMPLATE_FILE:-${PWD}/rl-distill-scripts/data/gemma3_it_fewshot_math.jinja}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-4096}" \
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-100}" LOG_TRAIN_GENERATIONS="${LOG_TRAIN_GENERATIONS:-100}" \
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}" \
N_GPUS_PER_NODE="${n_gpus}" NNODES=1 OFFLOAD="${OFFLOAD:-False}" SAVE_FREQ="${SAVE_FREQ:-25}" \
RAY_ADDRESS=local VERL_VLLM_PORT_BASE="${VERL_VLLM_PORT_BASE:-52000}" \
DATA_DIR="${DATA_DIR}" CKPTS_DIR="${CKPTS_DIR}" \
    bash rl-distill-scripts/gemma3_pt_fewshot_math_rl.sh \
      +ray_kwargs.ray_init._temp_dir="${RAY_TEMP_DIR}" \
      +ray_kwargs.ray_init.include_dashboard=False \
      +ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME="${HF_HOME}" \
      +ray_kwargs.ray_init.runtime_env.env_vars.CUDA_HOME="${CUDA_HOME:-}" \
      +ray_kwargs.ray_init.runtime_env.env_vars.PATH="${PATH}" \
      +ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
      +ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF="'${PYTORCH_CUDA_ALLOC_CONF:-}'" \
      +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ATTENTION_BACKEND="TRITON_ATTN" \
      +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_FLASHINFER_SAMPLER="'0'" \
      +ray_kwargs.ray_init.runtime_env.env_vars.VERL_FSDP2_LOCAL_LOAD="'1'" \
      +ray_kwargs.ray_init.runtime_env.env_vars.VERL_SKIP_VLLM_MM_WEIGHT_RELOAD="'${VERL_SKIP_VLLM_MM_WEIGHT_RELOAD}'" \
      "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_ROLLOUT_EXTRA_STOP='${VERL_ROLLOUT_EXTRA_STOP}'" \
      actor_rollout_ref.model.use_remove_padding=False \
      +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
      ++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${UPDATE_WEIGHTS_BUCKET_MB:-8192}" \
      actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
      ++actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes="${VLLM_KV_CACHE_MEMORY_BYTES}" \
      actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
      `# NOTE: the new engine path reads these from actor.fsdp_config (ActorConfig.__post_init__ sets` \
      `# engine = fsdp_config); the bare actor.* twins are legacy dp_actor fields — set BOTH.` \
      ++actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True \
      ++actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
      ++actor_rollout_ref.actor.fsdp_config.offload_policy="${FSDP_CPU_OFFLOAD_POLICY}" \
      ++actor_rollout_ref.actor.fsdp_config.max_padded_tokens_per_microbatch="${MAX_PADDED_TOKENS_PER_MICROBATCH}" \
      ++actor_rollout_ref.actor.fsdp_config.infer_max_padded_tokens_per_microbatch="${MAX_PADDED_TOKENS_PER_MICROBATCH}" \
      ++actor_rollout_ref.ref.fsdp_config.infer_max_padded_tokens_per_microbatch="${MAX_PADDED_TOKENS_PER_MICROBATCH}" \
      actor_rollout_ref.actor.router_replay.mode="${ROUTER_REPLAY_MODE}" \
      actor_rollout_ref.actor.fsdp_config.router_replay.mode="${ROUTER_REPLAY_MODE}" \
      actor_rollout_ref.rollout.enable_rollout_routing_replay="${ENABLE_ROLLOUT_ROUTING_REPLAY}" \
      ++actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
      `# gemma-4 has no flash-attn CE fast path; naive entropy does a monolithic logits.float() over` \
      `# 24576 tokens x 262k vocab (~26GB) -> OOM at train step 1. Chunked entropy + checkpointing fix it.` \
      ++actor_rollout_ref.actor.entropy_checkpointing=True \
      `# padded-forward logits are bsz x padded_len x 262k-vocab; dynamic packing by REAL tokens lets` \
      `# padding inflate that to ~26GB bf16 (OOM). Without rmpad (unsafe for gemma-4: no varlen attn,` \
      `# no monkey patch), cap the forward at ONE sequence: logits <= ~13GB, predictable memory.` \
      actor_rollout_ref.actor.use_dynamic_bsz=False \
      actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
      actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
      `# vLLM sleep/wake (cumem) is fragile on this stack (wake_up OOM after Adam states materialize;` \
      `# also cumem invalid-argument on E4B). Disable sleep entirely: engine stays resident at a small` \
      `# util (0.3 via env) and weight-sync loads into the live engine. Trainer gets ~54GB vs ~40 peak.` \
      actor_rollout_ref.rollout.free_cache_engine=False \
      ++actor_rollout_ref.rollout.enable_sleep_mode=False \
      ++trainer.early_stopping.enabled="${EARLY_STOPPING_ENABLED:-False}" \
      ++trainer.early_stopping.metric="${EARLY_STOPPING_METRIC:-val-core/math/acc/mean@16}" \
      ++trainer.early_stopping.mode="${EARLY_STOPPING_MODE:-max}" \
      ++trainer.early_stopping.patience="${EARLY_STOPPING_PATIENCE:-5}" \
      ++trainer.early_stopping.min_delta="${EARLY_STOPPING_MIN_DELTA:-0.0}" \
      ++trainer.early_stopping.include_initial_validation="${EARLY_STOPPING_INCLUDE_INITIAL_VALIDATION:-True}" \
      `# Forward ordinary experiment overrides first. The rollout-correction settings below are` \
      `# deliberately last so every Gemma 4 run uses the same always-on token TIS contract.` \
      "$@" \
      actor_rollout_ref.rollout.calculate_log_probs=True \
      algorithm.rollout_correction.rollout_is=token \
      algorithm.rollout_correction.rollout_is_threshold=2.0 \
      algorithm.rollout_correction.rollout_is_batch_normalize=False \
      algorithm.rollout_correction.rollout_rs=null \
      algorithm.rollout_correction.rollout_rs_threshold=null \
      algorithm.rollout_correction.bypass_mode=False || status=$?
# Compiled rollout mode requires a writable, reusable cache per child. The
# matrix wrappers isolate VLLM/TRITON/TorchInductor cache roots so concurrent
# packed E2B children cannot corrupt one another's artifacts.
# Forensics on failure: the silent native worker deaths never reach the pod stdout — dump the Ray
# worker stderr files + kernel OOM evidence so the next failure is diagnosable from Datadog/kubectl.
if [ "${status}" -ne 0 ]; then
  echo "===== FORENSICS: ray worker stderr tails ====="
  for f in "${RAY_TEMP_DIR}"/session_*/logs/worker-*.err; do
    [ -s "$f" ] || continue
    if grep -qiE "error|fatal|abort|segfault|SIG|out of memory|Traceback" "$f"; then
      echo "--- $f ---"; tail -40 "$f"
    fi
  done
  echo "===== FORENSICS: raylet + dmesg ====="
  tail -30 "${RAY_TEMP_DIR}"/session_*/logs/raylet.err 2>/dev/null || true
  dmesg 2>/dev/null | tail -30 || true
fi

if [ "${status}" -eq 0 ] && [ -n "${FULL_CHECKPOINT_S3_URI:-}" ]; then
  FINAL_CHECKPOINT_STEP="$("${VENV}/bin/python" -c \
    'import json, sys; print(int(json.load(open(sys.argv[1]))["final_step"]))' \
    "${CKPTS_DIR}/run_outcome.json")" || status=$?
fi
if [ "${status}" -eq 0 ] && [ -n "${RUN_ARTIFACT_S3_URI:-}" ]; then
  "${VENV}/bin/python" rl-distill-scripts/full_checkpoint_s3.py publish-best-hf \
    --checkpoint-root "${CKPTS_DIR}" \
    --s3-uri "${RUN_ARTIFACT_S3_URI}" || status=$?
fi
if [ "${status}" -eq 0 ] && [ -n "${FULL_CHECKPOINT_S3_URI:-}" ]; then
  "${VENV}/bin/python" rl-distill-scripts/full_checkpoint_s3.py complete-run \
    --s3-uri "${FULL_CHECKPOINT_S3_URI}" \
    --expected-step "${FINAL_CHECKPOINT_STEP}" \
    --expected-world-size "${n_gpus}" \
    --model "${GEMMA4_MODEL}" \
    --difficulty "${DIFFICULTY_DATASET}" \
    --seed "${SEED}" \
    --wandb-run-id "${WANDB_RUN_ID}" \
    --hf-repo "${HF_PUSH_REPO:-${DEFAULT_HF_PUSH_REPO}}" \
    --receipt-path "${CKPTS_DIR}/run_complete.json" || status=$?
fi
# The wandb uploader can die mid-run ("Fatal error while uploading data") while metrics keep
# writing to the local .wandb transaction log; a final `wandb sync` backfills the server copy.
# (Observed 2026-07-27 on the E4B 8k run — pod-log parsing recovered it, but this is cheaper.)
echo "===== final wandb sync (backfill any uploader failures) ====="
"${VENV}/bin/wandb" sync --sync-all 2>&1 | tail -5 || true

echo "RUN_DONE rc=${status}"
exit "${status}"
