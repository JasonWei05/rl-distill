#!/usr/bin/env bash
# ScaleTrain entrypoint: Gemma 4 PT DeepScaleR **strict-4/4** DAPO RL (single node, full node's GPUs).
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
# Which model via GEMMA4_MODEL env (google/gemma-4-E2B | google/gemma-4-E4B). Per-seed via DATA_SEED.
#
# Launch (E2B, one node, borrowing OFF):
#   python rl-distill-scripts/scale_train/launch_st_job.py --cluster eks --n-instances 1 \
#     --gpus-per-instance 8 --priority high --job-name gemma4-e2b-ds4of4s \
#     --build-config-key train-rl-distill --team egp --product train.enterprise_rlvr \
#     --run-file run_gemma4_pt_deepscaler_4of4strict_rl.sh \
#     --env-vars GEMMA4_MODEL=google/gemma-4-E2B,DATA_SEED=42
#   # E4B: --job-name gemma4-e4b-ds4of4s --env-vars GEMMA4_MODEL=google/gemma-4-E4B,DATA_SEED=42
#   # NOTE: --run-file is resolved relative to scale_train/ (bare filename), NOT the repo root — a
#   #       repo-root-relative path doubles to .../scale_train/rl-distill-scripts/scale_train/... and 404s.
set -euxo pipefail
cd /workspace/rl-distill

GEMMA4_MODEL="${GEMMA4_MODEL:-google/gemma-4-E2B}"
case "${GEMMA4_MODEL}" in
  *E2B*) MODEL_TAG=gemma4-e2b ;;
  *E4B*) MODEL_TAG=gemma4-e4b ;;
  *)     MODEL_TAG="$(echo "${GEMMA4_MODEL}" | tr '/A-Z' '-a-z')" ;;
esac

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

export RAY_DATA_HOME="${RAY_DATA_HOME:-/tmp/verl}"; export DATA_DIR="${RAY_DATA_HOME}/data"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"; export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb}"; export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
# ScaleTrain pods on ml-gpu-batch are IPv6-only (pod IP 2602:fb33:...), interface eth0.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET6}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"

n_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((n_gpus - 1)))"
mkdir -p "${DATA_DIR}"
SEED="${DATA_SEED:-42}"

# model (gated; HF_TOKEN from env) + strict-4/4 data (from HF, idempotent)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${GEMMA4_MODEL}')"
DATA_DIR="${DATA_DIR}" bash rl-distill-scripts/data/prepare_deepscaler_4of4strict_rl_data.sh

status=0
MODEL_TAG="${MODEL_TAG}" MODEL_REPO="${GEMMA4_MODEL}" MODEL_PATH="${GEMMA4_MODEL}" \
WRAP_LAYER_CLS="Gemma4TextDecoderLayer" \
HF_PUSH_REPO="${HF_PUSH_REPO:-JWei05/DAPO-${MODEL_TAG}-PT-DeepScaleR-4of4strict-seed${SEED}}" \
EXP_NAME="${EXP_NAME:-DAPO-${MODEL_TAG}-pt-DeepScaleR-4of4strict-seed${SEED}}" \
DATA_SEED="${SEED}" \
TRAIN_FILE="${DATA_DIR}/deepscaler_4of4strict_rl_train.parquet" \
VAL_FILES="['${DATA_DIR}/deepscaler_4of4strict_rl_val200_x16.parquet']" \
GEMMA3_CHAT_TEMPLATE_FILE="${GEMMA3_CHAT_TEMPLATE_FILE:-${PWD}/rl-distill-scripts/data/gemma3_it_fewshot_math.jinja}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-20480}" OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-4096}" \
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-100}" LOG_TRAIN_GENERATIONS="${LOG_TRAIN_GENERATIONS:-100}" \
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}" \
N_GPUS_PER_NODE="${n_gpus}" NNODES=1 OFFLOAD="${OFFLOAD:-False}" SAVE_FREQ="${SAVE_FREQ:-25}" \
RAY_ADDRESS=local VERL_VLLM_PORT_BASE="${VERL_VLLM_PORT_BASE:-52000}" \
DATA_DIR="${DATA_DIR}" CKPTS_DIR="${RAY_DATA_HOME}/ckpts/${MODEL_TAG}-deepscaler-4of4strict-seed${SEED}" \
    bash rl-distill-scripts/gemma3_pt_fewshot_math_rl.sh \
      +ray_kwargs.ray_init._temp_dir="/tmp/ray_${MODEL_TAG}_ds4of4strict_seed${SEED}" \
      +ray_kwargs.ray_init.include_dashboard=False \
      +ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME="${HF_HOME}" \
      +ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
      +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ATTENTION_BACKEND="TRITON_ATTN" \
      +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_FLASHINFER_SAMPLER="'0'" \
      +ray_kwargs.ray_init.runtime_env.env_vars.VERL_FSDP2_LOCAL_LOAD="'1'" \
      "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_ROLLOUT_EXTRA_STOP='<end_of_turn>,<start_of_turn>'" \
      actor_rollout_ref.model.use_remove_padding=False \
      +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
      ++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${UPDATE_WEIGHTS_BUCKET_MB:-8192}" \
      actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER:-True}" \
      `# NOTE: the new engine path reads these from actor.fsdp_config (ActorConfig.__post_init__ sets` \
      `# engine = fsdp_config); the bare actor.* twins are legacy dp_actor fields — set BOTH.` \
      ++actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True \
      ++actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
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
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
      `# vLLM sleep/wake (cumem) is fragile on this stack (wake_up OOM after Adam states materialize;` \
      `# also cumem invalid-argument on E4B). Disable sleep entirely: engine stays resident at a small` \
      `# util (0.3 via env) and weight-sync loads into the live engine. Trainer gets ~54GB vs ~40 peak.` \
      actor_rollout_ref.rollout.free_cache_engine=False \
      ++actor_rollout_ref.rollout.enable_sleep_mode=False \
      "$@" || status=$?
# enforce_eager=True (above): p5 8-rank init died natively in the vLLM engine-build phase while
# single-rank eager/compile/sleep all pass (diag job) — concurrent per-rank torch.compile/CUDA-graph
# JIT storms are the remaining suspect class. Eager skips them (slower rollouts; revisit once stable).
# Forensics on failure: the silent native worker deaths never reach the pod stdout — dump the Ray
# worker stderr files + kernel OOM evidence so the next failure is diagnosable from Datadog/kubectl.
if [ "${status}" -ne 0 ]; then
  echo "===== FORENSICS: ray worker stderr tails ====="
  for f in /tmp/ray_${MODEL_TAG}_ds4of4strict_seed${SEED}/session_*/logs/worker-*.err; do
    [ -s "$f" ] || continue
    if grep -qiE "error|fatal|abort|segfault|SIG|out of memory|Traceback" "$f"; then
      echo "--- $f ---"; tail -40 "$f"
    fi
  done
  echo "===== FORENSICS: raylet + dmesg ====="
  tail -30 /tmp/ray_${MODEL_TAG}_ds4of4strict_seed${SEED}/session_*/logs/raylet.err 2>/dev/null || true
  dmesg 2>/dev/null | tail -30 || true
fi
# The wandb uploader can die mid-run ("Fatal error while uploading data") while metrics keep
# writing to the local .wandb transaction log; a final `wandb sync` backfills the server copy.
# (Observed 2026-07-27 on the E4B 8k run — pod-log parsing recovered it, but this is cheaper.)
echo "===== final wandb sync (backfill any uploader failures) ====="
"${VENV}/bin/wandb" sync --sync-all 2>&1 | tail -5 || true

echo "RUN_DONE rc=${status}"
exit "${status}"
