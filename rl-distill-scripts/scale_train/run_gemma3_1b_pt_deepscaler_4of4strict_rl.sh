#!/usr/bin/env bash
# ScaleTrain entrypoint: Gemma 3 1B PT DeepScaleR **strict-4/4** DAPO RL (single node, 2 GPUs).
# Same recipe as the local sweep (GRPO n=16, 12-shot prompt, 8192 response + 2048 overlong buffer,
# temp 1.0 val=train, SAVE_FREQ=25 + HF push /25, wandb, val_before_train), on the strict-graded 4/4
# split (JWei05/DeepScaleR-4of4-strict-RL). Strict boxed-only grader is the math_verify default (baked
# into the image from the working tree). Uploads 100 random TRAIN traces/step + 100 VAL traces/eval to
# wandb (LOG_{TRAIN,VAL}_GENERATIONS). Per-seed via DATA_SEED (default 42) -> EXP_NAME/HF_PUSH_REPO/ckpt.
#
# Launch (seed 42, borrowing OFF):
#   python rl-distill-scripts/scale_train/launch_st_job.py --cluster eks --n-instances 1 \
#     --gpus-per-instance 2 --priority high --job-name gemma3-1b-ds4of4s-s42 \
#     --build-config-key train-rl-distill --team egp --product train.enterprise_rlvr \
#     --run-file rl-distill-scripts/scale_train/run_gemma3_1b_pt_deepscaler_4of4strict_rl.sh \
#     --env-vars DATA_SEED=42
# seed 43, borrowing ON: add --allow-borrowing and --env-vars DATA_SEED=43 (+ job-name ...-s43).
set -euxo pipefail
cd /workspace/rl-distill

VENV="${VENV:-/workspace/rl-distill/.venv}"           # FSDP2 stack (torch 2.9 / vllm 0.15.1)
export PATH="${VENV}/bin:${PATH}"; source "${VENV}/bin/activate"
if [ -f .env ]; then set -a; source .env; set +a; fi

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

# model (gated; HF_TOKEN from .env) + strict-4/4 data (from HF, idempotent)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-3-1b-pt')"
DATA_DIR="${DATA_DIR}" bash rl-distill-scripts/data/prepare_deepscaler_4of4strict_rl_data.sh

status=0
MODEL_TAG=gemma3-1b-pt MODEL_REPO=google/gemma-3-1b-pt MODEL_PATH=google/gemma-3-1b-pt \
HF_PUSH_REPO="${HF_PUSH_REPO:-JWei05/DAPO-Gemma3-1B-PT-DeepScaleR-4of4strict-seed${SEED}}" \
EXP_NAME="${EXP_NAME:-DAPO-gemma3-1b-pt-DeepScaleR-4of4strict-seed${SEED}}" \
DATA_SEED="${SEED}" \
TRAIN_FILE="${DATA_DIR}/deepscaler_4of4strict_rl_train.parquet" \
VAL_FILES="['${DATA_DIR}/deepscaler_4of4strict_rl_val200_x16.parquet']" \
GEMMA3_CHAT_TEMPLATE_FILE="${GEMMA3_CHAT_TEMPLATE_FILE:-${PWD}/rl-distill-scripts/data/gemma3_it_fewshot_math.jinja}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}" OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-2048}" \
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-100}" LOG_TRAIN_GENERATIONS="${LOG_TRAIN_GENERATIONS:-100}" \
N_GPUS_PER_NODE="${n_gpus}" NNODES=1 OFFLOAD="${OFFLOAD:-False}" SAVE_FREQ="${SAVE_FREQ:-25}" \
RAY_ADDRESS=local VERL_VLLM_PORT_BASE="${VERL_VLLM_PORT_BASE:-52000}" \
DATA_DIR="${DATA_DIR}" CKPTS_DIR="${RAY_DATA_HOME}/ckpts/gemma3-1b-pt-deepscaler-4of4strict-seed${SEED}" \
    bash rl-distill-scripts/gemma3_pt_fewshot_math_rl.sh \
      +ray_kwargs.ray_init._temp_dir="/tmp/ray_1b_ds4of4strict_seed${SEED}" \
      +ray_kwargs.ray_init.include_dashboard=False \
      "$@" || status=$?
echo "RUN_DONE rc=${status}"
exit "${status}"
