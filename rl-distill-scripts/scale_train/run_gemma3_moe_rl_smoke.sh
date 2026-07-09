#!/usr/bin/env bash
# ScaleTrain entrypoint: Gemma3 4B PT MoE Megatron+vLLM RL smoke test.
#
# Runs inside the train-rl-distill-megatron image (see Dockerfile.megatron),
# which bakes .venv-megatron and the Megatron-Bridge fork. Prepares the
# DAPO-17k data in-container, then runs the same local smoke wrapper used for
# workstation validation on all GPUs allocated to the pod.
#
# Launch (4 GPUs, 4 experts, TP=2/EP=4):
#   python rl-distill-scripts/scale_train/launch_st_job.py \
#     --cluster eks --n-instances 1 --gpus-per-instance 4 \
#     --priority high --allow-borrowing \
#     --build-config-key train-rl-distill-megatron \
#     --job-name gemma3-moe-4e-smoke \
#     --run-file run_gemma3_moe_rl_smoke.sh \
#     --env-vars "NUM_EXPERTS=4"
set -euxo pipefail

cd /workspace/rl-distill

# The container entrypoint runs `sudo -E ... bash <this>`, and sudo's
# secure_path drops the venv from PATH (the CUDA base image has no system
# python3). Put the Megatron venv on PATH up front so the data-prep step and
# everything before the smoke wrapper's own `activate` can find python3.
MEGATRON_VENV="${MEGATRON_VENV:-/workspace/rl-distill/.venv-megatron}"
export PATH="${MEGATRON_VENV}/bin:${PATH}"
# shellcheck disable=SC1091
source "${MEGATRON_VENV}/bin/activate"

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export RAY_DATA_HOME="${RAY_DATA_HOME:-/tmp/verl}"
export DATA_DIR="${RAY_DATA_HOME}/data"
mkdir -p "${DATA_DIR}"

# Prepare the DAPO-17k train/test parquets used by the smoke.
if [ ! -f "${DATA_DIR}/dapo_17k_train.parquet" ]; then
    DATA_DIR="${DATA_DIR}" bash rl-distill-scripts/data/prepare_dapo_17k_split.sh
fi

n_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
smoke_gpus="$(seq -s, 0 $((n_gpus - 1)))"

NUM_EXPERTS="${NUM_EXPERTS:-4}"
if [ "${NUM_EXPERTS}" = "4" ] && [ "${n_gpus}" -ge 4 ]; then
    export ACTOR_TP="${ACTOR_TP:-2}"
    export REF_TP="${REF_TP:-2}"
fi

NUM_EXPERTS="${NUM_EXPERTS}" \
SMOKE_GPUS="${smoke_gpus}" \
SMOKE_STEPS="${SMOKE_STEPS:-2}" \
HF_MOE_LOCAL_DIR="${HF_MOE_LOCAL_DIR:-}" \
TRAIN_FILE="${DATA_DIR}/dapo_17k_train.parquet" \
VAL_FILE="${DATA_DIR}/dapo_17k_test.parquet" \
CKPTS_DIR="${RAY_DATA_HOME}/ckpts/moe-${NUM_EXPERTS}e-smoke" \
    bash rl-distill-scripts/gemma3_4b_pt_moe_megatron_rl_local_smoke.sh "$@"
