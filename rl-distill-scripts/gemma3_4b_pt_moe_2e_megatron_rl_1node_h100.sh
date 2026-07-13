#!/usr/bin/env bash
set -euo pipefail

# Gemma 3 4B PT -> 2E top-1 MoE DAPO RL on one 8xH100 node.
#
# This wrapper intentionally requires a freshly upcycled local checkpoint. It
# uses Megatron TP=4/EP=2 (all eight GPUs), native vLLM rollout with Triton
# attention, R2 router replay, and the router auxiliary load-balancing loss.
# See GEMMA3_MOE_RL_TRAINING.md for conversion and mandatory correctness gates.
#
# Usage:
#   bash rl-distill-scripts/gemma3_4b_pt_moe_2e_megatron_rl_1node_h100.sh
# Required model input: UPCYCLED_MOE_DIR, HF_MOE_LOCAL_DIR, or MODEL_PATH.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Fresh dense upcycle (normal sparse checkpoint, never canonical) --------
export NUM_EXPERTS=2
_upcycled="${UPCYCLED_MOE_DIR:-/tmp/gemma3-4b-pt-moe-2e-upcycled}"
if [ -z "${MODEL_PATH:-}" ] && [ -z "${HF_MOE_LOCAL_DIR:-}" ]; then
    if [ ! -f "${_upcycled}/config.json" ]; then
        echo "Missing fresh upcycle at ${_upcycled}." >&2
        echo "Set UPCYCLED_MOE_DIR/HF_MOE_LOCAL_DIR, then follow GEMMA3_MOE_RL_TRAINING.md." >&2
        exit 2
    fi
    export HF_MOE_LOCAL_DIR="${_upcycled}"
fi

# ---- Topology: one node, 8 H100 (DP=2 via TP4/PP1/CP1/EP2) -------------------
export NNODES="${NNODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# TP=4 is the validated 20k-response topology on 80 GB H100s. World=8 gives
# DP=2; EP=2 shards the experts across that data-parallel dimension.
export ACTOR_TP="${ACTOR_TP:-4}"
export ACTOR_PP="${ACTOR_PP:-1}"
export ACTOR_CP="${ACTOR_CP:-1}"
export ACTOR_EP="${ACTOR_EP:-2}"
export REF_TP="${REF_TP:-4}"
export REF_PP="${REF_PP:-1}"
export REF_CP="${REF_CP:-1}"
export REF_EP="${REF_EP:-2}"
export OFFLOAD="${OFFLOAD:-True}"

# ---- Single-node isolated Ray + loopback NCCL (not the 2-node cluster path) --
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"

# ---- MoE and rollout ---------------------------------------------------------
export ROUTER_REPLAY_MODE="${ROUTER_REPLAY_MODE:-R2}"
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-1e-3}"
export ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-True}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
export ROLLOUT_MODEL_IMPL="${ROLLOUT_MODEL_IMPL:-native}"
export ROLLOUT_ATTENTION_BACKEND="${ROLLOUT_ATTENTION_BACKEND:-TRITON_ATTN}"

# ---- Training HPs carried from the FSDP2 B200 1-node reference ---------------
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-$((1024 * 2))}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$((1024 * 20))}"
export ENABLE_OVERLONG_BUFFER="${ENABLE_OVERLONG_BUFFER:-True}"
export OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-$((1024 * 4))}"
export OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-1.0}"
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-64}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-16}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-32}"
export SAVE_FREQ="${SAVE_FREQ:-25}"
export HF_PUSH_FREQ="${HF_PUSH_FREQ:-25}"
export TEST_FREQ="${TEST_FREQ:-5}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"

# ---- Data: the DAPO-17k split prepared by data/prepare_dapo_17k_split.sh -----
export TRAIN_FILE="${TRAIN_FILE:-${HOME}/verl/data/dapo_17k_train.parquet}"
export VAL_FILE="${VAL_FILE:-${HOME}/verl/data/dapo_17k_test.parquet}"
for data_file in "${TRAIN_FILE}" "${VAL_FILE}"; do
    if [ ! -f "${data_file}" ]; then
        echo "Missing dataset ${data_file}; run data/prepare_dapo_17k_split.sh or override TRAIN_FILE/VAL_FILE." >&2
        exit 2
    fi
done

# ---- HF push: off by default (opt in explicitly) ----------------------------
export HF_PUSH_ENABLE="${HF_PUSH_ENABLE:-False}"

export EXP_NAME="${EXP_NAME:-DAPO-Gemma3-4B-PT-MoE-2E-Megatron-RL-1node-H100-$(date +%Y%m%d-%H%M)}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_megatron_moe_2e_1node}"

total_training_steps_args=()
if [ -n "${TOTAL_TRAINING_STEPS:-}" ]; then
    if ! [[ "${TOTAL_TRAINING_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "TOTAL_TRAINING_STEPS must be a positive integer; got ${TOTAL_TRAINING_STEPS}" >&2
        exit 2
    fi
    total_training_steps_args+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi

exec bash "${SCRIPT_DIR}/gemma3_4b_pt_moe_megatron_rl_20k.sh" \
    +ray_kwargs.ray_init._temp_dir="${RAY_TEMP_DIR}" \
    +ray_kwargs.ray_init.include_dashboard=False \
    "${total_training_steps_args[@]}" \
    "$@"
