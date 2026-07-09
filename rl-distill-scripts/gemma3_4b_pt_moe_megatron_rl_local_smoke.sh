#!/usr/bin/env bash
# Local single-node smoke test for the Gemma3 4B PT MoE Megatron+vLLM RL launcher.
#
# Runs a 2-step DAPO round trip (vLLM rollout -> Megatron log-probs -> actor
# update -> weight resync -> second rollout) with tiny batches on a subset of
# local GPUs, leaving every other GPU untouched.
#
#   NUM_EXPERTS=2 SMOKE_GPUS=4,6 bash gemma3_4b_pt_moe_megatron_rl_local_smoke.sh
#   NUM_EXPERTS=4 SMOKE_GPUS=4,6 bash gemma3_4b_pt_moe_megatron_rl_local_smoke.sh
#
# Extra hydra overrides may be appended.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NUM_EXPERTS=${NUM_EXPERTS:-2}
SMOKE_GPUS=${SMOKE_GPUS:-4,6}
SMOKE_STEPS=${SMOKE_STEPS:-2}

# Refuse to run on GPUs that other jobs are using.
IFS=',' read -ra gpu_list <<<"${SMOKE_GPUS}"
for gpu in "${gpu_list[@]}"; do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}")"
    if [ "${used_mib}" -gt 1024 ]; then
        echo "GPU ${gpu} is busy (${used_mib} MiB in use); pick free GPUs via SMOKE_GPUS" >&2
        exit 2
    fi
done
n_gpus=${#gpu_list[@]}

export CUDA_VISIBLE_DEVICES="${SMOKE_GPUS}"
export NUM_EXPERTS
export RAY_ADDRESS=local
export NNODES=1
export GPUS_PER_NODE=${n_gpus}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}

# Parallelism sized for the smoke world; EP stays <= world size.
if [ "${n_gpus}" -ge 4 ] && [ "${NUM_EXPERTS}" = "4" ]; then
    default_ep=4
else
    default_ep=2
fi
export ACTOR_TP=${ACTOR_TP:-1}
export ACTOR_EP=${ACTOR_EP:-${default_ep}}
export REF_TP=${REF_TP:-${ACTOR_TP}}
export REF_EP=${REF_EP:-${ACTOR_EP}}

# Pre-downloaded pinned snapshots (matching the 2e/4e wrapper revisions).
if [ -z "${HF_MOE_LOCAL_DIR:-}" ]; then
    case "${NUM_EXPERTS}" in
    2) HF_MOE_LOCAL_DIR="/tmp/hf-gemma3-moe-rl-cache/models--JWei05--gemma3-4b-pt-moe-2e-top1-sft-16k/snapshots/952a11b802b63ef091f20ec2dfe08eb66376794c" ;;
    4) HF_MOE_LOCAL_DIR="/tmp/hf-gemma3-moe-rl-cache/models--JWei05--gemma3-4b-pt-moe-4e-top1-sft-16k/snapshots/cd87f6e541b1bc0fba8caef218c55601fbb0c533" ;;
    esac
fi
if [ -d "${HF_MOE_LOCAL_DIR}" ]; then
    export HF_MOE_LOCAL_DIR
else
    # Fall back to the launcher's snapshot download, pinned to the same
    # revisions the 2e/4e wrappers use.
    unset HF_MOE_LOCAL_DIR
    case "${NUM_EXPERTS}" in
    2) export HF_MOE_REVISION="${HF_MOE_REVISION:-952a11b802b63ef091f20ec2dfe08eb66376794c}" ;;
    4) export HF_MOE_REVISION="${HF_MOE_REVISION:-cd87f6e541b1bc0fba8caef218c55601fbb0c533}" ;;
    esac
fi

export TRAIN_FILE=${TRAIN_FILE:-"${HOME}/verl/data/dapo_17k_train.parquet"}
export VAL_FILE=${VAL_FILE:-"${HOME}/verl/data/dapo_17k_test.parquet"}

# Tiny round trip: 4 prompts x 4 responses, one mini-batch, short generations.
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-4}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-4}
export TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-4}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
export ENABLE_OVERLONG_BUFFER=${ENABLE_OVERLONG_BUFFER:-False}
# The DAPO reward manager asserts max_resp_len > overlong_buffer.len even
# when the buffer is disabled.
export OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-128}
export TEST_FREQ=${TEST_FREQ:--1}
export SAVE_FREQ=${SAVE_FREQ:--1}
export RESUME_MODE=${RESUME_MODE:-disable}
export ROUTER_REPLAY_MODE=${ROUTER_REPLAY_MODE:-disabled}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}
export REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-2}
export AGENT_LOOP_NUM_WORKERS=${AGENT_LOOP_NUM_WORKERS:-2}
export CKPTS_DIR=${CKPTS_DIR:-"/tmp/verl/ckpts/moe-${NUM_EXPERTS}e-smoke"}
export EXP_NAME=${EXP_NAME:-"moe-${NUM_EXPERTS}e-smoke-$(date +%Y%m%d-%H%M)"}

exec bash "${SCRIPT_DIR}/gemma3_4b_pt_moe_megatron_rl_20k.sh" \
    trainer.total_training_steps="${SMOKE_STEPS}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=1 \
    'trainer.logger=["console"]' \
    +ray_kwargs.ray_init._temp_dir=/tmp/ray_megatron_moe_smoke \
    +ray_kwargs.ray_init.include_dashboard=False \
    "$@"
