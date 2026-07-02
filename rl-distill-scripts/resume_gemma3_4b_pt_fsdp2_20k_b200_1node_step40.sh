#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export EXP_NAME=${EXP_NAME:-"DAPO-Gemma3-4B-PT-B200-1Node-DAPO17k-20kResp-20260701-1802"}
export RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
export CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/DAPO/${EXP_NAME}"}
export RESUME_STEP=${RESUME_STEP:-40}
export RESUME_CKPT_PATH=${RESUME_CKPT_PATH:-"${CKPTS_DIR}/global_step_${RESUME_STEP}"}

# W&B server-side rewind is not enabled for this project, so use a fresh run id
# and rely on verl's explicit step logging; the first logged metrics are step 40.
unset WANDB_RESUME
unset WANDB_FORK_FROM
unset WANDB_RESUME_FROM
export WANDB_RUN_ID=${WANDB_RUN_ID:-"g4b40$(date +%H%M%S)"}

# Keep the original 4-GPU setup and normal checkpoint contents.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export ACTOR_CKPT_SAVE_CONTENTS=${ACTOR_CKPT_SAVE_CONTENTS:-"[model,optimizer,extra,hf_model]"}
export SAVE_FREQ=${SAVE_FREQ:-20}
export HF_PUSH_ENABLE=${HF_PUSH_ENABLE:-True}

exec "${PROJECT_ROOT}/rl-distill-scripts/gemma3_4b_pt_fsdp2_20k_b200_1node.sh" \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${RESUME_CKPT_PATH}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_RUN_ID="${WANDB_RUN_ID}" \
    "$@"
