#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH=${MODEL_PATH:-"/tmp/hf_models/dapo-gemma3-27b-pt-step_000040/step_000040"}
export EXP_NAME=${EXP_NAME:-"DAPO-Gemma3-27B-PT-FSDP2-from-step40-seed43-$(date +%Y%m%d-%H%M)"}
export HF_PUSH_REPO=${HF_PUSH_REPO:-"JWei05/dapo-gemma3-27b-pt-from-step40-seed43"}
export DATA_SEED=${DATA_SEED:-43}

bash rl-distill-scripts/gemma3_27b_pt_fsdp2_20k_cpu_driver.sh "$@"
