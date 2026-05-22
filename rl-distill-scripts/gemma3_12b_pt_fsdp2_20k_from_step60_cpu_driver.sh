#!/usr/bin/env bash
set -euo pipefail

export MODEL_PATH=${MODEL_PATH:-"/tmp/hf_models/dapo-gemma3-12b-pt-step_000060/step_000060"}
export EXP_NAME=${EXP_NAME:-"DAPO-Gemma3-12B-PT-FSDP2-from-step60-seed43-$(date +%Y%m%d-%H%M)"}
export HF_PUSH_REPO=${HF_PUSH_REPO:-"JWei05/dapo-gemma3-12b-pt-from-step60-seed43"}
export DATA_SEED=${DATA_SEED:-43}

bash rl-distill-scripts/gemma3_12b_pt_fsdp2_20k_cpu_driver.sh "$@"
