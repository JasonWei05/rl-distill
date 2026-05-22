#!/usr/bin/env bash
set -euo pipefail

export NUM_EXPERTS=2
export HF_MOE_REVISION="${HF_MOE_REVISION:-952a11b802b63ef091f20ec2dfe08eb66376794c}"
export ACTOR_TP="${ACTOR_TP:-1}"
export ACTOR_EP="${ACTOR_EP:-2}"
export EXP_NAME="${EXP_NAME:-DAPO-Gemma3-4B-PT-MoE-2E-Megatron-RL-$(date +%Y%m%d-%H%M)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/gemma3_4b_pt_moe_megatron_rl_20k.sh" "$@"
