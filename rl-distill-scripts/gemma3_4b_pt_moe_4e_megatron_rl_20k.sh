#!/usr/bin/env bash
set -euo pipefail

export NUM_EXPERTS=4
export HF_MOE_REVISION="${HF_MOE_REVISION:-cd87f6e541b1bc0fba8caef218c55601fbb0c533}"
export ACTOR_TP="${ACTOR_TP:-2}"
export ACTOR_EP="${ACTOR_EP:-4}"
export EXP_NAME="${EXP_NAME:-DAPO-Gemma3-4B-PT-MoE-4E-Megatron-RL-$(date +%Y%m%d-%H%M)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/gemma3_4b_pt_moe_megatron_rl_20k.sh" "$@"
