#!/usr/bin/env bash
# RL on a matched maze model (Arm B: S0 direct RL; Arm A step 1: L0 RL), through TailRL's train.sh.
#
# Usage: bash rl-distill-scripts/maze/run_maze_rl.sh <model-base-dir> <suite-tag> <gpu> [extra train.sh args]
#   <model-base-dir> must contain ckpt-0 (the matched/*_base dirs do):
#     ~/tailrl_maze_runs/matched/primary_S0_base   ~/tailrl_maze_runs/matched/primary_L0_base
#   Env knobs: REWARD (binary = goal_reached [default] | binary_shortest), STEPS (2000), N_ROLLOUTS (16),
#              BATCH (256), SAVE_FREQ / TEST_FREQ (50), VAL_N (64), ACTOR_LR (1e-4), METHOD (grpo), SEED (0)
# Recipe (rl-distill RECIPES.md 1.1-1.2 at maze scale): GRPO, 16 rollouts/prompt, 256 prompts/step, one update per
# step, no KL / entropy terms, temperature 1, validation on the 1000 dev mazes every TEST_FREQ steps, every checkpoint
# kept with an HF export (best step chosen post hoc on dev), FLOPs ledger on. ATTN_IMPL=sdpa (no flash-attn on sm_120).
set -euo pipefail
MODEL_BASE_DIR=${1:?model base dir containing ckpt-0}; SUITE=${2:?suite tag}; GPU=${3:?gpu}; shift 3
cd "$HOME/TailRL/experiments/text_maze"
source "$HOME/TailRL/.venv/bin/activate"
export ATTN_IMPL=sdpa MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-null}" ACTOR_LR="${ACTOR_LR:-1e-4}"
exec bash scripts/train.sh --suite "$SUITE" --method "${METHOD:-grpo}" --model-base-dir "$MODEL_BASE_DIR" --ckpt-step 0 \
  --seed "${SEED:-0}" --gpu-ids "$GPU" --n-rollouts "${N_ROLLOUTS:-16}" --batch-size "${BATCH:-256}" \
  --total-steps "${STEPS:-2000}" --reward "${REWARD:-binary}" --reward-transform raw \
  --val-n "${VAL_N:-64}" --val-prompts 1000 --test-freq "${TEST_FREQ:-50}" --save-freq "${SAVE_FREQ:-50}" "$@"
