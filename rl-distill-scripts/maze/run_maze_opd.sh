#!/usr/bin/env bash
# On-policy distillation, policy-gradient form (PG-k1), of a maze teacher into a matched student (Arm A step 2,
# on-policy variant), through TailRL's train.sh with the fork's opd_token estimator.
#
# Usage: bash rl-distill-scripts/maze/run_maze_opd.sh <student-base-dir> <teacher-hf-dir> <suite-tag> <gpu> [extra train.sh args]
#   <student-base-dir> contains ckpt-0 (e.g. ~/tailrl_maze_runs/matched/primary_S0_base)
#   <teacher-hf-dir>   an HF model dir, e.g. <L0 RL run>/global_step_N/actor/huggingface
#   Env knobs: STEPS (400), BATCH (128), ACTOR_LR (1e-4), OPD_KL_COEF (1.0), OPD_MAX_CLAMP (5.0), SAVE_FREQ / TEST_FREQ (10), VAL_N (64)
# Recipe (rl-distill RECIPES.md 3.1-3.3 at maze scale): student samples 1 response per prompt, 128 prompts per step, one
# update per step; per-token advantage = -clip(log pi_S - log pi_T, -5, 5); zero task reward; teacher loaded as the
# reference policy; HF rollout == trainer so no IcePop mask is needed. The teacher's forward FLOPs are logged as tflops/ref.
set -euo pipefail
STUDENT_BASE_DIR=${1:?student base dir containing ckpt-0}; TEACHER=${2:?teacher HF dir}; SUITE=${3:?suite tag}; GPU=${4:?gpu}; shift 4
cd "$HOME/TailRL/experiments/text_maze"
source "$HOME/TailRL/.venv/bin/activate"
export ATTN_IMPL=sdpa MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-null}" ACTOR_LR="${ACTOR_LR:-1e-4}" OPD_KL_COEF="${OPD_KL_COEF:-1.0}" OPD_MAX_CLAMP="${OPD_MAX_CLAMP:-5.0}"
exec bash scripts/train.sh --suite "$SUITE" --opd --ref-model-dir "$TEACHER" --model-base-dir "$STUDENT_BASE_DIR" --ckpt-step 0 \
  --seed "${SEED:-0}" --gpu-ids "$GPU" --n-rollouts 1 --batch-size "${BATCH:-128}" \
  --total-steps "${STEPS:-400}" --reward zero --reward-transform raw \
  --val-n "${VAL_N:-64}" --val-prompts 1000 --test-freq "${TEST_FREQ:-10}" --save-freq "${SAVE_FREQ:-10}" "$@"
