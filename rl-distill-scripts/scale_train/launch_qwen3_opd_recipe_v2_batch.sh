#!/usr/bin/env bash
# Re-run of the from-base on-policy distillations with the v2 recipe (PROGRESS_LOG 2026-10-26): PG-k1 as before, but a CONSTANT
# learning rate of 1e-6 after a 50-step linear warmup (was 2e-6 / 20 -- the first ~50 steps overshot: grad norm ~3, response length
# 2.8k +- 0.8k with 8% truncation), 400 steps, no early stopping, validation + permanent checkpoint every 10, borrowing true,
# priority high, auto-resume. RUN_TAG pgk1-base-lr1e6w50 keeps the v1 runs intact. The 8B hard teacher is excluded (its RL is
# being re-run). Teachers are pinned to their RL best step; GPUs follow the student (1.7B: 2, 4B: 4).
#   bash launch_qwen3_opd_recipe_v2_batch.sh            # dry run: print each spec's resolved launch env
#   bash launch_qwen3_opd_recipe_v2_batch.sh --launch   # start one supervisor per spec (skips finished / running ones)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RUN_TAG=pgk1-base-lr1e6w50
COMMON=(RUN_TAG=${RUN_TAG} TOTAL_TRAINING_STEPS=400 TEST_FREQ=10 SAVE_FREQ=10 OPD_LR=1e-6 OPD_LR_WARMUP_STEPS=50
        ALLOW_BORROWING=true MAX_RELAUNCHES=200)
SPECS=(
  # 4B RL teacher -> 1.7B base
  q4be-s500-to-q1p7b-base q4bm-s510-to-q1p7b-base q4bh-s560-to-q1p7b-base
  # 8B RL teacher (easy / medium; hard excluded) -> 1.7B / 4B base
  q8be-s460-to-q1p7b-base q8bm-s480-to-q1p7b-base q8be-s460-to-q4b-base q8bm-s480-to-q4b-base
  # 30B-A3B RL teacher (medium / hard; no 30B easy RL) -> 1.7B / 4B base: the from-base counterpart of the two-stage 30B runs
  q30bm-s320-to-q1p7b-base q30bh-s410-to-q1p7b-base q30bm-s320-to-q4b-base q30bh-s410-to-q4b-base
  # same-size control: 1.7B RL teacher -> 1.7B base
  q1p7be-s470-to-q1p7b-base q1p7bm-s680-to-q1p7b-base q1p7bh-s910-to-q1p7b-base
)
MODE="${1:-}"
B=s3://scale-ml/genai/rl-distill
if [ "${MODE}" != --launch ]; then
  sed 's/^exec bash launch_st_with_code.sh/echo DRYRUN launch_st_with_code.sh/' launch_qwen3_onpolicy_distill.sh > ./.opd_v2_dry.sh
  trap 'rm -f ./.opd_v2_dry.sh' EXIT
fi
for spec in "${SPECS[@]}"; do
  if [ "${MODE}" = --launch ]; then
    if AWS_PROFILE=ml-worker aws s3 ls "${B}/qwen3-opd-full-checkpoints/${spec}-${RUN_TAG}/run_complete.json" >/dev/null 2>&1; then
      echo "SKIP ${spec}: complete"; continue
    fi
    if tmux has-session -t "rl-q3-opd-${spec}" 2>/dev/null; then echo "SKIP ${spec}: supervisor already running"; continue; fi
    env SPEC="${spec}" "${COMMON[@]}" bash start_qwen3_onpolicy_distill_supervisor.sh | sed 's/ + ckpt-janitor.*//'
  else
    env SPEC="${spec}" "${COMMON[@]}" bash ./.opd_v2_dry.sh --image IMG 2>&1 | tr ',' '\n' \
      | grep -E "^(DIFFICULTY_DATASET|ONPOLICY_DISTILL_TEACHER_S3_URI|ONPOLICY_DISTILL_TEACHER_TP|ACTOR_LR|ACTOR_LR_WARMUP_STEPS|TOTAL_TRAINING_STEPS|TEST_FREQ|SAVE_FREQ|EARLY_STOPPING_ENABLED|OFFLOAD|GEMMA4_INIT_MODEL_S3_URI|WANDB_RUN_ID)=|LAUNCH_ENV GEMMA4_MODEL|DRYRUN|FATAL" \
      | sed -E 's|s3://scale-ml/genai/rl-distill/||; s/LAUNCH_ENV //; s/DRYRUN launch_st_with_code.sh (--gpus-per-instance [0-9]+ --priority [a-z]+ ?(--allow-borrowing)?).*/\1/' \
      | sort -u | tr '\n' ' ' | sed "s/^/[${spec}] /"; echo
  fi
done
