#!/bin/bash
# 2XS / 3XS TIS grids (t3x2 / t3x3): the small grid's final layouts. 16 + 64 prompts packed on 1-GPU pods, 256 one per
# 1-GPU pod (update micro-batch 1024), 1024 data-parallel on 4-GPU pods; all runs to 10k steps (patience never binds).
#   CODE_REF=<commit> bash rl/launch_t3x_grids.sh
set -euo pipefail
L="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch_rl_small_grid.sh"
cd /mnt/efs/jasonwei/rl-distill  # launches run from the main checkout
export CODE_REF=${CODE_REF:?} FAST_DECODE=1 MAX_STEPS=10000 PATIENCE_OVERRIDE=2000 ALLOW_BORROWING=false DEADLINE_HOURS=72
for g in t3x2 t3x3; do
  GPUS=1 GRID=$g BATCHES="16 64" NPODS=4 EXTRA_ENVS_APPEND=",PREFIX_FORWARD=1" bash $L
  GPUS=1 GRID=$g BATCHES="256" NPODS=9 TAG_PREFIX_OVERRIDE=maze-rl-${g}b MICRO=1024 EXTRA_ENVS_APPEND=",PREFIX_FORWARD=1" bash $L
  GPUS=4 GRID=$g BATCHES="1024" NPODS=9 TAG_PREFIX_OVERRIDE=maze-rl-${g}x MICRO=1024 \
    EXTRA_ENVS_APPEND=",PREFIX_FORWARD=1,RUN_ALL_GPUS=1,REWARD_NUM_PROCESSES=32,ROLLOUT_MICRO_BATCH_SIZE=16384" bash $L
done
