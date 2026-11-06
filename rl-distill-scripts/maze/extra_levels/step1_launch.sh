#!/usr/bin/env bash
# Step 1 of "Extra start levels for the maze RL-curve law" on federer: regenerate the two pretraining trajectories
# (S init_seed 4, L init_seed 0, data_seed 0, 20k steps, a checkpoint every 20 steps), prune checkpoints outside the
# selection windows as they land (/work2 cannot hold all 34 GB), and stream-screen the stride-50 grid at k = 16 on the
# 1000 dev mazes with 4 polling workers (two per GPU is fine at k = 16).
#   TRAIN_GPUS="4 5" SCREEN_GPUS="7 7 4 5" bash step1_launch.sh
set -euo pipefail
source /work2/jiahaoz4/maze_extra/env.sh
cd /work2/jiahaoz4/rl-distill
: "${TRAIN_GPUS:?two GPU ids}" "${SCREEN_GPUS:?four GPU ids}"
RUNS_ROOT=$R MAX_STEPS=20000 SAVE_STEPS=20 EVAL_STEPS=50 TRAIN_GPUS="$TRAIN_GPUS" \
  RUN_SPECS="S_ds0_is4 S 4;L_ds0_is0_rep L 0" bash $M/launch_pretrain_matrix.sh
setsid nohup $PY /work2/jiahaoz4/maze_extra/prune_pretrain_rerun.py --root $R --loop 60 \
  > /work2/jiahaoz4/maze_extra/logs/prune_pretrain.log 2>&1 < /dev/null &
w=0
for g in $SCREEN_GPUS; do
  CUDA_VISIBLE_DEVICES=$g setsid nohup $PY $M/screen_watch.py --run-dirs $R/S_ds0_is4 $R/L_ds0_is0_rep \
    --stride 50 --k 16 --worker $w --n-workers 4 --tag stride50_k16 > $R/screen_worker_$w.log 2>&1 < /dev/null &
  w=$((w + 1))
done
echo "step 1 launched: training on GPUs $TRAIN_GPUS, screening workers on $SCREEN_GPUS, pruner loop every 60 s"
