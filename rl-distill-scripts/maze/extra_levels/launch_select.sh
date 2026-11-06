#!/usr/bin/env bash
# Step 2 on federer: one select_matched_init.py job (one run, one target), detached on one empty GPU.
#   bash launch_select.sh <S_ds0_is4|L_ds0_is0_rep> <target> <grid k> <out-name> <gpu> [extra select args]
# Log: <run>/<out-name without .json>.log. Finals at k = 1024, the three closest candidates (runbook Step 2).
set -euo pipefail
source /work2/jiahaoz4/maze_extra/env.sh
RUN=${1:?run}; TARGET=${2:?target}; K=${3:?grid k}; OUT=${4:?out name}; GPU=${5:?gpu}; shift 5
cd /work2/jiahaoz4/rl-distill
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $PY $M/select_matched_init.py --run $R/$RUN --target $TARGET --k $K --final-k 1024 \
  --n-final 3 --out-name $OUT "$@" > $R/$RUN/${OUT%.json}.log 2>&1 < /dev/null &
echo "select $RUN target $TARGET k $K on GPU $GPU (pid $!) -> $R/$RUN/${OUT%.json}.log"
