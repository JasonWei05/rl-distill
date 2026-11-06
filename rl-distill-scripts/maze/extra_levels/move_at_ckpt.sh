#!/usr/bin/env bash
# Move a running run right after its next checkpoint, so the resume redoes no finished step: wait for
# latest_checkpointed_iteration.txt in its real run dir to change, give the save 30 s to settle, then run move_run.sh.
#   setsid nohup bash move_at_ckpt.sh <S0|L0> <pair> <seed> <steps> <dest gpu> <ray port index> >> logs/move_<suite>.log 2>&1 &
set -euo pipefail
SIDE=$1; PAIR=$2; SEED=$3
TAG=$PAIR; [ "$SEED" != 0 ] && TAG="${PAIR}s${SEED}"
case "$SIDE" in S0) SUITE="${TAG}-s0" ;; L0) SUITE="${TAG}-l0" ;; *) echo "side S0|L0" >&2; exit 1 ;; esac
CK=$HOME/tailrl_maze_runs/checkpoints/tailrl-text-maze
d=$(find "$CK" -maxdepth 1 -type d -name "textmaze_${SUITE}_grpo_binary_*" | head -1)
t=$d/latest_checkpointed_iteration.txt
c0=$(cat "$t")
echo "$(date -u '+%m-%d %H:%M:%S') $SUITE: waiting for the checkpoint after $c0"
while [ "$(cat "$t")" = "$c0" ]; do sleep 5; done
sleep 30
echo "$(date -u '+%m-%d %H:%M:%S') $SUITE: checkpoint $(cat "$t"); moving to GPU $5"
bash /work2/jiahaoz4/maze_extra/move_run.sh "$@"
echo "$(date -u '+%m-%d %H:%M:%S') $SUITE: move_run.sh rc=$?"
