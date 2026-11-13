#!/usr/bin/env bash
# Start the next RL run on a card the moment the run on it ends, so the card is not left empty for others to take
# (another user took GPU 6 a minute after p5 finished). rl_launch.sh still refuses a card another user has a process on.
#   setsid nohup bash handoff.sh <suite to wait for> <SIDE> <pair> <seed> <steps> <gpu> <ray port index> > log 2>&1 &
set -uo pipefail
SUITE=$1; SIDE=$2; PAIR=$3; SEED=$4; STEPS=$5; GPU=$6; PORT=$7
echo "$(date '+%m-%d %H:%M:%S') waiting for $SUITE to end -> then $SIDE $PAIR seed $SEED, $STEPS steps on GPU $GPU (port index $PORT)"
while ps -u "$(id -un)" -o args= | grep -q -- "[s]cripts/train.sh --suite $SUITE "; do sleep 10; done
echo "$(date '+%m-%d %H:%M:%S') $SUITE ended"
SIDE=$SIDE RAY_PORT_INDEX=$PORT bash /work2/jiahaoz4/maze_extra/rl_launch.sh "$PAIR" "$SEED" "$STEPS" "$GPU"
echo "$(date '+%m-%d %H:%M:%S') rl_launch exit=$?"
