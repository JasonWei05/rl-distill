#!/usr/bin/env bash
# Move a running RL run to another card: check the card first (another user's process -> abort), stop the run, resume it
# there from its newest checkpoint (rl_launch.sh aliases the run dir and rotates the log; at most 9 steps are redone).
#   bash move_run.sh <S0|L0> <pair> <seed> <steps> <dest gpu> <ray port index (the run's own, freed by the stop)>
set -euo pipefail
SIDE=$1; PAIR=$2; SEED=$3; STEPS=$4; GPU=$5; PORT=$6
TAG=$PAIR; [ "$SEED" != 0 ] && TAG="${PAIR}s${SEED}"
case "$SIDE" in S0) SUITE="${TAG}-s0" ;; L0) SUITE="${TAG}-l0" ;; *) echo "side S0|L0" >&2; exit 1 ;; esac
L=/work2/jiahaoz4/maze_extra/rl_launch.sh
LOCK=/work2/jiahaoz4/maze_extra/locks/$SUITE.moving   # watchdog.sh leaves a run alone while this exists
touch "$LOCK"; trap 'rm -f "$LOCK"' EXIT
CHECK_ONLY=1 SIDE=$SIDE RAY_PORT_INDEX=$PORT bash "$L" "$PAIR" "$SEED" "$STEPS" "$GPU"
P=$(ps -u "$(id -un)" -o pid=,args= | awk -v s="experiment_name=textmaze_${SUITE}_" '/[v]erl.trainer.main_ppo/ && index($0, s) {print $1}')
[ -n "$P" ] || { echo "no running trainer for $SUITE" >&2; exit 1; }
echo "stopping $SUITE (pids $(echo $P))"; kill $P
# ps output is captured first: under pipefail an early-exiting grep -q can SIGPIPE ps and read as "not running"
for _ in $(seq 1 90); do out=$(ps -u "$(id -un)" -o args=); grep -q -- "[s]cripts/train.sh --suite $SUITE " <<< "$out" || break; sleep 5; done
# the stopped run's Ray can hold its port band for a while ("Port 75xx still busy ... refusing to start Ray"): if so,
# take a band no running run uses and nothing listens on
band_busy() { ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE ":(70|71|72|73|75|76|77|78|79)$(printf %02d "$1")$"; }
if band_busy "$PORT"; then
  # (an if, not &&: a stopped last run would make the loop, and so the assignment, fail under set -e)
  out=$(ps -u "$(id -un)" -o args=)
  used=$(for e in /work2/jiahaoz4/maze_extra/runs/*.env; do s=$(basename "$e" .env)
    if grep -q -- "[s]cripts/train.sh --suite $s " <<< "$out"; then grep '^PORT=' "$e" | cut -d= -f2; fi; done)
  for i in $(seq 0 11); do echo "$used" | grep -qx "$i" || band_busy "$i" || { echo "port band $PORT still busy; using $i"; PORT=$i; break; }; done
fi
SIDE=$SIDE RAY_PORT_INDEX=$PORT bash "$L" "$PAIR" "$SEED" "$STEPS" "$GPU"
