#!/usr/bin/env bash
# Launch ONE RL run of a matched pair (one size, one seed), detached on one GPU, with the budget-study cadence and the
# naming the extraction expects (rl_curve_family.py extract): suite <pair>[s<seed>]-<s0|l0>, log rl/<pair>[s<seed>]_<S0|L0>_rl.log.
#
# Usage: bash launch_single_rl.sh <pair> <S0|L0> <seed> <gpu>
#   Env: STEPS (4000), SAVE_FREQ (10), TEST_FREQ (10), VAL_N (16), MAX_ACTOR_CKPT_TO_KEEP (null = keep all),
#        FOREGROUND=1 to run in the foreground and return the run's exit code (gpu_queue.sh sets it)
#   Needs ~/tailrl_maze_runs/matched/<pair>_<S0|L0>_base/ckpt-0 (make_pair.py, or a Hub base linked by hand).
#   Re-running the same command with a larger STEPS resumes the run (train.sh resume_mode=auto) as long as the last
#   checkpoint still has its optimizer state (keep MAX_ACTOR_CKPT_TO_KEEP >= 2 or unset).
set -euo pipefail
PAIR=${1:?pair}; SIDE=${2:?S0 or L0}; SEED=${3:?seed}; GPU=${4:?gpu}
case "$SIDE" in S0|L0) ;; *) echo "side must be S0 or L0" >&2; exit 1;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
M="$HOME/tailrl_maze_runs/matched"; R="$HOME/tailrl_maze_runs/rl"; mkdir -p "$R"
[ -e "$M/${PAIR}_${SIDE}_base/ckpt-0/config.json" ] || { echo "missing $M/${PAIR}_${SIDE}_base/ckpt-0" >&2; exit 1; }
TAG=$PAIR; [ "$SEED" != 0 ] && TAG="${PAIR}s${SEED}"
SUITE="${TAG}-$(echo "$SIDE" | tr 'SL' 'sl')"   # S0 -> s0, L0 -> l0
export SAVE_FREQ=${SAVE_FREQ:-10} TEST_FREQ=${TEST_FREQ:-10} VAL_N=${VAL_N:-16} SEED STEPS=${STEPS:-4000}
if [ "${FOREGROUND:-0}" = 1 ]; then
  echo "$PAIR $SIDE seed $SEED on GPU $GPU, $STEPS steps, suite $SUITE (foreground) -> $R/${TAG}_${SIDE}_rl.log"
  exec bash "$HERE/run_maze_rl.sh" "$M/${PAIR}_${SIDE}_base" "$SUITE" "$GPU" > "$R/${TAG}_${SIDE}_rl.log" 2>&1 < /dev/null
fi
setsid nohup bash "$HERE/run_maze_rl.sh" "$M/${PAIR}_${SIDE}_base" "$SUITE" "$GPU" > "$R/${TAG}_${SIDE}_rl.log" 2>&1 < /dev/null &
echo "$PAIR $SIDE seed $SEED on GPU $GPU, $STEPS steps, suite $SUITE (pid $!) -> $R/${TAG}_${SIDE}_rl.log"
