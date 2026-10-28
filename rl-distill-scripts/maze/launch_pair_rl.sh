#!/usr/bin/env bash
# Launch the two RL runs of one matched pair (Arm B: S0 direct RL; Arm A step 1: L0 RL) detached, one GPU each, with the
# budget-study cadence: goal_reached reward, dev validation (mean@16) every 10 steps, a checkpoint every SAVE_FREQ steps.
#
# Usage: bash launch_pair_rl.sh <pair> <gpu-for-S0> <gpu-for-L0>
#   Env: S_STEPS (3000), L_STEPS (1500), SAVE_FREQ (5), TEST_FREQ (10), VAL_N (16)
#   Needs ~/tailrl_maze_runs/matched/<pair>_{S0,L0}_base/ckpt-0 (make_pair.py). Suites <pair>-s0 / <pair>-l0; logs in
#   ~/tailrl_maze_runs/rl/<pair>_{S0,L0}_rl.log. Re-running with the same suite resumes (train.sh resume_mode=auto).
set -euo pipefail
PAIR=${1:?pair}; GS=${2:?gpu for S0}; GL=${3:?gpu for L0}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
M="$HOME/tailrl_maze_runs/matched"; R="$HOME/tailrl_maze_runs/rl"; mkdir -p "$R"
for s in S0 L0; do [ -e "$M/${PAIR}_${s}_base/ckpt-0/config.json" ] || { echo "missing $M/${PAIR}_${s}_base/ckpt-0" >&2; exit 1; }; done
export SAVE_FREQ=${SAVE_FREQ:-5} TEST_FREQ=${TEST_FREQ:-10} VAL_N=${VAL_N:-16}
STEPS=${S_STEPS:-3000} setsid nohup bash "$HERE/run_maze_rl.sh" "$M/${PAIR}_S0_base" "${PAIR}-s0" "$GS" > "$R/${PAIR}_S0_rl.log" 2>&1 < /dev/null &
echo "$PAIR S0 RL on GPU $GS, ${S_STEPS:-3000} steps (pid $!) -> $R/${PAIR}_S0_rl.log"
STEPS=${L_STEPS:-1500} setsid nohup bash "$HERE/run_maze_rl.sh" "$M/${PAIR}_L0_base" "${PAIR}-l0" "$GL" > "$R/${PAIR}_L0_rl.log" 2>&1 < /dev/null &
echo "$PAIR L0 RL on GPU $GL, ${L_STEPS:-1500} steps (pid $!) -> $R/${PAIR}_L0_rl.log"
