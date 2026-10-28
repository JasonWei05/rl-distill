#!/usr/bin/env bash
# One-off GPU handoff for the p10 matrix (2026-10-27): stop the p10 RL runs once they have what the plan needs and give
# their GPUs to the matrix workers. p10s1_S0 (seed-1 replicate, session 1748384) stops at step >= 300 (B5 is at step 290),
# p10_S0 (session 1721858) at step >= 400 (plateau confirmed), p10s1_L0 is left to finish its 300 steps.
R=/home/jiahaoz4/tailrl_maze_runs; M=/home/jiahaoz4/rl-distill/rl-distill-scripts/maze; PLAN=$R/budget/p10/plan.json
laststep() { grep -oE "step:[0-9]+ - " "$1" | tail -1 | tr -dc "0-9"; }
gpufree() { for i in $(seq 1 30); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"); [ "${m:-99999}" -lt 1000 ] && return 0; sleep 3; done; return 1; }
stop_sid() { pkill -TERM -s "$1"; for i in $(seq 1 30); do pgrep -s "$1" >/dev/null || break; sleep 2; done; pkill -KILL -s "$1" 2>/dev/null; sleep 3; }
d0=0; d2=0; d3=0
while [ "$d0$d2$d3" != 111 ]; do
  s=$(laststep "$R/rl/p10s1_S0_rl.log")
  if [ $d2 = 0 ] && [ "${s:-0}" -ge 300 ]; then stop_sid 1748384; gpufree 2; echo "$(date -Is) stopped p10s1_S0 at step $s -> matrix worker GPU 2"; bash "$M/run_budget_matrix.sh" "$PLAN" 2; d2=1; fi
  s=$(laststep "$R/rl/p10_S0_rl.log")
  if [ $d0 = 0 ] && [ "${s:-0}" -ge 400 ]; then stop_sid 1721858; gpufree 0; echo "$(date -Is) stopped p10_S0 at step $s -> matrix worker GPU 0"; bash "$M/run_budget_matrix.sh" "$PLAN" 0; d0=1; fi
  if [ $d3 = 0 ] && grep -q "Final validation metrics" "$R/rl/p10s1_L0_rl.log" 2>/dev/null; then gpufree 3; echo "$(date -Is) p10s1_L0 finished -> matrix worker GPU 3"; bash "$M/run_budget_matrix.sh" "$PLAN" 3; d3=1; fi
  sleep 60
done
echo "$(date -Is) handoff watcher done"
