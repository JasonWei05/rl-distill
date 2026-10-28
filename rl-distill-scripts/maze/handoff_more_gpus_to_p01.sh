#!/usr/bin/env bash
# One-off (2026-10-27 12:25): add GPUs 1-3 to the p01 matrix when the p1 evaluation shards finish, and GPU 0 when the
# p01 S0 seed-1 replicate finishes its 4000 steps.
R=/home/jiahaoz4/tailrl_maze_runs; M=/home/jiahaoz4/rl-distill/rl-distill-scripts/maze; PLAN=$R/budget/p01/plan.json
gpufree() { for i in $(seq 1 120); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"); [ "${m:-99999}" -lt 1000 ] && return 0; sleep 5; done; return 1; }
while ps -eo args -ww | grep -F 'collect_budget_results.py --plan /home/jiahaoz4/tailrl_maze_runs/budget/p1/plan.json' | grep -v grep >/dev/null; do sleep 30; done
echo "$(date -Is) p1 evaluation shards finished"
for g in 1 2 3; do gpufree "$g"; echo "$(date -Is) GPU $g -> p01 matrix worker"; bash "$M/run_budget_matrix.sh" "$PLAN" "$g"; done
until grep -q 'Final validation metrics' "$R/rl/p01s1_S0_rl.log"; do sleep 60; done
while ps -eo args -ww | grep -F 'suite p01s1-s0 ' | grep -v grep >/dev/null; do sleep 10; done
gpufree 0; echo "$(date -Is) p01s1_S0 finished; GPU 0 -> p01 matrix worker"; bash "$M/run_budget_matrix.sh" "$PLAN" 0
