#!/usr/bin/env bash
# One-off (2026-10-27 11:45): when p01_L0 finishes its 2000 steps, give GPU 5 to the p1 matrix.
R=/home/jiahaoz4/tailrl_maze_runs; M=/home/jiahaoz4/rl-distill/rl-distill-scripts/maze
until grep -q 'Final validation metrics' "$R/rl/p01_L0_rl.log"; do sleep 30; done
while ps -eo args -ww | grep -F 'suite p01-l0 ' | grep -v grep >/dev/null; do sleep 10; done
for i in $(seq 1 60); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5); [ "${m:-99999}" -lt 1000 ] && break; sleep 5; done
echo "$(date -Is) p01_L0 finished -> p1 matrix worker on GPU 5"
bash "$M/run_budget_matrix.sh" "$R/budget/p1/plan.json" 5
