#!/usr/bin/env bash
# One-off (2026-10-27 ~08:45): when p1_S0 finishes its 3000 steps, plan the p1 budgets (smoothed saturation rule) and
# start the p1 distillation matrix on GPUs 2, 3, 6, 7 (GPU 3 is taken back from the p01n64 diagnostic, session 2368058).
R=/home/jiahaoz4/tailrl_maze_runs; M=/home/jiahaoz4/rl-distill/rl-distill-scripts/maze; PY=/home/jiahaoz4/TailRL/.venv/bin/python
C=$R/checkpoints/tailrl-text-maze
gpufree() { for i in $(seq 1 60); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"); [ "${m:-99999}" -lt 1000 ] && return 0; sleep 5; done; return 1; }
until grep -q 'Final validation metrics' "$R/rl/p1_S0_rl.log"; do sleep 30; done
while ps -eo args -ww | grep -F 'suite p1-s0 ' | grep -v grep >/dev/null; do sleep 10; done
gpufree 6
echo "$(date -Is) p1_S0 finished; planning"
mkdir -p "$R/budget/p1"
"$PY" "$M/plan_budgets.py" --pair p1 --s-root "$C/textmaze_p1-s0_grpo_binary_ckpt0_bs256_G16_1gpu_6_seed0" \
  --l-root "$C/textmaze_p1-l0_grpo_binary_ckpt0_bs256_G16_1gpu_7_seed0" --s-log "$R/rl/p1_S0_rl.log" --l-log "$R/rl/p1_L0_rl.log" \
  --out "$R/budget/p1/plan.json" || { echo "PLANNING FAILED"; exit 1; }
echo "$(date -Is) stopping the p01n64 diagnostic (session 2368058) to free GPU 3"
pkill -TERM -s 2368058; sleep 20; pkill -KILL -s 2368058 2>/dev/null; gpufree 3
bash "$M/run_budget_matrix.sh" "$R/budget/p1/plan.json" 2 3 6 7
echo "$(date -Is) p1 matrix launched"
