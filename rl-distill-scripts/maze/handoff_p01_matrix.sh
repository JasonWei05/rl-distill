#!/usr/bin/env bash
# One-off (2026-10-27 12:15): when p01_S0 finishes its 4000 steps, plan the p01 budgets with B5 = the whole S0 run (S0 never
# saturated from the 0.1% start: dev 0.08 -> 0.13 over 4000 steps) and start the p01 matrix on GPUs 4, 5, 6, 7 as each one is
# released by the p1 matrix workers. GPUs 1-3 are left for the p1 evaluation shards (add them to p01 by hand afterwards).
R=/home/jiahaoz4/tailrl_maze_runs; M=/home/jiahaoz4/rl-distill/rl-distill-scripts/maze; PY=/home/jiahaoz4/TailRL/.venv/bin/python
C=$R/checkpoints/tailrl-text-maze
S_ROOT=$C/textmaze_p01-s0_grpo_binary_ckpt0_bs256_G16_1gpu_4_seed0; L_ROOT=$C/textmaze_p01-l0_grpo_binary_ckpt0_bs256_G16_1gpu_5_seed0
gpufree() { for i in $(seq 1 120); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"); [ "${m:-99999}" -lt 1000 ] && return 0; sleep 5; done; return 1; }
until grep -q 'Final validation metrics' "$R/rl/p01_S0_rl.log"; do sleep 30; done
while ps -eo args -ww | grep -F 'suite p01-s0 ' | grep -v grep >/dev/null; do sleep 10; done
echo "$(date -Is) p01_S0 finished; planning with B5 = whole run"
mkdir -p "$R/budget/p01"
B5=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['cumulative_tflops']['total'])" "$S_ROOT/global_step_4000/flops_ledger.json")
"$PY" "$M/plan_budgets.py" --pair p01 --s-root "$S_ROOT" --l-root "$L_ROOT" --s-log "$R/rl/p01_S0_rl.log" --l-log "$R/rl/p01_L0_rl.log" \
  --b5-tflops "$B5" --out "$R/budget/p01/plan.json" || { echo "PLANNING FAILED"; exit 1; }
for g in 4 5 6 7; do
  pf="$R/budget/p1/jobs/worker_gpu$g.pid"
  while [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; do sleep 20; done   # p1 worker on this GPU still busy
  gpufree "$g"
  echo "$(date -Is) GPU $g free -> p01 matrix worker"
  bash "$M/run_budget_matrix.sh" "$R/budget/p01/plan.json" "$g"
done
echo "$(date -Is) p01 matrix launched on GPUs 4-7"
