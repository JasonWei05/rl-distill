#!/usr/bin/env bash
# One-off (2026-10-27 ~05:00): once the p10 extension evaluation (GPU 0) and the extension workers (GPUs 1-3) are done,
# start seed-1 replicates of the S0 RL runs of the harder pairs — p01s1-s0 on GPU 0 (is S0's stall at ~7% from the
# 0.1% start seed-dependent?) and p1s1-s0 on GPU 1 (Arm B error bar for the 1% pair). GPUs 2-3 stay free for the p1 matrix.
R=/home/jiahaoz4/tailrl_maze_runs; M=/home/jiahaoz4/rl-distill/rl-distill-scripts/maze; J=$R/budget/p10/jobs
gpufree() { for i in $(seq 1 40); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"); [ "${m:-99999}" -lt 1000 ] && return 0; sleep 5; done; return 1; }
while kill -0 2119661 2>/dev/null; do sleep 30; done            # ext evaluation on GPU 0
gpufree 0
SEED=1 STEPS=4000 SAVE_FREQ=5 TEST_FREQ=10 VAL_N=16 setsid nohup bash "$M/run_maze_rl.sh" "$R/matched/p01_S0_base" p01s1-s0 0 > "$R/rl/p01s1_S0_rl.log" 2>&1 < /dev/null &
echo "$(date -Is) launched p01s1-s0 on GPU 0 (pid $!)"
while [ -f "$J/worker_gpu1.pid" ] && kill -0 "$(cat "$J/worker_gpu1.pid")" 2>/dev/null; do sleep 30; done   # ext worker on GPU 1
gpufree 1
SEED=1 STEPS=3000 SAVE_FREQ=5 TEST_FREQ=10 VAL_N=16 setsid nohup bash "$M/run_maze_rl.sh" "$R/matched/p1_S0_base" p1s1-s0 1 > "$R/rl/p1s1_S0_rl.log" 2>&1 < /dev/null &
echo "$(date -Is) launched p1s1-s0 on GPU 1 (pid $!)"
