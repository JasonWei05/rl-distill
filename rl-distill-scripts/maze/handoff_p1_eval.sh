#!/usr/bin/env bash
# One-off (2026-10-27 12:20): when the p1 matrix is complete (30 DONE markers) and its workers on GPUs 1-3 have exited,
# evaluate every p1 point on dev (k=64) and then on test1000 (k=256), three shards on GPUs 1, 2, 3.
R=/home/jiahaoz4/tailrl_maze_runs; M=/home/jiahaoz4/rl-distill/rl-distill-scripts/maze; PY=/home/jiahaoz4/TailRL/.venv/bin/python
B=$R/budget/p1; J=$B/jobs; T=/home/jiahaoz4/tailrl_maze_data/test1000/test.parquet
S1=$R/checkpoints/tailrl-text-maze/textmaze_p1s1-s0_grpo_binary_ckpt0_bs256_G16_1gpu_1_seed1
gpufree() { for i in $(seq 1 120); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"); [ "${m:-99999}" -lt 1000 ] && return 0; sleep 5; done; return 1; }
until [ "$(ls $J/*.DONE 2>/dev/null | wc -l)" -ge 30 ]; do sleep 30; done
echo "$(date -Is) p1 matrix complete"
declare -A RX=([1]='^armB|^teacher' [2]='^p1-b[123]-' [3]='^p1-b[45]-')
for g in 1 2 3; do
  pf="$J/worker_gpu$g.pid"; while [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; do sleep 20; done
  gpufree "$g"
  extra=""; [ "$g" = 1 ] && extra="--extra-s-roots s1=$S1"
  CUDA_VISIBLE_DEVICES=$g setsid nohup bash -c "cd $M; $PY collect_budget_results.py --plan $B/plan.json --k 64 --labels-regex '${RX[$g]}' $extra; $PY collect_budget_results.py --plan $B/plan.json --k 256 --parquet $T --labels-regex '${RX[$g]}' $extra" > "$B/collect_shard_gpu$g.log" 2>&1 < /dev/null &
  echo "$(date -Is) eval shard on GPU $g: ${RX[$g]} (pid $!)"
done
