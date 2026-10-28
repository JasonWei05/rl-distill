#!/usr/bin/env bash
# One-off (2026-10-27 16:30): evaluate the p01 matrix in two passes so the two long OPD cells do not hold up everything:
#   pass 1 when only <= 2 jobs are still running: three shards (dev k=64 then test k=256) on three GPUs no job owns;
#   pass 2 when all 30 jobs are done: one shard over everything (cached points are skipped) for the last cells.
R=/home/jiahaoz4/tailrl_maze_runs; M=/home/jiahaoz4/rl-distill/rl-distill-scripts/maze; PY=/home/jiahaoz4/TailRL/.venv/bin/python
B=$R/budget/p01; J=$B/jobs; T=/home/jiahaoz4/tailrl_maze_data/test1000/test.parquet
S1=$R/checkpoints/tailrl-text-maze/textmaze_p01s1-s0_grpo_binary_ckpt0_bs256_G16_1gpu_0_seed1
gpufree() { for i in $(seq 1 120); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"); [ "${m:-99999}" -lt 1000 ] && return 0; sleep 5; done; return 1; }
owners() { for l in "$J"/*.lock; do [ -d "$l" ] && cut -d' ' -f1 "$l/owner"; done | sort -u | tr '\n' ' '; }
while true; do n=$(ls -d "$J"/*.lock 2>/dev/null | wc -l); d=$(ls "$J"/*.DONE 2>/dev/null | wc -l); [ "$n" -le 2 ] && [ "$d" -ge 28 ] && break; sleep 60; done
busy=" $(owners)"; free=(); for g in 0 1 2 3 4 5 6 7; do [[ $busy == *" $g "* ]] || free+=("$g"); done
echo "$(date -Is) pass 1: $(ls "$J"/*.DONE | wc -l) done, busy GPUs:$busy free: ${free[*]}"
declare -a RX=('^armB|^teacher' '^p01-b[123]-' '^p01-b[45]-')
for i in 0 1 2; do
  g=${free[$i]}; gpufree "$g"
  extra=""; [ "$i" = 0 ] && extra="--extra-s-roots s1=$S1"
  CUDA_VISIBLE_DEVICES=$g setsid nohup bash -c "cd $M; $PY collect_budget_results.py --plan $B/plan.json --k 64 --labels-regex '${RX[$i]}' $extra; $PY collect_budget_results.py --plan $B/plan.json --k 256 --parquet $T --labels-regex '${RX[$i]}' $extra" > "$B/collect_pass1_gpu$g.log" 2>&1 < /dev/null &
  echo "$(date -Is) pass-1 shard on GPU $g: ${RX[$i]} (pid $!)"
done
until [ "$(ls "$J"/*.DONE 2>/dev/null | wc -l)" -ge 30 ]; do sleep 60; done
echo "$(date -Is) matrix complete; pass 2"
while ps -eo args -ww | grep -F "collect_budget_results.py --plan $B/plan.json" | grep -v grep >/dev/null; do sleep 30; done
g=${free[0]}; gpufree "$g"
CUDA_VISIBLE_DEVICES=$g setsid nohup bash -c "cd $M; $PY collect_budget_results.py --plan $B/plan.json --k 64 --extra-s-roots s1=$S1; $PY collect_budget_results.py --plan $B/plan.json --k 256 --parquet $T --extra-s-roots s1=$S1" > "$B/collect_pass2_gpu$g.log" 2>&1 < /dev/null &
echo "$(date -Is) pass-2 shard on GPU $g (pid $!)"
