#!/usr/bin/env bash
# One-screen status of the budget-study queue: GPUs, RL runs (step, dev curve tail), selection jobs, matrix workers.
R="$HOME/tailrl_maze_runs"
echo "== $(date '+%F %T') GPUs (idx, util, mem MiB)"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';'; echo
echo "== RL runs"
for L in "$R"/rl/p*_rl.log; do
  n=$(basename "$L" _rl.log); last=$(grep -oE 'step:[0-9]+ - ' "$L" | tail -1 | tr -d ' -'); total=$(grep -oE 'total_training_steps=[0-9]+' "$L" | head -1 | cut -d= -f2)
  curve=$(grep -oE 'step:[0-9]+ - |val-aux/maze_17/goal_reached/mean@16:[0-9.]+' "$L" | awk -F: '/^step/{s=$2+0} /mean@16/{printf "%d=%.3f ", s, $NF}' | awk '{n=NF; for(i=(n>8?n-7:1);i<=n;i++) printf "%s ", $i}')
  init=$(grep -oE "'val-aux/maze_17/goal_reached/mean@16': [0-9.]+" "$L" | head -1 | cut -d' ' -f2)
  tfl=$(grep -oE 'tflops/cumulative_total:[0-9.]+' "$L" | tail -1 | cut -d: -f2); tb=$(grep -ciE 'traceback' "$L")
  s=${n,,}; s=${s/_/-}; alive=$(pgrep -f "suite $s " >/dev/null 2>&1 && echo run || echo "----")
  echo "  $n [$alive] ${last:-step:0}/${total:-?} init=${init:-?} dev: $curve cumTFLOPs=${tfl:-0} tracebacks=$tb"
done
echo "== selection (k=16 bracket -> candidates -> finals)"
for f in "$R"/select/*.log; do [ -f "$f" ] || continue; echo "  $(basename "$f" .log): $(grep -cE '^ +[0-9]+ +0\.' "$f") rows; $(grep -E 'final k=' "$f" | sed -E 's/\| short.*//' | cut -c1-80 | tr '\n' ';') $(grep -q '^wrote' "$f" && echo DONE)"; done
echo "== matched pairs: $(ls "$R"/matched 2>/dev/null | grep -E '_(S0|L0)$' | tr '\n' ' ')"
echo "== budget matrices"
for p in "$R"/budget/*/; do [ -d "$p/jobs" ] || continue
  echo "  $(basename "$p"): done $(ls "$p"/jobs/*.DONE 2>/dev/null | wc -l) failed $(ls "$p"/jobs/*.FAILED 2>/dev/null | wc -l) of $(grep -c . "$p"/jobs/all.list 2>/dev/null) jobs; running: $(for l in "$p"/jobs/*.lock; do [ -d "$l" ] && echo -n "$(basename "$l" .lock)@gpu$(cut -d' ' -f1 "$l/owner" 2>/dev/null) "; done); workers alive: $(for pf in "$p"/jobs/worker_gpu*.pid; do [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null && echo -n "$(basename "$pf" .pid | sed 's/worker_//') "; done)"
  for fl in "$p"/jobs/*.FAILED; do [ -f "$fl" ] && echo "    FAILED: $(basename "$fl" .FAILED): $(grep -iE 'error|Traceback' "${fl%.FAILED}.log" | tail -1 | cut -c1-120)"; done
done
echo "== disk: $(df -h /home | awk 'NR==2{print $4" free"}')"
