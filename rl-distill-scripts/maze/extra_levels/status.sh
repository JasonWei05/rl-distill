#!/usr/bin/env bash
# One-screen status of the extra-starts work on federer: pretraining, screening, selections, pairs, RL runs, disk, GPUs.
source /work2/jiahaoz4/maze_extra/env.sh
date
echo "== pretraining"
for r in S_ds0_is4 L_ds0_is0_rep2; do
  d=$R/$r; m=$(ls "$d" 2>/dev/null | grep -E '^(DONE|FAILED)$' | tr '\n' ' ')
  n=$(cat "$d"/screen_stride50_k16*.json 2>/dev/null | grep -o '"ckpt-[0-9]*"' | sort -u | wc -l)
  echo "  $r: step $(tail -1 "$d/train_metrics.jsonl" 2>/dev/null | grep -o '"step": [0-9]*' | grep -o '[0-9]*') ${m:+[$m]}| $(ls "$d" | grep -c '^ckpt-') ckpts on disk | $n grid points screened"
done
echo "== selections"
for f in "$R"/*/select_goal_reached_*.json; do
  [ -e "$f" ] || continue
  $PY - "$f" <<'EOF'
import json, sys, os
d = json.load(open(sys.argv[1]))
sel = d.get("_selection")
grid = [k for k in d if k.startswith(("avg-", "interp-")) and k.endswith(tuple(f"@k{k}" for k in (64, 256)))]
fin = [k for k in d if k.endswith("@k1024")]
best = sorted((abs(v["goal_reached"]["pass@1"] - (sel or {}).get("target", 0)), k, v["goal_reached"]["pass@1"]) for k, v in d.items() if k.endswith("@k1024"))
run = sys.argv[1].split("/")[-2]
print(f"  {run}/{os.path.basename(sys.argv[1])}: {len(grid)} grid, {len(fin)} finals{' DONE' if sel else ''}"
      + (f"; best final {best[0][1]} p@1 {best[0][2]:.4f}" if best else ""))
EOF
done
echo "== pairs in matched/manifest.json"
$PY -c "import json; m=json.load(open('$HOME/tailrl_maze_runs/matched/manifest.json'))['pairs']; [print(f'  {p}:', {s: round(v[s]['pass@1'], 4) if isinstance(v.get(s), dict) and 'pass@1' in v[s] else '?' for s in ('S0', 'L0')}) for p, v in m.items() if p in ('p03', 'p3', 'p5')]" 2>/dev/null
echo "== RL"
for log in "$HOME"/tailrl_maze_runs/rl/p*_S0_rl.log "$HOME"/tailrl_maze_runs/rl/p*_L0_rl.log; do
  [ -e "$log" ] || continue
  last=$(grep -E 'step:[0-9]+ - ' "$log" | tail -1)
  dev=$(grep -oE 'step:[0-9]+ - .*val-aux/maze_17/goal_reached/mean@16:[0-9.]+' "$log" | tail -1 | grep -oE '^step:[0-9]+|mean@16:[0-9.]+$' | tr '\n' ' ')
  echo "  $(basename "$log"): $(grep -oE '^.{0,0}step:[0-9]+' <<< "${last#*step:}" | head -1)step $(grep -oE 'step:[0-9]+' <<< "$last" | head -1 | cut -d: -f2), $(grep -oE 'timing_s/step:[0-9.]+' <<< "$last" | cut -d: -f2) s/step, PF $(grep -oE 'tflops/cumulative_total:[0-9.e+]+' <<< "$last" | cut -d: -f2 | awk '{printf "%.1f", $1/1000}') | last dev ${dev:-none} | errors $(grep -cE 'Traceback|OutOfMemory|Error executing' "$log")"
done
echo "== janitors"
echo "  rl:    $(tail -1 /work2/jiahaoz4/maze_extra/logs/rl_janitor.log 2>/dev/null | cut -c1-150)"
echo "  prune: $(tail -1 /work2/jiahaoz4/maze_extra/logs/prune_pretrain.log 2>/dev/null)"
echo "== disk: $(df -h /work2 | tail -1 | awk '{print $4 " free of " $2 " (" $5 ")"}')"
echo "== GPUs (index, used MiB, util; mine = processes of jiahaoz4)"
mine=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | while IFS=', ' read -r b p; do [ "$(ps -o user= -p "$p" 2>/dev/null)" = jiahaoz4 ] && echo "$b"; done | sort | uniq -c)
nvidia-smi --query-gpu=index,pci.bus_id,memory.used,utilization.gpu --format=csv,noheader,nounits | while IFS=', ' read -r i b u t; do
  c=$(grep -i "$b" <<< "$mine" | awk '{print $1}'); echo "  gpu$i: ${u} MiB ${t}% | my procs ${c:-0}"
done
