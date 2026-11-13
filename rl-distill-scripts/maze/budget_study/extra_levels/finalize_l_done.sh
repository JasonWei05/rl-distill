#!/usr/bin/env bash
# L runs end at the shared 266 PF ceiling (1,250 steps) with no extension rule: once watchdog.sh has marked one done
# (runs/<suite>.done), put FINALIZE in its run dir so the janitor uploads the rest and the final full checkpoint.
#   setsid nohup bash finalize_l_done.sh >> logs/finalize_l.log 2>&1 &
set -uo pipefail
D=/work2/jiahaoz4/maze_extra; CK=$HOME/tailrl_maze_runs/checkpoints/tailrl-text-maze
while true; do
  left=0
  for e in "$D"/runs/*-l0.env; do
    s=$(basename "$e" .env)
    if [ -e "$D/runs/$s.done" ]; then
      d=$(find "$CK" -maxdepth 1 -type d -name "textmaze_${s}_grpo_binary_*" | head -1)
      [ -n "$d" ] && [ ! -e "$d/FINALIZE" ] && { touch "$d/FINALIZE"; echo "$(date -u '+%m-%d %H:%M:%S') FINALIZE $s ($d)"; }
    else
      left=$((left + 1))
    fi
  done
  [ "$left" -eq 0 ] && { echo "$(date -u '+%m-%d %H:%M:%S') all L runs finalized"; break; }
  sleep 120
done
