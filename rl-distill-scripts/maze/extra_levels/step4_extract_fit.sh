#!/usr/bin/env bash
# Step 4 of "Extra start levels for the maze RL-curve law" on federer: assemble the S RL logs (an extended run's earlier
# segments were rotated to rl/<tag>_S0_rl.log.part<K> by rl_launch.sh), extract the curves, refit the S family, and stage
# the files the runbook asks to send back under /work2/jiahaoz4/maze_extra/extra/ (uploaded to the Hub repo's extra/).
#   PAIRS="p3 p03 [p30]" REPLICATES="p3s1=p3:1 p03s1=p03:1 p1s1=p1:1 [p30s1=p30:1]" bash step4_extract_fit.sh
set -euo pipefail
source /work2/jiahaoz4/maze_extra/env.sh
RUNS=$HOME/tailrl_maze_runs
X=/work2/jiahaoz4/maze_extra/extract_root   # rl/ = assembled logs, matched -> the real matched dir
OUT=/work2/jiahaoz4/maze_extra/extra
PAIRS=${PAIRS:-"p3 p03"}
REPLICATES=${REPLICATES:-"p3s1=p3:1 p03s1=p03:1 p1s1=p1:1"}
rm -rf "$X" && mkdir -p "$X/rl" "$OUT/rl_logs" "$OUT/select" "$OUT/screen"
ln -s "$RUNS/matched" "$X/matched"
tags="$PAIRS"; for r in $REPLICATES; do tags="$tags ${r%%=*}"; done
for t in $tags; do
  log=$RUNS/rl/${t}_S0_rl.log
  parts=$(ls -v "$log".part* 2>/dev/null || true)   # part1 < part2 < ... < part10, then the live log
  if [ -n "$parts" ]; then cat $parts "$log" > "$X/rl/${t}_S0_rl.log"; else cp "$log" "$X/rl/${t}_S0_rl.log"; fi
  cp "$X/rl/${t}_S0_rl.log" "$OUT/rl_logs/"
done
cd /work2/jiahaoz4/rl-distill
# shellcheck disable=SC2086
$PY $M/rl_curve_family.py extract --pairs $PAIRS --replicates $REPLICATES --runs "$X" --out "$OUT/rl_curves_extra.json"
$PY $M/rl_curve_family.py fit --data $M/data/rl_curves_v1.json "$OUT/rl_curves_extra.json" --model S0 --loo \
  --json-out "$OUT/rl_curve_family_S0.json" | tee "$OUT/rl_curve_family_S0.txt"
$PY $M/rl_curve_family.py fit --data $M/data/rl_curves_v1.json "$OUT/rl_curves_extra.json" --model S0 --loo --form SIG \
  --json-out "$OUT/rl_curve_family_S0_SIG.json" | tee "$OUT/rl_curve_family_S0_SIG.txt"
cp "$RUNS/matched/manifest.json" "$OUT/matched_manifest.json"
for run in S_ds0_is4 L_ds0_is0_rep; do
  for f in "$R/$run"/select_goal_reached_*pct.json "$R/$run"/select_goal_reached_*pct.log; do [ -e "$f" ] && cp "$f" "$OUT/select/${run}__$(basename "$f")"; done
  for f in "$R/$run"/screen_stride50_k16*.json; do [ -e "$f" ] && cp "$f" "$OUT/screen/${run}__$(basename "$f")"; done
done
ls -la "$OUT" "$OUT"/*/
