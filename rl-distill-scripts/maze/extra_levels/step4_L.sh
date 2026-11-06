#!/usr/bin/env bash
# Step 4 for L once its runs end: one extraction holding the S and the L curves (so extra/rl_curves_extra.json carries both),
# the L refit (PAPER + SIG) and per-curve C_mid; stages the files under maze_extra/extra/ for the Hub upload.
#   bash step4_L.sh
set -euo pipefail
source /work2/jiahaoz4/maze_extra/env.sh
RUNS=$HOME/tailrl_maze_runs
X=${X:-/work2/jiahaoz4/maze_extra/extract_root}
OUT=${OUT:-/work2/jiahaoz4/maze_extra/extra}   # X / OUT overridable for a dry run
PAIRS="p3 p03 p5"
REPLICATES="p3s1=p3:1 p03s1=p03:1 p5s1=p5:1 p1s1=p1:1 p01s1=p01:1"
rm -rf "$X" && mkdir -p "$X/rl" "$OUT/rl_logs"
ln -s "$RUNS/matched" "$X/matched"
tags="$PAIRS"; for r in $REPLICATES; do tags="$tags ${r%%=*}"; done
for t in $tags; do
  for side in S0 L0; do
    log=$RUNS/rl/${t}_${side}_rl.log
    [ -e "$log" ] || continue
    parts=$(ls -v "$log".part* 2>/dev/null || true)   # part1 < part2 < ... then the live log
    if [ -n "$parts" ]; then cat $parts "$log" > "$X/rl/${t}_${side}_rl.log"; else cp "$log" "$X/rl/${t}_${side}_rl.log"; fi
    cp "$X/rl/${t}_${side}_rl.log" "$OUT/rl_logs/"
  done
done
cd /work2/jiahaoz4/rl-distill
# shellcheck disable=SC2086
$PY $M/rl_curve_family.py extract --pairs $PAIRS --replicates $REPLICATES --runs "$X" --out "$OUT/rl_curves_extra.json"
for model in S0 L0; do
  $PY $M/rl_curve_family.py fit --data $M/data/rl_curves_v1.json "$OUT/rl_curves_extra.json" --model $model --loo \
    --json-out "$OUT/rl_curve_family_$model.json" | tee "$OUT/rl_curve_family_$model.txt"
  $PY $M/rl_curve_family.py fit --data $M/data/rl_curves_v1.json "$OUT/rl_curves_extra.json" --model $model --loo --form SIG \
    --json-out "$OUT/rl_curve_family_${model}_SIG.json" | tee "$OUT/rl_curve_family_${model}_SIG.txt"
  $PY /work2/jiahaoz4/maze_extra/cmid_per_curve.py $M/data/rl_curves_v1.json "$OUT/rl_curves_extra.json" \
    --fit "$OUT/rl_curve_family_$model.json" --model $model --json-out "$OUT/rl_cmid_per_curve_$model.json"
done
cp "$RUNS/matched/manifest.json" "$OUT/matched_manifest.json"
ls -la "$OUT" "$OUT/rl_logs"
