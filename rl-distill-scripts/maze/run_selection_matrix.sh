#!/usr/bin/env bash
# Run select_matched_init.py for every pretraining run that crossed the target, one process per run,
# round-robin over the given GPUs, detached. Logs to <run>/select.log; results in <run>/select_<metric>.json.
#
# Usage: bash rl-distill-scripts/maze/run_selection_matrix.sh [RUNS_ROOT] [GPUS]
#   RUNS_ROOT default ~/tailrl_maze_runs/pretrain20k; GPUS default "2 3 4 5 6 7"
#   Env: TAG (stride50_k16), METRIC (goal_reached), TARGET (0.10), CENTER_STEP (20), HALF_WINDOW (60),
#        K (64), FINAL_K (1024), N_FINAL (2), MARGIN (100), MARGIN_BELOW (250)
set -euo pipefail
RUNS_ROOT=${1:-$HOME/tailrl_maze_runs/pretrain20k}
read -r -a GPUS <<< "${2:-2 3 4 5 6 7}"
PY=$HOME/TailRL/.venv/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:-stride50_k16}; METRIC=${METRIC:-goal_reached}; TARGET=${TARGET:-0.10}
i=0
for run in "$RUNS_ROOT"/*_ds0_*; do
  [ -f "$run/screen_$TAG.json" ] || { echo "[skip] $(basename "$run"): no screen_$TAG.json"; continue; }
  if ! $PY "$HERE/find_crossing.py" --runs "$run" --tag "$TAG" --metric "$METRIC" --target "$TARGET" | grep -q "\["; then
    echo "[skip] $(basename "$run"): not crossed"; continue
  fi
  gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
  echo "[gpu $gpu] $(basename "$run")"
  ( CUDA_VISIBLE_DEVICES=$gpu setsid nohup $PY "$HERE/select_matched_init.py" --run "$run" --tag "$TAG" --metric "$METRIC" --target "$TARGET" \
      --center-step "${CENTER_STEP:-20}" --half-window "${HALF_WINDOW:-60}" --k "${K:-64}" --final-k "${FINAL_K:-1024}" --n-final "${N_FINAL:-2}" \
      --margin "${MARGIN:-100}" --margin-below "${MARGIN_BELOW:-250}" > "$run/select.log" 2>&1 < /dev/null & )
  i=$((i+1))
done
echo "launched $i selection jobs"
