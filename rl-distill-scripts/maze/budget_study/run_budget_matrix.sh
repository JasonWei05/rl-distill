#!/usr/bin/env bash
# Run every feasible Arm A cell of a budget plan (plan_budgets.py), both variants, on the given GPUs. One detached worker
# per GPU walks the shared job list and claims jobs with an atomic lock (mkdir), so workers can be added at any time by
# re-running this script with more GPU ids; DONE and FAILED jobs are skipped (delete a .FAILED marker to retry it).
#
# Usage: bash run_budget_matrix.sh <plan.json> <gpu> [gpu ...]
#   Env: VARIANTS ("opd off"), STUDENT_HF (~/tailrl_maze_runs/matched/<pair>_S0; STUDENT_BASE = ${STUDENT_HF}_base),
#        OPD_TEST_FREQ (50), OPD_VAL_N (16), OPD_BATCH (128)
# Layout: $BUDGET_DIR = ~/tailrl_maze_runs/budget/<pair>/ with jobs/all.list, jobs/<cell>-<variant>.{log,lock/,DONE,FAILED},
#         jobs/worker_gpu<g>.log, off/<cell>/, traces/<cell>/; on-policy runs are train.sh experiments with suite
#         "<cell>-opd" (checkpoints dir), their last step = the budget stop. A worker killed mid-job leaves a stale .lock
#         directory: remove it by hand before re-running.
set -euo pipefail
PLAN=${1:?plan.json}; shift; [ $# -ge 1 ] || { echo "need at least one GPU" >&2; exit 1; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY="$HOME/TailRL/.venv/bin/python"
PAIR=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['pair'])" "$PLAN")
export BUDGET_DIR=${BUDGET_DIR:-$HOME/tailrl_maze_runs/budget/$PAIR}
export STUDENT_HF=${STUDENT_HF:-$HOME/tailrl_maze_runs/matched/${PAIR}_S0}
export STUDENT_BASE=${STUDENT_BASE:-${STUDENT_HF}_base}
export OPD_TEST_FREQ=${OPD_TEST_FREQ:-200} OPD_VAL_N=${OPD_VAL_N:-16} OPD_BATCH=${OPD_BATCH:-128}  # a 16k-sample validation costs ~30 s vs 0.4 s per OPD step
export SEED=${SEED:-0}  # replicate seed for the distillation runs (train.sh --seed, off-policy trace/student seeds); use a separate BUDGET_DIR per seed
VARIANTS=${VARIANTS:-"opd off"}
mkdir -p "$BUDGET_DIR/jobs"
[ -d "$STUDENT_BASE/ckpt-0" ] || { echo "no student ckpt-0 under $STUDENT_BASE" >&2; exit 1; }

# shared job list: variant|cell|teacher_hf|D|n_traces, largest distillation budgets first so the long jobs start early
"$PY" - "$PLAN" "$VARIANTS" > "$BUDGET_DIR/jobs/all.list" <<'EOF'
import json, sys
plan = json.load(open(sys.argv[1])); variants = sys.argv[2].split()
cells = sorted((c for c in plan["cells"] if c["feasible"]), key=lambda c: -c["distill_tflops"])
for c in cells:
    for v in variants:
        print(f"{v}|{c['id']}|{c['teacher_hf']}|{c['distill_tflops']:.6f}|{c['off']['n_traces']}")
EOF
echo "$(grep -c . "$BUDGET_DIR/jobs/all.list") jobs in $BUDGET_DIR/jobs/all.list; done so far: $(ls "$BUDGET_DIR"/jobs/*.DONE 2>/dev/null | wc -l)"

for g in "$@"; do
  pidfile="$BUDGET_DIR/jobs/worker_gpu$g.pid"
  if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then echo "worker on GPU $g already running (pid $(cat "$pidfile")); not starting another"; continue; fi
  setsid nohup bash -c '
    HERE=$1; GPU=$2; LIST="$BUDGET_DIR/jobs/all.list"; echo $$ > "$BUDGET_DIR/jobs/worker_gpu$GPU.pid"
    while IFS="|" read -r v cell teacher d ntr; do
      [ -n "$v" ] || continue
      job="$BUDGET_DIR/jobs/${cell}-${v}"
      [ -f "$job.DONE" ] && continue
      [ -f "$job.FAILED" ] && { echo "skip $cell $v (FAILED marker present)"; continue; }
      mkdir "$job.lock" 2>/dev/null || continue          # claimed by another worker
      echo "$GPU $(date -Is)" > "$job.lock/owner"
      echo "$(date -Is) start $cell $v on GPU $GPU: D=$d traces=$ntr teacher=$teacher"
      if [ "$v" = opd ]; then
        MAX_TOTAL_TFLOPS=$d STEPS=1000000 SAVE_FREQ=1000000 TEST_FREQ=$OPD_TEST_FREQ VAL_N=$OPD_VAL_N BATCH=$OPD_BATCH \
          bash "$HERE/run_maze_opd.sh" "$STUDENT_BASE" "$teacher" "${cell}-opd" "$GPU" > "$job.log" 2>&1 && touch "$job.DONE" || touch "$job.FAILED"
      else
        bash "$HERE/run_offpolicy_budget.sh" "$teacher" "$STUDENT_HF" "$ntr" "$d" "$cell" "$GPU" > "$job.log" 2>&1 && touch "$job.DONE" || touch "$job.FAILED"
      fi
      rm -rf "$job.lock"
      echo "$(date -Is) end $cell $v: $([ -f "$job.DONE" ] && echo ok || echo FAILED)"
    done < "$LIST"
    echo "$(date -Is) worker GPU $GPU finished the list"
  ' _ "$HERE" "$g" >> "$BUDGET_DIR/jobs/worker_gpu$g.log" 2>&1 < /dev/null &
  echo "worker on GPU $g (pid $!) -> $BUDGET_DIR/jobs/worker_gpu$g.log"
done
