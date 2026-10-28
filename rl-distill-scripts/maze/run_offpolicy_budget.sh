#!/usr/bin/env bash
# One off-policy distillation cell of the budget study: generate n teacher traces (1 per prompt) plus a fixed 1024-prompt
# validation slice, then one epoch of exact forward KL into the student, capped at the cell's FLOPs budget
# (teacher generation + student updates; the validation traces are monitoring and not counted).
#
# Usage: bash run_offpolicy_budget.sh <teacher-hf> <student-hf> <n-traces> <budget-tflops> <cell-id> <gpu>
#   Env: BUDGET_DIR (~/tailrl_maze_runs/budget/<pair>, output root), TRAIN_PARQUET, OFF_BATCH (128), OFF_LR (1e-4),
#        TRACE_OFFSET (0: first train row used for this cell's traces), VAL_TRACE_OFFSET (1200000), VAL_TRACES (1024)
# Outputs: $BUDGET_DIR/traces/<cell>/{train,val}.parquet(+.flops.json), $BUDGET_DIR/off/<cell>/{ckpt-N, train_metrics.jsonl, DONE}
set -euo pipefail
TEACHER=${1:?teacher HF dir}; STUDENT=${2:?student HF dir}; N=${3:?n traces}; BUDGET=${4:?budget TFLOPs}; CELL=${5:?cell id}; GPU=${6:?gpu}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY="$HOME/TailRL/.venv/bin/python"
BUDGET_DIR=${BUDGET_DIR:?set BUDGET_DIR (e.g. ~/tailrl_maze_runs/budget/p10)}
TRAIN_PARQUET=${TRAIN_PARQUET:-$HOME/tailrl_maze_data/main_parquet/train.parquet}
OFF_BATCH=${OFF_BATCH:-128}; OFF_LR=${OFF_LR:-1e-4}
TRACE_OFFSET=${TRACE_OFFSET:-0}; VAL_TRACE_OFFSET=${VAL_TRACE_OFFSET:-1200000}; VAL_TRACES=${VAL_TRACES:-1024}
SEED=${SEED:-0}   # replicate seed: teacher sampling seeds 10*SEED (train) / 10*SEED+1 (val), student init/order seed SEED
TR="$BUDGET_DIR/traces/$CELL"; OUT="$BUDGET_DIR/off/$CELL"
mkdir -p "$TR" "$OUT"
export CUDA_VISIBLE_DEVICES=$GPU
BS=$(( N < OFF_BATCH ? N : OFF_BATCH )); MB=$(( BS < 32 ? BS : 32 )); STEPS=$(( (N + BS - 1) / BS )); WARM=$(( STEPS / 10 < 100 ? STEPS / 10 : 100 ))
cd "$HERE"
[ -f "$TR/train.parquet.flops.json" ] || "$PY" gen_teacher_traces.py --teacher "$TEACHER" --prompts "$TRAIN_PARQUET" \
    --offset "$TRACE_OFFSET" --n-prompts "$N" --k 1 --batch-prompts 4096 --seed $((SEED * 10)) --out "$TR/train.parquet"
[ -f "$TR/val.parquet.flops.json" ] || "$PY" gen_teacher_traces.py --teacher "$TEACHER" --prompts "$TRAIN_PARQUET" \
    --offset "$VAL_TRACE_OFFSET" --n-prompts "$VAL_TRACES" --k 1 --batch-prompts 4096 --seed $((SEED * 10 + 1)) --out "$TR/val.parquet"
"$PY" distill_offpolicy.py --student "$STUDENT" --train "$TR/train.parquet" --val "$TR/val.parquet" --out "$OUT" \
    --steps "$STEPS" --batch-size "$BS" --micro-batch-size "$MB" --warmup-steps "$WARM" --lr "$OFF_LR" \
    --val-every 50 --save-every 1000000 --tflops-budget "$BUDGET" --seed "$SEED"
echo "$(date -Is) steps=$STEPS batch=$BS traces=$N budget=$BUDGET" > "$OUT/DONE"
