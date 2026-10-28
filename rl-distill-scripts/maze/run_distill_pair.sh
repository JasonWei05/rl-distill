#!/usr/bin/env bash
# Arm A step 2, both variants, from ONE teacher checkpoint of the L0 RL run: on-policy PG-k1 (run_maze_opd.sh) on one
# GPU and off-policy exact forward KL (gen_teacher_traces.py -> distill_offpolicy.py) on another, both detached. The
# teacher's cumulative RL FLOPs up to that checkpoint (its flops_ledger.json) are copied next to the outputs so the
# Arm A curves can be offset by the teacher's cost.
#
# Usage: bash rl-distill-scripts/maze/run_distill_pair.sh <teacher-rl-ckpt-root> <teacher-step> <gpu-opd> <gpu-offpolicy> [tag]
#   <teacher-rl-ckpt-root>  e.g. ~/tailrl_maze_runs/checkpoints/tailrl-text-maze/textmaze_primary-l0_grpo_binary_ckpt0_bs256_G16_1gpu_1_seed0
#   tag defaults to l0step<teacher-step>
#   Env knobs: STUDENT_HF (~/tailrl_maze_runs/matched/primary_S0; STUDENT_BASE = ${STUDENT_HF}_base holds ckpt-0),
#              OPD_STEPS (400), OPD_BATCH (128), OPD_LR (1e-4),
#              TRACE_PROMPTS (65536), TRACE_K (1), VAL_TRACE_PROMPTS (2048), OFF_STEPS (512), OFF_BATCH (128), OFF_LR (1e-4)
#   Defaults give both variants ~50-65k student sequences (OPD 400 x 128 on-policy samples; off-policy one epoch over
#   65536 teacher traces), comparable to the ~16 RL steps' worth of sequences at 4096 per step.
# Outputs: ~/tailrl_maze_runs/distill/<tag>/{meta.json, teacher_rl_flops.json, opd.log, offpolicy.log, offpolicy/{ckpt-N,train_metrics.jsonl}}
#          traces: ~/tailrl_maze_runs/traces/<tag>/{train,val}.parquet (+ .flops.json with the teacher's generation FLOPs)
#          OPD checkpoints: ~/tailrl_maze_runs/checkpoints/tailrl-text-maze/textmaze_opd-<tag>_opd_token_zero_ckpt0_bs<OPD_BATCH>_G1_1gpu_<gpu>_seed0/
#          (same tag again resumes the OPD run through train.sh's auto resume; the off-policy chain skips finished stages)
set -euo pipefail
ROOT=${1:?teacher RL checkpoint root}; STEP=${2:?teacher step}; GPU_OPD=${3:?gpu for OPD}; GPU_OFF=${4:?gpu for off-policy}
TAG=${5:-l0step${STEP}}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY="$HOME/TailRL/.venv/bin/python"
RUNS="$HOME/tailrl_maze_runs"
STUDENT_HF=${STUDENT_HF:-$RUNS/matched/primary_S0}
STUDENT_BASE=${STUDENT_BASE:-${STUDENT_HF}_base}
TRAIN_PARQUET=${TRAIN_PARQUET:-$HOME/tailrl_maze_data/main_parquet/train.parquet}
OPD_STEPS=${OPD_STEPS:-400}; OPD_BATCH=${OPD_BATCH:-128}; OPD_LR=${OPD_LR:-1e-4}
TRACE_PROMPTS=${TRACE_PROMPTS:-65536}; TRACE_K=${TRACE_K:-1}; VAL_TRACE_PROMPTS=${VAL_TRACE_PROMPTS:-2048}
OFF_STEPS=${OFF_STEPS:-512}; OFF_BATCH=${OFF_BATCH:-128}; OFF_LR=${OFF_LR:-1e-4}
BP=$(( 8192 / TRACE_K )); [ "$BP" -gt 4096 ] && BP=4096   # <= 8192 sequences per generate call

TEACHER_HF="$ROOT/global_step_$STEP/actor/huggingface"
[ -f "$TEACHER_HF/config.json" ] || { echo "no teacher HF export at $TEACHER_HF" >&2; exit 1; }
[ -d "$STUDENT_BASE/ckpt-0" ] || { echo "no student ckpt-0 under $STUDENT_BASE" >&2; exit 1; }
OUT="$RUNS/distill/$TAG"; TR="$RUNS/traces/$TAG"
mkdir -p "$OUT" "$TR"
cp "$ROOT/global_step_$STEP/flops_ledger.json" "$OUT/teacher_rl_flops.json" 2>/dev/null || echo "warning: no flops_ledger.json for teacher step $STEP" >&2
cat > "$OUT/meta.json" <<EOF
{"teacher_root": "$ROOT", "teacher_step": $STEP, "teacher_hf": "$TEACHER_HF", "student_hf": "$STUDENT_HF", "launched": "$(date -Is)",
 "opd": {"steps": $OPD_STEPS, "batch": $OPD_BATCH, "lr": "$OPD_LR", "gpu": $GPU_OPD, "suite": "opd-$TAG"},
 "offpolicy": {"trace_prompts": $TRACE_PROMPTS, "trace_k": $TRACE_K, "val_trace_prompts": $VAL_TRACE_PROMPTS, "steps": $OFF_STEPS, "batch": $OFF_BATCH, "lr": "$OFF_LR", "gpu": $GPU_OFF}}
EOF

# on-policy (PG-k1): the teacher is the reference policy inside the RL loop
STEPS=$OPD_STEPS BATCH=$OPD_BATCH ACTOR_LR=$OPD_LR \
  setsid nohup bash "$HERE/run_maze_opd.sh" "$STUDENT_BASE" "$TEACHER_HF" "opd-$TAG" "$GPU_OPD" > "$OUT/opd.log" 2>&1 < /dev/null &
echo "OPD      -> GPU $GPU_OPD, log $OUT/opd.log (pid $!)"

# off-policy: teacher traces (train slice, then a disjoint val slice), then the exact forward-KL trainer
setsid nohup bash -c "
set -euo pipefail
export CUDA_VISIBLE_DEVICES=$GPU_OFF
cd '$HERE'
[ -f '$TR/train.parquet.flops.json' ] || '$PY' gen_teacher_traces.py --teacher '$TEACHER_HF' --prompts '$TRAIN_PARQUET' \
    --n-prompts $TRACE_PROMPTS --k $TRACE_K --batch-prompts $BP --seed 0 --out '$TR/train.parquet'
[ -f '$TR/val.parquet.flops.json' ] || '$PY' gen_teacher_traces.py --teacher '$TEACHER_HF' --prompts '$TRAIN_PARQUET' \
    --offset $TRACE_PROMPTS --n-prompts $VAL_TRACE_PROMPTS --k $TRACE_K --batch-prompts $BP --seed 1 --out '$TR/val.parquet'
'$PY' distill_offpolicy.py --student '$STUDENT_HF' --train '$TR/train.parquet' --val '$TR/val.parquet' \
    --out '$OUT/offpolicy' --steps $OFF_STEPS --batch-size $OFF_BATCH --lr $OFF_LR
echo DONE > '$OUT/offpolicy.DONE'
" > "$OUT/offpolicy.log" 2>&1 < /dev/null &
echo "off-pol  -> GPU $GPU_OFF, log $OUT/offpolicy.log (pid $!)"
echo "outputs: $OUT   traces: $TR"
