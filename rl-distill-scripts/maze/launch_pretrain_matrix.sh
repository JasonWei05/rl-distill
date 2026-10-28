#!/usr/bin/env bash
# Launch the matched-initialization pretraining matrix for the text-maze study (step 3 of the design doc).
#
# 8 runs, all with data_seed 0 (identical batch order for every size): S and L with init_seed 0/1/2
# (noise floor) plus one repeat of init_seed 0 per size (reproducibility check). Four runs share a GPU:
# a run is ~6000 steps of a 4M-13M model, minutes of compute, so packing is free; the screening on the
# other GPUs is the real work (see screen_watch.py).
#
# No generative evaluation inside the training script (its loop generates prompt by prompt on a random
# subset, slow and noisy); only validation loss every EVAL_STEPS. Checkpoints every SAVE_STEPS from step 0
# to MAX_STEPS; capability is measured post hoc on the fixed dev set.
#
# Usage: bash rl-distill-scripts/maze/launch_pretrain_matrix.sh [--dry-run]
#   RUNS_ROOT (default ~/tailrl_maze_runs/pretrain), TRAIN_GPUS (default "0 1"), MAX_STEPS (6000),
#   SAVE_STEPS (10), EVAL_STEPS (50) can be overridden in the environment.
set -euo pipefail

TAILRL=${TAILRL:-$HOME/TailRL}
EXP=$TAILRL/experiments/text_maze
PY=$TAILRL/.venv/bin/python
DATA=${TAILRL_MAZE_DATA_DIR:-$HOME/tailrl_maze_data}
RUNS_ROOT=${RUNS_ROOT:-$HOME/tailrl_maze_runs/pretrain}
MAX_STEPS=${MAX_STEPS:-6000}
SAVE_STEPS=${SAVE_STEPS:-10}
EVAL_STEPS=${EVAL_STEPS:-50}
DATA_SEED=${DATA_SEED:-0}
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

read -r -a GPUS <<< "${TRAIN_GPUS:-0 1}"

# name  hidden layers heads kv ffn
S_CFG="256 4 4 2 1024"
L_CFG="384 6 6 3 1536"

# run name -> size init_seed (the _rep run repeats init_seed 0 bit-for-bit in configuration)
RUNS=(
  "S_ds${DATA_SEED}_is0 S 0"
  "S_ds${DATA_SEED}_is1 S 1"
  "S_ds${DATA_SEED}_is2 S 2"
  "S_ds${DATA_SEED}_is0_rep S 0"
  "L_ds${DATA_SEED}_is0 L 0"
  "L_ds${DATA_SEED}_is1 L 1"
  "L_ds${DATA_SEED}_is2 L 2"
  "L_ds${DATA_SEED}_is0_rep L 0"
)

mkdir -p "$RUNS_ROOT"
i=0
for spec in "${RUNS[@]}"; do
  read -r name size init_seed <<< "$spec"
  if [ "$size" = S ]; then read -r hid lay heads kv ffn <<< "$S_CFG"; else read -r hid lay heads kv ffn <<< "$L_CFG"; fi
  gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
  out=$RUNS_ROOT/$name
  cmd=("$PY" "$EXP/scripts/sft_pretrain.py"
    --train_data "$DATA/sft_data/train_random.json" --val_data "$DATA/sft_data/test.json"
    --output_dir "$out" --experiment_name "$name"
    --hidden_size "$hid" --num_hidden_layers "$lay" --num_attention_heads "$heads" --num_key_value_heads "$kv" --intermediate_size "$ffn"
    --learning_rate 5e-4 --batch_size 32 --micro_batch_size 8 --lr_scheduler cosine
    --max_steps "$MAX_STEPS" --save_steps "$SAVE_STEPS" --eval_steps "$EVAL_STEPS"
    --seed "$DATA_SEED" --data_seed "$DATA_SEED" --init_seed "$init_seed")
  echo "[gpu $gpu] $name: ${cmd[*]}"
  if [ "$DRY_RUN" = 0 ]; then
    if [ -f "$out/DONE" ]; then echo "  [skip] $out/DONE exists"; i=$((i+1)); continue; fi
    mkdir -p "$out"
    ( cd "$EXP" && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$EXP PYTHONNOUSERSITE=1 setsid nohup bash -c \
        "$(printf '%q ' "${cmd[@]}") > '$out/train.log' 2>&1 && touch '$out/DONE' || touch '$out/FAILED'" \
        > /dev/null 2>&1 < /dev/null & )
    sleep 3
  fi
  i=$((i+1))
done
echo "launched ${#RUNS[@]} runs under $RUNS_ROOT (DRY_RUN=$DRY_RUN); logs: <run>/train.log; markers: DONE / FAILED"
