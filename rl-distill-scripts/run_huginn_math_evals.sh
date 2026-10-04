#!/usr/bin/env bash
# Huginn-0125 math evals on the 3 DeepScaleR band sets, sharded over local GPUs (one process per GPU walks the
# config list in order, so every config finishes on all shards before the next starts). Results:
# /tmp/huginn_eval/runs/<config>/<band>.shard<g>of<N>.jsonl ; grade with eval_huginn_math.py --grade.
# Standard Huginn inference = greedy, chat template + math system prompt; we add the boxed-answer instruction so the
# repo's boxed-only grader applies. Batch size shrinks with num_steps (KV per token = (4 + 4r) x 20.6 KiB).
set -u
cd /mnt/efs/jasonwei/rl-distill
GPUS=(${GPUS:-0 1 2 3 4 5 6 7}); N=${#GPUS[@]}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/tmp/huginn_eval/hf_home
# name num_steps temperature batch
CONFIGS=(
  "greedy_box_r32 32 0 8"
  "greedy_box_r4 4 0 48"
  "greedy_box_r8 8 0 32"
  "greedy_box_r16 16 0 16"
  "temp1_box_r32 32 1.0 8"
  "greedy_box_r64 64 0 4"
)
for g in $(seq 0 $((N - 1))); do
  (
    for cfg in "${CONFIGS[@]}"; do
      read -r name steps temp batch <<<"$cfg"
      mkdir -p /tmp/huginn_eval/runs/$name/logs
      CUDA_VISIBLE_DEVICES=${GPUS[$g]} .venv/bin/python rl-distill-scripts/eval_huginn_math.py \
        --out_dir /tmp/huginn_eval/runs/$name --shard $g/$N --samples_per_question 1 --temperature $temp \
        --num_steps $steps --prompt boxinstr --batch_size $batch --max_new_tokens 1024 \
        >> /tmp/huginn_eval/runs/$name/logs/shard$g.log 2>&1
      rc=$?  # capture before $(date) resets it
      echo "$(date -u +%FT%TZ) SHARD_DONE $name $g rc=$rc" >> /tmp/huginn_eval/runs/progress.log
    done
  ) &
done
wait
echo "$(date -u +%FT%TZ) ALL_DONE" >> /tmp/huginn_eval/runs/progress.log
