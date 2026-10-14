#!/usr/bin/env bash
# Re-evaluate RL checkpoints whose logged val mean@16 is within 1 SE (of a two-eval difference) of the logged best, with the
# run_qwen3_band_val_evals_local.sh protocol, to pick the best checkpoint without the max-over-many-evals bias.
# The logged-best checkpoints themselves are already evaluated there (rl_<size>_<band>_s<best>).
#   SIZE=4b GPUS=1 setsid bash rl-distill-scripts/run_qwen3_rl_ckpt_candidates_local.sh > <log> 2>&1 &     # SIZE: 4b | 1.7b
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SIZE="${SIZE:?4b or 1.7b}"; TAG_SIZE=${SIZE/./p}
case ${SIZE} in
  4b) declare -A STEPS=([medium]="530 540" [hard]="440 490 510 550" [very_hard]="520 540 560 570 610") ;;
  1.7b) declare -A STEPS=([medium]="530 560 580" [hard]="430 540 570 600" [very_hard]="820 960 970") ;;
  *) echo "FATAL: SIZE=${SIZE}" >&2; exit 2 ;;
esac
declare -A SHORT=([medium]=med [hard]=hard [very_hard]=vh)
for band in medium hard very_hard; do
  specs=""
  for s in ${STEPS[${band}]}; do
    specs+="rl_${TAG_SIZE}_${SHORT[${band}]}_s${s}|s3:s3://scale-ml/genai/rl-distill/qwen3-${SIZE}-base-rl-full-checkpoints/qwen30b-${band}-es5/global_step_${s}/actor/huggingface "
  done
  BAND=${band} MODEL_SPECS="${specs}" bash rl-distill-scripts/run_qwen3_band_val_evals_local.sh
done
echo "RL_CKPT_CANDIDATES_DONE ${SIZE} $(date -u +%FT%TZ)"
