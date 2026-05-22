#!/usr/bin/env bash
set -euo pipefail

cd /mlx_devbox/users/jason.wei/playground/rl-distill
source .venv/bin/activate

for g in 0 1 2 3 4 5 6 7; do
  env CUDA_VISIBLE_DEVICES="${g}" python3 rl-distill-scripts/data/check_cuda_slots.py
done
