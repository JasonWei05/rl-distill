#!/usr/bin/env bash
# Print the indices of GPUs that are empty right now: no compute process of anyone and < 1 GiB used.
# Usage: free_gpus.sh [min_total_MiB]   (e.g. 70000 to list only the 80 GB cards)
set -euo pipefail
MIN_TOTAL=${1:-0}
busy=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | sort -u)
nvidia-smi --query-gpu=index,pci.bus_id,memory.used,memory.total --format=csv,noheader,nounits |
while IFS=', ' read -r idx bus used total; do
  if [ "$used" -lt 1024 ] && [ "$total" -ge "$MIN_TOTAL" ] && ! grep -qi "$bus" <<< "$busy"; then echo "$idx"; fi
done
