#!/usr/bin/env bash
# Periodically (default every 15 min) back-fill Hub step folders that the trainer's async HFPusher
# failed to upload (a 10 GB upload can exhaust its retries on transient Hub timeouts; on 2026-09-03
# hard's step_000040 was lost that way), and prune the Hub to the newest GAP_FILLER_KEEP local steps.
#   nohup bash rl-distill-scripts/run_hf_hub_gap_filler.sh >> ~/verl/logs/g4_e2b_hf_gap_filler.log 2>&1 &
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${PROJECT_ROOT}"
if [ -f .env ]; then set -a; source .env; set +a; fi
VENV="${VENV:-${PROJECT_ROOT}/.venv-gemma4}"
while true; do
  echo "[$(date +%F' '%T)] gap filler pass"
  "${VENV}/bin/python" rl-distill-scripts/hf_hub_gap_filler.py || echo "[$(date +%F' '%T)] gap filler pass failed rc=$?"
  sleep "${GAP_FILLER_INTERVAL_SECONDS:-900}"
done
