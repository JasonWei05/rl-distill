#!/usr/bin/env bash
# GO/NO-GO gate on a local 4xGPU box (1 training step). PASS requires BOTH:
#   - step-0 validation/accuracy in [0.045, 0.075] (full 3200-sample val, mean@16)
#   - step-1 train/probs_ratio AND train/probs_ratio_clamped ~= 1.0 (+-0.01) in wandb
#     (val alone is NOT sufficient — the corrupted-training failure mode passes the
#      val band while probs_ratio_clamped pins at 0.80; PROGRESS_LOG 2026-07-30)
set -uo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_HERE}/env.sh"
_REPO_ROOT="$(cd "${_HERE}/../../.." && pwd)"
cd "${_REPO_ROOT}" || exit 1
if [ -f .env ]; then set -a; source .env; set +a; fi
export NEMORL_FORCE_LOCAL_RAY=1

GEMMA4_VARIANT="${GEMMA4_VARIANT:-e2b}"
"${NEMO_RL_DRIVER_VENV}/bin/python" "${REPRO_DIR}/run_grpo_repro.py" \
  --config "${REPRO_DIR}/config/dapo_gemma4_${GEMMA4_VARIANT}_pt_repro.yaml" \
  cluster.gpus_per_node="${GPUS_PER_NODE:-4}" \
  grpo.max_num_steps=1 \
  checkpointing.enabled=false \
  policy.tokenizer.chat_template="${_REPO_ROOT}/rl-distill-scripts/data/gemma3_it_fewshot_math.jinja" \
  logger.wandb.name="nemorl-dapo-gemma4-${GEMMA4_VARIANT}-pt-ds4of4strict-s42-8k-GATE-${RUN_SUFFIX:-local4g}"
rc=$?
echo "GATE_RUN_RC=${rc}"
exit "${rc}"
