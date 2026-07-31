#!/usr/bin/env bash
# Full DAPO repro training run on a local 4xGPU box. Launch only after
# run_gate.sh passes BOTH gate criteria (val band + step-1 probs_ratio ~= 1).
# MAX_STEPS caps the run; omit for open-ended (verl ran total_epochs=100).
set -uo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_HERE}/env.sh"
bash "${_HERE}/apply_nemo_rl_patches.sh" || exit 1
_REPO_ROOT="$(cd "${_HERE}/../../.." && pwd)"
cd "${_REPO_ROOT}" || exit 1
if [ -f .env ]; then set -a; source .env; set +a; fi
export NEMORL_FORCE_LOCAL_RAY=1

GEMMA4_VARIANT="${GEMMA4_VARIANT:-e2b}"
RUN_SUFFIX="${RUN_SUFFIX:-local4g-v2}"
RUN_NAME="nemorl-dapo-gemma4-${GEMMA4_VARIANT}-pt-DeepScaleR-4of4strict-seed42-8k-${RUN_SUFFIX}"
EXTRA_ARGS=()
if [ -n "${MAX_STEPS:-}" ]; then EXTRA_ARGS+=("grpo.max_num_steps=${MAX_STEPS}"); fi
"${NEMO_RL_DRIVER_VENV}/bin/python" "${REPRO_DIR}/run_grpo_repro.py" \
  --config "${REPRO_DIR}/config/dapo_gemma4_${GEMMA4_VARIANT}_pt_repro.yaml" \
  cluster.gpus_per_node="${GPUS_PER_NODE:-4}" \
  policy.tokenizer.chat_template="${_REPO_ROOT}/rl-distill-scripts/data/gemma3_it_fewshot_math.jinja" \
  logger.wandb.name="${RUN_NAME}" \
  logger.log_dir="/tmp/verl/logs/${RUN_NAME}" \
  checkpointing.checkpoint_dir="/tmp/verl/ckpts/${RUN_NAME}" \
  "${EXTRA_ARGS[@]}"
rc=$?
echo "TRAIN_RUN_RC=${rc}"
exit "${rc}"
