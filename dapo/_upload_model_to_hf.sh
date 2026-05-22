#!/usr/bin/env bash
# Upload the final (step 200) distilled ckpt on this node to the root of its HF repo.
# Usage: bash _upload_model_to_hf.sh <SIZE>    where SIZE is 4b or 12b
set -euo pipefail
SIZE="${1:?SIZE 4b or 12b}"

cd /mlx_devbox/users/jason.wei/playground/rl-distill
source .venv/bin/activate
set -a; source .env; set +a

case "${SIZE}" in
  4b)
    LOCAL_DIR=$(ls -d /tmp/verl/ckpts/distill/offpolicy-gemma3-4b-*/global_step_200/huggingface 2>/dev/null | tail -1)
    export REPO_ID=JWei05/gemma3-4b-it-off-policy-distilled-from-dapo27b
    export BASE_MODEL=google/gemma-3-4b-it
    ;;
  12b)
    LOCAL_DIR=$(ls -d /tmp/verl/ckpts/distill/offpolicy-gemma3-12b-*/global_step_200/huggingface 2>/dev/null | tail -1)
    export REPO_ID=JWei05/gemma3-12b-it-off-policy-distilled-from-dapo27b
    export BASE_MODEL=google/gemma-3-12b-it
    ;;
  *) echo "unknown size: ${SIZE}"; exit 1 ;;
esac

if [ -z "${LOCAL_DIR}" ] || [ ! -f "${LOCAL_DIR}/config.json" ]; then
    echo "ERROR: no step_200 huggingface/ dir found for ${SIZE}"
    echo "searched: /tmp/verl/ckpts/distill/offpolicy-gemma3-${SIZE}-*/global_step_200/huggingface"
    exit 1
fi
export LOCAL_DIR
echo "LOCAL_DIR=${LOCAL_DIR}"
echo "REPO_ID=${REPO_ID}"
echo ""
python3 dapo/_upload_model_root.py
