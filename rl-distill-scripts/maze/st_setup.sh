# Sourced by the maze ScaleTrain run-files (run_maze_screen_st.sh, run_maze_select_st.sh): refresh the shipped code, then
# build TailRL @ 5682c6a + maze/tailrl_text_maze.patch + maze/tailrl_rl_tis.patch and its venv under /tmp/maze (BUDGET_STUDY_RUNBOOK "Setting up on
# another machine"). Defines PROJECT_ROOT, M (this directory), T (TailRL checkout) and PY (its venv python).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${PROJECT_ROOT}"
export PATH="${PROJECT_ROOT}/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
if [ -n "${CODE_S3_URI:-}" ]; then
  aws s3 cp --only-show-errors "${CODE_S3_URI}" /tmp/rl-distill-code.tar.gz
  tar -xzf /tmp/rl-distill-code.tar.gz --unlink-first --recursive-unlink -C "${PROJECT_ROOT}"
  echo "CODE_REFRESHED $(sha256sum /tmp/rl-distill-code.tar.gz | cut -c1-16)"
fi
if [ -f .env ]; then set -a; source .env; set +a; fi
export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
export HF_HOME=/tmp/hf_cache HF_HUB_ENABLE_HF_TRANSFER=1 UV_CACHE_DIR=/tmp/uv_cache UV_HTTP_TIMEOUT=600 PYTHONNOUSERSITE=1
M="${PROJECT_ROOT}/rl-distill-scripts/maze"; T=/tmp/maze/TailRL; PY="${T}/.venv/bin/python"
mkdir -p /tmp/maze
if [ ! -d "${T}/.git" ]; then
  git clone -q https://github.com/Zanette-Labs/TailRL "${T}" && git -C "${T}" checkout -q 5682c6a && git -C "${T}" apply "${M}/tailrl_text_maze.patch" \
    && git -C "${T}" apply "${M}/tailrl_rl_tis.patch"   # RL recipe: rollout log-probs, token-level TIS, early stop + best_hf
fi
if [ ! -x "${PY}" ]; then
  (cd "${T}" && uv python install 3.10 && uv venv --python "$(uv python find --managed-python 3.10)" .venv \
    && uv pip install --python .venv/bin/python -c .uv-constraints.txt torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install --python .venv/bin/python -c .uv-constraints.txt -r experiments/text_maze/requirements.txt \
         transformers==4.57.6 "ray[default]==2.59.0" torchdata==0.11.0 peft==0.21.2 scipy hf_transfer)
fi
"${PY}" -c "import torch, transformers; assert torch.cuda.is_available(); print('MAZE_ENV_OK', torch.__version__, transformers.__version__)"
