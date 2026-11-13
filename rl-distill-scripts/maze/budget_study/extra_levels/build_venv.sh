#!/bin/bash
# Build ~/TailRL/.venv per BUDGET_STUDY_RUNBOOK "Setting up on another machine" item 2 (cu128 torch works on driver 550).
set -euxo pipefail
export UV_CACHE_DIR=/work2/jiahaoz4/.cache/uv UV_PYTHON_INSTALL_DIR=/work2/jiahaoz4/.local/share/uv/python TMPDIR=/work2/jiahaoz4/tmp/maze
cd /work2/jiahaoz4/TailRL
uv python install 3.10
uv venv --python "$(uv python find --managed-python 3.10)" .venv
uv pip install --python .venv/bin/python -c .uv-constraints.txt torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -c .uv-constraints.txt -r experiments/text_maze/requirements.txt \
  transformers==4.57.6 "ray[default]==2.59.0" torchdata==0.11.0 peft==0.21.2 scipy
.venv/bin/python -c "import torch, transformers, ray, scipy, peft, torchdata; print('VENV_OK', torch.__version__, torch.version.cuda, transformers.__version__, ray.__version__)"
