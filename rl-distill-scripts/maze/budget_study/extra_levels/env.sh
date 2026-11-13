# Source before any maze command on federer (GPUs 4-7 = A800 40GB; the root partition is full, everything lives on /work2).
set -a; . /work2/jiahaoz4/TailRL/.env; set +a
export PY=$HOME/TailRL/.venv/bin/python
export M=/work2/jiahaoz4/rl-distill/rl-distill-scripts/maze
export R=$HOME/tailrl_maze_runs/pretrain20k_rerun
export PYTHONNOUSERSITE=1
export SCREEN_MAX_SEQS=${SCREEN_MAX_SEQS:-4096}  # 8192 sequences per generate call OOMs an L checkpoint on a 40 GB A800 (4096: 28 GiB peak)
