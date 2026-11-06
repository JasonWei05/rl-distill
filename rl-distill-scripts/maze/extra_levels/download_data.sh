#!/bin/bash
# The five downloads of TailRL scripts/download_dataset.sh, run with ~/.local/bin/hf so they can overlap the venv build.
set -euxo pipefail
export HF_HOME=/work2/jiahaoz4/.cache/huggingface HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TMPDIR=/work2/jiahaoz4/tmp/maze
OUT=/work2/jiahaoz4/tailrl_maze_data; REPO=max-rl/maze_17x17_diverse_1.3m
mkdir -p $OUT/sft_data
hf download $REPO main_1.3M.jsonl --repo-type dataset --local-dir $OUT
hf download $REPO main_1.3M.jsonl.meta.json --repo-type dataset --local-dir $OUT
hf download $REPO train_random.json --repo-type dataset --local-dir $OUT/sft_data
hf download $REPO train_shortest.json --repo-type dataset --local-dir $OUT/sft_data
hf download $REPO test.json --repo-type dataset --local-dir $OUT/sft_data
echo DOWNLOAD_OK; du -sh $OUT
