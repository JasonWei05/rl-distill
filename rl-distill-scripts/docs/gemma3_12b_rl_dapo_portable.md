# Gemma 3 12B DAPO RL On Another Cluster

This is the portable recipe for continuing RL from:

```text
JWei05/DAPO-Gemma3-12B-PT-TopK128Distill-From-Gemma3-4B-PT-DAPO-17.4k-LR2e6-linear500-1node/step_000500
```

It prepares the same DAPO train/eval split used in the ScaleTrain run and launches
`rl-distill-scripts/gemma3_12b_it_fsdp2_20k.sh`.

## Cluster Assumptions

- 2 nodes, 8 H100 GPUs each.
- Python 3.12 and `uv`.
- A shared repo path on every node, or the same repo checkout path on every node.
- A shared data/model path across nodes, or the same local paths prepared on every node.
- Network connectivity between nodes for Ray and NCCL.
- Hugging Face access to `JWei05/*` and gated access to `google/gemma-3-12b-it`.

Put secrets in `.env` at the repo root. Do not commit this file.

```bash
HF_TOKEN=...
WANDB_API_KEY=...
WANDB_BASE_URL=https://api.wandb.ai
```

Use a W&B key for the host in `WANDB_BASE_URL`. Public W&B keys work with
`https://api.wandb.ai`; Scale internal W&B needs a Scale W&B key.

## 1. Environment

Run on every node:

```bash
cd /path/to/rl-distill
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
bash rl-distill-scripts/setup_env.sh
source .venv/bin/activate
```

The setup script installs the stack used for this run:

- `torch==2.9.1+cu128`
- `vllm==0.15.1`
- `flash-attn==2.8.3`
- `transformers==4.57.6`
- editable `verl` from this repo

## 2. Model

Use the helper below to download only `step_000500`, then patch missing tokenizer
and metadata files from `google/gemma-3-12b-it`.

Run on every node unless `MODEL_LOCAL_DIR` is on a shared filesystem:

```bash
cd /path/to/rl-distill
source .venv/bin/activate
set -a; source .env; set +a

export MODEL_REPO=JWei05/DAPO-Gemma3-12B-PT-TopK128Distill-From-Gemma3-4B-PT-DAPO-17.4k-LR2e6-linear500-1node
export MODEL_SUBFOLDER=step_000500
export BASE_MODEL_REPO=google/gemma-3-12b-it
export MODEL_LOCAL_DIR=/shared/hf_models/gemma3-12b-topk128-distill-step000500

python3 rl-distill-scripts/data/download_hf_subfolder.py \
  --repo-id "$MODEL_REPO" \
  --subfolder "$MODEL_SUBFOLDER" \
  --metadata-repo "$BASE_MODEL_REPO" \
  --output-dir "$MODEL_LOCAL_DIR"
```

The train script can also do this automatically if `PREPARE_MODEL=1`, but
preparing the model explicitly makes multi-node failures easier to diagnose.

## 3. Data

Prepare the DAPO-only split:

- Source: `JWei05/DAPO-17.4k` train plus validation parquet files.
- Eval: 100 random rows sampled with seed 42.
- Eval duplication: each eval row is repeated 16 times with a shared `uid`.
- Train: all remaining rows.
- Heldout tracking:
  - `dapo_17k_test_seed42_n100.heldout.json`
  - `dapo_17k_test_seed42_n100.heldout.txt`

Run once if `DATA_DIR` is shared; otherwise run on every node using the same
`DATA_DIR`, seed, test size, and repeat count.

```bash
cd /path/to/rl-distill
source .venv/bin/activate
set -a; source .env; set +a

export DATA_DIR=/shared/verl/data
export DAPO_SPLIT_SEED=42
export DAPO_TEST_SIZE=100
export DAPO_EVAL_REPEAT=16

bash rl-distill-scripts/data/prepare_dapo_17k_split.sh
```

Expected outputs:

```text
/shared/verl/data/dapo_17k_train.parquet
/shared/verl/data/dapo_17k_test.parquet
/shared/verl/data/dapo_17k_test_seed42_n100.heldout.json
/shared/verl/data/dapo_17k_test_seed42_n100.heldout.txt
```

The duplicated eval rows let validation report pass@1/pass@16-style metrics:
`mean@16` is the average single-sample score and `best@16/mean` is the
pass@16-style score.

## 4. Start Ray

On the head node:

```bash
cd /path/to/rl-distill
source .venv/bin/activate

export HEAD_IP=<head-node-ip>
export NCCL_SOCKET_IFNAME=<network-interface>
export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export NCCL_SOCKET_FAMILY=AF_INET

ray stop --grace-period 30 || true
ray start --head \
  --node-ip-address="$HEAD_IP" \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --num-gpus=8 \
  --disable-usage-stats
```

On each worker:

```bash
cd /path/to/rl-distill
source .venv/bin/activate

export HEAD_IP=<head-node-ip>
export NCCL_SOCKET_IFNAME=<network-interface>
export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export NCCL_SOCKET_FAMILY=AF_INET

ray stop --grace-period 30 || true
ray start --address="$HEAD_IP:6379" --num-gpus=8 --disable-usage-stats
```

Back on the head, verify Ray sees 16 GPUs:

```bash
ray status
```

If your cluster uses IPv6 or a fabric interface, set `NCCL_SOCKET_FAMILY` and
`NCCL_SOCKET_IFNAME` accordingly before starting Ray and before launching
training.

## 5. Launch Training

Run on the Ray head node:

```bash
cd /path/to/rl-distill
source .venv/bin/activate
set -a; source .env; set +a

export RAY_ADDRESS=http://127.0.0.1:8265
export NNODES=2
export RAY_DATA_HOME=/shared/verl
export TRAIN_FILE=/shared/verl/data/dapo_17k_train.parquet
export TEST_FILE=/shared/verl/data/dapo_17k_test.parquet
export VAL_FILES="['/shared/verl/data/dapo_17k_test.parquet']"

export MODEL_PATH=/shared/hf_models/gemma3-12b-topk128-distill-step000500
export PREPARE_MODEL=0
export GEMMA3_CHAT_TEMPLATE_FILE=/path/to/rl-distill/rl-distill-scripts/data/gemma3_it_chat_template.jinja

export EXP_NAME=Gemma3-12B-TopK128Distill-RL-DAPO17k
export CKPTS_DIR=/shared/verl/ckpts/DAPO/$EXP_NAME
export HF_PUSH_REPO=JWei05/DAPO-Gemma3-12B-TopK128Distill-RL-DAPO17k
export HF_PUSH_ENABLE=True

export TRAIN_PROMPT_BSZ=64
export GEN_PROMPT_BSZ=64
export N_RESP_PER_PROMPT=16
export TRAIN_PROMPT_MINI_BSZ=32
export ACTOR_LR=5e-7
export ACTOR_LR_WARMUP_STEPS=50
export TEST_FREQ=5
export SAVE_FREQ=20
export VAL_BEFORE_TRAIN=True
export OFFLOAD=False
export ACTOR_FSDP_SIZE=-1
export SP_SIZE=1
export GEN_TP=1
export VAL_N=1

bash rl-distill-scripts/gemma3_12b_it_fsdp2_20k.sh
```

Important defaults in this run:

- FSDP2 actor sharding over all actor GPUs: `ACTOR_FSDP_SIZE=-1`.
- Sequence parallelism disabled: `SP_SIZE=1`.
- vLLM rollout tensor parallelism disabled: `GEN_TP=1`.
- CPU offload disabled: `OFFLOAD=False`.
- Eval every 5 steps, save every 20 steps.
- Checkpoints include `hf_model`, so HF push can upload model checkpoints.

## 6. Optional Cluster Wrapper

If your cluster launcher exposes these env vars, you can use:

```bash
bash rl-distill-scripts/scale_train/run_gemma3_12b_it_rl_dapo.sh
```

Required or useful env vars:

```text
NNODES=2
NODE_RANK=0 or 1
MASTER_ADDR=<head-node-ip>   # or LEADER_ADDR
NPROC_PER_NODE=8
PROJECT_ROOT=/path/to/rl-distill
RAY_DATA_HOME=/shared/verl
MODEL_LOCAL_DIR=/shared/hf_models/gemma3-12b-topk128-distill-step000500
```

Rank 0 starts Ray head, prepares the DAPO split, waits for all GPUs, and launches
training. Other ranks join Ray and stay alive while the head is running.

## 7. Monitoring

On the head node:

```bash
ray status
tail -f /tmp/ray/session_latest/logs/dashboard.log
```

For training logs, capture stdout from the launch command:

```bash
setsid bash -lc 'bash rl-distill-scripts/gemma3_12b_it_fsdp2_20k.sh' \
  > /shared/verl/logs/gemma3_12b_rl_dapo.log 2>&1 &
tail -f /shared/verl/logs/gemma3_12b_rl_dapo.log
```

Check W&B for `project_name=DAPO` and the `EXP_NAME` you set.

## 8. Common Failures

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `MASTER_ADDR/LEADER_ADDR is required` | Multi-node wrapper does not know the head address. | Set `MASTER_ADDR=<head-node-ip>` on all nodes. |
| Ray shows only 8 GPUs | Worker did not join or joined another Ray head. | Stop Ray on workers and re-run `ray start --address="$HEAD_IP:6379"`. |
| HF download missing tokenizer/chat template | Only the checkpoint subfolder was downloaded. | Use `download_hf_subfolder.py --metadata-repo google/gemma-3-12b-it`. |
| W&B authentication error | Key does not match `WANDB_BASE_URL`. | Use public W&B key with `https://api.wandb.ai`, or a Scale W&B key with Scale W&B. |
| Data file not found on workers | Data path is local to head only. | Put `RAY_DATA_HOME` on shared storage or prepare identical paths on every node. |
| OOM during actor/ref model init | Too little GPU memory for no-offload config. | Try `OFFLOAD=True` first; then consider lower batch sizes. |
| SP tensor shape errors | Gemma 3 VLM path and SP are not stable here. | Keep `SP_SIZE=1`. |
