# DAPO on B200 — Gemma 3 training

For the portable 2-node H100 recipe that resumes RL from the distilled
Gemma 3 12B checkpoint
`JWei05/DAPO-Gemma3-12B-PT-TopK128Distill-From-Gemma3-4B-PT-DAPO-17.4k-LR2e6-linear500-1node/step_000500`,
see [`docs/gemma3_12b_rl_dapo_portable.md`](docs/gemma3_12b_rl_dapo_portable.md).

RL training (DAPO) for Gemma 3 IT models on NVIDIA **B200** GPUs. This directory is a self-contained recipe: environment setup, data prep, model download, and launch scripts.

Upstream DAPO algorithm description: [arXiv:2503.14476](https://arxiv.org/abs/2503.14476). This doc is about **running** it here.

---

## 1. Hardware / scale

| Model        | Nodes     | GPUs | Script                              |
| ------------ | --------- | ---- | ----------------------------------- |
| Gemma 3 4B  IT | 1 × B200  | 8    | `gemma3_4b_it_fsdp2_20k.sh`         |
| Gemma 3 12B IT | 2 × B200  | 16   | `gemma3_12b_it_fsdp2_20k.sh`        |
| Gemma 3 27B IT | 4 × B200  | 32   | `gemma3_27b_it_fsdp2_20k.sh`        |

Each node: 8 × B200 (sm_100, 183 GB HBM). Training uses FSDP2 (`fsdp_size=-1`, full shard) + vLLM for rollout. Multi-node training uses `bond0` / AF_INET6; single-node uses `lo` / AF_INET.

## 2. Environment (uv, on every node)

```bash
cd /mlx_devbox/users/jason.wei/playground/rl-distill
bash dapo/setup_env.sh
```

This creates `.venv/` and installs the known-working B200 stack:

| Package        | Version     | Why pinned |
| -------------- | ----------- | ---------- |
| `torch`        | 2.9.1+cu128 | B200 sm_100 needs cu128 |
| `vllm`         | 0.15.1      | Torch 2.9 compat; earlier vLLMs leak subprocesses / fail on B200 |
| `flash-attn`   | 2.8.3       | Must be **built** against the installed torch (`--no-build-isolation`) |
| `transformers` | 4.57.6      | vLLM 0.15.1 rope_scaling check breaks on 5.x |
| `verl`         | editable    | This repo: `uv pip install --no-deps -e .` |

> The legacy recipe `USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh && pip install --no-deps -e .` is **outdated** — it pulls vllm 0.8.x / torch 2.6 which don't support sm_100. Use `setup_env.sh`.

You'll also need a `.env` at the repo root with:

```bash
HF_TOKEN=hf_...
WANDB_API_KEY=...
```

Training scripts source `.env` automatically and forward both keys into Ray's runtime env.

## 3. Data (once, on any node that has HF access)

```bash
source .venv/bin/activate
bash dapo/data/prepare_all_datasets.sh
```

Writes to `${HOME}/verl/data/`. Datasets:
- Train: `dapo-math-17k.parquet` (BytedTsinghua-SIA/DAPO-Math-17k)
- Val: AIME 2024 / 2025 / 2026 (×32), MATH500 (×2), OlympiadBench (×2), MinervaMath (×4), GSM8K test

Custom DAPO/OpenMathInstruct2 mix:

```bash
source .venv/bin/activate
python3 rl-distill-scripts/data/split_dapo_openmath2_mix.py
```

The source dataset is `JWei05/DAPO-OpenMathInstruct2-34k` / `dapo_openmath2_mix.parquet`.
The split is deterministic with `--seed 42`: 1,500 random rows are held out for validation and the rest are used for training.
Outputs:
- Train: `${HOME}/verl/data/dapo_openmath2_mix_train.parquet` (33,296 rows)
- Val: `${HOME}/verl/data/dapo_openmath2_mix_val.parquet` (1,500 rows)

Reward scoring is routed to `math_verify` (LatexExtractionConfig only — only `\boxed{}` answers score). See `verl/utils/reward_score/`.

## 4. Model download (once per node with local storage)

Gemma 3 IT models + chat template patch (strip the strict role-alternation check that breaks verl's `initialize_system_prompt`):

```bash
python3 dapo/setup_model.py --size 4b  --variant it
python3 dapo/setup_model.py --size 12b --variant it
python3 dapo/setup_model.py --size 27b --variant it
```

Writes to `${HOME}/verl/models/gemma-3-{size}-it/`.

## 5. Launch training

All launch helpers use `setsid … nohup … &` so training survives SSH disconnect. Logs go to `/tmp/dapo_training.log` (4B) or `/tmp/dapo_12b_training.log` (12B).

### 4B — single node

```bash
bash dapo/_launch_training.sh   # starts Ray head + launches training
bash dapo/_check_status.sh      # quick health check
bash dapo/_get_error.sh         # tail log, filter noise
```

### 12B — 2 nodes

On head node (e.g. `881268`):
```bash
bash dapo/_start_12b_head.sh
```
On worker node (e.g. `881270`):
```bash
bash dapo/_join_12b_worker.sh   # joins the head via bond0/v6 address
```
Back on head, once `ray status` shows 16 GPUs:
```bash
bash dapo/_launch_12b.sh
bash dapo/_check_12b_status.sh
```

### 27B — 4 nodes

Same pattern as 12B, but join 3 worker nodes to the head before launching. The script verifies `NNODES=4` × 8 = 32 GPUs; adapt `_launch_12b.sh` or run the bash script directly once the cluster is up:

```bash
# on head, after all 4 nodes have joined:
bash dapo/gemma3_27b_it_fsdp2_20k.sh
```

### Restart / clean

- `_clean_restart.sh` — stop Ray gracefully (`--grace-period 30`, never `--force`), kill orphan GPU processes, relaunch.
- `_clean_restart_12b.sh` — 12B variant.
- `_stop_and_restart.sh` — 4B stop + relaunch without full cleanup.

## 6. Key config choices

Shared across all three sizes (set in each `*_it_fsdp2_20k.sh`):

- **No dynamic sampling**: `enable_filter_groups=False`, `gen_prompt_bsz == train_prompt_bsz == 512`. Exactly 512 prompts per step, no oversample-and-filter.
- `n_resp_per_prompt=16`, `train_prompt_mini_bsz=32`
- `max_prompt_length=2k`, `max_response_length=20k`
- Soft overlong penalty enabled (`overlong_buffer_cfg`, 4k buffer, factor 1.0). Overlong *filtering* from the paper is **not** implemented in verl — only the soft penalty.
- `sp_size=1` always — Gemma 3 loads as a VLM and the SP tensor-shape path has a bug (`temperature_rmpad` mismatch). Leave at 1.
- `wrap_policy.transformer_layer_cls_to_wrap=["Gemma3DecoderLayer"]` — FSDP can't auto-discover the layer class for Gemma 3 VLM.
- Per-size differences: 4B uses `offload=True`, `gen_tp=1`, single-node `lo`/AF_INET. 12B is `offload=False`, `gen_tp=1`, 2-node `bond0`/AF_INET6. 27B is `offload=True`, `gen_tp=2`, 4-node `bond0`/AF_INET6.

## 7. Custom bits in this fork

- `dapo/dapo_ray_trainer.py` — custom `fit()` with trajectory saving (`sample_prob=0.001`), `all_gen_acc` pre-filter accuracy, extra metrics (`train/acc_mean`, `train/acc_all_generated`, `train/overlong_penalty_mean`, `train/overlong_ratio`).
- `verl/utils/reward_score/math_verify.py` — `LatexExtractionConfig` only; bare-number answers don't score. Subprocess-pool with 30s timeout.
- `verl/utils/reward_score/__init__.py` — routes `math`, `aime*`, `math500`, `olympiadbench`, `minervamath`, `gsm8k` to math_verify; returns `{"score": s, "acc": s > 0.5}`.
- `verl/trainer/ppo/ray_trainer.py` — validation loop logs per-sample `response_length`.

## 8. Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `ray stop --force` breaks `/proc` | Use `ray stop --grace-period 30` only. |
| GPU memory still pinned after kill | Grep compute-apps PIDs from `nvidia-smi`, `kill -9`. `_clean_restart.sh` does this. |
| NCCL "no socket interface found" | Single node: `NCCL_SOCKET_IFNAME=lo`, `NCCL_SOCKET_FAMILY=AF_INET`. Multi-node: `bond0` / AF_INET6. |
| `rope_scaling` error at model load | `transformers` is 5.x — pin to 4.57.6. |
| FSDP can't find `Gemma3DecoderLayer` | Already handled via explicit `wrap_policy`. Don't remove it. |
| Multi-node cluster shows 8 GPUs not 16 | Worker hasn't joined. Run `_join_12b_worker.sh` on the worker, wait for `ray status` on the head to report 16. |
| CPU OOM during rollout | Known vLLM 0.15.1 subprocess leak; restart. Watch `free -g` during long runs. |
