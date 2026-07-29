# Few-shot-prompt math RL — Gemma 3 1B & 4B PT

DAPO RL that trains the **PT base models** on math using the **same unified few-shot prompt** for
training rollouts *and* every validation set. The prompt is the one that reproduced the eval
numbers (4B PT: GSM8K 37.4 / MATH 24.4) — see `GEMMA3_PT_EVAL_REPLICATION.md`.

## Status (live)

**1B PT — 3-seed local sweep.** Launcher `launch_1b_fewshot_seed_sweep.sh` (has a pre-flight that
reaps stale sweep Ray/vLLM actors + frees the ports, so relaunch is always clean). 3 concurrent
runs, each on its own GPU pair + an isolated local Ray (`address=local`, distinct `_temp_dir`,
distinct `VERL_VLLM_PORT_BASE`), wandb project `DAPO`, entity `rl-distill`.

| Seed | GPUs | vLLM port base | wandb run | HF push repo | State |
|---|---|---|---|---|---|
| 42 | 2,3 | 52000 | [rdb4on62](https://wandb.ai/rl-distill/DAPO/runs/rdb4on62) | `JWei05/DAPO-Gemma3-1B-PT-FewShotMath-seed42` | **training** (continuing) |
| 43 | — | 53000 | [rwmx0jg0](https://wandb.ai/rl-distill/DAPO/runs/rwmx0jg0) | `JWei05/DAPO-Gemma3-1B-PT-FewShotMath-seed43` | **stopped @ step 400** |
| 44 | — | 54000 | [gjjze9j5](https://wandb.ai/rl-distill/DAPO/runs/gjjze9j5) | `JWei05/DAPO-Gemma3-1B-PT-FewShotMath-seed44` | **stopped @ step 500** |

Plan executed 2026-07-19: ran all 3 until two crossed step 400 (seed43=403, seed44=504), then stopped
those two to free GPUs 4,5,6,7 for the local 4B (below). seed42 keeps training on 2,3. Checkpoints for
43/44 saved through step 400/500 (local + HF).

**4B PT — LOCAL run** on GPUs **2,3,6,7**: wrapper `gemma3_4b_pt_fewshot_math_rl.sh`, `N_GPUS=4`,
`OFFLOAD=True`, `RAY_ADDRESS=local`, `VERL_VLLM_PORT_BASE=55000`, temp dir `/tmp/ray_4b_local`,
SAVE_FREQ=25, HF repo `JWei05/DAPO-Gemma3-4B-PT-FewShotMath`, wandb exp `gemma3-4b-pt-fewshot-math-local`.
Same core config as the 1B (unified prompt, DAPO-val-x16-only). Log `~/verl/logs/4b_local.log`.
Distinct from the (separate, still-QUEUED) ScaleTrain 4B.

- **Why 2,3,6,7 not 4,5,6,7:** after stopping seed43/44, another user grabbed GPUs 4,5, so seed42
  was also stopped (ckpt through step 450) to free 2,3. **Only ever `kill` your own pids on a shared
  box** — `nvidia-smi -i <gpus> --query-compute-apps` returns *all* users' pids; filter to your own.
- **OOM saga (attempts 1–3 → 4):** all three 20k-response attempts OOM'd in the **actor update** on
  the same thing — Gemma3's **262k-vocab LM head** materializes `[~25k tokens, 262144]` logits
  (~13.4 GB) on top of a ~58 GB actor footprint + ~9.9 GB resident vLLM. **vLLM util tuning is a dead
  end**: 0.78→0.5→0.3 barely moved update-phase free memory (0.3 was actually *worse*), because vLLM
  returns its KV pool on sleep and the 9.9 GB weight residual + the logits are what dominate. Gemma3
  has **no escape** via the actor path: fused-kernel CE isn't wired for it, and `sp_size>1` (Ulysses
  sequence parallel) has a known Gemma3 shape bug. **`MAX_RESPONSE_LENGTH` stays 20480.**
- **Fix that keeps 20k — rollout tensor parallelism (`GEN_TP=2`, attempt 5):** util tuning was a dead
  end because the ~9.9 GB *resident vLLM weights* + the 13.4 GB logits are what overflow. `GEN_TP=2`
  shards the 4B vLLM weights across 2 GPUs/engine, cutting the during-update vLLM residual to ~6 GB
  (−~4 GB) — enough to fit the 13.4 GB logits that missed by 0.1 GB at TP=1. Config: `GEN_TP=2`,
  `ROLLOUT_GPU_MEMORY_UTILIZATION=0.5`, `expandable_segments:True`, response length unchanged at 20k.
  If still tight, escalate to `GEN_TP=4` (residual ~4 GB) or the ScaleTrain full-node 4B.

> **vLLM port isolation (learned the hard way).** For concurrent runs on one host, the lever is
> **`VERL_VLLM_PORT_BASE`** (default 52000, +100/replica), read by
> `vllm_async_server._set_vllm_port_floor`, which then patches vLLM's `get_open_port`. `VLLM_PORT`
> is *ignored* (verl overwrites it), so give each run a distinct base ≥1000 apart. The launcher's
> pre-flight also reaps stale sweep Ray actors + frees those ports before relaunch.

- wandb: <https://wandb.ai/rl-distill/DAPO> · logs: `~/verl/logs/1b_fewshot_sweep/seed{42,43,44}.log`
- Fill the wandb run URLs into the table once each run prints `View run at …`.

**4B PT — ScaleTrain, 1 node / 8×H100 (`p5.48xlarge`), priority high.** Run-file
`scale_train/run_gemma3_4b_pt_fewshot_math_rl.sh`, same core config, + `ROLLOUT_GPU_MEMORY_UTILIZATION=0.5`
and `GEN_TP=2` (via `--env-vars`). HF repo `JWei05/DAPO-Gemma3-4B-PT-FewShotMath`, wandb `DAPO`.
- **ROOT CAUSE of the repeated FAILEDs (confirmed from pod logs + describe): IPv6 cluster + `AF_INET`.**
  ml-gpu-batch pods are **IPv6-only** (pod IP `2602:fb33:...`, iface `eth0`). The run-file/core forced
  `NCCL_SOCKET_FAMILY=AF_INET` (IPv4) → NCCL filters out the IPv6 `eth0` →
  `DistBackendError: ncclInvalidUsage ... Bootstrap : no socket interface found` at the FSDP weight
  broadcast → exit 1, ~6 min in. **Fix: `NCCL_SOCKET_FAMILY=AF_INET6` + `NCCL_SOCKET_IFNAME=eth0`**
  (`GLOO_SOCKET_IFNAME=eth0`), in the run-file + forced via `--env-vars`. `eth0` was right all along;
  `lo` was a wrong guess and `--allow-borrowing` a red herring (dropped anyway; owned-quota high-prio
  schedules fine).
- **Diagnosis toolkit (hard-won):** CLI is TTY-only (`script -qec "scale-train get job <ID>" /dev/null`),
  `get job` shows no reason, no `scale-train logs` command. **kubectl SSO expires → every call silently
  returns empty (looks like "no pods" but is auth failure).** Revive via the EC2 instance role:
  `aws eks update-kubeconfig --name ml-gpu-batch --region us-west-2 --alias ir-ml-gpu-batch` (real
  cluster `ml-gpu-batch`, ns `train`). Failed pods persist a while → `kubectl logs` + `describe` give
  the real error. Always `AWS_REGION=us-west-2`.

**Validation during training: `dapo_rl_val100_x16` only.** The other val parquets below are still
built (standalone eval) but are intentionally *not* wired into these runs' `VAL_FILES`.

## The prompt (one source of truth, applied to train + val)

`data/gemma3_it_fewshot_math.jinja` is a Gemma-3 IT chat template with the **12 interleaved
MATH+GSM8K exemplars baked in** (as user/model turns), followed by the actual question. It is
generated from the eval's own exemplars by `data/build_fewshot_chat_template.py`, and was
**verified to render byte-identical** to the eval's `fewshot_as_multiturn` prompt.

Because verl applies `actor_rollout_ref.model.custom_chat_template` to both rollout and
validation, train and all 8 val sets see the exact same 12-shot prompt. Data parquets stay
plain (question + `\boxed{}` instruction + gold); nothing few-shot is baked into the data.

Regenerate the template if the exemplars change:
```bash
lm-evaluation-harness/.venv-eval/bin/python rl-distill-scripts/data/build_fewshot_chat_template.py
```

## Data

```bash
.venv-megatron/bin/python rl-distill-scripts/data/build_math_rl_data.py   # writes to ~/verl/data
```
Produces (plain verl format, `data_source` routes to `math_verify`):

| File | Rows | Role |
|---|---|---|
| `dapo_rl_train.parquet` | 17,198 | **train** (DAPO-Math-17k minus 100) |
| `dapo_rl_val100_x16.parquet` | 1600 | val — held-out DAPO (100 q × 16, seed 42) |
| `math__math_500_x2.parquet` | 1,000 | val — MATH500 ×2 |
| `math__gsm8k_test.parquet` | 1,319 | val — GSM8K |
| `math__olympiadbench_x2.parquet` | 1,348 | val — OlympiadBench ×2 |
| `math__minervamath_x4.parquet` | 1,088 | val — MinervaMath ×4 |
| `math__beyondaime_x8.parquet` | 800 | val — BeyondAIME ×8 |
| `math__aime2025_x32.parquet` | 960 | val — AIME 2025 ×32 |
| `math__aime2026_x32.parquet` | 960 | val — AIME 2026 ×32 |

Repeat factors follow this repo's existing avg@k convention (AIME ×32, MATH500 ×2, etc.);
`val_kwargs` sample at **temp 1.0 / top_p 1.0 / top_k −1** (same as training), so the repeats give
avg@k / pass@k / maj@k per unique question. `beyondaime` was added to the `math_verify` router in
`verl/utils/reward_score/__init__.py`. *(Only `dapo_rl_val100_x16` is validated during these runs;
the rest are for standalone eval.)*

### pass@k / maj@k
verl's `process_validation_metrics` groups a dataset's rows by a stable id and reports, per group
size, **mean@k** (avg accuracy), **best@k** (= pass@k, "any correct") and **maj@k** (majority vote).
Two hooks make this work here:
- `build_math_rl_data.py` writes `uid = "<tag>-<i>"` on each base question and **shares that uid
  across all repeats** — so the 16 copies of a DAPO-val question form one group of 16.
- the `math` reward branch returns `pred` (last `\boxed{}` content, via
  `math_verify.extract_prediction`) so maj@k can take a mode over predictions.
So `dapo_rl_val100_x16` yields `mean@16` / `pass@16` / `maj@16` over the 100 held-out questions.

## Run

**1B — LOCAL 3-seed sweep** (2,3 / 4,5 / 6,7; isolated local Ray per run; pre-flight cleans first):
```bash
bash rl-distill-scripts/launch_1b_fewshot_seed_sweep.sh          # seeds 42/43/44 → wandb project DAPO
# single run / smoke (one GPU pair; wrapper defaults to GPUs 5,7):
TOTAL_TRAINING_STEPS=1 VAL_BEFORE_TRAIN=False HF_PUSH_ENABLE=False \
  bash rl-distill-scripts/gemma3_1b_pt_fewshot_math_rl.sh
bash rl-distill-scripts/gemma3_1b_pt_fewshot_math_rl.sh          # single full run
```

**4B — ScaleTrain, full 8×H100 node, high priority + borrowing:**
```bash
python rl-distill-scripts/scale_train/launch_st_job.py \
  --cluster eks --n-instances 1 --gpus-per-instance 8 --priority high --allow-borrowing \
  --build-config-key train-rl-distill --team egp --product train.enterprise_rlvr \
  --job-name gemma3-4b-pt-fewshot-math \
  --run-file run_gemma3_4b_pt_fewshot_math_rl.sh
```

Both use the shared core `gemma3_pt_fewshot_math_rl.sh`. `val_before_train=True` → step-0 baseline
(≈ eval numbers). Hyperparameters match the most-recent dense-PT DAPO recipe. Notable defaults:

| Var | Default | Note |
|---|---|---|
| `MAX_PROMPT_LENGTH` | 4096 | few-shot prefix ≈1250 tok + question (max ≈2461); **don't lower** (`truncation=left` eats the exemplars) |
| `MAX_RESPONSE_LENGTH` | **20480** (20k) | matches the prior recipe |
| overlong buffer / factor | **4096 / 1.0** | soft overlong penalty (prior recipe) |
| train + **val** sampling | temp 1.0, top_p 1.0, top_k −1 | validation uses the SAME sampling as training |
| `SAVE_FREQ` / `HF_PUSH_FREQ` | **25 / 25** | checkpoint + HF push every 25 steps |
| `TEST_FREQ` | 10 | validation cadence (val = 8 datasets) |
| `LOG_VAL_GENERATIONS` | 64 | val generations → wandb Table |
| `TRAIN_PROMPT_BSZ` / `N_RESP_PER_PROMPT` | 64 / 16 | GRPO group size |
| `ACTOR_LR` | 1e-6 | |
| `HF_PUSH_REPO` | `JWei05/DAPO-Gemma3-{1B,4B}-PT-FewShotMath` | weight-only HF push |
| `LOG_VAL_GENERATIONS` | 64 | # val generations logged to a **wandb Table** each eval |
| `ROLLOUT_DATA_DIR` | null | if set, dump **every train** rollout (prompt/response/score) to `<dir>/<step>.jsonl` |
| `VALIDATION_DATA_DIR` | null | if set, dump **every val** generation to `<dir>/<step>.jsonl` |

## Generation traces

- **wandb**: `LOG_VAL_GENERATIONS=64` (default on) logs a 64-sample table of validation
  `(input, output, score)` each eval — the standard way to eyeball format / reward-hacking /
  degeneration. Logging *all* traces to wandb is not standard (volume/UI cost); sample to wandb,
  dump everything to disk.
- **full dump to disk (JSONL)**: `ROLLOUT_DATA_DIR=~/verl/traces/train VALIDATION_DATA_DIR=~/verl/traces/val`
  writes every rollout/val generation per step for offline analysis. Off by default.
- This fork also saves a tiny random trajectory sample to `{ckpt_dir}/trajectories/`.

## Design notes

- **Few-shot for RL is deliberate**: a raw PT base model is near-floor zero-shot, so the few-shot
  prompt gives non-degenerate, gradeable rollouts (boxed answers) from step 0 — the reward signal
  RL needs. Train/val share it for a clean before/after against the eval baseline.
- To train **zero-shot** instead, set `GEMMA3_CHAT_TEMPLATE_FILE=…/gemma3_it_chat_template.jinja`
  (the plain IT template) — data is unchanged.
- Prefix caching is on; the 1250-token few-shot prefix is shared across all rollouts, so vLLM
  caches it once.
