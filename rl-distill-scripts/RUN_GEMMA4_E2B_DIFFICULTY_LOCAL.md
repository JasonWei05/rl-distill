# Run Gemma 4 E2B difficulty RL (easy / medium / hard) locally on a 2-GPU H100 node

Trains `google/gemma-4-E2B` from scratch with DAPO/GRPO on each of the three
DeepScaleR difficulty bands, one run per band, on **2 GPUs**, directly with a local
Ray (no ScaleTrain). This reproduces the seed-42 E2B difficulty sweep and keeps a
**permanent checkpoint every 10 steps** so the best-step weights are retained.

- **Model:** `google/gemma-4-E2B` (dense), revision `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`
- **Data:** `JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k`, bands easy/medium/hard
  (3,000 train + 300 validation questions each). Both the model and the dataset are
  public: no `HF_TOKEN` is needed to download them.
- **Per band:** 2 GPUs (FSDP2 DP-2, TP-1), 80 GB cards. The optimizer recipe is
  **identical** to the 4-GPU sweep (same prompt batch 64, 16 responses/prompt, mini
  batch 32, seed 42): FSDP2 just holds 512 sequences per GPU per step instead of 256,
  so each step takes roughly twice as long. Per-GPU micro-batch memory is unchanged.
- The three bands are independent: run them one after another on the 2-GPU node.

The code path is `run_gemma4_pt_deepscaler_4of4strict_rl.sh` → `gemma3_pt_fewshot_math_rl.sh`
→ `python3 -m dapo.main_dapo`. The wrapper downloads the base model, prepares the band
dataset, and trains. It reads the GPU count from `CUDA_VISIBLE_DEVICES`, so nothing in
the scripts needs to change for 2 GPUs.

Validated on 2026-09-03 on a 2× H100 80GB node (driver 580.105, CUDA 13.0, Ubuntu,
Python 3.10 system interpreter): venv build, reward-scorer unit test, dataset hash
check, and a 2-step end-to-end RL smoke (rollout → reward → FSDP2 update → weight sync →
validation → checkpoint save) all passed. See "What was verified" at the end.

## 0. Prerequisites

- A node with **2** GPUs (80 GB each), an NVIDIA driver that supports CUDA 13
  (driver ≥ 580; check the `CUDA Version` field of `nvidia-smi`), `git`, `curl`.
- `uv` (installed by the block below if missing). It downloads its own Python 3.12, so
  the system Python version does not matter.
- Clone this repo. A repo-root `.env` (untracked) is **optional**: add `WANDB_API_KEY=...`
  only if you want Weights & Biases. `HF_TOKEN` is not needed for E2B or the dataset.

```bash
git clone git@github.com:JasonWei05/rl-distill.git
cd rl-distill
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# optional:
printf 'WANDB_API_KEY=xxx\n' > .env
```

If the driver only supports CUDA 12.x (`CUDA Version: 12.8` in `nvidia-smi`), build with
`GEMMA4_CUDA_VARIANT=cu129` in step 1 instead; the run block is unchanged.

## 1. Build the Gemma 4 environment

Builds `.venv-gemma4` (torch 2.11 cu130, vLLM 0.25.1, transformers 5.14.1, ray 2.56,
all prebuilt wheels, no compilation). One-time per node; takes a few minutes.

```bash
bash rl-distill-scripts/setup_env_gemma4.sh          # add GEMMA4_CUDA_VARIANT=cu129 for CUDA-12 drivers
```

It must end with `GEMMA4_ENV_CORE_OK`. Optional sanity checks in the new venv:

```bash
source .venv-gemma4/bin/activate
uv pip install pytest
python -m pytest -q tests/utils/reward_score/test_math_verify_strict_boxed_on_cpu.py   # 19 passed
```

## 2. Train one band

Pick a band and paste the block. It trains from the base model, saves a full
checkpoint (model + Adam + dataloader cursor) to a local directory every 10 steps,
validates every 10 steps, and early-stops after 5 non-improving validations (max 400
steps). Nothing is uploaded.

```bash
export BAND=easy                                    # easy | medium | hard
export RUN_ROOT="$HOME/gemma4-e2b-${BAND}-s42"
mkdir -p "$RUN_ROOT/ckpts" "$RUN_ROOT/data" "$HOME/hf_cache" "$HOME/verl/logs"
export PATH="$HOME/.local/bin:$PATH"

CUDA_VISIBLE_DEVICES=0,1 \
RAY_ADDRESS=local \
RAY_TEMP_DIR="/tmp/ray_g4_e2b_${BAND}" \
HF_HOME="$HOME/hf_cache" \
NCCL_SOCKET_IFNAME=lo NCCL_SOCKET_FAMILY=AF_INET GLOO_SOCKET_IFNAME=lo \
CKPTS_DIR="$RUN_ROOT/ckpts" \
DATA_DIR="$RUN_ROOT/data" \
GEMMA4_MODEL=google/gemma-4-E2B \
GEMMA4_MODEL_REVISION=d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f \
DIFFICULTY_DATASET_SOURCE=gemma4_26b_bands \
DIFFICULTY_DATASET="$BAND" \
DIFFICULTY_DATASET_REPO=JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k \
DIFFICULTY_DATASET_REVISION=a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db \
DATA_SEED=42 \
TOTAL_TRAINING_STEPS=400 \
TRAIN_PROMPT_BSZ=64 GEN_PROMPT_BSZ=64 N_RESP_PER_PROMPT=16 TRAIN_PROMPT_MINI_BSZ=32 \
ACTOR_LR=1e-6 ACTOR_LR_WARMUP_STEPS=20 \
MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=8192 MAX_MODEL_LEN=12288 \
OVERLONG_BUFFER_LEN=2048 ENABLE_OVERLONG_BUFFER=True OVERLONG_PENALTY_FACTOR=1.0 \
MICRO_BATCH_SIZE_PER_GPU=8 MAX_PADDED_TOKENS_PER_MICROBATCH=12288 \
SP_SIZE=1 GEN_TP=1 ACTOR_FSDP_SIZE=-1 \
FSDP_CPU_OFFLOAD_POLICY=True OFFLOAD=False \
ROUTER_REPLAY_MODE=disabled ROUTER_Z_LOSS_COEF=0.0 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.25 VLLM_KV_CACHE_MEMORY_BYTES=536870912 \
ROLLOUT_ENFORCE_EAGER=False VLLM_DISABLE_COMPILE_CACHE=0 \
TEST_FREQ=10 SAVE_FREQ=10 MAX_ACTOR_CKPT_TO_KEEP=100 \
ROLLING_CHECKPOINT_ENABLED=False HF_PUSH_ENABLE=False HF_PUSH_REQUIRED=False \
EARLY_STOPPING_ENABLED=True EARLY_STOPPING_METRIC='val-core/math/acc/mean@16' \
EARLY_STOPPING_MODE=max EARLY_STOPPING_PATIENCE=5 EARLY_STOPPING_MIN_DELTA=0.0 \
EARLY_STOPPING_INCLUDE_INITIAL_VALIDATION=True \
LOG_VAL_GENERATIONS=100 LOG_TRAIN_GENERATIONS=100 \
EXP_NAME="g4-e2b-${BAND}-s42-local" \
VENV="$PWD/.venv-gemma4" \
  bash rl-distill-scripts/scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh \
    'trainer.logger=["console"]' \
  2>&1 | tee "$HOME/verl/logs/g4_e2b_${BAND}.log"
```

Run the block again with `BAND=medium` and `BAND=hard` for the other two. Run it under
`nohup`/`tmux`: a 400-step run at 8k responses takes on the order of days on 2 GPUs.

### Weights & Biases (optional)

The launcher's default logger is `["console", "wandb"]`, and `wandb.init` blocks
waiting for a login if no key is present, so the block above pins
`trainer.logger=["console"]`. To log to W&B, put `WANDB_API_KEY=...` in `.env` and drop
the `'trainer.logger=["console"]'` line. `EXP_NAME` is the W&B run name.

### Why these environment settings

- `CUDA_VISIBLE_DEVICES=0,1` sets the world size; the wrapper derives `N_GPUS_PER_NODE`
  from it.
- `NCCL_SOCKET_IFNAME=lo` / `AF_INET` / `GLOO_SOCKET_IFNAME=lo`: the wrapper's defaults
  are the ScaleTrain pod values (`eth0`, IPv6). A local single node must use loopback.
- `HF_HOME=$HOME/hf_cache`: the wrapper defaults to `/tmp/hf_cache`. Keeping the ~10 GB
  model snapshot under `$HOME` survives reboots and is reused by every band.
- `RAY_TEMP_DIR` is per band so a stale Ray session from one band never collides with
  the next.
- `trainer.logger=["console"]` is a Hydra override forwarded verbatim; see the W&B note.

## 2b. All three bands back to back, with HF checkpoint upload (recommended)

`rl-distill-scripts/run_gemma4_e2b_difficulty_sequential_local.sh` chains easy → medium →
hard on the same 2 GPUs. Each band uses the section-2 recipe with these differences:

- **HF upload every 10 steps, newest 5 kept.** `HF_PUSH_ENABLE=True HF_PUSH_FREQ=10
  HF_PUSH_MAX_TO_KEEP=5`: each save's `actor/huggingface/` export is pushed to
  `<namespace>/DAPO-gemma4-e2b-PT-DeepScaleR-gemma26b-<band>-seed42-local2gpu` under
  `step_000010/`, `step_000020/`, ... and folders older than the newest 5 are deleted from
  the Hub. Local full checkpoints are still kept every 10 steps (`MAX_ACTOR_CKPT_TO_KEEP=100`).
- **Early stopping patience 4.** A validation "improves" only if it is strictly greater than
  the all-time best; 4 consecutive non-improving validations end the band. The initial
  step-0 validation counts as the first observation.
- The next band starts automatically once the previous one logs `RUN_DONE rc=0`. A band whose
  `ckpts/run_outcome.json` already exists is skipped, so the launcher can be re-run after a
  crash and will resume the interrupted band from its latest local checkpoint.
- Everything lands under `$HOME/gemma4-e2b-difficulty-s42/<band>/{ckpts,data}` with per-band
  logs in `$HOME/gemma4-e2b-difficulty-s42/logs/`.

It needs a **write-capable `HF_TOKEN`** in `.env` (or the environment). If the token is
missing at launch the script polls for it every 60 s instead of failing, so it can be
started first and the token added afterwards. The Hub namespace defaults to the token
owner; override with `HF_PUSH_NAMESPACE=<org>`. Repos are created public (the launcher's
`+trainer.hf_push.private=False`); append `trainer.hf_push.private=True` as an extra
argument for private repos.

```bash
printf 'HF_TOKEN=hf_xxx\n' > .env                      # write token; add WANDB_API_KEY=... for W&B
mkdir -p ~/verl/logs
nohup bash rl-distill-scripts/run_gemma4_e2b_difficulty_sequential_local.sh \
  > ~/verl/logs/g4_e2b_sequential.log 2>&1 &
tail -f ~/verl/logs/g4_e2b_sequential.log
```

Knobs: `DIFFICULTY_SEQUENCE="medium hard"` to run a subset, `EARLY_STOPPING_PATIENCE`,
`HF_PUSH_MAX_TO_KEEP`, `TOTAL_TRAINING_STEPS`, and every recipe variable from section 2
are env-overridable.

**Memory and speed defaults in the chained launcher (differ from section 2).** Set on
2026-09-03 after three OOM'd attempts on a 4x H100 80GB node running medium and hard side by
side (see section 2c). vLLM stays resident during training in this recipe (sleep mode is off in
the wrapper because wake-up was fragile on this stack), so the trainer only ever gets what the
engine leaves over.

| Setting | Section 2 (4-GPU sweep knobs) | Chained launcher default | Why |
|---|---|---|---|
| FSDP2 offload | `FSDP_CPU_OFFLOAD_POLICY=True` | same (`OFFLOAD=False`) | Phase-level offload (`OFFLOAD=True`, policy off) loads the full 2-way sharded fp32 state (params 9.6 + grads 9.6 + Adam 19.2 = 38 GB for the 4.8B-param E2B) onto the GPU for `update_actor`, and medium OOMed there at 47 GB allocated next to a 30 GB vLLM. On 4 GPUs that state halves, which is the only reason it fit in the sweep. |
| vLLM CUDA graphs | eager (`ROLLOUT_ENFORCE_EAGER=True`) | same | With `ROLLOUT_ENFORCE_EAGER=False` (FULL_AND_PIECEWISE graphs, `max_num_seqs` 1024) the resident engine grew to 46 to 58 GB per GPU and both bands OOMed at step 1. Eager costs ~50 s of generation per step. |
| vLLM KV cache | 512 MiB | 4 GiB | Rollout concurrency; generation is 50 to 200 s per step at 4 GiB. |
| Worker CPU threads | Ray default (1) | `OMP_NUM_THREADS=16` per Ray worker (`WORKER_OMP_THREADS`) | With the CPU offload policy the per-micro-batch gradient accumulation (`sharded_grad += new_grad` over 9.6 GB of fp32) and the AdamW step run on the host. Single-threaded, `update_actor` took ~710 s with the worker pinned at exactly 100% CPU. |
| vLLM allocator | - | `torch.cuda.empty_cache()` after each weight sync (fork change in `verl/workers/rollout/vllm_rollout/utils.py`) | The engine process grew from 20 GB to 29 GB at the first weight sync and 31.5 GB at the second, leaving 1.3 GB free on the hard band's GPUs. |
| FSDP2 grad offload copy | stock torch | pinned-memory accumulate path (`VERL_FSDP2_CPU_OFFLOAD_PINNED_ACCUM=1`, fork module `verl/utils/fsdp2_cpu_offload_pinned_patch.py`, `FSDP2_PINNED_ACCUM=0` to disable) | py-spy put the remaining update time in torch 2.11's `foreach_reduce`: when accumulating, it copies each micro-batch's 9.6 GB fp32 grad shard to a freshly allocated pageable CPU tensor with a blocking copy (~4 s each, ~100 per step, worker at exactly 100% CPU). The patch rewrites that one branch to copy into cached pinned memory asynchronously and sync the stream before the CPU add. Same bytes, same gradients. |
| vLLM cache after generation | - | released via an opt-in RPC (`VERL_ROLLOUT_RELEASE_CACHE_AFTER_GEN=1`; fork change in `agent_loop.py`, `replica.py`, `vllm_async_server.py`, vLLM worker extension) | Even with the sync-time release, the engine held 30 to 32 GB of generation workspace through the trainer's update; per-GPU peaks reached 80.5 of 81.5 GB with both bands running. Emptying the engine's caching allocator right after each generation round returns ~10 GB before old-logprob/update. |
| Micro-batch | 8 seqs / 12k padded tokens | unchanged | The padded-token cap already equals one max-length sample (4096 + 8192); a long sample is a singleton forward, so smaller micro-batches do not lower the peak. |

None of these change the optimizer trajectory (same data order, batch composition, loss, and
update count). Measured step times are in section 2c.

**Watchdog.** `rl-distill-scripts/watch_gemma4_e2b_difficulty_chain.sh` relaunches the chain
if it exits before `SEQUENTIAL_GEMMA4_E2B_DIFFICULTY_RUNS_DONE` (the launcher skips finished
bands and the trainer resumes from the latest local checkpoint). It gives up after 3
consecutive restarts with no new checkpoint.

```bash
nohup bash rl-distill-scripts/watch_gemma4_e2b_difficulty_chain.sh > ~/verl/logs/g4_e2b_watchdog.log 2>&1 &
```

## 2c. Two bands side by side on a 4-GPU node (medium + hard, 2 GPUs each)

When one band (e.g. easy) is already running elsewhere, the remaining two can share a 4-GPU
node. `rl-distill-scripts/run_gemma4_e2b_medium_hard_parallel_local.sh` starts two
single-band instances of the section-2b chain, each with its own watchdog:

| Band | `CUDA_VISIBLE_DEVICES` | `VERL_VLLM_PORT_BASE` | Chain log | Trainer log |
|---|---|---|---|---|
| medium | `0,1` | 52000 | `~/verl/logs/g4_e2b_medium_sequential.log` | `~/gemma4-e2b-difficulty-s42/logs/medium.log` |
| hard | `2,3` | 54000 | `~/verl/logs/g4_e2b_hard_sequential.log` | `~/gemma4-e2b-difficulty-s42/logs/hard.log` |

```bash
nohup bash rl-distill-scripts/run_gemma4_e2b_medium_hard_parallel_local.sh \
  > ~/verl/logs/g4_e2b_medium_hard_launch.log 2>&1 &
```

Isolation between the two instances: each Ray session is a separate local cluster (Ray
picks free GCS ports; per-band `RAY_TEMP_DIR`), vLLM port bases are disjoint, and
checkpoints/data/HF repos are already per band. Two things were changed so instances do
not step on each other: the chain launcher and the watchdog no longer run a node-wide
`ray stop` (they `pkill` only processes whose argv carries their own `ray_g4_e2b_<band>`
temp dir), and the watchdog identifies "its" chain by matching `DIFFICULTY_SEQUENCE` in
the launcher's `/proc/<pid>/environ` instead of a bare `pgrep`. The second chain starts
two minutes after the first so downloads and Ray bring-up are staggered. GPU-side
resources are identical to the single-chain case (each band still sees exactly 2 GPUs);
host RAM and CPU are shared, which is fine on a large node but worth checking on a
small one (each chain's FSDP2 offload holds an E2B copy plus Adam state in host memory).

**Measured on 2026-09-03 (4x H100 80GB, both bands running side by side, 8k responses,
64 prompts x 16, per GPU):**

| Config | update_actor | step (medium) | trainer peak | vLLM resident | outcome |
|---|---|---|---|---|---|
| phase-level offload, CUDA-graph vLLM (first launcher defaults) | - | - | 47 GB | 46 to 58 GB | OOM at step 1, both bands |
| phase-level offload, eager vLLM | 172 s | 331 s | 49 GB | 30 GB | OOM at step 2 (update_actor, 47 GB allocated + 30 GB vLLM) |
| CPU offload policy, 1 thread | 704 to 720 s | 830 to 980 s | 50 GB | 30 GB | stable, too slow |
| CPU offload policy, 16 threads | 596 to 614 s | 855 to 872 s | 50 GB | 30 GB | stable, too slow |
| CPU offload policy, 16 threads, pinned accumulate | 192 to 212 s | 293 to 471 s | 50 GB | 20 to 32 GB | stable for 12 steps, but GPU peaks of 80.5 of 81.5 GB: the engine kept its generation workspace through the update |
| + release vLLM cache after each generation round (current default) | 208 s | 498 s (8k response in batch) | 49.5 GB | 16.5 GB during update, ~19 GB during generation | stable; GPU peak 66 GB |

Generation adds 40 to 210 s per step depending on how many responses run to the 8192 cap
(hard runs longer than medium). Host RAM in use with both bands: ~180 GB of 885 GB.

**Hub uploads.** As of 2026-09-03 13:00 UTC every push from these runs fails with
`You have exceeded your public storage space` (the account holds ~5.65 TB across 83 public model
repos; the two run repos are ~40 GB each). Local checkpoints are the source of truth; nothing is
lost, but the Hub stays at medium step 40 / hard step 30 until space is freed or the plan is
upgraded. Do not start the gap filler below until then, since each failed attempt still streams the
10 GB LFS payload before the commit is rejected. The trainer's `HFPusher` uploads each 10-step export
(~10 GB) in a background thread and retries transient failures (now 8 attempts, backoff capped at 120 s, `HF_PUSH_MAX_RETRIES`);
training is never blocked by it, but a lost upload is remembered and makes the run exit non-zero at
its very end (`HF_PUSH_REQUIRED=True`), which the watchdog absorbs with one restart into the finished
state. To keep the Hub complete regardless, run the gap filler alongside the chains; every 15 min it
uploads any of the newest 5 completed local exports missing on the Hub and prunes older Hub folders:

```bash
nohup bash rl-distill-scripts/run_hf_hub_gap_filler.sh >> ~/verl/logs/g4_e2b_hf_gap_filler.log 2>&1 &
```

**Reference outcome (2026-09-03/04, this recipe, 2 GPUs per band, both bands side by side).**
Medium early-stopped at step 280 with best `val-core/math/acc/mean@16` 22.50% at step 240 (4.35% at
step 0); hard early-stopped at step 230 with best 13.63% at step 190 (2.98% at step 0). Wall clock
about 19 to 20 hours per band including two restarts; ~250 s per step once responses shortened.

## 3. Checkpoints

- A full checkpoint lands at `$RUN_ROOT/ckpts/global_step_{10,20,30,...}/` every 10 steps:
  `actor/` (FSDP2 model + Adam shards + a consolidated `huggingface/model.safetensors`
  inference export) and `data.pt` (dataset cursor).
- `MAX_ACTOR_CKPT_TO_KEEP=100` retains every 10-step checkpoint (40 for a 400-step run).
  Each E2B full checkpoint is ~68 GB, so 40 of them is ~2.7 TB. Lower it if disk is
  tight; the per-checkpoint `actor/huggingface/model.safetensors` (~10 GB) is the
  weight-only inference model if you only need weights, not resume state.
- The best step is the one whose validation `val-core/math/acc/mean@16` is highest;
  it is printed at each validation and recorded in `validation_early_stopping.json`
  inside the checkpoint directory.
- **Resume after a crash:** rerun the same block. `trainer.resume_mode=auto` (the
  default) picks up `latest_checkpointed_iteration.txt` in `CKPTS_DIR` and restores
  model, Adam, and dataset position. Keep every recipe knob identical.

## 4. Optional: durable checkpoints to S3 (the full rolling infra)

If the node can reach S3 with AWS credentials and you want the production infra (a
rolling "latest step always" checkpoint plus the permanent every-10 checkpoints mirrored
to S3, with seamless resume after a crash), set these instead of the local-only
checkpoint knobs above:

```bash
  FULL_CHECKPOINT_S3_URI=s3://<your-bucket>/gemma4-e2b-${BAND}-s42/full-checkpoints \
  RUN_ARTIFACT_S3_URI=s3://<your-bucket>/gemma4-e2b-${BAND}-s42/artifacts \
  WANDB_RUN_ID=g4-e2b-${BAND}-s42-local \
  ROLLING_CHECKPOINT_ENABLED=True ROLLING_CHECKPOINT_FREQ=1 MAX_ACTOR_CKPT_TO_KEEP=1 \
```

## 5. Quick 2-step smoke (recommended before the first real run)

Same wrapper, tiny sizes, a 4-question validation subset, and a checkpoint save at step
2. It exercises every stage of the loop on the real 2-GPU layout in well under an hour.
Prepare the band data first so the val subset can be cut from it:

```bash
source .venv-gemma4/bin/activate
mkdir -p "$HOME/gemma4-smoke/ckpts" "$HOME/gemma4-e2b-easy-s42/data"
python rl-distill-scripts/data/prepare_deepscaler_gemma4_26b_difficulty_rl_data.py \
  --data-dir "$HOME/gemma4-e2b-easy-s42/data" --band easy --validation-repeats 16
python - <<'PY'
import os, pandas as pd
d=os.path.expanduser("~/gemma4-e2b-easy-s42/data/")
df=pd.read_parquet(d+"deepscaler_gemma4_26b_easy_val300_x16.parquet")
df[df.uid.isin(df.uid.unique()[:4])].to_parquet(d+"smoke_val4_x16.parquet", index=False)
PY
```

Then run the step-2 block with these overrides on top (everything else unchanged):

```bash
CKPTS_DIR="$HOME/gemma4-smoke/ckpts" DATA_DIR="$HOME/gemma4-e2b-easy-s42/data" \
RAY_TEMP_DIR=/tmp/ray_g4_smoke DIFFICULTY_DATASET=easy \
VAL_FILES="['$HOME/gemma4-e2b-easy-s42/data/smoke_val4_x16.parquet']" \
TOTAL_TRAINING_STEPS=2 VAL_BEFORE_TRAIN=False \
TRAIN_PROMPT_BSZ=8 GEN_PROMPT_BSZ=8 N_RESP_PER_PROMPT=4 TRAIN_PROMPT_MINI_BSZ=4 \
ACTOR_LR_WARMUP_STEPS=1 MAX_RESPONSE_LENGTH=2048 MAX_MODEL_LEN=6144 OVERLONG_BUFFER_LEN=512 \
TEST_FREQ=2 SAVE_FREQ=2 MAX_ACTOR_CKPT_TO_KEEP=1 LOG_VAL_GENERATIONS=4 LOG_TRAIN_GENERATIONS=0 \
EXP_NAME=g4-e2b-smoke-2gpu \
```

Success is `RUN_DONE rc=0` at the end of the log and a `global_step_2/` directory under
the smoke `CKPTS_DIR`. Delete `$HOME/gemma4-smoke` afterwards (~68 GB).

## Notes

- **2 GPUs, dense model.** FSDP2 shards E2B across the 2 GPUs (`ACTOR_FSDP_SIZE=-1`),
  with the FSDP2 CPU offload policy on. `ROUTER_REPLAY_MODE=disabled` because E2B is
  not a MoE.
- Keep the recipe identical to reproduce the sweep: same dataset repo/revision,
  `DATA_SEED=42`, batch sizes, `N_RESP_PER_PROMPT=16`. The GPU count changes only the
  per-GPU share of each batch, not the optimizer trajectory's definition.
- Different GPU memory: lower `ROLLOUT_GPU_MEMORY_UTILIZATION` /
  `VLLM_KV_CACHE_MEMORY_BYTES` / `MAX_PADDED_TOKENS_PER_MICROBATCH` for smaller cards.
- Entropy near 9 or generations pinned at `MAX_RESPONSE_LENGTH` is a correctness alarm,
  not a tuning problem (see `GEMMA3_MOE_RL_TRAINING.md`).

## What was verified (2026-09-03, 2× H100 80GB, driver 580.105 / CUDA 13.0)

| Check | Result |
|---|---|
| `setup_env_gemma4.sh` (cu130) | `GEMMA4_ENV_CORE_OK`; torch 2.11.0+cu130, vLLM 0.25.1, transformers 5.14.1; all four Gemma 4 vLLM archs registered |
| `pytest tests/utils/reward_score/test_math_verify_strict_boxed_on_cpu.py` | 19 passed |
| `prepare_deepscaler_gemma4_26b_difficulty_rl_data.py --band easy` | `DATASET_READY`, SHA-256 of both source splits matched the pinned revision |
| 2-step smoke via `run_gemma4_pt_deepscaler_4of4strict_rl.sh` on `CUDA_VISIBLE_DEVICES=0,1` | `RUN_DONE rc=0`; steps 1 and 2 trained, rollout IS ratio mean 0.999, reward mean 0.125, response length ~123 tok, clip ratio 0 |
| Validation at step 2 (4 questions x16) | `val-core/math/acc/mean@16` = 0.125 logged; early-stopping state `best_step=2`, `validation_early_stopping.json` written |
| Checkpoint at step 2 | `global_step_2/{actor,data.pt}` with `model/optim/extra_state_world_size_2_rank_{0,1}.pt`, `actor/huggingface/`, `latest_checkpointed_iteration.txt`, `run_outcome.json`; 64 GB total |

A `nvidia-smi` sample taken mid-update in step 1 showed ~16 GB used per card (vLLM
resident at 0.25 utilization plus the FSDP2 actor with CPU offload). That is a sample,
not a measured peak, and the smoke used 2k responses; the production 8k-response,
64-prompt x16 batch has not yet been run to completion on this node.
