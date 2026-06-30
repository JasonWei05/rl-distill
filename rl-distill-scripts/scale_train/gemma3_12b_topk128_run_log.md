# Gemma 3 12B Top-k Distillation Run Log

This tracks the ScaleTrain runs for distilling Gemma 3 4B PT DAPO outputs into
Gemma 3 12B PT with teacher top-k 128 KL. It intentionally omits API keys and
tokens.

Last updated: 2026-06-29 21:00 UTC.

## Current Recommendation

Use the one-node linear-decay `2e-6` run unless it later shows instability:

- 1 H100 node, `p5.48xlarge`, high priority, borrowing enabled.
- Global train batch size: `64`.
- Student: Gemma 3 12B PT.
- Teacher: Gemma 3 4B PT.
- Dataset: `JWei05/DAPO-Gemma3-4B-PT-DAPO-17.4k`.
- Validation split: 500 random rows from the full dataset, seed `42`.
- Train split: all remaining rows.
- Teacher KL: top-k `128`, chunk size `16`.
- FSDP2: `engine.fsdp_size=8`, `engine.model_dtype=bfloat16`,
  `engine.ulysses_sequence_parallel_size=1`.
- Optimizer: `LR=2e-6`, `LR_WARMUP_STEPS=100`, `LR_SCHEDULER_TYPE=linear`,
  `MIN_LR_RATIO=0.25`, `clip_grad=1.0`, `TOTAL_TRAINING_STEPS=500`.
  This warms up to `2e-6` and linearly decays to `5e-7`.
- Eval/save: `TEST_FREQ=5`, `SAVE_FREQ=200`, `MAX_CKPT_TO_KEEP=8`,
  `HF_PUSH_MAX_TO_KEEP=8`.

## Active Run

| Job id | Name | Status | Created | Notes |
| --- | --- | --- | --- | --- |
| `job_d907gdhjo0vg07vr0nrg` | `gemma3-12b-topk128-lr2e6-lin500` | `IN_PROGRESS` | 2026-06-28 01:17:47 UTC | One-node relaunch with `LR=2e-6`, `LR_WARMUP_STEPS=100`, linear decay to `5e-7`, and `TOTAL_TRAINING_STEPS=500`. W&B: `https://wandb.ai/rl-distill/topk-distill/runs/yv9qwyje`. Initial val at step 0 was `0.107933`; step 1 completed with LR `2e-8` and loss `0.127968`, confirming warmup. Saves at steps 200, 400, and final step 500. HF push repo: `JWei05/DAPO-Gemma3-12B-PT-TopK128Distill-From-Gemma3-4B-PT-DAPO-17.4k-LR2e6-linear500-1node`. |

## Active RL Run

| Job id | Name | Status | Created | Notes |
| --- | --- | --- | --- | --- |
| `job_d91dpe97779g07opt7mg` | `gemma3-12b-rl-dapo-jasonwei` | `QUEUED` | 2026-06-29 20:59:37 UTC | Relaunch of the DAPO-only RL run from the distilled Gemma 3 12B checkpoint with borrowing disabled: `allow_borrowing=False`. Launched on 2 H100 nodes with high priority. Uses `OFFLOAD=False`, `ACTOR_FSDP_SIZE=-1`, `SP_SIZE=1`, `GEN_TP=1`, `TRAIN_PROMPT_BSZ=64`, `GEN_PROMPT_BSZ=64`, `TRAIN_PROMPT_MINI_BSZ=32`, `N_RESP_PER_PROMPT=16`, `VAL_N=1`, `TEST_FREQ=5`, and `SAVE_FREQ=20`. Eval split is 100 random DAPO rows, each duplicated 16 times with shared `uid` so validation reports pass@1/pass@16-style metrics. Monitor: `/tmp/gemma3_12b_rl_dapo_no_borrow_monitor.log`. |
| `job_d90pvopjo0vg07vr0o90` | `gemma3-12b-rl-dapo-jasonwei` | `CANCELED` | 2026-06-28 22:27:47 UTC | Previous DAPO-only RL run with borrowing enabled. Canceled on 2026-06-29 so it could be relaunched with `allow_borrowing=False`. |

Launch command:

```bash
python rl-distill-scripts/scale_train/launch_st_job.py \
  --cluster eks \
  --n-instances 1 \
  --priority high \
  --allow-borrowing \
  --product train.enterprise_rlvr \
  --team egp \
  --job-name gemma3-12b-topk128-lr2e6-lin500 \
  --env-vars "TOTAL_TRAINING_STEPS=500,SAVE_FREQ=200,TEST_FREQ=5,VAL_ROW_COUNT=500,DATA_SPLIT_SEED=42,EXP_NAME=Gemma3-12B-PT-TopK128Distill-From-4B-PT-DAPO17k-RowSplit-500-LR2e6LinearTo5e7-1Node,LR=2e-6,LR_WARMUP_STEPS=100,LR_SCHEDULER_TYPE=linear,MIN_LR_RATIO=0.25,TRAIN_BATCH_SIZE=64,MAX_CKPT_TO_KEEP=8,HF_PUSH_MAX_TO_KEEP=8,HF_PUSH_REPO=JWei05/DAPO-Gemma3-12B-PT-TopK128Distill-From-Gemma3-4B-PT-DAPO-17.4k-LR2e6-linear500-1node"
```

Monitor:

```bash
/tmp/monitor_gemma3_12b_lr2e6_linear500.sh
/tmp/gemma3_12b_lr2e6_linear500_monitor.log
```

## Prior Runs

| Job id | Name | Status | Created | What happened |
| --- | --- | --- | --- | --- |
| `job_d8vhbupjo0vg07vr0n60` | `gemma3-12b-smoke-borrow-jasonwei` | `FAILED` | 2026-06-27 00:14:51 UTC | Smoke attempt failed because the remote command referenced the local `/mnt/efs/...` path inside the container. Fixed by mapping repo-local run files to `/workspace/rl-distill/...` in the ScaleTrain launcher. |
| `job_d8vho6h7779g07opt5u0` | `gemma3-12b-topk128-row-jasonwei` | `FAILED` | 2026-06-27 00:40:58 UTC | Full/row-split attempt had the same remote path problem. Superseded by the launcher path fix. |
| `job_d8vlt497779g07opt61g` | `gemma3-12b-smoke-rowfix-jasonwei` | `FAILED` | 2026-06-27 05:24:33 UTC | Remote path was fixed, but the run OOMed at the full LM-head projection in `full_vocab_kl_loss.py`; config was still using `engine.model_dtype=fp32`. |
| `job_d8vm2fpjo0vg07vr0ncg` | `gemma3-12b-smoke-memfix-jasonwei` | `FAILED` | 2026-06-27 05:35:59 UTC | Reduced max length/chunking was not sufficient because student model dtype was still fp32; failed with the same OOM class. |
| `job_d8vmbth7779g07opt620` | `gemma3-12b-smoke-bf16-jasonwei` | `COMPLETED` | 2026-06-27 05:56:06 UTC | First successful smoke after setting `engine.model_dtype=bfloat16` and `FULL_VOCAB_KL_CHUNK_SIZE=16`. Completed 1 train step; finite loss around `0.161760`. |
| `job_d8vmfn1jo0vg07vr0ndg` | `gemma3-12b-smoke-20k-bf16-jasonw` | `CANCELED` | 2026-06-27 06:04:12 UTC | Production-context smoke at 20k length. Left queued/pending; canceled when we decided to skip further smoke tests and launch the full run. |
| `job_d8vor41jo0vg07vr0neg` | `gemma3-12b-topk128-full-jasonwei` | `CANCELED` | 2026-06-27 08:45:04 UTC | Full 2-node run with `LR=1e-5`, `LR_WARMUP_STEPS=100`, `TOTAL_TRAINING_STEPS=2000`. W&B: `https://wandb.ai/rl-distill/topk-distill/runs/edo30bjk`. It ran successfully past step 360, saved/pushed at step 250, but validation loss was unstable: best `0.082399` at step 80, spike to `0.155192` at step 300, recovery to about `0.0944` by step 360. Canceled to relaunch with lower LR and longer warmup. |
| `job_d9029d97779g07opt6d0` | `gemma3-12b-topk128-lr5e6-1k-jaso` | `CANCELED` | 2026-06-27 19:29:57 UTC | Lower-LR run with `LR=5e-6`, `LR_WARMUP_STEPS=200`, `TOTAL_TRAINING_STEPS=1000`. W&B: `https://wandb.ai/rl-distill/topk-distill/runs/qvq85nrt`. It reached about step 200, but validation started rising again as LR warmed toward `5e-6`. Canceled to relaunch at `2e-6` with linear decay to `5e-7`. |
| `job_d907761jo0vg07vr0nr0` | `gemma3-12b-topk128-lr2e6-lin500` | `CANCELED` | 2026-06-28 01:06:32 UTC | Two-node version of the `2e-6` linear-decay run. It was still queued when we decided to test whether one node is sufficient, so it was canceled before training started. |

## Code And Config Changes Made

- Added `rl-distill-scripts/gemma3_12b_pt_topk128_distill_from_4b_pt_fsdp2.sh`.
  - Sets Gemma 3 12B PT student and Gemma 3 4B PT teacher paths.
  - Uses `TEACHER_TOP_K=128`.
  - Defaults `FULL_VOCAB_KL_CHUNK_SIZE=16`.
  - Defaults `ENGINE_FSDP_SIZE=8`.
  - Defaults `ENGINE_MODEL_DTYPE=bfloat16`.
  - Defaults `ULYSSES_SEQUENCE_PARALLEL_SIZE=1`.
  - Defaults `TEST_FREQ=5`.
  - Enables HF push by default.
  - Added `LR_SCHEDULER_TYPE` and `MIN_LR_RATIO` env overrides.
- Added linear warmup/decay support for FSDP optimizer schedules.
  - `verl/utils/torch_functional.py` now has `get_linear_schedule_with_warmup`.
  - `verl/workers/config/optimizer.py` allows `lr_scheduler_type=linear`.
  - `verl/workers/engine/fsdp/transformer_impl.py` wires the linear scheduler.
- Added `rl-distill-scripts/data/split_dapo_gemma3_4b_pt_by_prompt_idx.py`.
  - Despite the historical filename, it now creates a random row split.
  - Validation is `VAL_ROW_COUNT` random rows from the full source dataset.
  - Train is every remaining row.
  - Writes heldout source row indexes to JSON and TXT.
- Added ScaleTrain launch wrappers under `rl-distill-scripts/scale_train/`.
  - `launch_st_job.py` builds the image, injects selected `.env` values, creates
    the k8s-preset job config, and calls `scale-train train`.
  - `run_gemma3_12b_pt_topk128_distill.sh` adapts ScaleTrain env vars to the
    distributed torchrun script.
  - `launch_st_job.py` maps local repo paths to `/workspace/rl-distill/...` for
    remote containers.
- Updated environment setup.
  - `rl-distill-scripts/setup_env.sh` installs the needed training/inference stack
    in a uv environment.
  - `setup.py` was adjusted for the local dependency set.
- Updated `/mnt/efs/jasonwei/src/models/docs/scale_train_docs.md` with a
  ScaleTrain launch/monitoring section for these jobs.

## Notes

- The W&B metric prefix `full_vocab_kl/*` is inherited from the loss module.
  For these runs, `full_vocab_kl/top_k=128` confirms the KL sum uses the
  teacher top 128 tokens.
- The validation set is fixed for each run. The split seed and heldout row
  indexes are stored under the remote run's `/tmp/verl/data/...` directory.
- Batch size stayed at `64` for the lower-LR relaunch so the main changed
  variable is the LR schedule, not the token/example budget per step.
