# Maze study on ScaleTrain: runbook

How to rerun the 25×25 maze pipeline on ScaleTrain: pretraining → exact base tuning → RL grid → report pages.
The commands are the ones the November 2026 runs used; the first RL example combines the settings those runs ended on. Results and decisions are in `../PROGRESS_LOG.md`. The earlier local budget study
(8-GPU box) is in `BUDGET_STUDY_RUNBOOK.md`.

## Where things are

| What | Where |
|---|---|
| Code (this folder) | branch `jiahaoz`, worktree `/mnt/efs/jasonwei/rl-distill-jiahaoz` (shares its object store with the main checkout) |
| TailRL (trainer, verl fork) | `Zanette-Labs/TailRL` @ `5682c6a` + `tailrl_text_maze.patch` + `tailrl_rl_tis.patch`; built by `st_setup.sh` on every pod |
| 25×25 data | `s3://scale-ml/genai/rl-distill/maze/data/dfs25_eps05/` (`sft/`, `main_parquet/`, `eval1000/`, `eval500/`, `test1000/`) |
| Pre-tokenized pretraining data | `.../data/dfs25_eps05/sft/train.tok-L1024/` |
| Pretraining runs and tuned bases | `s3://scale-ml/genai/rl-distill/maze/pt_dfs25_eps05_v1/<SIZE>/<SIZE>_is<seed>_a<attempt>/` (`tune3-<level>/best`) |
| RL grids | `.../maze/rl_small_dfs25_tis3s_v1`, `.../maze/rl_medium_dfs25_tis3s_v1` (`bases/`, `runs/<suite>/`, `pods/`) |
| W&B | `rl-distill/tailrl-text-maze-dfs25-eps05` |
| Supervisor state | `/mnt/efs/jasonwei/rl-distill/.scale_train_supervisors/<TAG>-<YYYYMMDD><SUP_TAG>/` |

Launchers `cd` to the main checkout and ship `git archive <CODE_REF>`. Commit on `jiahaoz` first and pass that commit as
`CODE_REF`. Running supervisors pin the commit they started on; old pinned commits stay reachable through
`jiahaoz-history-20261112`, so don't delete that branch while runs that started before 11-13 are alive.

Model sizes (`maze_pt_pipeline.py` `ARCH`): 3XS 0.55M, 2XS 1.0M, XS (small) 2.2M, S (medium) 3.9M, L (large) 13.3M.

## 1. Data (once per maze distribution)

```bash
# 25x25 depth-first mazes, 0-20% wall knockout, eps = 0.5 noisy demonstrations, 1.3M training mazes
~/TailRL/.venv/bin/python rl-distill-scripts/maze/build_dfs_eps_data.py --size 25 --eps 0.5 --out /tmp/maze_data/dfs25_eps05
aws s3 sync /tmp/maze_data/dfs25_eps05 s3://scale-ml/genai/rl-distill/maze/data/dfs25_eps05
```

Pre-tokenize the training set once. Otherwise every trainer on a pod tokenizes it itself, which takes about 35 min.
The tokenizer comes from any run's `init_model/` (the vocabulary is fixed). `--verify` re-encodes 2,000 random items with
the trainer's own code and requires identical ids and labels:

```bash
aws s3 sync s3://.../maze/pt_dfs25_eps05_v1/2XS/<run>/init_model /tmp/maze_tok --exclude model.safetensors
/tmp/jw_maze/TailRL/.venv/bin/python rl-distill-scripts/maze/tokenize_maze_sft.py --data <dir>/sft/train.json \
  --tokenizer /tmp/maze_tok --max-length 1024 --out <dir>/sft/train.tok-L1024 --tailrl /tmp/jw_maze/TailRL --verify 2000
aws s3 sync <dir>/sft/train.tok-L1024 s3://scale-ml/genai/rl-distill/maze/data/dfs25_eps05/sft/train.tok-L1024
```

The trainer (`--train_cache`) refuses a cache whose source byte size, sequence cap, vocabulary or probe encoding differs.
So rebuild the cache whenever `train.json` or `MAX_LENGTH` changes. The pod log prints `MAZE_PT_NO_CACHE` when no cache
was found.

## 2. Pretraining (one supervised pod per size)

`start_maze_pt_supervisor.sh` → `run_maze_pretrain_st.sh` → `maze_pt_pipeline.py`. Each size trains all its seeds on
one pod, screens checkpoints as they land and mirrors everything to `${OUT_S3}/${SIZE}`. With `NO_TUNE=1` it only trains
and screens; tuning is step 3. The pod sets `${OUT_S3}/${SIZE}/COMPLETE` when done; a relaunch resumes from S3.

```bash
cd /mnt/efs/jasonwei/rl-distill
OUT=s3://scale-ml/genai/rl-distill/maze/pt_dfs25_eps05_v1; D=s3://scale-ml/genai/rl-distill/maze/data/dfs25_eps05
EX="DATA_S3=$D/sft,TRAIN_JSON=train.json,EVAL_S3=$D/eval1000,WANDB_PROJECT_NAME=tailrl-text-maze-dfs25-eps05,MAX_LENGTH=1024,MAZE_MAX_NEW_TOKENS=300,SCREEN_MAX_SEQS=512,SEEDS=0-1-2-3-4,LR=5e-4,SAVE_STEPS=25,NO_TUNE=1"
CODE_REF=<jiahaoz commit> SIZE=2XS GPUS=4 MAX_STEPS=40000 STOP_AT=0.14 ALLOW_BORROWING=false OUT_S3=$OUT TAG=maze-e25-2xs \
  EXTRA_ENVS="$EX" bash /mnt/efs/jasonwei/rl-distill-jiahaoz/rl-distill-scripts/maze/start_maze_pt_supervisor.sh
```

- **LR 5e-4 for every size**, including 3XS and 2XS. At 1e-4 the tiny models stayed on the pre-solving plateau
  (exact pass@1 ~1e-7 after 10k steps; `pt_dfs25_eps05_lr1e4_v1`).
- `SEEDS` is dash-separated: env values passed through the supervisor cannot hold spaces.
- `STOP_AT` stops a seed once the median of its last 4 screens reaches that pass@1, else it stops at `MAX_STEPS`.
- A second supervisor for the same size must use another `TAG` and its own `SUP_TAG`. Example: a borrowing copy
  `TAG=maze-e25-3xs-b ALLOW_BORROWING=true`, raced against the dedicated one. Once one of them reaches `IN_PROGRESS`,
  stop the other supervisor and cancel its jobs by ID.

## 3. Exact base tuning (devbox GPU, minutes per base)

`exact_tune.py` scores each candidate's shortest-path pass@1 exactly from teacher-forced path probabilities
(`exact_pass.py`), without sampling. It averages checkpoints around the crossing, brackets the target and interpolates
weights until within 0.25% (relative). Run it from the TailRL text-maze dir with the TailRL venv, one GPU per job queue:

```bash
cd /tmp/jw_maze/TailRL/experiments/text_maze
CUDA_VISIBLE_DEVICES=<gpu> PYTHONNOUSERSITE=1 PYTHONPATH=$PWD /tmp/jw_maze/TailRL/.venv/bin/python \
  /mnt/efs/jasonwei/rl-distill-jiahaoz/rl-distill-scripts/maze/exact_tune.py \
  --run-s3 s3://.../maze/pt_dfs25_eps05_v1/XS/XS_is0_a11090115 --level p1 --target 0.01 \
  --work /tmp/jw_kube/tune25/work --parquet <eval1000/test.parquet> \
  --center-step 100 --half-window 200 --ckpt-stride 100 --batch-tokens 65536 --out-name tune3-p1 --upload
```

- Levels and targets: `p1` 0.01, `p3` 0.03, `p10` 0.10 (also `p01` 0.001 and `p03` 0.003).
- `--upload` writes `<run>/tune3-<level>/{best,tune.json,COMPLETE}`.
- Lower `--batch-tokens` on a shared GPU.
- `select_aligned_bases.py` / `align_passk.py` pick bases whose pass@k curves match across sizes.

RL bases are copies of the tuned models, one per level and seed. RL seed i starts from base seed i:

```bash
aws s3 sync s3://.../maze/pt_dfs25_eps05_v1/XS/XS_is0_a11090115/tune3-p1/best s3://.../maze/rl_small_dfs25_tis3s_v1/bases/p1_s0
```

## 4. RL grid (GRPO + TIS)

`launch_rl_small_grid.sh` builds the run list for a `GRID` and packs it onto pods:

- longest-processing-time packing, from per-step costs in the script's cost table;
- one `start_rl_pack_supervisor.sh` per pod → `run_maze_rl_pack_st.sh` → `rl_tis_run.sh` per run.

A run spec is `<level>.s<base seed>.<prompts per step>.<RL seed>`. A run's state lives in `${OUT_S3}/runs/<suite>/`
(latest checkpoint, `best_hf/`, `COMPLETE.json`); a relaunched pod skips finished runs and resumes the rest.

Recipe (`GRID=t3s` small, `t3m` medium):

- 16 samples per prompt;
- TIS loss: weight min(π_train/π_rollout, 2), one update per step, no old-log-prob pass;
- lr 1e-5 constant after a 20-step warmup;
- validation every 20 steps on 500 dev mazes × 16 samples;
- reward 1 only for a shortest path;
- prompt cap 660, response cap 300.

Prompts per step: small 16 / 64 / 256 (+ 1024 as `t3sx`), medium 8 / 32 / 128 (+ 512 as `t3mx`). Levels p1 / p3 / p10, RL
seeds 0 / 1 / 2.

```bash
L=/mnt/efs/jasonwei/rl-distill-jiahaoz/rl-distill-scripts/maze/launch_rl_small_grid.sh
# 1-GPU pods, several runs per GPU; runs to 10k steps (patience 2000 validations never binds)
CODE_REF=<commit> FAST_DECODE=1 MAX_STEPS=10000 PATIENCE_OVERRIDE=2000 ALLOW_BORROWING=false GPUS=1 \
  GRID=t3s BATCHES="16 64 256" NPODS=7 EXTRA_ENVS_APPEND=",PREFIX_FORWARD=1" bash $L
# the large rungs, each run data-parallel on a 4-GPU pod
CODE_REF=<commit> FAST_DECODE=1 MAX_STEPS=10000 PATIENCE_OVERRIDE=2000 ALLOW_BORROWING=false GPUS=4 MICRO=1024 \
  GRID=t3sx NPODS=9 WSFX_OVERRIDE=t3s4 EXTRA_ENVS_APPEND=",PREFIX_FORWARD=1,RUN_ALL_GPUS=1,REWARD_NUM_PROCESSES=32" bash $L
```

Always run with `DRY=1` first: it prints the pod plan and submits nothing.

Knobs:

- `NPODS=N`: spreads every run without a `COMPLETE.json` over pods `<prefix>-r<GPUS>-g0..`. Use it both for first launches
  and for resumes onto a different pod count. Stop the old pods first: no run may be trained by two pods.
- `PREFIX_DECODE=1`: shared-prompt rollout. One prompt KV cache per group, finished rows dropped, CUDA graphs; 8.8×
  faster generation at 4,096 rows. The t3 grids set it by default; it needs `FAST_DECODE=1`.
- `PREFIX_FORWARD=1`: shared-prompt update. Each distinct prompt runs once; same log-probs and gradients; turns gradient
  checkpointing off.
- `RUN_ALL_GPUS=1`: one run at a time, data-parallel on all GPUs of the pod. Checkpoints are
  `model_world_size_<N>_rank_<r>.pt`, and a run cannot resume on a different world size (the pack script refuses).
- `MICRO`: per-GPU cap on samples per update micro-batch. `BATCH·16 / GPUs` must be a multiple of it (checked before Ray
  starts). 1024 OOMed with 4 heavy runs on one GPU, hence the defaults of 512 (small) and 256 (medium) when packed.
- `WSFX_OVERRIDE`: a fresh W&B id suffix when a run restarts under a new layout (`t3s4` = the 4-GPU restart). Report
  scripts take the suffix per batch, e.g. `t3s;1024:t3s4`.
- `TAG_PREFIX_OVERRIDE`: separate pod names for a subset of the grid. `EXTRA_ENVS_APPEND=",K=V"`: more pod env (the last
  value of a key wins).

Speed at 10k steps: small 1024 prompts 8.3 s/step and medium 512 prompts 4.5 s/step, both on 4 GPUs with the shared update.

### Stopping and restarting

- **Stop a pod:** `touch <state>/STOP; tmux kill-session -t "=rl-<TAG>"` (keep the quotes; zsh expands a bare `=word`).
  Then cancel its job by ID with `scale-train cancel job <ID>`; the job ID is in `<state>/monitor.log`.
- **Restart under the same TAG:** `start_rl_pack_supervisor.sh` refuses while `${OUT_S3}/pods/<TAG>.COMPLETE` exists.
  Remove that marker or use another TAG.
- **Stray jobs:** ScaleTrain can re-queue a preempted job under its old ID after the supervisor already submitted a
  replacement, so the same runs get trained twice. While a grid is live, run a pod → job check that cancels any job that
  is not its supervisor's newest, or whose supervisor is stopped. Read the job ID from the pod label
  `scaletrain/job_id`.

## 5. Status and report pages

Run these on the devbox with `.env` sourced (W&B key).

```bash
cd /mnt/efs/jasonwei/rl-distill-jiahaoz/rl-distill-scripts/maze; P=s3://scale-ml/genai/rl-distill/maze
PY=/mnt/efs/jasonwei/rl-distill/.venv/bin/python; TPY=/tmp/jw_maze/TailRL/.venv/bin/python   # TPY: the page scripts need TailRL
$PY rl_grid_status.py --root $P/rl_small_dfs25_tis3s_v1 --prefix st3 --levels p1 p3 p10 \
  --batches 16 64 256 1024 --seeds 0 1 2 --json small.json                     # best / step / state of every run
$PY rl_flops_curves.py --grid "Small (2.2M)|XS|$P/rl_small_dfs25_tis3s_v1|st3|t3s;1024:t3s4|16 64 256 1024|0 1 2" \
  --wandb-path rl-distill/tailrl-text-maze-dfs25-eps05 --out flops.json      # essential RL FLOPs per run
$TPY report_tis3.py --flops flops.json --size "Small (2.2M)|small.json|16 64 256 1024|XS" \
  --work /tmp/jw_kube/tune25/work --maze-strip <val parquet> --out tis3.html  # RL grid page (10k-step budget)
$TPY report_shortest.py --root <local mirror of pt_dfs25_eps05_v1> --sizes 3XS,2XS,XS,S,L \
  --tune-name tune3 --levels p1,p3,p10 ... --out bases.html                    # pretraining + tuned bases page
```

`rl_grid_status.py` and `rl_flops_curves.py` take the same `"label|size|root|prefix|suffix|batches|seeds"` grid strings. `report_tis3.py` truncates every run at 10,000
steps: results are reported at that budget.

## 6. Tests

The tests need a TailRL checkout with both patches (`TAILRL_ROOT`). The TailRL venv has no pytest, so use the repo's.

```bash
cd /mnt/efs/jasonwei/rl-distill-jiahaoz; export TAILRL_ROOT=/tmp/jw_maze/TailRL PY=/mnt/efs/jasonwei/rl-distill/.venv/bin/python
$PY -m pytest -q tests/special_standalone/test_rl_distill_maze_prefix_forward.py   # CPU, 12 tests incl. 2-rank FSDP, ~5 min
$PY -m pytest -q tests/special_standalone/test_rl_distill_maze_token_cache.py      # CPU, 6 tests
CUDA_VISIBLE_DEVICES=<free gpu> $PY -m pytest -q tests/special_standalone/test_rl_distill_maze_prefix_decode.py  # GPU, 14 tests
```

The patches stack: in the TailRL checkout, `tailrl_text_maze.patch` is committed on top of `5682c6a` (local commit
"maze patch"), and `tailrl_rl_tis.patch` is exactly the working-tree diff against that commit. After editing TailRL code,
`git add -N` any new file, then regenerate the patch with `git -C $TAILRL_ROOT diff HEAD > rl-distill-scripts/maze/tailrl_rl_tis.patch`.
Run `bash -n` on every edited `.sh` file. Before a long grid, run a 40-step smoke pod of each layout (1-GPU packed and 4-GPU
data-parallel). Check that the runs reach `COMPLETE.json` and that every checkpoint shard plus the latest-step pointer
reached S3.

## Gotchas

- Never write checkpoints or data to EFS: pods use `/tmp` and S3, and the devbox uses `/tmp/jw_kube`, `/tmp/jw_maze`.
  When EFS stalls, `aws` and `kubectl` hang on their EFS configs; use the `/tmp/jw_kube/{k,a}` wrappers.
- Env values passed through `--env-vars` cannot contain spaces or `$(...)`.
- In zsh, `set -- $VAR` does not word-split and an unmatched glob aborts the command: run loops with `bash -c`.
- Pods keep the last 2 RL checkpoints, and only the latest one plus `best_hf/` is pushed to S3.
- Keep devbox polling slow (≥ 3 min per loop): the shared devbox was flagged for load from kubectl / aws polling.
