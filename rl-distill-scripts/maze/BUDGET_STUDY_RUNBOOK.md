# Budget study runbook (text-maze RL vs distillation, aiai-rtx-2, 8 GPUs)

Goal (set 2026-10-27; design doc section "Budget study"): three matched S/L pairs at goal_reached pass@1 ≈ 10% (`p10`,
frozen), 1% (`p1`) and 0.1% (`p01`); per pair, five total-FLOPs budgets from S₀ RL's own saturation, Arm B = S₀ RL at
that budget, Arm A = L₀ RL for a fraction ρ ∈ {0.5, 0.75, 0.9} of the budget then distillation (on-policy PG-k1 and
off-policy exact forward KL) with the rest; readout = goal_reached pass@1 on dev (final numbers on test1000).
Reward everywhere: `binary` (goal_reached). All scripts live in `rl-distill-scripts/maze/`; `PY=~/TailRL/.venv/bin/python`.
Data root: `R=~/tailrl_maze_runs` (`rl/`, `select/`, `matched/`, `budget/<pair>/`, `checkpoints/tailrl-text-maze/`).

Status at a glance: `bash queue_status.sh`.

## Stage A — selection of the lower-level pairs (GPUs 2–5, launched 2026-10-27 00:20)
Jobs: `select_matched_init.py --bracket … --out-name select_goal_reached_{1pct,0p1pct}.json` on `pretrain20k/S_ds0_is4`
and `pretrain20k/L_ds0_is0_rep`; logs `$R/select/<run>_<level>.log`; done when the log ends with `wrote …`.
- 1%: k=64 grid, finals k=1024, n-final 3 (S bracket 9000–9400 step 20, L 3180–3460).
- 0.1%: k=256 grid, finals k=2048, n-final 3 (S 8000–9000 step 40, L 2600–3040 step 40). Slow (hours).

## Stage B — freeze the pair and start its RL
```
CUDA_VISIBLE_DEVICES=<free> $PY make_pair.py --pair p1  --target 0.01  --tol 0.0005 --k 1024 --select-name select_goal_reached_1pct.json   --s-run $R/pretrain20k/S_ds0_is4 --l-run $R/pretrain20k/L_ds0_is0_rep
CUDA_VISIBLE_DEVICES=<free> $PY make_pair.py --pair p01 --target 0.001 --tol 0.0001 --k 2048 --select-name select_goal_reached_0p1pct.json --s-run $R/pretrain20k/S_ds0_is4 --l-run $R/pretrain20k/L_ds0_is0_rep
```
(interpolates between adjacent averaged candidates if the finals are not within tol; exit 2 = best effort outside tol —
still usable, report the gap). Then:
```
S_STEPS=3000 L_STEPS=1500 bash launch_pair_rl.sh p1  <gpuS> <gpuL>     # SAVE_FREQ 5, TEST_FREQ 10, VAL_N 16
S_STEPS=4000 L_STEPS=2000 bash launch_pair_rl.sh p01 <gpuS> <gpuL>
```
p10's RL is already running: suites `p10-s0` (GPU 0, 600 steps, ckpt every 2) and `p10-l0` (GPU 1, 300 steps, ckpt every 2),
logs `$R/rl/p10_{S0,L0}_rl.log`. Re-running a launch with the same suite resumes it (auto resume).

## Stage C — plan the budgets and run the matrix (needs S₀ saturated: ≥ 8 dev points and a plateau)
```
$PY plan_budgets.py --pair <pair> --s-root $R/checkpoints/tailrl-text-maze/textmaze_<pair>-s0_grpo_binary_ckpt0_bs256_G16_1gpu_<g>_seed0 \
   --l-root $R/checkpoints/tailrl-text-maze/textmaze_<pair>-l0_grpo_binary_ckpt0_bs256_G16_1gpu_<g>_seed0 \
   --s-log $R/rl/<pair>_S0_rl.log --l-log $R/rl/<pair>_L0_rl.log --out $R/budget/<pair>/plan.json
bash run_budget_matrix.sh $R/budget/<pair>/plan.json <gpu> [gpu …]      # round-robin workers; DONE markers make it re-runnable
```
Check the plan table first: the L₀ run must have checkpoints up to the largest ρ·B (0.9·B5) — if not, let it run longer
(resume with a larger STEPS) before launching those cells; infeasible small cells are listed and skipped. Once the plan
exists, the L₀ run may be stopped when its cumulative FLOPs exceed 0.9·B5 (free the GPU: `pkill -TERM -s <session leader pid>`;
never `pkill -f`). Workers: `$R/budget/<pair>/jobs/worker_gpu<g>.log`, per-job `<cell>-{opd,off}.{log,DONE,FAILED}`.
A FAILED job: read its log, fix, delete the FAILED marker, re-run `run_budget_matrix.sh` (skips DONE jobs).

## Stage D — evaluate
```
CUDA_VISIBLE_DEVICES=<free> $PY collect_budget_results.py --plan $R/budget/<pair>/plan.json --k 64            # dev, cached
CUDA_VISIBLE_DEVICES=<free> $PY collect_budget_results.py --plan $R/budget/<pair>/plan.json --k 256 --parquet ~/tailrl_maze_data/test1000/test.parquet
```
Prints the pass@1-by-budget table (Arm B | per ρ: teacher / OPD / off-policy). Record results in the design doc.

## Priorities when GPUs free up
1. make_pair + RL for a pair whose selection finished (2 GPUs per pair).  2. Matrix workers for a planned pair (any free GPU; add
GPUs by re-running `run_budget_matrix.sh` with more GPU ids — DONE jobs are skipped, but a job already running on another
worker is not detected, so only add GPUs after the first workers finished or for a new pair).  3. Evaluation.
Each tick: `queue_status.sh`; tracebacks in RL logs, FAILED jobs, disk (`df -h /home`; `$R/pretrain/` = 78 GB of superseded 6k-step runs, delete only with the user's OK).

## Setting up on another machine

Git holds only the code: nothing under `~/tailrl_maze_data`, `~/tailrl_maze_runs` or `~/TailRL/.venv` is committed. Everything
below rebuilds them from TailRL's public corpus and this study's Hub archive,
https://huggingface.co/datasets/hi-todayis-jh/tailrl-text-maze-budget-study (`used/<pair>/<label>/` = every evaluated checkpoint
incl. the matched bases, `archive/<run>.tar` = every RL run's HF exports + `flops_ledger.json` + console log, `meta/` = plans,
results, manifests). Keep the same layout under `$HOME` (`~/TailRL`, `~/tailrl_maze_data`, `~/tailrl_maze_runs`): every script
here defaults to it. The one-off helpers (`handoff_*.sh`, `prune_*.sh`, `stage_hub_archive.sh`) hardcode `/home/jiahaoz4` and
are not needed.

1. **TailRL fork.** `git clone https://github.com/Zanette-Labs/TailRL ~/TailRL && git -C ~/TailRL checkout 5682c6a`, then
   `git -C ~/TailRL apply <rl-distill>/rl-distill-scripts/maze/tailrl_text_maze.patch` (11 files: OPD estimator + teacher
   reference policy, FLOPs ledger + `MAX_TOTAL_TFLOPS` stop, `ATTN_IMPL` knob, flash-attn-free workers,
   `sft_pretrain.py --data_seed/--init_seed`, `src/maze_zero_reward.py`, `.uv-constraints.txt`; verified to apply to `5682c6a`).
2. **Environment** `~/TailRL/.venv` (Python 3.10, built with `uv`): torch 2.8.0 for your CUDA (`.uv-constraints.txt` pins
   `torch==2.8.0`; cu128 here), `experiments/text_maze/requirements.txt`, then pin **transformers 4.57.6** (5.x loads these
   checkpoints silently wrong), ray 2.59.0, and add `torchdata==0.11.0` + `peft==0.21.2` (imported by the vendored verl, absent
   from its requirements). flash-attn is not needed: every launcher here runs `ATTN_IMPL=sdpa`.
3. **`~/TailRL/.env`** (gitignored; sourced by `scripts/env.sh`): `TAILRL_MAZE_DATA_DIR=$HOME/tailrl_maze_data`,
   `HF_CACHE_DIR=$HOME/.cache/tailrl_maze_sft_ckpts`, `TAILRL_MAZE_RUN_DIR=$HOME/tailrl_maze_runs`, `WANDB_MODE=offline`.
   Then `ln -s eval1000 ~/TailRL/experiments/text_maze/data/main_parquet_eval1000` (train.sh's default validation dir;
   `data/eval1000` is committed in TailRL).
4. **Data** (~12 GB, all derived from the public corpus `max-rl/maze_17x17_diverse_1.3m`), from `~/TailRL/experiments/text_maze`:
   `bash scripts/download_dataset.sh && TEST_PROMPTS=1000 bash scripts/prepare_dataset.sh` → `main_parquet/` (1,298,992 train /
   1000 test rows) + `sft_data/` (pretraining json). Dev = `cp -r data/eval1000 ~/tailrl_maze_data/eval1000` (identical to the
   SFT held-out set; TailRL's default 256-prompt split would leak 744 dev mazes into training). Test = `~/tailrl_maze_data/test1000/`,
   1000 fresh mazes: `python src/build_dataset.py --m 1000 --seed_start 10000000 --out ~/tailrl_maze_data/test1000/test_fresh.jsonl`
   (other args default), then the two `to_rl_parquet.py` calls of `prepare_dataset.sh` with `--in_path .../test_fresh.jsonl
   --out_dir ~/tailrl_maze_data/test1000 --test_prompts 1000` (`--variants maze_17_continuous`, and `--variants maze_17 --canonical`
   → `test.parquet`). Generation is seeded, so this reproduces the set used here (not re-verified on another machine).
5. **Starting checkpoints** (instead of re-running the pretraining matrix + `make_pair.py`; `pretrain20k/` is not archived): for
   each pair `p10`, `p1`, `p01` download `used/<pair>/S0_base` and `used/<pair>/L0_base` from the Hub and lay them out as
   `~/tailrl_maze_runs/matched/<pair>_S0` (the HF dir) + `~/tailrl_maze_runs/matched/<pair>_S0_base/ckpt-0 -> ../<pair>_S0` (same
   for `L0`); `meta/matched_manifest.json` → `matched/manifest.json`. `launch_pair_rl.sh` and `run_budget_matrix.sh` take it from there.
6. **Reusing finished runs instead of recomputing.** Extract `archive/<pair>-s0.tar` / `<pair>-l0.tar` into
   `~/tailrl_maze_runs/checkpoints/tailrl-text-maze/` (they unpack as `textmaze_<suite>_grpo_binary_…/global_step_N/actor/huggingface`
   + ledgers, plus the console log → move it to `~/tailrl_maze_runs/rl/`), and copy `meta/<pair>/plan.json` + `results_*.json` to
   `~/tailrl_maze_runs/budget/<pair>/`. Those files store absolute `/home/jiahaoz4/…` paths: rewrite the prefix
   (`sed -i "s#/home/jiahaoz4#$HOME#g"`) if `$HOME` differs; `meta/used_manifest.json` maps every `used/` entry to its original run
   path. Not reproducible anywhere: resuming an RL run with its optimizer state (`*.pt` were deleted everywhere) and re-screening or
   re-averaging the pretraining trajectories (pruned; the 1.3 GB remnant lives only on aiai-rtx-2). The teacher traces of the
   off-policy cells (112 GB here) are not archived; `gen_teacher_traces.py --seed 0` regenerates them from the Hub teachers.
7. **GPU memory.** Everything is one GPU per job. The study ran on 96 GB cards; evaluation (`collect_budget_results.py` →
   `screen_sft_ckpts.screen_one`, ≤ 8192 sequences per generate call) peaked at 40–60 GB at k = 256 — on smaller cards lower the
   `8192 // k` cap in `screen_sft_ckpts.py`. Never co-locate an evaluation with an RL run or trace generation.
