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
