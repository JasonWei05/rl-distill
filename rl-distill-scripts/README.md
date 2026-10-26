# rl-distill: RL vs distillation on Gemma 4 and Qwen3

Last updated: 2026-10-26.

This directory holds everything fork-specific in this [verl](https://github.com/verl-project/verl) fork. The research question
is **when does distilling a stronger teacher beat training the student directly with RL** on math reasoning, and why. Both
are compared at matched data, prompts, graders and (where stated) FLOPs.

| | In scope |
|---|---|
| Models | Gemma 4 **E2B, E4B, 12B** (PT). Qwen3 base **1.7B, 4B, 8B**; **14B** RL was added 2026-10-26. Gemma 4 26B-A4B and Qwen3-30B-A3B appear only as teachers / band definers. |
| Recipes | (1) **DAPO/GRPO RL**. (2) **Off-policy top-k 128 forward-KL distillation** with the **tail bucket**. (3) **On-policy distillation, policy-gradient form (PG-k1)**. |
| Data | DeepScaleR, split into easy / medium / hard difficulty bands by a large base model's pass rate. |
| Hardware | H100 80 GB, ScaleTrain EKS `p5.48xlarge` (whole node or fractional `:1` / `:2` / `:4`). |

Everything else (Gemma 3, MoE upcycling, Megatron, SFT, few-shot RL, B200 recipes, Ouro/Huginn/IFM, NeMo-RL repro, local
one-off scripts) was removed on 2026-10-26 and is in git history (the last commit before the cleanup is `8aefaa5a`).

## Documentation map

| Doc | Read it for |
|---|---|
| [docs/RECIPES.md](docs/RECIPES.md) | The three recipes: objectives (incl. the tail-bucket and PG-k1 derivations), **exact current hyperparameters** with code pointers, per-size memory layouts, launch commands, eval protocols. |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Venvs, the ScaleTrain launch chain, supervisors / borrowing / janitors, checkpoints and resume, best_hf, S3 and W&B layout, monitoring, and the full gotcha list. |
| [docs/QWEN3_RESULTS.md](docs/QWEN3_RESULTS.md) | All Qwen3 results: RL grid, off-policy matrix (incl. 30B-teacher students), OPD v1/v2, pre-training vs capacity ablation, math suite, FLOP-matched comparison, artifact index. |
| [docs/QWEN3_MATH_SUITE_RESULTS.md](docs/QWEN3_MATH_SUITE_RESULTS.md) | Generated 8-set math-suite table (`eval/summarize_qwen3_math_suite.py`). |
| [docs/GEMMA4_RESULTS.md](docs/GEMMA4_RESULTS.md) | All Gemma 4 results: RL difficulty sweep + seeds, 21-run distillation grid, E4B-base → 12B control, teacher / LR ablations, full eval matrix, artifact index. |
| [PROGRESS_LOG.md](PROGRESS_LOG.md) | Running experiment log, newest first. Append an entry per experiment (goal, config, scripts, results, status). |
| [figures/](figures/) | Plots referenced by the results docs (+ the JSON / MD each analysis script writes next to its PNG). |

## Directory map

```
rl-distill-scripts/
├── README.md, PROGRESS_LOG.md
│
│   # Runtime core (flat on purpose: `dapo/` symlinks to these and the distill trainer imports them by bare name)
├── __init__.py, main_dapo.py, dapo_ray_trainer.py, hf_push.py   # RL trainer = `python3 -m dapo.main_dapo` (see dapo/ below)
├── config/                         # Hydra: dapo_trainer.yaml (RL/OPD), full_vocab_distill_fsdp2.yaml (distill); Gemma 4 eval registries
├── run_dapo_rl.sh                  # builds the dapo.main_dapo Hydra command (every RL / OPD run ends here; DRY_RUN=1 prints the config)
├── main_full_vocab_distill_fsdp2.py, full_vocab_distill_dataset.py, full_vocab_kl_loss.py   # off-policy top-k distillation trainer
├── gemma4_topk_distill_fsdp2.sh    # audited torchrun launcher for the distill trainer (used for Gemma 4 *and* Qwen3 students)
├── full_checkpoint_s3.py           # S3 checkpoint protocol: restore-latest, publish-best-hf, complete-run, cleanup-rolling, ...
├── assert_local_fs.sh              # refuses checkpoints / caches on NFS/EFS
├── setup_env.sh, setup_env_gemma4.sh, patch_vllm_gemma4_r3.py   # venv builders (.venv / .venv-gemma4)
│
├── scale_train/                    # ScaleTrain launchers (devbox side) and run-files (pod side)
│   ├── launch_st_with_code.sh, launch_st_job.py       # git-archive tarball + pre-built image -> `scale-train train`
│   ├── supervise_borrowing_job.py, borrowing_fallback_watcher.sh   # relaunch-until-done supervisor, queued->borrowing switch
│   ├── run_gemma4_pt_deepscaler_4of4strict_rl.sh      # THE pod run-file for RL and on-policy distillation (Gemma 4 + Qwen3)
│   ├── run_gemma4_bestckpt_trace_collection.sh, run_gemma4_distill_one.sh   # shared off-policy stages (traces, one distill run)
│   ├── launch_qwen3_30b_a3b_medium_rl.sh / start_qwen3_30b_a3b_medium_rl_supervisor.sh   # Qwen3 RL, every QWEN_SIZE
│   ├── launch_qwen3_onpolicy_distill.sh / start_qwen3_onpolicy_distill_supervisor.sh / launch_qwen3_opd_recipe_v2_batch.sh  # PG-k1 OPD
│   ├── run_qwen3_30bm_to_4b_distill_st.sh (+ launch_/start_)   # Qwen3 off-policy distill: traces -> distill -> evals, any TEACHER_SPEC/STUDENT
│   ├── run_qwen3_{math_suite,passk32_evals,own_nll,band_evals}_st.sh (+ launch_/start_), run_deepscaler_samples_st.sh   # Qwen3 eval jobs
│   ├── republish_best_hf.py, publish_qwen3_opd_best_hf.sh, stop_qwen3_opd_at_step.sh, run_qwen3_opd_math_suite_watcher.sh
│   ├── wandb_backfill_step_from_pod_log.py, build_es_migration_from_wandb.py, list_team_jobs.py, mirror_hf_to_s3.py, upload_fullckpt_to_hf.py
│   ├── gemma4/                     # Gemma-4-only launchers, supervisors, queues and eval/trace run-files
│   └── st_config/                  # Dockerfiles + build_manifest.yaml (images are pre-built; see OPERATIONS §2)
├── data/                           # dataset builders (DeepScaleR bands), trace generation / validation / S3 + HF bundles,
│   │                               # eval-data prep, samplers + graders (generate_deepscaler_4bpt_samples.py, grade_band_eval_samples.py),
│   │                               # own-sample NLL, prompt templates (*.jinja)
│   └── tests/                      # CPU tests of the trace / view / preflight tooling
├── eval/                           # Gemma 4 eval stack (eval_math_passk.py, run_gemma4_math_4gpu.py, eval_gemma4_ood.py, ...),
│   │                               # reverse_kl_topk.py, result summarizers / doc updaters
│   └── local/                      # devbox (local-GPU) band-val re-evals: Qwen3 RL best-checkpoint selection, Gemma 4 train-sample eval
├── analysis/                       # FLOP-matched plot, distill-vs-teacher / pre-distill reports, entropy, diversity, pass@k plots
├── utils/                          # reshard_fsdp2_checkpoint.py (FSDP2 world-size reshard), recover_wandb_from_podlog.py, patches/ (Ray on the devbox)
├── docs/                           # see the documentation map above
└── figures/
```

Outside this directory the fork touches:

- `dapo/` (repo root): the package `python3 -m dapo.main_dapo` imports. `__init__.py`, `main_dapo.py`, `dapo_ray_trainer.py`, `hf_push.py`
  and `config/` are **symlinks** into `rl-distill-scripts/`; `dapo/validation_early_stopping.py` (early stopping) is a real file.
- `verl/`: narrow fork edits (reward `verl/utils/reward_score/{math_verify,miles_sympy}.py` + routing, on-policy distillation in
  `verl/trainer/distillation/`, Gemma 4 / Qwen3 FSDP engine fixes, vLLM rollout port base, router replay, checkpoint helpers).
  Find them with `git log --author=Jason -- verl/`. Prefer new behavior in `rl-distill-scripts/` or `dapo/`.
- `tests/special_standalone/test_rl_distill_*.py` and a few `tests/{utils,workers,trainer}/...` fork tests.
- `lm-evaluation-harness/` (vendored upstream EleutherAI gitlink, unmodified): only the Gemma 4 OOD evals use it (`eval/eval_gemma4_ood.py`).

## Environments

Two uv venvs at the repo root, both editable installs of this repo (`uv pip install --no-deps -e .`):

| Venv | Build | Stack | Used for |
|---|---|---|---|
| `.venv` | `bash rl-distill-scripts/setup_env.sh` | torch 2.9.1+cu128, vLLM 0.15.1, transformers 4.57.6, flash-attn 2.8.3 | Devbox tooling (S3 janitors, republish, W&B backfill, launchers); baked into the ScaleTrain image (`aws` for the code bootstrap). Cannot load Gemma 4. |
| `.venv-gemma4` | `GEMMA4_CUDA_VARIANT=cu129 bash rl-distill-scripts/setup_env_gemma4.sh` | torch 2.11, vLLM 0.25.1, transformers 5.14.1, no flash-attn, torchcodec pinned 0.15.0 | **Every training and eval job, Gemma 4 and Qwen3.** Pods build it into `/tmp/.venv-gemma4` at job start (cu129: the p5 fleet has a CUDA 12.8 driver). |

A repo-root `.env` (untracked) holds `HF_TOKEN`, `WANDB_API_KEY` (and optionally `WANDB_BASE_URL`); `launch_st_job.py`
forwards them to the job. Details and pins: [OPERATIONS §1](docs/OPERATIONS.md#1-environments).

## The three recipes (current settings)

Full tables with code pointers: [docs/RECIPES.md](docs/RECIPES.md). All three use the same prompts, grader and bands:

- **Prompts.** Gemma 4: the 12-shot `data/gemma3_it_fewshot_math.jinja` template (the Gemma 4 prompt despite its name), stops on
  the literal `<end_of_turn>` / `<start_of_turn>` text. Qwen3: 0-shot native ChatML `data/qwen3_base_native_chat.jinja` with a
  `\boxed{}` instruction, stop ids `<|im_end|>` / `<|im_start|>`.
- **Reward / grader.** `verl/utils/reward_score/math_verify.py`: only the **last well-formed `\boxed{}`** is graded,
  `math_verify` OR a bounded SymPy fallback (`miles_sympy.py`); no box = 0.
- **Bands.** Qwen3: `JWei05/DeepScaleR-Easy-Medium-Hard-Qwen3-30B-A3B-Base-10k` (30B-A3B base k/8: easy ≥ 4, medium 2–3, hard 1;
  3,000 train / 300 val per band; very_hard discontinued 2026-10-25). Gemma 4: bands defined by the Gemma 4 26B-A4B PT pass rate
  (`DIFFICULTY_DATASET_SOURCE=gemma4_26b_bands`). Validation = the band's val300 × 16 (mean@16).

### 1. DAPO / GRPO RL

GRPO, n = 16 responses, 64 prompts / step with mini-batch 32 (2 optimizer steps per step), AdamW lr **1e-6** (20 warmup,
constant), clip 0.2 / 0.28, token-mean loss, no KL / entropy terms, prompt 4096 / response **8192** + 2048 soft overlong buffer,
token TIS (`rollout_is_threshold=2.0`), temp 1 / top-p 1 for train and val. Validation before training and every 10 steps;
**early stopping** on `val-core/math/acc/mean@16` (Qwen: patience 5 evals with a 1-point miss tolerance, `es5`; 14B uses
patience 10, `es10`); cap 400 steps by default (many runs extended to 500–1000). Rolling resumable checkpoint every 5 steps,
permanent checkpoint (+ HF export) every 10, `best_hf/` published at the end.

Call chain: `launch_*` → `launch_st_with_code.sh` → `launch_st_job.py` → pod `run_gemma4_pt_deepscaler_4of4strict_rl.sh` →
`run_dapo_rl.sh` → `python3 -m dapo.main_dapo` (`RayDAPOTrainer`).

| Size | GPUs | vLLM | Notes |
|---|---|---|---|
| Qwen3-1.7B / 4B / 8B | 2 / 4 / 4 | TP 1, one engine per GPU | KV 24 / 24 / 16 GiB; no offload |
| Qwen3-14B | 8 | 4 × TP 2 | 16 GiB KV per GPU, no offload |
| Gemma 4 E2B / E4B | 4 | TP 1 | `Gemma4TextDecoderLayer`; E4B: `OFFLOAD=True`, run with `VERL_GEMMA4_CUDNN_SDPA=0` |
| Gemma 4 12B | 8 | TP 1 | `Gemma4UnifiedTextDecoderLayer`, FSDP CPU-offload policy |

### 2. Off-policy top-k 128 forward-KL distillation (+ tail bucket)

Teacher traces: per prompt, sampled responses (temp 1, 8192 tokens) with the teacher's **top-128 token ids + log-probs at every
position** (`data/generate_gemma4_distill_traces.py`). The student minimizes forward KL over the teacher's top-128 plus, with
the **tail bucket** (`TOPK_KL_TAIL_BUCKET=true`), one extra "everything else" bin: teacher mass `1 − Σ top-128`, student mass
`logsumexp` over the complement of the teacher's top-128 (derivation in [RECIPES §2.2](docs/RECIPES.md#22-objective-derived-from-full_vocab_kl_losspy)).
The tail bucket is **off by default** in the launcher; only the 2026-10-26 Qwen base→base runs (`q1p7b-base-all` /
`q4b-base-all`) set it. Set it explicitly for new runs.

v2 recipe (current): 3,000 train q × 16 samples + 300 val q, global batch 128 sequences, lr **2e-6** (100 warmup, linear →
2e-7), **1000 steps**, val KL every 10, permanent S3 checkpoint every 250; FSDP2 fp32 master, bf16 params. Entry points:
`scale_train/run_qwen3_30bm_to_4b_distill_st.sh` (Qwen3: traces → distill → band evals in one resumable job) and
`scale_train/gemma4/run_gemma4_12br_trace_and_distill_st.sh` / `run_gemma4_e4b_base_distill_st.sh` (Gemma 4); both reach
`run_gemma4_distill_one.sh` → `gemma4_topk_distill_fsdp2.sh` → `torchrun main_full_vocab_distill_fsdp2.py`.

### 3. On-policy distillation, policy-gradient form (PG-k1)

The student samples one response per prompt; a colocated vLLM teacher scores the sampled tokens; per-token advantage
**A_t = sg[log π_teacher(y_t) − log π_student(y_t)]** (k1, no normalization, no task reward), clipped to **[−5, 5]**, with an
**IcePop** token mask (trainer/vLLM ratio outside [0.5, 2] → weight 0). 128 prompts × 1 sample, one update per step (PPO clip
inactive), temp 1, same RL contract otherwise (val every 10 on the band's val300 × 16).

| | v1 (`pgk1-v1`, `pgk1-base-v1`, `pgk1-pre-v1`) | **v2 (`pgk1-base-lr1e6w50`, current)** |
|---|---|---|
| LR | 2e-6, 20 warmup, constant | **1e-6, 50 warmup, constant** |
| Steps / stop | 120 (grid: 250), no early stopping | cap **400**; stop when no +0.5-pt val gain in 100 steps (patience 10, `min_delta` 0.005) |
| Capacity | borrowing false | **borrowing true**, priority high |

`SPEC=<teacher>[-s<STEP>]-to-<student>[-base]`, e.g. `q4bm-s510-to-q1p7b-base` = the 4B medium RL run's step-510 checkpoint
teaching the untrained 1.7B base on the medium band. Launch: `scale_train/launch_qwen3_onpolicy_distill.sh` (one run),
`start_qwen3_onpolicy_distill_supervisor.sh` (supervised), `launch_qwen3_opd_recipe_v2_batch.sh` (the v2 batch; no args = dry
run). The 2026-09 Gemma 4 on-policy runs used a different, supervised reverse-KL-on-top-k loss that is retired
([RECIPES §3.5](docs/RECIPES.md)).

## Launching on ScaleTrain (H100)

Launch from the **main checkout** with a clean, committed tree: `launch_st_with_code.sh` ships `git archive <CODE_REF>`
(default `HEAD`) as a tarball and unpacks it over the pre-built image's `/workspace/rl-distill`. Uncommitted changes are not
shipped.

```bash
cd rl-distill-scripts/scale_train
# RL (Qwen3, any size / band) under the relaunching supervisor + rolling-checkpoint janitor (tmux rl-q3-<size>-rl-<band>)
QWEN_SIZE=4b BAND=medium bash start_qwen3_30b_a3b_medium_rl_supervisor.sh
#   RL from a pre-distilled init: INIT_MODEL_S3_URI=<s3 HF export> RUN_TAG=<new tag> ...
# RL (Gemma 4 E2B/E4B seed x band sweep; 12B layout)
SEED=43 SIZES="e2b e4b" BANDS="easy medium hard" bash gemma4/start_gemma4_rl_seed_supervisors.sh
bash gemma4/launch_gemma4_12b_distilled_rl_medium.sh
# Off-policy distillation (Qwen3; traces + distill + evals, resumable)
TEACHER_SPEC=q4bh-hard STUDENT=q1p7b GPUS=2 bash start_qwen3_30bm_to_4b_distill_supervisor.sh
# On-policy distillation (PG-k1 v2 batch)
bash launch_qwen3_opd_recipe_v2_batch.sh            # dry run: resolved env per spec
bash launch_qwen3_opd_recipe_v2_batch.sh --launch   # one supervisor per spec (skips finished / running ones)
```

What happens after launch ([OPERATIONS §3–§5](docs/OPERATIONS.md)):

- **Supervisor** (`supervise_borrowing_job.py`, one tmux session `rl-<name>` per run, state in `.scale_train_supervisors/<name>-<date>/`):
  relaunches after preemption / failure / cancel until the S3 completion receipt (and `best_hf`) exists. Every relaunch ships the
  `CODE_REF` the run started on (pinned at supervisor start). Stop: `touch <state>/STOP`, then `tmux kill-session -t =rl-<name>`
  (note the `=`: exact-match target).
- **Borrowing.** RL and OPD v2 launch with borrowing true (preemptible); dedicated (`ALLOW_BORROWING=false`) jobs queued > 60 min are
  switched to borrowing by `borrowing_fallback_watcher.sh`.
- **Resume.** On every pod start the run-file restores the newest *complete* S3 checkpoint (`full_checkpoint_s3.py restore-latest`):
  model + Adam + LR/RNG + dataloader cursor + early-stopping state, so a preemption costs at most 5 steps.
- **Checkpoints.** Rolling slot every 5 steps (the previous one is deleted from the devbox by the `ckpt-janitor-<name>` tmux
  session, since pods cannot delete S3 objects) + permanent `global_step_N/` with the HF export every 10.
- **Outputs.** `s3://scale-ml/genai/rl-distill/<family>/<run>/best_hf/` + `run_outcome.json`, checkpoints under
  `<family>-full-checkpoints/<run>/` with `run_complete.json`. Qwen RL families: `qwen3-<size>-base-rl[-full-checkpoints]`;
  OPD: `qwen3-opd[-full-checkpoints]/<SPEC>-<RUN_TAG>`. HF Hub pushes are off (S3 only).
- **W&B.** Entity `rl-distill`; RL / OPD in project `DAPO` with fixed run ids (`q3-<size>-<band>-<RUN_TAG>-s42-v1`,
  `q3-opd-<SPEC>-<RUN_TAG>-s42`), so relaunches resume the same run. Distillation logs to `<family>-distill-v1` projects.
- **After a run.** `republish_best_hf.py` / `publish_qwen3_opd_best_hf.sh` re-publish `best_hf` at the true best saved step
  (needed after early stopping with `min_delta`); `stop_qwen3_opd_at_step.sh` stops an OPD run cleanly at a step.

## Evaluation

- **In-loop**: band val300 × 16, `val-core/math/acc/mean@16` (exact). verl's logged `best@N` is a **with-replacement bootstrap**
  (≈ pass@10 for N = 16): compare it only with other RL logs, never with offline pass@k.
- **Qwen3 offline**: `scale_train/run_qwen3_math_suite_st.sh` (8 sets: easy/medium/hard val, GSM8K, MATH500, AIME 2025/2026;
  models in `qwen3_math_suite_models.sh`, table via `eval/summarize_qwen3_math_suite.py`), `run_qwen3_passk32_evals_st.sh`
  (pass@1..32), `run_qwen3_own_nll_st.sh` (own-sample NLL = per-token entropy estimate), all with
  `data/generate_deepscaler_4bpt_samples.py` + `data/grade_band_eval_samples.py` (unbiased pass@k). Devbox re-evals for best-step
  selection: `eval/local/run_qwen3_band_val_evals_local.sh`, `eval/local/run_qwen3_rl_ckpt_candidates_local.sh`.
- **Gemma 4 offline**: `eval/eval_math_passk.py` + `eval/run_gemma4_math_4gpu.py` (registered math matrix, run with
  `EVAL_PREDICTIVE_TOPK_WIDTH=0`), `eval/eval_gemma4_ood.py` (MMLU-Pro / GPQA-Diamond / MMMLU-14k via `lm-evaluation-harness`),
  student-checkpoint pass@k (`eval/eval_student_checkpoints_passk.py`), offline reverse KL (`eval/reverse_kl_topk.py`).
  Pod jobs: `scale_train/gemma4/run_gemma4_eval_one_model_st.sh`, `run_gemma4_rl_distill_eval_*`.
- **Analysis / figures**: `analysis/qwen3_flops_matched_opd_vs_rl.py` (FLOP-matched OPD vs RL), `analysis/qwen3_distill_vs_teacher_report.py`,
  `analysis/qwen3_predistill_inspection.py`, `analysis/qwen3_opd_entropy_from_logs.py`, `analysis/qwen3_answer_diversity.py`,
  `analysis/plot_*.py`; outputs in `figures/`.

## Gotchas (the ones that cost runs)

Full list: [OPERATIONS §7](docs/OPERATIONS.md#7-gotchas).

- **Paths live supervisors depend on.** Supervisors relaunch from the main checkout by running these exact paths, so don't move
  or rename them while runs are live: `scale_train/{supervise_borrowing_job.py, launch_st_with_code.sh, launch_st_job.py,
  launch_qwen3_onpolicy_distill.sh, launch_qwen3_30b_a3b_medium_rl.sh, start_qwen3_onpolicy_distill_supervisor.sh,
  start_qwen3_30b_a3b_medium_rl_supervisor.sh, run_gemma4_pt_deepscaler_4of4strict_rl.sh, launch_qwen3_opd_recipe_v2_batch.sh,
  borrowing_fallback_watcher.sh}` and `full_checkpoint_s3.py`. (The run-file path must also exist in the pinned `CODE_REF`.)
- **Commit before launching**; pods run `git archive` of a commit. Use `git commit --no-verify` (pre-commit hooks can time out and eat work).
- ScaleTrain EKS pods are **IPv6-only**: NCCL/Gloo need `AF_INET6` + `eth0` (the run-file sets this from `SCALE_CLUSTER`).
- Pods run the run-file in a login shell that **resets PATH**: use absolute venv paths; job commands are `string.Template`-rendered (no `$(...)`).
- Never write checkpoints, caches or traces to EFS (`/mnt/efs`); `assert_local_fs.sh` enforces it. Use `/tmp` + S3.
- Gemma 4 in vLLM: always load the **native unified architecture** (never `hf_overrides` → `Gemma4ForCausalLM`; copy the base
  model's `processor_config.json` next to exports that lack it). The `-v1` best-checkpoint trace bundles are contaminated; use v2.
- `VERL_GEMMA4_CUDNN_SDPA=0` for E4B (cuDNN SDPA is slower and crashes on RL's variable shapes); `TORCH_CUDNN_SDPA_ENABLED=0` alone does nothing.
- Concurrent jobs on one host need distinct `VERL_VLLM_PORT_BASE` (setting `VLLM_PORT` does nothing).
- Killing things: never `pkill -f <pattern>` (matches your own shell); `tmux kill-session` SIGHUPs background children (use
  `setsid`); only ever `scale-train cancel` jobs you can name by ID.
- W&B `run.scan_history()` drops steps logged after a resume: query `run.history(keys=[key])` per key and merge by `_step`.

## Follow-ups (after the live runs finish)

- Rename `scale_train/launch_qwen3_30b_a3b_medium_rl.sh` → `launch_qwen3_rl.sh` (and its supervisor), and
  `run_gemma4_pt_deepscaler_4of4strict_rl.sh` → a model-neutral name; move shared Qwen3 / common scripts into
  `scale_train/{qwen3,common}/` like `scale_train/gemma4/`.
- Drop the remaining 26B / 30B-only branches from the run-file and launchers.
