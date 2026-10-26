# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [`verl-project/verl`](https://github.com/verl-project/verl) used as the research codebase "rl-distill":
**RL vs distillation** for math reasoning on **Gemma 4 E2B / E4B / 12B** and **Qwen3 base 1.7B / 4B / 8B (+14B RL)**, trained on
**H100** ScaleTrain (EKS `p5.48xlarge`) pods. Three recipes are in scope:

1. **DAPO/GRPO RL** (`python3 -m dapo.main_dapo`, FSDP2 + vLLM).
2. **Off-policy top-k 128 forward-KL distillation with a tail bucket** (`main_full_vocab_distill_fsdp2.py`).
3. **On-policy distillation, policy-gradient form ("PG-k1")**, run through the RL trainer with a colocated vLLM teacher.

Start with [`rl-distill-scripts/README.md`](rl-distill-scripts/README.md) (entry point + directory map), then
[`docs/RECIPES.md`](rl-distill-scripts/docs/RECIPES.md) (exact hyperparameters), [`docs/OPERATIONS.md`](rl-distill-scripts/docs/OPERATIONS.md)
(launch / supervise / checkpoint / S3 / W&B / gotchas) and the results docs
([Qwen3](rl-distill-scripts/docs/QWEN3_RESULTS.md), [Gemma 4](rl-distill-scripts/docs/GEMMA4_RESULTS.md)).
[`rl-distill-scripts/PROGRESS_LOG.md`](rl-distill-scripts/PROGRESS_LOG.md) is the running experiment log (newest first): add an
entry per experiment (goal, config, scripts, results, status) and keep it at that path.

Gemma 3, MoE upcycling, Megatron, SFT, few-shot RL, B200 recipes, Ouro/Huginn/IFM and the NeMo-RL repro were removed on
2026-10-26; they are in git history (last pre-cleanup commit `8aefaa5a`).

> **`AGENTS.md` is the upstream verl contribution policy.** Read it before proposing any PR to `verl-project/verl`
> (duplicate-work checks, no busywork PRs, human-accountable review). Those rules still apply; this file covers the fork.

## Where the runtime code is

- **`rl-distill-scripts/`** holds all fork-specific code. Its root keeps the runtime core *flat on purpose*: `main_dapo.py`,
  `dapo_ray_trainer.py` (`RayDAPOTrainer`: custom `fit()`, early stopping, rolling/permanent S3 checkpoints, best_hf), `hf_push.py`,
  `config/`, `run_dapo_rl.sh` (the Hydra command every RL/OPD run ends in), the distill trainer
  (`main_full_vocab_distill_fsdp2.py`, `full_vocab_distill_dataset.py`, `full_vocab_kl_loss.py` with `tail_bucket`,
  `gemma4_topk_distill_fsdp2.sh`) and `full_checkpoint_s3.py`. The distill trainer runs with cwd `rl-distill-scripts/` and imports
  siblings by bare name (`from full_checkpoint_s3 import ...`), so don't move these into subfolders without fixing imports.
- **`dapo/`** (repo root) is the package `python3 -m dapo.main_dapo` imports: `__init__.py`, `main_dapo.py`, `dapo_ray_trainer.py`,
  `hf_push.py` and `config/` are **symlinks** into `rl-distill-scripts/`; `dapo/validation_early_stopping.py` is a real file.
  Editing `rl-distill-scripts/dapo_ray_trainer.py` edits the runtime trainer.
- **`rl-distill-scripts/scale_train/`**: devbox launchers + pod run-files. `run_gemma4_pt_deepscaler_4of4strict_rl.sh` is the pod
  run-file for **all** RL and on-policy distillation (Gemma 4 and Qwen3, despite the name); `launch_qwen3_30b_a3b_medium_rl.sh` is the
  Qwen3 RL launcher for **every** `QWEN_SIZE` (despite the name). Gemma-4-only scripts are in `scale_train/gemma4/`.
- Other folders: `data/` (band builders, trace generation, samplers/graders, prompts), `eval/` (+ `eval/local/`), `analysis/`,
  `utils/`, `docs/`, `figures/`.
- **`verl/`** is upstream, edited narrowly (reward `verl/utils/reward_score/{math_verify,miles_sympy}.py` — only the last well-formed
  `\boxed{}` scores; on-policy distillation in `verl/trainer/distillation/`; Gemma 4 / Qwen3 FSDP engine fixes; vLLM port base;
  router replay). Don't reorganize upstream dirs (`verl/`, `recipe/`, `examples/`, `docs/`, upstream `tests/`). Prefer new behavior in
  `rl-distill-scripts/` or `dapo/`. Some Gemma 3 MoE fork edits remain in `verl/vllm/` (+ the `vllm.general_plugins` entry point in
  `pyproject.toml`); they are dormant for Gemma 4 / Qwen3.
- `lm-evaluation-harness/` is a vendored, unmodified EleutherAI gitlink used only by the Gemma 4 OOD evals
  (`eval/eval_gemma4_ood.py`). Don't hand-edit it.

## Environments

Both venvs live at the repo root of the **main checkout** (`/mnt/efs/jasonwei/rl-distill`) and are editable installs pointing at it.
When testing a worktree, set `PYTHONPATH=<worktree>:<worktree>/rl-distill-scripts` and check `python -c "import verl; print(verl.__file__)"`.

| Venv | Build | Use |
|---|---|---|
| `.venv` | `bash rl-distill-scripts/setup_env.sh` (torch 2.9.1+cu128, vLLM 0.15.1, transformers 4.57.6) | devbox tooling (S3 janitors, republish, W&B backfill), most CPU tests; baked into the ScaleTrain image. Cannot load Gemma 4. |
| `.venv-gemma4` | `GEMMA4_CUDA_VARIANT=cu129 bash rl-distill-scripts/setup_env_gemma4.sh` (torch 2.11, vLLM 0.25.1, transformers 5.14.1, no flash-attn) | every training / eval job (Gemma 4 **and** Qwen3; pods build it into `/tmp`), Gemma 4 CPU tests (`tests/workers/engine/test_gemma4_distill_logits_on_cpu.py`). |

A repo-root `.env` (untracked) holds `HF_TOKEN` / `WANDB_API_KEY`; `launch_st_job.py` forwards them to jobs.

## Launching and monitoring

- Jobs run on ScaleTrain from the **main checkout**: `launch_*.sh` → `launch_st_with_code.sh` (ships `git archive <CODE_REF>`, default
  HEAD, over a pre-built image) → `launch_st_job.py` → pod run-file. **Commit before launching**; uncommitted changes are not shipped.
- Long runs go through `start_*_supervisor.sh` → `supervise_borrowing_job.py` in tmux session `rl-<name>` (state in
  `.scale_train_supervisors/`), which relaunches after preemption / failure until the S3 completion receipt + `best_hf` exist and pins
  `CODE_REF` to the commit the run started on. Rolling checkpoints are pruned from the devbox by `ckpt-janitor-<name>` sessions
  (`full_checkpoint_s3.py cleanup-rolling`; pods can't delete S3 objects). `borrowing_fallback_watcher.sh` switches dedicated jobs
  queued > 60 min to borrowing.
- Stop a supervised run: `touch <state>/STOP`, then `tmux kill-session -t =rl-<name>` — **always use the `=` exact-match target**
  (a bare name prefix-matches other sessions).
- Live supervisors relaunch by running these paths from the main checkout, so **don't move, rename or change the CLI/env interface** of:
  `rl-distill-scripts/full_checkpoint_s3.py` and `rl-distill-scripts/scale_train/{supervise_borrowing_job.py, launch_st_with_code.sh,
  launch_st_job.py, launch_qwen3_onpolicy_distill.sh, launch_qwen3_30b_a3b_medium_rl.sh, start_qwen3_onpolicy_distill_supervisor.sh,
  start_qwen3_30b_a3b_medium_rl_supervisor.sh, run_gemma4_pt_deepscaler_4of4strict_rl.sh, launch_qwen3_opd_recipe_v2_batch.sh,
  borrowing_fallback_watcher.sh}` (+ the scripts the watcher's job table restarts). The run-file must also exist at that path in the
  pinned `CODE_REF`.
- Dry runs that never submit: `launch_qwen3_opd_recipe_v2_batch.sh` with no args; `launch_st_job.py --dry-run ...`;
  `DRY_RUN=1 bash rl-distill-scripts/run_dapo_rl.sh` (Hydra config only). `launch_st_with_code.sh` is *not* a dry run (it uploads the tarball).
- Debugging ScaleTrain: `scale-train get job <ID>` is TTY-only; use `env -u AWS_PROFILE kubectl get pods -n train` / `kubectl logs -n train <pod>`
  (instance role; the kubectl SSO token silently returns empty when expired). Pod-log markers: `CODE_BOOTSTRAPPED`, `RUN_DONE rc=N`.

## Tests

There is no unit-test suite for most launcher scripts; correctness is checked by the CPU tests below plus training metrics.

```bash
cd <worktree>; export PYTHONPATH=$PWD:$PWD/rl-distill-scripts
/mnt/efs/jasonwei/rl-distill/.venv/bin/python -m pytest -q tests/special_standalone/test_rl_distill_*.py \
  tests/utils/reward_score/test_math_verify_strict_boxed_on_cpu.py tests/workers/utils/test_losses.py rl-distill-scripts/data/tests
/mnt/efs/jasonwei/rl-distill/.venv-gemma4/bin/python -m pytest -q tests/workers/engine/test_gemma4_distill_logits_on_cpu.py
```

Known pre-existing failures (stub tokenizer `SimpleNamespace` has no `len()`): 4 in `tests/workers/engine/test_full_vocab_distill_dataset.py`,
12 in `rl-distill-scripts/data/tests/test_gemma4_distill_traces.py`, 2 dataset tests in `test_gemma4_distill_logits_on_cpu.py`.

## Gotchas (repo / infra facts)

- `git commit --no-verify`: the pre-commit hooks can time out and eat unstaged work (recoverable from `.cache/pre-commit` patches).
  The main checkout has many long-lived uncommitted files: stage explicit paths, rebase only with `--autostash`, and check the dirty
  count is unchanged afterwards.
- ScaleTrain EKS pods are **IPv6-only**: NCCL/Gloo need `NCCL_SOCKET_FAMILY=AF_INET6` + `eth0` (`AF_INET` → "no socket interface
  found"); GKE a3-mega pods are IPv4. The run-file picks from `SCALE_CLUSTER`.
- Pods run the run-file in a login shell that **resets PATH** (image `ENV PATH` dropped): call venv binaries by absolute path. Job
  commands are `string.Template`-rendered: never put `$(...)` or `$VAR` in them.
- The `ml-worker` AWS profile can submit jobs and write `s3://scale-ml` but has no `ecr:GetAuthorizationToken`: reuse the known-good
  `--image`; don't build.
- Never write checkpoints / HF exports / traces / caches to EFS (`/mnt/efs`, $HOME): one shared NFS connection; `assert_local_fs.sh`
  refuses it. Use `/tmp` + S3.
- Gemma 4: FSDP wrap `Gemma4TextDecoderLayer` (E2B/E4B) / `Gemma4UnifiedTextDecoderLayer` (12B); no flash-attn → `use_remove_padding=False`
  + `attn_implementation=sdpa`; weight-sync bucket `rollout.checkpoint_engine.update_weights_bucket_megabytes=8192`;
  `VERL_GEMMA4_CUDNN_SDPA=0` on E4B (`TORCH_CUDNN_SDPA_ENABLED=0` alone never reaches `update_actor`).
- Gemma 4 in vLLM: load the native unified architecture; never `hf_overrides` → `Gemma4ForCausalLM` (inflates multimodal-token
  logits, corrupts top-k logprobs). Copy the base model's `processor_config.json` next to exports missing it. Gemma 4 has no
  `<end_of_turn>` token: stops are literal strings. `gemma4-bestckpt-traces-topk128-v1` is contaminated; use v2.
- `.venv-gemma4` pins `torchcodec==0.15.0` (0.17 needs `libnvrtc.so.13` → `import vllm` fails on cu129 pods).
- Concurrent RL jobs on one host need distinct `VERL_VLLM_PORT_BASE` (≥1000 apart); `VLLM_PORT` is overwritten.
- verl's logged `best@N` is a with-replacement bootstrap (≈ pass@10 for N = 16); `mean@N` is exact. Use offline evals for pass@k.
- Distilling into an untied-head student (Qwen3-8B and larger; the 30B-A3B control used it) needs `VERL_FSDP2_WRAP_EMBED_LM_HEAD=0`
  and `VERL_FSDP2_LOCAL_LOAD=0` (otherwise NCCL hangs in the first distill validation); see `run_gemma4_distill_one.sh`.
- W&B `run.scan_history()` drops steps logged after a resume; use `run.history(keys=[key], samples=...)` per key and merge by `_step`.
  Re-logging into an old run id drops steps ≤ its last logged step (use a new id for weight-only continuations).
- Process hygiene on the shared devbox: never `pkill -f <pattern>` (matches the killing shell); `tmux kill-session` SIGHUPs a pane's
  background children (start long jobs with `setsid`); only `scale-train cancel` jobs you can name by ID.
- `sp_size=1` always (the Gemma VLM sequence-parallel path has a shape bug).
