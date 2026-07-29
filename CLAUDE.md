# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [`verl-project/verl`](https://github.com/verl-project/verl) used as a research
codebase ("rl-distill") for RL training and distillation of **Gemma 3** models (dense,
dense→MoE upcycled, PT and IT variants) on **B200 / H100**. Almost all of the fork-specific
work lives in two places, which this file focuses on:

- `rl-distill-scripts/` — launch scripts, docs, and custom code (datasets, losses, MoE
  converter, validation harnesses) for DAPO RL and distillation.
- `lm-evaluation-harness/` — an **unmodified** upstream EleutherAI clone (v0.4.13.dev0), used
  standalone to evaluate trained checkpoints. It is *not* imported by the training code.

`verl/` is the upstream framework, edited only lightly in this fork (see
[custom fork changes](#custom-fork-changes-in-verl)).

> **`CLAUDE.md` was formerly a symlink to `AGENTS.md`.** `AGENTS.md` is the upstream
> **verl contribution policy** — read it before proposing any PR to `verl-project/verl`
> (duplicate-work checks, no busywork PRs, human-accountable review). Those rules still apply;
> this file adds the fork-specific architecture that the symlink didn't cover.

## Two isolated environments — do not mix them

The two training stacks have incompatible pins and live in separate venvs at the repo root.
Pick the venv that matches the task; the wrong `vllm`/`transformers`/`torch` will fail at import
or model load.

| Venv | Built by | Stack | Used for |
|---|---|---|---|
| `.venv` | `rl-distill-scripts/setup_env.sh` | torch 2.9.1+cu128, vllm 0.15.1, transformers 4.57.6, flash-attn 2.8.3 (built) | FSDP2 DAPO RL + FSDP2 distillation |
| `.venv-megatron` | `setup_megatron.sh` | torch 2.10+cu129, vllm 0.18.0, transformers 5.3.0, Megatron-Core v0.16.0, vendored `third_party/Megatron-Bridge`, TransformerEngine v2.12 | Megatron DAPO RL, incl. Gemma 3 MoE |

```bash
# FSDP2 stack (one-shot, per fresh node)
bash rl-distill-scripts/setup_env.sh && source .venv/bin/activate

# Megatron / MoE stack (H100 example)
UV_NO_CONFIG=1 PIP_INDEX_URL=https://pypi.org/simple CUDA_HOME=/usr/local/cuda-12.9 \
  TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=64 bash setup_megatron.sh
# B200: TORCH_CUDA_ARCH_LIST=10.0
```

Both require a repo-root `.env` (untracked) with `HF_TOKEN=...` and `WANDB_API_KEY=...`. Launch
scripts source it and forward both into Ray's `runtime_env.env_vars`.

Editable install (`uv pip install --no-deps -e .`) is mandatory for the Megatron venv — vLLM
discovers the native `verl_gemma3_moe` MoE plugin through this package's entry point.

## Architecture — where the runtime code actually is

There are **two near-duplicate sibling directories**, and which one runs depends on how the entry
point is invoked. This is the most common source of confusion; know it before editing.

- **`dapo/`** (repo-root package) is the RL runtime. All `*.sh` RL launchers in
  `rl-distill-scripts/` invoke `python3 -m dapo.main_dapo`, so `dapo/main_dapo.py`,
  `dapo/dapo_ray_trainer.py`, and `dapo/config/*.yaml` are what execute — **not** the
  identically-named files under `rl-distill-scripts/`.
- `rl-distill-scripts/main_dapo.py` / `dapo_ray_trainer.py` are currently byte-identical copies of
  the `dapo/` versions, but they can drift. When changing RL trainer behavior, **edit `dapo/`** (or
  both, deliberately). `rl-distill-scripts/config/` similarly mirrors `dapo/config/`.
- **Distillation** is the exception: its launchers `cd rl-distill-scripts/` then
  `torchrun main_distill_offpolicy.py` / `main_full_vocab_distill_fsdp2.py`, so those entry points
  run **in place** from `rl-distill-scripts/`.

### Three training modalities

| Modality | Entry (invoked) | Config | Launch scripts | Venv |
|---|---|---|---|---|
| DAPO RL, FSDP2 | `dapo.main_dapo` (`dapo_trainer`) | `dapo/config/dapo_trainer.yaml` | `gemma3_{4b,12b,27b}_it_fsdp2_20k.sh`, `gemma3_4b_pt_*_b200_1node.sh` | `.venv` |
| DAPO RL, Megatron (dense + MoE) | `dapo.main_dapo --config-name=dapo_megatron_trainer` | `dapo/config/dapo_megatron_trainer.yaml` | `gemma3_4b_pt_megatron_20k.sh`, `gemma3_4b_pt_moe_*megatron*.sh` | `.venv-megatron` |
| Distillation, FSDP2 (SFT-based) | `main_distill_offpolicy.py` (forward-KL) / `main_full_vocab_distill_fsdp2.py` (top-k KL) | `config/distill_offpolicy.yaml`, `config/full_vocab_distill_fsdp2.yaml` | `gemma3_4b_it_distill_offpolicy.sh`, `gemma3_27b_pt_full_vocab_distill_from_4b_pt_fsdp2.sh` | `.venv` |

`main_dapo.py` subclasses verl's `TaskRunner` → `DAPOTaskRunner` and swaps in `RayDAPOTrainer`
(custom `fit()` with trajectory saving, pre-filter accuracy, and extra metrics). Distillation
subclasses verl's `SFTTrainer` and swaps the loss via `set_loss_fn()` — no verl core changes.

### Distillation internals

- Off-policy forward-KL: `distill_dataset.py` (`DistillSFTDataset`, loads pre-generated
  `teacher_log_probs`/`teacher_token_ids` from parquet) + `forward_kl_loss.py`
  (`log p_teacher − log p_student` on the sampled token; same gradient as CE, but a proper ≥0
  divergence). See `rl-distill-scripts/DISTILLATION.md`.
- Full-vocab top-k KL: `full_vocab_distill_dataset.py` + `full_vocab_kl_loss.py`. See
  `FULL_VOCAB_DISTILLATION_PLAN.md`.
- Teacher data is generated **offline** (`data/generate_teacher_data.py`, TP=2 data-parallel
  shards via `data/launch_teacher_gen.sh`, merged by `data/merge_teacher_shards.py`).

### Reward scoring

Math RL/eval routes through `verl/utils/reward_score/math_verify.py` (`LatexExtractionConfig`
only — **only `\boxed{}` answers score**; bare numbers do not). Strict single-box grading is the
default (`VERL_MATH_VERIFY_STRICT_BOXED=0` restores lenient): responses with zero or multiple
`\boxed{}` score 0. `verl/utils/reward_score/__init__.py` routes `math`, `math_dapo`, `math500`,
`olympiadbench`, `minervamath`, `gsm8k`, `beyondaime`, and any `aime*` source to it, returning
`{"score": s, "acc": s > 0.5, "pred": <extracted boxed answer>}` (`pred` feeds verl's maj@N
machinery). CPU test: `tests/utils/reward_score/test_math_verify_strict_boxed_on_cpu.py`.

## Common commands

```bash
# Data prep (writes to ${HOME}/verl/data by default)
bash rl-distill-scripts/data/prepare_all_datasets.sh          # DAPO-Math-17k + AIME/MATH500/... val sets
bash rl-distill-scripts/data/prepare_dapo_17k_split.sh        # train/test split used by MoE smoke + H100 wrapper
python3 rl-distill-scripts/data/split_dapo_openmath2_mix.py   # DAPO+OpenMathInstruct2 mix (seed 42)

# FSDP2 RL (single B200 node)
bash rl-distill-scripts/gemma3_4b_it_fsdp2_20k.sh

# Megatron dense RL (single node); MEGATRON_VENV defaults to repo-local .venv-megatron
MEGATRON_VENV="$PWD/.venv-megatron" bash rl-distill-scripts/gemma3_4b_pt_megatron_20k.sh

# MoE: known-good one-node 2E H100 recipe (after validation gates pass — see below)
UPCYCLED_MOE_DIR="$MOE_DIR" bash rl-distill-scripts/gemma3_4b_pt_moe_2e_megatron_rl_1node_h100.sh
```

There is no unit-test suite for `rl-distill-scripts/` — correctness is enforced by the MoE
validation gates and by comparing training metrics (entropy, response length, router loss) against
a dense baseline. verl's own tests live in `tests/` and run via `pytest`.

## Gemma 3 dense→MoE workflow (read `GEMMA3_MOE_RL_TRAINING.md` first)

The single most important rule: **do not start a long MoE run until validation gates A–D pass.**

1. **Convert** dense → MoE: `gemma3_moe_hf/create_gemma3_moe_from_dense_hf.py` duplicates each
   dense MLP into every expert and adds a deterministic random router. It emits two outputs — a
   normal **sparse** checkpoint (for training) and a **canonical** view (expert-0-over-full-batch,
   for exact tests only; **never train it**). Must print `GEMMA3_MOE_CHECKPOINT_VERIFIED`.
2. **Gates** (all must be exact, not "99%"):
   - A — HF layerwise parity: `check_gemma3_dense_moe_activations.py` → `HF_DENSE_MOE_ACTIVATION_PARITY_OK`
   - B — Megatron actor parity at prod topology: `check_gemma3_mcore_dense_moe_activations.py` → `MCORE_DENSE_MOE_ACTIVATION_PARITY_OK`
   - C — native vLLM parity: `diagnose_gemma3_vllm_parity.py` → `NATIVE_VLLM_DENSE_MOE_PARITY_OK`
   - D — bounded end-to-end RL round trip: `gemma3_4b_pt_moe_megatron_correctness_1node_h100.sh`
3. **Runtime defaults that must hold** (correctness, not tuning knobs): `ROLLOUT_MODEL_IMPL=native`
   (the generic vLLM Transformers backend is diagnostic-only — it produced entropy ≈9 and
   non-terminating responses), `ROLLOUT_ATTENTION_BACKEND=TRITON_ATTN`, `ROUTER_REPLAY_MODE=R2`.
   Entropy near 9 or generations hitting `MAX_RESPONSE_LENGTH` is a **correctness alarm** — never
   mask it by tuning entropy coeffs or response limits.

Router replay: **R2** (record routes in the actor's old-logprob pass, replay for the update) is the
supported default. `disabled` is diagnostic. **R3** (vLLM captures generation routes) is
experimental and gated behind `ALLOW_EXPERIMENTAL_R3=True`; a guardrail raises on an all-zero replay
map rather than silently routing every token to expert 0. Use R2 for real runs.

The Megatron venv forces a **fresh single-node local Ray** (`RAY_ADDRESS=local`, isolated
`_temp_dir`) so it never attaches to the long-lived FSDP2-venv Ray cluster (which would spawn
workers in the wrong interpreter → `AssertionError: Unknown backend: megatron`).

## lm-evaluation-harness

An unmodified upstream EleutherAI clone used standalone to evaluate trained checkpoints. **Treat it
as a vendored tool**: don't hand-edit it to fix bugs — those go upstream; local edits to a vanilla
clone are easy to lose and hard to review. It has its own git repo, `.venv`/install, and
`pyproject.toml` (extras: install a backend explicitly).

```bash
cd lm-evaluation-harness
pip install -e ".[vllm]"      # base package no longer bundles torch/transformers; vllm>=0.18

# Evaluate a checkpoint (vLLM backend). New CLI also supports `lm_eval run|ls|validate`.
lm_eval --model vllm \
  --model_args pretrained=/path/to/checkpoint/huggingface,dtype=bfloat16,gpu_memory_utilization=0.9 \
  --tasks aime,gsm8k --batch_size auto
```

Tasks are YAML/Python under `lm_eval/tasks/`; the evaluation engine is `lm_eval/evaluator.py` and
model backends are `lm_eval/models/`. Note the harness's task-level answer extraction is not the
same as this repo's `\boxed{}`-only `math_verify` reward — numbers in `EVAL_RESULTS.md` come from
the repo's own math scoring, so don't compare the two head-to-head without checking the scorer.

## Checkpoints, resume, and HF upload

- A `SAVE_FREQ` save writes `global_step_N/{actor,actor/huggingface,data.pt}`. `save_contents`
  includes `hf_model` (a weight-only HF snapshot for inference); resume `load_contents` uses only
  `model`/`optimizer`/`extra`. The `huggingface/` dir is an inference artifact, **not** an Adam
  checkpoint.
- Exact resume: `RESUME_MODE=resume_path trainer.resume_from_path=<STEP_DIR>` — keep data paths,
  shuffle seed, batch size, response count, and DP layout unchanged, and ensure `data.pt` exists.
- HF Hub upload runs **only on local-save steps**: set `HF_PUSH_ENABLE=True`, `HF_PUSH_REPO=...`,
  and make `HF_PUSH_FREQ` a multiple of `SAVE_FREQ`.

## Custom fork changes in verl

Beyond `dapo/` and `rl-distill-scripts/`, the fork touches verl narrowly:
`verl/utils/reward_score/{__init__.py,math_verify.py}` (routing + `\boxed{}`-only scoring) and
`verl/trainer/ppo/ray_trainer.py` (per-sample `response_length` logged in validation). Prefer adding
new behavior in `dapo/`/`rl-distill-scripts/` over editing `verl/` core.

## Gotchas

- `sp_size=1` always for Gemma 3 — it loads as a VLM and the sequence-parallel path has a
  `temperature_rmpad` shape bug.
- FSDP wrap must be explicit: `wrap_policy.transformer_layer_cls_to_wrap=["Gemma3DecoderLayer"]`
  (auto-discovery fails for the Gemma 3 VLM).
- `transformers` must be 4.57.6 in `.venv` (5.x breaks vLLM 0.15.1's `rope_scaling` check).
- `ray stop --grace-period 30` only — `--force` breaks `/proc`. `_clean_restart.sh` also kills
  orphan GPU PIDs from `nvidia-smi`.
- NCCL interface: single node `NCCL_SOCKET_IFNAME=lo` / `AF_INET`; multi-node `bond0` / `AF_INET6`.
- Megatron on B200: `TORCH_CUDA_ARCH_LIST="10.0"` **quoted** (Hydra otherwise coerces it to float
  and megatron-core's import-time JIT crashes).
