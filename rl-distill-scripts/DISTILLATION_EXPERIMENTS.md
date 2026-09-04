# Gemma 4 RL → Off-Policy Distillation Experiments

Distill the **RL'd** Gemma 4 policies into **base** (pretrained) Gemma 4 models, per
DeepScaleR difficulty band, using **off-policy top-128 forward-KL**. The teacher traces are
the same top-128 trace format already collected by
`scale_train/run_gemma4_bestckpt_trace_collection.sh`.

- **Models (4):** e2b, e4b, 12b, 26b-a4b (RL'd from `google/gemma-4-{E2B,E4B,12B,26B-A4B}`).
- **Datasets (3):** DeepScaleR difficulty bands **easy / medium / hard**
  (`JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k`, 3000 train + 300 val per band).
- **Students (2):** `google/gemma-4-E2B` (base) and `google/gemma-4-E4B` (base) only.

## 1. Best RL checkpoint per (model, band) — in-distribution val (mean@16)

Metric `val-core/math/acc/mean@16` (W&B entity `rl-distill`, project `DAPO`, logged every 10
steps). ▸ = run still training remotely, score may still improve.

| Model | Band | Best step | mean@16 | HF checkpoint (repo @ pinned commit → `step_NNNNNN/`) |
|---|---|---|---|---|
| e2b | easy | 130 | 0.4102 | `JWei05/DAPO-gemma4-e2b-PT-DeepScaleR-gemma26b-easy-seed42-local2gpu` @ `c82460136fb1` → `step_000130/` |
| e2b | medium | 240 | 0.2250 ▸ | `JWei05/DAPO-gemma4-e2b-PT-DeepScaleR-gemma26b-medium-seed42-local2gpu` @ `497e7964f98b` → `step_000240/` |
| e2b | hard | 190 | 0.1363 | `JWei05/DAPO-gemma4-e2b-PT-DeepScaleR-gemma26b-hard-seed42-local2gpu` @ `59762d43bf94` → `step_000190/` |
| e4b | easy | 100 | 0.7056 | `JWei05/DAPO-gemma4-e4b-PT-DeepScaleR-gemma26b-easy-seed42-26b-bands-es5` @ `345beec132e3` → `step_000100/` |
| e4b | medium | 90 | 0.2998 | `JWei05/DAPO-gemma4-e4b-PT-DeepScaleR-gemma26b-medium-seed42-26b-bands-es5` @ `5f90d25e193d` → `step_000090/` |
| e4b | hard | 120 | 0.1590 | `JWei05/DAPO-gemma4-e4b-PT-DeepScaleR-gemma26b-hard-seed42-26b-bands-es5` @ `627bd9d825ff` → `step_000120/` |
| 12b | easy | 70 | 0.8408 | `JWei05/DAPO-gemma4-12b-PT-DeepScaleR-gemma26b-easy-seed42-26b-bands-es5` @ `372aa8417b09` → `step_000070/` |
| 12b | medium | 120 | 0.5208 | `JWei05/DAPO-gemma4-12b-PT-DeepScaleR-gemma26b-medium-seed42-26b-bands-es5` @ `485326ce84d0` → `step_000120/` |
| 12b | hard | 140 | 0.2767 | `JWei05/DAPO-gemma4-12b-PT-DeepScaleR-gemma26b-hard-seed42-26b-bands-es5` @ `162d85023909` → `step_000140/` |
| 26b-a4b | easy | 80 | 0.9408 | `JWei05/DAPO-gemma4-26b-a4b-PT-DeepScaleR-gemma26b-easy-seed42-26b-bands-es5` @ `f72b7fc8af90` → `step_000080/` |
| 26b-a4b | medium | 140 | 0.6725 ▸ | `JWei05/DAPO-gemma4-26b-a4b-PT-DeepScaleR-gemma26b-medium-seed42-26b-bands-es5` @ `4da4c943785f` → `step_000140/` |
| 26b-a4b | hard | 180 | 0.4329 ▸ | `JWei05/DAPO-gemma4-26b-a4b-PT-DeepScaleR-gemma26b-hard-seed42` @ `7659c94add2a` → `step_000180/` |

All 12 teachers are on the Hub (public, JWei05) and **pinned to an immutable commit** in
`run_gemma4_bestckpt_trace_collection.sh` (`TEACHER_HF_REPO` / `TEACHER_HF_REVISION`). The
`local2gpu` and 26b medium/hard repos are written by still-running RL jobs and keep only a
rolling window of recent steps on `main`, so always fetch by the pinned commit, not `main`.
e4b/12b/26b-easy are model-only re-uploads of the S3 full checkpoints' `actor/huggingface/`
(`scale_train/upload_fullckpt_to_hf.py`; the S3 originals remain under
`s3://scale-ml/genai/rl-distill/gemma4-difficulty-s42-20260819-full-checkpoints/`). The 12b
exports lack `processor_config.json`; the collector provisions it from `google/gemma-4-12B`.

▸ = run still training (2026-09-04): e2b-medium, 26b-a4b medium/hard keep improving; the pinned
steps are the W&B peaks at pin time. Re-pin (W&B best → HF step → commit) when you freeze them.

## 2. Teacher trace generation (all 12 teachers)

Every RL checkpoint above is a distillation **teacher** and needs off-policy top-128 traces:
**8** responses / train question + 1 / val question, training sampling (temp 1.0 / top-p 1.0 /
top-k −1), RL few-shot template, capturing `teacher_topk_token_ids` + `teacher_topk_logprobs`
(width 128), `input_ids`, `response_mask`. Output S3
`gemma4-bestckpt-traces-topk128-v2/<spec>/{train,validation}/`.

> **v1 is contaminated — do not use.** It was generated through an `hf_overrides →
> Gemma4ForCausalLM` (text-only) load added to work around the 12b exports' missing
> `processor_config.json`. That load mis-maps the multimodal rows of the LM head: their logits
> inflate (a multimodal id in the top-5 at ~34% of positions vs 2/1076 for the correct load),
> stealing softmax mass from real tokens (top-k logprobs deflated by up to 6.5 nats) and leaking
> `<image|>` into 71.6% of responses (100% of length-cap runaways; strict-correct 0.165
> contaminated vs 0.924 clean). Greedy argmax is preserved, so it evades greedy spot-checks and
> `bad_words` suppression (vLLM reports raw pre-mask logprobs). The RL rollout (verl) loads the
> native unified arch with no override — that is the distribution being distilled. Fix (v2):
> load the native arch; provision the base model's byte-compatible `processor_config.json` when
> an export lacks it; record `teacher_load_architectures` in the hashed run config so v1 shards
> are invalidated on resume. See memory `gemma4-vllm-unified-load-required`.

Trace collection status (2026-09-04): all 12 teachers are in the queue and every teacher is
fetched from its pinned Hub export (§1). Running now on this box (`tmux trace-queue-v2`, 8 GPUs)
as a head start that the nodes reuse via S3; the two-node split is in §6. Verified on the
regenerated v2 shards: 0% multimodal-token leakage, 0% length-cap runaways, width-128 top-k,
correct-rates tracking val (12b-easy 0.75, 26b-easy 0.97 on first shards).

## 3. Distillation grid (21 runs)

Teacher = best RL checkpoint (§1). Student = base model of size ≤ teacher (distill "into smaller
base models"), **plus same-size self-distillation** for e4b→e4b and e2b→e2b. Per band:

| Teacher (RL) | → e4b base | → e2b base |
|---|---|---|
| 26b-a4b | ✅ | ✅ |
| 12b | ✅ | ✅ |
| e4b | ✅ (self) | ✅ |
| e2b | — (e2b < e4b) | ✅ (self) |

**Per band: e4b base ← 3 teachers, e2b base ← 4 teachers = 7 distillations. × 3 bands = 21 runs.**
Each teacher's traces are generated once and reused across its student(s).

## 4. Distillation recipe

- **Objective:** top-128 forward-KL (`main_full_vocab_distill_fsdp2.py` /
  `gemma4_topk_distill_fsdp2.sh`; `full_vocab_kl_loss.py`). The dataset loader
  `full_vocab_distill_dataset.py` already consumes `teacher_topk_token_ids` /
  `teacher_topk_logprobs` at `teacher_topk_width=128`, `input_ids`, `response_mask` — i.e.
  **directly compatible with the traces from §2**.
- **Data:** the teacher's train traces (all 3000 q × 8 = 24,000 examples) for that band. Validation:
  128 of the teacher's own **validation-split** generations (300 val questions × 1 sample in the bundle),
  a seed-42 deterministic subset, scored by the same top-128 KL every `TEST_FREQ=10` steps
  (`build_gemma4_distill_training_view.py --validation-source validation`, the `run_gemma4_distill_one.sh`
  default; `--validation-source train` restores carving validation questions out of the train roster).
- **Hyperparameters:** global batch **64**, **1000 steps** — set epochs to **100** as a
  non-binding cap so the step count is the only limit (≈2.7 epochs over 24k examples);
  LR **2e-6**, **100** warmup steps, **linear decay to 2e-7**.
  Note: `gemma4_topk_distill_fsdp2.sh`'s strict 8-GPU / batch-128 contract applies only to the
  old `gemma4-hf-bf16-sdpa-topk-overlay-v1` index schema. These runs use derived training
  views (`gemma4-distill-training-view-v1`, built by `build_gemma4_distill_training_view.py`
  from each teacher's trace bundle), for which batch 64 on 1-2 GPUs flows through unchanged.
- **Compute:** 8-GPU FSDP2 per the audited overlay contract (`MODEL_DTYPE=fp32`, BF16 FSDP
  forward params, FP32 reductions, `Gemma4TextDecoderLayer` wrap, 4096 padded-token
  chunk, max length 12288). Runs on `.venv` (FSDP2 stack).
- **Precedent:** `e2b-base-to-e4b-topk128-lr2e6-linear-b128-2ep` used this exact schedule shape.
- **Logging / artifacts:** W&B project `gemma4-bestckpt-distill-v2` (console + wandb), validation every 10
  steps. No periodic local checkpoints (`SAVE_FREQ=0`): the trainer saves once at the final step, that save
  holds only the HF export (`checkpoint.save_contents=["hf_model"]`, no FSDP/Adam shards), it is pushed to
  `JWei05/Distill-gemma4-<teacher_spec>-to-<student>-base/step_001000/` and the local copy is deleted after
  the upload succeeds (`HF_PUSH_DELETE_LOCAL=true`). Storage is HF-only: no S3 (`TRACE_S3_MIRROR_ENABLE=false`,
  `DISTILL_S3_ENABLE=false`).

## 5. Open decisions / TODO

1. Confirm the **21-run grid** in §3 (teacher set per student).
2. ~~Upload the S3-only best checkpoints to HF~~ ✅ done 2026-09-04 (all 12 now on HF).
3. **Trace corrections + additions** (§2 TODO): fix e4b steps; add 5 HF-sourced teacher specs.
4. Where to **run the 21 distillations** — this 8-GPU box vs ScaleTrain (overlay needs 8 GPUs).
5. Whether the still-training runs (e2b medium/hard, 26b-a4b medium/hard) should be **frozen at
   the current best** or allowed to finish before their traces are generated.

## 6. Two-node execution plan (plain local scripts — no ScaleTrain)

Everything below is plain bash run directly on a node (the scripts live under
`rl-distill-scripts/scale_train/` for historical reasons; nothing depends on ScaleTrain, and
`launch_gemma4_bestckpt_trace_matrix.py` is not used). Each node generates its teachers'
traces, then distills from them. Both phases are async GPU-pool queues; the distill queue
launches a run as soon as *its* teacher's trace bundle is COMPLETE, so it can be started right
after the trace queue on the same node.

Inputs are all on the Hub: teachers (the 12 best RL checkpoints, §1), datasets
(`JWei05/DeepScaleR-…`), students (`google/gemma-4-E4B` / `-E2B`), and the base
`processor_config.json` the 12b exports lack. **S3 is optional**: with scale-ml S3 write
access, shards mirror to `…-topk128-v2/` and nodes cooperate (completed shards are restored
and skipped); without it, set `TRACE_S3_MIRROR_ENABLE=false` and `DISTILL_S3_ENABLE=false`
and everything stays on local disk (traces under `/tmp/gemma4_bestckpt_traces_v2/`, views
under `/tmp/gemma4_distill_views/`). Prereqs per node: `.env` with `HF_TOKEN` (+ `WANDB_API_KEY`,
AWS creds if mirroring), the gemma4 venv (`bash rl-distill-scripts/setup_env_gemma4.sh`,
default `/tmp/.venv-gemma4`), 8 GPUs.

Split rationale: pair each node's slow teacher with a fast one so distillation starts early,
and balance total GPU-hours (generation + the distill runs that consume that node's teachers).
Rough GPU-hour budget (8 samples/q; distill ≈ 2 GPU·h per e2b-student run, 4 per e4b-student
run): 26b ≈ 17 gen + 18 distill, 12b ≈ 10 + 18, e4b ≈ 5 + 18, e2b ≈ 3 + 6 → node 1 ≈ 44,
node 2 ≈ 52 (vs ≈ 58 / 38 for a 26b+e4b / 12b+e2b split). Each node distills only from teachers
it generated, so there is no cross-node dependency.

| | Node 1 (teachers 26b + e2b) | Node 2 (teachers 12b + e4b) |
|---|---|---|
| **Traces** | 26b easy/medium/hard (2 GPUs, **TP2**) · e2b easy/medium/hard (1 GPU) | 12b easy/medium/hard (2 GPUs, DP2) · e4b easy/medium/hard (2 GPUs, DP2) |
| **Distill** | 26b→e4b (2 GPUs) · 26b→e2b, e2b→e2b (1 GPU) — **9 runs** | 12b→e4b, e4b→e4b (2 GPUs) · 12b→e2b, e4b→e2b (1 GPU) — **12 runs** |

```bash
# ---- node 1 ----
TRACE_QUEUE_SPECS=26b-easy:2,26b-medium:2,26b-hard:2,e2b-easy:1,e2b-medium:1,e2b-hard:1 \
  VENV=/tmp/.venv-gemma4 GPU_MEMORY_UTILIZATION=0.72 \
  bash rl-distill-scripts/scale_train/run_gemma4_bestckpt_trace_queue.sh
DISTILL_QUEUE_RUNS=26b-easy:e4b:2,26b-medium:e4b:2,26b-hard:e4b:2,26b-easy:e2b:1,26b-medium:e2b:1,26b-hard:e2b:1,e2b-easy:e2b:1,e2b-medium:e2b:1,e2b-hard:e2b:1 \
  bash rl-distill-scripts/scale_train/run_gemma4_distill_queue.sh

# ---- node 2 ----
TRACE_QUEUE_SPECS=12b-easy:2,12b-medium:2,12b-hard:2,e4b-easy:2,e4b-medium:2,e4b-hard:2 \
  VENV=/tmp/.venv-gemma4 GPU_MEMORY_UTILIZATION=0.72 \
  bash rl-distill-scripts/scale_train/run_gemma4_bestckpt_trace_queue.sh
DISTILL_QUEUE_RUNS=12b-easy:e4b:2,12b-medium:e4b:2,12b-hard:e4b:2,e4b-easy:e4b:2,e4b-medium:e4b:2,e4b-hard:e4b:2,12b-easy:e2b:1,12b-medium:e2b:1,12b-hard:e2b:1,e4b-easy:e2b:1,e4b-medium:e2b:1,e4b-hard:e2b:1 \
  bash rl-distill-scripts/scale_train/run_gemma4_distill_queue.sh
```

Single runs: `TEACHER_SPEC=12b-easy STUDENT=e4b DISTILL_GPU_IDS=0,1 bash
rl-distill-scripts/scale_train/run_gemma4_distill_one.sh` (recipe defaults: bs 64, 1000 steps,
epochs cap 100, lr 2e-6, warmup 100, linear → 2e-7; pin `STUDENT_REVISION` for reproducible
student identities). Smoke-test one teacher on a node with `TRACE_MAX_SHARDS=1` /
one `run_gemma4_distill_one.sh` before the full queues.

Trace/distill identity: the generator hashes only its own source + config (not the git
commit), so nodes may be at different commits as long as `generate_gemma4_distill_traces.py`
is byte-identical; a change to that file requires a new trace version (bump the `-v2` prefixes).
