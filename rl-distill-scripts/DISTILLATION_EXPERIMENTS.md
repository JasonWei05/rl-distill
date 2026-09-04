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

| Model | Band | Best step | mean@16 | Best checkpoint location |
|---|---|---|---|---|
| e2b | easy | 130 | 0.4102 | HF `JWei05/DAPO-gemma4-e2b-PT-DeepScaleR-gemma26b-easy-seed42-local2gpu` `/step_000130` |
| e2b | medium | 190 | 0.2117 ▸ | HF `JWei05/DAPO-gemma4-e2b-PT-DeepScaleR-gemma26b-medium-seed42-local2gpu` `/step_000190` |
| e2b | hard | 190 | 0.1363 ▸ | HF `JWei05/DAPO-gemma4-e2b-PT-DeepScaleR-gemma26b-hard-seed42-local2gpu` `/step_000190` |
| e4b | easy | 100 | 0.7056 | S3 `…/e4b-easy/global_step_100` — **now on HF** `…-26b-bands-es5/step_00NNNN` |
| e4b | medium | 90 | 0.2998 | S3 `…/e4b-medium/global_step_90` — **now on HF** `…-26b-bands-es5/step_00NNNN` |
| e4b | hard | 120 | 0.1590 | S3 `…/e4b-hard/global_step_120` — **now on HF** `…-26b-bands-es5/step_00NNNN` |
| 12b | easy | 70 | 0.8408 | S3 `…/12b-easy/global_step_70` (HF `…-12b-…-easy-…-26b-bands-es5` exists) |
| 12b | medium | 120 | 0.5208 | S3 `…/12b-medium/global_step_120` — **now on HF** `…-26b-bands-es5/step_00NNNN` |
| 12b | hard | 140 | 0.2767 | S3 `…/12b-hard/global_step_140` — **now on HF** `…-26b-bands-es5/step_00NNNN` |
| 26b-a4b | easy | 80 | 0.9408 | S3 `…/26b-a4b-easy/global_step_80` — **now on HF** `…-26b-bands-es5/step_00NNNN` |
| 26b-a4b | medium | 120 | 0.6702 ▸ | HF `JWei05/DAPO-gemma4-26b-a4b-PT-DeepScaleR-gemma26b-medium-seed42-26b-bands-es5` `/step_000120` |
| 26b-a4b | hard | 160 | 0.4306 ▸ | HF `JWei05/DAPO-gemma4-26b-a4b-PT-DeepScaleR-gemma26b-hard-seed42` `/step_000160` |

S3 root: `s3://scale-ml/genai/rl-distill/gemma4-difficulty-s42-20260819-full-checkpoints/`.
Each S3 `global_step_N/actor/huggingface/` is a loadable consolidated HF model (verified).

**Checkpoint availability: all 12 best models are now on HF (JWei05).** The 7 S3-only ones were
uploaded 2026-09-04 (model-only `actor/huggingface/`, into `step_NNNNNN/` subdirs of
`…-<band>-seed42-26b-bands-es5` repos). The e2b `local2gpu` and 26b medium/hard repos were
already there.

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

Status of the local trace queue (`run_gemma4_bestckpt_trace_queue.sh`, tmux `trace-queue`):

| Teacher | In current queue? | Teacher source | Notes |
|---|---|---|---|
| 12b easy/medium/hard | 🔁 regenerating → v2 | S3 (s70/s120/s140) | exports lack `processor_config.json` → provisioned from `google/gemma-4-12B` |
| 26b-a4b easy | ✅ queued | S3 (s80) | correct |
| e4b easy | ✅ queued | S3 (s100) | step corrected to the W&B peak (was s130) |
| e4b medium | ✅ queued | S3 (s90) | step corrected to the W&B peak (was s100) |
| e4b hard | ✅ queued | S3 (s120) | correct |
| e2b easy/medium/hard | ❌ **not in queue** | HF `…-local2gpu` (s130/s190/s190) | add specs; source from HF |
| 26b-a4b medium | ❌ **not in queue** | HF `…-26b-bands-es5` (s120) | add spec; source from HF |
| 26b-a4b hard | ❌ **not in queue** | HF `…-hard-seed42` (s160) | add spec; source from HF |

**TODO for traces:** (a) ~~fix e4b-easy→s100, e4b-medium→s90~~ ✅ done; (b) add the 5
HF-sourced teacher specs (e2b×3, 26b-medium, 26b-hard) — extend the collection script to
download the teacher from an HF repo `step_NNNNNN` subdir instead of the S3 path (those exports
already include `processor_config.json`); (c) regenerate everything into v2 on all 8 GPUs.

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
- **Data:** the teacher's train traces (3000 q × 8 = 24,000 examples) for that band.
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

| | Node 1 | Node 2 |
|---|---|---|
| **Traces** | 26b easy/medium/hard (2 GPUs, **TP2**) · e4b easy/medium/hard (2 GPUs, DP2) | 12b easy/medium/hard (2 GPUs, DP2) · e2b easy/medium/hard (1 GPU) |
| **Distill** | 26b→e4b, e4b→e4b (2 GPUs each) · 26b→e2b, e4b→e2b (1 GPU each) — 12 runs | 12b→e4b (2 GPUs) · 12b→e2b, e2b→e2b (1 GPU each) — 9 runs |

```bash
# ---- node 1 ----
TRACE_QUEUE_SPECS=26b-easy:2,26b-medium:2,26b-hard:2,e4b-easy:2,e4b-medium:2,e4b-hard:2 \
  VENV=/tmp/.venv-gemma4 GPU_MEMORY_UTILIZATION=0.72 \
  bash rl-distill-scripts/scale_train/run_gemma4_bestckpt_trace_queue.sh
DISTILL_QUEUE_RUNS=26b-easy:e4b:2,26b-medium:e4b:2,26b-hard:e4b:2,e4b-easy:e4b:2,e4b-medium:e4b:2,e4b-hard:e4b:2,26b-easy:e2b:1,26b-medium:e2b:1,26b-hard:e2b:1,e4b-easy:e2b:1,e4b-medium:e2b:1,e4b-hard:e2b:1 \
  bash rl-distill-scripts/scale_train/run_gemma4_distill_queue.sh

# ---- node 2 ----
TRACE_QUEUE_SPECS=12b-easy:2,12b-medium:2,12b-hard:2,e2b-easy:1,e2b-medium:1,e2b-hard:1 \
  VENV=/tmp/.venv-gemma4 GPU_MEMORY_UTILIZATION=0.72 \
  bash rl-distill-scripts/scale_train/run_gemma4_bestckpt_trace_queue.sh
DISTILL_QUEUE_RUNS=12b-easy:e4b:2,12b-medium:e4b:2,12b-hard:e4b:2,12b-easy:e2b:1,12b-medium:e2b:1,12b-hard:e2b:1,e2b-easy:e2b:1,e2b-medium:e2b:1,e2b-hard:e2b:1 \
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
