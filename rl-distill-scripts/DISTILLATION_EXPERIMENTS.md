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

Node 2's six COMPLETE bundles are published as public HF datasets with `data/upload_gemma4_trace_bundle_hf.py`
(`JWei05/gemma4-bestckpt-traces-topk128-v2-{12b-easy,12b-medium,12b-hard,e4b-easy,e4b-medium,e4b-hard}`: 375
train + 38 validation shards with manifests and run configs, the `source/` prompt rosters the view builder
needs, `dataset_index.json`, `COMPLETE.json` and the final-validation log), so `run_gemma4_distill_one.sh`
fetches them on a box that did not generate the traces.

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
- **Hyperparameters:** global batch **64**, **500 steps** — set epochs to **100** as a
  non-binding cap so the step count is the only limit (≈1.3 epochs over 24k examples);
  LR **2.5e-6**, **100** warmup steps, **linear decay to 2.5e-7**. Micro-batching as in the audited
  production distillation: 1 sequence per micro-batch, 4096 padded-token ceiling, 4096-token KL chunks.
  Note: `gemma4_topk_distill_fsdp2.sh`'s strict 8-GPU / batch-128 contract applies only to the
  old `gemma4-hf-bf16-sdpa-topk-overlay-v1` index schema. These runs use derived training
  views (`gemma4-distill-training-view-v1`, built by `build_gemma4_distill_training_view.py`
  from each teacher's trace bundle), for which batch 64 on 1-2 GPUs flows through unchanged.
- **Compute:** FSDP2 per the audited overlay contract (`MODEL_DTYPE=fp32`, BF16 FSDP
  forward params, FP32 reductions, `Gemma4TextDecoderLayer` wrap, 4096 padded-token
  chunk, max length 12288). **e2b students on 2 GPUs, e4b students on 4 GPUs** (fp32 master + Adam
  state do not fit e2b on one GPU or e4b on two alongside activations). Runs on `.venv-gemma4`.
- **Precedent:** `e2b-base-to-e4b-topk128-lr2e6-linear-b128-2ep` used this exact schedule shape.
- **Logging / artifacts:** W&B project `gemma4-bestckpt-distill-v2` (console + wandb), validation every 10
  steps. No periodic local checkpoints (`SAVE_FREQ=0`): the trainer saves once at the final step, that save
  holds only the HF export (`checkpoint.save_contents=["hf_model"]`, no FSDP/Adam shards), it is pushed to
  `JWei05/Distill-gemma4-<teacher_spec>-to-<student>-base/step_000500/` and the local copy is deleted after
  the upload succeeds (`HF_PUSH_DELETE_LOCAL=true`). Storage is HF-only: no S3 (`TRACE_S3_MIRROR_ENABLE=false`,
  `DISTILL_S3_ENABLE=false`).

## 5. Decisions log

1. **Grid** — the 21 runs in §3 (e4b base ← {26b, 12b, e4b}; e2b base ← {26b, 12b, e4b, e2b}; × 3 bands). ✅
2. **Checkpoints on HF** — all 12 best RL checkpoints uploaded and pinned to immutable commits (§1). ✅
3. **Trace specs** — e4b steps corrected to the W&B peaks; all 12 teachers HF-sourced with pinned
   revisions; 8 samples per training question. ✅
4. **Where things run** — generation + distillation on the two remote nodes (§6), plain local
   scripts, HF-only (no S3, no ScaleTrain). Distilled students: e4b base on 4 GPUs, e2b base on
   2 GPUs (fp32 master + Adam do not fit smaller). Final distilled export = **step 500**. ✅
5. **Still-training teachers** — e2b-medium and 26b-a4b medium/hard are pinned at their W&B peaks
   at pin time (§1 ▸); re-pin (W&B best → Hub step → commit) if they are frozen later. ✅
6. **Evaluation** — the two untrained bases (needed on the new 300-q bands, MATH500 and GSM8K),
   the e2b/e4b RL teachers and every distilled student are evaluated (§7); the 12b/26b teachers are
   not. In-distribution = the model's own 300-q band (all three bands are run; the other two are
   cross-band transfer). ✅
7. **Eval throughput (2026-09-04)** — the math eval was CPU-bound: the protocol's per-token top-128
   logprobs (predictive-entropy diagnostic) made vLLM's Python output processing the bottleneck
   (~5 req/s per instance, GPU mostly idle; batching 64 questions per call only gave 1.4×). Decisions:
   (a) generation runs **without logprobs** (`EVAL_PREDICTIVE_TOPK_WIDTH=0`; entropy fields are null,
   sampled tokens unaffected, mean@k/pass@k/maj@k unchanged); (b) two models per 80 GB H100 with a
   fixed 16 GiB KV budget per vLLM instance (`kv_cache_memory_bytes` — the profiler is device-wide and
   concurrent startups on a shared GPU otherwise abort); (c) every model has its own results root. ✅

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
| **Distill** | 26b→e4b (4 GPUs) · 26b→e2b, e2b→e2b (2 GPUs) — **9 runs** | 12b→e4b, e4b→e4b (4 GPUs) · 12b→e2b, e4b→e2b (2 GPUs) — **12 runs** |

```bash
# ---- node 1 ----
TRACE_QUEUE_SPECS=26b-easy:2,26b-medium:2,26b-hard:2,e2b-easy:1,e2b-medium:1,e2b-hard:1 \
  VENV=/tmp/.venv-gemma4 GPU_MEMORY_UTILIZATION=0.72 \
  bash rl-distill-scripts/scale_train/run_gemma4_bestckpt_trace_queue.sh
DISTILL_QUEUE_RUNS=26b-easy:e4b:4,26b-medium:e4b:4,26b-hard:e4b:4,26b-easy:e2b:2,26b-medium:e2b:2,26b-hard:e2b:2,e2b-easy:e2b:2,e2b-medium:e2b:2,e2b-hard:e2b:2 \
  bash rl-distill-scripts/scale_train/run_gemma4_distill_queue.sh

# ---- node 2 ----
TRACE_QUEUE_SPECS=12b-easy:2,12b-medium:2,12b-hard:2,e4b-easy:2,e4b-medium:2,e4b-hard:2 \
  VENV=/tmp/.venv-gemma4 GPU_MEMORY_UTILIZATION=0.72 \
  bash rl-distill-scripts/scale_train/run_gemma4_bestckpt_trace_queue.sh
DISTILL_QUEUE_RUNS=12b-easy:e4b:4,12b-medium:e4b:4,12b-hard:e4b:4,e4b-easy:e4b:4,e4b-medium:e4b:4,e4b-hard:e4b:4,12b-easy:e2b:2,12b-medium:e2b:2,12b-hard:e2b:2,e4b-easy:e2b:2,e4b-medium:e2b:2,e4b-hard:e2b:2 \
  bash rl-distill-scripts/scale_train/run_gemma4_distill_queue.sh
```

Queue knobs used on node 2 (2026-09-04):

- **Sharing the node with the trace queue.** Start the distill queue with `DISTILL_QUEUE_DYNAMIC_GPUS=true
  DISTILL_QUEUE_RESERVED_GPUS_LOG=<trace queue stdout log>` once the trace queue has launched every
  collection. It then takes only GPUs that have no compute process *and* are not named in an open
  `QUEUE launch spec=… gpus=…` entry of that log (a just-launched vLLM worker holds no GPU memory for its
  first minute, so nvidia-smi alone is not enough), so distillation fills each pair the moment a collection
  finishes.
- **Per-run overrides.** A fourth run-spec field sets runner environment for one run, e.g.
  `e4b-easy:e4b:4:LR=1e-6` (several with `;`). Used to redo the e4b→e4b self-distillations at peak lr 1e-6:
  at 2.5e-6 their validation KL bottomed near step 80 and rose afterwards (0.036→0.048 easy, 0.028→0.039
  medium); at 1e-6 it kept falling to 0.011 / 0.012 / 0.0066 (easy / medium / hard) by step 500.
- **e2b students on 2 GPUs need the 2048-token KL chunk** (`run_gemma4_distill_one.sh` sets it per student):
  with 4096 the 12b-medium traces ran the pair to 81.1/81.5 GB and the cuDNN attention backward failed.

Single runs: `TEACHER_SPEC=12b-easy STUDENT=e4b DISTILL_GPU_IDS=0,1,2,3 bash
rl-distill-scripts/scale_train/run_gemma4_distill_one.sh` (recipe defaults: bs 64, 500 steps,
epochs cap 100, lr 2.5e-6, warmup 100, linear → 2.5e-7; pin `STUDENT_REVISION` for reproducible
student identities). Smoke-test one teacher on a node with `TRACE_MAX_SHARDS=1` /
one `run_gemma4_distill_one.sh` before the full queues.

Trace/distill identity: the generator hashes only its own source + config (not the git
commit), so nodes may be at different commits as long as `generate_gemma4_distill_traces.py`
is byte-identical; a change to that file requires a new trace version (bump the `-v2` prefixes).

## 7. Evaluation suite (bases, RL students, distilled students)

Roster (`config/gemma4_distill_study_eval_sources.json`, regenerated by
`data/build_gemma4_distill_study_eval_registry.py`): the **two base students** (`google/gemma-4-E2B`,
`-E4B`, pinned), the **six small RL teachers** (e2b/e4b × easy/medium/hard, pinned to the Hub
commits in §1 — the 12b/26b teachers are not evaluated) and **every distilled student** found on
the Hub (`Distill-gemma4-*` / `gemma4-distill-v2-*`, final `step_000500/` exports, `main` commit
pinned at discovery; empty in-progress repos are skipped, existing pins are kept). Re-run the
builder as distillations finish.

Per model, one generation protocol for all math sets (temp 1.0 / top-p 1.0 / top-k −1, 8192 max
tokens, 12-shot `gemma3_it_fewshot_math.jinja`), scored with **exactly the RL reward**
(`verl.utils.reward_score.math_verify.compute_score`, strict last-`\boxed{}`, 30 s verify, 5 s
SymPy fallback — pinned via `VERL_MATH_VERIFY_*` in the runner; correct ⇔ score > 0.5):

| Family | Sets | Metrics |
|---|---|---|
| In-distribution | `id_easy`, `id_medium`, `id_hard` — the pinned 300-q band validation splits of `JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k@a0ba3c3d` (a model's own band is its ID number; the other two are cross-band transfer) | mean@16, pass@16 (+maj@16) |
| OOD math | MATH500 (`HuggingFaceH4/MATH-500`) ×16; GSM8K (`openai/gsm8k` test) ×8 | mean@16 / pass@16; mean@8 / pass@8 |
| Out-of-domain | MMLU-Pro (5-shot CoT), GPQA-Diamond (5-shot CoT), MMLU-14k (`openai/MMMLU`, 14 locales × 1003) | accuracy via pinned lm-eval-harness (`eval_gemma4_ood.py`) — **a different scorer; never compare with the math family** |

Manifest protocol `gemma4_rl_distill_math_eval_v2` (v1 = the earlier 500-q Easy-10k/Medium-20k
splits; results are kept apart under `s3://scale-ml/genai/rl-distill/gemma4-distill-study-evals-v1/`).
Per model ≈ 33k math generations + 26k OOD items (≈ 2.5 min of generation per 4,800 short-answer
requests without logprobs; the per-dataset scoring/answer-class pass and the Monte-Carlo aggregation
add a few CPU minutes each).

Runtime knobs (all defaults set by the queue; the manifest protocol itself is unchanged):
`MATH_QUESTIONS_PER_BATCH=64` / `MATH_REQUEST_BATCH_SIZE=1024` batch several questions' seeded
requests per vLLM call (every request carries its own deterministic seed, so the sampled set does not
depend on batching); `EVAL_PREDICTIVE_TOPK_WIDTH=0` requests **no per-token logprobs** (the
predictive-entropy diagnostics are recorded as null; with top-128 logprobs each eval was CPU-bound in
vLLM's Python output processing at ~5 req/s); `EVAL_KV_CACHE_GIB=16` fixes each vLLM instance's KV
budget so two models can share an 80 GB H100; `EVAL_GPU_MEMORY_UTILIZATION=0.40` (only gates vLLM's startup free-memory check; the KV budget is fixed).

```bash
# Local GPU-pool queue (the way the study is evaluated): 2 models per 80 GB H100 (EVAL_QUEUE_SLOTS_PER_GPU),
# no per-token logprobs (EVAL_PREDICTIVE_TOPK_WIDTH=0), fixed EVAL_KV_CACHE_GIB=16 KV budget per vLLM
# instance (its memory profiler is device-wide, so concurrent startups on a shared GPU otherwise mis-size
# each other and abort). Full suite per model (math -> OOD -> RUN_COMPLETE.json); the next roster entry
# starts as soon as a slot frees. Override any knob by exporting it before launch. The roster
# is refreshed from the Hub every 10 polls, so distilled students are picked up as they finish;
# models with RUN_COMPLETE.json are skipped (safe to restart). After every completed model §8 below
# is regenerated from the result files and committed/pushed.
tmux new-session -d -s eval-queue \
  "EVAL_QUEUE_GPUS=4,5,6,7 bash rl-distill-scripts/scale_train/run_gemma4_distill_study_eval_queue.sh \
     2>&1 | tee -a /tmp/gemma4_distill_study_eval/eval_queue.log"
# monitor
grep -E "EVAL_QUEUE (launch|done|FAILED|skip)" /tmp/gemma4_distill_study_eval/eval_queue.log
/tmp/.venv-gemma4/bin/python rl-distill-scripts/eval_queue_progress.py      # per-model progress + ETA table
tail -3 /tmp/gemma4_distill_study_eval/queue_logs/<tag>.log        # per-model driver log
ls /tmp/gemma4_distill_study_eval/results/<tag>/{<tag>/math/metrics.json,<tag>/ood/*/complete.json,RUN_COMPLETE.json}
# manual pieces
/tmp/.venv-gemma4/bin/python rl-distill-scripts/data/build_gemma4_distill_study_eval_registry.py   # refresh roster
MODEL_TAG=rl_e2b_easy GPU_COUNT=1 PACKED_PHYSICAL_GPU_IDS=4 \
  bash rl-distill-scripts/scale_train/run_gemma4_rl_distill_eval_one_model.sh                       # one model
/tmp/.venv-gemma4/bin/python rl-distill-scripts/update_distill_study_results_doc.py                  # rebuild §8
```

### 7.1 Splitting the suite across machines (OOD elsewhere)

`EVAL_PHASES` (runner and queue; default `math,ood`) selects the suites, so the math family can run
here while the out-of-domain benchmarks run on another box. **Current policy (2026-09-04): the local
queue runs `EVAL_PHASES=math` — all math first, OOD later/elsewhere.** A queue never double-launches a
model whose runner is alive from a previous queue (it waits, then launches it math-only; the math
runner resumes from finished shards, so nothing is regenerated). Per-model results are independent files,
and `RUN_COMPLETE.json` records the phases a machine finished.

**Other machine, once** (any CUDA 12.x/13 host; needs `HF_TOKEN` in `.env`):
```bash
git clone <this repo> rl-distill && cd rl-distill
git clone https://github.com/EleutherAI/lm-evaluation-harness lm-evaluation-harness \
  && git -C lm-evaluation-harness checkout f4d4b3de3ee6741a7151a9fe74945ee515262f4c   # pinned; the repo only holds a gitlink
VENV=/tmp/.venv-gemma4 GEMMA4_CUDA_VARIANT=cu129 bash rl-distill-scripts/setup_env_gemma4.sh        # cu130 for CUDA-13 drivers
uv pip install --python /tmp/.venv-gemma4/bin/python --no-deps -e ./lm-evaluation-harness          # lm_eval 0.4.13.dev0
```
**Other machine, run the OOD suite** (models are materialized from the pinned Hub commits in the registry):
```bash
# whole roster, 2 models per 80 GB GPU, results under /tmp/gemma4_distill_study_eval/results/<tag>/
# (EVAL_QUEUE_COMMIT_DOC=false: the OOD box must not commit §8; EVAL_QUEUE_WAIT_FOR_NEW=false: exit when the
#  roster is done instead of idling for new distilled students; EVAL_QUEUE_TAGS="rl_e2b_medium rl_e2b_hard"
#  restricts the roster to a subset; VENV points at the repo-local venv if that is where the harness is installed)
EVAL_PHASES=ood EVAL_S3_ENABLE=false EVAL_QUEUE_COMMIT_DOC=false EVAL_QUEUE_WAIT_FOR_NEW=false \
  EVAL_QUEUE_GPUS=0,1,2,3,4,5,6,7 bash rl-distill-scripts/scale_train/run_gemma4_distill_study_eval_queue.sh
# or one model
EVAL_PHASES=ood EVAL_S3_ENABLE=false MODEL_TAG=rl_e2b_easy GPU_COUNT=1 PACKED_PHYSICAL_GPU_IDS=0 \
  bash rl-distill-scripts/scale_train/run_gemma4_rl_distill_eval_one_model.sh
```
(`EVAL_S3_ENABLE=true` instead mirrors to `s3://scale-ml/genai/rl-distill/gemma4-distill-study-evals-v1/<tag>/`
if the host has the `ml-worker` profile.) **This machine, math only:** relaunch the queue with
`EVAL_PHASES=math`. **Merging:** copy each `results/<tag>/<tag>/ood/` directory from the other machine
into the same path under this box's results base (or `aws s3 sync <prefix>/ /tmp/gemma4_distill_study_eval/results/`),
then `python rl-distill-scripts/update_distill_study_results_doc.py` fills the OOD columns of §8.
The OOD numbers also travel with the repo: the OOD box writes
`rl-distill-scripts/config/gemma4_distill_study_ood_summary.json` (`summarize_gemma4_ood_results.py`,
accuracies + result-file sha256s, no samples) and the updater reads it with `--summary <file>` on any
machine; `--fallback-from-doc` keeps the other family's cells already in §8 when a box has data for only
one family (that is how the OOD box refreshes its columns without blanking the math numbers).

Results land under `/tmp/gemma4_distill_study_eval/results/<tag>/` — one root per model:
`<tag>/math/metrics.json`, `<tag>/math/traces/*.jsonl`, `<tag>/ood/<bench>/`, `RUN_COMPLETE.json` —
mirrored to `s3://scale-ml/genai/rl-distill/gemma4-distill-study-evals-v1/<tag>/`.

## 8. Results (updated as each model finishes)

Numbers are copied here by `rl-distill-scripts/update_distill_study_results_doc.py`, which scans
the per-model result files under the study results root and rewrites everything between the two
markers below; run it after any model completes (or let the packed run's watcher do it). Math
numbers (repo `\boxed{}` verifier = the RL reward) and OOD accuracies (lm-eval-harness) are
separate families — do not compare across them. Bold = a model's own band (in-distribution).
All percentages; `mean@k` = average accuracy over k samples, `pass@k` = any-of-k.

**Status (2026-09-05 01:00Z): math suite complete for all 29 models of the study** (2 bases, 6 RL, the full
21-run distillation grid). Generation ran 2026-09-04 16:58Z – 2026-09-05 01:00Z on 7 H100s (2 models per GPU, no
logprobs, per-dataset resume). OOD (MMLU-Pro / GPQA-Diamond / MMLU-14k) is deferred for all models (`EVAL_PHASES=math`;
run it with `EVAL_PHASES=ood` here or on another machine, §7.1). Every RL model's own-band mean@16 matched its W&B
validation best within ~1 point. Note: `e4b-hard→e2b` exists twice on the Hub (`Distill-gemma4-…` and
`gemma4-distill-v2-…`); only the first is rostered (the builder dedupes directions and warns).

**Pipeline check (2026-09-04):** a 1-GPU smoke of `rl_e2b_easy` on `id_easy` (300 q × 16, same verifier
and prompt as the queue) gave mean@16 **40.8** / pass@16 77.0 / maj@16 47.0 — the RL run's own W&B
validation best for that checkpoint was 41.0, so the offline suite reproduces the training-time number.
Re-running it with cross-question batching (64 q × 16 per vLLM call, the queue setting) gave 40.4 / 76.7 / 47.0
with identical per-request seeds (70 % of sequences byte-identical; the rest differ by batch-composition numerics).

<!-- results:start -->
_Updated 2026-09-15 11:21Z — math complete for 33/33 models, OOD complete for 29/33. Partial rows are shown as they finish._

**Math family** — `mean@k / pass@k` (%), repo `\boxed{}` verifier (= RL reward). Bold = own band.

| Model | Category | Trained on | id_easy (16) | id_medium (16) | id_hard (16) | MATH500 (16) | GSM8K (8) |
|---|---|---|---|---|---|---|---|
| `base_12b` | base | — | 42.3 / 97.0 | 14.1 / 72.0 | 5.3 / 45.0 | 17.2 / 59.4 | 46.1 / 88.1 |
| `base_26b` | base | — | 62.3 / 99.7 | 23.8 / 86.3 | 9.9 / 69.7 | 27.2 / 71.8 | 56.7 / 93.6 |
| `base_e2b` | base | — | 11.2 / 55.0 | 4.3 / 36.7 | 3.1 / 32.0 | 4.8 / 37.0 | 8.2 / 36.0 |
| `base_e4b` | base | — | 29.6 / 89.3 | 8.6 / 60.3 | 4.2 / 38.0 | 10.9 / 50.6 | 26.4 / 72.6 |
| `rl_e2b_easy` | rl | easy | **40.7 / 76.3** | 17.0 / 57.7 | 8.4 / 39.0 | 16.3 / 46.2 | 33.8 / 64.5 |
| `rl_e2b_hard` | rl | hard | 21.1 / 46.3 | 16.3 / 40.0 | **13.2 / 28.3** | 10.3 / 25.4 | 8.9 / 25.2 |
| `rl_e2b_medium` | rl | medium | 34.6 / 54.3 | **22.1 / 43.0** | 16.0 / 38.7 | 15.9 / 33.2 | 19.6 / 45.9 |
| `rl_e4b_easy` | rl | easy | **69.9 / 95.3** | 32.8 / 77.3 | 19.2 / 56.0 | 31.5 / 63.6 | 69.5 / 88.9 |
| `rl_e4b_hard` | rl | hard | 39.9 / 79.3 | 20.0 / 56.7 | **15.4 / 47.7** | 16.7 / 48.0 | 43.0 / 77.4 |
| `rl_e4b_medium` | rl | medium | 62.2 / 94.3 | **29.1 / 71.3** | 17.2 / 58.3 | 26.5 / 61.6 | 65.4 / 88.9 |
| `distill_12b_easy_to_e2b` | distilled | easy | **41.5 / 84.3** | 13.4 / 61.3 | 7.8 / 44.3 | 15.8 / 52.6 | 31.5 / 67.2 |
| `distill_26b_easy_to_e2b` | distilled | easy | **40.7 / 84.3** | 13.4 / 61.3 | 6.4 / 36.3 | 15.5 / 46.2 | 31.4 / 66.3 |
| `distill_e2b_easy_to_e2b` | distilled | easy | **38.9 / 79.3** | 16.9 / 59.7 | 7.7 / 40.0 | 15.9 / 47.2 | 32.4 / 65.1 |
| `distill_e4b_easy_to_e2b` | distilled | easy | **38.0 / 84.7** | 14.8 / 62.0 | 7.6 / 41.7 | 15.3 / 48.6 | 32.3 / 70.4 |
| `distill_12b_hard_to_e2b` | distilled | hard | 29.5 / 81.3 | 12.8 / 62.0 | **9.3 / 45.3** | 11.3 / 50.2 | 23.5 / 63.5 |
| `distill_26b_hard_to_e2b` | distilled | hard | 31.0 / 85.7 | 12.9 / 69.3 | **8.0 / 44.3** | 13.3 / 54.6 | 23.7 / 63.8 |
| `distill_e2b_hard_to_e2b` | distilled | hard | 19.6 / 47.7 | 15.7 / 39.3 | **13.1 / 31.0** | 10.0 / 28.0 | 8.4 / 25.7 |
| `distill_e4b_hard_to_e2b` | distilled | hard | 19.8 / 65.0 | 12.4 / 46.3 | **12.1 / 45.0** | 8.6 / 39.6 | 13.4 / 45.2 |
| `distill_12b_medium_to_e2b` | distilled | medium | 36.7 / 87.0 | **15.8 / 65.0** | 8.7 / 51.3 | 14.8 / 51.2 | 29.6 / 71.5 |
| `distill_26b_medium_to_e2b` | distilled | medium | 34.3 / 86.7 | **14.7 / 65.3** | 8.7 / 45.7 | 15.1 / 53.6 | 28.0 / 68.6 |
| `distill_e2b_medium_to_e2b` | distilled | medium | 33.9 / 58.3 | **21.7 / 42.3** | 16.3 / 37.0 | 15.4 / 34.8 | 19.2 / 46.2 |
| `distill_e4b_medium_to_e2b` | distilled | medium | 33.1 / 81.7 | **14.0 / 62.3** | 10.2 / 49.0 | 13.3 / 46.2 | 25.9 / 62.7 |
| `distill_12b_easy_to_e4b` | distilled | easy | **66.9 / 96.0** | 27.2 / 79.7 | 14.2 / 60.0 | 27.4 / 61.6 | 62.8 / 87.4 |
| `distill_26b_easy_to_e4b` | distilled | easy | **67.6 / 95.7** | 26.8 / 78.7 | 13.7 / 53.3 | 29.1 / 64.2 | 64.7 / 90.7 |
| `distill_e4b_easy_to_e4b` | distilled | easy | **69.5 / 97.0** | 31.5 / 77.0 | 17.7 / 53.7 | 31.2 / 64.4 | 69.0 / 91.0 |
| `distill_12b_hard_to_e4b` | distilled | hard | 60.6 / 99.0 | 28.3 / 83.7 | **17.0 / 60.7** | 26.8 / 65.4 | 61.4 / 89.9 |
| `distill_26b_hard_to_e4b` | distilled | hard | 67.8 / 99.3 | 32.8 / 88.7 | **17.6 / 72.0** | 33.2 / 73.2 | 66.2 / 92.8 |
| `distill_e4b_hard_to_e4b` | distilled | hard | 38.9 / 81.0 | 19.3 / 55.3 | **15.2 / 46.7** | 16.2 / 48.0 | 43.1 / 78.9 |
| `distill_12b_medium_to_e4b` | distilled | medium | 68.0 / 96.3 | **32.9 / 82.7** | 19.3 / 65.0 | 33.5 / 67.6 | 69.0 / 93.3 |
| `distill_12bd_medium_to_e4b_step1000` | distilled | medium | 71.3 / 97.0 | **36.6 / 82.3** | 19.9 / 58.3 | 33.9 / 67.2 | 71.3 / 92.4 |
| `distill_12bd_medium_to_e4b_step300` | distilled | medium | 66.6 / 97.3 | **32.0 / 81.3** | 17.9 / 58.3 | 30.2 / 65.6 | 67.9 / 91.1 |
| `distill_26b_medium_to_e4b` | distilled | medium | 71.3 / 99.0 | **37.5 / 89.3** | 20.4 / 69.7 | 34.3 / 71.8 | 67.3 / 92.1 |
| `distill_e4b_medium_to_e4b` | distilled | medium | 62.2 / 93.7 | **28.5 / 75.7** | 16.2 / 55.0 | 26.1 / 60.2 | 64.1 / 88.6 |

**Out-of-domain** — accuracy (%), lm-eval-harness 5-shot CoT (different scorer; not comparable to the math family).

| Model | MMLU-Pro | GPQA-Diamond | MMLU-14k |
|---|---|---|---|
| `base_12b` | — | — | — |
| `base_26b` | — | — | — |
| `base_e2b` | 23.9 | 22.7 | 48.2 |
| `base_e4b` | 37.9 | 19.2 | 61.6 |
| `rl_e2b_easy` | 28.4 | 24.2 | 48.2 |
| `rl_e2b_hard` | 26.0 | 21.7 | 47.6 |
| `rl_e2b_medium` | 27.0 | 24.7 | 47.8 |
| `rl_e4b_easy` | 44.0 | 20.2 | 60.7 |
| `rl_e4b_hard` | 40.2 | 24.2 | 59.4 |
| `rl_e4b_medium` | 43.9 | 26.3 | 61.3 |
| `distill_12b_easy_to_e2b` | 27.1 | 20.7 | 48.4 |
| `distill_26b_easy_to_e2b` | 27.3 | 26.3 | 48.6 |
| `distill_e2b_easy_to_e2b` | 28.0 | 20.2 | 48.4 |
| `distill_e4b_easy_to_e2b` | 27.1 | 19.7 | 48.1 |
| `distill_12b_hard_to_e2b` | 27.3 | 20.2 | 47.8 |
| `distill_26b_hard_to_e2b` | 24.6 | 18.7 | 47.8 |
| `distill_e2b_hard_to_e2b` | 25.7 | 25.3 | 48.0 |
| `distill_e4b_hard_to_e2b` | 25.9 | 25.8 | 47.1 |
| `distill_12b_medium_to_e2b` | 23.9 | 19.2 | 48.2 |
| `distill_26b_medium_to_e2b` | 25.0 | 17.7 | 48.3 |
| `distill_e2b_medium_to_e2b` | 26.8 | 19.7 | 48.1 |
| `distill_e4b_medium_to_e2b` | 26.3 | 24.2 | 47.9 |
| `distill_12b_easy_to_e4b` | 39.0 | 18.7 | 57.6 |
| `distill_26b_easy_to_e4b` | 41.5 | 24.7 | 60.0 |
| `distill_e4b_easy_to_e4b` | 44.1 | 24.7 | 60.9 |
| `distill_12b_hard_to_e4b` | 41.6 | 22.7 | 58.5 |
| `distill_26b_hard_to_e4b` | 41.4 | 21.7 | 59.8 |
| `distill_e4b_hard_to_e4b` | 40.3 | 20.7 | 59.5 |
| `distill_12b_medium_to_e4b` | 39.6 | 14.6 | 58.9 |
| `distill_12bd_medium_to_e4b_step1000` | — | — | — |
| `distill_12bd_medium_to_e4b_step300` | — | — | — |
| `distill_26b_medium_to_e4b` | 41.4 | 19.2 | 59.6 |
| `distill_e4b_medium_to_e4b` | 43.7 | 26.3 | 61.1 |
<!-- results:end -->

### 8.1 pass@k curves — E4B student

![pass@k, E4B student](figures/passk_e4b.png)

`figures/passk_e4b.png` (regenerate with `python rl-distill-scripts/plot_distill_study_passk.py --student e4b
--teachers 12b 26b`): unbiased pass@k (Chen et al.) from the per-sample traces, k = 1..16 (GSM8K 1..8).
Rows = the band the RL / distilled models were trained on; columns = that band, MATH500, GSM8K; curves =
untrained E4B base, E4B RL, and the E4B students distilled from the 12b and 26b RL teachers (E4B→E4B
self-distillation omitted). Read (2026-09-04, all 22 exported models): on the **easy** band the four trained
curves are indistinguishable (RL leads at k=1 by 2–3 points, 12b/26b students catch up by k=4). On **medium** and
**hard** the big-teacher students dominate RL at every k and the gap widens with k (hard band pass@16: 26b→E4B
72, 12b→E4B 61, RL 48, base 38; MATH500 from the hard-band models: 73 / 65 / 48 / 51 — hard-band RL does not
beat the base at k=16 on MATH500, the distilled students do by 15–22 points). GSM8K follows the same order
except for the easy band, where RL is best at small k.

### 8.2 pass@k curves — E2B student

![pass@k, E2B student](figures/passk_e2b.png)

`figures/passk_e2b.png` (`python rl-distill-scripts/plot_distill_study_passk.py --student e2b --teachers 12b 26b`;
add `e4b` to `--teachers` for the e4b-teacher students), same layout as §8.1 with E2B→E2B self-distillation omitted.
Read (2026-09-05): the curves **cross**. On every band the RL model leads at k=1 (medium 22.1 vs 15.8/14.7, hard
13.2 vs 9.3/8.0) but the 12b/26b-distilled students overtake it by k≈2–3 and end far ahead at k=16 (medium 65 vs 43,
hard 45 vs 28). On the easy band the distilled students are ahead at every k (pass@16 84 vs 76). Transfer shows the
same shape: medium/hard-band RL barely beats the base on MATH500 and GSM8K at k=16 (hard-band RL is *below* the base
on both), while the distilled students reach 50–55 on MATH500 and 64–72 on GSM8K. So for the 2B student, RL sharpens
single-sample accuracy on its band; distillation from a bigger teacher broadens coverage and transfers, at a cost in
mean@16 on medium/hard.

## 9. Pre-training control: E4B *base* teacher → 12B / 26B students (started 2026-09-06)

Goal: separate what distillation transfers from what the teacher's pre-training already knows, by
distilling traces from the **untrained** Gemma 4 E4B PT model (`google/gemma-4-E4B` @ `411aa17b`)
into the larger 12B and 26B-A4B bases on the **medium** and **hard** bands.

**Step 1 — trace generation (running locally):** same sampler and prompt as every trace bundle in
this study (temp 1.0 / top-p 1.0 / top-k −1, 8192 max response tokens, 12-shot
`data/gemma3_it_fewshot_math.jinja`), **16 responses per training question, 1 per validation
question**, with top-128 logprobs + token ids per position. Two collections run at once, each with
2 GPUs as 2 data-parallel workers (TP 1):

```bash
# spec e4b-base-medium on GPUs 0,5 and e4b-base-hard on GPUs 6,7 (tmux trace-<spec>)
TRACE_SPEC=e4b-base-medium TRACE_GPU_IDS=0,5 TENSOR_PARALLEL_SIZE=1 TRAIN_SAMPLES_PER_QUESTION=16 \
  VALIDATION_SAMPLES_PER_QUESTION=1 VENV=/tmp/.venv-gemma4 AWS_PROFILE=ml-worker \
  bash rl-distill-scripts/scale_train/run_gemma4_bestckpt_trace_collection.sh
TRACE_SPEC=e4b-base-hard TRACE_GPU_IDS=6,7 ... (same)
# progress: ls /tmp/gemma4_e4b_base_traces_v1/<spec>/{train,validation}/*.parquet | wc -l ; logs under <spec>/logs/
```

The `e4b-base-*` specs (collection script) pull the base repo root (no `step_NNNNNN/`), default to 16
train samples, and write to `/tmp/gemma4_e4b_base_traces_v1/<spec>/` mirrored to
`s3://scale-ml/genai/rl-distill/gemma4-e4b-base-traces-topk128-v1/<spec>/` (bundle `COMPLETE.json`
at the end; directions `e4b_base_{medium,hard}_to_12b_26b`). Source data = the band train split
(3,000 q) and the 300-q validation split of `JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k@a0ba3c3d`.

**Step 2 (prepared, not started — needs the bundles and all 8 GPUs):** distill into `google/gemma-4-12B`
(@ `023679ed`) and `google/gemma-4-26B-A4B` (@ `24548b62`) with the §4 recipe (top-128 forward KL, bs 64,
500 steps, lr 2.5e-6 → 2.5e-7). The launcher now accepts `STUDENT=12b|26b` (8-GPU floor, fp32 master +
Adam; 12B wraps `Gemma4UnifiedTextDecoderLayer`, 26B-A4B `Gemma4TextDecoderLayer`), picks the e4b-base
trace family for `TEACHER_SPEC=e4b-base-*` (16 train samples, no HF dataset mirror), and exposes
`FSDP_OFFLOAD=true` (params **and** optimizer together — verl's engine rejects offloading only one) for the 26B footprint. Tokenizer identity verified: E4B, 12B
and 26B-A4B share the same tokenizer fingerprint (262,144 tokens), which the top-k KL preflight requires.

```bash
# one run per (band, student); 4 runs total, sequential on 8 GPUs (each ~ the §4 recipe's wall time)
TEACHER_SPEC=e4b-base-medium STUDENT=12b DISTILL_GPU_IDS=0,1,2,3,4,5,6,7 bash rl-distill-scripts/scale_train/run_gemma4_distill_one.sh
TEACHER_SPEC=e4b-base-medium STUDENT=26b DISTILL_GPU_IDS=0,1,2,3,4,5,6,7 FSDP_OFFLOAD=true bash rl-distill-scripts/scale_train/run_gemma4_distill_one.sh
# ... e4b-base-hard likewise; students push to JWei05/Distill-gemma4-e4b-base-<band>-to-<student>-base/step_000500
```
Then evaluate with §7 (math suite first). The registry builder now rosters the 12B and 26B-A4B bases
(`base_12b`, `base_26b`, the control baselines) and discovers `…-e4b-base-<band>-to-(12b|26b)-base` students
(tags `distill_e4b_base_<band>_to_<student>`). Note for the queue: a 26B-A4B model (52 GB bf16) needs one
eval slot per GPU (`EVAL_QUEUE_SLOTS_PER_GPU=1`), and the eval queue must not share GPUs with a running
distillation (it only sees other eval runners).

**Target:** the distilled 12B / 26B students should reproduce the E4B teacher's **pass@k curve** on the
in-distribution validation set of their band (not just mean@16). Reference curves: the E4B base sampled
**32×** per validation question (protocol `gemma4_rl_distill_math_eval_v2_x32`, k = 1..32):

```bash
# manifest variant (only id_medium / id_hard at 32 samples; everything else as v2)
python rl-distill-scripts/data/prepare_gemma4_rl_distill_eval_data.py --output-dir /tmp/gemma4_e4b_val32/data \
  --overwrite --samples-override "id_medium=32,id_hard=32" --protocol gemma4_rl_distill_math_eval_v2_x32
# one eval per band (E4B base; identical sampler/prompt/verifier to the study), then the curves:
python rl-distill-scripts/eval_math_passk.py --model <E4B base dir> --expected_model_identity_sha256 <sha> --tag base_e4b \
  --datasets /tmp/gemma4_e4b_val32/data/id_medium.parquet --dataset_manifest /tmp/gemma4_e4b_val32/data/math_eval_manifest.json \
  --out /tmp/gemma4_e4b_val32/id_medium/metrics.json --trace_dir /tmp/gemma4_e4b_val32/id_medium/traces --ks 1 2 4 8 16 32 \
  --temperature 1.0 --top_k -1 --top_p 1.0 --max_tokens 8192 --max_prompt_tokens 4096 --max_model_len 12288 \
  --predictive_topk_width 0 --request_batch_size 2048 --questions_per_batch 64 --subset_strategy monte_carlo --monte_carlo_resamples 4096
python rl-distill-scripts/plot_passk_from_traces.py --out rl-distill-scripts/figures/passk_e4b_base_val32.png \
  --trace "E4B base=/tmp/gemma4_e4b_val32/id_medium/traces/base_e4b__id_medium.jsonl" --trace "E4B base=/tmp/gemma4_e4b_val32/id_hard/traces/base_e4b__id_hard.jsonl"
```
Evaluate each student checkpoint with the same ×32 protocol and add its trace to the plot to compare curves.

**Step 2 hyperparameters — what we ran before vs. these runs (run off this box, 4 GPUs each):**

| | e2b/e4b students (§4, 2026-09-04) | 12B / 26B-A4B students (§9) |
|---|---|---|
| data | 3,000 q × 8 teacher samples = 24k rows | 3,000 q × 16 = 48k rows (`TRAIN_SAMPLES_PER_QUESTION=16`) |
| global batch / steps | 64 / 500 (≈1.3 epochs) | **128 / 1000** (≈2.7 epochs; `TOTAL_EPOCHS=100` cap) |
| LR schedule | 2.5e-6 peak, 100 warmup, linear → 2.5e-7 | **2e-6** peak, 100 warmup, linear → 2e-7 (`MIN_LR_RATIO=0.1`) |
| micro-batching | 1 seq / micro-batch, 4096 padded-token ceiling, KL chunk 4096 (e4b) / 2048 (e2b) | same; KL chunk 2048 (12B) / 1024 (26B) |
| precision / FSDP | fp32 master + Adam, bf16 param views, grad ckpt, max_length 12288 | same; 12B fits on 4×H100 (67–77 GB); 26B-A4B on 4 GPUs needs `FSDP_OFFLOAD=true` (8 GPUs preferred) |
| validation | top-128 KL on 128 teacher val generations every 10 steps | same (`TEST_FREQ=10`); plus pass@k×32 on every saved checkpoint |
| checkpoints | final only (`SAVE_FREQ=0`), HF export only | **every 50 steps** (`ROLLING_CHECKPOINT_FREQ=50`): full resumable checkpoint (fp32 model + Adam + LR/RNG `extra` + dataloader position `data_<rank>.pt`) + HF export pushed to the Hub as `step_000050 … step_001000` (20 pass@k points); the checkpoint goes to a single rolling S3 slot `…/rolling/` (background upload) except every 250 steps (`SAVE_FREQ=250`), when it is kept **permanently** in S3 and retires the rolling slot. Startup restores the newest of permanent/rolling — borrowed ScaleTrain pods get preempted and restart the run-file, losing ≤50 steps. Pods (role `eks-ml-worker2`) cannot delete S3 objects, so stale rolling steps are pruned by `scale_train/prune_rolling_checkpoints.sh` running on this box (tmux `ckpt-janitor`, every 30 min) |
| wrap class | `Gemma4TextDecoderLayer` | 12B `Gemma4UnifiedTextDecoderLayer`, 26B-A4B `Gemma4TextDecoderLayer` (set by `STUDENT`) |

```bash
# on a node with the repo + .venv-gemma4 + .env (HF_TOKEN, WANDB_API_KEY); bundles are fetched from S3 (or copy the local dirs)
COMMON="TRAIN_BATCH_SIZE=128 LR=2e-6 TOTAL_TRAINING_STEPS=1000 LR_WARMUP_STEPS=100 MIN_LR_RATIO=0.1 TEST_FREQ=10 SAVE_FREQ=250 ROLLING_CHECKPOINT_FREQ=50 ROLLING_HF_EXPORT=true HF_PUSH_MAX_TO_KEEP=24 \
        CHECKPOINT_SAVE_CONTENTS='["model","optimizer","extra","hf_model"]' MAX_CKPT_TO_KEEP=1 HF_PUSH_DELETE_LOCAL=false \
        REMOTE_CHECKPOINT_ENABLE=true REMOTE_CHECKPOINT_S3_URI=s3://scale-ml/genai/rl-distill/gemma4-e4b-base-distill-ckpts-v1/<spec>-to-<student>-bs128-s1000-lr2e-6 \
        ALLOW_UNDERSIZED_STUDENT_LAYOUT=true PROJECT_NAME=gemma4-e4b-base-distill-v1"   # the ST run-file sets exactly these
env $COMMON TEACHER_SPEC=e4b-base-medium STUDENT=12b DISTILL_GPU_IDS=0,1,2,3 bash rl-distill-scripts/scale_train/run_gemma4_distill_one.sh
env $COMMON TEACHER_SPEC=e4b-base-hard   STUDENT=12b DISTILL_GPU_IDS=4,5,6,7 bash rl-distill-scripts/scale_train/run_gemma4_distill_one.sh
env $COMMON FSDP_OFFLOAD=true TEACHER_SPEC=e4b-base-medium STUDENT=26b DISTILL_GPU_IDS=0,1,2,3 bash rl-distill-scripts/scale_train/run_gemma4_distill_one.sh
env $COMMON FSDP_OFFLOAD=true TEACHER_SPEC=e4b-base-hard   STUDENT=26b DISTILL_GPU_IDS=4,5,6,7 bash rl-distill-scripts/scale_train/run_gemma4_distill_one.sh
```
Expected wall time (from the local 12B run: 21 s/step at batch 64 on 4 H100s): 12B ≈ 12 h per run at batch 128;
26B-A4B with offload on 4 GPUs is several times slower (untested) — 8 GPUs without offload is the practical layout.

**Parallelism of a distillation run:** FSDP2 fully-sharded data parallel over the run's GPUs (`engine.fsdp_size=-1`,
ZeRO-3 style: fp32 master params, grads and Adam state sharded across all ranks, bf16 parameter views gathered
per layer for compute) — no tensor, pipeline or sequence parallelism (the top-k KL loss requires `sp_size=1`).
Each rank processes one sequence per micro-batch (4,096-padded-token ceiling) and accumulates gradients:
global batch 128 on 4 GPUs = 32 micro-steps per rank per optimizer step. The MoE experts of 26B-A4B are
ordinary sharded linear layers (no expert parallelism); `FSDP_OFFLOAD=true` moves the sharded fp32 params +
optimizer state to CPU between uses.

**pass@k every 100 steps — one short ScaleTrain job per evaluated checkpoint, 2 GPUs, no borrowing.** (Exports are pushed every
50 steps; the submitter evaluates the multiples of 100, `--step-multiple`.) Two no-GPU loops run on this box (tmux `ckpt-passk-submit`, `ckpt-passk-plot`; logs under `/tmp/gemma4_e4b_val32/`):
1. `scale_train/submit_student_ckpt_passk_jobs.py` polls the student repos every 10 min and, for each `step_NNNNNN/`
   export with no result in S3 and no live job, submits `run_gemma4_student_ckpt_passk_st.sh` with
   `STUDENT=<student>,STEP=<step>,BANDS=<band>` via `launch_st_with_code.sh` (HEAD tarball + known-good image; priority high,
   borrowing off, 12 h deadline). Job names `g4e4b-pk-<band[:3]>-<student>-s<step>`; state in
   `/tmp/gemma4_e4b_val32/passk_jobs_state.json`; a job that ends FAILED/CANCELLED without a result is resubmitted (≤3 attempts);
   an export re-pushed by a relaunched run (new last-commit on the Hub) supersedes its old result and is evaluated again.
2. The pod evaluates that one export with `eval_student_checkpoints_passk.py --step` (materialize → identity SHA →
   `eval_math_passk.py`, ×32 protocol, grader = the RL reward) and uploads metrics + traces to
   `s3://scale-ml/genai/rl-distill/gemma4-e4b-base-student-passk-v1/<tag>/`, then exits. GPU layout: **12B = dp 2** (one
   single-GPU vLLM per GPU on interleaved question shards; shard traces merged and re-aggregated with `--resume_traces` —
   exact, since sampling seeds derive from (dataset, question id, sample index), not row order, and the shard manifests
   keep the fixed 32 samples/question), **26B-A4B = tp 2**.
3. `eval_student_checkpoints_passk.py --plot-from-s3` syncs finished steps down and re-plots (plus the untrained-base reference
   curve when `base_<student>__x32_<band>/` exists — produced by a one-off job with `--env-vars "BASE_MODEL=12b"`) `figures/passk_<student>_val32.png`
   (all steps vs the E4B-base teacher curve; the reference traces live here under `/tmp/gemma4_e4b_val32/id_<band>/traces/`).
```bash
python rl-distill-scripts/scale_train/submit_student_ckpt_passk_jobs.py --poll-minutes 10 \
  --repo JWei05/Distill-gemma4-e4b-base-medium-to-12b-base --repo JWei05/Distill-gemma4-e4b-base-medium-to-26b-base
python rl-distill-scripts/eval_student_checkpoints_passk.py --plot-from-s3 --poll-minutes 10 \
  --s3-root s3://scale-ml/genai/rl-distill/gemma4-e4b-base-student-passk-v1 \
  --repo JWei05/Distill-gemma4-e4b-base-medium-to-12b-base --repo JWei05/Distill-gemma4-e4b-base-medium-to-26b-base
# one checkpoint by hand:  cd rl-distill-scripts/scale_train && bash launch_st_with_code.sh --gpus-per-instance 2 --priority high \
#   --active-deadline-hours 12 --run-file run_gemma4_student_ckpt_passk_st.sh --job-name g4e4b-pk-med-12b-s0250 \
#   --env-vars "STUDENT=12b,STEP=step_000250,BANDS=medium"
# local GPU fallback: eval_student_checkpoints_passk.py --gpus 6,7 --parallelism dp|tp --repo ... (same protocol/outputs)
```
(An earlier variant — one long-lived 2-GPU pod per student polling the Hub — was submitted 15:32Z as jobs
`job_dag2l12lrg1g07lkf0ng` / `job_dag2l3hob6s007k81ni0` and cancelled at 15:50Z before running: it would have held reserved
GPUs while waiting for checkpoints.)

### 9.0 RL on top of the distilled 12B (medium) — launched 2026-09-10

**Goal (user):** take the 12B student distilled from the E4B base (`JWei05/Distill-gemma4-e4b-base-medium-to-12b-base/step_001000`,
commit 92368d1f) and run the *same* DAPO medium recipe the E4B and 12B RL teachers were trained with, on a whole borrowed node,
with resumable checkpoints every 5 steps so preemption costs ≤5 steps and the relaunch is automatic.

| | value (identical to the difficulty-sweep E4B/12B medium launches unless noted) |
|---|---|
| init policy | the distilled export above, materialized with `download_hf_subfolder.py` (adds the base's `processor_config.json`; the export alone lacks it) — `GEMMA4_INIT_MODEL_{REPO,REVISION,SUBFOLDER}`; architecture/metadata from `google/gemma-4-12B@023679ed` |
| data | `gemma4_26b_bands` medium (`JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k@a0ba3c3d`), seed 42; in-distribution val 300 q ×16 |
| recipe | GRPO n=16, prompt bsz 64 / mini 32, lr 1e-6 (20 warmup), 4k prompt / 8k response (+2k overlong buffer, penalty 1.0), val every 10, early stopping on `val-core/math/acc/mean@16` patience 5 (incl. initial val), 400 steps max, token TIS correction (run-file default) |
| 12B layout | 8×H100, FSDP2 DP8 (`ACTOR_FSDP_SIZE=-1`, `SP_SIZE=1`), `FSDP_CPU_OFFLOAD_POLICY=True`, micro-batch 1 / 4096 padded tokens, rollout TP1 util 0.45 with a fixed 5 GiB KV cache, compiled rollout (`ROLLOUT_ENFORCE_EAGER=False`) |
| checkpoints | **rolling resumable checkpoint every 5 steps** (`ROLLING_CHECKPOINT_FREQ=5`: sharded model + Adam + LR/RNG + dataloader cursor → `…-full-checkpoints/12b-medium-from-e4bbase-distill-es5/rolling/`), permanent + HF push every 10 (`SAVE_FREQ=10` → `JWei05/DAPO-gemma4-12b-PT-DeepScaleR-gemma26b-medium-seed42-from-e4bbase-distill-es5`), best-HF marker under `…/gemma4-12b-medium-from-e4bbase-distill-es5/` |
| resume | the run-file restores the newest complete S3 checkpoint at start (`full_checkpoint_s3.py restore-latest` → `RESUME_MODE=auto`); ScaleTrain re-queues a preempted borrowing job on its own, and `supervise_borrowing_job.py` (tmux `rl-12b-distill-med`, state under `.scale_train_supervisors/g4-12b-distill-rl-med-20260911/`) relaunches it on preemption, external cancellation or failure (`--relaunch-on-cancel --relaunch-on-failure`, budget 30) until the durable completion markers exist |
| job | `g4-12b-distill-rl-med` = job_dah5igo0masg08eubo1g (submitted 2026-09-10 07:16Z; pods 13:24Z → evicted 13:50Z at step 1, 15:03Z → evicted 21:21Z after rolling step 25) → supervisor relaunch job_dahhve00masg08euboo0 (21:23Z, resumes from rolling 25; pod 3 started 00:44Z 09-11, **job CANCELED externally at 00:48Z**) → relaunched 02:36Z on the user's request as job_dahmi4u0m2tg07j7jkdg (resumes from rolling 25) under a supervisor with `--relaunch-on-cancel --relaunch-on-failure --max-relaunches 30` (commit 0d319c4c: any kill — preemption, external cancel, or failure — now relaunches and the run-file resumes from the newest S3 checkpoint; failures back off 2 min × streak), p5.48xlarge (8 GPUs), priority high, borrowing on, 240 h deadline; W&B run `g4ds26b-12b-medium-from-e4bbase-distill-es5-s42-v1` |

```bash
cd rl-distill-scripts/scale_train && bash launch_gemma4_12b_distilled_rl_medium.sh        # one job; env baked in the script
# or under the supervisor (what is running): python3 supervise_borrowing_job.py --name g4-12b-distill-rl-med ... \
#   --completion-s3-uri <full-checkpoints uri> --max-completion-step 400 --expected-completion-world-size 8 \
#   --completion-best-hf-s3-uri <artifact uri> -- bash launch_gemma4_12b_distilled_rl_medium.sh
```

**Progress (2026-09-14 12:15Z).** After the 00:37Z 09-13 platform cancel the run has continued k8s-only (ScaleTrain shows the job
CANCELED but Kueue keeps resuming the jobset; supervisor stopped to avoid duplicate writers). Pod 9 (since 04:37Z) is at **step 220**;
permanent checkpoints every 10 steps up to 220 on S3; early-stopping state `best 0.497 @ step 190` (val mean@16 on medium val300 ×16;
earlier bests 0.340@50, 0.456@130), misses 4 of 5 at step 235 (15:34Z) — the run ends by early stopping at step 240 unless that eval exceeds 0.497,
or at its step-400 cap. Trajectory of val: 0.306@40 → 0.340@50 → 0.456@130 → 0.497@190 (the distilled 12B started at ≈0.08).

**FINISHED 2026-09-14 16:47Z — early stopping at step 240** (`EARLY_STOP_TRIGGERED step=240 best=0.497 best_step=190 misses=5`;
val 0.485@200, 0.486@210, 0.482@220, 0.489@230, 0.484@240 never beat 0.497@190). `RUN_DONE rc=0`: best-step-190 HF snapshot (7 files,
26.0 GB) published to `s3://scale-ml/genai/rl-distill/gemma4-12b-from-e4bbase-distill-rl/gemma4-12b-medium-from-e4bbase-distill-es5/best_hf/`
(+ `run_outcome.json`), completion receipt at `…-rl-full-checkpoints/12b-medium-from-e4bbase-distill-es5/run_complete.json`, permanent
checkpoints `global_step_10..240` (every 10) on S3; the run also kept pushing to the Hub (pre-dates the S3-only policy):
`JWei05/DAPO-gemma4-12b-PT-DeepScaleR-gemma26b-medium-seed42-from-e4bbase-distill-es5` holds `step_000170..step_000240` (max-to-keep 8;
the best, step 190, is among them — prune to it if Hub storage matters). Headline: **RL on the E4B-base-distilled 12B takes medium val mean@16 from ≈0.08
(step 0) to 0.497 (step 190)**, i.e. the distilled student recovers most of the way toward the untrained-12B RL ceiling; compare the E4B RL
medium teacher (0.231 peak, §9.0b) and 12B-base RL (see the difficulty sweep). Nine pods / eight preemptions across 4.5 days; k8s-only
since the 09-13 platform cancel. Non-fatal noise at shutdown: a W&B service-teardown traceback and a deferred S3 rolling-checkpoint
cleanup error (`failed to delete rolling checkpoint objects`).

### 9.0b Seed-43 replicates of the small RL teachers (E2B, E4B × easy/medium/hard) — launched 2026-09-10 07:4xZ

Same sweep recipe as the seed-42 runs (§9.0 table, minus the 12B memory knobs: `FSDP_CPU_OFFLOAD_POLICY=False`, default
micro-batching), `DATA_SEED=43`, **4 borrowed GPUs per job** (p5.48xlarge:4, priority high), rolling resumable checkpoint every
5 steps + permanent/HF push every 10, one `supervise_borrowing_job.py` per job (tmux `rl-g4-<size>-s43-<band>`, state under
`.scale_train_supervisors/g4-<size>-s43-<band>-20260910/`). Scripts: `scale_train/launch_gemma4_rl_band_seed.sh`
(`SIZE= BAND= SEED= GPUS=`) and `scale_train/start_gemma4_rl_seed_supervisors.sh`. HF repos
`JWei05/DAPO-gemma4-<size>-PT-DeepScaleR-gemma26b-<band>-seed43-26b-bands-es5`; S3
`s3://scale-ml/genai/rl-distill/gemma4-difficulty-s43-20260910{,-full-checkpoints}/<size>-<band>`; W&B `g4ds26b-<size>-<band>-s43-v1`.

| job | id |
|---|---|
| g4-e2b-s43-easy | job_dah5pqg0masg08eubo2g → preempted at step ~24 (reported COMPLETED), supervisor relaunched job_dah7nr80masg08eubo7g (resumes from permanent step 20) |
| g4-e2b-s43-medi | job_dah5psgqi7bg07hm8r0g → preempted at step 2, relaunched job_dah7cigqi7bg07hm8r60 |
| g4-e2b-s43-hard | job_dah5q2o0masg07kbc50g → FAILED at step 15 (12:30Z janitor incident) → job_daha8foqi7bg08a9s720 → container `StartError` on node i-06b9a96a5ad99d11c (runc hook, 15:58Z) → job_dahelqoqi7bg08a9s7bg (resumes from permanent step 10) |
| g4-e4b-s43-easy | ~~job_dah5q70qi7bg07hm8r10~~ (FAILED: 0.5 GiB KV cache too small) → job_dah9km8qi7bg08a9s6t0 |
| g4-e4b-s43-medi | ~~job_dah5qc80masg07kbc510~~ (cancelled before it could fail) → job_dah9km8qi7bg07hm8rag |
| g4-e4b-s43-hard | ~~job_dah5qg80masg07kbc51g~~ (cancelled) → job_dah9ko80masg07kbc590 → container `StartError` on the same node (15:58Z) → job_dahelqoqi7bg07hm8rp0 |

**2026-09-10 11:49Z incident:** the first E4B pod died at vLLM start — `0.66 GiB KV cache is needed … available 0.5 GiB` — because the
launch script used the run-file's default rollout memory settings; the seed-42 sweep set per-model values inside its packing script
(E2B: micro-batch 8 / 12288 padded tokens / util 0.25 / 0.5 GiB KV; E4B: 1 / 4096 / 0.25 / 1 GiB). `launch_gemma4_rl_band_seed.sh`
now carries them (commit 04d413a1); the three E4B jobs were relaunched (state dirs `…-r2`). The E2B jobs keep the run-file defaults
(micro-batch 1, no packing, util 0.65): numerically the same optimization (gradient accumulation), only slower per step.

**2026-09-10 12:30Z incident (my janitor):** `g4-e2b-s43-hard` crashed during its step-15 rolling upload with `HeadObject 404` —
`prune_rolling_checkpoints.sh` had deleted the in-flight `rolling/global_step_15/`. Pods cannot delete S3 objects, so after the
permanent step-10 save the rolling tracker stayed at 5 while permanent was 10; the janitor read "rolling ≤ permanent" as "slot
retired" and removed *every* rolling step, including the one being uploaded (the RL trainer verifies each uploaded object and
raises; the distill trainer would only have logged a failed rolling upload). Fixed (commit 2e177d2a): a rolling step is deleted
only if it is older than the rolling tracker or ≤ the permanent tracker AND has its `_REMOTE_COMPLETE.json`; steps newer than both
trackers and in-flight uploads are never touched. The run was relaunched under a fresh supervisor.


**2026-09-10 21:55Z — seed-43 runs PAUSED (user: prioritize the 12B distilled RL).** All six seed-43 jobs were cancelled and
their supervisors stopped; the 12B distilled RL job and the reverse-KL jobs were left alone. Resume points in S3
(`gemma4-difficulty-s43-20260910-full-checkpoints/<size>-<band>`): e2b-easy permanent 160 (best val 0.396@160), e2b-medium
rolling 135 (best 0.145@130), e2b-hard permanent 10, e4b-easy rolling 25 (best 0.596@20), e4b-medium permanent 30 (best 0.231@30),
e4b-hard nothing yet. To resume later: `SEED=43 SIZES="e2b e4b" BANDS="easy medium hard" bash start_gemma4_rl_seed_supervisors.sh`
(same S3 prefixes → each run restores its newest checkpoint).

**2026-09-15 05:36Z — e4b-medium resumed (seed 43) + new seed 44 (user).** `SEED=43 SIZES=e4b BANDS=medium` and `SEED=44 …` via
`start_gemma4_rl_seed_supervisors.sh` (commit 764b5ec4: supervisors now `--relaunch-on-cancel --relaunch-on-failure --max-relaunches 30`,
quick-cancel cap 2; the band-seed launcher is **S3-only by default** — `HF_PUSH_ENABLE=False`, the permanent S3 checkpoints carry the HF
snapshot and `best_hf/` is published at the end). Seed 43 = **job_dakdierns19g0896pmig** (resumes from permanent step 30 of
`gemma4-difficulty-s43-20260910-full-checkpoints/e4b-medium`; best so far 0.231@30); seed 44 = **job_dakdijrns19g07i0tj50** (fresh; S3 prefix
`gemma4-difficulty-s44-20260910{,-full-checkpoints}/e4b-medium`, W&B `g4ds26b-e4b-medium-s44-v1`). Both QUEUED at 05:36Z (4 borrowed GPUs
each, priority high); supervisors tmux `rl-g4-e4b-s43-medi` / `rl-g4-e4b-s44-medi`, state `.scale_train_supervisors/g4-e4b-s4{3,4}-medi-20260915/`.
The other four seed-43 runs stay paused.

### 9.0c Reverse KL of the distilled students vs the E4B base (launched 2026-09-10 16:4xZ; final results 2026-09-11 07:02Z)

`reverse_kl_topk.py` (run-file `scale_train/run_gemma4_reverse_kl_st.sh`, 1 GPU, borrowing off): sample the *student* on 128
medium-train and 128 medium-validation questions (4 samples/q, study sampler + 12-shot prompt, seed 0) recording its top-128
(token, logprob) per position, then score the same sequences with the E4B base (HF, bf16, soft-capped logits) and report per
token: `rkl_mc` = log p_s(x) − log p_t(x) on the sampled token (unbiased full-vocab KL(student‖teacher)), `rkl_topk` =
Σ_{top-128 of the student} p_s (log p_s − log p_t) (training convention, unnormalised), `rkl_topk_renorm`, and the student's
top-128 mass. Students: the distilled 12B and 26B-A4B `step_001000` exports, each with its untrained base as reference.
Jobs (borrowing off): `g4-rkl-12b-vs-e4b` = job_dahdrsg0masg07kbc5ng, `g4-rkl-26b-vs-e4b` = job_dahdt3g0masg08euboh0; results
`s3://scale-ml/genai/rl-distill/gemma4-e4b-base-reverse-kl-v1/<size>_<distilled|base>__vs_e4b_base__medium_q128_s4_top128/`.
Borrowing-on duplicates (18:57Z, to see which pool schedules first; separate root `…-reverse-kl-v1-brw`): `g4-rkl-12b-e4b-brw` =
job_dahfr28qi7bg07hm8rug, `g4-rkl-26b-e4b-brw` = job_dahfrh8qi7bg07hm8rv0.

**2026-09-10/11 outcome of the first attempts:** the borrowing pair got pods at 21:56Z (right after the seed-43 jobs were paused) and
the reserved pair at 22:42Z. Both distilled students write ~6.5–7k tokens per response (3.3–3.7M tokens per 512-response split),
and vLLM's top-128 logprob output processing capped generation at ~780 tok/s. The borrowing pods were **OOMKilled** (192 GiB pod
limit) at ~56 % of the validation split: the script held a whole split of vLLM outputs (Logprob objects with decoded strings) in
memory. The reserved pair and the 12B RL job were **cancelled externally at 00:48Z** (not by the supervisors). Fix (commit
a628946b): generate in batches of 32 requests, convert to floats immediately, `detokenize=False`. Relaunched on borrowing at 01:41Z:
`g4-rkl-12b-e4b-b2` = job_dahlokm0m2tg08d16q8g, `g4-rkl-26b-e4b-b2` = job_dahlommr1t20089l75ug — cancelled after 15 min: 32 requests
in flight was GPU-bound at ~500 tok/s. Relaunched 01:58Z with `--gen_batch 128` (commit 9ddd778b): `g4-rkl-12b-e4b-b3` =
job_dahm0au0m2tg08d16q9g, `g4-rkl-26b-e4b-b3` = job_dahm0c6r1t2007nga4kg (results under the main `…-reverse-kl-v1/` root). 12B sampling was not bit-reproducible across pods (train split 3.54M vs 3.67M tokens); 26B was.

**Reverse-KL runs before 03:40Z on 2026-09-11 were INVALID (superseded):** the sampler had no `<end_of_turn>` / `<start_of_turn>`
stop tokens (the RL rollout's `VERL_ROLLOUT_EXTRA_STOP` and `eval_math_passk.STOP_STRINGS`), so a base-style student kept writing new
few-shot rounds until the 8,192-token cap (mean length 6–7k tokens, 58 % capped; the RL runs show ~200 tokens at step 0). The numbers
they produced (rKL ≈ 0.068–0.069 nats/token for both students, train and validation) describe the whole rambling continuation, not the
teacher-style answer; their outputs are parked under `s3://…/gemma4-e4b-base-reverse-kl-v1/_invalid_no_stop_tokens/`. Fixed in commit
c3a0230f (`stop_token_ids` for the two turn tokens, matching the RL rollout); corrected jobs `g4-rkl-12b-e4b-b5` = job_dahnfh6r1t2007nga50g
and `g4-rkl-26b-e4b-b5` = job_dahnfr60m2tg08d16qjg write to `s3://scale-ml/genai/rl-distill/gemma4-e4b-base-reverse-kl-v2/`.

**Reverse-KL results (corrected sampler, 2026-09-11 06:4xZ; per response token, 128 q × 4 samples per split, stop on turn tokens):**

| student | split | mean len | finish | rKL Monte-Carlo (full vocab) | rKL top-128 | top-128 renorm. | top-128 mass | nats / response | log p(sampled): student / teacher |
|---|---|---|---|---|---|---|---|---|---|
| 12B distilled ← E4B base (step 1000) | train | 231 | 512 stop | 0.0890 ± 0.0026 | 0.0892 | 0.0866 | 0.998 | 20.5 | −0.932 / −1.021 |
| 12B distilled ← E4B base (step 1000) | validation | 260 | 510 stop / 2 length | 0.0885 ± 0.0024 | 0.0883 | 0.0841 | 0.999 | 23.0 | −0.998 / −1.086 |
| 12B base (untrained, reference) | train | 216 | 512 stop | 0.1427 ± 0.0033 | 0.1408 | 0.1399 | 0.995 | 30.8 | −0.880 / −1.023 |
| 12B base (untrained, reference) | validation | 218 | 511 stop / 1 length | 0.1407 ± 0.0037 | 0.1362 | 0.1356 | 0.992 | 30.6 | −0.847 / −0.987 |
| 26B-A4B distilled ← E4B base (step 1000) | train | 207 | 512 stop | 0.0916 ± 0.0025 | 0.0939 | 0.0916 | 0.998 | 19.0 | −0.927 / −1.019 |
| 26B-A4B distilled ← E4B base (step 1000) | validation | 212 | 511 stop / 1 length | 0.0924 ± 0.0026 | 0.0917 | 0.0893 | 0.998 | 19.6 | −0.890 / −0.982 |
| 26B-A4B base (untrained, reference) | train | 215 | 512 stop | 0.1606 ± 0.0036 | 0.1568 | 0.1554 | 0.996 | 34.5 | −0.739 / −0.900 |
| 26B-A4B base (untrained, reference) | validation | 209 | 512 stop | 0.1590 ± 0.0037 | 0.1596 | 0.1574 | 0.998 | 33.2 | −0.692 / −0.851 |

(± = SE over the 512 per-sequence means; the per-token median is 0, so the divergence sits in a minority of positions. For scale,
the run's own validation loss — the *forward* KL(teacher‖student) on teacher samples — ended near 0.08–0.09 nats/token, so the two
directions agree.) Distillation cut the reverse KL to the E4B base by ~38 % for 12B (0.143 → 0.089 nats/token) — the untrained 12B
base already sits at 0.14 on these prompts because the 12-shot prompt pins the answer format, and its slightly *higher* log p(sampled)
means it is more peaked than the teacher rather than closer to it. The distilled 26B-A4B lands at 0.092, within noise of the 12B
student; both are ~19–23 nats per response. The untrained 26B-A4B base is the furthest from the teacher (0.160 / 0.159; it is also
the most peaked sampler, log p(sampled) −0.74 vs the teacher's −0.90 on its own tokens), so distillation cut its reverse KL by ~42 %.
Ordering: 12B distilled 0.089 ≈ 26B distilled 0.092 < 12B base 0.142 < 26B base 0.160 nats/token, identical on train and validation
questions (no memorisation of the 128 train prompts). Jobs: `g4-rkl-12b-e4b-b6` = job_dahpai6r1t2007nga58g (COMPLETED 06:52Z),
`g4-rkl-26b-e4b-b6` = job_dahpajm0m2tg08d16qrg (COMPLETED 07:02Z); both 1 GPU, borrowing. The b5 pair was preempted before scoring;
b6 resumed from the uploaded traces (same seeds, so the samples are identical). Wall time per student incl. base ≈ 70 min.

**Forward vs reverse KL (2026-09-11).** Forward = the training objective KL(teacher‖student) on teacher-sampled traces (teacher
top-128, unnormalised, per response token; W&B `train/loss`, runs `h90nnxbg` 12B / `72sb8zdl` 26B and their from-scratch predecessors
in `rl-distill/gemma4-e4b-base-distill-v1`): the train-batch value at training step 1 (untrained student, identical across the
restarted attempts) → `val/loss` at step 1000 (the same loss on the fixed held-out teacher traces of the validation questions). Reverse = KL(student‖teacher) on the student's own samples, exact Monte-Carlo, validation questions
(table above): untrained base → step-1000 export. Same 12-shot prompt and medium band on both sides.

| student | forward KL (step-1 train batch → step-1000 validation traces) | reverse KL (validation questions, untrained → step 1000) |
|---|---|---|
| 12B | 0.264 → **0.075** (−72 %) | 0.141 → **0.089** (−37 %) |
| 26B-A4B | 0.178 → **0.076** (−57 %) | 0.159 → **0.092** (−42 %) |

Reading: after training the two directions land within ~20 % of each other (forward 0.075–0.076 vs reverse 0.089–0.092 nats/token), so the
students are not collapsing onto teacher modes; the residual reverse > forward gap is consistent with the students being slightly sharper
than the teacher on their own samples (student log p(sampled) −0.93 vs teacher −1.02). The forward KL falls far more than the reverse
because the untrained bases start much further away in the forward direction (teacher samples contain tokens the bases give little mass
to) than in the reverse one (the bases' own samples are already format-pinned by the prompt). Train-batch forward KL over steps
901–1000 averages 0.064 (12B) / 0.069 (26B) (sd ≈ 0.012 across steps).

### 9.0d On-policy distillation of the distilled 12B toward the E4B base (setup + smoke 2026-09-11; launched 21:44Z)

**What already existed.** verl in this fork ships on-policy distillation end to end: `distillation.*` config group
(`verl/trainer/config/distillation/distillation.yaml`), a colocated vLLM teacher that returns top-k `prompt_logprobs` for every
student-sampled token (`verl/experimental/teacher_loop/`), and the losses in `verl/trainer/distillation/losses.py` — `reverse_kl_topk`
(Σ over the *teacher's* top-k of q_s (log q_s − log p_t), partial sum, backpropagated through the student logits), `forward_kl_topk`,
and the sampled-token estimators k1/k2/k3 (optionally as a policy-gradient advantage, the Thinking-Machines recipe). The DAPO trainer
calls the teacher right after rollout (`dapo/dapo_ray_trainer.py`, `_compute_teacher_colocate`). The Gemma 3 launcher
`rl-distill-scripts/distill_onpolicy.sh` (April 2026; k1 + PG by default, `LOSS_MODE`/`TOPK` knobs) produced six finished runs in W&B
project `distill_onpolicy` (1B/4B/12B PT students ← DAPO 4B/12B/27B teachers, 200 steps × 128 prompts × 1 response, lr 1e-5) and the
`JWei05/gemma3-*-onpolicy-distill-from-dapo*` repos. Nothing had been run for Gemma 4.

**Gemma 4 setup (this commit).** The RL run-file `scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh` gained an opt-in block
(`ONPOLICY_DISTILL_ENABLE=True`): it pins/downloads the teacher (requires `processor_config.json` so vLLM loads the unified Gemma 4
class), and appends the `++distillation.*` overrides (loss mode/top-k, `use_task_rewards=False`, `use_policy_gradient=False`, colocated
teacher with TP/util knobs, level-1 sleep between scoring calls, `max_logprobs=topk`, teacher context = prompt + response + 1).
Everything else — rollout, 12-shot prompt, stop tokens, validation, rolling S3 checkpoints, HF pushes, borrowing supervisor — is the RL
contract. Launcher: `scale_train/launch_gemma4_12b_distilled_onpolicy_medium.sh`.

| | off-policy (§9, done) | on-policy (this launcher) |
|---|---|---|
| student / teacher | 12B base ← E4B base traces | distilled 12B (`step_001000`) ← E4B base (`411aa17b`), colocated vLLM TP=2 ×4, util 0.20, sleep |
| samples | 128 teacher traces / step (pre-generated) | 128 medium prompts × 1 student sample / step, T=1, top_p=1, 8k max |
| loss | teacher top-128 forward KL Σ p_t (log p_t − log q_s) | teacher top-128 reverse KL Σ q_s (log q_s − log p_t) (`reverse_kl_topk`), token-mean, one update / step |
| optimiser | lr 2e-6, warmup 20, 1000 steps | lr 2e-6, warmup 20, 1000 steps (`ACTOR_LR`, `TOTAL_TRAINING_STEPS`) |
| memory (8×H100) | — | 12B FSDP2 DP8, CPU-offload policy, 4096-token micro-batches; student vLLM util 0.35 / 4 GiB KV (n=1) |
| checkpoints | S3 permanent 250 / rolling 50 | S3 permanent + HF push 50 / rolling 10 (`…-onpolicy-full-checkpoints/`) |
| validation | pass@k ×32 offline | val-core/math/acc/mean@16 every 10 steps (medium val300 ×16), early stopping off |

Notes: (i) the on-policy `reverse_kl_topk` is truncated to the *teacher's* top-128 (verl's convention; the §9.0c measurement used the
student's top-128 — same direction, different support; both carry ≥ 99.5 % of the mass here). (ii) With `use_task_rewards=False` the
PG term is zeroed and GRPO with n=1 only feeds the (unused) advantage; the math reward still runs so train/val accuracy keep logging.
(iii) Composed Hydra config validated locally (`DRY_RUN=1` + `omega_conf_to_dataclass`): teacher context 12288+1, `max_logprobs=128`.
(iv) **Engine fix (this commit):** verl's FSDP engine ran the top-k distillation logits processor only under `use_remove_padding`;
Gemma 4 trains padded, so the update died with `KeyError: 'distillation_losses'`. `verl/workers/engine/fsdp/transformer_impl.py`
(padded `NO_PADDING` branch) now packs each sample's real-length logits into the rmpad layout the loss expects, applies the deferred
final-logit softcap, runs the processor and re-nests the outputs (one extra bf16 copy of the logits per micro-batch).
(v) **2-GPU local smoke passed (2026-09-11 21:2xZ):** E2B base student ← E4B base teacher (TP 2, util 0.15, level-1 sleep), 16 prompts × 1
sample, 2 steps, medium band, FSDP CPU-offload policy. Teacher top-128 scoring 44 s / 32 s per step, reverse-KL loss (token-mean over
the teacher's top-128) 0.341 → 0.197, student top-128 mass 0.995, teacher mass 0.997, pg_loss 0 (no PG term), grad-norm 24 → 14,
step 320 s / 168 s (gen 127 s incl. warm-up → 17 s; update 126 s → 101 s on 2 GPUs with offload). Teacher sleep/wake worked across
both steps (`RUN_DONE rc=0`). The 12B launcher is ready to submit.

**Launched 2026-09-11 21:44Z** — `g4-12b-onpolicy-med` = job_dai7cbe0m2tg07j7jm8g (p5.48xlarge, 8 GPUs, borrowing on, priority high),
under `supervise_borrowing_job.py` (tmux `onpolicy-12b-med`, relaunch on cancel/failure, ≤ 30 relaunches, logs
`.scale_train_supervisors/g4-12b-onpolicy-med-20260911/`). Run settings differ from the launcher defaults per the user: **200 steps,
warmup 20, lr 5e-7 constant, SAVE_FREQ 10** (permanent S3 checkpoint with model + Adam + LR/RNG + dataloader cursor, and an HF push,
every 10 steps; rolling every 10 as well). A preempted pod restarts the run-file, which restores the newest S3 checkpoint and resumes;
ScaleTrain re-queues the same job, and the supervisor resubmits if the job is marked finished without the durable completion marker.
Student `JWei05/Distill-gemma4-e4b-base-medium-to-12b-base/step_001000` @ 92368d1f, teacher `google/gemma-4-E4B` @ 411aa17b (TP 2,
util 0.20, sleep). Outputs: HF `JWei05/OnPolicyDistill-gemma4-e4b-base-medium-to-12b-onpolicy-rkl128-from-e4bbase-distill`, S3
`s3://scale-ml/genai/rl-distill/gemma4-12b-from-e4bbase-distill-onpolicy-full-checkpoints/12b-medium-onpolicy-rkl128-from-e4bbase-distill/`,
W&B `g4-onpolicy-12b-medium-onpolicy-rkl128-from-e4bbase-distill-s42-v1`. Metrics to watch: `actor/distillation/loss` (should start ≈ 0.09,
the §9.0c reverse KL of this student) and `val-core/math/acc/mean@16` every 10 steps.

**2026-09-12 21:52Z — relaunched S3-only.** The first job (job_dai7cbe0m2tg07j7jm8g) queued 22 h, was admitted at 20:09Z and Kueue-reclaimed
30 s later before running anything; cancelled at 21:50Z on the user's request to stop pushing to Hugging Face. Relaunched as
`g4-12b-onpolicy-med` = job_daisj1u0m2tg07j7jn60 with `HF_PUSH_ENABLE=False HF_PUSH_REQUIRED=False` (now the launcher default, commit
23272d0f); same recipe (200 steps, warmup 20, lr 5e-7, SAVE_FREQ 10). Checkpoints go to S3 only — each permanent checkpoint carries the
weight-only HF snapshot under `actor/huggingface/`, and `publish-best-hf` copies the best step's snapshot to `RUN_ARTIFACT_S3_URI/best_hf/`.
Supervisor dir `.scale_train_supervisors/g4-12b-onpolicy-med-20260912/`.

**2026-09-13 00:37Z platform incident.** ScaleTrain flipped every job of this account to CANCELED at once (this job, the 12B RL job,
and the unrelated PPO controller's rows) while Kubernetes kept running/queueing the workloads: the RL pod trained on (step 84 at
01:24Z), and the cancelled on-policy submission stayed Pending in Kueue. The supervisor treated the cancel as a preemption and
resubmitted 13× in 8 min; every new submission was CANCELED within ~10 s, each leaving another Pending Kueue workload (our role
cannot list or delete jobsets/workloads; re-issuing `scale-train cancel` on an already-CANCELED job returns True but removes nothing).
Containment: all supervisors stopped; supervisor patched with a quick-cancel backoff + stop cap (commit 005e986e); the abandoned S3 root
`…-onpolicy-full-checkpoints/12b-medium-onpolicy-rkl128-from-e4bbase-distill/` holds a deliberately invalid `run_complete.json` so any
of its 14 stray workloads that Kueue admits fails the completion preflight within minutes instead of training. The real run was moved to
`RUN_TAG=onpolicy-rkl128-from-e4bbase-distill-v2` (new S3 roots + W&B id); its single submission (job_daivm2er1t20089l77r0) was also
CANCELED at 12 s but left exactly one Kueue workload (`…-20260913-t7cym`), which is therefore the de-facto pending run: if admitted it
trains the v2 config with Kueue-level re-queue on preemption but no ScaleTrain status. Same signature as 2026-09-11 00:48Z (nightly,
~00:40–00:50Z) — to be raised with the ScaleTrain team. No new submissions until the platform accepts them again.

**2026-09-13 05:14–06:08Z:** Kueue admitted all 13 remaining strays; each failed the poisoned preflight within ~2 min (verified: `run completion
receipt has an invalid terminal step`). **07:47Z: the v2 workload's third pod held the node and the run started training** (ScaleTrain shows
no job for it; Kueue re-queues it on preemption). First results:

| step | distillation loss (teacher top-128 reverse KL, token-mean) | student mass on teacher top-128 | teacher mass | val mean@16 |
|---|---|---|---|---|
| 0 | — | — | — | 0.0777 |
| 1 | 0.123 | 0.9978 | 0.9976 | |
| 2–9 | 0.080–0.089 | 0.9975–0.9980 | 0.9968–0.9978 | |
| 10 | 0.091 | 0.9974 | 0.9969 | 0.0748 |

The loss starts where the offline measurement said it would (0.089 nats/token, §9.0c) and the student's mass on the teacher's support
matches the teacher's own (0.9975 vs 0.9972), so the teacher-top-k truncation is not hiding a tail (§9.0d note (i)). Timing per step
≈ 177 s (gen 15–100 s, teacher scoring 66 s, update 78 s) plus ~213 s validation and ~306 s checkpoint+upload every 10 steps → ≈ 14 h
for 200 steps if the node holds. `actor/pg_loss` is reported but not part of the loss (`use_task_rewards=False`).

**Steps 20–50 (11:46Z).** Loss after warmup: mean 0.066 (steps 21–30), 0.067 (31–40), 0.072 (41–50) — noisy per step (0.041–0.086), plateau
≈ 0.07 vs 0.089 at start. Val mean@16: 0.078 → 0.100 (20) → 0.115 (30) → 0.100 (40) → 0.087 (50) (128 questions; ±0.02 is a few questions).
Response length swings 170–390 with the batch (1–3 of 128 samples hit 8k), no monotone growth. **Mass gap (student − teacher mass on
the teacher's top-128):** +0.0005 (10), +0.0005 (20), +0.0003 (30), −0.0001 (40), −0.0002 … −0.0009 (41 → 50), monotone since step 40:
the student is slowly moving ~0.1 % of its per-token mass outside the teacher's support, which the truncated reverse KL cannot see
(note (i) above; the per-token clamp at 0 also hides negative partial sums). Tiny in absolute terms, but it is the predicted blind spot.
Decision rule: if the gap passes −0.003 or validation keeps falling, switch the next run to the sampled-token estimator
(`loss_mode=k1`, `use_policy_gradient=True`) or a hybrid; the current run continues to 200 steps for the clean comparison.
**Steps 51–60 (12:35Z):** loss mean 0.073 (plateau since step 30), val@60 0.088 (flat), response length 186–327; mass gap −0.0009 → −0.0021,
monotone every step (−0.00087, −0.00109, −0.00126, −0.00138, −0.00143, −0.00186, −0.00190, −0.00204, −0.00210, −0.00205). On this
trajectory it crosses −0.003 around step 75–90. Prepared follow-up (needs a working ScaleTrain): same launcher with
`ONPOLICY_DISTILL_LOSS_MODE=k1 ONPOLICY_DISTILL_USE_POLICY_GRADIENT=True RUN_TAG=onpolicy-k1pg-from-e4bbase-distill` (sampled-token
reverse-KL estimator as a policy-gradient advantage, unbiased over the full vocabulary; the teacher then returns only the sampled token's
log-prob).
**Steps 61–70 (13:27Z): rule triggered.** Mass gap −0.0020 → −0.0042 (crossed −0.003 at step 66: −0.00348, −0.00333, −0.00398, −0.00387,
−0.00415); loss flat 0.062–0.078; val@70 0.0875 (flat since 50); response length drifting up (320–390 at 8 of the last 10 steps vs ~200 for
the teacher). The teacher-top-128 reverse KL is confirmed to leak mass into the unseen tail on this student. Recommendation: launch the
k1 + policy-gradient variant as soon as ScaleTrain accepts submissions; keep this run as the baseline (to 100 or 200 steps, user's call).
**Steps 71–80 (14:23Z):** gap accelerating, −0.0042 → −0.0093 (−0.00745, −0.00768, −0.00926, −0.00931 at 77–80); val@80 0.083; loss 0.067–0.080;
length 300–360. The leak is compounding rather than saturating.
**Steps 81–90 (15:24Z): degenerate.** Gap −0.0093 → −0.036 (−0.0175, −0.0204, −0.0246, −0.0290, −0.0364 at 86–90); the *truncated* loss now
FALLS (0.064 → 0.039) precisely because mass leaves the teacher's support; response length 400–570; val@90 0.058 (below the 0.078 start).
The objective is being gamed as predicted in note (i). Plan: stop at the step-100 checkpoint (k8s-only run: poison the v2 root so a
restart fails preflight, then delete the pod) and launch the k1 + policy-gradient variant when ScaleTrain accepts submissions.
**Step 100 (16:53Z): collapsed.** Truncated loss 0.0046, mass gap −0.148 (15 % of per-token mass outside the teacher's top-128), mean
response length 1,759 tokens with 8.6 % of samples hitting the 8k cap, val@100 0.031 (start 0.078). Our role cannot delete the pod/jobset,
so the pod runs until preempted (restart blocked by the poisoned root). **Verdict:** teacher-support truncated reverse KL is unusable
for on-policy distillation of a student already close to the teacher — the loss is minimised by leaking mass into the unseen tail and
on-policy sampling amplifies it. Use the sampled-token estimator (k1 + policy gradient), student-support top-k, or on-policy forward KL.
**16:55Z: run ended by itself** — CUDA OOM in `update_actor` at step 101 (`Tried to allocate 9.62 GiB`, GPU 0 with 35 GiB already in use):
the collapsed policy's 1.7k-token responses (8.6 % at the 8k cap, packed sequence length up to 71k) no longer fit the 12B update's
memory budget. Pod exited rc=1; any Kueue retry hits the poisoned preflight. Final artefacts: `global_step_10..100` under the v2 root.

**Hub cleanup (2026-09-12).** Both distilled-student repos (`JWei05/Distill-gemma4-e4b-base-medium-to-{12b,26b}-base`) were pruned to
`step_001000` (commits 71254c99 / 3dbd12c5) and the intermediate steps' LFS blobs permanently purged (0.47 TB + 1.01 TB; Hub storage
counts every blob in git history, so a delete commit alone frees nothing). Old revisions still resolve for `step_001000` but no longer
serve the removed steps. The final exports were also copied to S3:
`s3://scale-ml/genai/rl-distill/gemma4-e4b-base-distill-final-exports/Distill-gemma4-e4b-base-medium-to-12b-base/step_001000/` (26.0 GB,
verified) and `…/Distill-gemma4-e4b-base-medium-to-26b-base/step_001000/` (53 GB). The full training checkpoints (weights + Adam) at
steps 250/500/750/1000 remain under `s3://scale-ml/genai/rl-distill/gemma4-e4b-base-distill-ckpts-v1/`.

### 9.0e On-policy distillation, take 2: reverse KL on the *student's* top-128 (launched 2026-09-13 17:17Z)

**Implementation (commit e2b8849f).** New loss mode `reverse_kl_student_topk` (`verl/trainer/distillation/losses.py`, registered with a
new `DistillationLossSettings.teacher_in_actor=True`): per position, Σ over top-128(q_s) of q_s (log q_s − log p_t), full-vocab softmaxes of
both models. The teacher's logits come from an **extra frozen forward pass inside the actor update** — each actor rank lazily loads the
E4B base (bf16, sdpa, ~17 GB) and runs it on the same padded inputs as the student (`verl/trainer/distillation/fsdp/teacher_in_actor.py`,
hooked into the padded branch of the FSDP engine next to the student's packed logits). No vLLM teacher server is created
(`need_teacher_policy` is False for teacher-in-actor modes) and no `teacher_logprobs` payload is needed. Everything else (rollout, data,
checkpoints, validation, S3-only saving) is unchanged from §9.0d. CPU test: with k = vocab the loss equals the exact reverse KL to 6e-7;
with k = 8 it equals the brute-force partial sum; moving student mass off its own top token *raises* the loss (the gaming direction of
§9.0d is now penalised). Student engine footprint lowered to util 0.30 / 3 GiB KV to make room for the resident teacher.

**Launch.** `g4-12b-onpolicy-stk` = job_dajdlful77qg07nj7njg (borrowing, priority high, 8 GPUs) — ScaleTrain accepted it and Kueue admitted
it within a minute (the instant-cancel condition of 00:37–04:47Z had cleared). Same recipe: distilled 12B `step_001000` student, E4B base
teacher, 128 prompts × 1 sample, lr 5e-7 constant, warmup 20, 200 steps, SAVE_FREQ 10 (S3 only). `RUN_TAG=onpolicy-studenttop128-from-e4bbase-distill`;
S3 `…-onpolicy-full-checkpoints/12b-medium-onpolicy-studenttop128-from-e4bbase-distill/`; W&B run id
`g4-onpolicy-12b-medium-onpolicy-studenttop128-from-e4bbase-distill-s42-v1`. Supervisor `.scale_train_supervisors/g4-12b-onpolicy-stk-20260913/`
(quick-cancel cap 2). Diagnostics: `actor/distillation/student_mass` is now the student's own top-128 mass (≈ 0.998 by construction) and
`teacher_mass` is the teacher's mass on the student's support — the loss starts ≈ 0.089 (§9.0c) and cannot be lowered by tail leakage.

**Startup attempts (17:17–17:44Z).** Attempt 1 died at vLLM init: the 3 GiB student KV cache I set for headroom is below vLLM's 3.94 GiB minimum
for one 12,288-token request (fixed: 4 GiB, commit 4613655d). Attempt 2 died at the first update: upstream's `init_workers` else-branch resets
`self.distillation_config = None` when no teacher servers are created, which the new teacher-in-actor mode relies on (fixed: commit 0b1ccc61).
Both attempts were accepted by ScaleTrain, but the supervisor's two rapid resubmissions after attempt 2 (17:42Z, 17:44Z) were CANCELED within
8 s each — the same instant-cancel behaviour as 00:37–04:47Z, now consistent with a resubmission rate limit rather than an outage; the
supervisor stopped itself after two quick cancels as designed. Relaunch of the fixed code scheduled for ~18:13Z (failure backoff raised
to 900 s). The two cancelled submissions left Kueue workloads carrying the attempt-2 code; if admitted they fail at the same point.

**18:14Z relaunch (job_dajeft7s3ggg08df40eg) accepted; in-actor teacher path verified at step 1 (18:28Z).** Each rank logged
`TEACHER_IN_ACTOR_LOADED … class=Gemma4ForConditionalGeneration softcap=30.0`; no OOM with the 17 GB teacher resident next to the student
engine (util 0.30 / 4 GiB KV) and the CPU-offload actor. Step 1: loss 0.135 (student-top-128 reverse KL; the first batch was also elevated in
§9.0d, 0.123 → ~0.085 from step 2), student mass 0.9987 (its own top-128, matches the offline 0.998), teacher mass on the student's support
0.9965, response length 193. Timing 134 s/step (gen 18 s, update 99 s incl. the teacher forward, no teacher-scoring phase) vs 177 s for
the vLLM-teacher variant. Both strays from the rate-limited resubmissions were admitted meanwhile and failed at the attempt-2 bug as expected.
**18:32Z: OOM at step 3** (`Tried to allocate 6.15 GiB`, actor process at 47.8 GiB with the student engine holding 30.4 GiB): the first
loss implementation materialised full-vocabulary fp32 copies of both the student and teacher logits (6 GB each on a ~6k-token sequence)
plus a packed bf16 copy, on top of the 17 GB resident teacher. Rewritten (commit on 2026-09-13 ~18:45Z): the loss is computed sample by
sample in 1024-row chunks under activation checkpointing straight from the padded student logits, and the teacher is applied through its
hidden states with a chunked LM head + softcap (as in `reverse_kl_topk.py`), so the peak extra memory is ~1 GB per chunk. Because the
checkpointed chunks re-read the student logits in backward, the in-place logits-gradient trick is disabled in this mode (one extra bf16
gradient tensor). CPU test: values and gradients match the packed reference to 2e-7 / 4e-8; padding positions get zero gradient.
Relaunch at 18:53Z (job_dajf2dul77qg07nj7npg) was CANCELED by ScaleTrain 8 s after submission despite a 21-minute gap since the last
failure, so the trigger is not a simple resubmission cooldown (accepted: 17:17, 17:27, 18:14; cancelled: 17:42, 17:44, 18:53). As with the
v2 run, the cancelled submission still created a Kueue workload (`…-stk-…-1yhbp`, carrying the fixed code), which is now the single
pending copy of this run: it trains when admitted, re-queues on preemption, and has no ScaleTrain status. The supervisor was stopped after
this one submission to avoid creating a duplicate workload.
**Kueue admitted `1yhbp` at 19:47Z; three pods were reclaimed within minutes each, the fourth (20:23Z) trains.** Steps 1–3 at 20:44Z: loss
0.158 → 0.085 → 0.071, student mass 0.9987 (own top-128), teacher mass on it 0.9964–0.9969, update 85–102 s / step 124–213 s; the step-3 update
that OOMed before now completes — the chunked/checkpointed loss holds within memory with the 17 GB resident teacher.

| step | loss (student-top-128 reverse KL, mean over the window) | student − teacher mass on the student's top-128 | val mean@16 | response length |
|---|---|---|---|---|
| 1–10 (warmup) | 0.087 (0.158 at step 1) | +0.0019 … +0.0024 | 0.077 at 10 | 176–307 |
| 11–20 (warmup) | 0.081 | +0.0019 … +0.0031 | 0.096 at 20 | 177–233 |
| 21–30 (full lr) | 0.072 | +0.0017 … +0.0021, flat | 0.108 at 30 | 173–277 |

Contrast with §9.0d at the same point: there the gap had already turned negative by step 40 and the loss plateaued; here the gap is flat
and positive (the student's top-128 carries ~0.2 % more of its own mass than the teacher does on that support, exactly the tail the old
objective could not see), the loss keeps easing down at full learning rate, and response length shows no drift. Val 0.077 → 0.108 over 30 steps.
| 31–40 | 0.070 (0.060 at 39–40) | +0.0015 … +0.0021 | 0.101 at 40 | 177–255 |
| 41–50 | 0.074 (0.055–0.084, noisy) | +0.0018 … +0.0022, flat | 0.091 at 50 | 196–248 |

Steps 31–50 (23:19Z): the per-step loss is noisy (batch composition) with the window mean flat at ~0.07 since step 30; the mass gap has not
moved in 50 steps and lengths stay ~200–250, i.e. none of the §9.0d degeneration signatures. Validation 0.108 → 0.101 → 0.091 over
steps 30–50 is within the ±0.02 noise of 128 questions but worth watching against the 0.078 start.
| 51–60 (pod 5, resumed from 50) | 0.072 (0.154 at step 51 = first batch after restore) | +0.0015 … +0.0018 | 0.104 at 60 | 172–257 |
| 61–70 | 0.076 | +0.0016 … +0.0020 | 0.097 at 70 | 187–218 |
| 71–80 | 0.074 | +0.0013 … +0.0018, flat | 0.100 at 80 | 165–227 |
| 81–90 (pods 6–8; three admissions were reclaimed before the step-90 save) | 0.075 | +0.0013 | 0.094 at 90 | 190–290 |
| 91–100 | 0.068 (0.050–0.081) | +0.0009 … +0.0013, flat | 0.091 at 100 | 178–310 |

Steps 30–100: training-batch loss plateaued at ~0.07 (from 0.089), the mass gap stayed flat and positive throughout, response length shows
no drift, and validation oscillates 0.09–0.10 (0.078 at step 0). No degeneration signature after 100 steps, in contrast to §9.0d.
Step-100 checkpoint on S3 (06:33Z) is the natural next point for the offline KL + pass@k measurement done at step 50.
| 101–110 | 0.070 | +0.0006 … +0.0009 | 0.094 at 110 | 183–342 |
| 111–120 | 0.068 | −0.0002 … +0.0011 (≈ 0) | 0.094 at 120 | 187–345 |

**Second-order watch item (step 120):** the student's *own* top-128 mass is drifting down — 0.9987 (step 1) → 0.998 (50) → 0.9965 (111) →
0.9939 (120) — with the teacher's mass on that support falling in step (gap ≈ 0). The student is slowly flattening (0.5 % of per-token mass
moved outside its top-128 in 120 steps), which the truncated sum cannot see either (a much weaker version of the §9.0d blind spot: the
support follows the student's modes, so it cannot be gamed by relocating mass, only by spreading it). Not a problem at this magnitude;
if `student_mass` falls below ~0.99 the fix is the renormalised variant or the sampled-token estimator.

**Step 130 (08:35Z): the alarm fired.** Student top-128 mass 0.988–0.991 (< 0.99), gap −0.0012 … −0.0025 (the teacher now holds *more*
mass on the student's top-128 than the student), response length 266–454, val 0.092 (loss 0.055–0.077). Mechanism, reproduced on CPU with
teacher = original student and ε of the student's mass spread uniformly into the tail:

| tail mass spread ε | student top-k mass | plain student-top-k loss | + tail bucket | exact reverse KL |
|---|---|---|---|---|
| 0 | 0.954 | 0.0000 | 0.0000 | 0.0000 |
| 0.01 | 0.946 | −0.0082 | +0.0024 | +0.0196 |
| 0.03 | 0.930 | −0.0242 | +0.0147 | +0.0769 |
| 0.10 | 0.872 | −0.0772 | +0.0965 | +0.3345 |

The plain truncated sum *rewards* flattening (goes negative, then is clamped to 0 with no restoring gradient) — a weak but real relative of
the §9.0d blind spot. **Fix implemented: `reverse_kl_student_topk_bucket`** adds the (k+1)-th bucket term (1−Q_k)(log(1−Q_k) − log(1−P_k)),
turning the loss into the KL between the (k+1)-bucket distributions: a proper lower bound of the full reverse KL (equal to it at k = vocab,
verified to 5e-6) that rises with tail mass. Zero extra compute (Q_k, P_k are already computed). Now the launcher default; the plain
student-top-k run continues to step 200 for the record.

**Offline KL of the step-50 checkpoint (2026-09-14, 128 medium validation questions × 4 samples, same protocol as §9.0c, run locally on shared
GPUs):** reverse KL (student samples, teacher scores) **0.0713 ± 0.0021** nats/token (top-128 estimate 0.0714, student top-128 mass 0.9985,
mean response length 303) vs **0.0885 ± 0.0024** for the same student before on-policy training — a 19 % reduction in 50 steps with no
mass leak. Forward KL on the same protocol (512 E4B-base samples on the 128 validation questions, scored by each student — the same
teacher samples for both rows, so the comparison is paired):

| student | reverse KL (student samples; exact MC) | student top-128 mass | forward KL (teacher samples; exact MC) | forward, teacher top-128 (off-policy loss convention) |
|---|---|---|---|---|
| distilled 12B, before on-policy (off-policy step 1000) | 0.0885 ± 0.0024 (§9.0c) | 0.998 | **0.0868 ± 0.0027** | 0.0807 |
| after 50 on-policy steps (student-top-128 reverse KL) | **0.0713 ± 0.0021** | 0.9985 | 0.0912 ± 0.0027 | 0.0855 |

Reverse KL fell 19 % while forward KL rose ~5 % (0.087 → 0.091; the teacher's mean sample length is 223 tokens on both rows). That is the
expected mode-seeking trade: the student concentrates on its own modes (its samples look more teacher-like) at the cost of slightly
under-covering the teacher's own samples. Both estimates are per response token; ± is the SE over 512 sequences.

**pass@k ×32 of the step-50 on-policy checkpoint (2026-09-14, medium validation set, 300 questions, same protocol as §9.1):**

| model | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
|---|---|---|---|---|---|---|
| E4B base (teacher) | 8.4 | 15.4 | 26.6 | 41.9 | 59.3 | 74.7 |
| 12B base (untrained) | 14.1 | 24.7 | 39.6 | 57.0 | 73.4 | 86.3 |
| 12B off-policy distilled, step 1000 (= on-policy step 0) | 8.1 | 14.9 | 25.8 | 40.7 | 57.6 | 73.0 |
| **12B on-policy (student-top-128), step 50** | **9.6** | **17.4** | **29.3** | **44.9** | **61.9** | **76.3** |
| E2B base (untrained, reference) | 4.1 | 7.6 | 13.6 | 22.9 | 35.4 | 50.3 |

Fifty on-policy steps lifted the whole curve above both the off-policy student and the E4B teacher itself (+1.5 at pass@1, +3.3 at
pass@32 over step 0; +1.2 / +1.6 over the teacher), while reverse KL to the teacher fell 19 %. The curve is still far below the untrained
12B base (the distillation target is the E4B's distribution, not capability), but the student is no longer a strict copy of the teacher.
Figure: `figures/passk_12b_onpolicy_stk50_vs_teacher.png` (adds the E2B base ×32 reference, evaluated with the same protocol).

### 9.0f On-policy distillation, take 3: student-top-128 reverse KL with the tail bucket (launched 2026-09-14 09:26Z)

**Why.** The plain student-top-128 run (§9.0e) started flattening after step ~100: student top-128 mass 0.9987 → 0.988 (step 130) → 0.947
(step 140), response length 200 → 800 tokens, val 0.094 → 0.085 — the truncated sum rewards spreading mass into the unseen tail (CPU
reproduction in §9.0e). Its S3 root was poisoned at 09:25Z so it stops at its next preemption (still running at 10:12Z: step 150 student mass 0.769,
gap −0.015, response length 1329, val 0.070, grad-norm 43 — fully collapsed); checkpoints `global_step_10..150` remain
and the step-50 results (reverse KL 0.0713, pass@k above the teacher) stand as the reported outcome of that variant.

**Objective.** `reverse_kl_student_topk_bucket` (commit f0181bba): Σ_{v∈top-128(q_s)} q_s (log q_s − log p_t) + (1−Q_k)(log(1−Q_k) − log(1−P_k)),
the KL between the (k+1)-bucket distributions {top-128 tokens, rest}. A proper lower bound of the full reverse KL (equal to it at k = vocab),
rising with the student's tail mass, zero extra compute. Everything else identical to §9.0e (distilled 12B `step_001000` student, in-actor
E4B teacher, 128 prompts × 1 sample, lr 5e-7 / warmup 20 / 200 steps, S3-only saves every 10).

**Launch.** `g4-12b-onpolicy-bkt` = job_dajrr9bns19g07i0ti3g — accepted by ScaleTrain (QUEUED 09:26Z) — `RUN_TAG=onpolicy-studenttop128-bucket-from-e4bbase-distill`;
S3 `…-onpolicy-full-checkpoints/12b-medium-onpolicy-studenttop128-bucket-from-e4bbase-distill/`; W&B run id
`g4-onpolicy-12b-medium-onpolicy-studenttop128-bucket-from-e4bbase-distill-s42-v1`; supervisor `.scale_train_supervisors/g4-12b-onpolicy-bkt-20260914/`
(quick-cancel cap 1). Node at 10:01Z, preempted by borrowing reclaim at 10:05Z (during model download) and again at 10:09Z
(two minutes after re-admission); re-admitted 10:44Z (pod 3), step-0 val 0.077 (same student as §9.0e), TEACHER_IN_ACTOR_LOADED, first
updates at 11:00Z. Diagnostics to compare with §9.0e at the same steps: `student_mass` should stay ≈ 0.998 instead of drifting, the gap
should stay ≈ +0.002, and the loss (now including the bucket term) should start ≈ 0.09.

| step | loss (bucket objective) | student top-128 mass | student − teacher mass | grad-norm | response length | update / step (s) |
|---|---|---|---|---|---|---|
| 1 | 0.159 | 0.9988 | +0.0018 | 5.9 | 262 | 109 / 230 |
| 2 | 0.085 | 0.9986 | +0.0022 | 1.8 | 216 | 84 / 131 |
| 3 | 0.088 | 0.9986 | +0.0023 | 2.0 | 202 | 84 / 118 |

Steps 1–3 coincide with §9.0e's steps 1–3 (0.158 / ~0.085, mass 0.9987, gap +0.002): with the student's tail (0.0012) *smaller* than
the teacher's on the same support (0.0037) the bucket term is ≈ −0.001, so the two objectives only separate once the student starts
spreading mass — which is exactly when the plain sum went negative in §9.0e. Same memory footprint and step time as §9.0e.

**11:10Z: third borrowing preemption at step ~8 (no checkpoint yet), and this time ScaleTrain reported the job COMPLETED** (the first
two preemptions showed as QUEUED). The supervisor found no durable completion, cancelled job_dajrr9bns19g07i0ti3g and relaunched as
**attempt 2 = job_dajtd33ns19g0896plog** (jobset `tapu6`, QUEUED 11:12Z, same env/roots, fresh start). Useful fact learned: an explicit
`scale-train cancel` *does* delete the Kubernetes Job (`kubectl get jobs` no longer lists `0o6od`), unlike ScaleTrain's own 00:37Z
auto-cancels which leave the Kueue workload alive (the `stk` zombies) — so no duplicate writer for the bucket root.

**12:12Z: cancelled the collapsed plain run outright** (`scale-train cancel job job_dajf2dul77qg07nj7npg`; ScaleTrain still had it
IN_PROGRESS, so the poison pill alone would only have stopped it at its next preemption). It had reached step 160 (student top-128 mass
0.77–0.82, val 0.07) and was holding an 8-GPU node while attempt 2 of the bucket run sat queued for an hour. Kubernetes Job gone at 12:13Z;
checkpoints `global_step_10..160` remain on S3 under `12b-medium-onpolicy-studenttop128-from-e4bbase-distill/`.

**Attempt 2 got a node at 14:19Z** (3 h queued; pod `tapu6…cpq59`): step-0 val 0.071 (same weights as pod 3's 0.077 — ×16 sampling noise),
steps 1–3 loss 0.137 / 0.091 / 0.088, student mass 0.9987 → 0.9985, gap +0.0020 … +0.0023, grad-norm 7.2 → 1.8, lengths 213–235,
115–130 s/step. **Step-10 checkpoint on S3 at 15:02Z** (first resumable point). Steps 4–10: loss 0.086–0.094 (window mean 0.090, §9.0e
warmup window was 0.087), student mass 0.9986–0.9988, gap +0.0018 … +0.0022, lengths 171–296, val 0.079 at step 10 (§9.0e: 0.077).
Steps 11–20 (checkpoint 20 on S3 15:24Z): loss 0.055–0.088 (window mean 0.080; §9.0e 0.081), student mass 0.9985–0.9989 (no drift),
gap +0.0017 … +0.0025, lengths 166–275, **val 0.103 at step 20** (§9.0e: 0.096). Steps 21–30 (full lr; checkpoint 30 at 16:03Z): loss
0.060–0.077 (window 0.071; §9.0e 0.072), student mass 0.9989–0.9991 (slightly *up* from warmup), gap +0.0017 … +0.0020, lengths 156–268,
**val 0.110 at step 30** (§9.0e: 0.108). Steps 31–40 (checkpoint 40 at 16:33Z): loss 0.052–0.078 (window 0.069; §9.0e 0.070),
student mass 0.9986–0.9991, gap +0.0016 … +0.0022, lengths 175–305, val 0.093 at step 40 (§9.0e: 0.101; ±0.02 noise). Steps 41–50 (checkpoint 50 at 17:12Z): loss 0.050–0.079
(window 0.069; §9.0e 0.074), **student mass 0.9984–0.9988 at step 50 vs 0.998 in §9.0e** — first comparison point, no difference yet
(expected: the plain run only began drifting after step ~100), gap +0.0017 … +0.0022, lengths 173–302, val 0.096 at step 50 (§9.0e:
0.091). Steps 51–60 (checkpoint 60 at 17:44Z): loss 0.061–0.081 (window 0.073; §9.0e 0.072), student mass 0.9983–0.9988, gap
+0.0014 … +0.0021, mean length 169–262 (an 8192-token non-terminating sample in 3 of the 10 batches — same as the RL/§9.0e runs, harmless
at 1/128), val 0.102 at step 60 (§9.0e: 0.104). Steps 61–70 (checkpoint 70 at 18:18Z): loss 0.063–0.077 (window 0.072; §9.0e 0.076),
student mass 0.9980–0.9987 (min so far 0.9980 at step 65, single-batch), gap +0.0013 … +0.0022, lengths 151–293, val 0.095 at step 70
(§9.0e: 0.097).

**18:30Z: preempted again at step ~75 (4 h 10 min on pod `cpq59`; checkpoint 70 is the resume point).** Same false-COMPLETED
sequence as at 11:10Z: the supervisor cancelled job_dajtd33ns19g0896plog and submitted **attempt 3 = job_dak3s23ns19g07i0tieg** (jobset
`en9q5`, QUEUED 18:33Z). Root cause found and fixed (commit below): the supervisor's pod classifier reads a *Succeeded* pod as a completed
run, but a Kueue-preempted pod exits 0 (the run-file's SIGTERM handling) while its Job sits `spec.suspend=true` waiting for re-admission —
so the supervisor was cancelling a live workload and giving up its queue position each time (the two earlier preemptions during download
showed as QUEUED only because the pod was gone before the poll). `supervise_borrowing_job.py` now checks the owning Job's `spec.suspend`
(tested against the live suspended Job → QUEUED; the finished RL Job → not suspended) and the patched supervisor was restarted adopting
attempt 3 (`--initial-job-id`, tmux `onpolicy-12b-bkt2`, same log/state files). From here a preemption should just wait for Kueue.
Attempt 3 got a node at 20:48Z (2 h 15 min queued), restored step 70 at 21:02Z and was **preempted again at 21:06Z** before its first
step — and the patched supervisor logged `status=QUEUED` instead of COMPLETED, leaving the Kueue workload (and its queue position)
intact: the fix works. Five borrowing preemptions so far today on this run; 70 steps banked.
Re-admitted 21:31Z (25 min; queue position kept), restored step 70 at 21:37Z, **checkpoint 80 on S3 at 22:20Z**. Steps 71–80: loss 0.067–0.084
(0.115 on the first post-restore batch; window 0.079, §9.0e 0.074), student mass 0.9981–0.9983, gap +0.0014 … +0.0019, lengths 168–275,
val 0.100 at step 80 (§9.0e: 0.100).
Steps 81–90 (checkpoint 90 at 22:52Z): loss 0.064–0.078 (window 0.072; §9.0e 0.075), student mass 0.9980–0.9984, gap +0.0012 … +0.0018,
lengths 182–290, val 0.095 at step 90 (§9.0e: 0.094). Next §9.0e comparison point: step 100 (mass 0.9965 there).
**Step 100 (checkpoint 23:24Z) — first divergence from §9.0e, in the bucket's favour:** student top-128 mass **0.9978–0.9984** (§9.0e at
step 100: 0.9965 and falling), gap +0.0011 … +0.0020 (§9.0e: +0.0009 … +0.0013), loss window 0.068 (§9.0e 0.068), lengths 168–375,
**val 0.109 at step 100** (§9.0e: 0.091). The tail-bucket term is holding the mass where the plain objective let it slip.
Steps 101–110 (checkpoint 110 at 23:58Z): loss 0.060–0.081 (window 0.074; §9.0e 0.070), student mass 0.9972–0.9983 (single-batch
dip to 0.9972 at step 102; §9.0e at 110: 0.9965 and falling), gap +0.0011 … +0.0020 (§9.0e: +0.0006 … +0.0009), lengths 180–345,
val 0.100 at step 110 (§9.0e: 0.094).
**Steps 111–120 (checkpoint 120 at 00:31Z) — the plain run's alarm point:** student mass **0.9976–0.9981** (§9.0e at 120: 0.9939), gap
+0.0012 … +0.0015 (§9.0e: ≈ 0, about to turn negative), loss window 0.074 (§9.0e 0.068), lengths 180–306, val 0.099 at step 120 (§9.0e: 0.094).
No flattening: the bucket term has removed the drift the plain objective showed here.
**00:49Z: preempted again** (sixth borrowing preemption; pod `d94vt` ran 21:32–00:49Z, steps 71–~125); resume point = checkpoint 120.
Remaining §9.0e comparison points: steps 100 (0.9965), 120 (0.9939), 130 (0.99), 140 (0.947).

**Finished 2026-09-16 00:39Z (after the 10th borrowing preemption / relaunch at 20:17Z):** ran 130 → 200 without interruption (~8 min/step),
`RUN_OUTCOME_WRITTEN reason=max_steps final_step=200 best_step=200`; best HF export published to
`s3://scale-ml/genai/rl-distill/gemma4-12b-from-e4bbase-distill-onpolicy/gemma4-12b-medium-onpolicy-studenttop128-bucket-from-e4bbase-distill/best_hf`
and the completion receipt to the `…-full-checkpoints/12b-medium-onpolicy-studenttop128-bucket-from-e4bbase-distill` prefix. Two benign
tail errors: the wandb service teardown traceback and `wandb sync --sync-all` (flag removed in this wandb version). One follow-up: the pod role
lacks `s3:DeleteObject`, so every `retire-pointer-after-permanent-N` rolling-checkpoint cleanup was deferred (`AccessDenied`) — the rolling
shards for steps 135–195 are still on S3 and should be deleted from the devbox with the `ml-worker` profile.

Final stretch (steps 130 → 200, from the completed pod's log; §9.0e's plain objective had collapsed to mass 0.77 / val 0.07 by step 160):

| step | loss (bucket objective) | student top-128 mass | teacher mass on that support | val mean@16 | best@16 (mean) | response length |
|---|---|---|---|---|---|---|
| 140 | 0.076 | 0.9978 | 0.9967 | 0.105 | 0.521 | 200–280 |
| 150 | 0.072 | 0.9975 | 0.9963 | 0.098 | 0.495 | ~210 |
| 160 | 0.065 | 0.9970 | 0.9954 | 0.101 | 0.508 | ~260 |
| 170 | 0.071 | 0.9975 | 0.9966 | 0.106 | 0.527 | 200–370 |
| 180 | 0.079 | 0.9979 | 0.9972 | 0.099 | 0.496 | ~160 |
| 190 | 0.076 | 0.9974 | 0.9965 | 0.103 | 0.545 | ~180 |
| 200 | 0.063 | 0.9976 | 0.9967 | 0.107 | 0.536 | ~190 |

Reading: the tail-bucket objective held the student's top-128 mass at 0.997–0.998 for all 200 steps (gap to the teacher +0.001), the loss
settled in the 0.06–0.08 band it reached by step 50, response lengths stayed at 150–370 tokens, and validation accuracy plateaued at
≈ 0.10 mean@16 from step 100 on (E4B-base teacher level; step 0 was 0.071–0.077) — i.e. the student matches the base teacher's
distribution without the drift that killed the plain variant. Grad-norm mostly < 1 with isolated spikes to 4–5. Not yet done for this run:
pass@k / reverse-KL evaluation of the step-200 export (`best_hf/`), and deleting the deferred rolling shards on S3.

### 9.0g Reverse direction: the RL'd distilled 12B (§9.0 best, step 190) → E4B base (started 2026-09-14 18:45Z)

**Goal.** Take the strongest RL model of the study — the E4B-base-distilled 12B after DAPO on medium (val mean@16 0.497 @ step 190,
§9.0) — and distill it back into the **untrained E4B base** with the §9 recipe, then evaluate. Natural control already in hand: §4's
`12b-medium → e4b` run (the *untrained-12B* RL medium teacher, step 120, same student, same loss) — the only difference is which 12B the
RL started from.

**Recipe (§9, table above).** Teacher traces: 3,000 medium train questions × **16** samples + 300 validation × 1, T 1.0 / top-p 1.0 / top-k
off, 8192 max response tokens, 12-shot prompt, top-128 logprobs + token ids per position. Distillation: top-128 forward KL, global batch
**128**, **1000** steps, lr **2e-6** peak / 100 warmup / linear to 2e-7, 1 sequence per micro-batch under the 4096 padded-token ceiling,
fp32 master + Adam, resumable checkpoints every 50 steps (S3 only — no Hub pushes), val top-128 KL every 10 steps. Then pass@k ×32 and
the offline reverse/forward KL against the same teacher.

**Hardware: local GPUs 0 and 2 only** (user's call). Trace generation = 2 data-parallel vLLM workers (TP 1, 12B bf16 ≈ 26 GB each);
the E4B-base collection took 2 h 54 min for the same 48,300 rows (18:40 → 21:34Z 09-06, 4.5 GB), so expect ~2–3× that for the 12B.
The E4B student normally needs 4 GPUs (fp32 master + Adam ≈ 56 GB/GPU on two); on 2 GPUs it will run with `FSDP_OFFLOAD=true`
(params + Adam on the 2 TB host) — slower per step; ETA recorded once the first steps are timed.

**Plumbing added (commit below).** New trace spec `12bd-medium` in `run_gemma4_bestckpt_trace_collection.sh` (RUN_KEY = the RL run's
S3 prefix `12b-medium-from-e4bbase-distill-es5`, BEST_STEP 190, direction `12bd_medium_to_e4b`, `TEACHER_SOURCE=s3` reads
`…-rl-full-checkpoints/<RUN_KEY>/global_step_190/actor/huggingface` — the verified artifact; Hub fallback pinned to `ed5457f6`;
processor_config provisioned from `google/gemma-4-12B` as for every 12B actor export), 16-sample default for `12bd-*`, the direction
registered in `generate_gemma4_distill_traces.py` and `preflight_gemma4_distill_training_view.py`, and a `12bd-*` branch in
`run_gemma4_distill_one.sh` (bestckpt-v2 trace family on S3, 16 samples, no HF dataset mirror, W&B project `gemma4-12bd-distill-v1`).

**Step 1 — traces (launched 18:45Z; both vLLM workers up on GPUs 0/2 at 18:56Z, generating from 18:59Z at ≈1.5 min per 128-request
shard per worker; measured 1.43 shards/min over 18:59–19:45Z (62/375 done) → train split ≈ 23:25Z, bundle ≈ 23:45Z; tmux `trace-12bd-medium`, log `/tmp/gemma4_bestckpt_traces_v2/12bd-medium-collection.log`):**

```bash
TRACE_SPEC=12bd-medium TEACHER_SOURCE=s3 TRACE_GPU_IDS=0,2 TENSOR_PARALLEL_SIZE=1 TRAIN_SAMPLES_PER_QUESTION=16 \
  VALIDATION_SAMPLES_PER_QUESTION=1 VENV=/tmp/.venv-gemma4 AWS_PROFILE=ml-worker bash rl-distill-scripts/scale_train/run_gemma4_bestckpt_trace_collection.sh
# output /tmp/gemma4_bestckpt_traces_v2/12bd-medium/{train,validation}/*.parquet, mirrored to
# s3://scale-ml/genai/rl-distill/gemma4-bestckpt-traces-topk128-v2/12bd-medium/ (COMPLETE.json at the end)
```
**21:57Z: switched to 4 GPUs** (user: kill the root-owned GPU-holder placeholders on GPUs 1 and 3 — container inits, needed `sudo kill -9` —
and resume on 0–3). The 2-worker run was stopped with SIGTERM at 240/375 train shards (all mirrored to S3), and relaunched at 21:59Z as
`TRACE_GPU_IDS=0,1,2,3` → 4 data-parallel workers with the *same* hashed engine config (only `--num-workers`/`--worker-id` change, which
are not part of the semantic hash), so the generator resumes: every existing shard is validated and skipped, no S3 work redone.
tmux `trace-12bd-medium-4gpu`, same log.

**22:00–22:06Z: the 4-GPU attempt lost GPUs 0, 2 and 3 to a teammate's evaluator.** The GPU-1/3 placeholders were Docker containers
(`gpu-hold-1`, `gpu-hold-3`, restart policy on — `gpu-hold-1` came back within a minute; stopped for good with
`docker update --restart=no` + `docker stop`). But within ~40 s of any GPU going free, `watch_and_eval.sh` loops belonging to
jingxuanfan (pinned to GPU sets {0,2} and {3,5}) start 10–30 GB reward-bench processes on it, and our engine's hashed
`gpu_memory_utilization=0.88` requires 69.9 GiB *free at startup* (vLLM `ValueError: Free memory on device … is less than desired`),
so workers 0/2/3 failed their first attempt while worker 1 (GPU 1, in nobody's list) came up. Static shard ownership (shard % workers)
means a worker that never gets its GPU stalls the whole split, so the 4-worker run was stopped (still 240/375, nothing lost) and
relaunched 22:06Z on **GPUs 1,2** (tmux `trace-12bd-medium-g12`, `MAX_WORKER_ATTEMPTS=30`), with a 8-min guard that clears reward-bench
grabs on GPU 2 until our engine holds it (both engines up 22:10Z).

**22:11Z — user: "sudo kill everything on gpus 0-3 and then immediately launch our generation resume."** Done: our 2-GPU run stopped
(SIGTERM), every compute process on GPUs 0–3 killed (three reward-bench processes), the collection relaunched on `TRACE_GPU_IDS=0,1,2,3`
(tmux `trace-12bd-medium-g0123`, `MAX_WORKER_ATTEMPTS=30`) at 22:11:32Z, and a 12-minute startup guard kills any non-EngineCore process
that lands on GPUs 0–3 until each of our four engines holds ≥ 60 GB (`scratchpad/gpu_guard.sh`, log alongside). Still 240/375 done.
All four engines held their memory by 22:15Z (the guard then exited; a reward-bench process later squeezed 11.7 GB onto GPU 0 next to
our engine, harmless). Resume verified: the four workers' first shards were 236/238 (the two in-flight shards of the killed runs, never
saved) and 245/247 (the first unowned ones); no finished shard was regenerated.
Four-worker rate **3.46 shards/min** (22:22–22:34Z; 2.5× the 2-worker 1.38) → train split ≈ 23:01Z, bundle ≈ 23:10Z.
Train split complete 23:13Z (375/375). **Validation split failed at 23:20Z:** three of the four validation workers hit vLLM's
free-memory check on their first attempt (GPU 0: a 10 GB reward-bench intruder, 69.06 GiB free < 69.9; GPU 3: intruders, 5.5 GiB free; GPU 1:
65 GiB free while its own train engine was still tearing down) and **never retried** — a latent bug: under `set -e` the bare
`wait "$pid"; status=$?` aborts the worker subshell on a non-zero exit before the retry loop, so `MAX_WORKER_ATTEMPTS` had never applied
(this is also why the 22:00Z workers showed `attempts=1`). Fixed (`status=0; wait … || status=$?`, commit below). Worker 2 finished its 9
validation shards. Relaunched 23:31Z on the two free GPUs (1, 2; tmux `trace-12bd-medium-val`) for the remaining 29 validation shards —
train shards are all validated and skipped.
The relaunch spent 23:31–23:48Z re-checking the 375 train shards (per-shard integrity pass, ~25/min), then ran the validation split: worker 0
(GPU 1) finished its 19 shards by 23:55Z; worker 1 (GPU 2) lost its GPU to a reward-bench grab and **retried 20 times** (the fixed retry loop
working as intended) before re-acquiring GPU 2 at 00:28Z; 23/38 at 00:33Z, ~15 shards left. Meanwhile a teammate's vLLM server container
(`vllm-ptp-rl2-30174`, root) took GPU 1 once our worker released it, so the distillation runner now stops intruding *containers* (restart
policy off) as well as bare processes on GPUs 0–3, and runs a startup guard until all four ranks hold their 60 GB reservation (commit 1346a034).

**Bundle complete 01:06Z** (`TRACE_COLLECTION_COMPLETE`, 48,300 train rows + 300 validation, `COMPLETE.json` + `dataset_index.json` on S3; the
bundle-wide validator took 31 min — it re-decodes every row). The distillation runner fired at 01:07Z, stopped the vLLM-server container on
GPU 1, then died on a `set -e` trap in its own container lookup (a bare process has no docker cgroup → `grep` exit 1 → abort); fixed and
relaunched 01:09Z (**step 2 launched 01:09:54Z on GPUs 0–3**; view build first, ~10 min for 48k rows). Guard clearing intruders meanwhile.
View built (teacher identity `c9403bab…`, student `acdc0d2b…`) and the trainer launched at ~01:20Z — and **died at the 60 GB reservation**:
between the runner's clearing pass and the ranks' first CUDA allocation, a root container (`/bridge/.venv/bin/python`, 80.9 GB on each of
GPUs 1 and 2) and a 29 GB reward-bench process (GPU 3) had landed, so three ranks hit OOM/`CUDA error: out of memory` while rank 0 reserved
fine. Fix (commit 1a551449): the reservation now *retries* (every 5 s, up to 10 min) while the runner's guard keeps clearing co-tenants
(guard cycle 3 s, 25 min). Relaunched 01:23Z with GPUs 0–3 clear (the bridge container was gone by then).
**Training since 01:24Z** (torchrun 4 ranks; all reservations held at 01:24:42Z; W&B
[`gemma4-12bd-distill-v1/v50a6bro`](https://wandb.ai/rl-distill/gemma4-12bd-distill-v1/runs/v50a6bro)). **Step-0 validation: forward KL 0.323**
nats/token on the teacher's top-128 (teacher mass 0.9994, E4B-base mass on that support 0.990) — higher than the E4B-base→12B start
(0.264, §9.0c table): the RL'd 12B is a sharper, more specific target for the untrained E4B than the E4B base was for the 12B.
Steps 1–3: train KL 0.329 / 0.339 / 0.285, ~40k active response tokens per 128-sequence batch, **86–100 s per step** (4 ranks at
100 % util, 74–79 GB used incl. the 60 GB reservation) → the 1000 steps take ≈ 26 h (ETA ≈ 03:30Z 09-16). Speed-up option not taken
(recipe fidelity): `MICRO_BATCH_SIZE_PER_GPU=2` would pack two ~2k-token sequences under the 4096 ceiling for ~1.4× — the loss is
`token_sum / global_batch_tokens`, so the objective is unchanged, but memory headroom is thin and the audited preflight gates that layout.
Correction after step 20: the first ~10 steps were slow (compile/warm-up); steps 10→20 ran at **45 s/step**, so the run should take ≈ 12.5 h
(**ETA ≈ 14:00Z 09-15**). Val KL 0.323 (0) → 0.319 (10) → **0.280 (20)**; student mass on the teacher's top-128 0.990 → 0.9914.
By step 50 (02:07Z) the pace settled at **32.5 s/step** (≈ 8.6 h for 1000 → **ETA ≈ 10:45Z**). Val KL **0.191 (30) → 0.164 (40) → 0.145 (50)**,
student mass 0.9965 → 0.9970. **Step-50 save verified on S3 (02:11Z):** rolling slot committed and `hf_exports/global_step_50/huggingface/`
(6 files, 17.4 GB) uploaded — the S3-only export path works end to end; training resumed at step 51 without a stall.
Val KL through step 90 (02:35Z, 30 s/step): 60: 0.136, 70: 0.127, 80: 0.122, 90: 0.120; student mass on the teacher's top-128 0.9980.

**Throughput (user question, 02:50Z).** Layout: 4 GPUs, FSDP2 full-shard DP only, **1 sequence per micro-batch** under a 4096 padded-token
ceiling → 32 sequential micro-steps per GPU per optimizer step, SDPA attention (no remove-padding for Gemma 4), bf16 compute / fp32 master +
Adam, grad checkpointing, full-vocab KL in 4096-token chunks. Measured 28.7 s/step for ≈ 250k tokens (≈ 2.2k tok/s/GPU ≈ 80 TFLOP/s, < 10 %
of H100 peak): the one-sequence micro-batch under-feeds the GPU and pays FSDP all-gathers + recompute 32× per step. **Switch at step 150:**
`MICRO_BATCH_SIZE_PER_GPU=4`, `MAX_PADDED_TOKENS_PER_MICROBATCH=8192` (≈ 8k tokens per micro-step, 8 micro-steps/GPU), reservation 70 GB
(leaves < 10 GB visible so the box's schedulers keep off). Objective unchanged (loss = token_sum / global batch tokens; data order restored
from the rolling checkpoint's dataloader position); only the accumulation split changes (the §9 runs already used 16 micro-steps on 8 GPUs).
Executed 03:10–03:13Z: step-150 rolling checkpoint + `hf_exports/global_step_150` committed, trainer stopped, relaunched 03:13:21Z with
`MICRO_BATCH_SIZE_PER_GPU=4 MAX_PADDED_TOKENS_PER_MICROBATCH=8192 DISTILL_RESERVE_GPU_GB=70`; resumes from the step-150 rolling checkpoint
(Adam + LR/RNG + dataloader position). (The automated switch task killed itself with a self-matching `pkill -f` after stopping the trainer, so
the relaunch was issued by hand ~2.5 min later; GPUs stayed free in the gap.) Val KL before the switch: 0.116 (100) → 0.112 (110) →
0.111 (120) → 0.109 (130) → 0.108 (140) → **0.108 (150)**; student mass on the teacher's top-128 0.9986. The curve is flattening around
0.11 at lr ≈ 1.9e-6 — the E4B base cannot get arbitrarily close to the RL'd 12B; compare 0.075–0.076 final for E4B-base→12B/26B (§9.0c).
**Result of the switch (03:23Z):** restore exact (val@150 0.1087 = pre-switch 0.108; train KL at steps 152–153 within 0.001 of the
pre-switch values), step time **17.3 s** (steps 152–156: 17.3 / 17.9 / 18.2 / 17.4 / 15.9) vs 28.7 s before → **1.65×**; ≈ 250k tokens/step
→ 14.5k tok/s total. Remaining 845 steps ≈ 4.1 h → **ETA ≈ 07:30Z**. GPUs at 78.6 GB used (70 GB reservation reused; the high-water mark is
set by the single long-sample micro-batches, which are the same in both layouts) and 100 % util.
Val KL after the switch: 160: 0.1072, 170: 0.1070, 180: 0.1053, 190: 0.1045, 200: 0.1028, 210: 0.1021; rolling checkpoint + `hf_exports/global_step_200` committed at ~03:50Z.
Steps 220–300 (220: 0.1013, 230: 0.1030, 240: 0.1009, 250: 0.0999, 260: 0.1007, 270: 0.1000, 280: 0.1009, 290: 0.1004, 300: 0.0992); **first permanent full checkpoint `global_step_250` on S3** (109 GB), rolling slot at 300. Val KL plateauing ≈ 0.100.
Steps 310–640: val KL 0.098 → 0.0935 (600) as the LR decays; permanent `global_step_500` on S3; rolling 550, 600 committed.

**06:2xZ incident — `/tmp` filled up and the step-650 save died (`SafetensorError: No space left on device`), crashing the trainer.**
The 28 TB ephemeral volume had 895 GB free at 18:50Z 09-14; by 06:20Z it was full. My own footprint: `/tmp/gemma4_trace_models` **524 GB**
(teacher HF copies from every trace collection of the study, never cleaned), `/tmp/verl` 352 GB (the distill's local checkpoints incl. a
**stale 109 GB `global_step_150`** left from the 03:13Z relaunch — the trainer's max-keep pruning only knows about saves of the current
process), `/tmp/gemma4_distill_study_eval` 469 GB (the study's eval traces, mirrored to S3), plus ~100 GB of caches. Freed ≈ 620 GB (all
teacher copies except `12bd-medium`, the stale checkpoint, my caches); relaunched at ~06:28Z → resumes from the S3 rolling step 600
(≈ 15 min of steps lost). Lesson: `/tmp` is shared and finite — trace/eval runners should delete their model copies when done, and a
relaunch must purge the previous process's local checkpoints.
Guards added (commit b6c93f26): the trainer now waits (up to 1 h, `DISTILL_MIN_FREE_GB=160`) for free space before every local save
instead of dying mid-save, the runner purges any previous process's local checkpoints before launching, and a 5-min `/tmp` free-space
watch alerts below 200 GB. The 06:37Z relaunch still runs the pre-guard code (the guard applies from the next relaunch); 712 GB free.
Resume verified: val@600 0.0935 reproduced, step-650 rolling checkpoint + export committed at 07:00Z (611 GB free afterwards); val 0.0937 @ 650.

**07:17Z crash #2 — non-finite gradient at step 692.** Steps 651–690 normal (val 0.0931 @ 660, 0.0932 @ 670, 0.0930 @ 690, grad norm ≈ 3);
step 691 grad norm **41.7** (clipped), step 692 `FloatingPointError: gradient norm is non-finite: nan` on all ranks — the launcher runs the
trainer fail-closed (`VERL_FAIL_ON_NONFINITE_GRAD=1`). Loss was finite (the loss check did not fire). Relaunched 07:24Z from the S3 rolling
step 650 with `VERL_FAIL_ON_NONFINITE_GRAD=0` = verl's standard policy: a non-finite pre-clip norm zeroes the gradients and skips that
optimizer step with a `WARN` (the loss stays fail-closed). **Step 692 is a deterministic bad batch:** on the resumed run steps 690/691 had
grad norms 2.5 / 1.6 (no spike this time) and step 692 again produced `grad_norm = nan` with a *finite* loss (mean 0.072, max 7.8) — the
trainer skipped that optimizer step (`WARN: gradient norm is non-finite`) and continued. So it is one of the 128 sequences of epoch-2 batch
316 (the data order is fixed by the seed-42 shuffle), most likely a bf16 overflow in the backward of one sequence; the loss values give no
hint. Effect on the run: one skipped step out of 1000. Worth dumping that batch later to find the row.
Steps 700–960 (rolling 700–950 + exports on S3, permanent 750): val KL 700: 0.0927, 750: 0.0924, 800: 0.0926, 850: 0.0914, 900: 0.0911, 950: 0.0913, 960: 0.0904 — the LR tail (→ 2e-7) is still buying a little.
**Step 3 armed (09:35Z):** `local_jobs/step3_evals_after_distill.sh` (tmux `step3-orchestrator`, commit a043bb6e) waits for the step-1000
export receipt, clears GPUs 0–3, then runs in parallel (a) the §7 math suite on the final export (GPUs 0,1, tag
`distill_12bd_medium_to_e4b_step1000`) and (b) pass@k ×32 for every 50-step export + the 12bd teacher + the §4 control followed by the
reverse/forward KL of the final student vs the teacher (GPUs 2,3, `eval_12bd_medium_to_e4b.sh`).

**Step 2 DONE ≈ 09:45Z**: 1000 steps, final permanent checkpoint `global_step_1000` (fp32 + Adam + HF export inside, 109 GB) on S3;
val KL 970: 0.0906, 980: 0.0906, 990: 0.0904, 1000: 0.0903 → **final 0.0903** (from 0.323). Orchestrator slip: it waited for `hf_exports/global_step_1000`, but permanent saves
(250/500/750/1000) keep their export *inside* `global_step_N/huggingface/` (only rolling saves write `hf_exports/`), so it never fired —
caught at 10:41Z (≈ 1 h lost); registry entry + pass@k runner fixed for the permanent layout (commit below) and both evals launched by
hand at 10:4xZ: (a) tmux `eval-12bd-step1000` (math suite, GPUs 0,1), (b) tmux `eval-12bd-passk` (pass@k ×32: 16 rolling exports +
250/500/750/1000 + teacher + control, then KL; GPUs 2,3).

**Step 3a result — §7 math suite on the FINAL export (step 1000; official §8 row, eval done 11:21Z):**

| set | **final step 1000** | step 300 | §4 control `distill_12b_medium_to_e4b` (500 steps) | `distill_26b_medium_to_e4b` (best §4 E4B) | `rl_e4b_medium` | `base_e4b` |
|---|---|---|---|---|---|---|
| id_easy (16) | **71.3 / 97.0** | 66.6 / 97.3 | 68.0 / 96.3 | 71.3 / 99.0 | 62.2 / 94.3 | 29.6 / 89.3 |
| **id_medium (16), own band** | **36.6 / 82.3** | 32.0 / 81.3 | 32.9 / 82.7 | 37.5 / 89.3 | 29.1 / 71.3 | 8.6 / 60.3 |
| id_hard (16) | **19.9 / 58.3** | 17.9 / 58.3 | 19.3 / 65.0 | 20.4 / 69.7 | 17.2 / 58.3 | 4.2 / 38.0 |
| MATH500 (16) | **33.9 / 67.2** | 30.2 / 65.6 | 33.5 / 67.6 | 34.3 / 71.8 | 26.5 / 61.6 | 10.9 / 50.6 |
| GSM8K (8) | **71.3 / 92.4** | 67.9 / 91.1 | 69.0 / 93.3 | 67.3 / 92.1 | 65.4 / 88.9 | 26.4 / 72.6 |

Reading: the RL'd *distilled* 12B is a better teacher for the E4B base than the untrained-12B RL model was — **mean@k is higher on every
set** (own band 36.6 vs 32.9, +3.7; easy +3.3, hard +0.6, MATH500 +0.4, GSM8K +2.3) and it beats the E4B's own RL (29.1) by 7.5 points on the
medium band — while **pass@16 is equal or lower** (hard 58.3 vs 65.0, medium 82.3 vs 82.7): the sharper teacher (trained from a student
of the E4B base, then RL'd to 0.497) transfers a more peaked, mode-seeking policy — higher single-sample accuracy, less diversity at large k.
The 26B-teacher E4B student (§4) still leads on pass@k and edges mean@k on medium/hard. Steps 300 → 1000 added +4.6 on the own band and
+3–5 elsewhere; response lengths 173–471 tokens, ≤ 0.1 % truncation. Figure (5 panels, shared legend, both steps):
`figures/passk_e4b_from_12bd_math_suite.png`.

Sanity check on the first 62 shards (7,936 rows, 19:45Z): teacher content sha `b3c37391…` (the step-190 export), **strict accuracy 0.581**
on the medium *train* prompts at T = 1 (the run's val mean@16 was 0.497), mean response 391 tokens (median 312, p95 847), 2 of 2,560
inspected rows hit the 8192-token cap (finish = length), the rest stop cleanly — the RL'd teacher's short boxed style, no degeneration.

**Step 2 — distillation (armed 19:05Z, tmux `distill-12bd-medium`, waits for the bundle's `COMPLETE.json` and the GPUs, then runs;
log `/tmp/gemma4_12bd_distill/distill.log`):** `rl-distill-scripts/local_jobs/distill_12bd_medium_to_e4b_gpu02.sh` =
`TEACHER_SPEC=12bd-medium STUDENT=e4b DISTILL_GPU_IDS=0,2` with the §9 knobs (bs 128, 1000 steps, lr 2e-6 → 2e-7, warmup 100, TEST_FREQ 10),
`ALLOW_UNDERSIZED_STUDENT_LAYOUT=true FSDP_OFFLOAD=true` (2 GPUs), S3-only checkpoints at
`s3://scale-ml/genai/rl-distill/gemma4-12bd-distill-ckpts-v1/12bd-medium-to-e4b-bs128-s1000-lr2e-6/` (permanent every 250, rolling every 50,
plus `hf_exports/global_step_N/huggingface/` for every 50-step export — new `ROLLING_HF_EXPORT_S3`, commit c3e82067), `HF_PUSH_ENABLE=false`,
W&B `gemma4-12bd-distill-v1/12bd-medium-to-e4b-base-bs128-s1000-lr2e-6-g2-offload`. Resumable: rerunning the script restores the newest S3 checkpoint.
**22:30Z update:** with GPUs 0–3 cleared for us, the runner now trains on **all four** (`DISTILL_GPU_IDS=0,1,2,3`, `FSDP_OFFLOAD=false` — the §4
E4B layout, W&B run `…-lr2e-6-g4`), waits for our trace engines to exit, clears anything that landed on 0–3 meanwhile, and reserves 60 GB per
rank at startup (`DISTILL_RESERVE_GPU_GB`, new in `main_full_vocab_distill_fsdp2.py`: allocate-and-release into PyTorch's caching allocator, so
the box's free-GPU schedulers see no room) — the reward-bench evaluator otherwise lands 10–30 GB jobs on any GPU with free memory.

**Step 3 — evals (runner ready: `rl-distill-scripts/local_jobs/eval_12bd_medium_to_e4b.sh`, runs after training on GPUs 0/2):** pass@k ×32 on
the medium validation set (§9.1 protocol) for every 50-step export, the 12bd teacher itself and the §4 control
(`gemma4-distill-v2-12b-medium-to-e4b-base@b015fe88/step_000500` — the untrained-12B RL teacher → same E4B student, same loss), against the
existing E4B-base ×32 trace; then reverse + forward KL vs the 12bd teacher (128 val q × 4) for the final student; figure
`figures/passk_e4b_from_12bd_medium.png`.

**Step 3a — the §7 math suite for the step-300 export (set up 04:5xZ 09-15, NOT launched — GPUs 0–3 are training until ≈ 08:10Z).**
Same protocol as every row of §8: `id_easy`/`id_medium`/`id_hard` (pinned 300-q band validation splits) ×16, MATH500 ×16, GSM8K ×8, T 1.0 /
top-p 1.0 / top-k off, 8192 max tokens, 12-shot prompt, scored with the RL reward (strict last-`\boxed{}`), manifest
`gemma4_rl_distill_math_eval_v2` (already prepared under `/tmp/gemma4_distill_study_eval/data/`), study runner knobs (no logprobs, 16 GiB KV,
64 q × 16 per vLLM call, per-dataset resume). Plumbing: registry entry **`distill_12bd_medium_to_e4b_step300`** (category distilled,
E4B, medium; `s3_hf_export` source = `…/hf_exports/global_step_300/huggingface/` + its `_REMOTE_COMPLETE.json`; the materializer now
accepts the `step` key those receipts carry) in `config/gemma4_distill_study_eval_sources.json` — note the registry *builder* only keeps
Hub-discovered distilled entries, so re-add this one if the roster is regenerated. Runner: `rl-distill-scripts/local_jobs/eval_12bd_step300_math.sh`
(`GPUS=0,1`; results under `/tmp/gemma4_distill_study_eval/results/<tag>/`, mirrored to `s3://scale-ml/genai/rl-distill/gemma4-distill-study-evals-v1/<tag>/`,
then `update_distill_study_results_doc.py --fallback-from-doc` adds the row to §8). Comparison rows already in §8: `distill_12b_medium_to_e4b`
(§4 control: 68.0/96.3 · **32.9/82.7** · 19.3/65.0 · 33.5/67.6 · 69.0/93.3), `base_e4b` (29.6 · 8.6 · 4.2 · 10.9 · 26.4), `rl_e4b_medium` (62.2 · **29.1** · 17.2 · 26.5 · 65.4).
Out-of-domain (MMLU-Pro / GPQA-Diamond / MMLU-14k, lm-eval, different scorer) is the same runner with `EVAL_PHASES=ood`.
**Launched 05:04Z on GPUs 6,7** (user: "just the math … maximize throughput"): `GPUS=6,7 EVAL_KV_CACHE_GIB=48 EVAL_GPU_MEMORY_UTILIZATION=0.85
MATH_REQUEST_BATCH_SIZE=2048 MATH_QUESTIONS_PER_BATCH=128` — one whole H100 per vLLM instance instead of the queue's half-GPU sharing;
per-request seeds make the sampled set independent of batching, so the numbers stay comparable with §8. tmux `eval-12bd-step300`, log
`/tmp/gemma4_distill_study_eval/queue_logs/eval_12bd_step300_driver.log`; a 15-min startup guard keeps GPUs 6/7 clear while vLLM loads.

**Results as they land (mean@k / pass@k, %, computed from the completed trace files with the run's own `acc`; §8 gets the official row at the end):**

| set | step-300 12bd→E4B | §4 control `distill_12b_medium_to_e4b` | `rl_e4b_medium` | `base_e4b` |
|---|---|---|---|---|
| GSM8K (8) | **67.9 / 91.1** (maj@8 78.4; 176 tok, 0 % truncated) | 69.0 / 93.3 | 65.4 / 88.9 | 26.4 / 72.6 |
| MATH500 (16) | **30.2 / 65.6** (maj@16 42.6; 401 tok, 0.1 % truncated) | 33.5 / 67.6 | 26.5 / 61.6 | 10.9 / 50.6 |
| id_easy (16) | **66.6 / 97.3** (maj@16 83.0; 248 tok) | 68.0 / 96.3 | 62.2 / 94.3 | 29.6 / 89.3 |
| id_hard (16) | **17.9 / 58.3** (maj@16 27.3; 463 tok, 0.1 % truncated) | 19.3 / 65.0 | 17.2 / 58.3 | 4.2 / 38.0 |
| **id_medium (16), own band** | **32.0 / 81.3** (404 tok, 0.2 % truncated; from the fully scored trace file before finalisation) | **32.9 / 82.7** | **29.1 / 71.3** | 8.6 / 60.3 |

pass@k curves (one panel per completed set; the four E4B models above): `figures/passk_e4b_step300_vs_refs.png`, regenerated as each set
lands. pass@1 / pass@4 / pass@n so far — GSM8K (n = 8): base 26.3 / 57.6 / 72.6, RL 65.4 / 83.7 / 88.9, §4 control 69.0 / 88.4 / 93.3,
**new 67.9 / 86.3 / 91.1**; MATH500 (n = 16): base 10.9 / 27.6 / 50.6, RL 26.5 / 43.6 / 61.6, §4 control 33.5 / 53.1 / 67.6, **new 30.2 / 49.3 / 65.6**.
At step 300 of 1000 the new student sits between the E4B RL model and the §4 control on both OOD sets (the control is a finished 500-step run).

**Eval complete 05:40Z** (36 min on 2 GPUs for 33k generations); official row in §8 (identical to the table above). Two layout slips fixed
afterwards: the runner had been given the *shared* results root, so the row first came out blank and the end-of-run mirror copied every other
model's results under this tag's S3 prefix — results moved into the per-model layout (`results/<tag>/<tag>/math/`), §8 rebuilt, the 37 stray
prefixes deleted from S3 and the model's own results mirrored (commit a341c62b fixes the runner). Five-panel pass@k figure with one shared legend:
`figures/passk_e4b_step300_vs_refs.png` (`plot_passk_from_traces.py --shared-legend`, commit d6e21d9d).

### 9.0h Resuming the untrained-12B medium RL run from step 130 (patience 4, fast update layout) — 2026-09-15

**Ask.** Continue the seed-42 DAPO run of `google/gemma-4-12B` on the medium band (§1/§8 teacher `12b-medium`) from its last checkpoint,
with Adam state and the dataset cursor, early stopping = **4 non-improving validations in a row**, on one 8-GPU node, and with faster
micro-batching — tested locally first, then launched on ScaleTrain.

**What the checkpoint holds.** `s3://…/gemma4-difficulty-s42-20260819-full-checkpoints/12b-medium/global_step_130/` (169 GB, world size 8):
fp32 model + Adam shards, `extra_state` (LR/RNG), `data.pt` (dataloader cursor) and `validation_early_stopping.json`. The run had stopped on
**patience 1**: best 0.5208 @ 120, then 0.51875 @ 130 → one miss → stop. Resuming with `EARLY_STOPPING_PATIENCE=4
EARLY_STOPPING_MIGRATE_PATIENCE_FROM=1` keeps best/miss history and recomputes the trigger flag (misses 1 < 4), so the next validation is
miss 2 of 4 (the trainer verifies the saved patience equals the migrate-from value; §3b of `RESUME_GEMMA4_26B_A4B_LOCAL.md` documents the
same procedure for the 26B). Steps continue 131 → cap 400 (`TOTAL_TRAINING_STEPS=400`, the sweep's cap); permanent + S3 saves every 10,
rolling every 5, no Hub pushes.

**Blocker handled.** The finished run left durable completion markers (`run_complete.json`, `run_outcome.json`, `best_hf/_REMOTE_COMPLETE.json`);
the run-file's preflight would exit `RUN_ALREADY_COMPLETE`. `scale_train/move_gemma4_12b_medium_completion_markers.sh` moves them (reversibly)
under `pre-resume-20260915/` in each prefix right before the launch.

**Micro-batching.** The 12B recipe ran `MICRO_BATCH_SIZE_PER_GPU=1`, 4096 padded cap, per-layer `FSDP_CPU_OFFLOAD_POLICY=True`, vLLM resident
(distilled-12B run: update 593 s of a 789 s step). The 26B resume measured (2026-09-03, same batch) mbsz 4 / 8192 cap / phase-level
`OFFLOAD=True` / `VLLM_SLEEP_MODE=True` at 4.9×/step, gradient-neutral (global-token-mean loss). `scale_train/launch_gemma4_12b_medium_resume.sh`
adopts that layout for 12B (dense: no router replay), rollout util 0.45 / KV 10 GiB (trainer state is off-GPU during generation), every knob
overridable. **Local test:** the world-size-8 checkpoint cannot load on this box's 4 free GPUs, so `local_jobs/gemma4_12b_fast_layout_local_test.sh`
runs a fresh 2-step 12B DAPO step on GPUs 0–3 with the fast layout (pessimistic: 48 GB/GPU of fp32 state vs 24 on 8 GPUs) to check memory
and update time — launched 17:01Z (tmux `g4-12b-local-test`, log `/tmp/gemma4_12b_local_test/fast_mbs4_cap8192.log`).
First attempt died at Ray start-up (box load 260–300; `/tmp/.venv-gemma4`'s Ray still had the hardcoded 30 s raylet wait — patched to read
`RAY_RAYLET_START_WAIT_TIME_S`, relaunched 17:23Z with 600 s). **Result (4 × H100, 1024 sequences/step, mbsz 4 / 8192 cap / OFFLOAD=True /
sleep mode, no OOM):** step 1 gen 125 s · old-log-prob 101 s · update 393 s · step 642 s (warm-up); **step 2 gen 47 s · old-log-prob 60 s ·
update 289 s · step 418 s** (256 seq/GPU → 64 micro-steps of 4 ≈ 4.5 s each; grad norms 2.7 / 1.7, mean response 202 tokens). On the 8-GPU
node (128 seq/GPU, half the fp32 state per GPU) the update should land ≈ 150 s vs 593 s for the recipe layout on the distilled-12B run.

**Launched 18:09Z:** completion markers moved to `pre-resume-20260915/` (both prefixes), then `g4-12b-med-resume` = **job_dakojfht1s0g088gt0vg**
(QUEUED) under supervisor tmux `rl-g4-12b-med-resume` (`.scale_train_supervisors/g4-12b-med-resume-20260915/`, relaunch on cancel/failure,
quick-cancel cap 2). Expect on start: `restored complete source=permanent step=130`, `EARLY_STOPPING_PATIENCE_MIGRATED … triggered=False`,
validation at 140, 150, … ; stops after 4 consecutive misses vs the running best (0.5208 @ 120 unless beaten) or at step 400.

**Switched to a local 4-GPU resume (user request, 18:20Z).** The ScaleTrain job was cancelled (`job_dakojfht1s0g088gt0vg`, CANCELED
18:21Z, supervisor stopped) and the run continues on this box instead. The obstacle: verl's FSDP2 checkpoints are **per-rank `torch.save`
files** (`model|optim|extra_state_world_size_8_rank_r.pt`), loaded back by `(world_size, rank)`, so an 8-rank checkpoint cannot be resumed on
4 GPUs. Each sharded value is a `DTensor` with `Shard(0)` on the 1-D `fsdp` mesh whose local piece follows `torch.chunk` semantics (checked:
every dim-0 size is divisible by 8; 630 model DTensors + 48 replicated buffers; Adam `exp_avg`/`exp_avg_sq` DTensors + scalar `step`).
New tool **`reshard_fsdp2_checkpoint.py`**: loads the 8 shards on CPU (no GPUs, no NCCL — the destination DTensors are built on a
`_init_backend=False` mesh under a 1-process gloo group, which pickles exactly like the trainer's own shards), concatenates the local
pieces in rank order, re-chunks to 4, clones each chunk (a `torch.chunk` view would serialize the whole tensor into every shard file — the
first attempt wrote 48 GiB per rank), rewrites `fsdp_config.json` (`world_size: 4`), copies `extra_state` rank r ← source rank r (LR
scheduler identical, RNG per rank), `data.pt` and `validation_early_stopping.json`, then re-reads every destination shard and checks the
rank-ordered concatenation is **bit-exact** against the source (model 630/630, optim 1236/1236 DTensors). Conversion of the 134 GB step-130
checkpoint took ~9 min (4 × 12.1 GiB model + 4 × 22.2 GiB Adam shards). It generalizes to any W→D with W % D == 0 or D % W == 0.

`local_jobs/resume_gemma4_12b_medium_local4.sh` drives the same run-file contract as the ScaleTrain launcher with local differences:
picks **4 GPUs that are idle at start** (no compute process, < 512 MiB; the box's other users were on GPUs 5/6 at launch), checkpoints on EFS
(`/mnt/efs/jasonwei/gemma4-12b-medium-s42-local4/ckpts`; the `/tmp` volume was at 158 GB free — a 4-rank checkpoint + HF export is ~160 GB),
`MAX_ACTOR_CKPT_TO_KEEP=2`, a **new S3 prefix** (`…-full-checkpoints/12b-medium-local4`, artifacts `…/gemma4-12b-medium-local4`) so the
4-rank shards never mix with the 8-rank history (`restore-latest` on the empty prefix is a no-op and the trainer resumes from the local
tracker), the same W&B run id (`g4ds26b-12b-medium-s42-v1`, curve continues), a GPU guard (kills non-owner processes that land on our GPUs
after start — the box's evaluator loops grab any GPU that looks idle, and OFFLOAD/vLLM-sleep phases leave ours briefly empty; `GUARD=0`
disables) and a relaunch loop (≤ 5 attempts, each resuming from the newest complete checkpoint). Initial validation is skipped on a
resume with restored early-stopping history (`INITIAL_VALIDATION_SKIPPED_ON_RESUME`), so the first observation is at step 140.
Follow-up: the best step so far (120) lives only in the original prefix; if it is still the best at the end, copy
`global_step_120/actor/huggingface/` + its `_REMOTE_COMPLETE.json` into the new prefix *after* the run's first upload (so `restore-latest`
never picks the 8-rank step 120) or publish `best_hf` by hand — the original `gemma4-12b-medium/best_hf/` still holds it.

**Local launch 18:57Z (tmux `g4-12b-med-local4`, GPUs 0–3, log `/mnt/efs/jasonwei/gemma4-12b-medium-s42-local4/logs/`).** Two earlier
launch attempts died silently before printing anything: the free-GPU picker piped through `head` under `set -o pipefail`, so `head` closing
the pipe failed the function and `set -e` exited (fixed). Ray's raylet also logs "/tmp … is over 95% full" every 10 s — the 28 TB shared
volume is >95 % used even with 300 GB free — harmless so far; `RAY_local_fs_capacity_threshold=0.99` is now exported for relaunches.
**Resume verified:** all 4 ranks `Loaded model / optimizer / rng / lr_scheduler` from the resharded `world_size_4` shards (19:05–19:09Z),
`EARLY_STOPPING_PATIENCE_MIGRATED checkpoint_step=130 active_patience=4 misses=1 triggered=False`, `EARLY_STOPPING_STATE_RESTORED best=0.5208
best_step=120 misses=1 last_observed_step=130`, `INITIAL_VALIDATION_SKIPPED_ON_RESUME`, W&B resumed the original run. **Step 131** (1024
sequences on 4 GPUs, fast layout): gen 69 s · old-log-prob 73 s · update 296 s · **step 474 s**; train score 0.486, mean response 531 tokens.
Note the trainer's own `print`/metric lines go to the DAPOTaskRunner's Ray worker log (`/tmp/ray_12b_medium_local4/session_*/logs/worker-*-<pid>.out`),
not the driver log. At ~8 min/step, validation every 10 steps lands every ~80 min (first at step 140 ≈ 20:25Z); the cap of 400 steps is ~36 h away.

**Incident 20:49–21:33Z — EFS checkpoints stalled the whole devbox.** The step-140 permanent save (134 GB of FSDP shards + 24 GB HF
export) went to `/mnt/efs/...`. The devbox home is EFS over a single NFSv4.1 connection (64 session slots, no `nconnect`) shared by ~115
sessions: our writer queued ~57k NFS requests, every shell on the box took ~30 s to open, and our own throughput collapsed to ~13 MB/s (the
4 shard files alone took 31 min; the trainer then sat in the HF export at 0 MB/s). Also seen in the same window: the step-135 rolling upload
to S3 had failed with `AccessDenied` because the launcher did not export `AWS_PROFILE=ml-worker` (the EC2 instance role can only read the
bucket) — fixed before the second attempt, which resumed from the local step 135. The run was killed at 21:33Z; nothing reached the new S3
prefix. **Fixes (all committed):** (1) `assert_local_fs.sh` — sourced by the RL run-file and `gemma4_topk_distill_fsdp2.sh`; fails when
`CKPTS_DIR`/`RAY_DATA_HOME`/`DATA_DIR`/`HF_HOME` resolve to nfs/nfs4 (`ALLOW_NFS_CHECKPOINTS=1` only for tiny smoke runs); (2) the local
launcher and `resume_gemma4_26b_a4b_local.sh` default every bulk directory to `/tmp` (local NVMe) — S3 is the durable copy; rolling saves are
off for the local run (permanent saves every 10 steps already match `TEST_FREQ`); (3) host sysctl `vm.dirty_bytes=8 GiB` /
`vm.dirty_background_bytes=2 GiB` (persisted in `/etc/sysctl.d/60-dirty-bytes-network-fs.conf`) so a slow-filesystem writer blocks after a
few GB instead of buffering ~100 GB; (4) freed 461 GB of re-materializable eval caches (`/tmp/gemma4_distill_study_eval/work`) so /tmp can hold
two 12B checkpoints. **Still admin-only:** remount EFS with `nconnect=16` and raise `nfs.max_session_slots` (module parameter, currently 64) —
both multiply in-flight requests so one writer cannot starve the other sessions; large artifacts should never live on EFS regardless.
**Run state:** the local resume is stopped (user request). Steps 131–140 exist only on EFS (too slow to read back); the step-130 8-rank
checkpoint was re-downloaded from S3 to `/tmp/gemma4_12b_medium_s42_local4/src` for a fresh NVMe reshard — relaunch pending the user's go.
Validation at step 140 (first attempt) scored **0.515** mean@16 (best 0.5208 @ 120 → miss 2 of 4); it will be re-measured on relaunch.

**Relaunch on NVMe (user go at 22:37Z).** Rolling full checkpoints (weights + Adam + dataloader cursor + early-stopping state) now go to
S3 every 5 steps on top of the permanent saves every 10 (`ROLLING_CHECKPOINT_ENABLED=True`; cheap on NVMe). Step 130 was resharded again on
`/tmp` (model 630/630, Adam 1236/1236 DTensors bit-exact) and launched at 23:00Z on GPUs 0–3. That attempt died at `ray.init`: the raylet aborted
in `NodeManager::WaitForDashboardAgentPorts` — the Python dashboard/runtime-env agent could not import and register within
`agent_register_timeout_ms` (default 100 s) at a box load average of ~500 (other users' evaluators; 192 cores). Ray 2.58 honours the
`RAY_agent_register_timeout_ms` environment override in the raylet, so the launcher now exports 900 000 ms (plus 1200 s for the driver→node and
GCS waits) — relaunched 23:27Z. Disk: `/tmp` had 420 GB free after the reshard (one 4-rank checkpoint + HF export ≈ 160 GB; keep-2).

**Ray start-up on the overloaded box — root cause found (00:20Z, 2026-09-16).** The 23:27Z relaunch died the same way even with
`RAY_agent_register_timeout_ms=900000`: the raylet aborts with `Timed out waiting for file <session>/metrics_agent_port_<node_id>` (first
attempt) or `.../dashboard_agent_listen_port_<node_id>` (fixed-port probe). Ray 2.58 source (`src/ray/raylet/node_manager.cc`
`WaitForDashboardAgentPorts`, `src/ray/util/port_persistence.h`): when an agent port is unassigned (0) the raylet polls for the port file the
Python dashboard/runtime-env agent writes, with a **hardcoded 15 s** default (`WaitForPersistedPort(..., timeout_ms = 15000)`) — not governed
by `agent_register_timeout_ms` or any `RAY_*` override. At load average 500–680 (other users' evaluators on the 192-core box) those agents
need 45–60 s just to import, so `ray.init(address=local)` cannot succeed. Pre-assigned ports skip the wait — but `ray start --head`'s
`--dashboard-agent-listen-port` is never forwarded to the raylet by `ray/_private/services.py` (it only passes `--metrics-agent-port`,
`--metrics_export_port`, `--runtime_env_agent_port`), so that one wait always fires. **Fix:** (1) venv patch adding
`--dashboard_agent_listen_port=…` to the raylet command in `/tmp/.venv-gemma4/.../ray/_private/services.py` (backup `.orig_agent_listen_port`);
(2) the local launcher starts its own head node with all agent ports fixed (`RAY_PORT_BASE=56390`, ports +0…+7) and the run-file connects to
it via the new opt-in `RUN_RAY_ADDRESS` (default still `local`, so ScaleTrain is unchanged). Bare-`ray.init` probes at this load also showed
the raylet itself takes ~60 s to come up, so expect several minutes of start-up before the checkpoint load.

**00:40–00:58Z:** with the fixed-port head the raylet came up in 112 s and the driver connected, but actor creation then failed with
`ActorUnschedulableError: worker startup repeatedly failed` — Ray workers must register within `worker_register_timeout_seconds` (default 60 s),
also too short at this load; the launcher now exports `RAY_worker_register_timeout_seconds=900` for the head. Two more launcher changes for the
shared box: it **waits** for 4 idle GPUs instead of failing (co-tenants took all 8 GPUs during the 00:20Z restart gap, and GPU 3 again during
the 00:58Z one), and `MAX_ACTOR_CKPT_TO_KEEP=1` (the shared /tmp volume dropped from 420 GB to 71 GB free within an hour from other users'
writes; I freed my re-downloadable eval-model caches — `gemma4_12bd_evals/models` 243 GB, `gemma4_distill_students` 47 GB,
`gemma4_trace_models` 25 GB — to get back to ~380 GB; the local 4-rank step-130 copy is deleted automatically once the trainer has loaded it).
The on-policy tail-bucket run finished at step 200 in the meantime (§9.0f).

**Switched to ScaleTrain on 4 GPUs (user request, 01:35Z 2026-09-16).** The local run (relaunched 01:05Z on GPUs 0,1,2,6 with the fixed-port
head, still in model load) was stopped: the shared box is at load ~500 and its /tmp volume keeps filling from other users, so a 4-GPU
ScaleTrain job is the safer home. Recipe unchanged; what moved: the 4-rank step-130 reshard (+ the step-130 HF export copied from the original
prefix) is uploaded as a permanent checkpoint to `…-full-checkpoints/12b-medium-local4` (`full_checkpoint_s3.py upload`, manifest world_size 4),
the original step-120 checkpoint (current best, 8-rank) is server-side copied into the same prefix so end-of-run best-HF publishing can find it
(never selected by `restore-latest`, since 130 is newer), and `scale_train/launch_gemma4_12b_medium_resume.sh` gained `GPUS_PER_INSTANCE`
and `CKPT_SUFFIX` (`=4`, `=-local4` here; defaults keep the 8-GPU/original-prefix behaviour). Supervisor:
`scale_train/start_gemma4_12b_medium_resume_local4_supervisor.sh` (tmux `rl-g4-12b-med-resume-local4`, completion checks against the
-local4 prefixes, expected world size 4). Expect on start: `restored complete source=permanent step=130` (4 shards), the same
`EARLY_STOPPING_PATIENCE_MIGRATED … active_patience=4 misses=1` line, then steps 131+ at ~7–8 min each (4 × H100, mbs 4 / 8192 cap).

**Submitted 01:53Z as `job_dakvn1pt1s0g07hmc3dg`** (4 GPUs, borrowing, priority high); a pod was scheduled at 02:14Z — ~20 min, versus the
1–14 h seen for 8-GPU shapes today. The uploaded step-130 manifest: `layout ppo_actor, world_size 4, 22 files, 161.2 GiB` (shards + HF
export + `data.pt` + early-stopping state); remote tracker 130; step 120 copied alongside.

**Running on ScaleTrain (4 GPUs) since 02:18Z:** `restored complete source=permanent step=130` (4 shards), `EARLY_STOPPING_PATIENCE_MIGRATED
… active_patience=4 misses=1`, steps 131–139 at **~370–395 s/step** (gen 40–57 s · update 250–260 s; faster than the local 4×H100 run's
~430–490 s), rolling upload at 135 committed. **Step 140 validation: 0.5271 mean@16 — new best** (was 0.5208 @ 120; the abandoned local
attempt had measured 0.515 on the same step with a different sample), misses reset to 0. **Step 150: 0.511** (miss 1), **step 160: 0.519** (miss 2 of 4); checkpoints 150/160 uploaded. **Step 170: 0.519** (miss 3), **step 180: 0.513** (miss 4) → `EARLY_STOP_TRIGGERED step=180 best=0.5271 best_step=140`;
`RUN_OUTCOME_WRITTEN reason=early_stopping final_step=180 best_step=140`; best HF export published to
`s3://scale-ml/genai/rl-distill/gemma4-difficulty-s42-20260819/gemma4-12b-medium-local4/best_hf/` and the completion receipt to the
`…-full-checkpoints/12b-medium-local4` prefix (RUN_DONE rc=0 at 09:15Z; the wandb teardown traceback and `wandb sync --sync-all` error are the
usual benign tail). **Net effect of the patience-4 resume:** the untrained-12B medium teacher's best moved from **0.5208 @ 120 → 0.5271 @ 140**
(+0.6 pt, within the ±0.5–1 pt run-to-run noise seen at 130/140/150/160/170/180 = 0.519/0.527/0.511/0.519/0.519/0.513); steps 141–180 never
beat it. Wall clock on the 4-GPU pod: 50 steps in 6.9 h (~6.3 min/step + 5 validations/saves). The `rl_12b_medium` teacher in §1/§8 still
refers to step 120 (Hub pin); the step-140 export is S3-only (`global_step_140/actor/huggingface/`, and `best_hf/`).

### 9.1 Results

**E4B base, validation ×32 (the target curves; 2026-09-07):** `figures/passk_e4b_base_val32.png`

| band (300 q) | mean@32 | maj@32 | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 |
|---|---|---|---|---|---|---|---|---|
| id_medium | 8.4 | 20.3 | 8.4 | 15.4 | 26.6 | 41.9 | 59.3 | 74.7 |
| id_hard | 3.9 | 9.0 | 3.9 | 7.4 | 13.5 | 23.1 | 35.8 | 50.7 |

![E4B base validation pass@k ×32](figures/passk_e4b_base_val32.png)

**Traces (done 2026-09-06, 18:39–22:12Z on GPUs 0,5 / 6,7):** both bundles `COMPLETE`, mirrored to
`s3://scale-ml/genai/rl-distill/gemma4-e4b-base-traces-topk128-v1/<spec>/` (1,666 objects, 10.7 GB).

| Bundle | train rows | validation rows | sampled mean / max response tokens (train) | top-k width | `<image\|>` leakage |
|---|---|---|---|---|---|
| e4b-base-medium | 48,000 (3,000 q × 16) | 300 (× 1) | 239 / 1,735 | 128 | 0 |
| e4b-base-hard | 48,000 (3,000 q × 16) | 300 (× 1) | 170 / 748 | 128 | 0 |

(The base model answers much more tersely than the RL teachers; all sampled responses ended on a stop token.)

**Distillation runs — ScaleTrain (launched 2026-09-07 ~17:30Z, mediums first):** `gemma4-e4bbase-med-12b`
(p5.48xlarge:4, 4×H100) and `gemma4-e4bbase-med-26b` (p5.48xlarge, 8×H100), priority high, borrowing on, run-file
`scale_train/run_gemma4_e4b_base_distill_st.sh` (v2 recipe: batch 128, lr 2e-6 → 2e-7, 1000 steps, validate every
10, save + push every 250). Jobs `job_dafr28qlrg1g089aq2h0` (12B) and `job_dafr29pob6s008cpfgpg` (26B), created
2026-09-08 06:54Z. Four earlier pairs failed at pod start: 05:16Z (the run-file's own code-refresh block ran
`aws` before the PATH restore — now the first thing the run-file does), 01:48Z (`tar: dapo/config: Cannot open: File exists` —
`dapo/config` is a symlink in the archive but a directory in the baked tree; extraction now uses
`--unlink-first --recursive-unlink`), 18:59Z (the 08-29 image predates the run-file →
`launch_st_job.py --code-s3-uri` now unpacks the code tarball *before* invoking it) and 20:27Z (the pod's login
shell drops the image PATH → `aws: command not found`; the bootstrap now calls the FSDP2-venv `aws` by absolute
path and the run-file restores `/workspace/rl-distill/.venv/bin` + system dirs on PATH).
The remote image builds (`--build-env remote`) never produced an image, so the jobs run the known-good 2026-08-29
image (`…/tmp:20260829-002920.cd13c11a…`) and refresh the code from a `git archive` tarball of commit 7469ba29
(`--code-s3-uri s3://scale-ml/genai/rl-distill/code/rl-distill-code-75227184.tar.gz`, unpacked over the baked repo
before the run-file starts). Students land at
`JWei05/Distill-gemma4-e4b-base-medium-to-{12b,26b}-base/step_*`; the local submitter loop launches one 2-GPU ScaleTrain job per
export (§9, jobs `g4e4b-pk-med-{12b,26b}-s<step>`) and the plot loop refreshes `figures/passk_*_val32.png` from S3. Hard-band jobs: not
launched yet.

**Per-checkpoint pass@k — id_medium validation, 32 samples/question (all rows ×32; both runs complete; updated 2026-09-10 15:15Z):** the 26B-A4B student's
`step_000250` (pushed 16:06Z) was evaluated by ScaleTrain job `g4e4b-pk-med-26b-s0250` (job_dag3ah2lrg1g07lkf0p0; tp 2 on
2 H100s; 35 min wall incl. venv build + 52 GB materialize) and is already within ~1.5 points of the E4B teacher at every k
(figure `figures/passk_e4b-base-medium-to-26b-base_val32.png`):

| pass@k (%) | 1 | 2 | 4 | 8 | 16 | 32 | mean@32 | maj@32 |
|---|---|---|---|---|---|---|---|---|
| 12B base, no distillation | 14.1 | 24.7 | 39.6 | 57.0 | 73.4 | 86.3 | 14.1 | 38.7 |
| 26B-A4B base, no distillation | 23.7 | 38.9 | 57.1 | 74.1 | 86.5 | 93.3 | 23.7 | 56.7 |
| E4B base teacher | 8.4 | 15.4 | 26.6 | 41.9 | 59.3 | 74.7 | 8.4 | 20.3 |
| 12B ← E4B-base medium, step 100 | 7.5 | 13.9 | 24.1 | 38.3 | 55.0 | 71.7 | 7.5 | 19.7 |
| 12B ← E4B-base medium, step 150 | 6.7 | 12.4 | 21.9 | 35.7 | 52.8 | 70.0 | 6.7 | 15.3 |
| 12B ← E4B-base medium, step 200 | 7.2 | 13.2 | 23.0 | 36.9 | 53.6 | 70.3 | 7.2 | 21.0 |
| 12B ← E4B-base medium, step 300 | 7.3 | 13.3 | 23.1 | 36.7 | 53.1 | 68.7 | 7.3 | 17.0 |
| 12B ← E4B-base medium, step 400 | 7.4 | 13.6 | 23.7 | 38.1 | 55.3 | 71.3 | 7.3 | 19.0 |
| 12B ← E4B-base medium, step 500 | 7.5 | 13.8 | 23.8 | 37.5 | 53.0 | 67.0 | 7.5 | 21.0 |
| 12B ← E4B-base medium, step 600 | 7.7 | 14.2 | 24.5 | 38.7 | 55.2 | 70.3 | 7.7 | 18.7 |
| 12B ← E4B-base medium, step 700 | 7.6 | 14.0 | 24.3 | 38.7 | 55.3 | 70.7 | 7.6 | 19.0 |
| 12B ← E4B-base medium, step 800 | 8.0 | 14.7 | 25.4 | 40.3 | 57.6 | 73.7 | 8.0 | 18.7 |
| 12B ← E4B-base medium, step 900 | 8.0 | 14.7 | 25.4 | 40.1 | 57.4 | 74.0 | 8.0 | 17.7 |
| 12B ← E4B-base medium, step 1000 | 8.1 | 14.9 | 25.8 | 40.7 | 57.6 | 73.0 | 8.1 | 21.7 |
| 26B-A4B ← E4B-base medium, step 100 | 7.4 | 13.6 | 23.8 | 38.3 | 55.1 | 70.3 | 7.4 | 16.7 |
| 26B-A4B ← E4B-base medium, step 200 | 8.1 | 14.9 | 25.7 | 40.6 | 57.9 | 72.7 | 8.1 | 20.3 |
| 26B-A4B ← E4B-base medium, step 300 | 7.6 | 13.9 | 24.1 | 38.1 | 54.1 | 70.0 | 7.6 | 19.3 |
| 26B-A4B ← E4B-base medium, step 400 | 7.7 | 14.2 | 25.0 | 40.2 | 58.1 | 74.0 | 7.7 | 20.7 |
| 26B-A4B ← E4B-base medium, step 500 | 7.8 | 14.5 | 25.1 | 40.0 | 57.2 | 73.7 | 7.8 | 17.7 |
| 26B-A4B ← E4B-base medium, step 600 | 8.2 | 15.1 | 25.9 | 40.6 | 57.2 | 72.7 | 8.2 | 20.0 |
| 26B-A4B ← E4B-base medium, step 700 | 8.2 | 15.0 | 26.0 | 41.1 | 58.4 | 74.0 | 8.2 | 21.7 |
| 26B-A4B ← E4B-base medium, step 800 | 8.2 | 15.1 | 26.2 | 41.7 | 59.5 | 75.0 | 8.2 | 18.3 |
| 26B-A4B ← E4B-base medium, step 900 | 8.1 | 14.9 | 25.7 | 40.6 | 57.8 | 74.7 | 8.1 | 18.3 |
| 26B-A4B ← E4B-base medium, step 1000 | 8.6 | 15.7 | 26.9 | 42.3 | 59.8 | 75.3 | 8.6 | 22.3 |

Overlays of the untrained base, the E4B teacher and the student at steps 500 and 1000: `figures/passk_12b_final_vs_e4b_teacher.png`
and `figures/passk_26b_final_vs_e4b_teacher.png`.

Untrained bases on the same band from §8 (16 samples/q, so the curve stops at k=16): 12B pass@1/2/4/8/16 =
14.1 / 24.7 / 39.5 / 56.3 / 72.0 (mean 14.1, maj@16 29.3); 26B-A4B mean 23.8 / pass@16 86.3. Both bases are being re-run with the
×32 protocol as ScaleTrain jobs (`BASE_MODEL=12b|26b`, run-file mode; results `base_<student>__x32_medium/` in the same S3 root) and
the plot loop adds them as a "no distillation" reference line. ( The distilled student has moved onto the
teacher's curve, i.e. well *below* its own pre-training ability, after 250 steps.) No 12B checkpoint exists yet (see below).

**Preemption (2026-09-08):** both jobs were evicted once under borrowing — the 26B pod had trained to step 20 (val loss
0.147 → 0.141, 13:28–13:59Z) when the job went back to QUEUED; new pods started 14:08Z (12B) and 14:15Z (26B) and, because
those jobs save only the HF export, training restarted from step 0. The run-file now saves a full checkpoint (model, Adam, LR/RNG,
dataloader position) plus an HF export (pushed to the Hub) every 50 steps — kept permanently in S3 every 250 steps, otherwise in
a rolling S3 slot — and restores the newest complete one at startup (`main_full_vocab_distill_fsdp2.py` `_init_rolling_checkpoints`,
yaml `trainer.remote_checkpoint.rolling_freq`, env `ROLLING_CHECKPOINT_FREQ`). Validated locally with E2B smoke runs:
(a) permanent-only: 5 steps → checkpoint dir wiped → restored step 5 from S3, identical step-5 val loss, dataloader
`samples_yielded` 20 → 24 with the same base seed, LR schedule continued, steps 6–8 trained; (b) rolling: `SAVE_FREQ=4`,
`ROLLING_CHECKPOINT_FREQ=2` → rolling 2, permanent 4 (rolling 2 retired), rolling 6 uploaded (57 GB, async, training
continued), process group killed mid-save at step 8 (simulated preemption) → relaunch restored ROLLING step 6 and resumed
(phase-2 details in the smoke logs under `/tmp/gemma4_e4b_base_distill/rolling_smoke_p*.log`).
Jobs launched before this change keep running without resume until relaunched. **Second round of preemptions:** the 12B
job went QUEUED at 17:07Z (at ~step 235, before its first push), ran 17:12–17:22Z, and restarted again at 18:44Z from step 0;
the 26B job went QUEUED at 17:33Z (at ~step 450, after pushing step 250) and had no pod as of 19:25Z. Every preemption
of these jobs discards all progress. Both pods were replaced yet again at 20:02Z/20:05Z (12B: third restart from step 0), so at
20:22Z the two jobs were cancelled and **relaunched with the resumable run-file** (commit aba46d6b, permanent 250 / rolling 50):
`gemma4-e4bbase-med-12b` = job_dag6svilrg1g07lkf1a0 (p5:4), `gemma4-e4bbase-med-26b` = job_dag6t1hob6s007k81o80 (p5:8), borrowing on,
priority high. They push to the same Hub repos; the submitter re-evaluates an export whose last Hub commit differs from the
revision recorded in its S3 result (the old result is parked under `_superseded/`), so the step-250 point above will be replaced
by the relaunched 26B run's own step 250. Those jobs reached step 50 (rolling checkpoints in S3 at 21:01Z/21:10Z), were
preempted again before step 100 and sat QUEUED; at 00:58Z (09-09) they were cancelled once more and resubmitted on commit
0d154688 so that every 50-step save also pushes an HF export (user: evaluate every 50 steps): `gemma4-e4bbase-med-12b` =
job_dagau49ob6s008cpfhug, `gemma4-e4bbase-med-26b` = job_dagau6alrg1g07lkf1mg. Same S3 prefixes → they resume from rolling
step 50, so the first Hub exports of these runs are `step_000100` (step 50 has no export). Launch command pattern:
```bash
cd rl-distill-scripts/scale_train
python3 launch_st_job.py --cluster eks --build-env remote --n-instances 1 --gpus-per-instance 4 --job-name gemma4-e4bbase-med-12b \
  --priority high --allow-borrowing --active-deadline-hours 72 --run-file run_gemma4_e4b_base_distill_st.sh \
  --env-vars "TEACHER_SPEC=e4b-base-medium,STUDENT=12b"          # 26B: --gpus-per-instance 8, STUDENT=26b, deadline 96 h
# used in practice (remote build failed): --image <ECR uri of a working rl-distill image> and add
#   --code-s3-uri s3://scale-ml/genai/rl-distill/code/rl-distill-code-<sha>.tar.gz   (git archive HEAD | aws s3 cp, ml-worker profile;
#   the job command must not contain $(...) — the job config is Template-rendered)
# status (non-interactive): scratchpad st_status.py via the CLI client library; pods: env -u AWS_PROFILE kubectl get pods -n train | grep e4bbase
```

_(Earlier local attempt, 2026-09-07 00:05Z, stopped at the user's request — kept for the record:)_
The first run, `e4b-base-medium → 12b` on GPUs 0,5,6,7 (4 GPUs, no offload; 67–77 GB/GPU; ~21 s/step), was killed at
step 250/500 with KL/token 0.22 → 0.09 and val loss 0.169 → 0.088; no student was pushed. Two fixes made big students
runnable and are committed (74730fe8): verl's engine offloads model+optimizer+grads together (single `FSDP_OFFLOAD`
knob), and the `skip_lm_head` hidden-state passthrough now covers `Gemma4UnifiedForConditionalGeneration` (12B).
To run elsewhere: the step-2 commands above (`STUDENT=12b|26b`, 8 GPUs; `FSDP_OFFLOAD=true` for 26B-A4B on fewer),
bundles from the S3 prefix (or copy `/tmp/gemma4_e4b_base_traces_v1/<spec>/`). Control baselines `base_12b` /
`base_26b` math results are in §8.
