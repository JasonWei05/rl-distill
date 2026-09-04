# Gemma 4 RL, distillation, and base-model evaluation matrix

Status: complete and final-audit passing. The production packed-node job
`job_d9u0svrcrk5gppe43ho0` (`g4eval-packed15-jasonwei`) ran on one high-priority,
borrowing-disabled EKS eight-GPU node and completed on 2026-08-12 at 23:09 UTC.
All 15 registered models produced valid `RUN_COMPLETE.json` receipts; the
packed completion receipt records zero retries, zero permanent failures, and no
missing remote completions. The immutable result root is
`s3://scale-ml/genai/rl-distill/gemma4-rl-distill-base-evals-v2/`.

The strict final audit passed on 2026-08-12 at 23:14 UTC. It independently
verified 422,280 math samples, 394,080 effective OOD items, 12,564 registered
files, and 11,541,013,272 registered bytes. It generated `curves.csv`, all 12
planned lineage/dataset comparison figures, and 48 individual model/dataset
pass@k-plus-maj@k figures under `curves/by_model_dataset/`. The canonical
machine-readable report and generated table are respectively
`results/gemma4-rl-distill-base-evals-v2/final_audit.json` and
`results/gemma4-rl-distill-base-evals-v2/results_table.generated.md`.

Each lineage comparison also overlays the RL teacher behind every distilled
student. Teacher curves use the same color as their student with a faint,
dashed style so the student comparisons remain visually dominant.

Earlier canaries and retries established the scheduler and environment fixes
but are not result sources. In particular, the legacy
`transformers==4.57.6`/`vllm==0.15.1` image could not load Gemma 4's
`gemma4_unified` architecture. The successful job used the proven Gemma 4 cu129
stack (`transformers==5.14.1`, `vllm==0.25.1`) with isolated per-model caches.

## Final audited results

Math cells are `mean@k / maj@k / pass@k` in percent. OOD cells are harness
accuracy in percent.

The tables are organized by output architecture: distilled checkpoints are
grouped by student architecture, while base and direct-RL checkpoints are
grouped by their own architecture. Thus every E2B result is together, followed
by every E4B result and then every 12B result.

### E2B models

This groups the E2B PT base, direct E2B RL checkpoints, and every distilled
student whose target architecture is E2B.

| Model | Easy ID | Medium ID | MATH500 | GSM8K | MMLU-Pro | GPQA-Diamond | MMMLU-14K |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_e2b` | 5.70 / 12.00 / 35.00 (@16) | 3.35 / 4.80 / 26.80 (@16) | 4.72 / 9.60 / 34.40 (@16) | 8.13 / 12.59 / 34.95 (@8) | 23.81 | 19.70 | 48.20 |
| `rl_e2b_easy_total_step360` | 24.46 / 28.40 / 49.60 (@16) | — | 20.61 / 22.20 / 46.80 (@16) | 31.09 / 37.07 / 63.68 (@8) | 27.98 | 20.71 | 47.81 |
| `rl_e2b_medium_step180` | — | 9.72 / 11.80 / 34.40 (@16) | 12.12 / 12.00 / 42.00 (@16) | 23.80 / 28.43 / 56.71 (@8) | 27.07 | 19.70 | 48.32 |
| `distill_e4b_easy_to_e2b_step1000` | 19.01 / 28.00 / 57.40 (@16) | — | 15.40 / 23.40 / 52.60 (@16) | 29.14 / 40.33 / 68.23 (@8) | 26.25 | 23.23 | 46.92 |
| `distill_e4b_medium_to_e2b_step1000` | — | 8.72 / 14.40 / 39.00 (@16) | 12.54 / 18.20 / 50.20 (@16) | 21.95 / 31.92 / 59.06 (@8) | 27.03 | 23.74 | 47.72 |
| `distill_12b_easy_to_e2b_step1000` | 20.31 / 31.80 / 60.80 (@16) | — | 17.18 / 25.60 / 59.40 (@16) | 32.20 / 46.70 / 74.45 (@8) | 23.78 | 20.20 | 48.31 |
| `distill_12b_medium_to_e2b_step1000` | — | 11.51 / 16.60 / 45.60 (@16) | 16.68 / 26.20 / 55.00 (@16) | 30.06 / 43.44 / 71.42 (@8) | 24.26 | 17.17 | 47.81 |

### E4B models

This groups the E4B PT base, direct E4B RL checkpoints, and every distilled
student whose target architecture is E4B.

| Model | Easy ID | Medium ID | MATH500 | GSM8K | MMLU-Pro | GPQA-Diamond | MMMLU-14K |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_e4b` | 12.76 / 23.20 / 51.40 (@16) | 6.64 / 13.00 / 33.40 (@16) | 10.94 / 21.20 / 49.20 (@16) | 26.54 / 44.28 / 72.86 (@8) | 37.87 | 19.19 | 61.58 |
| `rl_e4b_easy_step160` | 33.21 / 40.80 / 67.60 (@16) | — | 30.66 / 36.20 / 64.20 (@16) | 67.96 / 76.50 / 89.92 (@8) | 42.18 | 18.69 | 58.33 |
| `rl_e4b_medium_step060` | — | 18.52 / 24.80 / 53.40 (@16) | 27.94 / 35.60 / 61.80 (@16) | 60.98 / 71.27 / 87.87 (@8) | 44.10 | 29.29 | 61.05 |
| `distill_12b_easy_to_e4b_step1000` | 39.14 / 54.40 / 77.20 (@16) | — | 35.39 / 46.40 / 71.00 (@16) | 69.42 / 81.58 / 92.95 (@8) | 37.87 | 12.63 | 59.78 |
| `distill_12b_medium_to_e4b_step1000` | — | 23.36 / 30.40 / 60.60 (@16) | 34.19 / 45.20 / 68.80 (@16) | 68.88 / 81.58 / 93.10 (@8) | 41.53 | 24.24 | 59.74 |

### 12B models

There are no distilled 12B students in this matrix, so this table contains the
12B PT base and the two direct 12B RL teacher checkpoints.

| Model | Easy ID | Medium ID | MATH500 | GSM8K | MMLU-Pro | GPQA-Diamond | MMMLU-14K |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_12b` | 18.19 / 31.00 / 57.80 (@16) | 10.05 / 17.20 / 40.40 (@16) | 17.19 / 29.60 / 58.60 (@16) | 46.40 / 69.98 / 89.31 (@8) | 45.25 | 23.23 | 65.96 |
| `rl_12b_easy_step160` | 53.99 / 66.60 / 86.60 (@16) | — | 47.83 / 59.20 / 79.00 (@16) | 85.24 / 90.30 / 97.04 (@8) | 50.48 | 24.24 | 66.15 |
| `rl_12b_medium_first_step140` | — | 34.40 / 41.20 / 72.00 (@16) | 44.35 / 54.20 / 74.80 (@16) | 83.13 / 89.92 / 96.06 (@8) | 52.03 | 23.23 | 66.12 |

## Scope

The registered matrix contains 15 unique models:

### E2B model registry

| Tag | Model | Lineage | Immutable artifact |
|---|---|---|---|
| `base_e2b` | E2B PT base | base; Easy and Medium ID | `google/gemma-4-E2B@d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` |
| `rl_e2b_easy_total_step360` | E2B RL | Easy | `s3://scale-ml/genai/rl-distill/verl-full-checkpoints/g4-e2b-easy-cont200-plus100-0807/global_step_160/actor/huggingface/` |
| `rl_e2b_medium_step180` | E2B RL | Medium | `JWei05/DAPO-gemma4-e2b-PT-DeepScaleR-medium20k-seed42-8k@bf240cf792843b04833d9e01257f7152615a66d4/step_000180` |
| `distill_e4b_easy_to_e2b_step1000` | E4B Easy RL -> E2B | Easy | `gemma4-offpolicy-distill-v1/topk128-e4b-easy-to-e2b-seed42-8k8-val500/global_step_1000/huggingface/` |
| `distill_e4b_medium_to_e2b_step1000` | E4B Medium RL -> E2B | Medium | `gemma4-offpolicy-distill-v1/topk128-e4b-medium-to-e2b-seed42-8k8-val500/global_step_1000/huggingface/` |
| `distill_12b_easy_to_e2b_step1000` | 12B Easy RL -> E2B | Easy | `gemma4-offpolicy-distill-v1/topk128-12b-easy-to-e2b-seed42-8k8-val500/global_step_1000/huggingface/` |
| `distill_12b_medium_to_e2b_step1000` | 12B Medium RL -> E2B | Medium | `gemma4-offpolicy-distill-v1/topk128-12b-medium-to-e2b-seed42-8k8-val500/global_step_1000/huggingface/` |

### E4B model registry

| Tag | Model | Lineage | Immutable artifact |
|---|---|---|---|
| `base_e4b` | E4B PT base | base; Easy and Medium ID | `google/gemma-4-E4B@411aa17b749aa952df1359d2dcea73917a544d9a` |
| `rl_e4b_easy_step160` | E4B RL | Easy | `JWei05/DAPO-gemma4-e4b-PT-DeepScaleR-easy10k-seed42-8k@34474993813ec3f2c3dfdc87d09e6269d5c5965a/step_000160` |
| `rl_e4b_medium_step060` | E4B RL | Medium | `JWei05/DAPO-gemma4-e4b-PT-DeepScaleR-medium20k-seed42-8k-medium-only-20260806@cddf55318f616c572d4f9e18636f25b4db76a25b/step_000060` |
| `distill_12b_easy_to_e4b_step1000` | 12B Easy RL -> E4B | Easy | `gemma4-offpolicy-distill-v1/topk128-12b-easy-to-e4b-seed42-8k8-val500/global_step_1000/huggingface/` |
| `distill_12b_medium_to_e4b_step1000` | 12B Medium RL -> E4B | Medium | `gemma4-offpolicy-distill-v1/topk128-12b-medium-to-e4b-seed42-8k8-val500/global_step_1000/huggingface/` |

### 12B model registry

| Tag | Model | Lineage | Immutable artifact |
|---|---|---|---|
| `base_12b` | 12B PT base | base; Easy and Medium ID | `google/gemma-4-12B@023679ed352de9bb66cc873c9009ce3482585c08` |
| `rl_12b_easy_step160` | 12B RL | Easy | `JWei05/DAPO-gemma4-12b-PT-DeepScaleR-easy10k-seed42-8k@9c42ffaf845e8be158ccff6a87a1352177016144/step_000160` |
| `rl_12b_medium_first_step140` | 12B RL, Medium off-policy teacher | Medium | `JWei05/DAPO-gemma4-12b-PT-DeepScaleR-medium20k-seed42-8k-medium-only-20260806@2c195f8f86de44cc00f51ce5539950b6841a5461/step_000140` |

The E2B checkpoint is continuation `global_step_160`, corresponding to total
training step 360. Its Easy-ID `mean@16` was `0.249375`, higher than the final
total-step-400 checkpoint (`0.246`). Its `_REMOTE_COMPLETE.json` was verified on
2026-08-11 and binds a complete four-rank checkpoint plus the HF export.

The E2B Medium run `nnivxmap` reached its highest validation score at step 170,
which was not a save step. Among the saved 20-step checkpoints, step 180 is best
with Medium-ID `mean@16 = 0.10075`. Immutable HF commit
`bf240cf792843b04833d9e01257f7152615a66d4` contains the complete
`step_000180` export.

The 12B Medium selection intentionally remains the first-run step-140 checkpoint.
That exact immutable model generated the off-policy traces used for both 12B
Medium distillation runs. The later duplicate 12B Medium run is excluded from
this matrix even though one of its checkpoints has a higher validation score;
the comparison is lineage-matched to the actual distillation teacher.

The source registry is
`config/gemma4_rl_distill_eval_sources.json`. It is the canonical roster and
also records each model's Easy/Medium routing.

## Packed-scheduler reference provenance

The mixed-size scheduling design was checked against the requested short-horizon
Qwen 2.5 3B reference in `/mnt/efs/jasonwei/src/models/tmp/scalar-ppo`. The
relevant file is `docs-ppo/run_pack_queue.sh`, mirrored byte-for-byte at
`SkyRL/scale/train/examples/deepscaler/run_pack_queue.sh` (SHA256
`9d009b49fcbb7c12fa0f8a3c7da63f8da9f40f3d4b00558824320fa0233f290f`).
Both copies were **untracked** in the heavily dirty `jason/scalar-ppo` working
tree on 2026-08-12; they are not part of the branch's latest commit
`f68d3d0b0a`. This run therefore records the exact working-tree file hash and
does not incorrectly attribute the scheduler to a committed revision.

The packed Gemma 4 scheduler preserves the reference's important invariants:

- one full eight-GPU node and a physical free-GPU pool;
- mixed one- and two-GPU tasks with greedy first-fit backfill and no phase
  barrier;
- isolated child process groups and targeted shutdown rather than global
  cleanup while neighboring tasks are live;
- explicit per-task completion evidence before GPUs are returned; and
- immediate launch of the next queued task that fits after a child exits.

The evaluation-specific implementation replaces Ray port/temp isolation with
per-model vLLM, Triton, TorchInductor, model, log, and result directories. It
also adds durable S3 state, content-bound `RUN_COMPLETE.json` receipts,
three-attempt retry, recovery skipping only valid remote completions, signal
propagation, and a correct success/failure exit code. The reference shell file
ends with unconditional `exit 1`; that historical defect is intentionally not
copied.

This behavior was exercised, not only dry-run inspected. Retry 2's durable
state began with the registered five-task 8/8-GPU packing at 00:13 UTC. By
00:16:23 UTC, `rl_12b_easy_step160` was still live on GPUs 2-3 while released
GPUs had already backfilled `rl_e4b_easy_step160` on GPU 0,
`rl_e4b_medium_step060` on GPU 1, `rl_e2b_medium_step180` on GPU 4,
`distill_e4b_easy_to_e2b_step1000` on GPU 5, and
`rl_e2b_easy_total_step360` on GPU 6. GPU 7 was between launches during the
configured 20-second stagger. That S3 state directly demonstrates mixed-size
asynchronous backfill without waiting for the surviving 12B task or a phase
barrier.

## Math protocol

Every math task uses the original unified 12-shot raw-completion template:

`data/gemma3_it_fewshot_math.jinja`, SHA256
`27b8801d8b61a413a9bb3b54b6f55e16217eff3e55f7c560377c8a162dd63c1c`.

Sampling is unchanged from the audited three-model evaluation:

- temperature `1.0`, top-p `1.0`, top-k disabled (`-1`);
- prompt limit `4,096`, response limit `8,192`, context `12,288`;
- top-128-plus-residual predictive-entropy statistics;
- independent deterministic per-question/per-sample seeds;
- the smallest power-of-two sample count making requests strictly greater than
  2,000.

| Dataset | Pinned source | Unique | Samples/question | Requests/model |
|---|---|---:|---:|---:|
| Easy ID | `JWei05/DeepScaleR-Easy-10k@0c3e81d98fad8783f6ab93cf3732ce58f159b555` held-out val | 500 | 16 | 8,000 |
| Medium ID | `JWei05/DeepScaleR-Medium-20k@c3db94f80a3abe079fdf457fe01555544b8bc2dd` held-out val | 500 | 16 | 8,000 |
| MATH500 | `HuggingFaceH4/MATH-500@6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` | 500 | 16 | 8,000 |
| GSM8K | `openai/gsm8k@740312add88f781978c0658806c59bc2815b9866` | 1,319 | 8 | 10,552 |

Easy-trained models receive Easy ID; Medium-trained models receive Medium ID.
Each base receives both ID sets. Every model receives MATH500 and GSM8K.
Correctness is the current project policy: grade only the last well-formed
`\boxed{}`/`\fbox{}` and mark correct when either `math_verify` or the bounded
Miles-style SymPy fallback accepts it.

This is 422,280 math generations across all 15 models.

## Out-of-domain protocol

OOD benchmarks retain their harness-native prompts, not the math template:

| Benchmark | Items/model | Task | Protocol |
|---|---:|---|---|
| MMLU-Pro | 12,032 | `mmlu_pro` | 5-shot generative CoT, greedy, `max_gen_toks=2048` |
| GPQA-Diamond | 198 | `gpqa_diamond_cot_n_shot` | 5-shot generative CoT, greedy |
| MMMLU-14K | 14,042 | `gemma4_mmmlu14k` | registered 5-shot likelihood subset |

All use `add_bos_token=True`, full datasets, and no `--limit`. The pinned
lm-eval commit is `f4d4b3de3ee6741a7151a9fe74945ee515262f4c`.
Dataset revisions remain MMLU-Pro `b189ec765aa7ed75c8acfea42df31fdae71f97be`,
GPQA `633f5ee89ab8ad4522a9f850766b73f62147ffdd`, and MMMLU
`325a01dc3e173cac1578df94120499aaca2e2504`.

This is 26,272 benchmark items/model and 394,080 across all models.

## Scripts and execution boundary

- `run_gemma4_rl_distill_base_evals.py`: top-level planner; dry-run by default.
- `data/prepare_gemma4_rl_distill_eval_data.py`: pins and materializes the four
  math datasets plus their sampling manifest.
- `data/materialize_gemma4_eval_models.py`: prints the 15 download/sync commands
  by default; with `--execute`, writes a content-bound resolved model registry.
- `run_gemma4_math_4gpu.py`: existing four-GPU math scheduler, generalized to
  accept the resolved registry and per-model ID routing.
- `eval_math_passk.py`: unchanged generation, trace, grading, entropy, and
  aggregation engine.
- `run_gemma4_ood_4gpu.py`: existing four-GPU OOD scheduler, generalized to
  accept the resolved registry.
- `eval_gemma4_ood.py`: unchanged pinned lm-eval wrapper.
- `data/prepare_gemma4_mmmlu14k.py`: unchanged deterministic MMMLU task builder.
- `scale_train/run_gemma4_rl_distill_eval_one_model.sh`: one model's complete
  math-then-OOD sequence, with resume-aware S3 restore/upload.
- `scale_train/run_gemma4_rl_distill_eval_packed.py`: greedy mixed-size GPU-pool
  scheduler, durable S3 state, per-model logs, retries, and exact remote
  completion verification.
- `scale_train/run_gemma4_rl_distill_eval_packed_node.sh`: prepares shared data
  once, verifies eight visible GPUs, and starts the packed scheduler.
- `scale_train/launch_gemma4_rl_distill_eval_packed.py`: submits exactly one
  full-node, high-priority, borrowing-disabled ScaleTrain job.
- `audit_gemma4_rl_distill_eval_results.py`: independently rehashes each
  per-model completion manifest, verifies exact math UID/sample-index coverage
  and full-sample metrics from traces, checks every OOD result/sample receipt,
  and emits a report-ready JSON and Markdown table.
- `finalize_gemma4_rl_distill_eval_results.sh`: after the packed completion
  marker exists, synchronizes the immutable result tree, runs the final audit,
  and generates the verified pass@k/maj@k curve set.

The packed node allocates two GPUs to each 12B model and one GPU to every E2B or
E4B model. The initial work-conserving packing is:

- `base_12b` on GPUs 0-1;
- `rl_12b_easy_step160` on GPUs 2-3;
- `rl_12b_medium_first_step140` on GPUs 4-5;
- `base_e2b` on GPU 6;
- `base_e4b` on GPU 7.

When any model finishes its complete math and OOD sequence, its physical GPUs
return to the pool and the first queued model that fits launches immediately;
there is no phase barrier. Shared math/MMMLU data is prepared once. Per-model
model and result directories remain isolated, and valid remote
`RUN_COMPLETE.json` markers are skipped on a recovery launch. Failed tasks are
retried up to three times while unrelated model tasks continue.

The scheduler-only dry run is:

```bash
python rl-distill-scripts/scale_train/run_gemma4_rl_distill_eval_packed.py --dry-run
```

The one-job ScaleTrain dry run is:

```bash
python rl-distill-scripts/scale_train/launch_gemma4_rl_distill_eval_packed.py --dry-run
```

The production launch omits `--dry-run`. Its S3 result root is
`s3://scale-ml/genai/rl-distill/gemma4-rl-distill-base-evals-v2/`.

After all 15 remote completion markers and the packed completion marker exist,
the local finalization command is:

```bash
bash rl-distill-scripts/finalize_gemma4_rl_distill_eval_results.sh
```

It writes `final_audit.json`, a generated Markdown result table, `curves.csv`,
12 lineage/dataset comparison figures, and 48 individual model/dataset
pass@k-plus-maj@k figures below
`rl-distill-scripts/results/gemma4-rl-distill-base-evals-v2/`. The canonical
results document is updated only from that passing audit, never directly from
an in-progress S3 tree.

## Plan of action

1. Build and submit the one-node/eight-GPU packed ScaleTrain job with borrowing
   disabled.
2. Monitor the durable packed state and per-model logs; repair and relaunch only
   if a model exhausts its in-job retries or the pod fails.
3. Require valid remote completion markers for all 15 models.
4. Audit exact math UID/sample-index coverage, merged mean@k, maj@k, pass@k,
   entropy outputs, model identities, and trace hashes.
5. Audit all MMLU-Pro, GPQA-Diamond, and MMMLU-14K result keys, effective counts,
   sample logs, model identities, and output hashes.
6. Generate four lineage-matched pass@k/maj@k comparison sets. Each E2B Easy
   or Medium plot includes the base model, direct-RL model, E4B-teacher
   distillation, and 12B-teacher distillation; each E4B plot includes the base,
   direct-RL, and 12B-teacher distillation. ID, MATH500, and GSM8K remain
   separate figures for every set.
7. Add the verified tables and graph references to
   `GEMMA4_DISTILL_VS_RL_EXPERIMENTS.md`.
