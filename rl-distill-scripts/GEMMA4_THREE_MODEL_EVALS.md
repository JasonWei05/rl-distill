# Gemma 4 E2B three-model evaluations

## Objective

Evaluate the following models under one auditable protocol:

1. Gemma 4 E2B base.
2. The final E4B-RL-to-E2B off-policy distillation checkpoint (step 750).
3. The final E2B RL checkpoint (step 125).

This evaluation supports the broader distillation-versus-RL project documented in
`GEMMA4_DISTILL_VS_RL_EXPERIMENTS.md`. Full evaluations are intentionally not launched by the
preparation commands below.

## References inspected

Project documentation and code:

- `FEWSHOT_MATH_RL.md`
- `GEMMA3_PT_EVAL_REPLICATION.md`
- `GEMMA4_DISTILL_VS_RL_EXPERIMENTS.md`
- `eval_math_passk.py`, `eval_gemma4_math.py`, `eval_gemma4_ood.py`, and `gemma4_eval_metrics.py`
- Existing math data converters and evaluation tests

External/pinned evaluation sources:

- Gemma 4 Technical Report, arXiv `2607.02770`
- `lm-evaluation-harness` at commit `f4d4b3de3ee6741a7151a9fe74945ee515262f4c`
  (`lm_eval==0.4.13.dev0`)
- Harness MMLU-Pro, GPQA, and OpenAI-MMMLU task definitions and READMEs

The Gemma 4 report's Table 5 evaluates final instruction-tuned models, generally in thinking mode.
It reports E2B MMLU-Pro 60.0, GPQA-Diamond 43.4, and MMMLU 67.4, but does not disclose enough
prompting and sampling details to reproduce those scores exactly. Our PT/RL/distilled models and
raw-completion harness protocol are therefore a controlled comparison among our three models, not a
strict reproduction of Table 5.

## Model artifacts

| Tag | Artifact | Expected identity |
|---|---|---|
| `base_e2b` | pinned `google/gemma-4-E2B` snapshot `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` | `bde9e800223cdd62228ce39e0305398f6ada05b98adaf438b0b3d3d3c3015561` |
| `distilled_e2b_step750` | `e4b-rl100-to-e2b-topk128-750-seed42/global_step_750/huggingface` | `beef12a146c3373a049467b42412520929228849570ba4cf495107e4597add03` |
| `rl_e2b_step125` | vLLM-ready materialization of `JWei05/nemorl-dapo-gemma4-e2b-pt-rl-step-125` | `cafb31ba68a05aaedd22584230b6bc3100bad61f710e7d33ce4da019acb7e25c` |

The raw RL export lacks `processor_config.json` and 60 shared-KV aliases. Do not evaluate that
directory directly. `data/materialize_gemma4_eval_checkpoint.py` copies the processor metadata from
the immutable base revision, expands the aliases with the existing Gemma 4 checkpoint helper,
rebuilds the safetensors index, and records source/output hashes.

## Math protocol

All math datasets use the exact 12-shot prompt from
`data/gemma3_it_fewshot_math.jinja`, SHA256
`27b8801d8b61a413a9bb3b54b6f55e16217eff3e55f7c560377c8a162dd63c1c`.

Generation configuration:

| Setting | Value |
|---|---:|
| temperature | 1.0 |
| top-k | -1 |
| top-p | 1.0 |
| maximum response tokens | 8,192 |
| maximum prompt tokens | 4,096 |
| total model context | 12,288 |
| predictive entropy top-k width | 128 |

“Duplicate by two until greater than 2,000 questions” is implemented as independent seeded samples
per unique question. The smallest power-of-two factor whose total is **strictly greater than 2,000**
is used:

| Dataset | Unique questions | Samples/question | Total requests/model |
|---|---:|---:|---:|
| In-distribution validation (full) | 200 | 16 | 3,200 |
| In-distribution validation (overlap-free diagnostic) | 193 | 16 | 3,088 derived from the full-set traces |
| MATH500 | 500 | 8 | 4,000 |
| GSM8K | 1,319 | 2 | 2,638 |
| OlympiadBench | 674 | 4 | 2,696 |
| MinervaMath | 272 | 8 | 2,176 |
| AIME 2025 | 30 | 128 | 3,840 |
| AIME 2026 | 30 | 128 | 3,840 |

The parquets contain one row per unique question. `eval_math_passk.py` consumes the generated
manifest and creates the independent samples with deterministic per-question/per-sample seeds. It
streams full response traces to JSONL and stores compact top-128-plus-residual predictive-entropy
statistics by default. Full per-token top-128 arrays are optional and are not required for this
comparison. The 193-question overlap-free result is filtered and re-aggregated from the full
200-question traces; it does not trigger another 3,088 generations. The resulting production total
is 22,390 math generations per model and 67,170 across all three models.

Correctness uses the repository's strict single-`\boxed{}` `math_verify` scorer. Semantic answer
classes are built with the same verifier before majority voting. Unbiased pass@k uses all available
samples. The conservative default reports mean@k and plurality maj@k only at the full observed sample
count; smaller-k mean/majority curves require an explicit subset estimator. Semantic equivalence is
required to succeed in both verifier directions. A timeout, asymmetric comparison, or proposed
equivalence that conflicts with the independently computed correctness label is recorded and treated
as non-equivalent. This prevents an uncertain or internally inconsistent verifier result from
merging answer classes without aborting the full evaluation.

### Pinned math sources

| Dataset | Revision |
|---|---|
| `HuggingFaceH4/MATH-500` | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |
| `openai/gsm8k` | `740312add88f781978c0658806c59bc2815b9866` |
| `math-ai/OlympiadBench` | `4faaf1e6ec17d11a4218a9bf4c049ecaf954dd84` |
| `math-ai/minervamath` | `ee46ddc498933b1977577953250ca5c66be64f96` |
| `MathArena/aime_2025` | `c94da77eb22bbd6439e62a323bec18493a421302` |
| `MathArena/aime_2026` | `d2de22f3c656b4f56cf8981212186377d1e23bc3` |

The in-distribution source is
`/tmp/verl/data/deepscaler_4of4strict_rl_val200_x16.parquet`, SHA256
`a92f69f247e352889ed06601aeb415434880c1402603914e73045668f78b647c`. Seven of its 200 unique
questions also occur in the training parquet. The preparation script emits both the full 200-question
set and a 193-question overlap-free diagnostic.

## Priority math results

The priority ID-validation, MATH500, and GSM8K matrix completed on 2026-07-31. The final run contains
exactly 29,514 traces. `priority_core_audit.json` independently re-read every final JSONL trace,
verified each question's exact sample-index set, recomputed all full-sample metrics and entropy
summaries, and matched the merged metric files. The overlap-free 193-question rows below were then
filtered and re-aggregated from the saved full-200 traces without repeating inference. Their source
trace identities, clean-parquet identity, counts, and metrics are recorded in
`priority_core_clean193.json`.

| Dataset | Model | Full k | mean@k | maj@k | pass@k | Sequence entropy (nats) |
|---|---|---:|---:|---:|---:|---:|
| ID validation (full 200) | Base E2B | 16 | 4.97% | 9.00% | 34.50% | 0.8989 |
| ID validation (full 200) | Distilled E2B step 750 | 16 | **19.00%** | **29.50%** | **53.50%** | 0.2915 |
| ID validation (full 200) | RL E2B step 125 | 16 | 17.84% | 23.50% | 49.00% | 0.2638 |
| ID validation (clean 193) | Base E2B | 16 | 5.05% | 9.33% | 34.20% | 0.8990 |
| ID validation (clean 193) | Distilled E2B step 750 | 16 | **18.81%** | **29.53%** | **52.85%** | 0.2923 |
| ID validation (clean 193) | RL E2B step 125 | 16 | 17.88% | 23.32% | 48.70% | 0.2634 |
| MATH500 | Base E2B | 8 | 4.58% | 6.40% | 23.60% | 0.9205 |
| MATH500 | Distilled E2B step 750 | 8 | **14.82%** | **19.00%** | **40.80%** | 0.3163 |
| MATH500 | RL E2B step 125 | 8 | 14.75% | 16.80% | 37.80% | 0.2729 |
| GSM8K | Base E2B | 2 | 7.92% | 2.73% | 13.57% | 0.8257 |
| GSM8K | Distilled E2B step 750 | 2 | **30.10%** | 17.21% | **42.84%** | 0.3019 |
| GSM8K | RL E2B step 125 | 2 | 29.76% | **18.80%** | 41.02% | 0.2966 |

At this checkpoint, distillation is stronger than direct E2B RL on all three datasets by mean@k and
pass@k. It is also stronger by maj@k on ID validation and MATH500; RL is 1.59 points higher by
maj@2 on GSM8K. Both trained models substantially outperform base E2B. Their predictive entropy is
similar to one another and much lower than base E2B under this top-128-plus-residual lower-bound
estimator.

## Out-of-domain protocol

OOD evaluations use the harness-native raw-completion prompts, not the 12-shot math template. They
use full splits, never `--limit`, and add one BOS token for continuity with the Gemma 3 PT
replication.

| Benchmark | Harness task | Few-shot | Native inference |
|---|---|---:|---|
| MMLU-Pro | `mmlu_pro` | 5 | generative CoT, greedy, `max_gen_toks=2048` |
| GPQA-Diamond (recommended primary) | `gpqa_diamond_cot_n_shot` | 5 | generative CoT, greedy |
| GPQA-Diamond (continuity alternative) | `gpqa_diamond_n_shot` | 5 | multiple-choice likelihood |
| MMMLU | registered reduced-MMMLU tasks | 5 | multiple-choice likelihood, 14,042 items total |

Pinned dataset revisions are checked before the harness starts. The two native harness datasets must
also have their pinned revision at the current Hub default; the custom MMMLU YAML pins its revision
directly and therefore does not require the Hub default branch to remain unchanged:

- `TIGER-Lab/MMLU-Pro@b189ec765aa7ed75c8acfea42df31fdae71f97be`
- `Idavidrein/gpqa@633f5ee89ab8ad4522a9f850766b73f62147ffdd`
- `openai/MMMLU@325a01dc3e173cac1578df94120499aaca2e2504`

GPQA-Diamond access has been verified and exposes 198 rows. Standard English MMLU contains 14,042
test questions across 57 subjects. Full MMMLU repeats those questions across 14 translated locales,
for 196,588 evaluations. This study instead evaluates exactly 14,042 MMMLU rows: every underlying
MMLU test question appears once, with its locale assigned deterministically while balancing locales
within subjects. This preserves all 57 subjects and all 14 locales without evaluating 14
translations of every question. The reduced manifest and task definitions are pinned and must not
use harness `--limit`, which would select ordered prefixes rather than the preregistered sample.
The generated task tree is stored under
`/lambda/nfs/Jason-scale/rl-distill-evals/gemma4-three-model/data/mmmlu14k_tasks`: it contains 798
locale/subject leaf tasks, gives every locale exactly 1,003 evaluation rows, and retains each native
locale/subject prompt plus a full-subject pool for the five few-shot examples. Its 815 registered
files are SHA256-bound by `manifest.json`.

The OOD wrapper loads the repository `.env` before checking revisions or launching the harness. This
is required because the accepted GPQA account token is stored there rather than globally exported.

## Out-of-domain results

The production OOD matrix started on 2026-07-31. Completed results are added here only after the
harness exits successfully and the scheduler verifies the expected result key, effective sample
count, logged-sample count, model identity, and content hashes of the result/sample artifacts.

| Benchmark | Model | Score | Standard error | Samples |
|---|---|---:|---:|---:|
| MMLU-Pro, 5-shot CoT | Base E2B | 24.04% | 0.38% | 12,032 |
| MMLU-Pro, 5-shot CoT | Distilled E2B step 750 | 25.96% | 0.39% | 12,032 |
| MMLU-Pro, 5-shot CoT | RL E2B step 125 | **27.29%** | 0.40% | 12,032 |
| GPQA-Diamond, 5-shot CoT | Base E2B | 22.73% | 2.99% | 198 |
| GPQA-Diamond, 5-shot CoT | Distilled E2B step 750 | 19.70% | 2.83% | 198 |
| GPQA-Diamond, 5-shot CoT | RL E2B step 125 | **24.75%** | 3.07% | 198 |
| MMMLU-14K, 5-shot | Base E2B | **48.16%** | 0.41% | 14,042 |
| MMMLU-14K, 5-shot | Distilled E2B step 750 | 48.03% | 0.41% | 14,042 |
| MMMLU-14K, 5-shot | RL E2B step 125 | 48.14% | 0.41% | 14,042 |

MMLU-Pro uses the harness `exact_match,custom-extract` aggregate. GPQA-Diamond uses the harness
`exact_match,flexible-extract` aggregate. MMMLU uses `acc,none`; `acc_norm,none` is identical for
these four-choice tasks. `ood_results_audit.json` verifies all nine successful scheduler tasks,
result sample counts, and logged sample coverage. The three MMMLU results differ by at most 0.13
percentage points, far below their approximately 0.41-point standard errors.

## Preparation and preflight

The pinned harness is installed in `/tmp/.venv-gemma4-e2e`.

```bash
/tmp/.venv-gemma4-e2e/bin/python \
  rl-distill-scripts/data/prepare_gemma4_three_model_eval_data.py

/tmp/.venv-gemma4-e2e/bin/python \
  rl-distill-scripts/data/materialize_gemma4_eval_checkpoint.py \
  --source-dir /tmp/nemorl-e2b-step125-hf.sg62B9 \
  --output-dir /lambda/nfs/Jason-scale/rl-distill-evals/gemma4-three-model/models/nemorl-e2b-rl-step125-vllm \
  --expected-source-model-sha256 987bc77de8dd3d705b90f10ecf8f6ff4dd6166d6c1554cc925b49cd085ae8e30

/tmp/.venv-gemma4-e2e/bin/python \
  rl-distill-scripts/run_gemma4_three_model_evals.py --preflight
```

The launcher defaults to printing an auditable command manifest only. `--preflight` validates
dataset/model/harness identities and runs OOD dry-runs. `--execute` is the explicit switch that
starts generation/evaluation. A single model can be selected with, for example,
`--models base_e2b` and assigned to a GPU using `CUDA_VISIBLE_DEVICES`.

Production order is fixed as follows:

1. Priority pass: run the full 200-question ID validation set, MATH500, and GSM8K for all three
   models. This is 9,838 generations per model and 29,514 total. Results are isolated under
   `/lambda/nfs/Jason-scale/rl-distill-evals/gemma4-three-model/priority-core-results`.
2. Verify and report the priority-pass metrics before resuming the remaining math datasets.
3. Run the remaining math benchmarks for all three models. `run_gemma4_math_4gpu.py` uses GPUs 0-3
   as four independent vLLM workers, dynamically schedules non-overlapping dataset shards, and
   merges shard metrics only after every selected shard succeeds.
4. Derive the clean 193-question ID report from the full 200-question traces when assembling the
   complete math report.
5. Run the three OOD families only after the math matrix passes validation. OOD uses the same
   four-GPU worker policy where the task implementation permits independent shards.

## Completed verification

- 234 focused evaluation, metrics, launcher, trace, upload, overlay, and FSDP2-safety tests pass;
  Ruff passes on all new/modified Python files, and both modified shell launchers pass `bash -n`.
- All math parquets were materialized and their row counts, UID counts, and SHA256 hashes match the
  manifest.
- The RL evaluation checkpoint contains the pinned processor metadata and all 60 expanded shared-KV
  aliases. Its Transformers last-token logits are bit-exact against the raw RL export
  (`max_abs_logit_diff=0.0`).
- All three models complete a live vLLM generation.
- All three models complete a one-question 12-shot math evaluation with top-128 predictive
  statistics and strict grading.
- One leaf task from each OOD family completed end to end through lm-eval: `mmlu_pro_math`,
  `gpqa_diamond_cot_n_shot`, and `mmmlu_de_de_abstract_algebra`. This diagnostic alone used
  `--limit 1`; production commands never use `--limit`.

## Registered protocol decisions

1. GPQA-Diamond uses the 5-shot generative CoT harness task.
2. Both the full 200-question and overlap-free 193-question ID results are reported.
3. Math uses sampled decoding only; no greedy math diagnostic is run.
4. Reduced MMMLU contains exactly 14,042 deterministic subject/locale-stratified items.
5. Mean@k, maj@k, and pass@k are reported only at each dataset's full sample count.
6. Majority is unique plurality and ties are wrong.
7. GPQA retains the harness-native response cap.

## Execution status

- Protocol finalized: 2026-07-31.
- Math matrix: the initial full-suite workers were stopped on 2026-07-31 at the user's request after
  the verifier fix was validated. Their partial traces are not accepted results. The priority pass
  targets only ID-full-200, MATH500, and GSM8K across all three models. It completed with all 29,514
  final generated traces, three merged model reports, and a successful independent trace-level
  audit. The clean-193 diagnostic was subsequently derived from the 9,264 applicable saved ID traces
  with no additional generation and recorded in `priority_core_clean193.json`.
- During the priority pass, `math_verify` incorrectly proposed equivalence between answers with
  conflicting correctness labels (for example, a wrong `45/8` expression and the correct `37/8`
  expression). The correctness-stratified fail-closed policy above was added and validated before
  reporting results. Complete generation traces produced before the fix are regraded from their
  stored predictions; generation is not repeated, and the recovery is recorded in shard metrics.
- OOD matrix: the earlier waiting process was stopped with the full-suite workers. The registered
  MMLU-Pro, GPQA-Diamond 5-shot CoT, and 14,042-item MMMLU matrix is launched after the audited
  priority report. The nine-task, four-GPU production matrix started and completed on 2026-07-31.
  All nine tasks exited successfully; the scheduler dynamically assigned MMLU and GPQA tasks as
  GPUs became available. The final task-level and sample-log coverage audit passed.
