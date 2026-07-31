# Gemma 4 Distillation vs. Reinforcement Learning Experiments

Status: canonical design, preparation runbook, and execution ledger. Both complete local generation
bundles now exist and validate: 48,615 train plus 1,000 validation traces in each direction, with
exact token IDs, response masks, and stored vLLM top-128 targets. Expanded cross-engine diagnostics
show that vLLM and an unsharded HF BF16+SDPA training-shaped forward are close but not bit-identical.
The immutable vLLM bundles remain the generation record; the intended primary training path derives a
separate, identity-bound unsharded-HF target overlay over the exact stored token sequences. That
rescore and the E2B-base-to-E4B student run are now authorized, subject to the parity, preflight, and
one-step smoke gates below. Both immutable vLLM source
bundles are now public on Hugging Face at independently verified commits: E2B-base traces at
`JWei05/gemma4-e2b-base-topk128-traces@e32aaa02681ae83b3d7256b1b155c9084da2f289` and E4B-RL
step-100 traces at `JWei05/gemma4-e4b-rl100-topk128-traces@2b6e49a0a456ee9d67b16a1dc61785562bee90c9`.
The E2B-base-to-E4B validation contract is the deterministic clean 128-question subset recorded
below, with one trace per question. The opposite distillation line, production benchmark matrix, and
post-distillation RL remain unstarted and unauthorized.

Last updated: 2026-07-31 UTC.

This document supersedes stale Gemma 4 run-status notes in `PROGRESS_LOG.md`. It combines the
scientific motivation, current artifacts, experimental design, data and evaluation contracts,
implementation plan, and unresolved decisions for the Gemma 4 RL-versus-distillation project.

## Executive summary

The project tests the claim in DeepSeek-R1 Section 4.1 that reinforcement learning on a larger
base model followed by distillation into a smaller model can outperform direct RL on the smaller
model. It also asks when that claim holds, why it holds, and when it fails.

There are two experimental lines:

1. Compare direct RL on Gemma 4 E2B against off-policy distillation of a Gemma 4 E4B RL teacher
   into Gemma 4 E2B base.
2. Distill Gemma 4 E2B base into Gemma 4 E4B base, test whether the resulting E4B behaves like
   E2B despite its different pretraining and larger parameter count, and then run RL from that
   distilled E4B initialization.

The trace payload design is fixed and complete: five responses per selected train row and validation
UID, with the teacher's vLLM top-128 token IDs and full-vocabulary-normalized log probabilities at
every response position. The bundles also store exact prompt, response, and combined token IDs plus
the response mask. Distillation must consume those stored IDs directly and must never re-tokenize a
trace. Before primary training, the exact sequences are intended to be teacher-forced through an
unsharded HF BF16+SDPA path, using the model class selected by verl, into a separate top-128 overlay;
the original vLLM targets remain immutable for
provenance and engine-difference analysis. The immutable artifacts preserve the historical 9,723-row
training roster and all 200 validation UIDs, including seven overlapping question texts. For the
E2B-base-to-E4B run, reuse the complete training split and use one response from each of 128
deterministically selected, unique, train-disjoint validation prompts. The full 200 validation UIDs
and clean 193-question roster remain provenance/evaluation pools rather than this run's in-loop
validation payload.

The framework-critical pieces now run end to end at smoke scale. Gemma 4 verl completes rollout,
strict reward, FSDP2 update, vLLM weight sync, checkpoint save, and checkpoint resume. The former
Gemma-3-only distillation fast path now supports Gemma 4 hidden-state projection, final-logit
softcapping, stored top-k targets, token-weighted validation, activation-checkpointed vocabulary
chunks, and lazy Parquet row-group ingestion. The resumable trace producer, validator/indexer,
identity-bound source-bundle preflight, 64-sample math evaluator, and pinned OOD wrapper are implemented.
An E4B-generated trace has passed the complete path into a real E2B student update.

Preparation is therefore at complete-generation and end-to-end gate status for both directions, not
experiment status. The source+overlay preflight and launcher routing are implemented and CPU-tested;
the remaining execution boundary is the separately authorized unsharded-HF overlay rescore and artifact
review. The later 750-step runs still require decisions about the exact top-k objective, metric
conventions, model-checkpoint destinations, and fairness axes. The source
train file has 9,723 rows but only 9,543 distinct question texts, and seven of the 200 validation
question texts also occur in train. The bundles deliberately preserve that historical roster for
exact RL comparability; the overlap hashes and clean-193 evaluation requirement must remain visible
in every derived index and report.

## Scientific motivation

DeepSeek-R1 Section 4.1 compares two routes to a strong smaller reasoning model:

- direct large-scale RL on the smaller/base model; and
- RL on a stronger model followed by distillation into the smaller model.

Their result was that the distilled Qwen-32B model substantially outperformed a Qwen-32B model
trained directly with large-scale RL, despite the large compute spent on the direct-RL route. This
project investigates whether that result generalizes to the Gemma 4 E2B/E4B family and attempts to
separate model scale, pretraining differences, RL optimization, and supervision density.

### Research questions

1. Does E4B RL followed by E4B-to-E2B distillation outperform direct E2B RL?
2. If so, where does the advantage come from?
3. Is the advantage consistent across in-distribution math, out-of-distribution math, and
   non-math capabilities?
4. At what model sizes, problem difficulties, data budgets, and compute budgets does the claim
   hold or fail?
5. How much of the E4B advantage is caused by parameter count versus different or additional
   pretraining?
6. Can an E4B model first distilled from E2B retain E2B-like behavior while gaining a larger
   parameterization that makes subsequent RL easier?

### Hypotheses

#### H1: mode availability and discoverability

Larger pretrained models may contain useful modes that smaller models cannot discover through
their own sampled RL trajectories. RL may primarily reweight or collapse modes already present
after pretraining rather than inventing entirely new modes. A larger model can therefore find a
solution mode that the smaller model rarely or never samples, while the smaller model may still be
able to represent that mode once dense distillation targets expose it.

Prediction: E4B should solve some questions for which E2B base has pass@64 near zero. Distillation
should improve E2B most on those questions if the E2B has enough capacity to imitate the E4B trace.

#### H2: model-size-dependent RL optimization difficulty

RL may be less stable and less sample-efficient at smaller scales, while supervised token-level
distillation remains comparatively well behaved. Larger models may converge faster, collapse less,
and tolerate sparse/noisy rewards better. Dense token-level targets then transfer the optimized
behavior into a smaller model through an easier objective.

Prediction: E4B RL should have better learning speed or stability than E2B RL, while E2B
distillation should show smooth held-out top-k loss and fewer optimization pathologies than E2B RL.

#### H3: sparse-reward learnability frontier

RL only supplies useful within-group advantage on questions that a model solves sometimes but not
always. Questions it never solves produce no positive trajectory and little or no useful learning
signal. The set of questions with non-degenerate rewards should grow with model size. Distillation
can provide the smaller model with dense targets on questions where its own RL sampler never finds a
successful response.

Prediction: questions with E2B-base pass@64 = 0 but E4B-base or E4B-RL pass@64 > 0 should be a key
stratum. Improvement on that stratum after distillation directly tests this hypothesis.

### Important limitation of line 2

Distilling E2B into E4B only on this strict-4/4 math split cannot erase or fully control all effects of
E4B pretraining. Under the current row-preserving plan it covers 9,723 train source rows plus 200
validation UIDs, but only 9,736 distinct question texts across the combined split because train has
duplicate texts and seven validation texts leak into train. It only constrains E4B on the prompt and
trajectory states covered by that math data. If the intended claim is that the distilled E4B should
behave like E2B globally, the distillation corpus needs broad non-math prompts as well. The current
requested design is enough to test E2B-like behavior on the math distribution and to measure
spillover on non-math benchmarks, but it is not equivalent to equalizing the two models' pretraining
histories.

## Confirmed current state

### Repository and framework status

- Repository: `rl-distill`, branch `main`; the pushed preparation baseline is `d3906463`. The final
  trace-evidence, cross-engine audit, rescorer, and documentation update is reviewed and pushed as a
  separate preparation-only change; it does not authorize an experiment.
- `3ed6f620` fixes the known verl Gemma 4 non-remove-padding attention-mask bug in the FSDP,
  AutoModel, and TorchTitan paths. `tests/workers/utils/test_padding.py` contains a targeted
  regression for the at-cap response case.
- `e7413889` adds PPO probability-ratio diagnostics; it is observability, not another correctness
  fix.
- A fresh two-H100 verl Gemma 4 E2B gate exercised rollout, strict reward, FSDP2 update, vLLM
  weight synchronization, and checkpointing for steps 1 and 2. Probability-ratio means were
  `1.00000002` and `1.000299`; finite gradient norms were `11.86` and `9.09`.
- A separate auto-resume gate loaded step 2 model, optimizer, RNG, scheduler, and dataloader state,
  continued at `2/3`, completed step 3 with ratio mean `1.000165` and gradient norm `10.4763`, and
  saved a complete step-3 checkpoint. The dataloader advanced from 8 to 12 yielded samples rather
  than restarting.
- The RL smoke used a 512-token response cap. Fifteen of sixteen resume-gate responses stopped
  before the cap and one reached it. The configured Gemma stop strings were active, but the current
  logs do not identify the exact matched string per request. This remains the verified RL rollout
  cap; the separate trace generator has passed its full 8,192-token boundary contract.
- `actor_rollout_ref.rollout.gpu_memory_utilization=0.25` passed on the two-H100 colocated gate;
  `0.45` OOMed in the same setup.
- The top-k distillation path now supports Gemma 4 hidden-state selection and reapplies the true
  `final_logit_softcapping` value before student log-probabilities. A real-weight comparison matched
  the native Gemma 4 logits exactly (`max_abs=0`) and backward passed.
- Precomputed top-k ingestion, lazy Parquet row-group loading, and activation-checkpointed vocabulary
  chunks passed CPU tests and real two-rank FSDP2 updates. A synthetic full
  8,192-response-token gate used FP32 master parameters with BF16 compute, processed 16,384 active
  response tokens, produced finite loss/gradient values, ran step-zero and step-one validation, and
  saved a resumable checkpoint.
- Ragged hidden-state selection now supports multiple sequences per microbatch. An eight-H100 E4B
  gate completed three optimizer steps at microbatch size 2 and KL chunk size 4,096 using the full
  4,096-prompt-plus-8,192-response contract, then passed a smaller final validation microbatch and
  saved model, optimizer, dataloader, and directly loadable HF state. Peak device memory was 65,394
  MiB and steady-state step time averaged 109.45 seconds for 786,432 global tokens.
- A live E4B-RL step-100 trace was generated through vLLM 0.25.1 with exact rank-1-through-128
  full-vocabulary-normalized log probabilities and stored token IDs. That trace then passed a real
  two-rank E2B update, step-zero/step-one validation, checkpoint save, resume, and serving check.
  E2B-base traces similarly passed a real E4B update/save/serve gate. Independent 8,192-cap live
  generation gates passed for both teachers.
- Both production-scale local trace bundles are complete and indexed. Each contains 48,615 train
  rows and 1,000 validation rows across 1,216 train and 25 validation shards. The E4B-RL teacher
  bundle contains 14,049,865 response tokens; the E2B-base teacher bundle contains 8,257,057.
- Expanded vLLM-versus-unsharded-HF diagnostics covered 32 traces per teacher and 1,870 E2B or 1,975 E4B
  response positions. The two unsharded-HF paths, native Gemma 4 forward and manual projection, matched exactly,
  while vLLM/HF top-128 overlap was 0.98504 for E2B and 0.97654 for E4B. The calibrated diagnostic
  thresholds pass, but the nonzero log-probability differences rule out treating the engines as
  bit-identical.
- `data/rescore_gemma4_training_topk.py` now prepares an immutable, one-to-one unsharded-HF target overlay
  without editing the vLLM source bundle. It binds the source index, shard and manifest hashes,
  trace-ID sets, exact teacher identity, causal shift, Gemma 4 softcap, BF16+SDPA contract, and native
  forward parity. This tooling is CPU-tested only; no production rescore has been launched.
- Deterministic trace-contract failures now return a distinct generator status, and the supervisor
  refuses identical-seed retries for that status instead of looping on an unrecoverable row.
- The validation loader retains a final partial batch only when exact distillation coverage is
  requested and computes token-weighted aggregates. Generic SFT retains its historical
  `drop_last=True` behavior.
- Clean repository HEAD also cannot reproduce the E4B NeMoRL run without additional work: the
  supervisor expects local NeMo patches that are stored in the private E4B HF repository but are
  not applied by a committed script. Provenance and patch application must be repaired before the
  post-distillation E4B RL phase.
- The private E4B repository's `repro/manifest.json` refers to an older v1 run rather than
  `6e5192d8`; the reconstructed teacher must receive a new, verified provenance manifest.
- The initial private E2B-teacher upload reached the Hub only after full local validation and failed
  closed with `Private repository storage limit reached`; the corresponding early E4B upload-result
  log was empty. After explicit public-upload authorization, both complete source bundles were
  published successfully and independently verified at immutable commits.

### Verified implementation status

| Gate | Result | Durable evidence |
|---|---|---|
| verl Gemma 4 RL steps 1-2 | pass | `/tmp/verl_gemma4_gate_phase2.log`; `/tmp/verl-gemma4-e2e-gate2/global_step_2` |
| verl checkpoint resume to step 3 | pass | `/tmp/verl_gemma4_gate_resume3.log`; `/tmp/verl-gemma4-e2e-gate2/global_step_3` |
| E4B step-100 raw chained-delta reconstruction | pass | `/tmp/verl/models/nemorl-gemma4-e4b-step100`; raw chain SHA256 `d565a3ff371906ca31a5e355472d70366b6956c0e82a914de4ea8a7c0085630c` |
| E4B vLLM-ready packaging | pass | `/tmp/verl/models/nemorl-gemma4-e4b-step100-vllm`; pinned `processor_config.json`; 54 deterministic shared-KV aliases; 2,130 tensors; SHA256 `830d47f78008b56787798a21a5e53d4e402a405bb899a9db5e18b7b83371110f` |
| Raw-versus-expanded Transformers parity | pass | bit-identical logits (`max_abs=0`) |
| Expanded E4B vLLM load/generation | pass | vLLM 0.25.1; `/tmp/vllm_e4b_expanded_gate.log` |
| Gemma 4 native-vs-distill projection | pass | exact logits (`max_abs=0`) plus backward |
| Precomputed top-128 two-rank FSDP2 update | pass | `/tmp/gemma4_precomputed_topk_checkpointed.log`; `/tmp/gemma4-precomputed-topk-distill-checkpointed/global_step_1` |
| Partial final validation batch | pass | two validation rows on two ranks with global batch four; step-zero and step-one validation both ran |
| Synthetic 8,192-response-token FP32-master/BF16 update | pass | 16,384 active response tokens; `/tmp/gemma4_precomputed_topk_8k_fp32.log`; `/tmp/gemma4-precomputed-topk-8k-fp32-gate/global_step_1` |
| Live E4B top-128 trace generation | pass at 32-token smoke cap | `/tmp/gemma4_e4b_live_trace_smoke_v4.log`; `/tmp/gemma4-e4b-live-trace-smoke-v4` |
| Live E4B trace to E2B update | pass | finite loss `0.2897999`, gradient norm `42.8811`, validation, and save; `/tmp/gemma4_live_trace_distill_gate.log`; `/tmp/gemma4-live-trace-distill-gate/global_step_1` |
| Live E2B-base trace generation | pass at five-response smoke scale | `/tmp/gemma4-e2b-base-live-trace-smoke`; exact top-128 bundles validated |
| Live E2B trace to E4B update/save | pass | loss `0.2406794`, finite update, validation, 54-alias HF save; `/tmp/gemma4_e2b_trace_to_e4b_student_gate.log` |
| Distillation checkpoint resume | pass | loaded step 1 model/optimizer/scheduler/RNG/data state and saved step 2; `/tmp/gemma4_distill_resume_gate.log` |
| E2B/E4B saved-checkpoint serving | pass | `/tmp/vllm_e2b_saved_checkpoint_gate.log`; `/tmp/vllm_e4b_student_saved_checkpoint_gate.log` |
| E4B live generation at 8,192 cap | pass | 149 response tokens, natural `<end_of_turn>` stop, exact top-128; `/tmp/gemma4_e4b_trace_cap8192_gate.log` |
| E2B live generation at 8,192 cap | pass | 40 response tokens, natural `<end_of_turn>` stop, exact top-128; `/tmp/gemma4_e2b_trace_cap8192_gate.log` |
| Complete E4B-RL-to-E2B source bundle | pass locally | 49,615 rows, 14,049,865 response tokens; index `8b5712e0f5dea3388340a9bc91a6ceee40ff2ff990e66b7238090e240daeda6c` |
| Complete E2B-base-to-E4B source bundle | pass locally | 49,615 rows, 8,257,057 response tokens; index `d170166abad89588880f5c0a9eac43006f9a32bbe27cec28d7eb97c65288dbcc` |
| Expanded E2B vLLM-versus-unsharded-HF diagnostic | pass calibrated thresholds | 32 traces / 1,870 positions; native projection max abs 0; top-128 overlap 0.98504; `/tmp/gemma4-e2b-vllm-vs-verl-parity-expanded.json` |
| Expanded E4B vLLM-versus-unsharded-HF diagnostic | pass calibrated thresholds | 32 traces / 1,975 positions; native projection max abs 0; top-128 overlap 0.97654; `/tmp/gemma4-e4b-vllm-vs-verl-parity-expanded.json` |
| Unsharded-HF target-overlay rescorer | prepared, not run | focused CPU tests and exact chunked/native fixture parity; production GPU rescore requires separate authorization; real FSDP2 audit remains pending |
| Public source-trace publication | pass and independently verified | E2B commit `e32aaa02681ae83b3d7256b1b155c9084da2f289`; E4B commit `2b6e49a0a456ee9d67b16a1dc61785562bee90c9`; 2,485 registered files each; every path/size and content identity verified remotely |

Expanded cross-engine results used to calibrate the committed audit are:

| Teacher | Traces / positions | Native vs manual max abs | Tie-safe top-1 | Top-10 overlap | Top-128 overlap | Weighted abs logprob delta mean | Probability L1 mean | Sampled-token delta p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E2B base | 32 / 1,870 | 0 | 0.99572 | 0.98294 | 0.98504 | 0.017996 | 0.017343 | 0.09419 |
| E4B RL step 100 | 32 / 1,975 | 0 | 0.99747 | 0.97494 | 0.97654 | 0.010180 | 0.010221 | 0.06082 |

These reports establish a controlled numeric difference, not a model-quality result. They do not
authorize replacing the source bundle in place, rescoring all rows, or starting distillation.

The exact E2B saved-checkpoint serving gate used training/save log
`/tmp/gemma4_checkpoint_serving_gate.log`, artifact
`/tmp/gemma4-checkpoint-serving-gate/global_step_1/huggingface`, and serving log
`/tmp/vllm_e2b_saved_checkpoint_gate.log`. Its `processor_config.json` SHA256 is
`32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c`; the packaged model has
2,012 tensors, including 60 shared-KV aliases, and an 11,051,930,326-byte `model.safetensors`.
vLLM 0.25.1 loaded that artifact and generated successfully.

The E4B materializer is implemented as a lock-driven utility at
`nemo_rl_repro/local/materialize_hf_checkpoint_chain.py`. It pins the base revision, checkpoint
repository revision, every delta link, target byte length, and target SHA256; it refuses overwrite,
validates the full chain, copies pinned processor metadata, deterministically expands the 54
shared-KV aliases required by vLLM, writes a directly HF/vLLM-loadable `model.safetensors`, and emits
a provenance manifest. The expanded artifact is local only and must not be uploaded without explicit
authorization. The current lock uses `google/gemma-4-E4B@411aa17b749aa952df1359d2dcea73917a544d9a`
and checkpoint repository revision `7c8d8cd4642465f418aa101a2e884cd5607ba8f1`.
The materializer imports the local sparse-delta codec and therefore requires the `zstandard`
package; the verified environment used `zstandard==0.25.0`. Install it with `uv` in the environment
that runs the materializer before validation or reconstruction.

### Upstream verl integration decision

Do **not** rebase this dirty, verified fork directly onto current upstream `main`. Rechecked against
the official refs on 2026-07-31, upstream `main` is `05466835` and `release/v0.8.0` is `bee9f6f4`;
the fork and official main have diverged by 36 fork-only and 488 upstream-only commits, with 13
practical conflict files and a Transformers-version migration. Preserve this exact verified branch,
then create a separate worktree from upstream `release/v0.8.0` and forward-port the local concerns
with independent gates. The v0.8 line is 36 fork-only and 285 upstream-only commits apart, has fewer
conflicts, and is the safer first integration target;
move to a later 0.9/main baseline only after the v0.8 port is green.

The forward port must preserve and retest Gemma 4 softcap relocation/HF restoration, padded
full-sequence attention masks, hidden-state response selection, memory-bounded top-k backward,
exact stored-token flow, stop handling, checkpoint/HF hooks, validation coverage, PPO ratio health
metrics, and the Gemma-3-MoE vLLM compatibility code. Do not resurrect legacy worker files removed
upstream; port behavior into the current engine paths.

### Existing Gemma 4 RL runs

Both runs produced useful, learning checkpoints. Neither completed its originally intended horizon,
and both W&B runs ended in a crashed state.

| Model | W&B run | Training observed | Latest durable checkpoint | Validation status | Artifact status |
|---|---|---:|---:|---|---|
| Gemma 4 E2B PT RL | `rl-distill/DAPO/e1du1oyu` | through step 149 | step 125 | strict validation accuracy 18.44% at step 140 | `JWei05/nemorl-dapo-gemma4-e2b-pt-rl-step-125`; public, BF16, directly HF-loadable |
| Gemma 4 E4B PT RL | `rl-distill/DAPO/6e5192d8` | through step 120 | step 100 | 33.47% at step 100; 32.72% at step 120 | `JWei05/nemorl-dapo-gemma4-e4b-pt-500step/checkpoints/step_100`; private chained delta, not directly HF-loadable |

Additional caveats:

- The E2B step-125 checkpoint consumed 8,000 training prompts. There is no step-150 save.
- The E4B repository name contains `500step`, but the run did not reach 500 steps.
- E4B step 120 showed a raw probability-ratio anomaly (`1.0716` mean and a very large maximum).
  It likely triggered the supervisor, but the termination cause is not proven. Step 100 predates
  that anomaly and is the selected teacher checkpoint.
- E4B step 100 has been reconstructed in order from the pinned E4B base and deltas at steps
  20, 40, 60, 80, and 100. The raw reconstruction matches the locked chain SHA256. A second,
  vLLM-ready package adds the pinned processor metadata and 54 shared-KV alias tensors that the raw
  NeMo/Transformers export omitted. The expansion has its own locked SHA256, is bit-identical to the
  raw model under Transformers, and loads/generates in vLLM 0.25.1.
- The expanded E4B checkpoint has not been uploaded. A normal private HF snapshot would simplify
  distributed generation, but publishing it is an external artifact mutation and requires explicit
  user authorization plus a final immutable provenance review.

### Training and validation data

The Gemma 4 runs use the DeepScaleR strict-4/4 split, not the older DAPO-Math-17k split described in
parts of `FEWSHOT_MATH_RL.md`. The current files were audited by extracting the final user message
exactly as the trace generator does.

| Split | File | Stored rows | Generation units under current code | Distinct question texts | Use here |
|---|---|---:|---:|---:|---|
| Source pool | `deepscaler_acc4of4_strict.parquet` | 9,923 recorded historically | not re-audited locally | not re-audited locally | provenance only |
| Train | `deepscaler_4of4strict_rl_train.parquet` | 9,723 | 9,723 row-level units | 9,543 | row-faithful trace generation and distillation train |
| Validation source | `deepscaler_4of4strict_rl_val200_x16.parquet` | 3,200 | 200 UID-deduplicated units | 200 | provenance/evaluation pool |
| E2B-to-E4B validation | `deepscaler_4of4strict_rl_val128_clean_seed42.parquet` | 128 | 128 UID-deduplicated units | 128 | one-trace in-loop validation |

The split was shuffled with seed 42. The source validation parquet repeats each question 16 times for
the legacy RL validator. The training parquet has no UID, so the implemented generator gives every
source row a deterministic row-derived UID and preserves all 9,723 rows. That preserves the historical
schedule but means 180 repeated question-text rows receive duplicate trace multiplicity.

The audit also found **seven validation question texts in train**. Each leaked text occurs once in
train, so dropping all leaked train rows would leave 9,716 train rows and 9,536 distinct train texts.
Across the current unmodified train and validation files there are only 9,736 distinct question texts,
not 9,923. The trace-artifact policy is to preserve the exact historical files because the existing
RL checkpoints trained on that roster. The dataset index records the seven overlap hashes, and both
preflight and upload reject the dataset unless `--allow-question-overlap` is explicitly supplied.
The E2B-to-E4B production bundle avoids the exception: it excludes those seven prompts, samples 128
of the 193 clean validation UIDs with Python's seeded sampler at seed 42, and then restores source
validation order. Its Parquet SHA256 is
`934f53eebf08775899b9705628324f1cd3e4c17dc6e9064774b1ee49587dbb99`, its selection-file SHA256 is
`9443e38436669498e1404e3cc20142ef8c54024cebcc943939f9f42f0c56c976`, and its ordered roster SHA256
is `173b2da5af12e12d264fa0346b8641b058b04078df6621868078f9d1c7fca921`.

Each immutable source bundle contains five traces per row-level train unit and five per validation
UID. The E2B-to-E4B derivative reuses all source training traces and selects one trace per clean
validation UID:

| Split | Generation units | Distinct texts | Traces/unit | Distillation rows |
|---|---:|---:|---:|---:|
| Train | 9,723 | 9,543 | 5 | 48,615 |
| Validation | 128 | 128 | 1 | 128 |
| Total | 9,851 | 9,671 after cross-split deduplication | split-specific | 48,743 |

The exact immutable bundle registrations are:

| Direction | Local root | Dataset index SHA256 | Experiment SHA256 | Response tokens | Approx. local size |
|---|---|---|---|---:|---:|
| E4B-RL step 100 to E2B | `/lambda/nfs/Jason-scale/rl-distill-traces/gemma4-e4b-rl100-topk128` | `8b5712e0f5dea3388340a9bc91a6ceee40ff2ff990e66b7238090e240daeda6c` | `b515965fcdc31caebae1e71cf696731f4271182bdaca8af8326888d0b196af92` | 14,049,865 | 6.0 GiB |
| E2B base to E4B | `/lambda/nfs/Jason-scale/rl-distill-traces/gemma4-e2b-base-topk128` | `d170166abad89588880f5c0a9eac43006f9a32bbe27cec28d7eb97c65288dbcc` | `f77e9dd611371030cc7752d4f4d0c92d4890448d9fe48594efdf3e223a5a409e` | 8,257,057 | 3.7 GiB |

Both indexes register 1,216 train shards and 25 validation shards, tokenizer SHA256
`f3ab24e73e9022f7b8d77113f543debd3779ef1e96c6452c68aaa9f3e6b81d17`, template SHA256
`27b8801d8b61a413a9bb3b54b6f55e16217eff3e55f7c560377c8a162dd63c1c`, and the same seven
cross-split overlaps. E4B-RL generation-config SHA256 values are
`1806394beb727900f89757453972aa15c6dc6ef7f74e70c3a3cb8a06087072a0` (train) and
`8369ef7a4bb65138ee4fa261f071cb79eba30c942a77e6df6154e2261d9eb1ce` (validation). E2B-base
values are `a2c1aacb4166472a6da7c0af223270b0feb289c1d612b08ea5c090184687d75c` (train) and
`b60835b8eaba9c1dc3e4d3a0f7cefd0046140ab5e652c8217ef18c1e6678770c` (validation).
The E4B teacher content/model/registered-identity hashes are respectively
`830d47f78008b56787798a21a5e53d4e402a405bb899a9db5e18b7b83371110f`,
`b9d90fb76e033d610c93dc36fc60b50ad40029c9f5af567abc316822949b7f08`, and
`46c469dc4e59ffad57d8889c1b6f0a7ce822192610819197e615a720dc591bf3`. The E2B values are
`76dc84a5a805a2c8b91e9ccc00b8dbf8f4a99bf0d56ab25832f6e6addd4f7f57`,
`bde9e800223cdd62228ce39e0305398f6ada05b98adaf438b0b3d3d3c3015561`, and
`2d48d343709dcae087d6ff2def9f09d2950ca66dc2183a8bee38850c4ddbbb36`.

At global batch size 128, 750 optimizer steps consume 96,000 sequence examples across almost two
passes over the 48,615-row training trace set. With eight data-parallel ranks, the drop-last sampler
and loader produce 379 optimizer steps per epoch, so two epochs expose 758 available steps and the
750-step cap stops eight updates before the end of epoch two. The exact trace IDs skipped by the two
independently shuffled epoch tails must be recorded after the run. Possible future clean-roster or
text-deduplicated ablations need separate coverage calculations and run identities.

The validator, production preflight, dataset supervisor, and uploader accept explicit split-specific
contracts while retaining 9,723 x 5 train and 200 x 5 validation as backward-compatible defaults. The
authorized E2B-to-E4B contract is exactly 9,723 x 5 train and 128 x 1 validation. A different roster
must use a separately named dataset and new immutable roster hash; it must never weaken these checks.

### Prompt, tokenization, sampling, and reward contract

- The exact math prompt is `data/gemma3_it_fewshot_math.jinja`. Despite the filename, both Gemma 4
  NeMoRL configurations use it.
- It is a 12-shot multi-turn prompt containing four Minerva/MATH and eight GSM8K exemplars,
  interleaved and normalized to end in `\boxed{}`.
- The actual question is the final user turn and contains the instruction
  `Please output the final answer within \boxed{}.`
- Render with `add_generation_prompt=True` and encode with `add_special_tokens=False`; the template
  already emits the single BOS.
- RL used a 4,096-token prompt cap, an 8,192-token response cap, and a 12,288-token total cap.
- A full tokenizer audit found train prompt lengths of 1,540-2,518 tokens and validation prompt
  lengths of 1,541-1,866 tokens. No selected prompt reaches the 4,096-token limit.
- RL sampling used temperature 1.0, top-p 1.0, and no sampling top-k.
- Gemma 4 must stop on the strings `<end_of_turn>` and `<start_of_turn>`.
- Math correctness uses strict `math_verify`: exactly one boxed/fboxed final answer; zero or multiple
  boxes score zero.

`top-k=128` in this project refers to the **logged teacher distribution used for distillation**. It
does not mean sampling from only 128 tokens; generation sampling remains unrestricted unless an open
decision below changes it.

## Experiment matrix

### Line 1: direct E2B RL versus E4B-RL-to-E2B distillation

Required primary models:

| Label | Initialization / checkpoint | Role |
|---|---|---|
| `e2b_base` | `google/gemma-4-E2B@d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` | direct baseline and distillation student initialization |
| `e2b_rl_125` | `JWei05/nemorl-dapo-gemma4-e2b-pt-rl-step-125` | direct-small-model RL result |
| `e4b_rl_100` | reconstructed E4B RL step 100 | teacher and upper reference |
| `e4b_rl100_to_e2b_250` | E2B base distilled for 250 steps | learning-curve checkpoint |
| `e4b_rl100_to_e2b_500` | E2B base distilled for 500 steps | learning-curve checkpoint |
| `e4b_rl100_to_e2b_750` | E2B base distilled for 750 steps | requested final distilled model |

Primary question: under the selected evaluation and fairness rule, is `e4b_rl100_to_e2b_*`
stronger than `e2b_rl_125`?

### Line 2: E2B-base-to-E4B distillation followed by RL

Required models:

| Label | Initialization / checkpoint | Role |
|---|---|---|
| `e2b_base` | `google/gemma-4-E2B@d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` | teacher and target behavior |
| `e4b_base` | `google/gemma-4-E4B@411aa17b749aa952df1359d2dcea73917a544d9a` | undistilled larger-model baseline |
| `e2b_to_e4b_250` | E4B base distilled from E2B base for 250 steps | learning-curve checkpoint |
| `e2b_to_e4b_500` | same at 500 | learning-curve checkpoint |
| `e2b_to_e4b_750` | same at 750 | requested final pre-RL initialization |
| `e2b_to_e4b_750_rl_*` | post-distillation E4B RL checkpoints | test of RL after behavioral initialization |

The pre-RL comparison asks whether `e2b_to_e4b_*` matches E2B base on the strict validation split,
math OOD sets, entropy/diversity measures, and non-math benchmarks. The post-RL comparison asks
whether the E2B-distilled E4B learns differently from normal E4B RL and whether it reduces the
advantage attributable to E4B's pretraining rather than its parameter count.

## Shared trace-generation contract

### Fixed design: immutable vLLM source plus precomputed unsharded-HF target overlay

For both directions, completed generation requested 128 teacher log probabilities from vLLM at
every generated response position and stored:

- the exact 128 teacher vocabulary IDs by rank;
- their full-vocabulary-normalized log probabilities;
- the sampled token ID and sampled-token log probability separately;
- all exact prompt, response, and concatenated token IDs; and
- the response mask used for training.

Use `int32` for vocabulary IDs and `float16` for stored top-k log probabilities, then cast to float32
for loss computation. Store normalized log probabilities rather than raw logits so both teacher
top-k mass and the existing partial-KL objective remain recoverable. If a future objective
renormalizes over the top-k support, it can renormalize these stored values.

Approximate uncompressed payload for top-k IDs plus FP16 log probabilities is 768 bytes per response
token. Under the current row-preserving 49,615-trace roster this is roughly 9.5 GB at 250 average
response tokens, 19 GB at 500, 38 GB at 1,000, and 312 GB in the pathological all-8,192-token case.
Files must therefore be sharded, written incrementally, and streamed or memory-mapped during
training; the full dataset must not be loaded into RAM as one pandas DataFrame.

vLLM may return the sampled token in addition to the requested top-128 when it falls outside the
top-128. The stored `teacher_topk_*` arrays contain exactly ranks 1 through 128; the sampled token is
retained separately and must not silently become a 129th distillation target.

The vLLM arrays are the immutable generation-engine record, but the primary distillation targets are
intended to come from a separately indexed unsharded-HF overlay over those exact token sequences. This
choice follows the expanded cross-engine audit: native HF forward and the manual distillation
projection agree exactly, but vLLM and HF differ slightly in top-k membership and log probabilities.
The overlay must:

- preserve the exact source `input_ids`, `response_token_ids`, response mask, trace order, and split;
- never re-tokenize or regenerate a response;
- teacher-force each full stored sequence through the exact local teacher using BF16, SDPA,
  `use_cache=False`, the causal predecessor shift, Gemma 4 final-logit softcapping, and FP32
  full-vocabulary normalization;
- store top-128 IDs/log probabilities and sampled-token log probabilities in a disjoint output tree;
- bind every output shard to the source dataset index, source Parquet and manifest hashes, source
  trace-ID set, target model identity, rescorer source hash, and environment; and
- pass exact chunked-projection-versus-native-forward parity before bulk scoring.

This remains a precompute design. The student trainer must not load an online teacher. The source
vLLM bundle must never be overwritten; engine comparisons and any alternative target policy remain
reproducible from the two separately indexed layers.

### Required trace schema

Every row must include at least:

- schema version and globally unique trace ID;
- experiment direction (`e4b_rl100_to_e2b` or `e2b_base_to_e4b`);
- split (`train` or `validation`), source dataset, source UID, prompt index, and sample index 0-4;
- question text, gold answer, and strict grade/prediction for diagnostics;
- teacher repository/path, immutable revision or content hash, tokenizer revision, and tokenizer
  vocabulary/config hash;
- chat-template path and SHA256;
- sampling seed and all sampling parameters;
- `prompt_token_ids`;
- `response_token_ids`;
- `input_ids = prompt_token_ids + response_token_ids`;
- `response_mask = [0] * prompt_length + [1] * response_length`;
- `teacher_topk_token_ids`, shape `[response_length, 128]`;
- `teacher_topk_logprobs`, shape `[response_length, 128]`;
- sampled-token log probabilities;
- prompt length, response length, finish reason, vLLM stop reason, and whether the 8,192-token cap
  was reached;
- shard ID, row-within-shard, generation timestamp, generator commit, and environment versions.

### Integrity requirements

Before a shard is accepted:

1. `len(input_ids) == len(response_mask)`.
2. `input_ids == prompt_token_ids + response_token_ids`.
3. The response mask contains exactly `prompt_length` zeros followed by `response_length` ones.
4. Both top-k tensors have exactly one 128-wide row per response token.
5. Top-k vocabulary IDs are in range and unique within each token position.
6. Log probabilities are finite and sorted by teacher rank.
7. The sampled-token log probability agrees with the top-k value whenever the sampled token is in
   the top 128.
8. Re-decoding the stored response IDs matches the stored response text under an explicitly defined
   normalization.
9. Train and validation UIDs are disjoint, the E2B-to-E4B validation split contains exactly 128 UIDs
   x 1 sample, and train/validation question-text hashes do not overlap.
10. The manifest records row counts, token counts, truncation counts, shard hashes, and distribution
    quantiles for response length and teacher top-k mass.

Production-length live gates settled the stop behavior. With
`include_stop_str_in_output=false` and `skip_special_tokens=false`, vLLM omits `<end_of_turn>` from
`completion.text` but retains its exact seven sampled token IDs in `completion.token_ids`. The trace
therefore preserves those IDs in `response_token_ids` and gives them response-mask value 1. This is
intentional: the request is to retain every sampled ID and the RL rollout trains sampled terminal
tokens. `response_text` remains the exact decode of stored IDs, while `vllm_response_text` separately
records vLLM's stop-string-omitted text for audit. Both E4B and E2B naturally stopped this way in the
8,192-cap gate and their bundles passed full validation.

Generation was resumable: it processed bounded chunks, atomically finalized each shard, and skipped
only shards whose manifest/hash checks passed. Deterministic trace-validation failures now stop the
supervisor immediately rather than retrying the same seed. The unsharded-HF overlay scorer follows the
same atomic, hash-validated, resumable-shard rule when separately authorized.

## Off-policy top-k distillation contract

### Fixed training hyperparameters

Unless superseded by an explicit decision below, both directions share:

| Setting | Value |
|---|---|
| Student initialization | base E2B for line 1; base E4B for line 2 |
| Teacher targets | precomputed unsharded-HF top-128 overlay bound to the immutable vLLM trace bundle; real FSDP2 equivalence still requires an engine audit |
| Global sequence batch | 64, pending confirmation |
| Optimizer steps | 750 |
| Shuffle | yes |
| Optimizer | AdamW |
| Betas | `(0.9, 0.98)` |
| Weight decay | `0.1` |
| Peak learning rate | `5e-6` |
| Warmup | 100 optimizer steps |
| Decay | cosine after warmup |
| Final learning rate | `5e-7` (`0.1` of peak) |
| Validation cadence | every 10 optimizer steps |
| W&B | required |
| Checkpoint/HF cadence | steps 250, 500, and 750 |
| Tokenization | consume stored IDs directly; no re-tokenization |
| Loss mask | response prediction positions only |

The intended LR interpretation is linear warmup to `5e-6` over the first 100 optimizer steps,
followed by cosine decay to exactly `5e-7` at step 750. This interpretation is listed for confirmation
because scheduler endpoints can otherwise differ by one update.

### Gemma 4 correctness requirements

- Use the Gemma-4-capable environment from `setup_env_gemma4.sh` or a pinned equivalent.
- Use `Gemma4TextDecoderLayer`, SDPA, and `use_remove_padding=false` unless a tested replacement is
  introduced.
- Freeze the vision and audio towers and train the language model unless explicitly changed.
- Preserve and correctly apply Gemma 4 `final_logit_softcapping` before computing student
  log-probabilities. The new Gemma 4 hidden-state path does this and must remain covered by exact
  native-logit and HF-save tests.
- Compute the response-position shift correctly: the logits at position `t-1` predict response token
  `input_ids[t]`.
- Require the overlay and source indexes together. Verify the exact source dataset-index SHA256,
  overlay dataset-index SHA256, teacher identity, student identity, and one-to-one trace-ID/order
  binding before the trainer receives shard lists.
- Do not silently truncate stored traces during training. Either fit the full 4,096 + 8,192 contract
  or fail with an actionable error.
- Log teacher top-k mass, student mass on teacher support, loss quantiles, active response-token count,
  gradient norm, learning rate, sequence lengths, throughput, and memory.
- Validation must include every trace in the selected held-out roster: 128 for the authorized
  E2B-to-E4B run. The SFT validation loader retains a final partial batch only for exact distillation
  coverage and computes a token-weighted aggregate. The selected row count is divisible by all eight
  data-parallel ranks, so `DistributedSampler` does not pad with duplicate rows.
- Precomputed hidden-state projection supports ragged microbatches by mapping packed response
  positions back to per-sample padded coordinates. Training batches remain strictly divisible by
  the configured microbatch size; forward-only validation permits a smaller final microbatch while
  preserving any configured force-group boundary. The verified eight-H100 default is microbatch 2.
- Vocabulary projection chunks must use activation checkpointing during training; otherwise each
  chunk's full-vocabulary autograd intermediates remain live until backward and defeat the memory
  bound at 8,192 response tokens.

### Distillation checkpoints and artifacts

Each save must contain:

- a directly HF-loadable student model;
- tokenizer/config/chat-template provenance;
- optimizer and scheduler state for resume;
- trainer state, RNG states, data-shuffle position, and source-manifest hashes;
- the complete resolved configuration;
- W&B run ID and repository commit;
- validation metrics through that step.

Step 750 is both a cadence save and the explicit final save. Upload completion must be awaited before
the job exits.

## Evaluation protocol

### Models to evaluate

At minimum, evaluate all models in the two experiment tables. The open ledger asks whether every
250-step checkpoint receives the complete expensive evaluation or only the final/best checkpoints.

### In-distribution math

- Dataset: the 200 unique strict-4/4 held-out validation questions.
- Never evaluate the pre-repeated 3,200-row parquet as though it contained 3,200 questions.
- Apply the exact 12-shot math prompt and strict boxed-only grader.
- The requested final evaluation draws 64 responses per question.
- Report both the continuity view over all 200 questions and the clean 193-question view. For the
  latter, pass the trace bundle's `dataset_index.json` through
  `eval_math_passk.py --exclude_overlap_hashes_from_index ...`; the evaluator validates and removes
  the seven registered question-text SHA256 overlaps before generation.

### Out-of-distribution math

Follow the prompt/scoring conventions in `FEWSHOT_MATH_RL.md` and
`GEMMA3_PT_EVAL_REPLICATION.md`:

- GSM8K test;
- MATH500;
- AIME 2024;
- AIME 2025;
- AIME 2026;
- OlympiadBench text-only subset;
- MinervaMath.

The exact dataset sources/revisions and whether 64 samples are required for every math set remain
open decisions.

### Out-of-domain/non-math benchmarks

Use full datasets, harness-native raw-completion few-shot prompts, and no 12-shot math/chat template:

| Benchmark | Shots | Reported metric |
|---|---:|---|
| MMLU | 5 | accuracy |
| WinoGrande | 5 | accuracy |
| TriviaQA | 5 | exact match |
| HellaSwag | 10 | normalized accuracy |
| ARC-Challenge | 25 | normalized accuracy |

Never report an ordered `--limit N` subset as the benchmark result. If an OpenAI-compatible endpoint
is used, `tokenized_requests=False` is required to avoid double BOS. Direct vLLM is preferred if it
supports the pinned Gemma 4/harness path.

### Relationship to the Gemma 3 reference protocols

This plan intentionally inherits the parts of `FEWSHOT_MATH_RL.md` and
`GEMMA3_PT_EVAL_REPLICATION.md` that were empirically validated:

- the exact unified 12-shot math template and strict single-box grader;
- sampled math decoding at temperature 1.0, top-p 1.0, and sampling top-k disabled;
- a separately reported greedy@1 diagnostic because RL can improve sampled behavior while degrading
  the argmax path;
- full OOD splits, raw-completion prompts, historical shot counts, and no ordered `--limit` subset;
  and
- single-BOS handling, including `tokenized_requests=False` for an OpenAI-compatible endpoint.

It does **not** inherit the old DAPO-Math-17k train split, the repeat-factor-dependent k values, or
verl's empirical `best@k` as this project's unbiased pass@k. Gemma 4 uses the audited strict-4/4
split, exactly 64 evaluation samples/question unless D42 changes scope, and the estimator specified
below.

### Unbiased pass@k

For each question with `n=64` samples and `c` correct samples, use the user-specified unbiased
without-replacement estimator:

```python
import numpy as np


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def pass_at_k_dataset(ns, cs, k):
    return float(np.mean([pass_at_k(n, c, k) for n, c in zip(ns, cs)]))
```

This is not what verl's empirical `best@k` implementation computes for `k < 64`. The reworked
`eval_math_passk.py` calls `gemma4_eval_metrics.py`, which implements this formula directly and is
tested against the equivalent combinatorial form `1 - C(n-c, k) / C(n, k)`.

### mean@k and maj@k

Mean accuracy has no meaningful k-dependence in expectation unless a subset convention is chosen,
and maj@k depends on subset, answer-equivalence, invalid-output, and tie rules. The new evaluator no
longer hard-codes the legacy raw-string/first-occurrence behavior: it enforces stable sample grouping,
uses semantic math-equivalence classes with a normalized fallback, and exposes full-only, fixed-prefix,
or deterministic Monte Carlo subsets plus plurality/strict-majority choices. Its conservative current
default treats invalid outputs as abstentions and ties as wrong, and records those conventions in the
result metadata.

That implementation flexibility does not resolve the scientific choice. D33-D38 must select the k
set, subset rule, invalid-output behavior, and tie rule before production metrics are reported; the
implementation must be changed and retested if a different D37/D38 policy is selected.

### Entropy and behavior diagnostics

At minimum record response length, finish/stop reason, truncation rate, strict formatting rate,
sample accuracy, and answer diversity. “Entropy” itself remains undefined and is an open decision.
Potentially useful, distinct quantities are:

1. model predictive entropy over the vocabulary at response positions;
2. the same metric approximated by top-k plus a residual-mass bucket;
3. empirical entropy of normalized final answers over 64 samples; and
4. sequence-level diversity metrics.

These quantities answer different questions and must not be reported under one ambiguous label.

### Required secondary greedy evaluation

Prior Gemma 3 experiments showed that sampled and greedy rankings can diverge after RL. A greedy@1
math evaluation is therefore recommended as a secondary diagnostic even if sampled@64 is primary.
It must be reported separately and must not be mixed into the sampled comparison.

## Analysis tied to the hypotheses

In addition to aggregate benchmarks, report question-level paired analyses.

### Difficulty and signal strata

Using the 64-sample base evaluations, partition questions into strata such as:

- E2B never solves, E4B solves at least once;
- both solve sometimes;
- E2B solves sometimes, E4B nearly always solves;
- both never solve; and
- both nearly always solve.

Measure distillation gains within each stratum. The first stratum is the cleanest test of the
sparse-reward/discoverability hypothesis.

### Transfer diagnostics

For each question, retain:

- E2B-base, E4B-base, and E4B-RL correct counts out of 64;
- distilled-student correct counts;
- teacher top-k mass and entropy summaries;
- student loss before and after distillation;
- whether the teacher trace was correct, invalid, duplicated, or truncated; and
- response-length and answer-mode changes.

### Optimization diagnostics

Compare RL and distillation using learning curves rather than final checkpoints alone:

- validation accuracy/loss versus optimizer step;
- examples, response tokens, and estimated FLOPs consumed;
- gradient norm and update stability;
- entropy/diversity collapse;
- truncation and stop rates; and
- wall-clock/GPU-hours.

Because the existing E2B RL and requested distillation runs use different numbers of sampled
sequences and different objectives, “distillation wins” must be qualified by a declared fairness
axis: endpoint quality, examples, generated tokens, wall-clock, or compute.

## Reusable code and required changes

### Reuse directly or extend

- `data/generate_teacher_data.py`: earlier exact-token-ID prompting and sharding patterns; the
  Gemma 4 production path is now `data/generate_gemma4_distill_traces.py`.
- `data/launch_teacher_gen.sh`: multi-GPU generation launcher.
- `data/merge_teacher_shards.py`: merge skeleton; integrity validation must be strengthened.
- `data/audit_gemma4_cross_engine_topk.py`: immutable-index-bound, globally length-stratified
  vLLM-versus-unsharded-HF diagnostic with explicit calibrated thresholds and machine-readable pass/fail
  output.
- `data/rescore_gemma4_training_topk.py`: separate, resumable unsharded-HF target-overlay builder over
  exact stored sequences. It does not mutate the source bundle and requires native-forward parity
  before scoring.
- `full_vocab_distill_dataset.py`: direct consumption of stored IDs/masks and precomputed top-k
  arrays, with shape/range/rank/mass checks and lazy row-group loading so each rank does not
  replicate the entire trace corpus in RAM.
- `main_full_vocab_distill_fsdp2.py`: FSDP2 SFT trainer, deterministic sampling, W&B, validation,
  checkpoints, HF push, exact-width checks, and token-weighted held-out loss.
- `verl/trainer/distillation/fsdp/losses.py`: precomputed top-k partial forward-KL math and
  teacher/student support-mass metrics; reuse only after adapting it to the SFT-style trainer and
  validating Gemma 4 softcapping.
- `hf_push.py`: asynchronous serialized uploads whose final `wait()` is fail-closed; exhausted
  retries, missing checkpoint files, missing Ray owners, and shutdown timeouts propagate as run
  failures instead of being logged and dropped.
- `eval_math_passk.py`: deterministic 64-sample Gemma 4 generation, exact prompt tokenization, stop
  strings, strict grading, bounded eight-request vLLM batches, question-at-a-time semantic grouping,
  streamed audit JSONL, compact in-memory metric rows, and predictive top-k entropy diagnostics.
- `gemma4_eval_metrics.py` and `eval_gemma4_math.py`: unbiased pass@k, mean/majority conventions,
  semantic answer classes, stable grouping, and offline aggregation over saved traces.
- `eval_gemma4_ood.py`: fail-closed wrapper for the five pinned lm-eval tasks and historical shot
  counts, with no ordered `--limit` path.
- `run_eval_matrix.py` and `GEMMA3_PT_EVAL_REPLICATION.md`: OOD task/shot/metric protocol references.

### Do not use as the requested solution

- `main_distill_offpolicy.py` / `distill_dataset.py`: sampled-token CE/KL with re-tokenization, not
  top-128 distribution matching and not drift-safe.
- Online-teacher mode for the production runs: the selected design consumes precomputed targets and
  must not reload or score the teacher inside student training.
- Any older top-k launcher that lacks the Gemma 4 hidden-state patch, explicit softcap restoration,
  checkpointed vocabulary chunks, stored-token ingestion, or exact validation coverage.
- The legacy verl `best@k`/older pass@k logic unchanged: it is empirical “any correct” on the
  available group, not the requested unbiased estimator. Use `gemma4_eval_metrics.py` for this
  project.

## Prepared entry points and runbook status

The preparation uses parameterized entry points rather than separate one-off scripts for each
direction. Their current status is:

1. `nemo_rl_repro/local/materialize_hf_checkpoint_chain.py` — implemented and verified
   - reconstruct and verify the E4B step-100 checkpoint;
   - copy pinned processor metadata and expand shared-KV aliases for vLLM;
   - emit an immutable provenance manifest.
   - publication is intentionally not automatic and is not authorized yet.
2. `data/generate_gemma4_distill_traces.py` — implemented; both complete local bundles produced
   - one generator for both teacher directions and both splits;
   - exact IDs/masks, precomputed top-128, resumable shards, and integrity manifest;
   - `data/gemma4_model_identity.py` binds local models to canonical config/index metadata and every
     safetensors shard, or remote models to an immutable revision;
   - deterministic trace-validation failures are non-retryable in the production supervisor.
3. `data/validate_gemma4_distill_traces.py` — implemented, unit tested, and run on both full bundles
   - validate/merge manifests and produce train/validation dataset indexes.
   - records cross-split text hashes, emits location-independent bundle paths, and can fail
     immediately on leakage.
4. `data/upload_gemma4_distill_dataset.py` and `data/run_gemma4_trace_dataset.sh` — implemented,
   reviewed, focused-test verified, and used for both local bundles
   - run resumable one-worker-per-GPU train generation followed by validation generation;
   - isolate each vLLM worker in a process group so supervisor termination cannot orphan GPU jobs;
   - regenerate and fully validate the complete index before any Hub mutation;
   - refuse partial/smoke bundles, non-empty destination branches, dirty source trees, missing
     identities, and changed-after-validation files;
   - stage content-verified snapshots, bind the commit to the observed parent revision, and upload
     privately by default; public visibility requires an explicit uploader flag and is verified
     before and after the commit;
   - report the immutable Hub commit OID without exposing the token, including through chained
     exception tracebacks;
   - the initial private E2B attempt failed closed at the storage quota; later explicit public uploads
     completed for both bundles and were independently verified at their immutable commits.
5. `data/audit_gemma4_cross_engine_topk.py` — implemented and focused CPU-test verified
   - binds expected dataset-index and teacher-identity hashes, verifies every registered shard before
     global length-stratified selection, and validates selected rows against the complete trace schema;
   - enforces the Gemma 4 BF16+SDPA+softcap contract and explicit calibrated thresholds;
   - the expanded E2B/E4B diagnostic reports were produced by the precursor audit and pass those
     thresholds; the integrated script has not been relaunched on GPU under this preparation-only
     instruction.
6. `data/rescore_gemma4_training_topk.py` and `data/GEMMA4_TRAINING_TOPK_RESCORER.md` — implemented,
   reviewed, and focused CPU-test verified; production scoring not run
   - writes a disjoint one-to-one target overlay, with atomic manifests/index and strict source
     dataset/shard/manifest/trace-ID binding;
   - uses full-sequence teacher forcing, causal predecessor scoring, BF16+SDPA, Gemma 4 softcap, FP32
     full-vocabulary normalization, and exact chunked/native parity.
7. `data/preflight_gemma4_training_topk_overlay.py` plus overlay routing in
   `gemma4_topk_distill_fsdp2.sh` — implemented and focused-test verified
   - accepts only schema `gemma4-hf-bf16-sdpa-topk-overlay-v1` and requires the exact immutable source
     index through `SOURCE_DATASET_INDEX`;
   - validates source and overlay self-hashes, rescore configuration, parity receipt, exact teacher
     identity, student/tokenizer identity, every source/overlay shard pair, copied row content,
     target tensors, trace order/set, row/token totals, and the overlap exception;
   - emits only verified overlay train/validation paths through the existing no-`eval` launcher
     contract. The combined source/overlay/launcher suite passed 48 tests.
8. `data/preflight_gemma4_topk_distill.py` — implemented and unit tested for source vLLM bundles
   - verifies index/config/shard hashes, explicit split-specific roster/sample counts, prompt/sampling
     contracts, tokenizer identity, top-k integrity, teacher identity, student identity, and
     cross-split leakage before a production launcher receives file lists;
   - the launcher automatically selects the overlay preflight instead for the overlay schema.
9. `main_full_vocab_distill_fsdp2.py`, `config/full_vocab_distill_fsdp2.yaml`, and
   `gemma4_topk_distill_fsdp2.sh` — trainer and production launcher verified
   - one precomputed-top-k trainer parameterized by teacher metadata, student model, files, and
     output repositories.
   - the trainer has passed synthetic 8,192-token, both live-teacher/student directions,
     checkpoint-resume, exact-validation, and self-contained HF-save gates.
   - production use must require the dataset index/preflight path; direct Parquet inputs may remain
     only behind an explicit smoke-only override.
10. `eval_math_passk.py`, `gemma4_eval_metrics.py`, and `eval_gemma4_math.py` — implemented and unit
   tested
   - generation defaults to exactly 64 deterministic samples/question, streams finalized audit
     JSONL one question at a time, and keeps only compact metric summaries in host memory;
   - offline aggregation implements the requested unbiased pass@k and exposes the still-open
     mean@k/maj@k conventions as explicit choices rather than silently fixing them.
11. `eval_gemma4_ood.py` — implemented and unit tested
   - pinned lm-eval invocation for MMLU, WinoGrande, TriviaQA, HellaSwag, and ARC-Challenge;
   - the harness submodule/environment is not initialized in this workspace, so no real OOD
     benchmark has been launched.

Each expensive entry point supports a tiny bounded smoke configuration and records or prints its
resolved configuration before work begins. The runbook is deliberately fail-closed:

1. **Complete locally:** generate and fully validate both immutable vLLM source bundles.
2. **Complete remotely:** publish both source bundles publicly and independently verify the immutable
   commits. E2B is commit `e32aaa02681ae83b3d7256b1b155c9084da2f289` with index
   `d170166abad89588880f5c0a9eac43006f9a32bbe27cec28d7eb97c65288dbcc`; E4B is commit
   `2b6e49a0a456ee9d67b16a1dc61785562bee90c9` with index
   `8b5712e0f5dea3388340a9bc91a6ceee40ff2ff990e66b7238090e240daeda6c`. Each commit contains
   exactly 2,485 registered files. Remote verification compared every path and size, Parquet LFS
   SHA-256, and non-LFS Git blob SHA-1.
3. **Optional diagnostic:** rerun the integrated cross-engine audit if a final committed-script
   report is required, then review its explicit pass/fail JSON.
4. **Authorized for E2B-to-E4B:** run the native-forward parity gate and bulk unsharded-HF target
   rescore into a separate output root.
5. After the complete overlay exists, run the overlay-specific preflight
   and launcher route with both `DATASET_INDEX` and `SOURCE_DATASET_INDEX`; only its emitted shard
   lists may reach the trainer.
6. Run the authorized one-step, no-upload E4B student smoke.
7. After smoke, code, and artifact review pass, launch the authorized 750-step E2B-to-E4B run.
8. Generate immutable evaluation traces, aggregate math metrics offline, then run the full pinned OOD
   matrix without `--limit`.

Source-bundle publication is complete. E2B target rescoring, its E4B smoke, and its 750-step run are
authorized. The opposite direction, benchmark production, and post-distillation RL remain future work.

### Historical/resume-only trace commands — DO NOT RUN

Both bundles are already complete. These commands are retained only to document the exact registered
invocations and to support a separately approved repair/resume. Leaving the guard at `NO` makes a
copied block inert.

E4B-RL-step-100 to E2B trace bundle:

```bash
GEMMA4_TRACE_RESUME_AUTHORIZED=NO
test "${GEMMA4_TRACE_RESUME_AUTHORIZED}" = YES && \
  PYTHON_BIN=/tmp/.venv-gemma4-e2e/bin/python \
  TRAIN_PARQUET=/tmp/verl/data/deepscaler_4of4strict_rl_train.parquet \
  VALIDATION_PARQUET=/tmp/verl/data/deepscaler_4of4strict_rl_val200_x16.parquet \
  rl-distill-scripts/data/run_gemma4_trace_dataset.sh \
    --teacher-model /tmp/verl/models/nemorl-gemma4-e4b-step100-vllm \
    --teacher-content-sha256 830d47f78008b56787798a21a5e53d4e402a405bb899a9db5e18b7b83371110f \
    --direction e4b_rl100_to_e2b \
    --output-root /lambda/nfs/Jason-scale/rl-distill-traces/gemma4-e4b-rl100-topk128 \
    --hf-repo-id JWei05/gemma4-e4b-rl100-topk128-traces \
    --gpus 0,1 \
    --allow-question-overlap
```

E2B-base to E4B trace bundle:

```bash
GEMMA4_TRACE_RESUME_AUTHORIZED=NO
test "${GEMMA4_TRACE_RESUME_AUTHORIZED}" = YES && \
  PYTHON_BIN=/home/ubuntu/rl-distill/.venv-gemma4/bin/python \
  TRAIN_PARQUET=/tmp/verl/data/deepscaler_4of4strict_rl_train.parquet \
  VALIDATION_PARQUET=/tmp/verl/data/deepscaler_4of4strict_rl_val128_clean_seed42.parquet \
  VALIDATION_SOURCE_DATASET=deepscaler_4of4strict_rl_val128_clean_seed42 \
  EXPECTED_TRAIN_QUESTIONS=9723 \
  EXPECTED_VALIDATION_QUESTIONS=128 \
  TRAIN_SAMPLES_PER_QUESTION=5 \
  VALIDATION_SAMPLES_PER_QUESTION=1 \
  rl-distill-scripts/data/run_gemma4_trace_dataset.sh \
    --teacher-model /tmp/hf_cache/hub/models--google--gemma-4-E2B/snapshots/d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f \
    --teacher-content-sha256 76dc84a5a805a2c8b91e9ccc00b8dbf8f4a99bf0d56ab25832f6e6addd4f7f57 \
    --direction e2b_base_to_e4b \
    --output-root /tmp/verl/gemma4-e2b-base-topk128-v128 \
    --hf-repo-id JWei05/gemma4-e2b-base-topk128-traces \
    --gpus 0,1,2,3,4,5,6,7
```

### E2B-base-to-E4B production distillation

The primary experiment must use the finalized unsharded-HF overlay, not the immutable source's vLLM
targets. Set `DATASET_INDEX` to the overlay index and `SOURCE_DATASET_INDEX` to the exact vLLM source
index so the launcher selects the strict overlay preflight. After the overlay passes preflight and a
one-step no-upload gate, launch the 750-step run with global batch 128, microbatch 2, and a
4,096-token vocabulary-projection chunk. Train for at most two epochs, warm up to `2e-6` over 100
optimizer steps, then decay linearly to `2e-7` at step 750. Validation runs over all 128 registered
rows with no distributed padding. Save and upload directly loadable HF checkpoints at steps 250,
500, and 750.

The first overlay launch performs the complete source/overlay content preflight and writes
`training_preflight_receipt.json` beside the overlay index. Later launches reuse that receipt after
checking both index identities, the exact student snapshot, validator source hashes, the requested
roster/identity contract, and filesystem metadata for every registered artifact. Set
`REFRESH_PREFLIGHT_RECEIPT=true` to force a new full scan after any intentional artifact or validator
change; receipt mismatches fail closed instead of silently rescanning.

```bash
GEMMA4_E2B_TO_E4B_PRODUCTION_AUTHORIZED=YES
test "${GEMMA4_E2B_TO_E4B_PRODUCTION_AUTHORIZED}" = YES && \
  MODEL_PATH=/home/ubuntu/.cache/huggingface/models--google--gemma-4-E4B/snapshots/411aa17b749aa952df1359d2dcea73917a544d9a \
  DATASET_INDEX=/tmp/verl/datasets/gemma4-e2b-base-topk128-hf-overlay-v128-seed42/dataset_index.json \
  SOURCE_DATASET_INDEX=/tmp/verl/datasets/gemma4-e2b-base-topk128-traces-e32aaa02681a-val128-seed42/dataset_index.json \
  DISTILL_DIRECTION=e2b_base_to_e4b \
  EXPECTED_TEACHER_IDENTITY_SHA256=2d48d343709dcae087d6ff2def9f09d2950ca66dc2183a8bee38850c4ddbbb36 \
  EXPECTED_STUDENT_IDENTITY_SHA256=acdc0d2bcb8f676593b5387807da1cd1b84a9e26fa279db4a86f54a211055b2d \
  EXPECTED_TRAIN_QUESTIONS=9723 EXPECTED_VALIDATION_QUESTIONS=128 \
  EXPECTED_TRAIN_SAMPLES_PER_QUESTION=5 EXPECTED_VALIDATION_SAMPLES_PER_QUESTION=1 \
  MICRO_BATCH_SIZE_PER_GPU=2 FULL_VOCAB_KL_CHUNK_SIZE=4096 TRAIN_BATCH_SIZE=128 \
  LR=2e-6 LR_WARMUP_STEPS=100 LR_SCHEDULER_TYPE=linear MIN_LR_RATIO=0.1 \
  TOTAL_EPOCHS=2 TOTAL_TRAINING_STEPS=750 SAVE_FREQ=250 TEST_FREQ=10 \
  PROJECT_NAME=gemma4-distill-vs-rl EXP_NAME=e2b-base-to-e4b-topk128-lr2e6-linear-b128-2ep-750-v128-seed42 \
  HF_PUSH_ENABLE=true HF_PUSH_REPO=JWei05/gemma4-e2b-base-to-e4b-topk128-distill \
  rl-distill-scripts/gemma4_topk_distill_fsdp2.sh
```

Evaluation remains similarly held. After D33-D40 are resolved, use `eval_math_passk.py` for the
64-sample generation/scoring pass, `eval_gemma4_math.py` for immutable offline re-aggregation, and
`eval_gemma4_ood.py` for the pinned full lm-eval matrix. Do not use `--max_questions`,
`--allow_nonstandard_sample_count`, `--allow-variable-samples`, `--allow-lexical-majority`,
`--skip-harness-git-check`, or lm-eval `--limit` for reported production results.

## Verification gates before expensive runs

1. **Passed:** reconstruct E4B step 100, verify every raw delta link/final hash, expand the 54
   shared-KV aliases, copy pinned processor metadata, prove raw/expanded Transformers-logit parity,
   and load/generate with vLLM 0.25.1.
2. **Passed:** preflight compares the trace tokenizer with the exact student tokenizer and binds
   teacher/student identities. E2B is pinned to
   `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`; E4B is pinned to
   `411aa17b749aa952df1359d2dcea73917a544d9a`. Local generation binds every exact safetensors file
   and relevant metadata to content/model hashes.
3. **Partially passed:** the generator and RL configuration point to the same template SHA256, and
   the reference Gemma 3 work established byte-identical eval rendering. A recorded multi-prompt
   generator-versus-RL token parity fixture is still desirable before production.
4. **Passed in both directions:** five E2B-base responses and live E4B-RL responses produced exact
   top-128 targets; each direction drove a finite real student update. Separate 8,192-cap live gates
   naturally stopped and fully validated for both teachers.
5. **Passed for both complete source bundles:** exact IDs/masks, top-k rank/shape/mass, row/config
   provenance, deterministic seeds, all shard hashes, decode checks, exact roster sizes, and the
   explicit seven-question overlap registration. The separate source+overlay preflight and launcher
   schema routing are implemented and passed their focused combined suite; they have not been run on
   a complete overlay because no bulk rescore has started.
6. **Passed:** objective, mask shift, teacher/student mass, Gemma 4 softcap, activation-checkpointed
   chunks, partial validation coverage, and a full 8,192-response-token FP32-master/BF16-compute
   update with 16,384 active tokens.
7. **Passed for both students:** real two-rank FSDP2 forward/backward, validation, and checkpoint
   save produced finite values for E2B and E4B. Their HF snapshots include pinned processor metadata
   plus 60 E2B or 54 E4B shared-KV aliases and load/generate in vLLM 0.25.1.
8. **Passed:** live E4B-to-E2B and E2B-to-E4B validation/update/save gates completed with HF push
   disabled.
9. **Passed:** RL resume and distillation resume loaded model, optimizer, scheduler, RNG, and data
   state and continued to a new saved step. The source-dataset uploader is fail-closed and focused
   tested; production upload is performed only after complete trace validation. Training-checkpoint
   uploads are also fail-closed: `HFPusher.wait()` re-raises background upload failure or timeout,
   while preserving an already-active training exception as the primary failure.
10. **Implemented and focused-test verified:** unbiased pass@k exact combinatorics, stable sample
    grouping, semantic majority classes, invalid/tie handling, and predictive-entropy summaries.
    The scientific mean@k/maj@k choices remain open even though the code exposes them.
11. **Pending:** reproduce a known base-model math and OOD score before reporting new checkpoints.
12. **Passed for trace generation:** both teachers loaded with the 12,288 context contract and
    generated naturally stopped responses under an 8,192-token cap. This is separate from the RL
    rollout boundary, whose 512-token verl smoke remains the verified RL configuration.
13. **Explicitly registered:** preserve the 180 duplicate-text train-row occurrences. The immutable
    source registers seven train/validation overlaps, while the E2B-to-E4B derivative excludes them
    and validates the clean 128 x 1 roster without an overlap exception.
14. **Cross-engine diagnostic passed:** expanded E2B and E4B reports cover 32 traces each and pass the
    calibrated top-1/top-k/log-probability thresholds. Native HF versus manual projection is exact;
    vLLM versus HF is close but non-identical, motivating a disjoint training-target overlay.
15. **Prepared and authorized for E2B-to-E4B:** the unsharded-HF rescorer, parity receipt,
    source+overlay preflight, and launcher routing are focused-test verified. Bulk scoring and
    complete overlay finalization remain unstarted. A real finalized overlay must pass that preflight
    before the E4B student smoke.
16. **Final preparation gate:** complete the consolidated test/lint review, inspect the full diff for
    secrets or artifacts, commit with AI-assistance attribution, and push the reviewed branch before
    starting the authorized E2B target rescore and E4B distillation.

## Decision and ambiguity ledger

Items marked **resolved** are part of the registered trace/preparation contract. Items still phrased
as questions remain unresolved for the later 750-step distillation or evaluation runs. Recommended
defaults are proposals, not silently adopted decisions.

### Models, checkpoints, and comparison target

- **D01 — Base revisions: resolved.** Use
  `google/gemma-4-E2B@d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` and
  `google/gemma-4-E4B@411aa17b749aa952df1359d2dcea73917a544d9a`. Never use floating `main`.
- **D02 — E4B step-100 distribution: resolved for trace generation.** Use the verified local
  vLLM-ready artifact at `/tmp/verl/models/nemorl-gemma4-e4b-step100-vllm`, content SHA256
  `830d47f78008b56787798a21a5e53d4e402a405bb899a9db5e18b7b83371110f`. No E4B model upload is
  needed or authorized for these local production workers.
- **D03 — Distilled checkpoint selection.** Should complete evaluations cover steps 250, 500, and
  750, only step 750, or the best held-out-loss checkpoint as well? Recommended: inexpensive held-out
  loss for all; full generation/OOD for 250/500/750 and any distinct best checkpoint.
- **D04 — Definition of “better” in line 1.** What is the primary dataset/metric and how are mixed
  benchmark wins aggregated? Recommended: declare strict validation mean@64 as the primary,
  with pass@k/maj@k, OOD math, and non-math reported separately rather than collapsed into one score.
- **D05 — Fairness axis.** Is the headline comparison endpoint quality, equal optimizer steps,
  equal generated sequences/tokens, equal wall-clock, or estimated compute? Recommended: report the
  requested endpoint comparison and add example-, token-, and GPU-hour-normalized curves; do not
  claim a compute-matched result unless we run one.
- **D05A — Line-2 behavior-matching domain.** Is the E2B-to-E4B distillation intentionally limited
  to the strict-4/4 math corpus, or should it add broad non-math prompts to support the stronger
  claim that E4B behaves like E2B generally? Recommended: keep the requested math-only run as the
  primary controlled experiment, explicitly narrow its claim, and add a broad-prompt variant only
  as a separately named follow-up.
- **D05B — Duplicate train question texts: resolved for primary trace artifacts.** Preserve all
  9,723 historical row-level generation units, including the 180 repeated-text occurrences. Report
  row-weighted and distinct-text-weighted analyses later; any deduplicated experiment needs a new
  dataset name and roster hash.
- **D05C — E2B-to-E4B validation roster: resolved.** Preserve the historical training rows, but use
  128 unique prompts sampled at seed 42 from the 193 validation prompts whose exact question hashes
  are absent from train. Generate one validation trace per prompt. Retain the full-200 and clean-193
  rosters for later continuity reporting, but do not use the seven leaked prompts for this run's
  in-loop validation.

### Generation and trace data

- **D06 — Sampling distribution: resolved.** Temperature 1.0, top-p 1.0, sampling top-k disabled,
  max response 8,192, max prompt 4,096, and total context 12,288.
- **D07 — Sampling seeds: resolved.** Derive each seed from global seed 42, split, UID, and sample
  index so every trace is independently reproducible.
- **D08 — Stop-string inclusion: resolved.** Preserve matched stop-associated sampled token IDs and
  train on them. Live E4B/E2B gates showed that vLLM retains the exact seven `<end_of_turn>` IDs while
  omitting the string from `completion.text`; the schema stores exact decoded and vLLM text separately.
- **D09 — Prompt overflow: resolved for this roster.** Reject rather than truncate, but the full
  audit found a 2,518-token train maximum and 1,866-token validation maximum, so no selected row is
  affected.
- **D10 — Trace filtering.** Train on every trace, including incorrect, duplicate, empty, malformed,
  and max-length responses, or filter some classes? Recommended: retain every trace in the artifact,
  but exclude empty/corrupt rows; make incorrect/truncated filtering a declared experimental toggle
  rather than silently filtering.
- **D11 — Duplicate responses: resolved.** Retain multiplicity because frequency is part of the
  sampled teacher distribution.
- **D12 — Storage precision: resolved.** Use `int32` IDs and FP16 full-vocabulary-normalized log
  probabilities, cast to FP32 for the loss, with no additional lossy transform.
- **D12A — Target-engine policy: resolved.** Preserve the vLLM top-128 values as the immutable
  generation record, but use a separately indexed unsharded-HF-rescored top-128 overlay as the primary
  distillation targets. Both layers use the exact stored token IDs; no re-tokenization and no online
  teacher are allowed.
- **D13 — Source trace repositories: resolved and independently verified.** Both repositories are
  public. E2B-base traces are at `JWei05/gemma4-e2b-base-topk128-traces`, immutable commit
  `e32aaa02681ae83b3d7256b1b155c9084da2f289`, index
  `d170166abad89588880f5c0a9eac43006f9a32bbe27cec28d7eb97c65288dbcc`. E4B-RL step-100 traces are at
  `JWei05/gemma4-e4b-rl100-topk128-traces`, immutable commit
  `2b6e49a0a456ee9d67b16a1dc61785562bee90c9`, index
  `8b5712e0f5dea3388340a9bc91a6ceee40ff2ff990e66b7238090e240daeda6c`. Each commit has 2,485
  registered files; independent remote verification matched every path and size, Parquet LFS SHA-256,
  and non-LFS Git blob SHA-1. Overlay repository names and upload tooling remain unresolved and
  unauthorized.
- **D14 — Gold/reward metadata: resolved.** Compute and store strict grade/prediction during
  generation without filtering training rows by correctness.

### Distillation objective and optimization

- **D15 — Exact top-k objective.** Choose one:
  1. existing-style unrenormalized partial forward KL over teacher top-128 support, using full-vocab
     normalized teacher/student log probabilities and clamping negative per-token partial sums to 0;
  2. the same partial sum without the clamp;
  3. teacher-renormalized top-128 cross-entropy/KL against the student's full-vocabulary softmax; or
  4. both distributions renormalized only over teacher top-128 support.
  The smoke-tested launcher currently defaults to option 2: this preserves the historical local
  partial-KL formula, while a clamp would silently zero the gradient for negative partial
  contributions. This is an implementation default, not a scientific decision. Option 3 plus an
  explicit residual-tail bucket is another principled candidate, but would be a new objective and
  needs a separately named run. Confirm the primary objective before production training.
- **D16 — Temperature.** Is distillation temperature 1.0? Recommended: yes. Precomputed normalized
  top-k log probabilities do not permit exact reconstruction of a different full-vocabulary
  temperature without additional statistics.
- **D17 — Auxiliary hard-label loss.** Pure top-k KL, or mix in CE on the sampled teacher token?
  Recommended: pure top-k forward KL for the main experiment; optionally run CE as a separately
  named ablation.
- **D18 — Loss aggregation.** Response-token mean, sequence mean, or another weighting?
  Recommended: response-token mean to match the existing RL token-level loss convention, while
  reporting sequence-weighted validation metrics as diagnostics.
- **D19 — Batch interpretation: resolved.** Use global batch 128 sequences per optimizer update,
  split across eight data-parallel ranks. With microbatch 2, each rank accumulates eight
  microbatches per optimizer step.
- **D20 — Gradient clipping and epsilon.** These were not specified. Recommended: global grad norm
  1.0 and AdamW epsilon `1e-8`.
- **D21 — Precision/offload.** Confirm BF16 model/compute, FP32 optimizer states, activation
  checkpointing, and only the offload needed to fit. Recommended: start BF16 with activation
  checkpointing, FP32 master parameters/Adam states, and no offload on 8xH100; enable offload only if
  a measured smoke requires it. A two-H100 8,192-token FP32-master/BF16-compute gate has passed, but
  the final 8xH100 production memory/offload policy is still a declared run choice.
- **D22 — Multimodal towers.** Freeze vision/audio towers and train only the language model?
  Recommended: yes, matching the RL runs.
- **D23 — Sequence overflow in training.** Fail rather than truncate any stored 12,288-token trace?
  Recommended: yes; generation should already enforce the contract.
- **D24 — Shuffle seed and coverage: resolved.** Keep seed 42, global batch 128, two epochs, and a
  750-step cap. This consumes 96,000 sequence draws and stops eight optimizer updates before the end
  of epoch two; record the exact trace IDs omitted by both shuffled epoch tails.
- **D25 — Scheduler endpoint: resolved.** Warm up linearly to `2e-6` over 100 optimizer steps, then
  decay linearly to `2e-7` at step 750 (`MIN_LR_RATIO=0.1`).

### During-training validation, logging, and artifacts

- **D26 — Validation every 10 steps.** Does this mean held-out top-k loss only, generated math
  accuracy, or both? Recommended: all held-out trace losses every 10 steps (1,000 for the full 200 or
  965 for a clean 193-question subset); a cheaper greedy or small fixed sampled generation every 50
  steps; full sampled@64 only at saved checkpoints.
- **D27 — Step-zero validation.** Run validation before the first optimizer update? Recommended: yes.
- **D28 — Generative validation sample count.** If generated validation runs during training, how
  many responses/question? Recommended: greedy@1 plus four sampled responses on a fixed 50-question
  panel during training; reserve 200x64 for checkpoint evaluation.
- **D29 — W&B naming.** Confirm entity/project and run names. Recommended: entity `rl-distill`,
  project `gemma4-distill-vs-rl`, with direction/model/teacher-step/seed in each run name.
- **D30 — Model HF repositories.** What repo names, visibility, and step layout should be used?
  Recommended: one private repo per direction with `step_000250`, `step_000500`, and `step_000750`
  revisions/folders, then make selected final artifacts public only after verification.
- **D31 — Checkpoint retention/resume.** Keep all three local/HF saves and auto-resume from the
  latest complete checkpoint? Recommended: yes, including optimizer/scheduler/RNG/data position.
- **D32 — Hardware target.** What GPU/node budget should scripts assume? Recommended: one 8xH100
  node per distillation run initially, with microbatch 1 and global accumulation to 64.

### Evaluation metrics and datasets

- **D33 — k values.** Which k values should be reported from 64 samples? Recommended:
  `[1, 2, 4, 8, 16, 32, 64]`.
- **D34 — mean@k convention.** Since expected mean accuracy is invariant to k, should we report only
  `mean@64`, repeat the same unbiased estimate at every k, use fixed prefixes, or Monte Carlo
  subsets? Recommended: report `mean@64` once as the primary mean metric; if `mean@k` labels are
  required, use deterministic Monte Carlo subsets and clearly identify the estimator.
- **D35 — maj@k convention.** For k < 64, use fixed prefixes or expected majority accuracy over
  random without-replacement subsets? Recommended: deterministic Monte Carlo subsets from the 64
  samples, with enough resamples to make Monte Carlo error negligible and a recorded seed.
- **D36 — Majority answer equivalence.** Vote by raw boxed string, normalized string, or semantic
  math equivalence? Recommended: form semantic equivalence classes using the same math verifier,
  with a stable normalized-string fallback when verification fails.
- **D37 — Invalid/no-box votes.** Should invalid responses all vote as one `<none>` class, each count
  separately, or abstain? Recommended: count each as an invalid abstention for majority selection,
  but keep them in the denominator and mark the result wrong if every response abstains.
- **D38 — Majority ties.** How should tied answer classes be scored? Recommended: conservative wrong
  unless one deterministic, predeclared tie-breaker is selected; never first-occurrence order.
- **D39 — Entropy definition.** Which entropy is required? Recommended: report both response-token
  predictive entropy (response-only, before sampling temperature, token- and sequence-weighted) and
  empirical normalized-answer entropy across 64 samples. State whether predictive entropy is exact
  full vocab or a top-k-plus-residual approximation.
- **D40 — Equivalence criterion.** What numerical standard establishes that E2B base and
  E2B-to-E4B are “the same”? Recommended: declare per-metric equivalence margins before evaluation and use paired
  bootstrap 95% confidence intervals over questions; the margins themselves require user values.
- **D41 — Confidence intervals.** Should all sampled math comparisons include paired bootstrap CIs?
  Recommended: yes, with a fixed bootstrap seed and at least 10,000 question-level resamples.
- **D42 — 64 samples scope.** Is 64 required only on the 200-question in-distribution validation set
  or on every math benchmark? Recommended: 64 on ID, MATH500, and all AIME sets; use a lower but
  declared count for full GSM8K/OlympiadBench/MinervaMath if compute is constrained. If the goal is
  one uniform protocol and budget permits, use 64 everywhere.
- **D43 — Math decoding.** Confirm sampled math uses temperature 1.0, top-p 1.0, top-k disabled,
  max new tokens 8,192, and Gemma stop strings. Recommended: yes.
- **D44 — Greedy secondary evaluation.** Required for every math model/checkpoint or only final
  checkpoints? Recommended: all saved checkpoints; it is cheap and detects distributional shifts.
- **D45 — MATH500 source.** Use `HuggingFaceH4/MATH-500` or the older converted parquet?
  Recommended: pin `HuggingFaceH4/MATH-500` and preserve a mapping to the historical parquet for
  continuity.
- **D46 — OlympiadBench scope.** Use all 674 text-only questions or the older 500-question subset?
  Recommended: all 674 text-only questions.
- **D47 — AIME sources.** Confirm LLM360 for AIME 2024 and MathArena for 2025/2026, with immutable
  revisions. Recommended: yes.
- **D48 — BeyondAIME.** Include it in addition to AIME 2024/25/26? Recommended: include as an
  optional secondary set because prior repo evaluations already use it, but do not replace an AIME
  set with it.
- **D49 — OOD harness path.** Direct `lm_eval --model vllm` or OpenAI-compatible server?
  Recommended: direct vLLM if Gemma 4 support is verified; otherwise server with
  `tokenized_requests=False`.
- **D50 — Harness/task revisions.** Pin historical lm-eval 0.4.13 for continuity or a newer verified
  commit? Recommended: pin an exact current commit after reproducing one historical baseline; record
  task YAML hashes.
- **D51 — Response diagnostics.** Make response length, finish reason, stop rate, strict-format rate,
  and truncation rate mandatory for every sampled math evaluation? Recommended: yes.
- **D52 — Evaluation seeds.** One fixed 64-sample batch or multiple independent 64-sample runs?
  Recommended: one fixed batch for the full matrix plus a second seed on the primary ID comparison
  if budget permits.

### Post-distillation E4B RL

- **D53 — RL framework.** NeMoRL or verl? Recommended: use the same NeMoRL recipe as the existing E4B
  baseline for the cleanest comparison, but only after committing/applying the missing patch and
  passing a ratio≈1 gate. If verl is chosen, first production-validate the attention-mask fix.
- **D54 — RL horizon and seeds.** Total steps, number of seeds, and stop rule are unspecified.
  Recommended: at least one seed through 125 steps for comparison with available curves, then extend
  to 500 only if healthy; two or three seeds are needed for strong optimization claims.
- **D55 — RL initialization checkpoint.** Start from step 750 or the best held-out-loss distilled
  checkpoint? Recommended: declare step 750 as primary in advance and optionally run best-validation as an
  ablation.
- **D56 — RL recipe.** Confirm the same strict-4/4 data, 12-shot prompt, 64 prompts x16 generations,
  8,192 response cap, strict reward, optimizer, warmup, and validation protocol as the existing E4B
  run. Recommended: yes; change only initialization.
- **D57 — Matched comparator.** Compare post-distillation E4B RL to existing E4B RL step 100, a new
  base-E4B RL rerun under repaired code, or both? Recommended: both, because the existing run has
  provenance/termination caveats.
- **D58 — Optimizer state.** Start RL with a fresh optimizer/scheduler rather than carrying the
  distillation optimizer? Recommended: yes.

## Decision log

- **2026-07-30 — Precomputed top-k chosen.** Store teacher top-128 IDs and normalized log
  probabilities during generation instead of loading/scoring the teacher online during
  distillation. Use sharded, resumable storage and stream it during training.
- **2026-07-30 — Preserve verified verl fork; forward-port separately.** Do not rebase the current
  dirty/verified branch onto upstream main. Freeze it, then port the required concerns in a separate
  `release/v0.8.0` worktree with fresh unit, RL, distillation, checkpoint, and resume gates.
- **2026-07-30 — E4B step 100 materialized locally.** The lock-driven chain through steps
  20/40/60/80/100 produced SHA256
  `d565a3ff371906ca31a5e355472d70366b6956c0e82a914de4ea8a7c0085630c`. The vLLM-ready expansion
  adds pinned processor metadata and 54 shared-KV aliases, producing SHA256
  `830d47f78008b56787798a21a5e53d4e402a405bb899a9db5e18b7b83371110f`; raw/expanded logits are
  bit-identical and vLLM 0.25.1 generation passed. No E4B artifact was uploaded.
- **2026-07-30 — verl Gemma 4 smoke and resume passed.** Two fresh steps and a resumed third step
  completed rollout/reward/update/weight-sync/checkpoint paths with ratio means near one and finite
  gradients. This verifies the 512-token smoke configuration, not yet the 8,192-token boundary.
- **2026-07-30 — Gemma 4 precomputed distillation smoke passed.** Hidden-state projection,
  softcapping, exact stored-token targets, partial validation coverage, activation-checkpointed
  vocabulary chunks, FSDP2 backward, and HF save passed on two ranks.
- **2026-07-30 — Synthetic full-length distillation update gate passed.** A two-rank E2B update with
  FP32 master parameters, BF16 compute, and 8,192 active response positions per rank processed
  16,384 response tokens with finite loss/gradient values, validation, and checkpoint save. This
  validates training memory/shape behavior, not live 8,192-token teacher generation.
- **2026-07-31 — Live E4B trace-to-E2B gate passed.** The expanded E4B-RL step-100 artifact generated
  a real top-128 trace through vLLM; exact stored IDs/targets then drove a finite E2B update with
  validation and checkpoint save. This was a one-question, one-response, 32-token-cap smoke only.
- **2026-07-31 — Source split audit found duplicates and leakage.** Train has 9,723 row-level units
  but 9,543 distinct question texts. Validation has 200 distinct UIDs/texts repeated to 3,200 rows.
  Seven validation texts occur once in train. Primary traces preserve the exact historical roster;
  overlap is an explicit indexed exception and clean evaluation uses the remaining 193 questions.
- **2026-07-31 — E2B-to-E4B validation reduced to a clean deterministic 128 x 1 roster.** Deduplicate
  validation by UID, exclude the seven train-overlapping prompts, sample 128 of 193 at seed 42, and
  preserve source order. The resulting Parquet and ordered roster hashes are registered above; the
  production validator, uploader, and preflight enforce the split-specific counts.
- **2026-07-31 — E2B trace corpus uploaded and production-preflighted.** Hub dataset revision
  `e32aaa02681ae83b3d7256b1b155c9084da2f289` contains 48,615 train and 1,000 original validation
  traces. The training split is reused byte-for-byte; sample index 0 from each selected clean
  validation UID forms the 128-row derivative. Its combined local index SHA256 is
  `efe76f5a53225e97081a750495ceb8ebe26b1a72a3ea8e0b74ea079a87828c0a`, and production preflight
  passed against pinned E4B revision `411aa17b749aa952df1359d2dcea73917a544d9a`.
- **2026-07-31 — Both production-length teacher gates passed.** E4B generated 149 tokens and E2B 40
  under the 8,192 cap; both naturally stopped on `<end_of_turn>`, stored exact `[response, 128]`
  targets, and passed bundle validation. vLLM retained the seven stop token IDs while omitting them
  from completion text; those sampled IDs remain trainable by decision D08.
- **2026-07-31 — Both student directions, serving, and distillation resume passed.** Live E4B traces
  updated E2B, live E2B traces updated E4B, and a saved distillation job resumed with model,
  optimizer, scheduler, RNG, and data state. Saved E2B/E4B snapshots include processor metadata and
  all 60/54 shared-KV aliases and load/generate in vLLM 0.25.1.
- **2026-07-31 — Both full vLLM source bundles completed locally.** Each immutable index registers
  48,615 train and 1,000 validation rows across 1,241 shards. E4B-RL-to-E2B contains 14,049,865
  response tokens; E2B-base-to-E4B contains 8,257,057. Full validation, decode checks, exact roster
  checks, and the explicit seven-overlap exception passed.
- **2026-07-31 — Initial private publication path failed closed.** The first E2B-teacher upload failed
  after validation because the private repository storage quota was exhausted; the corresponding
  early E4B upload-result log was empty. This is retained as execution history, not current status.
- **2026-07-31 — Both source bundles published publicly and independently verified.** E2B-base traces
  are at commit `e32aaa02681ae83b3d7256b1b155c9084da2f289`; E4B-RL step-100 traces are at commit
  `2b6e49a0a456ee9d67b16a1dc61785562bee90c9`. Each public commit contains exactly 2,485 registered
  files. Independent remote verification matched every path and size, LFS SHA-256 for each Parquet,
  Git blob SHA-1 for every non-LFS file, and the expected dataset-index SHA-256.
- **2026-07-31 — Cross-engine audit motivates a separate training overlay.** Expanded E2B/E4B
  diagnostics show exact native-versus-manual HF projection but small vLLM-versus-HF differences.
  Preserve the vLLM bundles as immutable generation records; precompute primary distillation targets
  into a disjoint unsharded-HF overlay over the exact same token IDs. Never re-tokenize and never load
  an online teacher in student training. This diagnostic does not establish FSDP2 numerical
  equivalence; that requires a separate real-engine audit.
- **2026-07-31 — E2B-to-E4B production authorized after repository push.** The rescorer, audit,
  focused CPU tests, and guarded operator documentation are prepared. After the training-engine
  overlay passes parity, complete preflight, and a one-step no-upload E4B gate, launch the 750-step
  SFT with W&B logging and private HF checkpoint uploads.
  The opposite distillation direction, benchmark production, and post-distillation RL remain future
  work.
- **2026-07-31 — Initial `5e-6` production schedule stopped at step 130.** W&B run `yswil8j8`
  completed through validation step 130 without OOM or a checkpoint save. The operator judged the
  peak learning rate too high and stopped the run before its first step-250 upload. Restart from the
  immutable E4B base with global batch 128, two epochs capped at 750 steps, 100-step warmup to
  `2e-6`, and linear decay to `2e-7`.
