# Progress Log

> Gemma 4 E2B/E4B distillation-vs.-RL planning, current artifact status, and the production runbook
> now live in [`GEMMA4_DISTILL_VS_RL_EXPERIMENTS.md`](GEMMA4_DISTILL_VS_RL_EXPERIMENTS.md). That
> canonical document supersedes older Gemma 4 status statements in this chronological log.

A running log of experiments/work in this repo. Newest entries on top. Each entry records the goal,
what was run (config + exact scripts/data), results, and status, so work is resumable and auditable.

---

## 2026-08-06 — Borrowing-enabled extra runs have durable full-checkpoint recovery

Three high-priority borrowing jobs were submitted: 12B Medium
`job_d9qhmp5cufhjo4csicv0`, E4B Medium `job_d9qhmp55herl6oiqnf6g`, and the
four-GPU E2B Easy continuation `job_d9qhpj55herl6oiqnf70`. All three were
accepted and initially `QUEUED` with no worker pod. Dedicated tmux supervisors
monitor pod phases and relaunch a failed/preempted attempt.

Every new step-20 checkpoint is uploaded to a run-specific S3 prefix with
model, Adam, scheduler/RNG extras, and `data.pt`; a completion marker is written
last and only manifest-complete checkpoints are restored. The old E2B Easy
step-200 full checkpoint was lost with its pod's `/tmp`, and the HF export is
weights-only, so its first extra-100-step attempt necessarily uses fresh Adam
plus a deterministic 200-batch dataloader fast-forward. Subsequent resumes are
exact after the first new full checkpoint. Focused checkpoint/recovery tests
pass (`4 passed`) and Ruff passes.

---

## 2026-08-05 — Repaired 12B production startup and step 1 passed

Replacement 12B production job `job_d9pibql5heri67n90crg` waited on its pinned
guaranteed-quota H100 node until 15:48 UTC, then started with zero pod restarts.
All eight vLLM servers initialized and the repaired text-only FSDP-to-vLLM sync
passed the exact point where the predecessor failed on
`embed_vision.pos_embedding`. W&B run `8t55mm42` started and the full initial
validation completed at 18.7375% in-distribution, 46.7887% GSM8K, and 16.7500%
MATH500.

The first real 12B optimizer update then completed in 824.7 seconds. Gradient
norm was finite at 12.94; token-TIS truncation was `9.9419e-06`; PPO ratio
metrics were present; and reward groups were 40/64 mixed, 24/64 all zero, and
0/64 all one. Generation took 90.8 seconds, old-log-probability computation
87.0 seconds, actor update 640.3 seconds, and weight synchronization 3.2
seconds. The repaired 12B verl path is therefore proven through step 1; its
required step-20 checkpoint/export remains under continuous monitoring.

Step 2 subsequently completed in 793.1 seconds and included an actual
8,192-token response. Gradient norm remained finite at 13.42, TIS truncation
was `8.3494e-06`, PPO ratio clipping was nonzero, and the pod still had zero
restarts. Reward groups were 35/64 mixed, 29/64 all zero, and 0/64 all one.

---

## 2026-08-05 — E4B production passed step-20 checkpoint/export

Replacement E4B production job `job_d9pht75cufhik662291g` completed step 20,
saved `global_step_20`, uploaded a new full HF export, completed the scheduled
in-distribution/GSM8K/MATH500 validation, and continued through step 21 with
zero pod restarts. The new HF commit is
`f5993c209242fca83cf42d271c9122763f18da78`; its `step_000020` tree contains
seven files totaling 17,366,959,359 bytes, including the 17,334,775,284-byte
`model.safetensors`. Step-20 validation reached 22.425% in-distribution,
50.634% GSM8K, and 20.425% MATH500. Token TIS, PPO ratio, and reward-group
metrics remained present; step 20 reported 37/64 mixed groups, 24/64 all zero,
and 3/64 all one.

---

## 2026-08-05 — Packed E2B production passed step-20 checkpoint/export

Both four-GPU children of `job_d9phvft5herivi4ihheg` completed the scheduled
step-10 in-distribution/GSM8K/MATH500 sweep and continued training. Easy changed
from 5.4125% to 5.7625% in-distribution, 7.7765% to 8.0797% on GSM8K, and
4.5625% to 5.0125% on MATH500. Medium changed from 3.2125% to 3.4000%
in-distribution and 8.1014% to 8.5888% on GSM8K; MATH500 was effectively flat
at 4.6875% versus 4.6500%.

Medium step 8 logged a non-finite pre-clip gradient norm while every loss,
rollout-correction, PPO-ratio, and reward metric remained finite. The active
FSDP engine's anomaly guard zeroed the gradients and skipped that optimizer
update, so weights were not corrupted. Steps 9 and 10 then completed with
finite gradient norms 5.19 and 5.29, respectively, including an 8,192-token
response at step 9. Easy continued through step 11 with finite metrics.

Both children later completed step 20, ran the second scheduled validation,
saved the local checkpoint, and uploaded a new full HF export. Easy's new commit
is `9ce950247f3e9e56df942f397b668a87a8dccb60`; Medium's is
`f329ac40b18154b4f46171145e31cb19eed008f4`. Each `step_000020` export has seven
files totaling 11,084,114,210 bytes, including the 11,051,930,326-byte
`model.safetensors` plus tokenizer, processor, config, generation config, and
chat template. Easy step-20 validation reached 6.3375% in-distribution, 10.0292%
GSM8K, and 5.8750% MATH500. Medium reached 3.7625%, 8.9028%, and 5.2125%,
respectively. Easy continued through step 21 after the save.

Both packed children subsequently reached step 40 and produced another full
HF export. Easy commit `a87df6d5c09abca1795fbbaf9089748b317b59f3`
landed at 15:34 UTC and Medium commit
`be22f986c8f7197521bb32177a599a77690da879` at 15:41 UTC. Each
`step_000040` tree contains seven files totaling 11,084,114,210 bytes, including
the 11,051,930,326-byte `model.safetensors`.

---

## 2026-08-05 — 12B text-only vLLM weight sync skips unused vision branch

The first replacement 12B job reached vLLM initialization but failed its
initial FSDP-to-vLLM synchronization because the training model emitted
`embed_vision.pos_embedding`, which vLLM's Gemma 4 Unified model intentionally
does not expose. The rollout is text-only, so the existing optional multimodal
reload filter now recognizes Gemma 4's `embed_vision` naming, and the 12B matrix
profile enables that filter explicitly. Language-model weights remain synced.
The failed job was `job_d9phq3t5herivi4ihhe0`; it was canceled successfully and
the tested replacement is `job_d9pibql5heri67n90crg` (borrowing disabled, high
priority). The focused vLLM-filter and matrix suite passes 22/22.

At 2026-08-05 13:46 UTC the replacement created worker pod
`g4-12b-easy-med-full-jasonwei-2026080-bsyyl-workers-0-0-vldn4`. The pod is
still `Pending` with zero restarts and no container startup: it is pinned to a
currently occupied H100 node, while the autoscaler reports max node-group size.
This is a capacity wait, not a recurrence of the weight-sync failure.

---

## 2026-08-05 — Replacement E4B step 1 confirms throughput improvement

E4B production job `job_d9pht75cufhik662291g` completed step 1 in 559.3
seconds, down from roughly 830–850 seconds/step on the preceding production
run. Generation took 26.2 seconds, old-log-probability computation 69.6 seconds,
and the actor update 454.8 seconds. The step processed 1,024 responses with
finite loss and gradient norm. TIS truncation was `1.51798e-05`, PPO ratio
metrics were present, and the reward-group split was 35/64 mixed, 29/64 all
zero, and 0/64 all one. After three completed steps, the new run's median is
562.9 seconds/step versus 837.9 seconds over 38 completed steps in its exact-name
predecessor W&B run `xlc5hevs`, a 32.8% step-time reduction.

---

## 2026-08-05 — Packed E2B microbatch-8 production step 1 passed

The Easy child of packed production job `job_d9phvft5herivi4ihheg` completed a
real four-GPU step 1 in 311.9 seconds with microbatch 8 and the 12,288-token
adaptive ceiling. It included an 8,192-token response, finite gradient norm
`4.90`, zero TIS truncation, and PPO clip fraction `2.96e-06`. Generation took
34.7 seconds, old-log-probability computation 37.1 seconds, and actor update
232.0 seconds. Reward groups were 18/64 mixed, 46/64 all zero, and 0/64 all
one. This is the production-scale confirmation of the local throughput canary.
The Medium child independently passed step 1 in 310.6 seconds with an 8,192-
token response, finite gradient norm `13.32`, zero TIS truncation, and a group
split of 10/64 mixed, 54/64 all zero, and 0/64 all one. The matched production
comparisons now establish the throughput gain beyond the one-step canary: Easy
has a 293.5-second median over its first six steps versus 893.3 seconds over 38
steps in exact-name predecessor `zcuagr1z` (67.1% lower), while Medium has a
299.6-second median over its first five steps versus 842.6 seconds over 38 steps
in exact-name predecessor `bsfow2oe` (64.4% lower).

---

## 2026-08-05 — E2B microbatch-8 / 12,288-token adaptive canary passed

A four-GPU Gemma 4 E2B canary completed a full optimizer update with compiled
vLLM rollout, an 8,192-token response limit, microbatch 8, and a 12,288 padded-
token adaptive ceiling. It wrote the complete `global_step_1` checkpoint and
ended with `RUN_DONE rc=0`. Step time was 140.0 seconds including 54.5 seconds
of checkpointing; generation took 46.7 seconds, actor update 31.0 seconds, and
old-log-probability computation 4.17 seconds. Token TIS, PPO ratio clipping, and
reward-group count/percentage metrics were present and finite. The E2B packed
production profile now uses microbatch 8 and the 12,288-token ceiling.

---

## 2026-08-05 — Math reward uses math_verify OR bounded Miles SymPy

The strict last-box reward now marks an answer correct when either Hugging Face
`math_verify` or the Miles/DeepScaleR SymPy equivalence grader accepts the same
extracted final box. `math_verify` runs first; SymPy runs only after a miss and
has a five-second worker-local timeout so pathological simplification cannot
stall the reward pool. A stratified 1,800-trace audit found 11 grader
disagreements, all manually valid equivalents; the OR recovered the one
SymPy-only case (`(15)x - 80.` versus `15x - 80`) while retaining the ten
`math_verify` acceptances. The verl focused suite passes 19/19 and the NeMo-RL
parity gate passes 8/8.

---

## 2026-08-05 — Math reward now grades the last well-formed box

The boxed-only `math_verify` path now grades the last balanced `\\boxed{}` or
`\\fbox{}` in a completion. A completion with no well-formed box still scores
zero, but an earlier boxed answer no longer invalidates a later correction. The
majority-vote prediction extractor and the NeMo-RL parity port use the same
selection rule. Historical results documented below retain the exactly-one-box
semantics used when those evaluations were run.

---

## 2026-08-05 — Production image repaired; E4B KV-cache failure fixed selectively

**Image repair.** The production image had excluded `verl/third_party` because
the Docker ignore rule `third_party` matched both the root dependency checkout
and the package directory. The rule is now scoped to `third_party/**`, and the
Dockerfile asserts that `verl/third_party/vllm/__init__.py` exists. A direct
container import verified the packaged module.

**E4B failure and recovery.** E4B job `job_d9p7utt5heri8vqpv6v0` failed at
vLLM startup because a 512 MiB explicit KV cache was too small for the
12,288-token window; vLLM required about 0.66 GiB. E4B and 12B now use a 1 GiB
explicit cache. The selectively relaunched E4B job is
`job_d9p8f4d5heri8vqpv720`; it passed the previous crash point, initialized all
eight vLLM servers and the agent-loop manager, and entered the 200-step trainer.
The focused matrix suite passes (`7 passed`).

**Current jobs.** Healthy jobs were not reset. The final monitored set is 12B
`job_d9p7ntd5heri67n90cd0` (`IN_PROGRESS`, worker capacity-pending), E2B packed
4+4 `job_d9p87355heri67n90ceg` (`IN_PROGRESS`, worker running), and E4B
`job_d9p8f4d5heri8vqpv720` (`IN_PROGRESS`, worker running). All retain EKS,
high priority, borrowing disabled, 200 steps, validation every 10 steps, and
checkpoint/Hugging Face export every 20 steps.

---

## 2026-08-03 — Local 8K gate passed; three production jobs launched

**Local gate.** Gemma 4 E2B Easy-10k completed ten real optimizer steps on four
H100s with the production 4,096/8,192 prompt/response window, microbatch 2, 64
prompts x16 responses, and explicit 512 MiB vLLM KV cache. All ten steps had
finite loss, gradient, and reward metrics; steps 1, 3, 4, 5, 8, 9, and 10 each
included at least one genuine 8,192-token response. Step 10 also completed the
full in-distribution/GSM8K/MATH500 validation sweep. The complete step-10 FSDP,
optimizer, extra-state, and Hugging Face checkpoint was written, the container
exited zero, and the log ended with `RUN_DONE rc=0`.

**Production topology.** Six logical 200-step runs resolve to three
borrowing-false, high-priority, full-node EKS jobs. One 12B job runs easy then
medium sequentially; one E4B job does the same; the E2B job runs easy and
medium concurrently on disjoint four-GPU halves. Validation remains every ten
steps and checkpoints/HF pushes every twenty. The three production jobs were
submitted after the local gate passed and were initially queued with no worker
pods admitted:

- 12B sequential: `job_d9o8g16rkvumpnejdjgg`
- E4B sequential: `job_d9o8hcurkvum8ubf6hn0`
- E2B packed 4+4: `job_d9o8i4ihu8ojb1b2cua0`

The previous remote smoke jobs were canceled successfully.

**Queue reset.** At the user's request on 2026-08-03, the three production jobs
above were canceled and immediately relaunched to move the matrix to the back
of the queue. Every cancellation returned `Result: True`; the briefly admitted
12B worker was still `Pending` and disappeared before relaunch. Replacement
jobs retain EKS, high priority, borrowing disabled, one full node per topology,
200 steps, validation every 10 steps, and save/export every 20 steps:

- 12B sequential: `job_d9oegh2hu8ojnqcvi9u0`
- E4B sequential: `job_d9oeimahu8ojnqcvi9ug`
- E2B packed 4+4: `job_d9oekfahu8ojnqcvi9v0`

All three replacements were initially `QUEUED` with no worker pods.

**Second queue reset.** At the user's request on 2026-08-04, those three
queued replacements were canceled and immediately resubmitted once more. All
three old job records now report `CANCELED`. The new jobs preserve the same
EKS/high-priority/borrowing-disabled production configuration and were
verified `QUEUED` after submission:

- 12B sequential: `job_d9ol3655hergioua77u0`
- E4B sequential: `job_d9ol47t5hergioua77ug`
- E2B packed 4+4: `job_d9ol4ut5hergioua77v0`

**Entry points.** The sequential 12B/E4B wrapper is
`scale_train/run_gemma4_easy_medium_sequential.sh`; E2B remains in
`scale_train/run_gemma4_e2b_easy_medium_packed.sh`. The matrix launcher resolves
all six runs into the three intended jobs, and focused tests pass.

---

## 2026-08-02 — Gemma 4 Easy-10k/Medium-20k six-run RL matrix prepared

**Scope.** Prepared but did not submit six seed-42 DAPO runs: Gemma 4 12B, E4B,
and E2B on both DeepScaleR Easy-10k and Medium-20k. The 12B/E4B runs reserve one
eight-GPU node each. The two E2B runs share one full node with four GPUs each,
isolated CUDA visibility, Ray directories, vLLM port ranges, caches, W&B dirs,
and checkpoint roots. This is five ScaleTrain jobs containing six logical runs.

**Validation contract.** Each run evaluates its matching 500-question
in-distribution validation set at x16 (8,000 rows), GSM8K's 1,319 questions at
x7 (9,233 rows), and MATH-500's 500 questions at x16 (8,000 rows), before
training and every ten steps. All repeated rows carry stable per-question UIDs.
The pinned preparation completed end to end and verified train/validation
non-overlap, exact row counts, uniform repeat counts, and a SHA-256 manifest.

**Training contract.** The matrix retains the current 8k Gemma 4 recipe:
64 prompts x16 responses, mini-batch 32, LR 1e-6 with 20 warmup steps, 4,096
prompt + 8,192 response tokens, 2,048-token soft-overlong buffer, strict boxed
reward, save every 25 steps, and a 500-step production horizon. 12B/E4B use
microbatch 1; E2B uses the validated configured microbatch 2 with adaptive
singleton splitting above a 4,096 padded-token ceiling. All use FSDP2 CPU
offload policy. The new 12B path selects `Gemma4UnifiedTextDecoderLayer` and is
gated behind a one-step canary because it has not previously trained in this
repository.

**Entry points.** See `GEMMA4_DIFFICULTY_RL_RUNS.md`,
`scale_train/launch_gemma4_difficulty_matrix.py`, and
`scale_train/run_gemma4_e2b_easy_medium_packed.sh`. The default launcher action
is dry-run; `--launch` is required for submission. Focused tests pass, and all
five smoke-job commands resolve against the full-node EKS build configuration.

---

## 2026-07-31 — Finite Gemma 4 gradient spikes accepted; deterministic-cuDNN experiment withdrawn

**Operator decision.** Large but finite Gemma 4 pre-clip gradient norms are not a stop condition for
this experiment. The scoped `torch.backends.cudnn.deterministic=True` change and its report-v5
contract were withdrawn. Its clean eight-GPU gate produced a finite norm of `127.41`; this is now
retained as an observed Gemma 4 training value rather than classified as numerical corruption.

**Production contract.** Training returns to the normal verified stack from `9a3e103f`: cuDNN SDPA
for training target parity, non-cuDNN SDPA for interleaved validation, singleton microbatches, and
the 4,096 padded-token ceiling. The finite gradient-magnitude ceiling is disabled by default while
gradient norms remain logged. Non-finite loss, non-finite gradients, invalid probability masses,
crashes, stalls, checkpoint failures, and upload failures still fail closed and are monitored by the
supervisor. A fresh clean receipt is required because production receipts bind the repository
commit and audited source hashes.

---

## 2026-07-31 — Validation-interleaved cuDNN backward anomaly isolated and fixed

**Failure beyond singleton batching.** A clean-HEAD three-step W&B smoke (`5b475fb2`) used the
audited singleton-microbatch contract and still reached gradient norm `74.66` on step 3 after
validation ran at steps 0, 1, and 2. The fail-closed gate stopped before the optimizer update; no
checkpoint from that run is resumable.

**Isolation.** The same exact three train batches with only step-0 and final validation passed in
W&B run `0ea93c05` at gradient norms `14.78`, `21.02`, and `12.61`, proving that the third batch and
optimizer sequence are safe without an intervening validation forward. Disabling cuDNN SDPA for
all forwards also kept gradients finite (`14.73`, `18.63`, `23.20`) but failed target parity:
weighted log-probability drift `0.01826` and sampled-token p95 `0.09447`.

**Fix and gate.** Training retains cuDNN SDPA so it matches the immutable BF16 target overlay;
in-process validation uses the non-cuDNN SDPA backend so it cannot contaminate the next backward.
The audit now executes the real optimizer/scheduler sequence and exact validation cadence. The
mixed-backend diagnostic passed with exact ordered top-128 support, weighted drift `0.000168`,
sampled-token p95 `0.000864`, and sequential gradient norms `14.78`, `21.20`, `14.95`. A clean-HEAD
receipt and fresh three-step W&B checkpoint smoke remain required before production relaunch.

---

## 2026-07-31 — E2B-to-E4B paired-microbatch anomaly isolated; singleton production contract

**Failure.** Fresh W&B smoke `b50c8b0e` used the BF16-forward/FP32-master stack and the previously
authorized microbatch-2/5,120-padded-token policy. Gradient norms were `13.62`, `20.16`, then
`356.42`; layers 0-1 `per_layer_input_gate` dominated the third step. The supervisor stopped before
the anomalous optimizer update, so this run is not resumable.

**Isolation.** An exact replay of the same first three deterministic train batches with per-
microbatch diagnostics reproduced the third-batch failure at gradient norm `62.96`. The different
magnitude establishes nondeterminism in the bad backward path. Microbatch 1 on the same batches
produced `14.84`, `21.26`, and `16.10`, completed validation at loss `0.20946`, and saved a complete
step-3 checkpoint. Disabling activation checkpointing or vocabulary-projection chunk checkpointing
did not repair paired batches; the evidence points to BF16 cuDNN SDPA backward when two padded
sequences share a microbatch, not the earlier NeMoRL shared-KV/use-cache bug.

**Fix and gate.** Production now requires microbatch 1, keeps cuDNN SDPA for target parity, and uses
a defensive 4,096 padded-token ceiling. The training-engine receipt is upgraded to bind the exact
runtime contract and exercise all first three seed-42 batches (384 distinct rows) rather than only
the first batch. Focused audit/verifier/launcher tests pass (`34 passed`), and the broader Gemma 4
data, launcher, supervisor, checkpoint, batching, gradient-diagnostic, and engine suite passes
(`252 passed`). A clean eight-GPU receipt and fresh three-step W&B smoke are still required before
launching the 750-step run.

---

## 2026-07-31 — Gemma 4 source traces complete; cross-engine target policy and preparation tooling

**Scope.** Preparation only for the two Gemma 4 distillation-vs.-RL lines. No 750-step
distillation, benchmark production, post-distillation RL, or model training run was started. The
canonical status and ambiguity ledger are in
[`GEMMA4_DISTILL_VS_RL_EXPERIMENTS.md`](GEMMA4_DISTILL_VS_RL_EXPERIMENTS.md).

**Complete local vLLM source bundles.** Both directions contain 48,615 train plus 1,000 validation
rows (1,216 train and 25 validation shards), five responses per registered question unit, exact
stored token IDs/response masks, and top-128 full-vocabulary-normalized vLLM targets:

| Direction | Dataset index | Experiment | Response tokens | Local root |
|---|---|---|---:|---|
| E4B-RL step 100 → E2B | `8b5712e0f5dea3388340a9bc91a6ceee40ff2ff990e66b7238090e240daeda6c` | `b515965fcdc31caebae1e71cf696731f4271182bdaca8af8326888d0b196af92` | 14,049,865 | `/lambda/nfs/Jason-scale/rl-distill-traces/gemma4-e4b-rl100-topk128` |
| E2B base → E4B | `d170166abad89588880f5c0a9eac43006f9a32bbe27cec28d7eb97c65288dbcc` | `f77e9dd611371030cc7752d4f4d0c92d4890448d9fe48594efdf3e223a5a409e` | 8,257,057 | `/lambda/nfs/Jason-scale/rl-distill-traces/gemma4-e2b-base-topk128` |

Both indexes preserve the historical 9,723-row train roster, 200 validation UIDs, 180 duplicate-text
train occurrences, and seven train/validation text overlaps. Clean evaluation must also report the
193-question non-overlap subset. Sampling is temperature/top-p 1.0, top-k disabled, with
4,096/8,192/12,288 prompt/response/context caps. The tokenizer and template hashes are
`f3ab24e73e9022f7b8d77113f543debd3779ef1e96c6452c68aaa9f3e6b81d17` and
`27b8801d8b61a413a9bb3b54b6f55e16217eff3e55f7c560377c8a162dd63c1c`.

**HF publication status.** Both source bundles are public and independently verified at immutable
commits:

| Direction | Public repository | Commit | Dataset index |
|---|---|---|---|
| E4B-RL step 100 → E2B | `JWei05/gemma4-e4b-rl100-topk128-traces` | `2b6e49a0a456ee9d67b16a1dc61785562bee90c9` | `8b5712e0f5dea3388340a9bc91a6ceee40ff2ff990e66b7238090e240daeda6c` |
| E2B base → E4B | `JWei05/gemma4-e2b-base-topk128-traces` | `e32aaa02681ae83b3d7256b1b155c9084da2f289` | `d170166abad89588880f5c0a9eac43006f9a32bbe27cec28d7eb97c65288dbcc` |

Each commit contains exactly 2,485 registered files. Independent remote verification compared every
path and size, LFS SHA-256 for every Parquet, and Git blob SHA-1 for every non-LFS file. The initial
private E2B upload had previously failed closed with `Private repository storage limit reached`, and
the corresponding early E4B upload-result log was empty; those facts remain historical rather than
current publication status.

**Cross-engine audit.** Expanded vLLM-versus-unsharded-HF diagnostics used exact stored sequences and
covered 32 traces per teacher: 1,870 E2B positions and 1,975 E4B positions. The two unsharded-HF
paths (native Gemma 4 forward and manual projection) had max absolute error 0. Tie-safe top-1 was 0.99572/0.99747,
top-128 overlap 0.98504/0.97654, weighted absolute log-probability delta mean 0.017996/0.010180,
probability-L1 mean 0.017343/0.010221, and sampled-token delta p95 0.09419/0.06082 for E2B/E4B.
Both pass the calibrated diagnostic thresholds, but the engines are not bit-identical.

**Target policy.** Keep each vLLM bundle immutable as the generation record. The intended primary
distillation path preserves the exact stored IDs and derives a separate unsharded-HF top-128 target overlay
with full-sequence teacher forcing, BF16+SDPA, causal predecessor scoring, Gemma 4 softcapping, and
FP32 full-vocabulary normalization. This remains offline precomputation: no online teacher and no
re-tokenization. A real FSDP2-engine target audit is still required before claiming numerical
equivalence to the distributed training forward.

**Prepared code.** Added the immutable-index-bound cross-engine audit, a disjoint resumable overlay
rescorer with exact chunked/native parity mode, focused CPU tests, and guarded operator documentation.
Added `preflight_gemma4_training_topk_overlay.py` and schema-based launcher routing: an overlay is
accepted only with its exact source index, parity receipt, target/student identities, and verified
one-to-one shard/row/token binding. The combined source/overlay/launcher suite passed 48 tests. The
trace generator now returns a distinct status for deterministic schema failures, and the supervisor
refuses identical-seed retries. The source-bundle uploader now commits only content-verified staged
snapshots to an otherwise empty destination branch, pins the observed parent commit, defaults to
private visibility, and suppresses raw provider exceptions so tokens cannot reappear in formatted
tracebacks. Bulk GPU rescoring, overlay generation/finalization, and overlay upload were not run.

**Status.** Both immutable source bundles are publicly published and remotely verified. Repository
review/test/commit/push remains preparation only. Integrated GPU audit, bulk rescore, student smoke,
both experimental lines, production evaluation, and post-distillation RL remain held.

---

## 2026-07-30 (fixed run, 100 steps) — verl val-curve parity EXACT; truncation bimodality does NOT reproduce

**Run.** `e1du1oyu` (`nemorl-dapo-gemma4-e2b-pt-DeepScaleR-4of4strict-seed42-8k-local4g-v2`), the
relaunch on the fixed stack (see ROOT CAUSE entry below), local 4×H100, ~4.5–6 min/step, ckpts at
25/50/75/100.

**Validation parity vs verl (`recbw9dcxso`) — every PARITY_CHECKLIST milestone matched:**

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nemo (fixed) | 5.91 | 6.47 | 7.28 | 10.78 | 13.50 | 14.22 | 13.56 | 14.97 | 15.28 | 16.69 | **16.78** |
| verl ref | 6.16 | 6.00 | 6.78 | 10.0 | 12.59 | 14.19 | — | — | — | — | **16.59** |

Step-1 `probs_ratio` 1.000020; grad_norm sane throughout. The corrupted run (`ihdn67bj`) had
flatlined at 5.78 by step 30 — the fix is what unlocked learning, and the two frameworks now
agree to within mean@16 noise across the entire curve.

**Bimodality verdict (the repro's purpose), 99 training steps:** NeMo-RL clean steps (n=55)
grad_norm 0.25–3.81; truncated steps (n=44) 0.44–15.0. Spikes >3 concentrate on truncated steps
(7 vs 2), so a mild truncation-keyed heavy tail exists — but the regimes **overlap**, vs verl's
disjoint clean 1.3–6.8 / truncated 83–3108. `compare_grad_norms.py`: "regimes overlap — differs
from verl". With 44 truncated steps and zero events near verl's regime, on a run with exact val
parity, the evidence points to verl's extreme bimodality being **framework-specific behavior on
truncated sequences, not a property of the DAPO algorithm or the data**. Caveats: NeMo logs the
last of 2 mini-batch grad norms (hides ~half the candidate events, not all 44); vLLM sampler
numerics differ; distributional comparison only. (Supersedes the retracted `ihdn67bj` verdict —
same conclusion, but that one was measured on a corrupted objective and inadmissible.)

**Status.** Run continues open-ended past step 100 for curve extension; next verl-side step is
deciding whether to hunt the verl-only mechanism (e.g. rmpad/padded-path logit handling on at-cap
sequences) or accept the framework-specific classification.

---

## 2026-07-30 — NeMo-RL repro running LOCALLY on 4×H100 (gate PASSED 0.0591; 2 config/doc fixes)

**Goal.** Stand up the gemma-4 E2B NeMo-RL repro on a local 4×H100 box (192-222-53-241, driver 580 /
CUDA 13 — no cuda-compat needed) and get the full training run going, per the nemo_rl_repro README.

**Setup (fresh box, ~50 min end-to-end).** sudo apt cuda-toolkit-13-0 + cudnn9/libcudnn9-dev-cuda-13 +
librdmacm-dev; Kitware cmake 4.0.3; uv 0.12. Driver venv `/tmp/nemo-rl-venv`
(`UV_PROJECT_ENVIRONMENT=... uv sync --locked --extra automodel --no-install-package deep-ep`, torch
2.11.0+cu130, TE 2.15.0), worker venvs `/tmp/nemo-rl-worker-venvs`, HF cache `/tmp/hf-home` (E2B
prefetched), data via `prepare_deepscaler_4of4strict_rl_data.sh` → `/tmp/verl/data`. Launch wrappers
on the box: `/tmp/nemo-rl-local/{env.sh,build_venv.sh,run_gate.sh,run_train.sh}`. CPU gates re-passed
here: strict reward 6/6, tokenization 5/5 (via the real nemo_rl processor path in the py3.13 venv).

**Fixes (each root-caused).**
1. README prereq gap: `git submodule update --init third_party/nemo-rl` is not enough — the uv sync
   dies on `nemo-gym ... not a workspace member`. Nested submodules required: **`--recursive`**
   (Gym/Automodel/Megatron-Bridge). README updated.
2. `checkpointing.keep_top_k: null` in the committed repro yaml **fails MasterConfig pydantic
   validation** on nemo-rl 5f89b3ae (`CheckpointingConfig.keep_top_k` is `NotRequired[int]` — absent
   OK, explicit null not; the pod gate must have run a pre-commit working copy). Fixed in-config:
   `keep_top_k: 1000000` (keep-all intent, valid int).
3. First 4-GPU gate: val PASSED then the train step **OOM'd** (12.0 GiB fp32 alloc, 11.45 free).
   Root cause chain: at 4 ranks the FSDP shards are 2× the 8-GPU pod run's, and the 12 GiB transient
   is a **full fp32 [12288, 262144] logits tensor in train()** — upstream nemo-rl bug:
   `LossPostProcessor.__call__` (`nemo_rl/models/automodel/train.py`) never forwards `chunk_size` to
   `prepare_loss_input`, so `policy.logprob_chunk_size: 4096` is silently ignored in the training
   loss path (honored only in the logprob pass). Zero-edit fix: **`policy.dtensor_cfg.cpu_offload=true`**
   (CLI override in the local wrappers; gradient-identical, exercised in their own grpo-deepscaler
   recipes) — frees the static shard memory so the transient fits.

**Gate (4 GPUs, full 3200-sample val): PASSED.** step-0 `validation/accuracy = 0.0591` ∈ [0.045,
0.075] (pod 0.0550, old local 2-GPU 0.0508); reference policy auto-skipped; train step 1 completed:
loss 0.2581, avg reward 0.0576, mean gen length 194.7 tok, 346 s/step (81% policy_training — the
cpu_offload tax; generation 41 s).

**Run `ihdn67bj` (36 steps) — KILLED: training was silently corrupted from step 1** (see the
follow-up entry below). Its val curve (5.91→6.22→6.25→5.78 at 0/10/20/30 vs verl 6.16→6.00→6.78→10.0)
is a *no-learning* baseline, not a comparison point; step-25 ckpt is garbage-trained (discardable).

---

## 2026-07-30 (later) — ROOT CAUSE: act-ckpt + gemma-4 KV-shared layers = garbage training forward

**Symptom (user-flagged).** `train/probs_ratio` bulk 0.05–0.45 with min 0 / max up to 1.7e6, and
`train/probs_ratio_clamped` pinned at ~0.80 (the lower PPO clip bound) — deterministic from step 1
(gate + full run logged identical values), while `gen_kl_error`/`policy_kl_error` ≈ 8e-4 (tiny).
Since those two metrics compare prev↔generation (NOT curr), and the step-1 train-data dump shows
prev↔gen agreeing to −0.0008 ± 0.04 nats over 199k tokens, the **training forward (`curr`) was the
corrupted side** — and the loss consumes the same `ratios` tensor, so training optimized garbage.
(Step-1 LR is 1e-14 under LinearLR warmup: both mini-batches are at numerically identical weights,
so real policy movement is excluded.)

**Root cause (proven by offline CPU repro, `/tmp/nemo-rl-local/repro_curr_vs_prev.py`).** gemma-4
E2B has KV-shared layers (`num_kv_shared_layers > 0`). nemo-rl PR #2224 makes every trainer forward
request `use_cache=True` because on transformers < 5.5.2 the shared layers otherwise fall back to
untrained K/V projections (upstream **Automodel#1705** — the exact same bug, root-caused by NVIDIA in
April). But HF transformers 5.5.0 **force-disables the cache in grad-mode when gradient checkpointing
is on** (our worker logs: "`use_cache=True` is incompatible with gradient checkpointing. Setting
`use_cache=False`"). Repro on identical base weights, dumped step-1 sequence: use_cache=False forward
diverges from the dumped logprob-pass values by **−10.4 nats mean** (98.9% of loss ratios < 0.8, min
0 — the exact wandb signature); flipping only use_cache=True matches to **−0.0001 ± 0.04** (ratios
1.0007). So `activation_checkpointing: true` (the 07-29 pod OOM "fix") silently poisons ALL gemma-4
E2B training on the locked transformers 5.5.0; no pod run ever logged a train step, so it went unseen.
**The 8-GPU ScaleTrain config/image has the same poison — do not launch pods until fixed there.**

**Fix stack (local, all verified):**
1. transformers **5.5.0 → 5.5.4** in the driver + DTensor-policy-worker venvs (within nemo-rl's
   `>=5.5.0,<5.6.0` pin; HF #45312 in ≥5.5.2 dissociates KV sharing from the cache — nemo-rl's own
   TODO endorses this). Repro on 5.5.4: use_cache=False now matches (−0.0001 ± 0.04); tokenization
   parity re-passed 5/5. NOTE: a bare worker-venv rebuild reverts to the locked 5.5.0 — reapply.
2. Act-ckpt must stay ON for memory (without recompute, activations at 12288 seq = 75.1 GiB → OOM
   even with cpu_offload; measured). Now safe on 5.5.4.
3. **`nemo_rl_repro/sitecustomize.py` (new adapter file, zero vendored edits):** on ≥ 5.5.2 it
   disables nemo-rl's `_needs_kv_cache_for_shared_layers` workaround via a lazy meta-path hook
   (loads in every Ray actor through the PYTHONPATH the launcher already exports; also copied into
   the worker venv site-packages). Needed because the workaround's use_cache=True path CRASHES on
   5.5.4 (`modeling_gemma4.py` `shared_kv_states` KeyError 13 in `get_logprobs`); with the patch all
   passes run use_cache=False, which is now correct.
4. Also fixed en route: `keep_top_k: null` → int (pydantic), cpu_offload=true (4-rank static memory),
   and documented the (memory-only, numerically-inert) `LossPostProcessor` chunk_size gap — upstream
   fixed the Megatron twin in #2871/#2872 but never the automodel side.

**Cross-check vs upstream/PR #2224 (user question).** Our pin (5f89b3ae, 2026-07-27) already contains
PR #2224 (merged 2026-06-14) + follow-ups (#3297, #3124); config matches every gemma-4 correctness
knob in their E2B recipe; nothing in `5f89b3ae..origin/main` touches the affected files. NVIDIA's
recipe never hit this because it runs 4096-token sequences with act-ckpt OFF; their known-good recipe
also runs TIS on (we deliberately match verl: off).

**Two more landmines found while re-gating on 5.5.2+ (each root-caused, both defused by
`nemo_rl_repro/sitecustomize.py`, zero vendored edits):**
- 5.5.4's new KV-sharing passes anchor K/V between layers by MUTATING a plain `shared_kv_states`
  dict kwarg. **FSDP2's pre-forward input cast (`MixedPrecisionPolicy.cast_forward_inputs=True`
  default) rebuilds kwargs containers** (`_apply_to_tensors`), so each fully-sharded decoder layer
  receives a fresh COPY — anchors write into copies, shared layers read empty dicts →
  `modeling_gemma4.py` KeyError 13 in get_logprobs, with either use_cache mode, custom or stock
  class. Fix: force `cast_forward_inputs=False` (value-wise a no-op: model/activations already
  bf16; fp32 output cast gated separately on output_dtype).
- The Automodel custom gemma4 class predates the 5.5.2 mechanism entirely →
  `+policy.dtensor_cfg.automodel_kwargs.force_hf=true` (stock HF class; refit/generation validated
  through the gate). Dead ends tried and documented: TP=2 (E2B forward breaks under the dtensor TP
  plan: mixed Tensor/DTensor aten.where), act-ckpt off (75.1 GiB activations OOM).

**GATE PASSED end-to-end (run `4ndtkcry`): step-1 `probs_ratio = 1.000017`,
`probs_ratio_clamped = 1.000017` (was pinned 0.80), ratio min/max 0.795/1.298 (was 0/1.7e6),
grad_norm 0.74, val 0.0600 ∈ [0.045, 0.075], rc=0.** Full run relaunched as wandb
`DAPO/nemorl-dapo-gemma4-e2b-pt-DeepScaleR-4of4strict-seed42-8k-local4g-v2` (ckpts/logs
`/tmp/verl/{ckpts,logs}/...-local4g-v2`). Final local recipe = locked stack + transformers 5.5.4
(driver + policy-worker venvs; reverts on worker-venv rebuild — reapply) + sitecustomize (both
patches) + act-ckpt ON + cpu_offload ON + force_hf. verl-comparison caveats unchanged (vLLM venv
untouched; training math identical). Upstream-worthy reports: Automodel#1705-class bug reachable
via nemo-rl on the LOCKED stack whenever act-ckpt is on for E2B/E4B; the FSDP2
cast_forward_inputs dict-copy breaking 5.5.2+ gemma-4 KV sharing; `LossPostProcessor` chunk_size
gap (automodel twin of #2871).

---

## 2026-07-29 — NeMo-RL repro on ScaleTrain: parity gate PASSED at scale; baked image; OOM→act-ckpt

**Goal.** Run the 10-step NeMo-RL comparison (`MAX_STEPS=10`) on ScaleTrain to collect cross-framework
grad norms vs the verl reference (recovery `recbw9dcxso`, bimodal law: clean steps 1.3–6.8, ≥1
truncated response → 83–3108). Directive: keep relaunching, borrowing-true + borrowing-false pair.

**Headline result.** On 8×H100 with the full 3200-sample val, the step-0 GO/NO-GO gate returned
**validation/accuracy = 0.0550** — inside the verl baseline band [0.045, 0.075] and matching the local
2-GPU gate (0.0508). The whole parity stack (12-shot chat template byte-identical tokenization, strict
boxed-only reward port, PT model, sampling params, stop strings) is now validated at production scale.

**Pod-attempt ledger** (each fixed one blocker; all fixes in `scale_train/run_nemorl_gemma4_e2b_repro.sh`):
1. `job_d9kofjhq` — TE build: `cudnn.h` not found (pip cudnn wheel lands mid-`uv sync`) → apt
   `libcudnn9-dev-cuda-13` before the sync.
2. (launch killed) — docker context swept a stale multi-GB `third_party/nemo-rl/venvs/` from an early
   local run → dockerignored + deleted.
3. (launcher crash) — my `rm -rf` of that dir raced the launcher's repo staging copy → sequenced.
4. `job_d9kpfd1q` — data prep died on `python3: command not found` (pod login shell resets PATH; image
   venv dropped) and was non-fatal → gate crashed 1 h later on missing parquet. Fix: absolute venv
   python + hard fail-fast (`DATA_PREP_OK` marker).
5. `job_d9kqj9hq` — **GATE_PASS 0.0550**, then the first training step **OOM'd** (71.8 GiB PyTorch
   alloc / 80 GiB H100). Root cause: verl-parity `max_total_sequence_length=12288` is 3× NVIDIA's own
   E2B recipe (4096) with activation recompute off. Fix (parity-neutral, gradient-identical):
   `policy.dtensor_cfg.activation_checkpointing: true` (their 26B/31B recipes use it) +
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
6. Current pair on the fix: `job_d9l2cgpq` (borrow-on) + `job_d9l2huer` (borrow-off), queued 16:24Z.

**Baked image** (user: no more hour-long pod setup). New `st_config/Dockerfile.nemorl` + build config
`train-rl-distill-nemorl` (78.9 GB): CUDA-13 toolkit/compat/cuDNN-dev, Kitware cmake, driver venv with
TE compiled, and both worker venvs (`VllmGenerationWorker`, `DTensorPolicyWorkerV2`) prefetched to
`/opt/ray_venvs` via their `prefetch_venvs.py` (positional filters; deep-ep needs
`/usr/local/cuda-13.0/lib64/stubs` on `LIBRARY_PATH` in the GPU-less build). Heavy layers depend only
on `third_party/nemo-rl` → cached across launches; run script auto-detects baked artifacts and
fast-skips (startup ~95 → ~35–40 min, dominated by HF download + gate).

**Analysis prep.** `nemo_rl_repro/compare_grad_norms.py` — pulls the newest nemorl wandb run +
`recbw9dcxso`, tabulates `train/grad_norm` vs `train/truncation_rate`, prints a bimodality verdict.
Verified both frameworks log the same quantity (pre-clip global L2 per optimizer step; NeMo-RL:
`nemo_automodel scale_grads_and_clip_grad_norm`, no PP/EP scaling in our dense no-PP case).

**Status.** Attempt-6 pair queued; monitors on job status + pod logs. Next: gate re-pass → 10 steps →
run `compare_grad_norms.py` for the cross-framework verdict.

**2026-07-30 UPDATE — first 10 NeMo-RL steps (local 4×H100, run `ihdn67bj`).** Attempt-6 nb pod
crashed at vLLM init (`expandable_segments` incompatible with vLLM's memory pool — removed; act-ckpt
is the real OOM fix). Attempt-7 ScaleTrain pair queued for hours (cluster packed); meanwhile local
4-GPU gates (`m2jrli33` failed step 1, `b9b03y2f` finished: val 0.0591, grad_norm 2.43) proved 4-way
training fits with activation checkpointing (~346 s/step), and the full run `ihdn67bj` delivered
steps 1–10. **Result: verl's truncation bimodality does NOT reproduce** — grad norms 1.4–37.7 with
6/10 steps containing 1–2 truncated responses and zero spikes near verl's 83–3108 regime
(`compare_grad_norms.py` verdict: "regimes overlap"). Caveats: NeMo logs last-of-2 optimizer-step
grad norm (verl averaged both); truncated-in-all-wrong-group coincidence may not have occurred yet
at these truncation rates. ScaleTrain pair cancelled as redundant (jobs d9l8k5hq/d9l8tdpq); the
local run continues past step 10 to sharpen the test as responses lengthen. Env images published:
ECR `tmp:rl-distill-nemorl-cu130-20260729` (full) and hf.co/JWei05/nemorl-gemma4-cu130-env
(env-only, public).

> **⚠ RETRACTED (2026-07-30, see the ROOT CAUSE entry above): the bimodality verdict from
> `ihdn67bj` is INVALID.** That run's training forward was silently corrupted from step 1
> (act-ckpt × KV-shared layers on transformers 5.5.0 — `train/probs_ratio_clamped` pinned at
> 0.80), so its grad norms are gradients of a garbage objective, not the DAPO loss. The
> gate `b9b03y2f` cited here had the same poison in its train step (its VAL number remains
> valid). The bimodality test restarts from scratch on the fixed run `e1du1oyu` (`-local4g-v2`).

---

## 2026-07-28 — NeMo-RL exact-replication adapters for the gemma-4 E2B DAPO run (CPU gates green)

**Goal.** Reproduce `DAPO-gemma4-e2b-pt-DeepScaleR-4of4strict-seed42` (**8k variant**: 8192 max
response + 2048 overlong buffer) in NVIDIA NeMo-RL (`third_party/nemo-rl`, commit `5f89b3ae`) with
zero edits to the vendored checkout, so framework-level differences can be isolated from recipe
effects. Follows the verified interface study (workflow `wf_e4fa6102-3e1`).

**What was built** (all under `rl-distill-scripts/nemo_rl_repro/`):
- `rl_distill_nemo/deepscaler_dataset.py` — `DeepScalerStrictParquet`, verl-parquet → NeMo response
  dataset (dotted-path reference; no registry edit).
- `rl_distill_nemo/strict_math_env.py` — `math_strict` env: verbatim port of our strict boxed-only
  scorer from `verl/utils/reward_score/math_verify.py` (exactly-one-`\boxed{}`, LatexExtractionConfig
  only, spawn-pool verify with 30s timeout + pool pre-warm) wrapped in NeMo's
  `BaseMathEnvironment`/worker pattern.
- `run_grpo_repro.py` — wrapper that registers the env + `PY_EXECUTABLES.SYSTEM` actor entry and
  execs nemo-rl's `examples/run_grpo.py:main()`; exports PYTHONPATH so Ray actors import the adapters.
- `config/dapo_gemma4_e2b_pt_repro.yaml` — inherits their `grpo_math_1B.yaml` (NOT the gemma4-it
  recipe: its TE FusedAdam kwargs would merge into and crash `torch.optim.AdamW` — `_override_` is
  top-level-only in their loader); encodes every verl knob (64×16, gbs 512, token-mean, 0.2/0.28/c10,
  AdamW 1e-6/wd0.1, warmup 20, shaping 2048/1.0/**8192 explicit**, temp/top_p 1.0 top_k null, stops
  `<end_of_turn>/<start_of_turn>`, val 3200@10 + at-start, seed 42, save 25) + the recipe's gemma-4
  deltas (freeze towers, mbs/logprob 1, chunk 4096, packing off/dyn-batching on, gpu_mem_util 0.5
  flagged). Wandb: `DAPO/nemorl-dapo-gemma4-e2b-pt-DeepScaleR-4of4strict-seed42-8k`, entity rl-distill.
- `PARITY_CHECKLIST.md` — matched-knob table, unmatchable caveats (approx-entropy, grad_norm
  last-of-2, vLLM sampler numerics, stop-string in response length), go/no-go gate.

**Results (CPU gates).** `tests/test_strict_reward.py`: **PASS 6/6**, port and verl reference agree
(correct/wrong/two/no box, equivalent latex `1/2`≡`\frac{1}{2}`, malformed latex no-crash).
`tests/test_tokenization_parity.py`: **PASS 5/5** byte- and token-identical vs the verl reference
(12-shot template ~1.6k tok prompts, single BOS) — via the marked 3-line replicated processor path
(`import nemo_rl.models.policy` asserts transformers <5.12; our `.venv-gemma4*` has 5.14.1 — their
own uv venv is unaffected). Ran from `/tmp/.venv-gemma4-cu129` (EFS was RPC-saturated, load ~800).
Config verified end-to-end through their inheritance loader (all overrides + interpolations resolve).

**Status / next.** Adapters + gates done; **launch pending** the separately-built nemo-rl uv venv.
Launch: see `run_grpo_repro.py` header; then the PARITY_CHECKLIST go/no-go — step-0
`validation/accuracy ∈ [0.045, 0.075]` (verl baseline 6.16%@n=3200) with `grpo.max_num_steps=1`
before the full run. Deviation from the study: `grpo.max_num_epochs=100` (verl ran
`trainer.total_epochs=100`; study said 1, which would stop at ~151 steps).

**Goal.** Get verl DAPO RL working on `google/gemma-4-E2B`/`E4B` (PT), then land one production E2B
+ one E4B run: DeepScaleR strict-4/4 split, 12-shot prompt, 20k max response, seed 42, 1 node.

**Stack.** New `.venv-gemma4` (`setup_env_gemma4.sh`): torch 2.11 + vllm 0.25.1 + transformers 5.14.1,
**cu129 variant** (runs on the p5 fleet's CUDA-12.8 driver — gemma-4 does *not* need CUDA 13).
ScaleTrain runner: `scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh` (reuses the
`train-rl-distill` image; builds the venv in /tmp at job start).

**Fix ladder (each root-caused; all in-repo, gemma-3 paths untouched).**
1. cu129 venv variant (driver compat).
2. `VERL_FSDP2_LOCAL_LOAD=1` — FSDP2 loader NCCL broadcast desync (rank0 +8 collectives) → deadlock.
3. `VERL_ROLLOUT_EXTRA_STOP` — gemma-4 has no `<end_of_turn>` token; without stop strings responses
   never terminate (20480-token rambles). With stops: mean response ~193 tokens, 87% single-boxed.
4. math-verify dep + `MATH_VERIFY_OK` fail-fast guard (missing dep silently scored 0.0).
5. flash-attn-free padding fallback in `verl/utils/attention_utils.py` (transformers `_unpad_input`).
6. Chunked entropy: flags on `actor_rollout_ref.actor.fsdp_config.*` + non-rmpad branch patch in
   `verl/workers/engine/fsdp/transformer_impl.py` (fp32 entropy blob was 21–40GB).
7. `use_dynamic_bsz=False` + all `*_micro_batch_size_per_gpu=1` (padded-forward logits OOM).
8. `free_cache_engine=False` + `enable_sleep_mode=False` (vLLM cumem sleep/wake broken on cu129);
   resident engine at util 0.25 (E2B/4×H100) / 0.3+OFFLOAD (E4B/8×H100).
9. v16: fused `F.cross_entropy` logprobs — killed `logprobs_from_logits_v2`'s fp32 autograd retention…
10. …but `F.cross_entropy` is on **autocast's fp32 cast-list**: one 21.5GiB fp32 upcast of the
    [22k, 262k] logits on any near-max-length sample (killed both v16 twins, steps 1–3). **v17:
    `_ChunkedLogprobsFromLogits`** in `verl/utils/torch_functional.py` — chunked fp32 gather−logsumexp
    forward, softmax-recompute backward (2GiB fwd transient, zero extra retention).
11. v17 died on its own backward's 10.8GB `empty_like` grad buffer → **v18: in-place backward**
    (grad written into the logits buffer; flash-attn `inplace_backward` contract).
12. v18 still ~51-53GB retained at step-1 backward → the non-rmpad NO_PADDING branch of
    `prepare_model_outputs` itself: `cat([logits.unbind()])` logits duplicate (+10.6GB) + its padded
    base-grad in backward (+10.6GB), unconditional `div_(temperature)` DivBackward full-vocab grad at
    temp 1.0, chunked entropy bypassing `entropy_checkpointing`, and graph-attached meta tensors across
    the accumulation loop. **v19 fixes all four** (padded-direct logprobs — bit-identical to packed;
    div skip at temp 1.0; checkpointed chunked entropy; detached metas). Composed update-flow peak at
    [22057, 262144]: base+3GiB. E2B arms also gain OFFLOAD=True for Adam headroom at steps ≥2.
13. v19 still died at step 1 — **gemma-4's own forward applies `final_logit_softcapping=30.0`**
    (`logits/cap → tanh → *cap`): 2-3 extra full-vocab tensors per forward, tanh retains its 10.8GB
    output for backward. Found in minutes by the new **local harness**
    `gemma4_local_update_memtest.py`, which runs the real TrainingWorker + prod config on
    a worst-case batch (4× 22,057-token samples) through old-logprob → update → post-Adam update, with
    memory-history snapshots and a 55.7GB budget check (79.19 − 23.5 resident vLLM). **v20: build_model
    nulls `text_config.final_logit_softcapping` and prepare_model_outputs re-applies it fused into the
    chunked logprobs (backward ×(1−tanh²), exact to 8e-6) or chunked in-place for no-grad passes; vLLM
    reads the untouched checkpoint config.** Harness green: old-logprob 26.4GB, update 44.1GB with real
    pg_loss (first-ever completed worst-case update), post-Adam 61.2GB at world-1 (optimizer term
    quarters at prod world-4 → ~33GB/rank projected). **Standing rule: no ScaleTrain submission without
    a green harness run.**

**Baselines (strict-4/4 val, temp 1.0, 12-shot, step-0; from authoritative single-line `step:0` logs
— earlier numbers parsed from wandb's wrapped pprint dumps were wrong).** E2B acc mean@16 5.2–6.1%,
best@16 26–30%, resp len ~170–185 tok (many reproductions); E4B acc mean@16 **12.4–12.6%**, best@16
41–45%, resp len ~250–270 tok (6×). Training confirmed healthy when it runs: E2B 4 steps (entropy
0.94–1.06, sane pg_loss), E4B 2 steps — pre-v19 durability was luck of batch lengths. ~8 min/step at
micro=1 (perf debt: flash-attn CE build / padding-aware budgets later).

**Update (2026-07-27 ~14:00Z): 8k runs learning fast.** E2B-8k (`bw9dcxso`): val 6.16→6.00→6.78→
10.0→**12.59%** (step 40) — >2× baseline; step-25 ckpt pushed (NOTE: same HF repo/path as the 20k
run's step-25 — superseded it; old snapshot only in HF git history. Add a config suffix to
HF_PUSH_REPO if configs must be kept separate). E4B-8k (`1tyrqscv`): val 12.06→13.97→**22.97%**
(step 20), step-25 ckpt pushed. 8k curves dominate the 20k config's early trajectory
(E2B-20k was 7.25% at step 20; E4B-20k 13.41% at step 10). ~3-4 min/step at 8k vs ~8-12 at 20k.

**Update (2026-07-27 ~06:55Z): restarted at 8k response length (user request).** The 20k v20 pair
trained cleanly (E2B: 26 steps, val 5.03→5.63→7.25%, step-25 ckpt pushed to HF
`JWei05/DAPO-gemma4-e2b-PT-DeepScaleR-4of4strict-seed42/step_000025`; E4B: 16 steps, val
12.97→13.41%) and was then replaced by a `MAX_RESPONSE_LENGTH=8192` + `OVERLONG_BUFFER_LEN=2048`
pair (env-only change, same image): E2B-8k job_d9jg45k89ikduq9ss6fg (4×H100), E4B-8k
job_d9jg4q9q1fpdvq22m7u0 (8×H100), borrowing false, priority high.

**Status (2026-07-27 ~01:15Z). v20 WORKS — production runs live.** The harness-validated v20 quad all
passed step 0 val at baseline and trained: E2B twins cleared steps 1–5+ (past v14's all-time record of
4 luck-steps; step-2 post-Adam regime included), E4B twins cleared steps 1–3. Consolidated per the
goal (one run per model, borrowing false): **E2B = job_d9j9k0c89ikd2e6dlipg / wandb `bo3fu4ow`
(4×H100, util 0.25, offload), E4B = job_d9j9kh9q1fpdkf56p82g / wandb `fsrp64mz` (8×H100, util 0.3,
offload)** — DeepScaleR strict-4/4, 12-shot, 20k max response, seed 42, priority high. -brw twins
cancelled healthy (flushed steps 5/3 on exit). Watchpoints: step-10 val, step-25 checkpoint + HF push.
Deferred: math-verify trace spot-check + response-length distribution; perf debt ~8 min/step (micro=1).
Full forensic detail in memory `gemma4-verl-rl-enablement.md`.

---

## 2026-07-22 — 1B-PT RL relaunch on the **strict 4/4** split + heavy wandb trace logging

**Goal.** Relaunch the two 1B-PT DeepScaleR-4/4 RL runs on a fresh split cut from the **strict-graded**
4/4 pool, under the strict boxed-only grader (now the math_verify default), and upload far more traces
to wandb: 100 random **train** rollouts every train step + 100 random **val** traces every eval.

**Data.** New split from `deepscaler_acc4of4_strict.parquet` (9,923 strict-4/4 questions), shuffle seed 42:
**train 9,723 / val 200** → `deepscaler_4of4strict_rl_{train,val200_x16}.parquet` (val 200 uids ×16 = 3,200,
`val_kwargs.n=1` → pass/maj/mean@16). **Local only** (1B reads local files; not pushed to HF).

**Code change (fork).** `dapo/dapo_ray_trainer.py` (symlink → `rl-distill-scripts/dapo_ray_trainer.py`):
added `RayDAPOTrainer._log_train_generations_to_wandb(batch)` — a **fresh per-step** `train/generations`
wandb Table (one row per trace: step/input/output/score; no unbounded accumulation), gated on
`trainer.log_train_generations`, called in `fit()` right after the disk-dump block. Core launcher
`gemma3_pt_fewshot_math_rl.sh` exposes it via `LOG_TRAIN_GENERATIONS` → `+trainer.log_train_generations`
(new key, `+`). Val traces use the existing `LOG_VAL_GENERATIONS` (bumped to 100). Strict grading is the
math_verify default (`VERL_MATH_VERIFY_STRICT_BOXED`), so no extra flag.

**Runs** — same recipe (GRPO n=16, max_response 8192 + 2048 overlong buffer, temp 1.0 train=val, 12-shot
prompt, SAVE_FREQ=25, HF push /25, wandb, val_before_train), via
`launch_1b_deepscaler_4of4strict_seed_sweep.sh`: seed 42 → GPUs 2,3 / port 52000; seed 43 → GPUs 4,5 /
port 53000; isolated Ray `/tmp/ray_1b_ds4of4s_seed{42,43}`; both `LOG_{VAL,TRAIN}_GENERATIONS=100`.
wandb `DAPO-gemma3-1b-pt-DeepScaleR-4of4strict-seed{42,43}`; HF
`JWei05/DAPO-Gemma3-1B-PT-DeepScaleR-4of4strict-seed{42,43}`.

**Ray startup timeout — root cause + fix (two parts).** Both seeds repeatedly died with Ray "The current
node timed out during startup" (even seed 42 alone, first — so NOT the 150s-stagger collision). Real root
cause: **the shared box is oversubscribed** — `load average ~180` with 83 users vs 192 CPUs (memory was
fine, 1.4 TB free). `ray/_private/node.py` gives the raylet only a **hardcoded 30s** to register with the
GCS (`raylet_start_wait_time_s = 30`); under that CPU starvation the raylet can't register in time and the
driver SIGKILLs it (the raylet.err "crash" trace is just that teardown, not a self-crash → a longer wait
succeeds). Two-part fix, both in `gemma3_1b_pt_fewshot_math_rl.sh`:
  1. **Widen the window.** Patched `.venv/.../ray/_private/node.py:411` to read an env var:
     `raylet_start_wait_time_s = int(os.environ.get("RAY_RAYLET_START_WAIT_TIME_S", "30"))`, and the wrapper
     exports `RAY_RAYLET_START_WAIT_TIME_S=300`. ⚠️ **This node.py edit is venv-local and is lost on a fresh
     `setup_env.sh`** — reapply it then (the wrapper's env export is a harmless no-op without it).
  2. **Cut our own startup churn.** `+ray_kwargs.ray_init.num_cpus=32` — the raylet prestarts one Python
     worker per CPU, so capping CPUs (a 2-GPU FSDP run needs only ~8: `max_colocate_count=3` →
     `{CPU:3,GPU:1}` bundles) avoids a 192-way EFS-venv import storm on top of the existing load.
  3. **Two more startup timeouts, same root cause.** With the driver-wait widened, the raylet then aborted
     `node_manager.cc:3362 Check failed ... Timed out waiting for file .../metrics_agent_port` — the
     metrics/dashboard agent (another EFS-venv Python import) was too slow to write its port file. These are
     real C++ RayConfig fields (env-overridable, confirmed via `strings _raylet.so`), so the wrapper now also
     exports `RAY_agent_register_timeout_ms=300000` and `RAY_worker_register_timeout_seconds=600`. (Stuck
     drivers go D-state on EFS I/O, so `kill -9` is pending until the syscall returns — expect slow cleanup.)

  4. **The un-fixable one (why it still won't start right now).** Even solo, the raylet keeps aborting on
     `Timed out waiting for file .../metrics_agent_port`: the metrics/dashboard agent (yet another EFS-venv
     Python import) takes ~90–117s to start under load but the raylet's wait for its port file is a **fixed
     ~86s hardcoded C++ timeout with NO env override** (confirmed `agent_register_timeout_ms` reaches the
     raylet via `os.environ.copy()` but doesn't govern this wait; can't patch a compiled `.so`). So the only
     real cure is the **external CPU load dropping** — box was at load ~210 when first flagged and climbed to
     ~280 (81 users) over the session; GPUs 1–5 are free the whole time, so it's purely CPU/EFS startup
     starvation, not our config.

All of this is captured in memory `ray-local-startup-timeouts-loaded-box`.

**Status — ARMED, waiting on the box.** Everything is staged and correct; the sole blocker is the overloaded
shared box. Deployed a **load-gated retry orchestrator** `launch_1b_ds4of4strict_retry.sh` (nohup, deadline
180min): it polls `/proc/loadavg` and only attempts a launch when 1-min load < `LOAD_GATE=160` (so it never
piles onto a peak-load box), brings seeds up **one at a time** (seed 42 on GPUs 2,3 then seed 43 on 4,5), and
stops the instant each is past Ray bringup. A persistent monitor watches its output. It will auto-launch both
runs the moment the box calms below 160 — nothing further to do on our side until then.

---

## 2026-07-22 — Strict boxed-only grading + full strict re-eval of GEMMA3_PT_EVAL_REPLICATION.md

**Change.** `verl/utils/reward_score/math_verify.py` now scores **strict** by default
(`VERL_MATH_VERIFY_STRICT_BOXED`): exactly one `\boxed{}` (0 or ≥2 → 0), verify only that box — vs the
old lenient whole-output `parse` + max-over-candidates that credited bare-LaTeX echoes / hedges. Tests:
`tests/utils/reward_score/test_math_verify_strict_boxed_on_cpu.py` (14 pass).

**Re-eval.** Rescored saved traces (sampled base/RL matrix `~/verl/eval_traces`; 4B-IT DeepScaleR ×4)
and regenerated the untraced ones on GPUs 2–5 with `--enforce_eager --trace_dir ~/verl/eval_traces_strict`
(greedy base/RL tables; per-difficulty 4B-PT/1B-PT on the **strict** difficulty buckets; few-shot sweep;
longest-2-shot; DeepScaleR-250). Scripts in scratch: `strict_rescore_traces.py`, `rescore_itgen_strict.py`,
`regen_strict.py`, `regen_strict2.py`, `sweep_strict.py`.

**Findings.** Strict ≈ lenient for **greedy** (single clean box); **meaningfully lower for sampled/temp-1.0**
(degeneration → no-box/echo false positives removed, e.g. DeepScaleR-250 4B temp-1.0 4.0→1.6). 4B-IT barely
changes (mean@4 38.8→37.9). Base→RL conclusions hold (RL still ~doubles DAPO-val pass@16). The 4B-IT
difficulty ranking transfers to PT models (4B-PT greedy 0/4→0% … 4/4→27%). Verified: all 2,530
scored-correct of 25,401 traces have exactly one box (0 violations); no false negatives spotted. All in the
new "Strict grading" section of `GEMMA3_PT_EVAL_REPLICATION.md`.

---

## 2026-07-22 — DeepScaleR **4/4-subset** RL: 1B ×2 seeds (local) + 4B ×1 (ScaleTrain)

**Goal.** Replace the prior full-DeepScaleR RL experiments with RL on the **4/4 subset** — the 10,404
questions Gemma-3-4B-IT solved 4/4 (from the difficulty filtering below). Clean, verified-solvable
problems; the 4B-PT only gets ~19% greedy on them, so there's RL headroom.

**Data.** 4/4 split (shuffle seed 42): **train 10,204 / val 200**. Files `deepscaler_4of4_rl_train.parquet`,
`deepscaler_4of4_rl_val200_x16.parquet` (200×16=3,200, `val_kwargs.n=1` → **pass/mean/maj@16**). Uploaded
to HF **`JWei05/DeepScaleR-4of4-RL`** (both local + pod pull the identical split).

**Runs** — same recipe as the DeepScaleR runs (GRPO, n=16, **max_response 8192 + 2048 overlong buffer**
[penalty ramps 6144→8192; changed from 20k/4k on the 2026-07-22 restart], temp 1.0 train=val,
**12-shot prompt** `gemma3_it_fewshot_math.jinja`, SAVE_FREQ=25, HF push /25, wandb, val_before_train):

- **1B ×2 seeds, local** (`launch_1b_deepscaler_4of4_seed_sweep.sh`): seed 42 → GPUs 2,3 / port 52000;
  seed 43 → GPUs 4,5 / port 53000; isolated Ray `/tmp/ray_1b_ds4of4_seed{42,43}`. wandb
  `DAPO-gemma3-1b-pt-DeepScaleR-4of4-seed{42,43}`; HF `JWei05/DAPO-Gemma3-1B-PT-DeepScaleR-4of4-seed{42,43}`.
- **4B ×1 seed, ScaleTrain** (`scale_train/run_gemma3_4b_pt_deepscaler_4of4_rl.sh` +
  `data/prepare_deepscaler_4of4_rl_data.sh`): 1×8-GPU H100, **priority high + borrowing**. wandb
  `DAPO-gemma3-4b-pt-DeepScaleR-4of4`; HF `JWei05/DAPO-Gemma3-4B-PT-DeepScaleR-4of4`.

**Status.** Launched. 1B seed 42 → GPUs 2,3 (fine); seed 43 hit a transient Ray-startup timeout when both
local Ray clusters started at once under EFS load → relaunched alone on GPUs 4,5 (fix next time: start the
two local seeds fully sequentially, not 45s-staggered). 4B ScaleTrain job (QUEUED, priority high + borrowing). Data split was cut from `deepscaler_acc4of4.parquet`
(see entry below).

**Restart (2026-07-22):** all 3 restarted to change **max_response 20480→8192** and **overlong buffer
4096→2048** (penalty ramps 6144→8192). 1B stagger bumped 45s→150s to avoid the Ray-startup collision.
Active jobs: 1B seeds 42 (GPUs 2,3) / 43 (GPUs 4,5) local; **4B ScaleTrain `job_d9g2rupq1fp5np2n185g`**
(QUEUED). (Superseded IDs: canceled `job_d9g1o4hq1fp5np2n1850`.)

---

## 2026-07-21/22 — DeepScaleR difficulty filtering (via Gemma-3-4B-IT pass rate) + 4B-PT check

**Goal.** Partition `agentica-org/DeepScaleR-Preview-Dataset` (40,315 competition-math problems) by
difficulty so we can select learnable-difficulty subsets for RL (drop always-solved / never-solved,
keep the middle), rather than training on the full, mostly-too-hard set.

**Approach — what we settled on.**
- First tried bucketing by the dataset's own `solution` length → **abandoned: 81.7% of `solution`
  fields are empty** (only 7,391 of 40,315 non-empty), so it's not a usable difficulty signal.
- Switched to a **model-based difficulty measure**: generate with **Gemma-3-4B-IT**, 4 samples per
  question, and use the per-question pass rate (`n_correct` out of 4) as difficulty.

### Step 1 — Inference: Gemma-3-4B-IT over all of DeepScaleR, 4×
- Model `google/gemma-3-4b-it`; **every** question (40,315) × **4 samples**.
- Sampling: **temp 1.0, top_p 1.0, top_k −1**; **8k context (max_prompt 2048 + max_response 6144)**.
  (Started at 12k/8192, tightened to 8k on request — smaller KV → higher concurrency.)
- Prompt: **short 2-shot** (`data/gemma3_it_fewshot_math_2shot.jinja`, domain + 15-trees).
- **DP=4** across GPUs 2,3,4,5 (dataset sharded 4 ways, TP=1 each). ~250–300 concurrent seqs/GPU,
  ~68–84% util, ~40k output tok/s aggregate, ~1.5 h wall.
- Script: `data/deepscaler_it_gen.py` (feeds token-id prompts, single BOS; saves per question the 4
  `response_texts` + `response_lens` + math_verify `scores`/`accs` + `n_correct`/`any_correct`).
- Output: `~/verl/data/deepscaler_it_gen/shard_{0..3}.parquet` (full traces kept).
- Note: run each shard's launch **staggered** or with a per-shard compile cache next time — 4 vLLM
  procs sharing the HF cache on EFS made the weight-load slow (~10 min D-state), and sharing the
  torch.compile inductor cache later corrupted it (see Step 3 caveat).

**IT-gen results (40,315 q):** overall **mean@4 = 38.8%**, **pass@4 = 53.7%**. Response length is a
strong difficulty signal — mean@4 by response-length quintile: **79.5 → 45.8 → 28.4 → 22.5 → 17.8%**
(shortest→longest). Lengths: median 1,264 tok, p95 2,021, max 4,252 (nothing hit the 6,144 cap).

### Step 2 — The 5 accuracy subsets (by `n_correct` of 4)
Merged (`merge_and_analyze_deepscaler_it_gen.py`) → `deepscaler_it_gen_merged.parquet` (all traces),
plus 5 **RL-ready** parquets `~/verl/data/deepscaler_it_gen/deepscaler_acc{0..4}of4.parquet`
(data_source=`math`, `\boxed{}` prompt, ground-truth):

| subset | questions | share | median resp len (IT) |
|---|---|---|---|
| 0/4 (never solved) | 18,670 | 46.3% | 1,405 |
| 1/4 | 4,755 | 11.8% | 1,472 |
| 2/4 | 3,251 | 8.1% | 1,419 |
| 3/4 | 3,235 | 8.0% | 1,260 |
| 4/4 (always solved) | 10,404 | 25.8% | 617 |

Bimodal: ~46% never solved, ~26% always solved, ~28% in the learnable middle (1–3/4).

### Step 3 — Sanity eval: Gemma-3-4B-PT on the subsets (greedy + temp 1.0)
- `google/gemma-3-4b-pt`, **12-shot** prompt (`data/gemma3_it_fewshot_math.jinja`), **100 random
  questions/bin** (seed 42), both **greedy (temp 0)** and **temp 1.0**, max_tokens 4096, max_model_len
  8192, via `eval_math_passk.py --enforce_eager`. Samples: `~/verl/data/deepscaler_acc{0..4}of4_s100.parquet`.

**4B-PT accuracy by IT-difficulty subset (mean@1, %):**

| IT subset | greedy | temp 1.0 |
|---|---|---|
| 0/4 | 3.0 | 2.0 |
| 1/4 | 5.0 | 0.0 |
| 2/4 | 4.0 | 1.0 |
| 3/4 | 7.0 | 2.0 |
| 4/4 | **19.0** | **8.0** |

- **Difficulty transfers across models**: 4B-PT accuracy rises with the 4B-IT pass rate; clearly
  highest on the 4/4 bin, near-floor on 0/4. Confirms the IT-pass-rate binning is a valid difficulty axis.
- But **compressed/low**: even the easiest (4/4) bin is only 19% greedy for the PT — DeepScaleR is hard
  for the 4B-PT, and IT-4/4 (stronger model + its own 2-shot prompt) ≠ easy-for-PT. Middle bins (1–3/4)
  are ~3–7% and statistically flat at n=100 (±~3–5%); the real signal is the 4/4 jump.
- **Greedy > temp 1.0** everywhere (base-PT pattern seen throughout, e.g. GEMMA3_PT_EVAL_REPLICATION.md).
- **Caveat:** needed `--enforce_eager` (flag added to `eval_math_passk.py`) — the DP=4 IT-gen corrupted
  the shared torch.compile inductor cache (`UnpicklingError: pickle data was truncated`); eager skips
  compilation and sidesteps it.

**Artifacts.**
- Scripts: `data/deepscaler_it_gen.py`, `data/merge_and_analyze_deepscaler_it_gen.py`,
  `eval_math_passk.py` (now supports `--enforce_eager`).
- Data (local): `~/verl/data/deepscaler_it_gen/` (shards, merged, 5 `acc*of4` subsets),
  `~/verl/data/deepscaler_acc*of4_s100.parquet`.
- Related earlier: `data/build_deepscaler_rl_data.py` + `data/prepare_deepscaler_rl_data.sh` built the
  DeepScaleR RL train/val split, uploaded to HF `JWei05/DeepScaleR-RL`.

**Status.** IT-gen + subset build + 4B-PT sanity eval all complete; nothing running. The earlier
DeepScaleR RL runs (1B local, 4B ScaleTrain) were stopped/cancelled — superseded by this filtering
work. Next: decide which subset(s) to train RL on (likely the learnable middle 1–3/4).

## 2026-09-13 — Hugging Face Hub cleanup (user request)

**Rule:** delete model repos last modified before 2026-07-30 (datasets untouched; repos created before but modified after the cutoff kept); the two Gemma 4 `4of4strict-seed42` RL repos (2026-07-27/28) were explicitly kept. 37 repos deleted (irreversible), 77 model repos remain.

Earlier the same day: `JWei05/Distill-gemma4-e4b-base-medium-to-{12b,26b}-base` pruned to `step_001000` with the other steps' LFS blobs purged (0.47 TB + 1.01 TB); their `step_001000` exports copied to `s3://scale-ml/genai/rl-distill/gemma4-e4b-base-distill-final-exports/`.

Deleted repos (several are still referenced by legacy Gemma 3 launch/eval scripts, which now point at nothing):

- `JWei05/dapo-gemma3-1b-pt`
- `JWei05/gemma3-4b-pt-sft-nemotron-cascade2-16k`
- `JWei05/gemma3-4b-pt-sft-distill-from-27b`
- `JWei05/gemma3-4b-pt-sft-distill-from-12b`
- `JWei05/dapo-gemma3-4b-pt-sft-nc2-16k`
- `JWei05/gemma3-1b-pt-sft-distill-from-4b`
- `JWei05/gemma3-1b-pt-sft-distill-from-12b`
- `JWei05/gemma3-1b-pt-sft-distill-from-27b`
- `JWei05/gemma3-1b-pt-onpolicy-distill-from-dapo4b-step60`
- `JWei05/gemma3-1b-pt-onpolicy-distill-from-dapo12b-step80`
- `JWei05/gemma3-1b-pt-onpolicy-distill-from-dapo27b-step80`
- `JWei05/gemma3-4b-pt-onpolicy-distill-from-dapo12b-step80`
- `JWei05/gemma3-4b-pt-onpolicy-distill-from-dapo27b-step80`
- `JWei05/gemma3-4b-pt-moe-2e-top1-sft-16k`
- `JWei05/gemma3-4b-pt-moe-4e-top1-sft-16k`
- `JWei05/dapo-gemma3-4b-pt`
- `JWei05/gemma3-4b-pt-sft-distill-from-27b-rl-step40-seed43`
- `JWei05/gemma3-4b-pt-sft-distill-from-12b-rl-step20-seed43`
- `JWei05/gemma3-4b-pt-sft-distill-from-27b-rl-step40-seed43-all33296-n4`
- `JWei05/gemma3-4b-pt-sft-distill-from-12b-rl-step20-seed43-all33296-n4`
- `JWei05/qwen35-9b-swe-save-smoke`
- `JWei05/Qwen3-4B-Instruct-2507-Scalar-Critic`
- `JWei05/DAPO-Gemma3-4B-PT-RL-DAPO17k`
- `JWei05/Qwen3-4B-Instruct-2507-Hybrid-Scalar-Reasoning-Critic`
- `JWei05/Qwen3-4B-Instruct-2507-Reasoning-Critic`
- `JWei05/Qwen2.5-3B-Instruct-DeepScaleR-Canonical-Critic`
- `JWei05/dapo-gemma3-4b-pt-moe-2e-megatron-rl`
- `JWei05/DAPO-Gemma3-1B-PT-FewShotMath-seed43`
- `JWei05/DAPO-Gemma3-1B-PT-FewShotMath-seed44`
- `JWei05/DAPO-Gemma3-1B-PT-FewShotMath-seed42`
- `JWei05/DAPO-Gemma3-4B-PT-FewShotMath`
- `JWei05/DAPO-Gemma3-4B-PT-FewShotMath-seed43`
- `JWei05/DAPO-Gemma3-1B-PT-DeepScaleR`
- `JWei05/DAPO-Gemma3-4B-PT-DeepScaleR`
- `JWei05/DAPO-Gemma3-1B-PT-DeepScaleR-4of4-seed42`
- `JWei05/DAPO-Gemma3-1B-PT-DeepScaleR-4of4-seed43`
- `JWei05/DAPO-Gemma3-1B-PT-DeepScaleR-4of4strict-seed42-local`

## 2026-09-14 — 12B distilled RL (medium) finished
- `g4-12b-distill-rl-med` (RL on top of the E4B-base→12B off-policy distilled student, DeepScaleR medium, seed 42, ES patience 5): early stopping at step 240, best val mean@16 **0.497 @ step 190** (from ≈0.08 at step 0). Best HF snapshot + receipt on S3 (`gemma4-12b-from-e4bbase-distill-rl/gemma4-12b-medium-from-e4bbase-distill-es5/best_hf/`). Details: DISTILLATION_EXPERIMENTS.md §9.0.
- On-policy distillation take 3 (student-top-128 reverse KL + tail bucket) running: job_dajtd33ns19g0896plog, step 40+, no flattening so far (§9.0f). Plain student-top-k run (take 2) collapsed by step 150 and was cancelled; its step-50 results stand (§9.0e).

## 2026-09-16 — devbox-008 restart backup

Infra is stop/starting devbox-008 (erases `/opt/dlami/nvme` = `/tmp` and `/dev/shm`). Everything of ours on `/tmp` that had no S3/Hub copy
was synced to `s3://scale-ml/genai/rl-distill/devbox-008-backup-20260916/` (≈62 GB): `gemma4_12bd_evals/passk`, `gemma4_distill_study_eval/results`,
`gemma4_e4b_val32`, `gemma4_stk50_kl`, `local_eval_raw`, `gemma4_distill_views`, `jasonwei_hf_stage/onpolicy_stk_step50`, the paper draft
(`tmax-research.*/paper`, `paper.tar`), two eval jsonl dumps and the local 12B-resume logs. Not uploaded (recoverable elsewhere): 26B checkpoint
copies (S3), Hub staging dirs (uploads finished), trace bundles (S3), HF/vLLM caches, agent scratch. `local_eval_models` (569 GB of Qwen2.5-3B
GRPO eval exports) was left for the user to decide. `/dev/shm` held only 238 MB of ours. The local venv `/tmp/.venv-gemma4` is rebuilt by
`setup_env_gemma4.sh` + `rl-distill-scripts/patches/`. GitHub `main` pushed through `dd6647dc`.
