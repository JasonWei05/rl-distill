# Progress Log

A running log of experiments/work in this repo. Newest entries on top. Each entry records the goal,
what was run (config + exact scripts/data), results, and status, so work is resumable and auditable.

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
