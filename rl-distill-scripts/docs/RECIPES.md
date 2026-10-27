# Training recipes: RL, off-policy distillation, on-policy distillation

Last updated: 2026-10-26

This is the reference for the three training recipes in this repo: DAPO/GRPO RL, off-policy top-k forward-KL
distillation, and on-policy distillation in policy-gradient form (PG-k1). It covers Gemma 4 (E2B / E4B / 12B / 26B-A4B
base) and Qwen3 base (1.7B / 4B / 8B / 30B-A3B). Every value below was read from the code as of this date, with a
`file:line` pointer. Anything not read directly from code is marked *(inferred)*.

- Results are in [GEMMA4_RESULTS.md](GEMMA4_RESULTS.md) and [QWEN3_RESULTS.md](QWEN3_RESULTS.md). The chronological record is in [../PROGRESS_LOG.md](../PROGRESS_LOG.md).
- ScaleTrain operations are in [OPERATIONS.md](OPERATIONS.md): images, supervisors, borrowing, preemption, S3 janitors, and the W&B and S3 layout.
- Repo overview: [../README.md](../README.md).

Code under `verl/` is shown as a path, not a link.

---

## 0. Shared plumbing

### 0.1 Job launch chain (RL and on-policy distillation)

```
scale_train/launch_*.sh            builds an env-var string (recipe + per-size memory layout), picks GPUs / priority / borrowing
  -> scale_train/launch_st_with_code.sh   git archive <CODE_REF|HEAD> -> s3://scale-ml/genai/rl-distill/code/rl-distill-code-<sha>.tar.gz
  -> scale_train/launch_st_job.py         ScaleTrain job (cluster eks, team egp, forwards HF_TOKEN / WANDB_API_KEY / WANDB_BASE_URL)
  -> pod: scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh   (one run-file for Gemma 4 and Qwen3, RL and OPD)
  -> run_dapo_rl.sh                       composes the Hydra command
  -> python3 -m dapo.main_dapo            dapo/config/dapo_trainer.yaml (defaults: verl ppo_trainer)
```

- [`launch_st_with_code.sh`](../scale_train/launch_st_with_code.sh) ships only **committed** code. It runs `git archive` on `CODE_REF`, which defaults to `HEAD` (L187-198), and runs it on the pinned ECR image (L183).
- The run-file does everything per pod:
  - builds `/tmp/.venv-gemma4` with `setup_env_gemma4.sh`, using the cu129 variant (L134-143);
  - fails fast if `math_verify` is missing (L161);
  - resolves the model and its family contract (L37-66);
  - prepares the band data (L511-526);
  - restores the newest complete S3 checkpoint (L535-543);
  - runs `run_dapo_rl.sh` with the overrides at L552-641;
  - then publishes `best_hf/` and the completion receipt (L660-681).
- `dapo/` is a package of symlinks into `rl-distill-scripts/`: `main_dapo.py`, `dapo_ray_trainer.py`, `hf_push.py` and `config/`. The one exception is [`dapo/validation_early_stopping.py`](../../dapo/validation_early_stopping.py), which is a real file.
- `main_dapo.py` swaps in `RayDAPOTrainer`. It also injects the credentials into Ray's `runtime_env` from the driver environment, so they never appear in Hydra (L78-114).
- [`run_dapo_rl.sh`](../run_dapo_rl.sh) supports `DRY_RUN=1`, which prints the resolved Hydra job config without starting Ray (L114-120). Its in-file defaults (Gemma 3, 20k responses, `SAVE_FREQ=25`) are historical. The run-file and launchers always override them.

### 0.2 Prompt contracts per model family

Source: run-file L56-66.

| | Gemma 4 | Qwen3 base |
|---|---|---|
| Chat template | [`data/gemma3_it_fewshot_math.jinja`](../data/gemma3_it_fewshot_math.jinja): 12-shot (12 user/model exemplar turns), same for train and val | [`data/qwen3_base_native_chat.jinja`](../data/qwen3_base_native_chat.jinja): 0-shot native ChatML |
| Instruction | from the dataset rows | data prep appends ` Please reason step by step, and put your final answer within \boxed{}.` to every problem (`prepare_deepscaler_qwen3_30b_difficulty_rl_data.py` L26, L63-71) |
| Extra stop strings | `<end_of_turn>`, `<start_of_turn>` | `\nHuman:`, `\nuser\n` |
| Stop token ids | none | `151645,151643,151644` (`<\|im_end\|>`, `<\|endoftext\|>`, `<\|im_start\|>`) |
| FSDP2 per-rank local load | on (`VERL_FSDP2_LOCAL_LOAD=1`) | off (rank-0 broadcast; local load needs tied embeddings) |

### 0.3 Reward (RL reward, and the accuracy metric for every recipe)

- The grader is `verl/utils/reward_score/math_verify.py` `compute_score` (L249-294). Boxed-only grading is on by default (`VERL_MATH_VERIFY_STRICT_BOXED=1`).
  - A response with no well-formed `\boxed{}` scores 0.
  - Only the **last** well-formed box is graded.
  - Grading is `math_verify`, or the bounded Miles-style SymPy fallback in `verl/utils/reward_score/miles_sympy.py`, which has a 5 s SIGALRM budget (`VERL_MATH_SYMPY_TIMEOUT`, L205-212).
  - The whole call has a 30 s timeout. A timeout scores 0.
- The reward manager is `dapo` (`verl/workers/reward_manager/dapo.py` L120-131). It adds a soft overlong penalty: `min(-(len - (max_resp - buf)) / buf * factor, 0)`.
  - With 8192 responses, a 2048 buffer and factor 1.0, the penalty ramps linearly from 0 at 6144 tokens to -1 at 8192.
  - OPD runs turn the penalty off (`ENABLE_OVERLONG_BUFFER=False`).

### 0.4 Datasets (DeepScaleR difficulty bands)

Each band is 3,000 train questions plus 300 validation questions. Validation is expanded ×16 (rows per UID), so verl
reports `mean@16` / `maj@16` with `VAL_N=1`. Files are pinned by SHA-256.

| Source (`DIFFICULTY_DATASET_SOURCE`) | Prep script | HF dataset @ revision | Bands |
|---|---|---|---|
| `qwen3_30b_bands` (all Qwen runs) | [`data/prepare_deepscaler_qwen3_30b_difficulty_rl_data.py`](../data/prepare_deepscaler_qwen3_30b_difficulty_rl_data.py) | `JWei05/DeepScaleR-Easy-Medium-Hard-Qwen3-30B-A3B-Base-10k` @ `be4dcd8b` (`very_hard` @ `9bec69ca`) | 30B-A3B-Base pass count out of 8: easy 4-8, medium 2-3, hard 1, very_hard 0/8 then ≥1/16 on resample |
| `gemma4_26b_bands` (all Gemma 4 runs) | [`data/prepare_deepscaler_gemma4_26b_difficulty_rl_data.py`](../data/prepare_deepscaler_gemma4_26b_difficulty_rl_data.py) | `JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k` @ `a0ba3c3d` | easy / medium / hard (Gemma-4-26B-PT-defined) |
| `legacy` (historical) | [`data/prepare_deepscaler_easy_medium_rl_data.py`](../data/prepare_deepscaler_easy_medium_rl_data.py), `prepare_deepscaler_4of4strict_rl_data.sh` | `JWei05/DeepScaleR-Easy-10k`, `-Medium-20k` (+ GSM8K, MATH-500 val) | strict4of4 / easy / medium (Gemma 4 runs before the band sweep) |

The band builders are `data/build_deepscaler_qwen3_30b_difficulty_dataset.py` (thresholds at L77-79) and
`data/build_deepscaler_qwen3_30b_very_hard_band.py`.

---

## 1. DAPO / GRPO RL

### 1.1 Objective

This is GRPO with the DAPO token-level tweaks. Dynamic sampling (filter groups) is **off**
(`ENABLE_FILTER_GROUPS=False`, `run_dapo_rl.sh` L40), so the recipe is GRPO + clip-higher + token-mean + soft overlong
penalty + token TIS.

For each prompt, sample G = 16 responses with reward r_i ∈ {0,1} (plus the overlong penalty). The advantage is the group
z-score Â_i = (r_i − mean(r)) / std(r), broadcast to every token of response i. Standard-deviation normalisation is
verl's GRPO default *(inferred: not overridden)*. The loss is

```
L = (1/Σ|y_i|) Σ_i Σ_t  w_t · max( −Â_i ρ_t , −Â_i clip(ρ_t, 1−0.2, 1+0.28) )   (dual-clip c = 10 for Â<0)
ρ_t = π_θ(y_t) / π_old(y_t)   (π_old = trainer recompute before the update)
w_t = min( π_old(y_t) / π_vLLM(y_t), 2.0 )      token TIS between trainer and rollout engine
```

There is no KL term (`use_kl_loss=False`, `use_kl_in_reward=False`) and no entropy bonus (`entropy_coeff=0`).

### 1.2 Hyperparameters (Gemma 4 and Qwen3 share them)

| Knob | Value | Where |
|---|---|---|
| adv estimator / group size | `grpo`, `N_RESP_PER_PROMPT=16` | `run_dapo_rl.sh` L26; launchers |
| prompts per step / mini-batch | `TRAIN_PROMPT_BSZ=64` (= `GEN_PROMPT_BSZ`), `TRAIN_PROMPT_MINI_BSZ=32` → 1024 responses, **2 optimizer steps per training step** | Qwen launcher L78-79; Gemma launcher L123-127 |
| optimizer | AdamW lr **1e-6**, linear warmup **20** steps, then constant (verl `lr_scheduler_type: constant`), weight decay 0.1, betas (0.9, 0.999), grad clip 1.0 | `run_dapo_rl.sh` L92-93, L175-182; verl `_generated_ppo_trainer.yaml` L14-23 |
| PPO clip | `clip_ratio_low=0.2`, `clip_ratio_high=0.28`, `clip_ratio_c=10.0` | `run_dapo_rl.sh` L28, L159-161 |
| loss aggregation | `token-mean` | L38 |
| KL / entropy | none / `entropy_coeff=0` | L27, L181 |
| lengths | prompt 4096 (left truncation), response **8192**, `MAX_MODEL_LEN=12288` | launchers |
| overlong | soft buffer 2048, factor 1.0 | launchers; §0.3 |
| rollout correction | `rollout_is=token`, `rollout_is_threshold=2.0` (TIS clamp), no batch-normalise, no rejection sampling, `bypass_mode=False`; always last on the command line | run-file L634-641 |
| sampling | temp 1.0, top-p 1.0, top-k −1 for train **and** val (`do_sample=True`) | `run_dapo_rl.sh` L79, L190-197 |
| rollout engine | vLLM, chunked prefill, prefix caching, `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, flashinfer sampler off, compiled (not eager: `ROLLOUT_ENFORCE_EAGER=False`) | run-file L574-590 |
| validation | before training and every `TEST_FREQ=10` steps on the band's `val300_x16` (mean@16, 300 q × 16) | launchers; run-file L514-518 |
| max steps | `TOTAL_TRAINING_STEPS=400` (default; many Qwen runs were extended to 500 / 600 / 1000 with this knob) | launchers |
| data seed | `DATA_SEED=42` (Gemma seed replicates: 43-47) | launchers |

### 1.3 Early stopping

The implementation is `ValidationEarlyStopping.observe`
([`dapo/validation_early_stopping.py`](../../dapo/validation_early_stopping.py) L379-431). Its state is saved in every
checkpoint (`validation_early_stopping.json`), so a resumed run keeps its best score and miss count.

- The metric is `val-core/math/acc/mean@16`, mode `max`. The pre-training validation counts as an observation (`include_initial_validation=True`).
- **Improvement** means `score > best + min_delta`. An improvement sets best = score and resets the miss streak to 0.
- With `miss_tolerance > 0`, a score that is not an improvement but is still ≥ `best − miss_tolerance` is **not** a miss, and it resets the streak to 0. Only a score below `best − miss_tolerance` adds a miss.
- With `miss_tolerance = 0`, every non-improvement adds a miss.
- The run stops after `patience` consecutive misses. A permanent checkpoint is forced at the stopping step (`dapo_ray_trainer.py` L1043-1044).

| Recipe | metric | patience | min_delta | miss_tolerance | Source |
|---|---|---|---|---|---|
| RL (Gemma 4 and Qwen3, standard) | mean@16 | **5** evals (50 steps) | 0.0 | Qwen **0.01**; Gemma launchers pass none → **0.0** | Qwen launcher L74; `launch_gemma4_rl_band_seed.sh` L122; run-file L626-629 |
| Qwen RL ablations / reruns (e.g. 4B' RL `p4b1000-es10`, 8B hard `rerun-es10`) | mean@16 | 10 | 0.0 | 0.01 | `EARLY_STOPPING_PATIENCE=10` on the Qwen launcher |
| OPD v2 (§3) | mean@16 | 10 | **0.005** | 0.0 | `launch_qwen3_opd_recipe_v2_batch.sh` L137-138; `launch_qwen3_onpolicy_distill.sh` L104-106 |

Extra early-stopping knobs:

- `EARLY_STOPPING_RESET_PATIENCE_AT_STEP=N` clears the miss streak once, when resuming from exactly `global_step_N`, and keeps the all-time best. Use it to extend a run that already early-stopped (`dapo_ray_trainer.py` L233-240, L327-334).
- `EARLY_STOPPING_MIGRATE_PATIENCE_FROM` changes patience on a restored state.
- `EARLY_STOPPING_LEGACY_STATE_B64` seeds the history for a checkpoint that predates the early-stopping state. It is built with [`scale_train/build_es_migration_from_wandb.py`](../scale_train/build_es_migration_from_wandb.py).

### 1.4 Checkpoints and the published model

- **Rolling** checkpoint every `ROLLING_CHECKPOINT_FREQ=5` steps. It holds sharded fp32 model + Adam + LR/RNG + StatefulDataLoader cursor in a single rolling S3 slot under `FULL_CHECKPOINT_S3_URI` (`dapo_ray_trainer.py` L112-127). The janitor that prunes superseded rolling slots runs outside the pod; see [OPERATIONS.md](OPERATIONS.md).
- **Permanent** checkpoint every `SAVE_FREQ=10` steps. It holds the same contents plus `actor/huggingface` (`ACTOR_CKPT_SAVE_CONTENTS=[model,optimizer,extra,hf_model]`). Local retention is `MAX_ACTOR_CKPT_TO_KEEP` (Qwen 2, Gemma 6).
- On every pod start the run-file calls `full_checkpoint_s3.py restore-latest` (L535-543). That makes preemption transparent: an exact resume restores Adam, data cursor and early-stopping state.
- At the end, `run_outcome.json` records the best step. The run-file then runs `full_checkpoint_s3.py publish-best-hf` (L665-669), which copies that step's HF export to `RUN_ARTIFACT_S3_URI/best_hf/`. `complete-run` writes the completion receipt that supervisors check.
- Without early stopping, `best_score` is null and **best_hf = the final step**. Republish the best-val step by hand: [`scale_train/publish_qwen3_opd_best_hf.sh`](../scale_train/publish_qwen3_opd_best_hf.sh), [`scale_train/republish_best_hf.py`](../scale_train/republish_best_hf.py).
- Hub pushes:
  - Off for Qwen runs (`HF_PUSH_ENABLE=False`).
  - Off by default for Gemma runs (`launch_gemma4_rl_band_seed.sh` L133, since 2026-09-15).
  - The 12B launcher does not set `HF_PUSH_ENABLE`, so it inherits `run_dapo_rl.sh`'s default `True` / `HF_PUSH_REQUIRED=True`.

### 1.5 Initialization and weight-only continuation

Initialization (run-file L337-412):

- **Default:** the pinned Hub base `GEMMA4_MODEL@GEMMA4_MODEL_REVISION`. The variable is named `GEMMA4_*` for Qwen too.
- **`INIT_MODEL_S3_URI`** (Qwen launcher L76, which maps it to `GEMMA4_INIT_MODEL_S3_URI`): initialize from an HF export on S3, for example a distilled student's `global_step_1000/huggingface`. For Gemma 4 the base model's `processor_config.json` is added, because Gemma 4 must load through the unified VLM class.
- **`GEMMA4_INIT_MODEL_REPO` / `_REVISION` / `_SUBFOLDER`**: a pushed Hub export subfolder, used by the 12B launcher.

**Weight-only continuation** is for a lost Adam checkpoint that still has an HF export:

1. Initialize from the export.
2. Give the run a new S3 prefix and W&B id.
3. Set a short warmup.
4. Set `DATALOADER_SKIP_BATCHES=<steps already done>`. `_fast_forward_fresh_dataloader` (`dapo_ray_trainer.py` L373-398) replays the seeded sampler to the old data cursor. It refuses to run if a checkpoint was restored, and the run-file resets it to 0 once one exists (L539-542).

### 1.6 Per-size memory layouts

These are all single-node layouts on H100-80GB:

- FSDP2 actor (`ACTOR_FSDP_SIZE=-1`, so DP = GPUs).
- vLLM rollout colocated on the same GPUs.
- `use_dynamic_bsz=False`, with a fixed micro-batch plus a padded-token cap. Long samples run alone: run-file L192-200 and L609-614.
- The old-logprob and ref passes can use a larger forward-only micro-batch (`INFER_MICRO_BATCH_SIZE_PER_GPU` / `INFER_MAX_PADDED_TOKENS_PER_MICROBATCH`).
- KV cache is set in bytes, which overrides the util-based sizing.

Qwen3 ([`launch_qwen3_30b_a3b_medium_rl.sh`](../scale_train/launch_qwen3_30b_a3b_medium_rl.sh) L40-68). It is the RL launcher for **every** `QWEN_SIZE` despite its name. `BAND=easy|medium|hard|very_hard`.

| Size | GPUs | GEN_TP | KV / engine | MBS / padded cap | OFFLOAD | vLLM sleep | Other |
|---|---|---|---|---|---|---|---|
| 30B-A3B | 8 | 2 (4 engines) | 12 GiB | 4 / 8192 | True (phase-level) | on | R3 router replay, `Qwen3MoeDecoderLayer` |
| 14B | 8 | 2 (4 engines) | 16 GiB per GPU | 4 / 8192 | False | on | weight-sync bucket 4096 MB; added 2026-10-26 (`es10` runs: cap 1000, patience 10) |
| 8B | 4 | 1 | 16 GiB | 4 / 8192 | False | on | weight-sync bucket 4096 MB |
| 4B | 4 | 1 | 24 GiB | 4 / 8192 | False | on | (easy runs: `MBS=8 PADDED_TOKENS=16384`) |
| 1.7B | 2 | 1 | 24 GiB | 4 / 8192 | False | on | (easy runs: 8 / 16384) |

All Qwen sizes use `ROLLOUT_GPU_MEMORY_UTILIZATION=0.10`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and
`VERL_VLLM_PORT_BASE=54000`. Dense sizes wrap `Qwen3DecoderLayer` (run-file L43-46).

Gemma 4. The E2B / E4B layouts are in [`gemma4/launch_gemma4_rl_band_seed.sh`](../scale_train/gemma4/launch_gemma4_rl_band_seed.sh) L105-114. The 12B layout is in [`gemma4/launch_gemma4_12b_distilled_rl_medium.sh`](../scale_train/gemma4/launch_gemma4_12b_distilled_rl_medium.sh) L164.

| Size | GPUs | GEN_TP | rollout util / KV | MBS / padded cap | Offload | vLLM sleep | Wrap class |
|---|---|---|---|---|---|---|---|
| E2B | 4 | 1 | 0.25 / 0.5 GiB | 8 / 12288 | none | off | `Gemma4TextDecoderLayer` |
| E4B | 4 | 1 | 0.25 / 1 GiB | 4 / 8192 (since 2026-09-17; was 1 / 4096) | `OFFLOAD=True` | on | `Gemma4TextDecoderLayer` |
| 12B | 8 | 1 | 0.45 / 5 GiB | 1 / 4096 | `FSDP_CPU_OFFLOAD_POLICY=True` | off | `Gemma4UnifiedTextDecoderLayer`, `VERL_SKIP_VLLM_MM_WEIGHT_RELOAD=1` |
| 26B-A4B | 8 | — | — | — | — | — | `Gemma4TextDecoderLayer`, requires R3 (run-file L228-233); its launcher is no longer in the tree |

The 2026-09-19 Gemma seeds 45-47 ran with a 4× batch (`TRAIN_PROMPT_BSZ=256`, `TRAIN_PROMPT_MINI_BSZ=128`,
`WANDB_TAG=bs256-v1`). Every other Gemma 4 run uses 64 / 32.

### 1.7 Gemma 4 specifics vs Qwen3 specifics

Gemma 4. These overrides are applied for every model, but only Gemma 4 needs them:

- `actor_rollout_ref.model.use_remove_padding=False` and `override_config.attn_implementation=sdpa` (run-file L585-586). There is no flash-attn varlen path for Gemma 4, and it has no monkey patch.
- `rollout.checkpoint_engine.update_weights_bucket_megabytes=8192` (L587). The per-layer embedding table is a single ~4.5 GB tensor, which overflows the 2 GB default weight-sync bucket.
- Chunked entropy + entropy checkpointing (L593-605). A monolithic `logits.float()` over a 262k vocab OOMs.
- `VERL_GEMMA4_CUDNN_SDPA` (default 1, L580-581): the FSDP engine forces cuDNN SDPA on for Gemma 4 training forwards unless this is 0. `TORCH_CUDNN_SDPA_ENABLED=0` alone does not reach `update_actor`. Recommendation: 0 on E4B (it is slower there and hit a cuDNN graph failure at seed 44 step 79), 1 on 26B-A4B (about 7% faster). The E4B launcher only takes it through `EXTRA_ENV_VARS`.
- Teacher and init exports must carry the base `processor_config.json`, so the unified (VLM-class) vLLM load is used. Never use a bare causal-LM override.
- 12-shot prompt (§0.2).

Qwen3:

- 0-shot ChatML with the boxed instruction appended in data prep, plus the stop ids/strings in §0.2.
- Rank-0 broadcast load.
- 30B-A3B only: R3 router replay (`ROUTER_REPLAY_MODE=R3`, `verl/models/transformers/qwen3_moe_router_replay.py`) and `GEN_TP=2`. At TP 1 the 61 GB of bf16 weights leave no headroom at weight sync.

### 1.8 Launch

```bash
cd rl-distill-scripts/scale_train
# Qwen3 (any size / band); supervised (relaunch on preemption) -- see OPERATIONS.md
QWEN_SIZE=4b BAND=medium bash start_qwen3_30b_a3b_medium_rl_supervisor.sh
QWEN_SIZE=1.7b BAND=hard RUN_TAG=es5 bash launch_qwen3_30b_a3b_medium_rl.sh        # one unsupervised submission
#   knobs: TOTAL_TRAINING_STEPS, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_RESET_PATIENCE_AT_STEP, INIT_MODEL_S3_URI,
#          MBS / PADDED_TOKENS, ALLOW_BORROWING (default true), ROLLING_CHECKPOINT_ENABLED, ASYNC_CHECKPOINT_UPLOAD
# Gemma 4 E2B / E4B (any band / seed); supervisors: gemma4/start_gemma4_rl_seed_supervisors.sh
SIZE=e4b BAND=medium SEED=43 bash gemma4/launch_gemma4_rl_band_seed.sh
# Gemma 4 12B layout (defaults to RL from the E4B-base-distilled 12B student export)
bash gemma4/launch_gemma4_12b_distilled_rl_medium.sh
```

Naming:

- Qwen run tag `es5` gives:
  - S3 prefixes `qwen3-<size>-base-rl{,-full-checkpoints}`;
  - W&B id `q3-<size>-<band>-es5-s42-v1`;
  - experiment name `DAPO-qwen3-<size>-base-pt-DeepScaleR-qwen30b-<band>-seed42-es5`.
- The full set of S3 paths is listed in [OPERATIONS.md](OPERATIONS.md).

---

## 2. Off-policy top-k forward-KL distillation (with optional tail bucket)

### 2.1 Pipeline

1. **Teacher traces.**
   - [`scale_train/run_gemma4_bestckpt_trace_collection.sh`](../scale_train/run_gemma4_bestckpt_trace_collection.sh) selects the teacher with a `TRACE_SPEC`. Specs cover the Gemma RL best checkpoints, the Gemma E4B base, the Qwen RL teachers `q30bm`/`q30bh`/`q30bvh`/`q4bm`/`q4bh`/`q4bvh`, and the Qwen bases `q4b-base-*` / `q1p7b-base-*`.
   - It calls [`data/generate_gemma4_distill_traces.py`](../data/generate_gemma4_distill_traces.py), which samples on vLLM with the family's RL prompt contract (§0.2).
   - Sampling: temp 1.0, top-p 1.0, top-k −1, prompt ≤ 4096, response ≤ 8192, `max_model_len` 12288 (L384-416). Each position records the sampled token ids and the **teacher's top-128 logprobs** (`logprobs=128`, stored as fp16).
   - Samples per training question: 16 for the RL-teacher v2 / Qwen / base specs, and 8 for the original Gemma v1 study specs (L254). Each validation question gets 1 sample.
   - Qwen padding rows of the LM head are suppressed with a −100 logit bias and `processed_logprobs`, so the sampled and recorded distributions match (L610-620).
   - Helpers: [`data/gemma4_distill_trace_schema.py`](../data/gemma4_distill_trace_schema.py), [`data/validate_gemma4_distill_traces.py`](../data/validate_gemma4_distill_traces.py), [`data/gemma4_trace_s3.py`](../data/gemma4_trace_s3.py), [`data/upload_gemma4_trace_bundle_hf.py`](../data/upload_gemma4_trace_bundle_hf.py).
   - Bundles are mirrored to S3 shard by shard (resumable). They are complete once `COMPLETE.json` is written.
2. **Training view.** [`data/build_gemma4_distill_training_view.py`](../data/build_gemma4_distill_training_view.py) selects the training rows (all train questions × all samples) and the validation rows (by default `VALIDATION_SOURCE=validation`: `VALIDATION_QUESTIONS` of the teacher's own validation generations, seed 42). It writes a provenance-bound `dataset_index.json`. Preflight: [`data/preflight_gemma4_distill_training_view.py`](../data/preflight_gemma4_distill_training_view.py).
3. **Runner.** [`scale_train/run_gemma4_distill_one.sh`](../scale_train/run_gemma4_distill_one.sh) resolves the bundle (local, then HF dataset, then S3), builds or caches the view, snapshots the pinned student, computes the teacher and student identity SHAs, and runs:
4. **Launcher.** [`gemma4_topk_distill_fsdp2.sh`](../gemma4_topk_distill_fsdp2.sh) does `cd rl-distill-scripts` and runs `torchrun main_full_vocab_distill_fsdp2.py` (L686-698):
   - [`main_full_vocab_distill_fsdp2.py`](../main_full_vocab_distill_fsdp2.py) is verl's `SFTTrainer` with `set_loss_fn(FullVocabKLLoss)` (L89-114).
   - [`full_vocab_distill_dataset.py`](../full_vocab_distill_dataset.py) provides `input_ids`, `loss_mask` = response mask, and the nested `teacher_topk_token_ids` / `teacher_topk_logprobs`.
   - [`full_vocab_kl_loss.py`](../full_vocab_kl_loss.py) is the loss.
   - [`config/full_vocab_distill_fsdp2.yaml`](../config/full_vocab_distill_fsdp2.yaml) is the config.

### 2.2 Objective (derived from `full_vocab_kl_loss.py`)

Notation:

- Context x_{<t}: the teacher's prompt and sampled response prefix. The trace is fixed, so this is off-policy.
- Teacher top-k set S_t (k = 128), with stored log-probs log p_v for v ∈ S_t. These are the teacher's full-softmax log-probs at temperature 1, not renormalized over S_t.
- Student logits z, log q_v = z_v − logsumexp(z).

The per-position loss is

```
ℓ_t = Σ_{v∈S_t} p_v (log p_v − log q_v)                                     # top-k forward KL  (L458-462)
      + [tail_bucket]  p_tail (log p_tail − log q_tail)                       # (L463-466, L254-269)
p_tail = max(0, 1 − Σ_{v∈S_t} p_v)                     (clamped: fp16 storage can sum to slightly > 1)
log q_tail = logsumexp(z_{V∖S_t}) − logsumexp(z)        (logsumexp over the complement, NOT log(1 − Σ q_v))
term is 0 where p_tail = 0
```

The batch loss is a global token mean over response-prediction positions:
`L = Σ_{t∈resp} ℓ_t / batch_num_tokens × dp_size` (L536-547).

- The response mask is shifted by one inside each sample, so the logits at t−1 predict response token t. It never crosses sample boundaries (L46-52).
- Temperature is fixed at 1. The `T²` scale is 1.
- `CLAMP_MIN_TOPK_KL=false` by default, so ℓ_t is not clamped. Optional `clamp_min_kl` clamps ℓ_t ≥ 0 per position (L497-498).
- Logged: `full_vocab_kl/{mean,max,min}`, `teacher_mass` = Σ_{S_t} p, and `student_mass` = Σ_{S_t} q.

Why this form (still-true rationale from the retired plan / experiment docs):

- **Why top-k and not full vocab.** Materialising full teacher logits is tens to hundreds of TB. Gemma's vocab is 262k, with hundreds of response tokens × hundreds of thousands of rows. Running the teacher online doubles training compute. Top-128 logprobs per position are cheap to store, and teacher top-128 mass is typically about 0.96-0.99 (logged).
- **Why the tail bucket.** Without it, the top-k sum is minimised by q = the teacher head renormalised onto S_t, with **zero** tail mass. The student learns to delete the teacher's tail and ends up less entropic than the teacher. In the 2026-10-26 Qwen 4B base → 30B base reference runs:
  - student top-128 mass 0.986 vs teacher 0.970;
  - own-sample entropy stayed at the 30B base's 0.43-0.46 nats instead of the teacher's 0.96;
  - 46% of the 4B base's own-sample NLL comes from the 5.3% of sampled tokens outside its top-128.
  
  This is the known "ghost token" bias (Sparse Logit Sampling, ACL'25; TA-OPD; TAD). With the bucket, ℓ_t is the **exact** forward KL between the (k+1)-bucket coarse-grainings {S_t tokens, everything else} of p and q. It is a proper lower bound on the full forward KL, equal to it at k = |V|, and it penalises tail-mass deletion. With the bucket on, student top-128 mass tracked the teacher's (0.961 vs 0.964).
- **Why logsumexp over the complement for log q_tail.** `log(1 − Σ q_v)` cancels catastrophically once the student puts nearly all its mass on S_t, and it needs a clamp with a dead gradient (L262-263).
- **Memory.**
  - The student LM head is applied to compacted active hidden states in chunks of `FULL_VOCAB_KL_CHUNK_SIZE` positions (`use_hidden_logits_processor`).
  - Each chunk is activation-checkpointed (`CHECKPOINT_DISTILL_CHUNKS=true`, L501-513).
  - Logits are cast to fp32 per chunk. So a 262k-vocab logits tensor is never materialised for the whole micro-batch.
  - The chunk size changes memory only, not the objective.

**Which runs used the tail bucket.** `TOPK_KL_TAIL_BUCKET` defaults to `false` (`gemma4_topk_distill_fsdp2.sh` L425). It is turned on only for the Qwen band-union base→base specs `q1p7b-base-all` / `q4b-base-all`
(`run_qwen3_30bm_to_4b_distill_st.sh` L82, from 2026-10-26). All Gemma 4 v1/v2 distillations and the Qwen RL-teacher
distillations (q30b*/q4b* → q4b/q1p7b) were trained **without** it. A new off-policy run should set
`TOPK_KL_TAIL_BUCKET=true` explicitly.

### 2.3 Hyperparameters

| Knob | v2 recipe (current: Qwen, Gemma 12B/26B teachers, E4B-base control) | v1 Gemma study (21-run grid, 2026-09) |
|---|---|---|
| traces | 3000 train q × **16** + 300 val q × 1 (band union: 9000 × 8 + 900 × 1) | 3000 × **8** + 300 × 1 |
| global batch | **128** sequences | 64 |
| steps / epochs | **1000** (epochs cap 100, non-binding) | 500 (cap 100) |
| LR | **2e-6**, 100 warmup, **linear → 2e-7** (`MIN_LR_RATIO=0.1`) | 2.5e-6, 100 warmup, linear → 2.5e-7 |
| optimizer | AdamW, weight decay 0.1, betas (0.9, 0.98), grad clip 1.0 | same |
| val KL | every 10 steps, on 128 teacher val generations (band union: 300) | every 10 on 128 |
| checkpoints | permanent (fp32 model + Adam + extra + HF) every **250** to S3; Gemma runners also roll every 50 with the HF export | final step only, HF export only, pushed to `JWei05/Distill-gemma4-<spec>-to-<student>-base/step_000500` |
| precision | `MODEL_DTYPE=fp32` master, bf16 FSDP params, fp32 reduce / buffers, activation checkpointing, sdpa, no remove-padding | same |

Sources:

- v2: `run_qwen3_30bm_to_4b_distill_st.sh` L77-88; `gemma4/run_gemma4_12br_trace_and_distill_st.sh` L64-72; `gemma4/run_gemma4_e4b_base_distill_st.sh` L76-87.
- v1: the defaults of `run_gemma4_distill_one.sh` L293-321, which [`gemma4/run_gemma4_distill_queue.sh`](../scale_train/gemma4/run_gemma4_distill_queue.sh) uses unchanged.
- Shared: `gemma4_topk_distill_fsdp2.sh` L406-458, L550-559, L645-651.

Per-student layout (`run_gemma4_distill_one.sh` L33-55, plus runner overrides):

| Student | GPUs (min) | wrap class | KL chunk | micro-batch / padded cap |
|---|---|---|---|---|
| Qwen3-1.7B | 2 | `Qwen3DecoderLayer` | 4096 | 4 / 8192 (Qwen runner) |
| Qwen3-4B | 4 (runner allows 2) | `Qwen3DecoderLayer` | 4096 | 4 / 8192 |
| Qwen3-30B-A3B | 8 | `Qwen3MoeDecoderLayer`, `VERL_FSDP2_WRAP_EMBED_LM_HEAD=0` (untied) | 1024 | 1 / 4096 |
| Gemma 4 E2B | 2 | `Gemma4TextDecoderLayer` | 2048 (4096 OOMed on 2 GPUs) | 1 / 4096 |
| Gemma 4 E4B | 4 | `Gemma4TextDecoderLayer` | 4096 | v2 runners 4 / 8192; v1 1 / 4096 |
| Gemma 4 12B | 4 (runner) / 8 (min) | `Gemma4UnifiedTextDecoderLayer` | 2048 | 1 / 4096 |
| Gemma 4 26B-A4B | 8 | `Gemma4TextDecoderLayer` | 1024 | 1 / 4096 |

### 2.4 Launch

Qwen3: traces, distillation and band-val evals run in one job. Each stage resumes from S3. The entry point is
[`scale_train/run_qwen3_30bm_to_4b_distill_st.sh`](../scale_train/run_qwen3_30bm_to_4b_distill_st.sh).

| `TEACHER_SPEC` | teacher | band | `STUDENT` |
|---|---|---|---|
| `q30bm-medium` / `q30bh-hard` / `q30bvh-very_hard` | 30B-A3B RL best | medium / hard / very_hard | `q4b`, `q1p7b` |
| `q4bm-medium` / `q4bh-hard` / `q4bvh-very_hard` | 4B RL best | same | `q1p7b` |
| `q1p7b-base-all` / `q4b-base-all` | untrained 1.7B / 4B base (pinned Hub) | easy ∪ medium ∪ hard, 8 samples/q, **tail bucket on** | `q4b` / `q1p7b` |

```bash
cd rl-distill-scripts/scale_train
TEACHER_SPEC=q4bh-hard STUDENT=q1p7b GPUS=2 bash start_qwen3_30bm_to_4b_distill_supervisor.sh   # supervised
TEACHER_SPEC=q30bm-medium STUDENT=q4b bash launch_qwen3_30bm_to_4b_distill_st.sh               # one submission (GPUS=4, borrowing on)
```

The Qwen job's stages:

- Stage 3 evaluates steps 250/500/750/1000 with the band-eval protocol (n = 16). Band-union specs evaluate steps 0/250/500/750/1000 on all three bands at n = 32, plus own-sample NLL.
- Outputs:
  - `s3://scale-ml/genai/rl-distill/<family>-traces-topk128-v1/<spec>`
  - `<family>-distill-ckpts-v1/<spec>-to-<student>-bs128-s1000-lr2e-6/global_step_<N>/huggingface`
  - `<family>-distill-evals-v1/...`

Gemma 4:

- [`gemma4/run_gemma4_12br_trace_and_distill_st.sh`](../scale_train/gemma4/run_gemma4_12br_trace_and_distill_st.sh): an RL teacher (`12br`/`12bo`/`26bd`/`26bw` specs) into the E4B base. It does traces + v2 distillation in one 4-GPU job; launcher [`gemma4/launch_gemma4_12br_trace_distill_st.sh`](../scale_train/gemma4/launch_gemma4_12br_trace_distill_st.sh).
- [`gemma4/run_gemma4_e4b_base_distill_st.sh`](../scale_train/gemma4/run_gemma4_e4b_base_distill_st.sh): E4B-base traces into the 12B / 26B students (`--env-vars "TEACHER_SPEC=e4b-base-medium,STUDENT=12b"`, 4 or 8 GPUs).
- [`gemma4/run_gemma4_distill_queue.sh`](../scale_train/gemma4/run_gemma4_distill_queue.sh): the v1 study queue, a local GPU pool. Entries are `spec:student:gpus[:VAR=val;...]`.
- One run by hand: `TEACHER_SPEC=12b-easy STUDENT=e4b DISTILL_GPU_IDS=0,1,2,3 bash scale_train/run_gemma4_distill_one.sh`.

---

## 3. On-policy distillation, policy-gradient form (PG-k1)

### 3.1 Objective

The student samples y ~ π_θ with vLLM. A colocated vLLM **teacher** scores every sampled token with `prompt_logprobs` over
prompt + response, in a single prefill with `max_tokens=1` (`verl/experimental/teacher_loop/teacher_manager.py` L31-44). For
`k1`, `num_logprobs = 0`, so only the sampled token's log-prob is returned. The per-token reverse-KL estimate is used as a
**per-token advantage** (`verl/trainer/distillation/losses.py`):

```
k1_t = log π_θ(y_t) − log π_T(y_t)                         (kl_penalty "k1" = logprob − ref_logprob)
A_t  = − clip(k1_t, −5, 5) = sg[ clip(log π_T(y_t) − log π_θ(y_t), −5, 5) ]      (loss_max_clamp=5, L297-299; advantages=−loss.detach(), L301-317)
L    = token-mean over response tokens of  w_t · max(−A_t ρ_t, −A_t clip(ρ_t, 0.8, 1.2))   (vanilla PPO loss, L303-317)
w_t  = r_t if 0.5 ≤ r_t ≤ 2 else 0,   r_t = π_old(y_t)/π_vLLM(y_t)                          (IcePop mask, rollout_corr_helper.py L566-601)
```

How the terms fit together:

- **No task reward.** `use_task_rewards=False` zeroes the PPO task term, and the distillation coefficient becomes 1.0 (L255-263). The math reward is still computed, but only for the accuracy metrics.
- **No advantage normalisation, no discounting, no sum over future tokens.** A_t is the token's own k1.
- **One update per batch.** π_old is the trainer's recompute just before the update, so ρ_t ≈ 1 and the PPO clip is inactive. That clip uses `DistillationLossConfig`'s 0.2 / 0.2 (`verl/workers/config/distillation.py` L71-75), not the actor's 0.2 / 0.28.
- **Gradient.** −E_{y~π_θ}[ Σ_t (log π_T(y_t) − log π_θ(y_t)) ∇log π_θ(y_t) ]. This is the per-token (non-reward-to-go) policy-gradient form of minimising the reverse KL(π_θ ‖ π_T).
- **`k1` cannot be backpropagated directly.** Its gradient with respect to θ does not depend on the teacher. The config raises if `k1` is used without `use_policy_gradient` (`distillation.py` L107-111).
- **Logged:** `distillation/k1` (mean k1), `distillation/k3` (`exp(Δ) − Δ − 1`, a non-negative unbiased estimate of the reverse KL), and `distillation/abs_loss`. Under student sampling, mean k1 estimates +KL(π_θ ‖ π_T) and is ≥ 0 in expectation.

Why this form (still-true rationale from the retired OPD write-up):

- Sampled-token reverse KL as a per-token advantage is what most frontier reports use: Thinking Machines, MiMo/MOPD, GLM-5, Nemotron-Cascade 2 / Nemotron 3 Ultra, Kimi K3. MOPD (arXiv 2606.30406) finds it is about as good as top-k logit distillation and more robust when teacher and student distributions differ.
- The recipe is Nemotron-Cascade 2 OPD plus MOPD's [−5, 5] advantage clip. IcePop is taken from Nemotron-Cascade 2 / GLM-5.
- Reverse KL is mode-seeking. It pulls the forward-KL (mode-covering) off-policy students back to about the teacher's entropy and diversity.
- The teacher needs only one log-prob per token, so there is no top-k payload.
- Two more consequences of `ROLLOUT_IS_THRESHOLD=0.5_2.0`:
  - Tokens whose trainer/vLLM ratio falls outside [0.5, 2] get **zero** weight. They are masked, not clamped.
  - In-range tokens keep weight r_t, not 1.

### 3.2 Wiring

- **Run-file** `ONPOLICY_DISTILL_*` section (L422-510):
  - Pins and materialises the teacher, either `ONPOLICY_DISTILL_TEACHER_S3_URI` (an HF export) or `ONPOLICY_DISTILL_TEACHER_REPO@REVISION`.
  - Emits `++distillation.*` overrides:
    - `teacher_model.enable_resource_pool=False` and `n_gpus_per_node=<all GPUs>`, so the teacher is **colocated** on the actor's GPUs;
    - `inference.tensor_model_parallel_size`;
    - `gpu_memory_utilization`;
    - `enforce_eager=True`;
    - `temperature=1.0`;
    - `max_model_len = prompt + response + 1`;
    - `enable_sleep_mode` / `free_cache_engine=True` (level-1 sleep parks the teacher in host RAM between scoring calls);
    - `engine_kwargs.vllm.max_logprobs`;
    - `distillation.num_workers=8`.
- **Trainer:**
  - `DAPOTaskRunner` adds the teacher resource pool (`main_dapo.py` L168).
  - In `RayDAPOTrainer.fit`, right after rollout, `_should_compute_teacher_colocate` → `_compute_teacher_colocate` merges `teacher_logprobs` into the batch (`dapo_ray_trainer.py` L824-829). Both come from `verl/trainer/ppo/ray_trainer.py` L517-526.
  - The actor loss is `distillation_ppo_loss` (`losses.py` L199-266).
- **Fork fix 07e9460f.** It sets `global_batch_info` (dp size, batch tokens) before the loss (`losses.py` L246-252). Without it, the first micro-batch of a run was normalised by its own token count, and the first micro-batch of every later step by the previous step's count. The v1 batch (`RUN_TAG=pgk1-v1`) is pinned to `CODE_REF=5c6d47d5`, which predates the fix (`launch_qwen3_onpolicy_distill.sh` L118-120). Every later batch runs on HEAD.

### 3.3 Hyperparameters

The common settings come from [`launch_qwen3_onpolicy_distill.sh`](../scale_train/launch_qwen3_onpolicy_distill.sh) L79-117:

- Loss: `ONPOLICY_DISTILL_LOSS_MODE=k1`, `USE_POLICY_GRADIENT=True`, `USE_TASK_REWARDS=False`, `LOSS_COEF=1.0`, `LOSS_MAX_CLAMP=5.0`, `LOG_PROB_MIN_CLAMP=null`, `TOPK=1`.
- IcePop: `ROLLOUT_IS_THRESHOLD=0.5_2.0`.
- Batch: `N_RESP_PER_PROMPT=1`, `TRAIN_PROMPT_BSZ = TRAIN_PROMPT_MINI_BSZ = GEN_PROMPT_BSZ = 128`, so there are **128 prompts × 1 sample and one optimizer step per training step**.
- No filter groups, no overlong penalty.
- Everything else is the Qwen RL contract:
  - 0-shot ChatML, prompt 4096 / response 8192 / max_model_len 12288, temp 1 / top-p 1 / top-k off;
  - validation on the band's val300 × 16 (mean@16) before training and every `TEST_FREQ`;
  - rolling checkpoint every 5 and permanent (HF export) every `SAVE_FREQ`, then `best_hf/`;
  - Hub push off.
- Data: the student trains on the **teacher's band** train split (`DIFFICULTY_DATASET_SOURCE=qwen3_30b_bands`, `DIFFICULTY_DATASET=<band>`).

**Standard = v1** (user, 2026-10-27). v2 was an ablation that did not improve on v1 (slower at every step; 4B → 1.7B best val easy 73.9 vs 74.2, medium 36.1 vs 38.0, hard 18.4 vs 19.5; QWEN3_RESULTS §5.5).

| | **v1 = standard** (`pgk1-v1`, 2026-10-20; `pgk1-base-v1` / `pgk1-pre-v1`, 10-23 → 10-25) | v2 ablation (`pgk1-base-lr1e6w50`, 2026-10-26, not adopted) |
|---|---|---|
| LR schedule | **2e-6, 20-step linear warmup, constant** (`OPD_LR` / `OPD_LR_WARMUP_STEPS` defaults) | 1e-6, 50-step warmup, constant |
| steps | 120 (`TOTAL_TRAINING_STEPS` default; first four runs launched at 150 and stopped at 120); from-base sweep / grid 250 | cap **400** |
| early stopping | off (fixed steps); the 10-25 extensions used the v2 rule | **on**: patience 10 evals × `TEST_FREQ` 10 = "no +0.5-pt gain in 100 steps" (`min_delta=0.005`, no miss tolerance) |
| val / permanent ckpt | every 10 (from-base sweep: val every 5) | every 10 |
| borrowing | false (supervisor default) | **true**, priority high, `MAX_RELAUNCHES=200` |
| student init | v1: off-policy distilled student `global_step_1000` (`<spec>-bs128-s1000-lr2e-6`); `-base` suffix → untrained base | untrained base (all 7 specs) |
| code | `5c6d47d5` (pre-fix) for `pgk1-v1`; HEAD afterwards | HEAD (`8186d486` at launch) |

v2 sources: [`launch_qwen3_opd_recipe_v2_batch.sh`](../scale_train/launch_qwen3_opd_recipe_v2_batch.sh) L135-144. Why v2 tried a lower LR: at 2e-6 / 20 warmup the first ~50 steps are volatile (grad norm ~3, response length 2.8k ± 0.8k, 8% truncation). That transient did not hurt v1's final accuracy, and halving the LR cost 2-4 points at every matched step, so v1 stays the standard.

Because `min_delta` is set, the early-stopping "best" lags the true maximum. Re-publish `best_hf` at the best saved
checkpoint after a v2 run finishes.

**Teachers** (`launch_qwen3_onpolicy_distill.sh` L26-66). `SPEC=<teacher>[-s<STEP>]-to-<student>[-base]`:

| Teacher tag | Source | vLLM TP | teacher util | Notes |
|---|---|---|---|---|
| `q1p7b{e,m,h}` | 1.7B RL `best_hf`, or `global_step_<STEP>` with `-s<STEP>` | 1 | 0.15 | `-base` students only (same-size control) |
| `q4b{e,m,h,vh}` | 4B RL `best_hf` / `-s<STEP>` | 1 | 0.20 | 8 GiB bf16 weights |
| `q8b{e,m,h}` | 8B RL `best_hf` / `-s<STEP>` | 1 | 0.32 | 16.4 GiB weights + ~9 GiB KV; `-base` students only |
| `q30b{m,h,vh}` | 30B-A3B RL `best_hf` / `-s<STEP>` | **2** | 0.45 | about 30.5 GiB/GPU weights; forces actor `OFFLOAD=True` |
| `qb1p7b{e,m,h}` / `qb4b{e,m,h}` | **untrained** base, pinned Hub revision (`ea980cb0` / `906bfd4b`) | 1 | 0.15 / 0.20 | no `-s<STEP>`; the band letter picks the prompts. Used for the abandoned on-policy pre-distillation |

The intermediate-checkpoint teacher path is `qwen3-<size>-base-rl-full-checkpoints/qwen30b-<band>-es5/global_step_<STEP>/actor/huggingface`.

**Students:**

| Student | GPUs | Init |
|---|---|---|
| `q1p7b` | 2 | the off-policy distilled `global_step_1000` export |
| `q4b` | 4 | the off-policy distilled `global_step_1000` export (30B teachers only) |
| `q1p7b-base` / `q4b-base` | 2 / 4 | the pinned base |

**Student layout:** `GEN_TP=1`, vLLM sleep on, MBS 4 / 8192 padded cap, `ROLLOUT_GPU_MEMORY_UTILIZATION=0.10`, 8 GiB KV
(`n=1` needs little KV), no FSDP CPU offload policy.

**v2 ablation spec list** (dry-run output 2026-10-26, borrowing true, priority high):

| Pair | Specs | GPUs |
|---|---|---|
| 4B → 1.7B | `q4be-s500` / `q4bm-s510` / `q4bh-s560` `-to-q1p7b-base` | 2 |
| 8B → 1.7B | `q8be-s460` / `q8bm-s480` `-to-q1p7b-base` | 2 |
| 8B → 4B | `q8be-s460` / `q8bm-s480` `-to-q4b-base` | 4 |

Exclusions: no 30B teachers, no same-size controls, and the 8B hard teacher waits for its RL rerun.

### 3.4 Launch

```bash
cd rl-distill-scripts/scale_train
# standard (v1) run: lr 2e-6 / 20 warmup are the defaults; give it >= 400 steps (v1 bests landed at 280-430)
SPEC=q4bm-s510-to-q1p7b-base RUN_TAG=my-tag TOTAL_TRAINING_STEPS=400 bash start_qwen3_onpolicy_distill_supervisor.sh
# the v2 ablation batch (kept for the record, not the standard recipe):
bash launch_qwen3_opd_recipe_v2_batch.sh            # dry run: prints each v2 spec's resolved env (safe)
```

Outputs:

- S3: `s3://scale-ml/genai/rl-distill/qwen3-opd{,-full-checkpoints}/<SPEC>-<RUN_TAG>`.
- W&B: `q3-opd-<SPEC>-<RUN_TAG>-s42` in project `DAPO`.

[`stop_qwen3_opd_at_step.sh`](../scale_train/stop_qwen3_opd_at_step.sh) stops a run cleanly at a given step: it waits for that step's validation, checkpoint and W&B flush, then publishes `best_hf`. Supervision is described in [OPERATIONS.md](OPERATIONS.md).

### 3.5 Retired: Gemma 4 on-policy distillation (supervised reverse KL on the student's top-k)

The Gemma 4 on-policy runs of 2026-09 (distilled 12B → E4B base teacher) used a different, **supervised** loss through
the same run-file. That recipe is no longer used, and its launchers are no longer in the tree.

- Losses: `reverse_kl_student_topk` and, later, `reverse_kl_student_topk_bucket` (`verl/trainer/distillation/losses.py` L422-458).
- Settings: `use_policy_gradient=False`, lr 5e-7, 20 warmup, 128 × 1, 200 steps.
- Per position: Σ over the **student's** top-128 of q (log q − log p), plus optionally the bucket term (1−Q_k)(log(1−Q_k) − log(1−P_k)). Q_k and P_k are the masses that the student and teacher put on that support.
- The teacher was a frozen HF model run inside the actor update (`verl/trainer/distillation/fsdp/teacher_in_actor.py`), not a vLLM server.

Lessons that still hold:

- A truncated reverse KL on the **teacher's** top-k (`reverse_kl_topk`, the run-file's default `ONPOLICY_DISTILL_LOSS_MODE`) can be lowered by moving student mass off the teacher's support.
- The plain student-top-k sum *rewards* spreading mass into the tail. The student collapsed after about 130 steps.
- The bucket term turns the sum into a proper lower bound on the full reverse KL that rises with tail mass.

---

## 4. Evaluation protocols

- **RL / OPD in-loop validation.**
  - Band val300 × 16 at temp 1 / top-p 1 / 8192 tokens, graded by the RL reward. `val-core/math/acc/mean@16` is exact.
  - **Caveat:** verl's `best@N` / `maj@N` come from `bootstrap_metric` (`verl/trainer/ppo/metric_utils.py` L420-460). It resamples N of the n responses **with replacement** (1000 bootstraps), so the logged `best@16` ≈ pass@10, not pass@16.
  - Compare `best@N` only with other RL logs. Use the offline evals below for real pass@k.
- **Qwen band evals / pass@k.**
  - Generator: [`data/generate_deepscaler_4bpt_samples.py`](../data/generate_deepscaler_4bpt_samples.py). It uses vLLM with the Qwen prompt contract (0-shot ChatML + boxed instruction, the §0.2 stops, temp 1, top-p 1, top-k −1, 8192 tokens, `TRITON_ATTN`).
  - Grader: [`data/grade_band_eval_samples.py`](../data/grade_band_eval_samples.py), strict last-box `math_verify` (the RL reward).
  - The summary reports mean@n, unbiased pass@k = 1 − C(n−c, k)/C(n, k) for every k ≤ n, and pass^k (`run_qwen3_30bm_to_4b_distill_st.sh` L158-172).
  - Run-files:
    - [`scale_train/run_qwen3_band_evals_st.sh`](../scale_train/run_qwen3_band_evals_st.sh): n = 16 on teacher-band sets;
    - [`scale_train/run_qwen3_passk32_evals_st.sh`](../scale_train/run_qwen3_passk32_evals_st.sh): n = 32, TP-2 engines;
    - [`scale_train/run_qwen3_math_suite_st.sh`](../scale_train/run_qwen3_math_suite_st.sh): easy / medium / hard (/ very_hard) val, GSM8K, MATH500, AIME 2025 / 2026; n = 16, AIME 32. `SINGLE_PASS=1` packs one model's sets into one vLLM call with [`data/math_suite_single_pass.py`](../data/math_suite_single_pass.py). The model registry is [`scale_train/qwen3_math_suite_models.sh`](../scale_train/qwen3_math_suite_models.sh), the summary script is [`eval/summarize_qwen3_math_suite.py`](../eval/summarize_qwen3_math_suite.py), and launchers / supervisors are `launch_qwen3_*` / `start_qwen3_*`.
- **Own-sample NLL** (entropy estimate). [`data/score_own_sample_nll.py`](../data/score_own_sample_nll.py) teacher-forces a model's own temperature-1 samples through the same model with vLLM `prompt_logprobs`. Mean −log p per response token = E_{y~π}[−log π(y)], the per-token entropy along its own trajectories. This is the same estimator as `rollout_corr/training_log_ppl` in RL / OPD logs. Run-file: [`scale_train/run_qwen3_own_nll_st.sh`](../scale_train/run_qwen3_own_nll_st.sh).
- **Gemma 4 eval stack.**
  - [`eval/eval_math_passk.py`](../eval/eval_math_passk.py): deterministic 64 samples per question, temp 1 / top-p 1, 8192 tokens, the 12-shot prompt. Unbiased without-replacement pass@k comes from [`eval/gemma4_eval_metrics.py`](../eval/gemma4_eval_metrics.py) (L75-97). Throughput note: run with `EVAL_PREDICTIVE_TOPK_WIDTH=0` (top-128 logprobs make vLLM CPU-bound) and 2 models per GPU with a 16 GiB KV budget each.
  - [`eval/run_gemma4_math_4gpu.py`](../eval/run_gemma4_math_4gpu.py): the registered math matrix on 4 GPUs.
  - [`eval/eval_gemma4_ood.py`](../eval/eval_gemma4_ood.py): out-of-distribution evals with the vendored `lm-evaluation-harness` (0.4.13.dev0). Profile `gemma4-report` = MMLU-Pro, GPQA-Diamond and a 14,042-item MMMLU subset. It fails closed on identity mismatches and never uses `--limit`.
  - Student-checkpoint pass@k: [`eval/eval_student_checkpoints_passk.py`](../eval/eval_student_checkpoints_passk.py) / [`gemma4/run_gemma4_student_ckpt_passk_st.sh`](../scale_train/gemma4/run_gemma4_student_ckpt_passk_st.sh).
  - Offline reverse KL of a student vs a teacher: [`eval/reverse_kl_topk.py`](../eval/reverse_kl_topk.py) / [`gemma4/run_gemma4_reverse_kl_st.sh`](../scale_train/gemma4/run_gemma4_reverse_kl_st.sh).
- Scores from the harness's own answer extraction are not comparable to the boxed-only `math_verify` numbers. All math numbers in the results docs use the repo's scorer.
