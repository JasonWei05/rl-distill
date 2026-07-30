# NeMo-RL exact-replication parity checklist

Target: replicate the verl run **DAPO-gemma4-e2b-pt-DeepScaleR-4of4strict-seed42 (8k variant)**
(`rl-distill-scripts/gemma3_pt_fewshot_math_rl.sh` via
`scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh` with `MAX_RESPONSE_LENGTH=8192
OVERLONG_BUFFER_LEN=2048`) in NeMo-RL (`third_party/nemo-rl`, commit `5f89b3ae`) as wandb run
`DAPO/nemorl-dapo-gemma4-e2b-pt-DeepScaleR-4of4strict-seed42-8k`.

Config: `config/dapo_gemma4_e2b_pt_repro.yaml` (inherits nemo-rl `examples/configs/grpo_math_1B.yaml`).
Launcher: `run_grpo_repro.py` (registers the `math_strict` env before setup).

## Matched knobs

| Knob | verl (reference run) | NeMo-RL key | Value |
|---|---|---|---|
| Model | google/gemma-4-E2B (PT) | `policy.model_name` / `policy.tokenizer.name` | `google/gemma-4-E2B` |
| 12-shot prompt | `model.custom_chat_template` = gemma3_it_fewshot_math.jinja | `policy.tokenizer.chat_template` (loads `.jinja` file) | same file; token parity gated by `tests/test_tokenization_parity.py` |
| Train data | deepscaler_4of4strict_rl_train.parquet | `data.train.data_path` (dotted-path dataset class) | same parquet, 9723 rows |
| Val data | deepscaler_4of4strict_rl_val200_x16.parquet (mean@16) | `data.validation.data_path` | same parquet; 3200 rows ⇒ `validation/accuracy` = mean@16 over 200 prompts |
| Reward | strict boxed-only `math_verify` (`verl/utils/reward_score/math_verify.py`) | env `math_strict` (`rl_distill_nemo/strict_math_env.py`, verbatim port) | gated by `tests/test_strict_reward.py` |
| Overlong soft penalty | buffer 2048, factor 1.0, max_resp 8192 | `grpo.reward_shaping.{overlong_buffer_length,overlong_buffer_penalty,max_response_length}` | 2048 / 1.0 / **8192 (explicit — default would be 12288!)** |
| Reward scaling | none | `grpo.reward_scaling.enabled` | false |
| Dynamic sampling | `ENABLE_FILTER_GROUPS=False` | `grpo.use_dynamic_sampling` + `batch_multiplier` | false / 1 |
| Prompts/step × n | 64 × rollout.n=16 | `grpo.num_prompts_per_step` × `num_generations_per_prompt` | 64 × 16 = 1024 |
| Mini-batch | ppo_mini_batch 32 prompts (512 samples) ⇒ 2 optim steps | `policy.train_global_batch_size` | 512 ⇒ 2 optim steps/rollout step |
| Loss agg | `loss_agg_mode=token-mean` | `loss_fn.token_level_loss` | true (masked-mean over global batch tokens) |
| Advantage | GRPO group mean/std (ddof=1, +1e-6) | `use_leave_one_out_baseline=false`, `normalize_rewards=true` | numerically identical |
| Clipping | low 0.2 / high 0.28 / c 10.0 | `loss_fn.ratio_clip_{min,max,c}` | 0.2 / 0.28 / 10 |
| KL / entropy bonus | KL off, entropy_coeff 0 | `loss_fn.reference_policy_kl_penalty=0` (ref policy auto-skipped); no entropy bonus exists | 0.0 / matched by construction |
| TIS | none | `use_importance_sampling_correction=false`, `truncated_importance_sampling_*=null` | all off |
| Optimizer | AdamW lr 1e-6, wd 0.1, betas (0.9,0.999), eps 1e-8, fp32 states | `policy.optimizer` torch.optim.AdamW | identical |
| LR schedule | constant with 20 warmup steps | LinearLR(1e-8→1, 20) + ConstantLR, milestones [20] | stepped once per optim... see caveat: per **rollout** step (= verl per training step) |
| Grad clip | 1.0 | `policy.max_grad_norm` | 1.0 |
| Lengths | prompt 4096 + response 8192 | `max_total_sequence_length` 12288, `max_new_tokens` 8192, `data.max_input_seq_length` 4096 | matched |
| Sampling | temp 1.0, top_p 1.0, top_k −1 (val identical) | `policy.generation.{temperature,top_p,top_k}` | 1.0 / 1.0 / null (→ −1); validation reuses training sampling params |
| Stop | `VERL_ROLLOUT_EXTRA_STOP='<end_of_turn>,<start_of_turn>'` | `policy.generation.stop_strings` | same two strings (+ their auto `stop_token_ids=[eos]`, inert for PT) |
| Seed | DATA_SEED 42 | `grpo.seed` | 42 (different RNG stream — see caveats) |
| Validation cadence | TEST_FREQ 10, val_before_train | `grpo.val_period=10`, `val_at_start=true`, `max_val_samples=3200`, `val_batch_size=3200` | matched |
| Checkpoints | SAVE_FREQ 25 | `checkpointing.save_period` | 25 |
| Checkpoint retention | verl keeps every save | `checkpointing.keep_top_k` | `null` (base yaml default of 3 would prune to top-3 by val:accuracy) |
| Gemma-4 handling | wrap Gemma4TextDecoderLayer, sdpa, no rmpad, micro-bsz 1 | recipe deltas: dtensor automodel `freeze_config` (vision+audio towers frozen), `train_micro_batch_size=1`, `logprob_batch_size=1`, `logprob_chunk_size=4096`, packing off / dynamic batching on | kept from their dapo-gemma4-e2b-it recipe |
| Epochs | `trainer.total_epochs=100` | `grpo.max_num_epochs=100` | **deviation from the interface study's `max_num_epochs=1`** — study missed verl's `trainer.total_epochs=100` (gemma3_pt_fewshot_math_rl.sh:215); 1 epoch would stop at ~151 steps |

## Flagged (not matched / needs launch-time attention)

- **`policy.generation.vllm_cfg.gpu_memory_utilization: 0.5`** — recipe value, untested at
  `max_model_len=12288` on 1n8g. verl ran 0.65 (sleep mode disabled). Retune on OOM; does not
  affect training math.
- **Config base**: inherits `grpo_math_1B.yaml` directly, NOT their gemma4 recipe — the loader's
  `_override_` marker is top-level-only, so the recipe's TE FusedAdam kwargs
  (`master_weights`, `exp_avg_dtype`, ...) would merge into and crash `torch.optim.AdamW`.
  All recipe deltas we keep are re-encoded explicitly in the yaml.
- **Verify pool warmup**: the strict env's spawn pool children re-import the heavy nemo_rl chain
  (verl's reward module is import-light), so `StrictBoxedVerifyWorker.__init__` pre-warms all
  pool workers — otherwise the first samples' 30s verify timeout could eat the import.

## Unmatchable caveats (from the interface study — keep next to any comparison)

1. **Entropy metric semantics differ**: NeMo logs `train/approx_entropy` =
   −E[(π_cur/π_gen)·log π_cur] on sampled tokens, not verl's exact full-softmax token entropy.
   Comparable in trend only (an entropy≈9 blowup still shows); never compare absolute values.
2. **grad_norm is last-of-2, not mean-of-2**: NeMo logs one pre-clip global L2 norm per rollout
   step — the value from the *last* of the 2 mini-batch updates (worker overwrites per global
   batch; rank-0) — while verl logs/averages per mini-batch. The bimodality test
   (~1.3–6.8 on truncation-free steps vs ≫80 on truncation steps) remains valid.
3. **vLLM version differs** from our verl gemma4 stack (vllm 0.25.1): sampler numerics differ.
   Expect step-0 val within binomial noise of 6.16%, not identical.
4. **Generation-length accounting for the overlong penalty**: NeMo counts assistant-message
   tokens with `include_stop_str_in_output=True` hardcoded — the `<end_of_turn>` text/tokens
   count toward the response length and are visible to the scorer (harmless for boxed-count;
   ~1-token penalty-onset delta vs verl).
5. **Sampled-data order**: both seed 42 but different RNG streams — per-step prompt sequences
   will not match. All comparisons are distributional (val curve, grad-norm vs truncation-rate
   scatter, `train/mean_gen_tokens_per_sample` trend), never step-exact.
6. **Mini-batch partitioning** is shard-then-split vs verl's split-then-shard; advantages are
   computed on the full 1024 before either, so only per-optimizer-step sample grouping differs.
7. **LinearLR start_factor is 1e-8**, not exactly 0 at step 0 (torch constraint) — negligible.
8. **Over-long prompts are masked (loss_multiplier=0), not dropped** — no-op for this data
   (max prompt ~2461 tok < 4096).

## Go / no-go gate (before committing GPU-days)

1. `tests/test_tokenization_parity.py` and `tests/test_strict_reward.py` both PASS (CPU).
2. Launch with `grpo.max_num_steps=1`:
   - **step-0 `validation/accuracy` ∈ [0.045, 0.075]** (verl baseline 6.16% @ n=3200,
     95% binomial CI ±0.8%) — outside this window = NO-GO, debug prompt/reward path first;
   - **step-1 `train/probs_ratio` AND `train/probs_ratio_clamped` ≈ 1.0 (±0.01)** — added after
     the 2026-07-30 incident: a corrupted TRAINING forward (garbage curr_logprobs) passes the
     val band while `probs_ratio_clamped` pins at 0.80. Val accuracy alone is NOT a valid gate;
   - logs show `Loading chat template from file`, env `math_strict` created, 2 optimizer steps
     per rollout step (`num_global_batches=2`), reference policy auto-skipped;
   - a `val_data_step*.jsonl` sample starts with the 12-shot prefix and ends
     `<start_of_turn>model\n`; `validation/avg_length` plausible; `train/truncation_rate > 0`
     at step 1.
3. Full-run comparison vs the verl run (bw9dcxso): `validation/accuracy` at steps 10/20/30/50/100
   (verl: 6.78 / 10.0 / 14.19 / 16.59% at 20/30/50/100), grad-norm-vs-truncation bimodality,
   `train/mean_gen_tokens_per_sample` trend.
