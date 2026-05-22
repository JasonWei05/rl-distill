# Full-Vocabulary Gemma 3 4B PT to 27B PT Distillation Plan

## Goal

Train a `google/gemma-3-27b-pt` student to match `google/gemma-3-4b-pt` teacher next-token logits on the generated DAPO dataset:

- Source dataset: `JWei05/DAPO-Gemma3-4B-PT-DAPO-17.4k`
- Train: 262,368 rows, 16 responses per original train prompt
- Validation: 1,000 rows, 1 response per validation prompt
- Useful columns already present:
  - `input_ids`: full prompt plus response token ids
  - `response_mask`: 1 on response tokens, 0 on prompt/template tokens
  - `messages`: OpenAI-style prompt/response text
  - `teacher_token_ids` and `teacher_log_probs`: sampled-response token ids/logprobs only

The requested objective is true token-level full-vocabulary KL:

```text
KL(P_4B(. | x_<t) || Q_27B(. | x_<t))
  = sum_v P_4B(v | x_<t) * (log P_4B(v | x_<t) - log Q_27B(v | x_<t))
```

The loss should be applied only where the next token is in the response. In practice this means shifting the response mask to align logits at position `t - 1` with label token `input_ids[t]`.

## Important Finding

We do not currently have an exact full-vocabulary off-policy distillation script for this dataset.

Existing code covers nearby cases:

- `rl-distill-scripts/main_distill_offpolicy.py`
  - Uses `SFTTrainer` plus `DistillSFTDataset`.
  - Optimizes only the sampled teacher token logprob from `teacher_log_probs`.
  - This is not full-vocab KL. The gradient is equivalent to supervised CE on the sampled 4B response tokens.
- `rl-distill-scripts/forward_kl_loss.py`
  - Computes `teacher_log_prob(sampled_token) - student_log_prob(sampled_token)`.
  - Useful baseline, not distribution matching.
- `verl/trainer/distillation/losses.py`
  - Has top-k distillation modes: `forward_kl_topk`, `reverse_kl_topk`.
  - Requires `teacher_logprobs` and `teacher_ids` tensors over top-k support.
  - This is sparse/top-k, not full vocab.
- `recipe/gkd/megatron`
  - Has an on-policy teacher service returning top-k logprobs and indices.
  - Megatron-only and designed for student-generated rollouts.
  - Also top-k, not full vocab.
- `Megatron-Bridge/docs/training/distillation.md`
  - Documents ModelOpt KD, which runs teacher and student forward passes and replaces the loss with logits KL.
  - This is the closest existing path for exact full-vocab online distillation.

## Storage Reality Check

Do not try to materialize full teacher logits into the Hugging Face dataset.

Gemma vocab is large. Full logits for roughly 262k rows with hundreds of response tokens per row would be tens to hundreds of TB even in bf16/float16. The practical design is to keep the existing dataset as tokenized contexts and compute teacher logits online during training.

## Preferred Plan: verl FSDP2 Online Full-Vocab KL

This is the preferred implementation if we want to stay in the training stack already used by `main_distill_offpolicy.py`.

### Why FSDP2 is feasible

The FSDP language-model engine already has the key hook we need. In `verl/workers/engine/fsdp/transformer_impl.py`, `FSDPEngineWithLMHead.prepare_model_outputs()` computes unpacked student logits, then conditionally calls:

```python
outputs = logits_processor_func(student_logits=logits_rmpad.unsqueeze(0), data=micro_batch)
```

That path is currently gated by `distillation_use_topk` and used for top-k distillation, but the contract is general enough for full-vocab KL: the hook receives differentiable student logits and can return a per-token loss tensor that gets carried into `model_output`.

### Proposed FSDP2 design

Add a new FSDP2 SFT-style entry point:

```text
rl-distill-scripts/full_vocab_distill_dataset.py
rl-distill-scripts/full_vocab_kl_loss.py
rl-distill-scripts/main_full_vocab_distill_fsdp2.py
rl-distill-scripts/gemma3_27b_pt_full_vocab_distill_from_4b_pt_fsdp2.sh
rl-distill-scripts/config/full_vocab_distill_fsdp2.yaml
```

Training flow:

1. Load the 27B PT student through the normal verl FSDP2 `TrainingWorker`.
2. Load a frozen `google/gemma-3-4b-pt` teacher in each rank/process, initially as a plain HF `AutoModelForCausalLM` in bf16 with `torch.no_grad()`.
3. Read `input_ids` and `response_mask` from `JWei05/DAPO-Gemma3-4B-PT-DAPO-17.4k`.
4. During the student forward pass, use the logits hook to compute full-vocab KL from teacher logits.
5. Backprop only through the student logits; teacher logits stay detached.

Loss:

```python
teacher_logp = log_softmax(teacher_logits / T, dim=-1)
student_logp = log_softmax(student_logits / T, dim=-1)
per_token_kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)
loss = masked_mean(per_token_kl, shifted_response_mask) * T * T
```

Masking detail:

- `response_mask[t] == 1` means token `input_ids[t]` is part of the assistant response.
- The logits at position `t - 1` predict `input_ids[t]`.
- Therefore the KL mask for logits must be `response_mask` shifted left onto prediction positions.

### Minimal code changes

1. Generalize or reuse the existing FSDP logits-processor gate:
   - quick path: set non-tensor flag `distillation_use_topk=True` even though the loss is full-vocab
   - cleaner path: rename/generalize to `use_logits_processor=True`
2. Implement a custom loss object/function with two call modes:
   - logits hook call: `loss_fn(student_logits=..., data=...)` computes `full_vocab_kl` per token
   - final loss call: `loss_fn(model_output=..., data=...)` masks and averages `model_output["full_vocab_kl"]`
3. Add a pretokenized dataset class that consumes `input_ids` and `response_mask` directly instead of re-tokenizing `messages`.
4. Add a launch script matching current FSDP2 scripts:
   - `engine.strategy=fsdp2`
   - `model.path=/path/to/gemma-3-27b-pt`
   - `teacher_model.path=/path/to/gemma-3-4b-pt`
   - `data.train_files=/home/tiger/verl/data/dapo_gemma3_4b_pt_teacher_v7_independent/train.parquet`
   - `data.val_files=/home/tiger/verl/data/dapo_gemma3_4b_pt_teacher_v7_independent/validation.parquet`

### Memory plan

Full-vocab KL is expensive because it needs teacher and student logits over vocab. To make FSDP2 practical:

- Start with `micro_batch_size_per_gpu=1`.
- Cap smoke-test `max_length` aggressively, for example 1024 or 2048.
- Compute KL in chunks over flattened active prediction positions, for example 128-512 positions per chunk.
- Select response-prediction positions before KL when possible, so prompt positions do not allocate full teacher/student log-prob tensors.
- Keep the teacher in bf16 and no-grad.
- If the 4B teacher replicated on every rank is too costly, move to a teacher FSDP/no-shard wrapper or separate teacher workers after the smoke test.

### Smoke test

Before a full run, run:

- 1 node
- 1-2 GPUs if the model fits, otherwise 8 GPUs
- 2-8 samples
- `max_length=1024`
- no checkpoint save
- validation disabled

The smoke test should verify:

- tokenizer vocab and token IDs match between 4B and 27B
- loss is finite
- gradients exist only on the 27B student
- mask alignment is correct by comparing sampled-token CE on `teacher_token_ids`
- GPU memory leaves headroom

## Plan A: Megatron-Bridge / ModelOpt Online Full-Vocab KD

This is the cleanest path if we can get Gemma 3 27B and 4B providers loading correctly through Megatron-Bridge.

### Why this path

- It is the only existing code path that is explicitly meant for full-logit KD.
- It runs both teacher and student in the same training step and backprops only through the student.
- It avoids storing teacher logits.
- It already has a KD abstraction:
  - `Megatron-Bridge/src/megatron/bridge/models/distillation_provider.py`
  - `Megatron-Bridge/src/megatron/bridge/training/post_training/distillation.py`
  - `Megatron-Bridge/examples/distillation/llama/distill_llama32_3b-1b.py`

### Implementation sketch

Add a Gemma script, for example:

```text
rl-distill-scripts/full_vocab_distill_gemma3_27b_from_4b.py
```

Core shape:

```python
from megatron.bridge import AutoBridge
from megatron.bridge.models.distillation_provider import convert_to_distillation_provider
from megatron.bridge.training.distill import distill
from megatron.bridge.training.post_training.distillation import ModelOptDistillConfig

cfg = ...  # start from an existing Gemma/Megatron training ConfigContainer
cfg.model = AutoBridge.from_hf_pretrained("google/gemma-3-27b-pt").to_megatron_provider(load_weights=True)
teacher = AutoBridge.from_hf_pretrained("google/gemma-3-4b-pt").to_megatron_provider(load_weights=True)

kd_config = ModelOptDistillConfig()
kd_config.logit_layers = ["output_layer", "output_layer"]
kd_config.intermediate_layer_pairs = []
kd_config.skip_lm_loss = True
kd_config.logit_kl_temperature = 1.0

cfg.model = convert_to_distillation_provider(cfg.model, teacher, kd_config)
distill(config=cfg)
```

### Required work

1. Verify `AutoBridge.from_hf_pretrained()` supports both Gemma 3 27B PT and 4B PT locally.
2. Build or reuse a Megatron-compatible dataset loader for pretokenized rows:
   - consume `input_ids`
   - consume `response_mask`
   - set the loss mask to shifted response-token positions
3. Make student and teacher parallelism match, because `DistillationProvider` enforces shared:
   - `tensor_model_parallel_size`
   - `pipeline_model_parallel_size`
   - `context_parallel_size`
   - `seq_length`
   - `pipeline_dtype`
4. Disable intermediate-layer KD unless we intentionally want it. 4B and 27B hidden sizes/layer counts differ, so logit-only KD is the safe first run.
5. Run a tiny smoke test:
   - max 2-8 samples
   - max sequence length capped
   - no checkpoint save
   - confirm loss is finite and only student grads update
6. Scale up with checkpoint/HF push.

### Risks

- Need a working Gemma 3 27B Megatron-Bridge provider and HF weight import path.
- ModelOpt may materialize full logits for both teacher and student at sequence length, so memory can be high.
- Teacher and student must share vocab/tokenizer. This should be verified before training.
- The Bridge training dataset path may need more adaptation than the loss path.

## Plan B: Patch verl SFT FSDP for Online Full-Vocab KL

This stays closer to the existing `rl-distill-scripts/main_distill_offpolicy.py` flow and can use the uploaded dataset directly.

### Why this path

- Reuses current FSDP2 SFT launch style.
- Reuses the exact dataset columns we already uploaded.
- Can be built incrementally from current off-policy distillation code.

### Implementation sketch

Add:

```text
rl-distill-scripts/full_vocab_distill_dataset.py
rl-distill-scripts/full_vocab_kl_loss.py
rl-distill-scripts/main_full_vocab_distill.py
rl-distill-scripts/gemma3_27b_pt_full_vocab_distill_from_4b_pt.sh
```

Dataset:

- Load `input_ids` and `response_mask` directly.
- Avoid re-applying chat templates, because the token ids are already in the desired format.
- Produce:
  - `input_ids`
  - `attention_mask`
  - `position_ids`
  - `response_mask`
  - `loss_mask` derived from shifted `response_mask`

Loss path:

- Run student forward through existing `TrainingWorker`.
- Keep student logits available via the FSDP engine's `logits_processor_func` hook.
- Run frozen teacher forward on the same `input_ids` under `torch.no_grad()`.
- Compute full-vocab KL on response-token prediction positions:

```python
teacher_logp = log_softmax(teacher_logits / T, dim=-1)
teacher_p = teacher_logp.exp()
student_logp = log_softmax(student_logits / T, dim=-1)
loss = (teacher_p * (teacher_logp - student_logp)).sum(-1)
loss = (loss * shifted_response_mask).sum() / shifted_response_mask.sum()
loss = loss * (T * T)
```

Memory controls:

- Compute the KL in sequence chunks, for example 128-512 prediction positions at a time.
- Select only response-prediction positions before full-vocab KL when possible.
- Keep teacher in bf16 and no grad.
- Start with micro-batch size 1 and dynamic batching.

### Required work

1. Extend the FSDP engine path so the loss hook can consume full student logits for selected positions, not only sampled-token logprobs.
2. Add teacher model loading in `main_full_vocab_distill.py`.
3. Make sure distributed ranks do not each redundantly allocate more teacher memory than expected.
4. Add metrics:
   - `full_vocab_kl/mean`
   - `full_vocab_kl/max`
   - `teacher_entropy/mean`
   - `student_teacher_argmax_agreement`
   - `sampled_token_ce` for comparison with the existing sampled-token baseline
5. Smoke-test on 8 rows before any full run.

### Risks

- Loading a 27B student plus 4B teacher inside the same FSDP process may be memory tight.
- The current SFT engine intentionally reduces logits to sampled-token logprobs; full-vocab KL needs a custom logits path.
- This is more repo-specific code than Plan A.

## Plan C: Top-K Approximation Using Existing verl Distillation

This is not exact full-vocab KL, but it is the fastest practical approximation using code already present.

### Why this path

- `verl/trainer/distillation/losses.py` already implements `forward_kl_topk` for FSDP and Megatron.
- `recipe/gkd/megatron` already has teacher-server code for top-k logprobs.
- Storage is manageable: top-k IDs/logprobs for K=128/256/1024 instead of full vocab.

### Implementation sketch

1. Generate top-k teacher distributions from `google/gemma-3-4b-pt` on the existing `input_ids`.
2. Store columns compatible with verl:
   - `teacher_ids`: shape `[seq_len, topk]`
   - `teacher_logprobs`: shape `[seq_len, topk]`
   - `response_mask`
3. Train `google/gemma-3-27b-pt` with:
   - `distillation.enabled=true`
   - `distillation.distillation_loss.loss_mode=forward_kl_topk`
   - `distillation.distillation_loss.use_policy_gradient=false`
   - `distillation.distillation_loss.topk=K`

### Risks

- It is sparse KL, not full-vocab KL.
- Quality depends on teacher top-k mass. Log `teacher_mass` and choose K accordingly.
- Large K increases dataset size and training IO.

## Plan D: Sampled-Token Baseline with Existing Script

This is already implementable, but it does not satisfy full-vocab matching.

Use the existing sampled-token off-policy path:

```bash
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/validation.parquet \
MODEL_PATH=/path/to/gemma-3-27b-pt \
EXP_NAME=gemma3-27b-pt-sampled-token-distill-from-4b-pt \
bash rl-distill-scripts/gemma3_12b_it_distill_offpolicy.sh
```

We would make a dedicated 27B PT launch script rather than reuse the 12B IT one directly.

This trains 27B to place probability mass on 4B-sampled answer tokens. It is a useful baseline because it is cheap and uses the current dataset as-is, but it cannot match the full 4B distribution.

## Recommendation

Given the FSDP2 preference, start with the FSDP2 online full-vocab KL path above and keep Plan D as the cheap baseline.

Suggested order:

1. Run a tiny FSDP2 sampled-token baseline with current code to confirm 27B PT SFT plumbing works on the dataset.
2. Implement the FSDP2 full-vocab smoke test with a frozen 4B teacher and 27B student.
3. If loss/memory look good, scale sequence length and batch size gradually.
4. If replicated 4B teacher memory is the blocker, shard or isolate the teacher forward.
5. If full-vocab remains too expensive, use Plan C with K=256 or K=1024 and log teacher top-k mass.

## Open Questions to Resolve Before Running

- Do `google/gemma-3-4b-pt` and `google/gemma-3-27b-pt` have identical tokenizer vocab IDs in our local environment?
- Do we want loss on response tokens only, or also prompt/template tokens? Current recommendation: response tokens only.
- Should we distill all 16 responses per prompt, or deduplicate prompts/contexts for a first run?
- What max sequence length should the first full-vocab run use? Current dataset has responses up to 20,480 tokens, which will be expensive for full-vocab KL.
- Which hardware target should be assumed for the first real run: the two H100 nodes used for data generation, or a B200 node where previous FSDP SFT scripts were run?

## Concrete Next Step

Implement a smoke-test script for the FSDP2 path:

```text
rl-distill-scripts/smoke_full_vocab_distill_gemma3_27b_from_4b_fsdp2.py
```

The smoke test should:

- load 2-8 rows from `JWei05/DAPO-Gemma3-4B-PT-DAPO-17.4k`
- truncate to a short max length
- verify tokenizer/vocab compatibility
- run frozen 4B teacher and 27B student forward
- compute masked full-vocab KL in chunks
- print loss and memory stats
- not write checkpoints

If that works, turn the smoke script into the distributed training entry point.
