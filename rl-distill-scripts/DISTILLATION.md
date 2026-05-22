# Distillation for Gemma 3 on B200

Two distillation modes for training Gemma 3 4B IT using a Gemma 3 27B IT teacher.

## Overview

| Mode | Who samples? | Loss | Status |
|------|-------------|------|--------|
| **Off-policy** | Teacher generates responses offline | Forward KL: `log p_teacher(x) - log p_student(x)` per token | Implemented |
| **On-policy** | Student generates during training | Reverse KL: `log p_student(x) - log p_teacher(x)` per token | Planned (~50 lines, verl already has 90% of it) |

Both use only the sampled token's log prob — no top-k or full distribution needed. This is cheaper and simpler than verl's built-in top-k distillation.

---

## Off-policy distillation

### Step 1: Generate teacher data

The teacher (27B) generates responses to the DAPO-Math-17k prompts. We collect the full token IDs and per-token log probs for every generated token.

**Architecture**: TP=2 per vLLM instance, data-parallel across remaining GPUs. On 2 B200 nodes (16 GPUs total): 8 shards, each using 2 GPUs.

**Teacher model**: `JWei05/dapo-gemma3-27b-it` at revision `step_000040`

**Responses per prompt**: 4

#### Run on GPU nodes

Node 0:
```bash
cd /mlx_devbox/users/jason.wei/playground/rl-distill
bash rl-distill-scripts/data/launch_teacher_gen.sh 0
```

Node 1:
```bash
cd /mlx_devbox/users/jason.wei/playground/rl-distill
bash rl-distill-scripts/data/launch_teacher_gen.sh 1
```

This launches 4 shards per node. Each shard handles ~2,125 prompts × 4 responses. Logs: `~/verl/data/teacher_gen/logs/shard_*.log`.

Monitor:
```bash
tail -f ~/verl/data/teacher_gen/logs/shard_*.log
```

#### Merge shards

After all 8 shards finish:
```bash
python3 rl-distill-scripts/data/merge_teacher_shards.py
```

Output: `~/verl/data/teacher_27b_step40_n4.parquet` (~68k rows)

Columns:
- `messages`: `[{role: "user", content: ...}, {role: "assistant", content: ...}]`
- `teacher_log_probs`: list of floats, one per generated token
- `teacher_token_ids`: list of ints, full generated token ID sequence
- `prompt_idx`: index into original dapo-math-17k dataset

#### Parquet format details

The `teacher_log_probs` list is aligned 1:1 with `teacher_token_ids`. Position `i` holds `log p_teacher(token_i | prompt, token_0..token_{i-1})` — the teacher's log probability of sampling that token given the context up to that point.

### Step 2: Train student with forward KL

Uses verl's SFT trainer with a custom dataset and loss function. No modifications to verl core.

```bash
cd /mlx_devbox/users/jason.wei/playground/rl-distill
bash rl-distill-scripts/gemma3_4b_it_distill_offpolicy.sh
```

#### Loss function

Per-token forward KL on the sampled token:

```
loss_t = log p_teacher(x_t) - log p_student(x_t)
```

Averaged over all response tokens in the batch. The gradient is identical to cross-entropy (the teacher log prob term is constant w.r.t. student params), but the loss value is a proper KL divergence — non-negative, zero when student matches teacher.

Metrics logged to wandb: `forward_kl/mean`, `forward_kl/max`.

#### How it works

1. **`distill_dataset.py`** (custom SFT dataset): Loads the teacher-generated parquet. For each sample, tokenizes the messages as usual, then places `teacher_log_probs` at the response token positions (where `loss_mask=1`). Returns a `teacher_log_probs` tensor alongside the standard `input_ids`, `loss_mask`, etc.

2. **`forward_kl_loss.py`** (custom loss): Replaces verl's default `sft_loss`. Left-shifts both `loss_mask` and `teacher_log_probs` by one token to align with the model's next-token log probs, then computes `teacher_lp - student_lp` masked to response tokens.

3. **`main_distill_offpolicy.py`** (entry point): Subclasses verl's `SFTTrainer`, swaps in the forward KL loss via `set_loss_fn()`.

4. **`config/distill_offpolicy.yaml`** (hydra config): Points `data.custom_cls` to `distill_dataset.py:DistillSFTDataset`.

---

## On-policy distillation (planned)

verl already supports estimator-based distillation losses that work with per-token log probs (`loss_mode="kl"`, `use_topk=False`). The student generates → teacher scores each token → loss = `student_logprob - teacher_logprob`.

The only missing piece: the DAPO trainer's `fit()` method overrides the base PPO trainer and doesn't call `_compute_teacher_colocate()`. Wiring that in is ~5 lines.

Changes needed:
- `dapo_ray_trainer.py`: Add `_compute_teacher_colocate()` call after rollout in `fit()`
- `config/dapo_trainer.yaml`: Add distillation config defaults
- New training script with distillation flags

---

## File inventory

All files are in `rl-distill-scripts/`:

### Data generation
| File | Description |
|------|-------------|
| `data/generate_teacher_data.py` | Runs teacher through vLLM, collects token IDs + log probs per response |
| `data/launch_teacher_gen.sh` | Launches 4 DP shards per node (TP=2 each) |
| `data/merge_teacher_shards.py` | Merges shard parquets, verifies completeness |

### Off-policy training
| File | Description |
|------|-------------|
| `distill_dataset.py` | Custom SFT dataset that loads `teacher_log_probs` from parquet |
| `forward_kl_loss.py` | Forward KL loss: `teacher_log_prob - student_log_prob` per token |
| `main_distill_offpolicy.py` | Entry point: subclasses `SFTTrainer` with custom loss |
| `config/distill_offpolicy.yaml` | Hydra config for off-policy distillation |
| `gemma3_4b_it_distill_offpolicy.sh` | Training launch script for Gemma 3 4B |

### Key design decisions

- **Pre-generated teacher data** rather than online teacher generation. Simpler, decouples teacher inference from student training, and the dataset can be reused across runs.
- **Forward KL with teacher log probs** rather than plain cross-entropy. Same gradients, but the loss value is a proper divergence (≥0, =0 when matched), giving a meaningful convergence signal.
- **No verl core modifications**. Everything is custom dataset/loss/entry-point layered on top.
- **TP=2 data-parallel** for generation rather than TP=16. Better throughput for the 27B model — TP=2 is sufficient to fit in memory on B200, and 8-way DP across 2 nodes parallelizes the 68k generations.
