---
library_name: transformers
pipeline_tag: text-generation
license: gemma
base_model: google/gemma-4-E4B
datasets:
- JWei05/gemma4-e2b-base-topk128-traces
- JWei05/gemma4-e2b-base-topk128-hf-overlay-v128-seed42
language:
- en
tags:
- gemma4
- knowledge-distillation
- top-k-logprobs
- math
---

# Gemma 4 E4B distilled from E2B-base traces

This repository contains the final step-750 Hugging Face export from top-k distribution
distillation of Gemma 4 E2B-base math traces into Gemma 4 E4B. The student was initialized from
`google/gemma-4-E4B@411aa17b749aa952df1359d2dcea73917a544d9a`.

The source responses came from
[`JWei05/gemma4-e2b-base-topk128-traces`](https://huggingface.co/datasets/JWei05/gemma4-e2b-base-topk128-traces/tree/e32aaa02681ae83b3d7256b1b155c9084da2f289).
For training, the exact stored response token IDs were rescored through the Hugging Face BF16+SDPA
training engine. The resulting top-k-128 overlay is published at
[`JWei05/gemma4-e2b-base-topk128-hf-overlay-v128-seed42`](https://huggingface.co/datasets/JWei05/gemma4-e2b-base-topk128-hf-overlay-v128-seed42).

## Training configuration

| Parameter | Value |
| --- | --- |
| Student | Gemma 4 E4B |
| Teacher targets | Gemma 4 E2B-base, HF BF16+SDPA full forward |
| Objective | Stored-support top-k-128 distillation with full-vocabulary normalization |
| Training / validation rows | 48,615 / 128 |
| Global batch size | 128 |
| Microbatch per GPU | 1 |
| GPUs | 8 |
| Distributed engine | FSDP2 |
| Maximum sequence length | 12,288 |
| Vocabulary-projection chunk | 4,096 tokens |
| Optimizer | AdamW, betas `(0.9, 0.98)`, weight decay `0.1` |
| Learning rate | 100-step warmup to `2e-6`, then linear decay to `2e-7` |
| Duration | 750 optimizer steps, capped just before two complete epochs |
| Checkpoint cadence | 250 steps |

The run used BF16 forward parameter views with FP32 master parameters and FP32 reductions. Gradient
checkpointing was enabled. cuDNN was left in its normal nondeterministic operating mode.

## Training results

| Metric | Initial | Final step 750 |
| --- | ---: | ---: |
| Training loss | `0.194667` at step 1 | `0.078773` |
| Validation loss | `0.209595` at step 0 | `0.093663` |
| Learning rate | `2e-8` at step 1 | `2e-7` |
| Gradient norm | `14.8093` at step 1 | `1.5481` |

The W&B run is
[`rl-distill/gemma4-distill-vs-rl/85803e85`](https://wandb.ai/rl-distill/gemma4-distill-vs-rl/runs/85803e85).
It completed all 750 optimization steps without an OOM or non-finite update. The original launcher
reported a post-training failure because the deferred uploads targeted a private repository whose
storage quota was exhausted; the final model was subsequently published and independently checked
through an unauthenticated Hub request.

## Provenance

- Training code revision: `rl-distill@2f88a4ff`
- Run name:
  `e2b-base-to-e4b-topk128-lr2e6-linear-b128-2ep-750-normalcudnn-v1-2f88a4ff-20260731`
- Training overlay index SHA-256:
  `124a1b904b60963fb2b1d422107bec593a8ef1053cce34fff40bfb6314d1a16e`
- Student base revision: `411aa17b749aa952df1359d2dcea73917a544d9a`
- Teacher trace repository revision: `e32aaa02681ae83b3d7256b1b155c9084da2f289`
- Training overlay repository revision: `4f60c51340eb3a58efddff26e6a086a92c6e2123`
- Transformers version: `5.14.1`
- PyTorch version used for rescoring: `2.11.0+cu130`

## Limitations

This checkpoint optimizes agreement with the teacher distributions on a math-trace corpus. The
loss reduction is not by itself evidence of general capability improvement. Reported downstream
math and out-of-distribution evaluations should use the pinned evaluation protocol in the training
repository. Users must comply with the Gemma license and the terms of the source data.
