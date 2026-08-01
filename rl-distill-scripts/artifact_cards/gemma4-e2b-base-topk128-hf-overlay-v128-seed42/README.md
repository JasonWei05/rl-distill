---
pretty_name: Gemma 4 E2B Base Top-k-128 HF Training Overlay
task_categories:
- text-generation
language:
- en
tags:
- gemma4
- knowledge-distillation
- top-k-logprobs
- math
configs:
- config_name: default
  data_files:
  - split: train
    path: train/*.parquet
  - split: validation
    path: validation/*.parquet
---

# Gemma 4 E2B base top-k-128 HF training overlay

This is the immutable training-engine overlay used to distill traces from Gemma 4 E2B base into
Gemma 4 E4B. It preserves the prompts, responses, and exact response token IDs from
[`JWei05/gemma4-e2b-base-topk128-traces`](https://huggingface.co/datasets/JWei05/gemma4-e2b-base-topk128-traces/tree/e32aaa02681ae83b3d7256b1b155c9084da2f289),
but replaces the source vLLM top-k targets with targets recomputed by the Hugging Face training
engine.

This repository is a reproducibility artifact for the corresponding distillation run. It is not a
new independently collected corpus.

## Contents

| Split | Rows | Response tokens | Shards |
| --- | ---: | ---: | ---: |
| train | 48,615 | 8,087,407 | 1,216 |
| validation | 128 | 18,437 | 16 |
| total | 48,743 | 8,105,844 | 1,232 |

The validation split contains one trace from each of 128 deterministic, unique, train-disjoint
validation prompts selected with seed 42. The Parquet shards use `int32` token IDs and `float16`
stored log probabilities.

## Target computation

- Teacher: `google/gemma-4-E2B` snapshot
  `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`
- Engine: unsharded Hugging Face Transformers forward
- Compute dtype: BF16
- Attention implementation: SDPA
- Vocabulary normalization: FP32 full-vocabulary `logsumexp` after the BF16 language-model head
  and final-logit softcap
- Top-k width: 128 over a 262,144-token vocabulary
- Alignment: response token at input position `i` is scored by logits at `i - 1`
- Maximum sequence length: 12,288 tokens

The original response token IDs are never re-tokenized. `rescore_config.json`,
`parity_receipt.json`, and `dataset_index.json` record the complete semantic configuration and
integrity identities.

## Integrity identities

- Dataset index SHA-256:
  `124a1b904b60963fb2b1d422107bec593a8ef1053cce34fff40bfb6314d1a16e`
- Source dataset index SHA-256:
  `efe76f5a53225e97081a750495ceb8ebe26b1a72a3ea8e0b74ea079a87828c0a`
- Rescore configuration SHA-256:
  `1c409d348a85f8d66941e7cb0700279af127147d82c39e6079aab929fcfca17a`
- Teacher weight-content SHA-256:
  `76dc84a5a805a2c8b91e9ccc00b8dbf8f4a99bf0d56ab25832f6e6addd4f7f57`
- Teacher model-identity SHA-256:
  `bde9e800223cdd62228ce39e0305398f6ada05b98adaf438b0b3d3d3c3015561`

## Intended use

Use this overlay when reproducing the E2B-base-to-E4B top-k distillation experiment with the
repository's `gemma4_topk_distill_fsdp2.sh` launcher. Consumers should validate
`dataset_index.json` before training and preserve the registered shard order.

The generated traces and probabilities remain subject to the terms governing the source dataset
and Gemma models. Review those terms before redistribution or downstream use.
