# Gemma 4 HF training-shaped top-128 rescorer

> [!CAUTION]
> **DO NOT RUN any command in this document without separate, explicit approval.**
> Preparing or reviewing this tooling does not authorize GPU use, trace
> rescoring, Hugging Face uploads, or either experimental training line.

`rescore_gemma4_training_topk.py` derives HF BF16+SDPA targets from an already
validated Gemma 4 vLLM trace bundle. It uses verl's Hugging Face model-class
resolver, but it is an unsharded Transformers forward, not the FSDP2 training
engine. It never edits the source bundle. It writes a separate, one-to-one
overlay containing the exact source IDs, masks, text, row order, and new
top-128 targets.

The scorer uses the exact local teacher identity, BF16 + SDPA, full-sequence
batch-1 teacher forcing with `use_cache=False`, the causal predecessor shift,
Gemma 4 final-logit softcapping, FP32 full-vocabulary normalization, int32 token
IDs, and FP16 log probabilities. Each Parquet and manifest file is published
with an atomic replacement, sequentially. The pair is not transactionally
visible as one unit, so every resume/finalize path fails closed unless both
files agree. Validation rehashes the immutable source Parquet and compares
every copied overlay field row by row. Each shard is bound to the source
dataset index, source Parquet SHA, source manifest SHA, and source manifest
`trace_ids_sha256`.

## Required order

After separate approval, operate in this order:

1. `inspect`: validate identities and write the immutable rescore config.
2. `parity`: compare the chunked scorer exactly with native HF full forward
   and write the mandatory, run-bound parity receipt.
3. `score`: create resumable overlay shards, normally with one worker per GPU.
4. `finalize`: validate every shard and create the overlay dataset index.

`score` and `finalize` reject a missing, modified, or mismatched parity receipt.
Do not use cached or independently windowed forwarding.
Gemma 4 has periodic full-attention layers, and cached BF16 forwarding is not
numerically identical to the training-shaped full forward.

This parity gate establishes internal equivalence between two unsharded HF
paths only. Before treating these targets as numerically identical to verl
FSDP2 training outputs, run a separately approved one-row audit through the
actual FSDP2 engine and compare the serialized targets exactly.

## Command template

Set these paths only after the source bundle and exact teacher snapshot have
been reviewed:

```bash
PY=/tmp/.venv-gemma4-e2e/bin/python
SCRIPT=rl-distill-scripts/data/rescore_gemma4_training_topk.py
SOURCE_INDEX=/path/to/source/dataset_index.json
MODEL_PATH=/path/to/exact/local/hf/teacher
OUTPUT_ROOT=/path/to/separate/hf-target-overlay
```

Read-only identity/config inspection except for creating `OUTPUT_ROOT` and its
`rescore_config.json`:

```bash
$PY "$SCRIPT" inspect \
  --source-dataset-index "$SOURCE_INDEX" \
  --model-path "$MODEL_PATH" \
  --output-root "$OUTPUT_ROOT"
```

Mandatory native-forward parity gate; this loads the model on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 $PY "$SCRIPT" parity \
  --source-dataset-index "$SOURCE_INDEX" \
  --model-path "$MODEL_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --parity-rows 8 \
  --parity-max-response-tokens 512
```

Two resumable scoring workers:

```bash
CUDA_VISIBLE_DEVICES=0 $PY "$SCRIPT" score \
  --source-dataset-index "$SOURCE_INDEX" \
  --model-path "$MODEL_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --worker-id 0 --num-workers 2

CUDA_VISIBLE_DEVICES=1 $PY "$SCRIPT" score \
  --source-dataset-index "$SOURCE_INDEX" \
  --model-path "$MODEL_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --worker-id 1 --num-workers 2
```

Final validation and index creation after every worker exits successfully:

```bash
$PY "$SCRIPT" finalize \
  --source-dataset-index "$SOURCE_INDEX" \
  --model-path "$MODEL_PATH" \
  --output-root "$OUTPUT_ROOT"
```

`finalize` does not upload. The strict source+overlay preflight is
`preflight_gemma4_training_topk_overlay.py`, and the distillation launcher
selects it automatically from the overlay schema when `SOURCE_DATASET_INDEX`
is supplied. A dedicated overlay uploader and another explicit approval are
still required before any Hugging Face mutation.

## Pre-run gates

- Source train and validation indexes are complete and immutable.
- `MODEL_PATH` identity exactly equals the source teacher identity.
- Focused CPU tests pass.
- The parity mode passes exactly on serialized top-k IDs/log probabilities and
  sampled-token log probabilities, and its receipt validates.
- A separately approved one-row real-FSDP2 audit passes before making any
  FSDP2-equivalence claim.
- A separately approved 100–500-row benchmark establishes runtime and memory.
- Output paths are new and outside the source bundle.
- GPU allocation and operator ownership are explicitly approved.

Again: **DO NOT RUN without separate, explicit approval.**
