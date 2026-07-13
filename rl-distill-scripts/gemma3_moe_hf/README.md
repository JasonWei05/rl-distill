# Gemma 3 dense-to-MoE Hugging Face checkpoint

This directory contains the reference Hugging Face implementation and the
converter for Gemma 3 4B dense-to-top-1-MoE upcycling. For the complete setup,
validation, vLLM, and RL workflow, see
[`../GEMMA3_MOE_RL_TRAINING.md`](../GEMMA3_MOE_RL_TRAINING.md).

Each dense MLP becomes a bias-free router plus 2 or 4 full-size experts. Every
expert copies the dense gate, up, down, and post-feedforward RMSNorm tensors.
The only newly initialized tensor is `mlp.router.weight`.

## Convert and verify

```bash
VIRTUAL_ENV="$PWD/.venv-megatron" uv run --active --no-sync python \
  rl-distill-scripts/gemma3_moe_hf/create_gemma3_moe_from_dense_hf.py \
  --dense-model google/gemma-3-4b-pt \
  --output-dir /shared/checkpoints/gemma3-4b-pt-moe-2e \
  --canonical-output-dir /shared/checkpoints/gemma3-4b-pt-moe-2e-canonical \
  --num-experts 2
```

`--dense-model` may also point to an existing local snapshot. Verification is
enabled by default and must finish with `GEMMA3_MOE_CHECKPOINT_VERIFIED`.

The normal output uses sparse dispatch and is the training checkpoint. The
canonical output reuses the same immutable files through hard links when
possible, but sets `gemma3_moe_canonical_dense_init=true` for exact
dense-equivalence tests. Never train the canonical view.

## Load the reference model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/shared/checkpoints/gemma3-4b-pt-moe-2e"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    dtype="auto",
)
```

The checkpoint intentionally stores the expert count as
`gemma3_moe_num_experts`. Generic `num_experts` keys cause vLLM's Transformers
backend to substitute a fused MoE block that cannot represent Gemma 3's
per-expert post-MLP RMSNorm.

## Run with native vLLM

Install this repository editable so vLLM can discover the plugin entry point:

```bash
uv pip install --python .venv-megatron/bin/python -e .
```

Then use the native model:

```python
from vllm import LLM

llm = LLM(
    model="/shared/checkpoints/gemma3-4b-pt-moe-2e",
    trust_remote_code=True,
    model_impl="native",
    attention_backend="TRITON_ATTN",
    dtype="bfloat16",
    enforce_eager=True,
)
```

`verl/vllm/gemma3_moe.py` keeps vLLM's native Gemma 3 attention and cache path
and implements the custom expert layout. `model_impl="transformers"` is a
diagnostic fallback, not a supported rollout path. Eager execution is the
validated mode for the token-routed expert loop.

## Other conversion direction

`convert_gemma3_moe_distckpt_to_hf.py` exports an existing Megatron distributed
checkpoint into the same HF layout. It is separate from fresh dense
upcycling; use `create_gemma3_moe_from_dense_hf.py` when initialization must
match `google/gemma-3-4b-pt`.
