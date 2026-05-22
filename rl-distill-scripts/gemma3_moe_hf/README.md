# Gemma 3 MoE HF Export

This directory contains the custom Hugging Face remote-code model and converter
for the Gemma 3 4B MoE checkpoints produced by the Megatron SFT runs.

The exported architecture is text-only `Gemma3MoeForCausalLM`. Each Gemma 3
dense MLP is replaced with:

- `mlp.router`: a bias-free linear router `[hidden_size -> num_experts]`
- `mlp.experts.{i}.gate_proj`
- `mlp.experts.{i}.up_proj`
- `mlp.experts.{i}.down_proj`
- `mlp.experts.{i}.post_layernorm`

Routing matches the Megatron upcycled checkpoints: `num_experts_per_tok=1`,
`router_score_function="softmax"`, and `router_pre_softmax=False`. With this
Megatron setting, the top-1 selected expert output is not multiplied by a
softmax probability because softmax over the single selected logit is `1`.

Convert a checkpoint:

```bash
python rl-distill-scripts/gemma3_moe_hf/convert_gemma3_moe_distckpt_to_hf.py \
  --hf-repo-id JWei05/gemma3-4b-pt-moe-4e-top1-sft-16k \
  --cache-dir /tmp/hf-gemma3-moe-cache \
  --output-dir /tmp/gemma3-4b-pt-moe-4e-top1-sft-16k-hf \
  --num-experts 4
```

Load from Hugging Face:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "JWei05/gemma3-4b-pt-moe-4e-top1-sft-16k"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype="auto")
```

vLLM 0.11 can instantiate this via the Transformers backend:

```python
from vllm import LLM

llm = LLM(
    model="JWei05/gemma3-4b-pt-moe-4e-top1-sft-16k",
    trust_remote_code=True,
    model_impl="transformers",
    dtype="bfloat16",
)
```

This is correctness-oriented. A native vLLM model using fused MoE kernels would
be a separate performance optimization because these checkpoints use per-expert
post-MLP RMSNorms, which do not directly match vLLM's standard `FusedMoE`
layout.
