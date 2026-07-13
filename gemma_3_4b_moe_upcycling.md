# Gemma 3 4B Dense-to-MoE Upcycling Design

## Scope

This implementation converts the text backbone of `google/gemma-3-4b-pt`
from dense feed-forward blocks to 2- or 4-expert, top-1 MoE blocks. Every
expert begins as a literal copy of the dense MLP. Attention, embeddings,
normalization, RoPE, and the LM head are unchanged.

The runnable procedure is in
[`rl-distill-scripts/GEMMA3_MOE_RL_TRAINING.md`](rl-distill-scripts/GEMMA3_MOE_RL_TRAINING.md).
This document records the invariants the converter and three execution
backends must preserve.

## Layer transformation

The dense feed-forward branch is:

```text
residual
  -> pre-feedforward RMSNorm
  -> gate/up projections + gated GELU
  -> down projection
  -> post-feedforward RMSNorm
  -> residual add
```

The upcycled branch is:

```text
residual
  -> pre-feedforward RMSNorm
  -> bias-free router -> top-1 expert id
  -> selected copied {gate, up, down, post-RMSNorm} expert
  -> residual add
```

For every layer and every expert, conversion performs these exact mappings:

| Dense tensor | MoE destination |
|---|---|
| `mlp.gate_proj.weight` | `mlp.experts.{e}.gate_proj.weight` |
| `mlp.up_proj.weight` | `mlp.experts.{e}.up_proj.weight` |
| `mlp.down_proj.weight` | `mlp.experts.{e}.down_proj.weight` |
| `post_feedforward_layernorm.weight` | `mlp.experts.{e}.post_layernorm.weight` |

All non-MLP tensors are copied without transformation. The router has no dense
source, so each layer initializes it deterministically from
`router_seed + layer_index`.

## Routing invariant

The supported router configuration is fixed:

```text
num_experts_per_tok = 1
router_score_function = softmax
router_pre_softmax = false
```

Selecting top-1 before softmax means the combine softmax contains one value,
so its weight is exactly `1`. If softmax ran over all experts first, the
selected probability would be less than one and would scale the MLP branch;
the converted model would no longer be dense-equivalent.

With post-top-k softmax and top-1, the task loss has no useful differentiable
path through the discrete router choice. Normal training therefore uses the
auxiliary load-balancing loss (`router_aux_loss_coef=1e-3` by default). A
balanced two-expert router loss is expected to be near `1`; it is a balancing
metric, not a dense-parity metric.

## Mathematical equality versus bit equality

Because every expert is identical and its combine weight is one, the ordinary
sparse model computes the same mathematical function as the dense model at
initialization. It is not guaranteed to be bit-identical in bf16:

- sparse dispatch partitions tokens into different GEMM batch shapes;
- the Megatron MoE graph separates operations that the dense graph fuses;
- bf16 rounding differences accumulate across 34 decoder layers;
- a small logit perturbation can flip an argmax when two vocabulary logits are
  close.

Consequently, `99%` top-1 agreement is not accepted as proof that conversion
is correct. The strict gate uses `gemma3_moe_canonical_dense_init=true`. In
that mode the router still runs, but expert 0 is evaluated over the full token
batch through the dense execution graph. The result must match dense weights,
every checked activation boundary, logits, and top-1 decisions exactly.

Canonical mode is a test fixture, not a training mode: it bypasses sparse
dispatch and does not specialize experts. The converter can create a canonical
view that hard-links the normal checkpoint's immutable files and changes only
`config.json`.

## Runtime implementations

### Hugging Face reference

`rl-distill-scripts/gemma3_moe_hf/` contains the custom
`Gemma3MoeForCausalLM` config/model and the dense converter. The converter:

- accepts a local snapshot or Hub model ID;
- resolves and serializes the effective Gemma 3 text config instead of relying
  on version-dependent defaults;
- writes one weight shard per decoder layer;
- duplicates all four dense MLP tensor classes into every expert;
- avoids generic `num_experts` keys that trigger an incompatible fused-MoE
  substitution;
- verifies every copied tensor and output key by default.

The remote-code model is the portable reference and the HF activation gate. It
is not the production vLLM rollout implementation.

### Megatron actor and reference

The vendored Megatron-Bridge additions are in:

- `third_party/Megatron-Bridge/src/megatron/bridge/models/gemma/gemma3_provider.py`
- `third_party/Megatron-Bridge/src/megatron/bridge/models/gemma/gemma3_moe_bridge.py`

Normal execution uses Megatron-Core `MoELayer` with `SequentialMLP` experts.
Grouped GEMM is disabled because Gemma 3 has a post-MLP RMSNorm inside each
expert. The bridge maps HF router and expert tensors with TP/EP-aware sharding.

The canonical MCore layer creates an unregistered dense helper whose norm
weights alias the authoritative MoE state and whose FC weights are
differentiable TP slices of the first EP-local expert (all experts are exact
duplicates at this gate). It adds no checkpoint or optimizer parameters. The
helper exists only while canonical mode is enabled.

### Native vLLM rollout

`verl/vllm/gemma3_moe.py` retains vLLM's native Gemma 3 attention, KV cache,
residual fusion, tensor-parallel projections, and logits processor. It replaces
only the dense MLP with the router and per-expert post-normalized MLPs. Its
loader explicitly packs Q/K/V and each expert's gate/up pair.

The implementation is registered through the `vllm.general_plugins` entry
point in `pyproject.toml`. Production rollout must use:

```text
ROLLOUT_MODEL_IMPL=native
ROLLOUT_ATTENTION_BACKEND=TRITON_ATTN
```

The generic vLLM Transformers backend produced incorrect long, non-terminating
rollouts for this model and remains diagnostic-only.

## Validation contract

A new upcycle is eligible for RL only after all applicable gates pass:

1. Converter verification prints `GEMMA3_MOE_CHECKPOINT_VERIFIED`.
2. HF dense/MoE activation comparison prints
   `HF_DENSE_MOE_ACTIVATION_PARITY_OK`.
3. Production-topology MCore comparison prints
   `MCORE_DENSE_MOE_ACTIVATION_PARITY_OK` with exact parameters, activations,
   logits, `top1_agreement=1.00000000`, and no first mismatch.
4. Native vLLM greedy output from the canonical view matches the dense native
   reference and terminates normally.
5. The bounded end-to-end correctness launcher uses the ordinary sparse view
   and completes rollout, old-log-prob, update, and weight resynchronization
   before a long training run starts.

Ordinary sparse diagnostics may report bf16 drift. They cannot replace the
canonical proof.

## Training boundaries

- Actor and reference use the Megatron backend. The HF/FSDP actor path does not
  implement the required router auxiliary loss and R2 replay for this model.
- `R2` router replay is the supported default. Native vLLM contains the route
  capture hook needed by `R3`, but the colocated rollout-to-trainer handoff is
  still experimental and guarded against all-zero captures.
- Normal training must use the non-canonical checkpoint and
  `gemma3_moe_canonical_dense_init=false`.
- A high entropy or responses repeatedly reaching the length cap is a rollout
  correctness alarm. Confirm the native plugin and Triton attention path
  before changing optimization hyperparameters.
