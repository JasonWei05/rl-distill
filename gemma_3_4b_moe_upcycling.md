# Gemma 3 4B Dense-to-MoE Upcycling Design

## Goal

Convert Gemma 3 4B from a dense-MLP transformer into a top-1 Mixture-of-Experts
model by replacing every dense MLP with an MoE layer. Each expert is initialized
as an exact copy of the original dense MLP, while the router is randomly
initialized and trainable.

The initial converted model should preserve the dense model's logits exactly.
After conversion, training can specialize the duplicated experts through the
router and the expert weights.

This design currently supports:

- `Gemma3MoEModelProvider4B`: 2 experts, top-1 routing.
- `Gemma3MoEModelProvider4B4E`: 4 experts, top-1 routing.

## High-Level Architecture

Original Gemma 3 4B decoder layer:

```text
x
|-- input RMSNorm
|-- self attention with QK norm and local/global Gemma 3 attention
|-- post-attention RMSNorm
|-- residual add
|-- dense MLP:
|     linear_fc1 -> fast_gelu gated activation -> linear_fc2 -> post-MLP RMSNorm
|-- residual add
```

Upcycled MoE Gemma 3 4B decoder layer:

```text
x
|-- input RMSNorm
|-- self attention with QK norm and local/global Gemma 3 attention
|-- post-attention RMSNorm
|-- residual add
|-- pre-MoE RMSNorm
|-- router: hidden_size -> num_experts
|-- top-1 token dispatch
|-- selected expert:
|     linear_fc1 -> fast_gelu gated activation -> linear_fc2 -> post-MLP RMSNorm
|-- token combine
|-- residual add
```

Everything outside the MLP is unchanged. The attention pattern remains Gemma 3's
existing local/global schedule. The MoE replacement only changes the feed-forward
branch.

## Provider Changes

The implementation lives in:

- `Megatron-Bridge/src/megatron/bridge/models/gemma/gemma3_provider.py`
- `Megatron-Bridge/src/megatron/bridge/models/gemma/__init__.py`
- `Megatron-Bridge/src/megatron/bridge/models/__init__.py`

The new layer spec is `gemma3_moe_layer_spec`. It keeps Gemma 3 attention intact
and replaces the dense MLP with:

```text
MoELayer
  router: Megatron-Core TopKRouter
  experts: SequentialMLP
    local_experts[i]:
      linear_fc1 = TEColumnParallelLinear
      activation = TEActivationOp
      linear_fc2 = TERowParallelLinearTorchRMSNorm
```

`SequentialMLP` is used instead of `TEGroupedMLP` because Gemma 3 needs a
post-MLP RMSNorm inside each expert. Grouped GEMM is therefore disabled for this
variant:

```python
moe_grouped_gemm = False
```

The 2E provider sets:

```python
num_moe_experts = 2
moe_router_topk = 1
moe_ffn_hidden_size = 10240
moe_router_load_balancing_type = "aux_loss"
moe_aux_loss_coeff = 1e-3
moe_router_pre_softmax = False
moe_token_dispatcher_type = "alltoall"
```

The 4E provider inherits the 2E provider and only changes:

```python
num_moe_experts = 4
```

## Why `moe_router_pre_softmax = False`

For exact dense-logit preservation, top-1 routing must combine the selected
expert output with weight `1`.

If `moe_router_pre_softmax = True`, Megatron computes softmax over all experts
first, selects the top expert, and uses that selected expert's probability as
the combine weight. With random router weights, that probability is usually less
than `1`, so even identical experts produce scaled-down MLP outputs and logits no
longer match the dense model.

With `moe_router_pre_softmax = False`, Megatron selects top-1 first and then
softmaxes the single selected score. Softmax over one value is exactly `1`, so
the selected duplicated expert is an identity replacement for the dense MLP.

Tradeoff: the router does not get a useful main-loss gradient from top-1
post-top-k softmax. The router is trained through the configured auxiliary load
balancing loss. `TransformerConfig` was relaxed to allow this case when an aux
loss coefficient is enabled, and emits a warning documenting the tradeoff.

## RMSNorm Detail

Gemma 3 uses zero-centered RMSNorm:

```text
RMSNorm(x) * (1 + weight)
```

rather than the usual:

```text
RMSNorm(x) * weight
```

The implementation adds `Gemma3RMSNorm`, a small torch module that matches
Gemma 3's zero-centered gamma behavior. It is used for the post-output norms in
`TERowParallelLinearLayerNorm` and in the MoE expert output projection wrapper.

This also avoids a Transformer Engine runtime issue encountered when using TE
RMSNorm inside `SequentialMLP` experts:

```text
Output rsigma is not allocated
```

Using the same torch RMSNorm path in dense and MoE variants also makes the
dense-vs-MoE logit identity test bit-exact.

## Upcycling Semantics

The implementation updates:

```text
Megatron-LM/megatron/core/transformer/moe/upcycling_utils.py
```

The default non-granular upcycling path now duplicates dense MLP parameters
exactly when `granularity == 1`.

For each dense layer:

```text
dense mlp.linear_fc1.weight
  -> expert 0 linear_fc1.weight
  -> expert 1 linear_fc1.weight
  -> ...

dense mlp.linear_fc2.weight
  -> expert 0 linear_fc2.weight
  -> expert 1 linear_fc2.weight
  -> ...

dense mlp.linear_fc2.post_layernorm.weight
  -> expert 0 linear_fc2.post_layernorm.weight
  -> expert 1 linear_fc2.post_layernorm.weight
  -> ...
```

The router is not copied from the dense model, because the dense model has no
router. It remains randomly initialized and trainable.

`fast_gelu` is treated as Gelu-compatible by the upcycling helper so Gemma 3's
activation is accepted.

## Parameter Count Shape

The dense Gemma 3 4B MLP per layer has one FFN. The MoE variants have `N`
full-size FFNs per layer:

```text
2E: attention + 2 * dense_ffn + router
4E: attention + 4 * dense_ffn + router
```

The active parameter count per token remains close to dense for the MLP path
because top-1 routing executes one expert per token. The total stored parameter
count increases with the number of experts.

## Verification

Two scripts were added:

- `rl-distill-scripts/smoke_gemma3_moe_upcycle.py`
- `rl-distill-scripts/logit_test_gemma3_moe_4b.py`

The smoke test builds a tiny Gemma 3 dense model and a tiny 2E MoE model, runs
Megatron's upcycling path, strict-loads the converted state dict, and verifies
that every expert's `fc1`, `fc2`, and post-MLP RMSNorm weights exactly match the
dense source.

The full logit test builds the full Gemma 3 4B architecture with deterministic
random weights, upcycles into 2E and 4E variants, and compares logits against
the dense model on the same token input.

Observed H100 result:

```text
2E: equal=True allclose_atol0=True max_abs=0 mean_abs=0
4E: equal=True allclose_atol0=True max_abs=0 mean_abs=0
Dense Gemma3 4B, 2E MoE, and 4E MoE logits are exactly identical
```

The exact command used:

```bash
mlx worker login 911777 -- cd /mlx_devbox/users/jason.wei/playground/rl-distill '&&' \
  CUDNN_HOME=/mlx_devbox/users/jason.wei/playground/rl-distill/.venv/lib/python3.12/site-packages/nvidia/cudnn \
  NVRTC_HOME=/mlx_devbox/users/jason.wei/playground/rl-distill/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc \
  CURAND_HOME=/mlx_devbox/users/jason.wei/playground/rl-distill/.venv/lib/python3.12/site-packages/nvidia/curand \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python rl-distill-scripts/logit_test_gemma3_moe_4b.py
```

## Operational Notes

The current implementation is validated for single-rank H100 execution with:

```text
tensor_model_parallel_size = 1
pipeline_model_parallel_size = 1
expert_model_parallel_size = 1
expert_tensor_parallel_size = 1
```

The architecture uses Megatron-Core's standard MoE layer, token dispatcher, and
upcycling utilities, so expert parallel configurations should be the next thing
to validate before large training runs.

The broader Megatron-LM upcycling unit test file was attempted, but this local
environment cannot run it cleanly because it expects Apex gradient accumulation
fusion and includes world-size combinations larger than the single-rank worker
run. The Gemma-specific smoke and full 4B logit identity tests pass.

## Open Follow-Ups

1. Validate the same logit identity test with `expert_model_parallel_size > 1`.
2. Run a short train step and confirm aux-loss router gradients are present.
3. Add a checkpoint conversion script that loads an actual Gemma 3 4B checkpoint,
   instantiates `Gemma3MoEModelProvider4B` or `Gemma3MoEModelProvider4B4E`, and
   saves the upcycled Megatron checkpoint.
4. Decide whether to keep exact identity routing or switch to pre-softmax routing
   after a warmup period if main-loss router gradients are important.
