# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare dense Gemma3 4B logits against 2E and 4E upcycled MoE variants.

The strict test uses the canonical dense-init graph by default. Pass
``--no-canonical-dense-init`` only to measure ordinary sparse bf16 drift; that
diagnostic mode reports differences without treating them as a conversion
failure.
"""

import argparse
import gc
import os
import socket

import torch
import torch.distributed as dist
from megatron.bridge.models.gemma import (
    Gemma3ModelProvider4B,
    Gemma3MoEModelProvider4B,
    Gemma3MoEModelProvider4B4E,
)
from megatron.core import parallel_state
from megatron.core.activations import fast_gelu
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe import upcycling_utils


def _unwrap_activation(output: object) -> torch.Tensor:
    """Return the activation tensor from MCore's Tensor-or-(Tensor, bias) APIs."""
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected Tensor or tuple[Tensor, ...], got {type(output)!r}")
    return output


def _capture_hook(target: dict[str, torch.Tensor], name: str):
    def hook(_, __, output):
        # MCore uses [sequence, batch, hidden].  Both sides use the same layout,
        # so retain it exactly and move a detached copy off GPU immediately.
        target[name] = _unwrap_activation(output).detach().cpu()

    return hook


def _capture_layer_outputs(model) -> tuple[dict[str, torch.Tensor], list]:
    """Capture the comparable boundaries of every MCore decoder layer.

    The dense and MoE graphs intentionally have different internal MLP
    structure (fused pre-MLP norm vs standalone norm + ``SequentialMLP``), but
    these boundaries have identical semantic meaning:

    - attention output after Gemma's post-attention RMSNorm;
    - MLP output after Gemma's post-MLP RMSNorm, before residual addition;
    - decoder-layer output after the MLP residual addition.
    """
    captures: dict[str, torch.Tensor] = {}
    handles = []
    for layer_idx, layer in enumerate(model.decoder.layers):
        prefix = f"layer={layer_idx:02d}"
        # In canonical dense-init mode the TransformerLayer deliberately
        # bypasses ``layer.mlp`` after using it only for routing.  Hook the
        # unregistered dense-compatible helper instead so this boundary keeps
        # its usual "post-MLP / pre-residual" meaning.
        mlp_boundary = getattr(layer, "_canonical_dense_mlp", layer.mlp)
        handles.extend(
            [
                layer.self_attention.register_forward_hook(_capture_hook(captures, f"{prefix}/attn")),
                mlp_boundary.register_forward_hook(_capture_hook(captures, f"{prefix}/mlp")),
                layer.register_forward_hook(_capture_hook(captures, f"{prefix}/layer_output")),
            ]
        )
    return captures, handles


def _report_layer_parity(dense_captures: dict[str, torch.Tensor], moe_captures: dict[str, torch.Tensor]) -> str | None:
    """Print all layer boundaries and return the first non-bit-exact one."""
    first_nonexact: str | None = None
    for layer_idx in range(len(dense_captures) // 3):
        for component in ("attn", "mlp", "layer_output"):
            key = f"layer={layer_idx:02d}/{component}"
            dense = dense_captures[key]
            moe = moe_captures[key]
            if dense.shape != moe.shape:
                raise AssertionError(f"{key}: shape mismatch {dense.shape} != {moe.shape}")
            diff = (dense.float() - moe.float()).abs()
            exact = torch.equal(dense, moe)
            print(f"{key} exact={exact} max_abs={diff.max().item():.8g} mean_abs={diff.mean().item():.8g}")
            if not exact and first_nonexact is None:
                first_nonexact = key
    print(f"first_mcore_dense_moe_nonexact={first_nonexact or 'none'}")
    return first_nonexact


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _init_dist() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", _free_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(1234)


def _provider(provider_cls, seq_len: int, *, canonical_dense_init: bool = False):
    provider_kwargs = dict(
        seq_length=max(seq_len, 16),
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
        moe_permute_fusion=False,
    )
    if canonical_dense_init:
        provider_kwargs["gemma3_moe_canonical_dense_init"] = True
    provider = provider_cls(**provider_kwargs)
    provider.finalize()
    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    return provider


def _inputs(vocab_size: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # A deterministic nontrivial token pattern makes this an architecture-level
    # test rather than a property of an all-zero input.
    input_ids = torch.arange(seq_len, dtype=torch.long, device="cuda").unsqueeze(0) * 997 + 1
    input_ids %= vocab_size
    position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device="cuda").unsqueeze(0)
    # Megatron 4-D boolean masks use ``True`` for positions that must be
    # excluded.  Match the real causal path rather than masking the entire
    # attention matrix with an all-ones tensor.
    attention_mask = ~torch.tril(
        torch.ones(
            (input_ids.size(0), 1, input_ids.size(1), input_ids.size(1)),
            dtype=torch.bool,
            device="cuda",
        )
    )
    return input_ids, position_ids, attention_mask


@torch.no_grad()
def _logits(model, input_ids, position_ids, attention_mask):
    model.eval()
    return model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    ).detach()


def _compare(
    name: str,
    dense_model,
    dense_logits,
    input_ids,
    position_ids,
    attention_mask,
    *,
    seq_len: int,
    force_expert_zero: bool,
    report_activations: bool,
    canonical_dense_init: bool,
) -> None:
    moe_model = (
        _provider(
            Gemma3MoEModelProvider4B if name == "2E" else Gemma3MoEModelProvider4B4E,
            seq_len,
            canonical_dense_init=canonical_dense_init,
        )
        .provide()
        .cuda()
    )

    # Megatron-Core's generic upcycling helper only recognizes a function
    # literally named ``gelu``.  Gemma3 intentionally uses the equivalent
    # ``fast_gelu`` implementation; preserve that implementation behind the
    # helper's expected name for this architecture-level test.
    def gelu(x):
        return fast_gelu(x)

    dense_model.config.activation_func = gelu
    moe_model.config.activation_func = gelu
    moe_model.load_state_dict(upcycling_utils.upcycle_state_dict([moe_model], [dense_model])["model"], strict=True)
    if force_expert_zero:
        with torch.no_grad():
            for param_name, param in moe_model.named_parameters():
                if param_name.endswith(".mlp.router.weight"):
                    param.zero_()
    dense_captures: dict[str, torch.Tensor] | None = None
    moe_captures: dict[str, torch.Tensor] | None = None
    dense_handles = []
    moe_handles = []
    if report_activations:
        dense_captures, dense_handles = _capture_layer_outputs(dense_model)
        moe_captures, moe_handles = _capture_layer_outputs(moe_model)
    try:
        # Recompute the dense output with hooks attached. ``dense_logits`` was
        # intentionally computed before building the MoE so the normal path
        # remains lean when activation reporting is disabled.
        if report_activations:
            dense_logits = _logits(dense_model, input_ids, position_ids, attention_mask)
        moe_logits = _logits(moe_model, input_ids, position_ids, attention_mask)
    finally:
        for handle in dense_handles:
            handle.remove()
        for handle in moe_handles:
            handle.remove()

    diff = (moe_logits - dense_logits).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    same = torch.equal(moe_logits, dense_logits)
    close = torch.allclose(moe_logits, dense_logits, rtol=0.0, atol=0.0)
    print(
        f"{name}: seq_len={seq_len} force_expert_zero={force_expert_zero} "
        f"canonical_dense_init={canonical_dense_init} "
        f"equal={same} allclose_atol0={close} max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} "
        f"top1_agreement={(moe_logits.argmax(-1) == dense_logits.argmax(-1)).float().mean().item():.8f}"
    )
    first_nonexact = None
    if report_activations:
        assert dense_captures is not None and moe_captures is not None
        first_nonexact = _report_layer_parity(dense_captures, moe_captures)
    if canonical_dense_init and (not same or first_nonexact is not None):
        raise AssertionError(f"{name} logits differ from dense Gemma3 4B")

    del moe_logits, moe_model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--force-expert-zero", action="store_true")
    parser.add_argument(
        "--report-activations",
        action="store_true",
        help="Print dense-vs-MoE attention, MLP, and layer outputs at every MCore layer.",
    )
    parser.add_argument(
        "--canonical-dense-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exercise the exact MCore initialization path (default: true).",
    )
    args = parser.parse_args()
    _init_dist()
    try:
        dense_provider = _provider(Gemma3ModelProvider4B, args.seq_len)
        dense_model = dense_provider.provide().cuda()
        input_ids, position_ids, attention_mask = _inputs(dense_provider.vocab_size, args.seq_len)
        dense_logits = _logits(dense_model, input_ids, position_ids, attention_mask)

        _compare(
            "2E",
            dense_model,
            dense_logits,
            input_ids,
            position_ids,
            attention_mask,
            seq_len=args.seq_len,
            force_expert_zero=args.force_expert_zero,
            report_activations=args.report_activations,
            canonical_dense_init=args.canonical_dense_init,
        )
        _compare(
            "4E",
            dense_model,
            dense_logits,
            input_ids,
            position_ids,
            attention_mask,
            seq_len=args.seq_len,
            force_expert_zero=args.force_expert_zero,
            report_activations=args.report_activations,
            canonical_dense_init=args.canonical_dense_init,
        )
        if args.canonical_dense_init:
            print("MCORE_RANDOM_WEIGHT_DENSE_MOE_PARITY_OK")
        else:
            print("mode=ordinary_sparse_moe (diagnostic only; exactness is not expected)")
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
