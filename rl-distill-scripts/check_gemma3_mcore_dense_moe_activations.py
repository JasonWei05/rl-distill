#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Strict real-weight MCore dense-vs-MoE activation gate for Gemma 3.

It loads the original dense HF checkpoint and a canonical dense-init MoE
checkpoint independently through Megatron-Bridge, then compares the
semantically identical boundaries of every decoder layer:

* post-attention output;
* fused pre-MLP norm/FC1 output;
* FC2 output including Gemma's post-MLP RMSNorm;
* complete MLP output;
* decoder-layer output after the residual connection.

Unlike a HF-only parity test, this exercises the exact MCore graph used by the
actor.  The MoE checkpoint must have
``gemma3_moe_canonical_dense_init=true``.  The script exits nonzero on the
first non-bit-exact result.

Example:

  CUDA_VISIBLE_DEVICES=0 .venv-megatron/bin/python \\
    rl-distill-scripts/check_gemma3_mcore_dense_moe_activations.py \\
    --moe-model /checkpoints/gemma3-4b-pt-moe-2e-canonical --seq-len 2048

  # The production 8xH100 topology (TP=4, EP=2):
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
    .venv-megatron/bin/torchrun --nproc-per-node 8 \\
    rl-distill-scripts/check_gemma3_mcore_dense_moe_activations.py \\
    --moe-model /checkpoints/gemma3-4b-pt-moe-2e-canonical \\
    --tp 4 --ep 2 --seq-len 2048 --check-parameters
"""

from __future__ import annotations

import argparse
import gc
import os
import socket
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoTokenizer

DEFAULT_DENSE_MODEL = "google/gemma-3-4b-pt"
DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "data" / "gemma3_it_chat_template.jinja"
DEFAULT_PROMPT = "Solve the equation 3x + 7 = 22. Explain your reasoning."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--moe-model", required=True)
    parser.add_argument(
        "--data",
        default=None,
        help="Optional parquet containing a DAPO prompt chat-list column.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used when --data is omitted.")
    parser.add_argument("--chat-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--check-parameters",
        action="store_true",
        help=(
            "Also require every dense MLP tensor to exactly match the local "
            "canonical expert tensor before running activations."
        ),
    )
    parser.add_argument("--tp", type=int, default=1, help="MCore tensor-parallel degree.")
    parser.add_argument("--ep", type=int, default=1, help="MCore expert-parallel degree.")
    parser.add_argument(
        "--attention-backend",
        choices=("flash", "fused", "unfused"),
        default="flash",
        help="Use the same MCore attention implementation on both sides.",
    )
    parser.add_argument(
        "--allow-noncanonical",
        action="store_true",
        help=(
            "Diagnostic mode: report ordinary sparse-MoE drift instead of "
            "requiring the canonical dense-init path and exact equality."
        ),
    )
    return parser.parse_args()


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _init_distributed(args: argparse.Namespace) -> int:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", _free_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    world_size = int(os.environ["WORLD_SIZE"])
    expected_world_size = args.tp * args.ep
    if world_size != expected_world_size:
        raise ValueError(f"WORLD_SIZE={world_size}, but --tp {args.tp} * --ep {args.ep} = {expected_world_size}")

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=args.ep,
        expert_tensor_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(1234)
    return dist.get_rank()


def _tokens(args: argparse.Namespace) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(args.moe_model, trust_remote_code=True)
    if args.data is not None:
        import pandas as pd

        messages = [dict(message) for message in pd.read_parquet(args.data).iloc[args.prompt_index]["prompt"]]
    else:
        messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=args.chat_template.read_text(),
    )
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    if args.seq_len < input_ids.shape[1]:
        raise ValueError(f"--seq-len ({args.seq_len}) is shorter than the rendered prompt ({input_ids.shape[1]})")
    repeats = (args.seq_len + input_ids.shape[1] - 1) // input_ids.shape[1]
    return input_ids.repeat(1, repeats)[:, : args.seq_len].cuda()


def _configure_provider(
    provider: object,
    seq_len: int,
    attention_backend: str,
    *,
    tp: int,
    ep: int,
) -> None:
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.enums import AttnBackend

    provider.params_dtype = torch.bfloat16
    provider.bf16 = True
    provider.fp16 = False
    provider.tensor_model_parallel_size = tp
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = ep
    provider.expert_tensor_parallel_size = 1
    provider.sequence_parallel = tp > 1
    provider.variable_seq_lengths = True
    provider.seq_length = max(seq_len, 16)
    provider.gradient_accumulation_fusion = False
    provider.moe_permute_fusion = False
    # MCore validates this when variable sequence lengths are enabled, even
    # for the dense VLM wrapper.
    provider.moe_token_dispatcher_type = "alltoall"
    provider.attention_backend = AttnBackend[attention_backend]
    provider.finalize()
    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()


def _load_mcore_model(
    path: str,
    seq_len: int,
    attention_backend: str,
    *,
    tp: int,
    ep: int,
):
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(path, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    _configure_provider(provider, seq_len, attention_backend, tp=tp, ep=ep)
    # ``ModelProvider.provide`` defaults these to ``None`` for pipeline
    # composition.  This standalone gate owns the entire model, and the dense
    # VLM wrapper needs ``pre_process=True`` to create/scatter text embeddings.
    model = provider.provide(pre_process=True, post_process=True).cuda().eval()
    bridge.load_hf_weights([model], path)
    return model


def _forward_model(model: nn.Module) -> nn.Module:
    # The dense Gemma checkpoint is a VLM container; no images are present in
    # this test, so its fully loaded text backbone is the correct comparison.
    return getattr(model, "language_model", model)


def _run_model(
    model: nn.Module,
    language_model: nn.Module,
    *,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> object:
    """Run the model without bypassing a VLM wrapper's SP input scatter.

    The dense Gemma checkpoint is a VLM checkpoint.  Its wrapper embeds text
    and scatters the resulting sequence across the TP group before invoking
    ``language_model``.  Calling the inner language model directly works at
    TP=1 but gives every TP rank a full sequence at TP>1, then its output
    gather duplicates that sequence.  Hooks are still registered on the
    language model below, but the wrapper must own the dense forward.
    """
    if model is not language_model:
        return model(
            input_ids=input_ids,
            position_ids=position_ids,
            runtime_gather_output=True,
        )
    return model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        runtime_gather_output=True,
    )


def _unwrap(output: object) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected Tensor or tuple[Tensor, ...], got {type(output)!r}")
    return output


def _capture_hook(
    captures: dict[str, torch.Tensor], name: str
) -> Callable[[nn.Module, tuple[object, ...], object], None]:
    def hook(_: nn.Module, __: tuple[object, ...], output: object) -> None:
        # MCore tensors are [sequence, batch, hidden].  Both models share that
        # layout, so retain it rather than introducing an unnecessary transpose.
        captures[name] = _unwrap(output).detach().cpu()

    return hook


def _register_hooks(
    model: nn.Module, *, moe: bool
) -> tuple[dict[str, torch.Tensor], list[torch.utils.hooks.RemovableHandle]]:
    captures: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for index, layer in enumerate(model.decoder.layers):
        prefix = f"layer={index:02d}"
        # In canonical mode the registered MoELayer is used for route accounting
        # only.  Its unregistered helper is the semantic dense MLP boundary.
        mlp = getattr(layer, "_canonical_dense_mlp", layer.mlp) if moe else layer.mlp
        for name, module in (
            ("attn", layer.self_attention),
            ("mlp_fc1", mlp.linear_fc1),
            ("mlp_fc2", mlp.linear_fc2),
            ("mlp", mlp),
            ("layer_output", layer),
        ):
            handles.append(module.register_forward_hook(_capture_hook(captures, f"{prefix}/{name}")))
    return captures, handles


def _report(
    name: str,
    dense: torch.Tensor,
    moe: torch.Tensor,
    *,
    rank: int,
) -> bool:
    if dense.shape != moe.shape:
        raise AssertionError(f"{name}: shape mismatch {tuple(dense.shape)} != {tuple(moe.shape)}")
    diff = (dense.float() - moe.float()).abs()
    exact = torch.tensor(int(torch.equal(dense, moe)), device="cuda", dtype=torch.int)
    diff_sum = torch.tensor(diff.sum().item(), device="cuda", dtype=torch.float64)
    diff_count = torch.tensor(diff.numel(), device="cuda", dtype=torch.float64)
    diff_max = torch.tensor(diff.max().item(), device="cuda", dtype=torch.float32)
    dist.all_reduce(exact, op=dist.ReduceOp.MIN)
    dist.all_reduce(diff_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(diff_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(diff_max, op=dist.ReduceOp.MAX)
    is_exact = bool(exact.item())
    if rank == 0:
        print(f"{name} exact={is_exact} max_abs={diff_max.item():.8g} mean_abs={(diff_sum / diff_count).item():.8g}")
    return is_exact


def _check_canonical_mlp_parameters(
    dense_model: nn.Module,
    moe_model: nn.Module,
    *,
    rank: int,
) -> str | None:
    """Check that every EP-local canonical expert is a literal dense copy."""
    dense_forward = _forward_model(dense_model)
    moe_forward = _forward_model(moe_model)
    first_nonexact: str | None = None
    for layer_index, (dense_layer, moe_layer) in enumerate(
        zip(dense_forward.decoder.layers, moe_forward.decoder.layers, strict=True)
    ):
        refresh = getattr(moe_layer, "refresh_canonical_dense_mlp_weights", None)
        if refresh is not None:
            refresh()
        canonical = getattr(moe_layer, "_canonical_dense_mlp", None)
        if canonical is None:
            raise AssertionError(f"layer {layer_index} has no canonical dense MLP")
        tensors = (
            ("fc1_weight", dense_layer.mlp.linear_fc1.weight, canonical.linear_fc1.weight),
            (
                "pre_ff_norm",
                dense_layer.mlp.linear_fc1.layer_norm_weight,
                canonical.linear_fc1.layer_norm_weight,
            ),
            ("fc2_weight", dense_layer.mlp.linear_fc2.weight, canonical.linear_fc2.weight),
            (
                "post_ff_norm",
                dense_layer.mlp.linear_fc2.post_layernorm.weight,
                canonical.linear_fc2.post_layernorm.weight,
            ),
        )
        for component, dense, moe in tensors:
            name = f"param/layer={layer_index:02d}/{component}"
            if not _report(name, dense, moe, rank=rank) and first_nonexact is None:
                first_nonexact = name
    return first_nonexact


def main() -> None:
    args = _parse_args()
    rank = _init_distributed(args)
    from megatron.core import parallel_state

    dense_model = moe_model = None
    dense_handles: list[torch.utils.hooks.RemovableHandle] = []
    moe_handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        input_ids = _tokens(args)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
        # In MCore's four-dimensional convention, True marks masked keys.
        attention_mask = ~torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device="cuda"))

        dense_model = _load_mcore_model(
            args.dense_model,
            seq_len,
            args.attention_backend,
            tp=args.tp,
            # A dense model has no expert partitioning.  It still runs in the
            # same initialized TP/EP process-group world as MoE, but its own
            # TransformerConfig must retain EP=1.
            ep=1,
        )
        dense_forward = _forward_model(dense_model)
        dense_captures, dense_handles = _register_hooks(dense_forward, moe=False)
        with torch.inference_mode():
            dense_logits = (
                _unwrap(
                    _run_model(
                        dense_model,
                        dense_forward,
                        input_ids=input_ids,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                    )
                )
                .detach()
                .cpu()
            )
        for handle in dense_handles:
            handle.remove()
        dense_handles = []

        moe_model = _load_mcore_model(
            args.moe_model,
            seq_len,
            args.attention_backend,
            tp=args.tp,
            ep=args.ep,
        )
        moe_layers = _forward_model(moe_model).decoder.layers
        canonical_enabled = bool(moe_layers) and all(
            getattr(layer, "gemma3_moe_canonical_dense_init", False) for layer in moe_layers
        )
        if not canonical_enabled and not args.allow_noncanonical:
            raise AssertionError(
                "MoE checkpoint did not enable gemma3_moe_canonical_dense_init; "
                "this gate would not be a dense-equivalence test."
            )
        moe_forward = _forward_model(moe_model)
        first_nonexact_parameter = None
        if args.check_parameters:
            first_nonexact_parameter = _check_canonical_mlp_parameters(dense_model, moe_model, rank=rank)
        moe_captures, moe_handles = _register_hooks(moe_forward, moe=canonical_enabled)
        with torch.inference_mode():
            moe_logits = (
                _unwrap(
                    _run_model(
                        moe_model,
                        moe_forward,
                        input_ids=input_ids,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                    )
                )
                .detach()
                .cpu()
            )
        for handle in moe_handles:
            handle.remove()
        moe_handles = []

        first_nonexact: str | None = None
        for layer_index in range(len(dense_forward.decoder.layers)):
            for component in ("attn", "mlp_fc1", "mlp_fc2", "mlp", "layer_output"):
                key = f"layer={layer_index:02d}/{component}"
                if (
                    not _report(
                        key,
                        dense_captures[key],
                        moe_captures[key],
                        rank=rank,
                    )
                    and first_nonexact is None
                ):
                    first_nonexact = key

        if dense_logits.shape[0] == seq_len:
            dense_logits = dense_logits.transpose(0, 1)
        if moe_logits.shape[0] == seq_len:
            moe_logits = moe_logits.transpose(0, 1)
        logits_exact = _report(
            "logits",
            dense_logits,
            moe_logits,
            rank=rank,
        )
        top1_matches = torch.tensor(
            (dense_logits.argmax(dim=-1) == moe_logits.argmax(dim=-1)).sum().item(),
            device="cuda",
            dtype=torch.float64,
        )
        top1_count = torch.tensor(dense_logits.argmax(dim=-1).numel(), device="cuda", dtype=torch.float64)
        dist.all_reduce(top1_matches, op=dist.ReduceOp.SUM)
        dist.all_reduce(top1_count, op=dist.ReduceOp.SUM)
        if rank == 0:
            print(f"top1_agreement={(top1_matches / top1_count).item():.8f}")
            if args.check_parameters:
                print(f"first_nonexact_parameter={first_nonexact_parameter or 'none'}")
            print(f"first_nonexact={first_nonexact or 'none'}")
        if not canonical_enabled:
            if rank == 0:
                print("mode=ordinary_sparse_moe (diagnostic only; exactness is not expected)")
        if not args.allow_noncanonical and (
            first_nonexact_parameter is not None or first_nonexact is not None or not logits_exact
        ):
            raise AssertionError("MCore dense/MoE activation parity gate failed")
        if canonical_enabled and first_nonexact_parameter is None and first_nonexact is None and logits_exact:
            if rank == 0:
                print("MCORE_DENSE_MOE_ACTIVATION_PARITY_OK")
    finally:
        for handle in dense_handles:
            handle.remove()
        for handle in moe_handles:
            handle.remove()
        del dense_model, moe_model
        gc.collect()
        torch.cuda.empty_cache()
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
