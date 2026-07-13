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

"""Compare dense and freshly-upcycled Gemma 3 activations layer by layer.

This is the initialization correctness gate that must pass before interpreting
any RL metric.  It feeds identical, fixed token ids to direct Transformers
forwards (no vLLM, Megatron, cache, sampling, or router replay involved).

Examples:

  # Strict correctness gate using a canonical dense-init checkpoint.
  CUDA_VISIBLE_DEVICES=0 python rl-distill-scripts/check_gemma3_dense_moe_activations.py \\
      --moe-model /checkpoints/gemma3-4b-pt-moe-2e-canonical \\
      --seq-len 2048 --all-components

  # Diagnostic control: route every token to expert 0.  This preserves the
  # dense MLP's full-batch GEMM shape and distinguishes weight-mapping errors
  # from route-partition numerical effects.
  CUDA_VISIBLE_DEVICES=0 python rl-distill-scripts/check_gemma3_dense_moe_activations.py \\
      --moe-model /checkpoints/gemma3-4b-pt-moe-2e \\
      --seq-len 2048 --allow-noncanonical --force-expert-zero
"""

from __future__ import annotations

import argparse
import gc
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEFAULT_DENSE_MODEL = "google/gemma-3-4b-pt"
DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "data" / "gemma3_it_chat_template.jinja"
DEFAULT_PROMPT = "Solve the equation 3x + 7 = 22. Explain your reasoning."


def parse_args() -> argparse.Namespace:
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
        "--force-expert-zero",
        action="store_true",
        help="Zero every router weight so top-1 deterministically selects expert 0.",
    )
    parser.add_argument(
        "--all-components",
        action="store_true",
        help="Print matching attention/norm stages too, not only the MLP/layer outputs.",
    )
    parser.add_argument(
        "--allow-noncanonical",
        action="store_true",
        help="Report ordinary sparse-MoE drift instead of requiring exact canonical parity.",
    )
    return parser.parse_args()


def _unwrap(output: object) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected Tensor or tuple[Tensor, ...], got {type(output)!r}")
    return output


def _capture_hook(name: str, captures: dict[str, torch.Tensor]) -> Callable[[nn.Module, tuple, object], None]:
    def hook(_: nn.Module, __: tuple, output: object) -> None:
        # Store bf16 CPU copies.  A 2K context for all 34 layers is small enough
        # for host RAM and avoids retaining a GPU autograd/forward graph.
        captures[name] = _unwrap(output).detach().cpu()

    return hook


def _text_layers(model: nn.Module) -> nn.ModuleList:
    """Return the text decoder layers for either dense multimodal or MoE text HF models."""
    root = model.model
    if hasattr(root, "layers"):
        return root.layers
    if hasattr(root, "language_model") and hasattr(root.language_model, "layers"):
        return root.language_model.layers
    raise AttributeError(f"Could not find text decoder layers under {type(model).__name__}")


def _register_dense_hooks(
    model: nn.Module, captures: dict[str, torch.Tensor]
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for index, layer in enumerate(_text_layers(model)):
        prefix = f"layer={index:02d}"
        for name, module in (
            ("input_norm", layer.input_layernorm),
            ("attn", layer.self_attn),
            ("post_attn_norm", layer.post_attention_layernorm),
            ("pre_ff_norm", layer.pre_feedforward_layernorm),
            ("dense_mlp", layer.mlp),
            # The MoE's MLP output includes the duplicated post-MLP norm.
            ("ff_output", layer.post_feedforward_layernorm),
            ("layer_output", layer),
        ):
            handles.append(module.register_forward_hook(_capture_hook(f"{prefix}/{name}", captures)))
    return handles


def _register_moe_hooks(model: nn.Module, captures: dict[str, torch.Tensor]) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for index, layer in enumerate(_text_layers(model)):
        prefix = f"layer={index:02d}"
        for name, module in (
            ("input_norm", layer.input_layernorm),
            ("attn", layer.self_attn),
            ("post_attn_norm", layer.post_attention_layernorm),
            ("pre_ff_norm", layer.pre_feedforward_layernorm),
            # Equivalent to dense MLP followed by post_feedforward_layernorm.
            ("ff_output", layer.mlp),
            ("layer_output", layer),
        ):
            handles.append(module.register_forward_hook(_capture_hook(f"{prefix}/{name}", captures)))
    return handles


def _token_ids(args: argparse.Namespace) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(args.moe_model, trust_remote_code=True)
    if args.data is not None:
        import pandas as pd

        row = pd.read_parquet(args.data).iloc[args.prompt_index]
        messages = [dict(message) for message in row["prompt"]]
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
        raise ValueError(f"--seq-len {args.seq_len} is shorter than the rendered prompt ({input_ids.shape[1]})")
    repeats = (args.seq_len + input_ids.shape[1] - 1) // input_ids.shape[1]
    return input_ids.repeat(1, repeats)[:, : args.seq_len].cuda()


def _load_and_capture(
    path: str,
    input_ids: torch.Tensor,
    captures: dict[str, torch.Tensor],
    register_hooks: Callable[[nn.Module, dict[str, torch.Tensor]], list[torch.utils.hooks.RemovableHandle]],
    *,
    trust_remote_code: bool,
    force_expert_zero: bool = False,
) -> torch.Tensor:
    model = (
        AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=trust_remote_code,
            dtype=torch.bfloat16,
        )
        .cuda()
        .eval()
    )
    if force_expert_zero:
        with torch.no_grad():
            for layer in _text_layers(model):
                layer.mlp.router.weight.zero_()

    handles = register_hooks(model, captures)
    try:
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits.detach().cpu()
    finally:
        for handle in handles:
            handle.remove()
        del model
        torch.cuda.empty_cache()
        gc.collect()
    return logits


def _report_tensor(name: str, dense: torch.Tensor, moe: torch.Tensor) -> tuple[bool, float, float]:
    if dense.shape != moe.shape:
        raise AssertionError(f"{name}: shape mismatch {tuple(dense.shape)} != {tuple(moe.shape)}")
    diff = (dense.float() - moe.float()).abs()
    exact = torch.equal(dense, moe)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    return exact, max_abs, mean_abs


def main() -> None:
    args = parse_args()
    moe_config = AutoConfig.from_pretrained(args.moe_model, trust_remote_code=True)
    canonical_enabled = bool(getattr(moe_config, "gemma3_moe_canonical_dense_init", False))
    if not canonical_enabled and not args.allow_noncanonical:
        raise AssertionError(
            "The MoE checkpoint does not enable gemma3_moe_canonical_dense_init. "
            "Use the canonical view for the strict gate, or pass --allow-noncanonical for diagnostics."
        )
    input_ids = _token_ids(args)
    prompt_source = f"data[{args.prompt_index}]" if args.data is not None else "--prompt"
    print(f"input_tokens={input_ids.shape[1]} prompt_source={prompt_source} canonical_dense_init={canonical_enabled}")

    dense_captures: dict[str, torch.Tensor] = {}
    dense_logits = _load_and_capture(
        args.dense_model,
        input_ids,
        dense_captures,
        _register_dense_hooks,
        trust_remote_code=False,
    )
    moe_captures: dict[str, torch.Tensor] = {}
    moe_logits = _load_and_capture(
        args.moe_model,
        input_ids,
        moe_captures,
        _register_moe_hooks,
        trust_remote_code=True,
        force_expert_zero=args.force_expert_zero,
    )

    compared_components = ("ff_output", "layer_output")
    if args.all_components:
        compared_components = ("input_norm", "attn", "post_attn_norm", "pre_ff_norm", *compared_components)

    first_nonexact: str | None = None
    dense_layer_count = len([key for key in dense_captures if key.endswith("/layer_output")])
    moe_layer_count = len([key for key in moe_captures if key.endswith("/layer_output")])
    if dense_layer_count != moe_layer_count:
        raise AssertionError(f"decoder layer count mismatch: dense={dense_layer_count}, moe={moe_layer_count}")
    for layer_index in range(dense_layer_count):
        prefix = f"layer={layer_index:02d}"
        for component in compared_components:
            key = f"{prefix}/{component}"
            exact, max_abs, mean_abs = _report_tensor(key, dense_captures[key], moe_captures[key])
            print(f"{key} exact={exact} max_abs={max_abs:.8g} mean_abs={mean_abs:.8g}")
            if not exact and first_nonexact is None:
                first_nonexact = key

    logit_exact, logit_max_abs, logit_mean_abs = _report_tensor("logits", dense_logits, moe_logits)
    top1_agreement = float((dense_logits.argmax(dim=-1) == moe_logits.argmax(dim=-1)).float().mean())
    print(
        f"logits exact={logit_exact} max_abs={logit_max_abs:.8g} "
        f"mean_abs={logit_mean_abs:.8g} top1_agreement={top1_agreement:.8f}"
    )
    print(f"first_nonexact={first_nonexact or 'none'}")
    if canonical_enabled and (first_nonexact is not None or not logit_exact or top1_agreement != 1.0):
        raise AssertionError("HF dense/MoE activation parity gate failed")
    if canonical_enabled:
        print("HF_DENSE_MOE_ACTIVATION_PARITY_OK")
    else:
        print("mode=ordinary_sparse_moe (diagnostic only; exactness is not expected)")


if __name__ == "__main__":
    main()
