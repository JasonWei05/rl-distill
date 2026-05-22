#!/usr/bin/env python3
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

"""Create a fresh Gemma 3 MoE HF checkpoint from dense Gemma 3 HF weights.

This is the non-SFT upcycling path: every dense MLP is duplicated into
``num_experts`` experts, and each layer gets a randomly initialized top-1
router. Because all experts start identical, initial logits match the dense
model regardless of which expert the random router selects.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DENSE_PREFIXES = ("language_model.model.", "model.")
SUPPORT_SKIP_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-model", type=Path, required=True, help="Dense Gemma 3 HF checkpoint directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Empty output directory for MoE HF checkpoint.")
    parser.add_argument("--num-experts", type=int, required=True, choices=(2, 4))
    parser.add_argument("--num-experts-per-tok", type=int, default=1)
    parser.add_argument("--router-aux-loss-coef", type=float, default=1e-3)
    parser.add_argument("--router-init-std", type=float, default=None)
    parser.add_argument("--router-seed", type=int, default=1234)
    parser.add_argument("--dtype", default=None, choices=("bfloat16", "float16", "float32"))
    return parser.parse_args()


class DenseTensorLoader:
    def __init__(self, root: Path):
        self.root = root
        index_path = root / "model.safetensors.index.json"
        if index_path.is_file():
            with index_path.open() as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            safetensors = sorted(root.glob("*.safetensors"))
            if len(safetensors) != 1:
                raise FileNotFoundError(f"Expected {index_path} or a single safetensors file under {root}")
            with safe_open(safetensors[0], framework="pt", device="cpu") as f:
                self.weight_map = {key: safetensors[0].name for key in f.keys()}
        self._handles: dict[str, Any] = {}

    def close(self) -> None:
        self._handles.clear()

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def get(self, key: str) -> torch.Tensor:
        filename = self.weight_map[key]
        handle = self._handles.get(filename)
        if handle is None:
            handle = safe_open(self.root / filename, framework="pt", device="cpu")
            self._handles[filename] = handle
        return handle.get_tensor(key)

    def dense_key(self, suffix: str) -> str:
        for prefix in DENSE_PREFIXES:
            key = prefix + suffix
            if self.has(key):
                return key
        raise KeyError(f"Could not find dense tensor for suffix {suffix!r}")


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def cast_tensor(tensor: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    if dtype is None:
        return tensor.detach().cpu().contiguous()
    return tensor.detach().cpu().to(dtype=dtype).contiguous()


def write_shard(output_dir: Path, filename: str, tensors: dict[str, torch.Tensor], weight_map: dict[str, str]) -> int:
    tensors = {name: tensor.contiguous() for name, tensor in tensors.items()}
    save_file(tensors, output_dir / filename, metadata={"format": "pt"})
    for name in tensors:
        weight_map[name] = filename
    return sum(tensor_nbytes(tensor) for tensor in tensors.values())


def copy_support_files(dense_model: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in dense_model.iterdir():
        if src.name == "config.json" or src.name == "model.safetensors.index.json" or src.is_dir():
            continue
        if any(src.name.endswith(suffix) for suffix in SUPPORT_SKIP_SUFFIXES):
            continue
        shutil.copy2(src, output_dir / src.name)

    code_dir = Path(__file__).resolve().parent
    for filename in ("configuration_gemma3_moe.py", "modeling_gemma3_moe.py"):
        shutil.copy2(code_dir / filename, output_dir / filename)


def build_config(
    dense_model: Path,
    output_dir: Path,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    router_aux_loss_coef: float,
    dtype_name: str | None,
) -> dict[str, Any]:
    with (dense_model / "config.json").open() as f:
        base_config = json.load(f)

    text_config = dict(base_config.get("text_config", base_config))
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key in base_config:
            text_config[key] = base_config[key]

    dense_dtype = dtype_name or base_config.get("dtype") or base_config.get("torch_dtype") or text_config.get("dtype")
    text_config.update(
        {
            "architectures": ["Gemma3MoeForCausalLM"],
            "auto_map": {
                "AutoConfig": "configuration_gemma3_moe.Gemma3MoeConfig",
                "AutoModel": "modeling_gemma3_moe.Gemma3MoeModel",
                "AutoModelForCausalLM": "modeling_gemma3_moe.Gemma3MoeForCausalLM",
            },
            "dtype": dense_dtype or "bfloat16",
            "model_type": "gemma3_moe",
            "num_experts": num_experts,
            "num_local_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "router_aux_loss_coef": router_aux_loss_coef,
            "router_pre_softmax": False,
            "router_score_function": "softmax",
            "router_dtype": None,
            "tie_word_embeddings": True,
            "transformers_version": "4.56.1",
        }
    )

    with (output_dir / "config.json").open("w") as f:
        json.dump(text_config, f, indent=2, sort_keys=True)
        f.write("\n")
    return text_config


def dtype_from_name(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def create(args: argparse.Namespace) -> None:
    dense_model = args.dense_model
    output_dir = args.output_dir
    if not (dense_model / "config.json").is_file():
        raise FileNotFoundError(f"{dense_model} does not contain config.json")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} already exists and is not empty")

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_support_files(dense_model, output_dir)

    dtype = dtype_from_name(args.dtype)
    config = build_config(
        dense_model,
        output_dir,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        router_aux_loss_coef=args.router_aux_loss_coef,
        dtype_name=args.dtype,
    )
    router_std = args.router_init_std
    if router_std is None:
        router_std = float(config.get("initializer_range", 0.02))

    loader = DenseTensorLoader(dense_model)
    weight_map: dict[str, str] = {}
    total_size = 0
    num_layers = int(config["num_hidden_layers"])
    hidden_size = int(config["hidden_size"])
    shard_count = num_layers + 1

    try:
        embed = cast_tensor(loader.get(loader.dense_key("embed_tokens.weight")), dtype)
        norm = cast_tensor(loader.get(loader.dense_key("norm.weight")), dtype)
        total_size += write_shard(
            output_dir,
            f"model-00001-of-{shard_count:05d}.safetensors",
            {
                "model.embed_tokens.weight": embed,
                "model.norm.weight": norm,
            },
            weight_map,
        )

        for layer_idx in range(num_layers):
            print(f"Writing fresh MoE layer {layer_idx + 1}/{num_layers}", flush=True)
            src_prefix = f"layers.{layer_idx}."
            dst_prefix = f"model.layers.{layer_idx}."
            tensors = {
                dst_prefix + "input_layernorm.weight": cast_tensor(
                    loader.get(loader.dense_key(src_prefix + "input_layernorm.weight")), dtype
                ),
                dst_prefix + "self_attn.q_norm.weight": cast_tensor(
                    loader.get(loader.dense_key(src_prefix + "self_attn.q_norm.weight")), dtype
                ),
                dst_prefix + "self_attn.k_norm.weight": cast_tensor(
                    loader.get(loader.dense_key(src_prefix + "self_attn.k_norm.weight")), dtype
                ),
                dst_prefix + "self_attn.q_proj.weight": cast_tensor(
                    loader.get(loader.dense_key(src_prefix + "self_attn.q_proj.weight")), dtype
                ),
                dst_prefix + "self_attn.k_proj.weight": cast_tensor(
                    loader.get(loader.dense_key(src_prefix + "self_attn.k_proj.weight")), dtype
                ),
                dst_prefix + "self_attn.v_proj.weight": cast_tensor(
                    loader.get(loader.dense_key(src_prefix + "self_attn.v_proj.weight")), dtype
                ),
                dst_prefix + "self_attn.o_proj.weight": cast_tensor(
                    loader.get(loader.dense_key(src_prefix + "self_attn.o_proj.weight")), dtype
                ),
                dst_prefix + "post_attention_layernorm.weight": cast_tensor(
                    loader.get(loader.dense_key(src_prefix + "post_attention_layernorm.weight")), dtype
                ),
                dst_prefix + "pre_feedforward_layernorm.weight": cast_tensor(
                    loader.get(loader.dense_key(src_prefix + "pre_feedforward_layernorm.weight")), dtype
                ),
            }

            router_gen = torch.Generator(device="cpu")
            router_gen.manual_seed(args.router_seed + layer_idx)
            router = torch.randn(args.num_experts, hidden_size, generator=router_gen, dtype=torch.float32) * router_std
            tensors[dst_prefix + "mlp.router.weight"] = cast_tensor(router, dtype or torch.bfloat16)

            gate = cast_tensor(loader.get(loader.dense_key(src_prefix + "mlp.gate_proj.weight")), dtype)
            up = cast_tensor(loader.get(loader.dense_key(src_prefix + "mlp.up_proj.weight")), dtype)
            down = cast_tensor(loader.get(loader.dense_key(src_prefix + "mlp.down_proj.weight")), dtype)
            post_ln = cast_tensor(loader.get(loader.dense_key(src_prefix + "post_feedforward_layernorm.weight")), dtype)
            for expert_idx in range(args.num_experts):
                expert_prefix = dst_prefix + f"mlp.experts.{expert_idx}."
                tensors[expert_prefix + "gate_proj.weight"] = gate.clone()
                tensors[expert_prefix + "up_proj.weight"] = up.clone()
                tensors[expert_prefix + "down_proj.weight"] = down.clone()
                tensors[expert_prefix + "post_layernorm.weight"] = post_ln.clone()

            total_size += write_shard(
                output_dir,
                f"model-{layer_idx + 2:05d}-of-{shard_count:05d}.safetensors",
                tensors,
                weight_map,
            )

        with (output_dir / "model.safetensors.index.json").open("w") as f:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2, sort_keys=True)
            f.write("\n")
    finally:
        loader.close()

    print(f"Wrote fresh Gemma3-MoE checkpoint to {output_dir}")


def main() -> None:
    create(parse_args())


if __name__ == "__main__":
    main()
