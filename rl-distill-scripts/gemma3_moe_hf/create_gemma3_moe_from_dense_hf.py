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
router. This is mathematically dense-equivalent at initialization, but normal
bf16 sparse dispatch can still differ numerically because it changes GEMM
batch geometry. Use ``--canonical-dense-init`` for a strict dense-equivalence
correctness checkpoint.

The converter accepts either a local Hugging Face snapshot or a Hub model ID.
It materializes the effective Gemma 3 text configuration, copies every
non-MLP tensor, duplicates every dense MLP into every expert, and verifies the
finished checkpoint by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
    parser.add_argument(
        "--dense-model",
        default="google/gemma-3-4b-pt",
        help="Dense Gemma 3 HF checkpoint directory or Hub model ID.",
    )
    parser.add_argument("--revision", default=None, help="Optional dense-model Hub revision.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional Hugging Face download cache.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Empty output directory for MoE HF checkpoint.")
    parser.add_argument(
        "--canonical-output-dir",
        type=Path,
        default=None,
        help=(
            "Also create a correctness-only view with canonical dense-init enabled. "
            "Weight files are hard-linked when the filesystem permits it."
        ),
    )
    parser.add_argument("--num-experts", type=int, required=True, choices=(2, 4))
    parser.add_argument("--num-experts-per-tok", type=int, default=1, choices=(1,))
    parser.add_argument("--router-aux-loss-coef", type=float, default=1e-3)
    parser.add_argument("--router-init-std", type=float, default=None)
    parser.add_argument("--router-seed", type=int, default=1234)
    parser.add_argument(
        "--canonical-dense-init",
        action="store_true",
        help=(
            "Write a correctness-only checkpoint mode that evaluates router decisions "
            "but executes the canonical dense expert over the full token batch."
        ),
    )
    parser.add_argument("--dtype", default=None, choices=("bfloat16", "float16", "float32"))
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify copied tensors, duplicated experts, router shapes, and config invariants (default: true).",
    )
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


def resolve_dense_model(source: str, revision: str | None, cache_dir: Path | None) -> Path:
    local_path = Path(source).expanduser()
    if local_path.is_dir():
        return local_path.resolve()
    if local_path.exists():
        raise NotADirectoryError(f"--dense-model must be a directory or Hub model ID; got {local_path}")

    from huggingface_hub import snapshot_download

    print(f"Downloading dense checkpoint {source!r} revision={revision or 'main'}", flush=True)
    return Path(
        snapshot_download(
            repo_id=source,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            repo_type="model",
        )
    ).resolve()


def prepare_empty_output_dir(output_dir: Path) -> None:
    """Create an output directory, failing closed on files or existing data."""
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"{output_dir} already exists and is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)


def build_config(
    dense_model: Path,
    output_dir: Path,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    router_aux_loss_coef: float,
    dtype_name: str | None,
    canonical_dense_init: bool,
) -> dict[str, Any]:
    # Official Gemma 3 snapshots intentionally keep config.json compact and
    # rely on Transformers defaults for many architecture fields. Resolve the
    # config through Transformers and serialize the effective text config so
    # the upcycled checkpoint does not depend on whichever defaults a later
    # Transformers release happens to provide.
    import transformers
    from transformers import AutoConfig

    base_config = AutoConfig.from_pretrained(dense_model, trust_remote_code=False)
    text_config_object = getattr(base_config, "text_config", base_config)
    text_config = text_config_object.to_dict()
    base_config_dict = base_config.to_dict()
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(base_config, key, None)
        if value is not None:
            text_config[key] = value

    # Generic expert keys trigger an incompatible FusedMoE interception in
    # vLLM's Transformers backend. The native plugin and remote-code model use
    # the Gemma-specific key below.
    text_config.pop("num_experts", None)
    text_config.pop("num_local_experts", None)
    text_config.pop("_name_or_path", None)

    dense_dtype = (
        dtype_name
        or text_config.get("dtype")
        or text_config.get("torch_dtype")
        or base_config_dict.get("dtype")
        or base_config_dict.get("torch_dtype")
    )
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
            "gemma3_moe_num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "router_aux_loss_coef": router_aux_loss_coef,
            "router_pre_softmax": False,
            "router_score_function": "softmax",
            "router_dtype": None,
            "gemma3_moe_canonical_dense_init": canonical_dense_init,
            "tie_word_embeddings": True,
            "transformers_version": transformers.__version__,
        }
    )

    # Normalize non-string mapping keys (for example id2label) exactly as they
    # will appear on disk so the verifier compares the serialized contract.
    serialized_config = json.loads(json.dumps(text_config))
    with (output_dir / "config.json").open("w") as f:
        json.dump(serialized_config, f, indent=2, sort_keys=True)
        f.write("\n")
    return serialized_config


def dtype_from_name(name: str | torch.dtype | None) -> torch.dtype | None:
    if name is None:
        return None
    if isinstance(name, torch.dtype):
        return name
    name = str(name).removeprefix("torch.")
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float": torch.float32,
        "float32": torch.float32,
    }[name]


def _assert_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: dtype {actual.dtype} != {expected.dtype}")
    if not torch.equal(actual, expected):
        max_abs = float((actual.float() - expected.float()).abs().max())
        raise AssertionError(f"{name}: tensor differs from dense source (max_abs={max_abs:.8g})")


def verify_checkpoint(
    dense_model: Path,
    output_dir: Path,
    config: dict[str, Any],
    *,
    num_experts: int,
    cast_dtype: torch.dtype | None,
    router_dtype: torch.dtype,
) -> None:
    """Fail closed unless every output tensor has the intended source."""
    with (output_dir / "config.json").open() as f:
        saved_config = json.load(f)
    if saved_config != config:
        raise AssertionError("Saved config.json does not match the materialized MoE config")
    if saved_config.get("architectures") != ["Gemma3MoeForCausalLM"]:
        raise AssertionError("config.json does not select Gemma3MoeForCausalLM")
    if int(saved_config.get("gemma3_moe_num_experts", -1)) != num_experts:
        raise AssertionError("config.json expert count does not match --num-experts")
    if saved_config.get("num_experts_per_tok") != 1 or saved_config.get("router_pre_softmax") is not False:
        raise AssertionError("Gemma3 MoE conversion requires top-1 post-top-k softmax routing")
    if "num_experts" in saved_config or "num_local_experts" in saved_config:
        raise AssertionError("config.json contains a generic expert-count key")

    dense_loader = DenseTensorLoader(dense_model)
    moe_loader = DenseTensorLoader(output_dir)
    expected_names: set[str] = set()

    def check_copy(output_name: str, dense_suffix: str) -> None:
        expected_names.add(output_name)
        expected = cast_tensor(dense_loader.get(dense_loader.dense_key(dense_suffix)), cast_dtype)
        _assert_equal(output_name, moe_loader.get(output_name), expected)

    try:
        check_copy("model.embed_tokens.weight", "embed_tokens.weight")
        check_copy("model.norm.weight", "norm.weight")
        common_tensors = (
            "input_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
        )
        expert_tensors = (
            ("gate_proj.weight", "mlp.gate_proj.weight"),
            ("up_proj.weight", "mlp.up_proj.weight"),
            ("down_proj.weight", "mlp.down_proj.weight"),
            ("post_layernorm.weight", "post_feedforward_layernorm.weight"),
        )
        hidden_size = int(config["hidden_size"])
        for layer_idx in range(int(config["num_hidden_layers"])):
            source_prefix = f"layers.{layer_idx}."
            output_prefix = f"model.layers.{layer_idx}."
            for suffix in common_tensors:
                check_copy(output_prefix + suffix, source_prefix + suffix)

            router_name = output_prefix + "mlp.router.weight"
            expected_names.add(router_name)
            router = moe_loader.get(router_name)
            if router.shape != (num_experts, hidden_size):
                raise AssertionError(f"{router_name}: shape {tuple(router.shape)} != {(num_experts, hidden_size)}")
            if router.dtype != router_dtype:
                raise AssertionError(f"{router_name}: dtype {router.dtype} != {router_dtype}")
            if not torch.isfinite(router).all():
                raise AssertionError(f"{router_name} contains non-finite values")
            if torch.count_nonzero(router).item() == 0:
                raise AssertionError(f"{router_name} is entirely zero")

            for output_suffix, dense_suffix in expert_tensors:
                expected = cast_tensor(
                    dense_loader.get(dense_loader.dense_key(source_prefix + dense_suffix)),
                    cast_dtype,
                )
                for expert_idx in range(num_experts):
                    output_name = output_prefix + f"mlp.experts.{expert_idx}.{output_suffix}"
                    expected_names.add(output_name)
                    _assert_equal(output_name, moe_loader.get(output_name), expected)

        actual_names = set(moe_loader.weight_map)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            unexpected = sorted(actual_names - expected_names)
            raise AssertionError(f"Checkpoint key mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    finally:
        dense_loader.close()
        moe_loader.close()
    print("GEMMA3_MOE_CHECKPOINT_VERIFIED", flush=True)


def create_canonical_view(source_dir: Path, output_dir: Path) -> None:
    """Create a config-only canonical view while reusing immutable weights."""
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("--canonical-output-dir must differ from --output-dir")
    prepare_empty_output_dir(output_dir)

    linked = copied = 0
    for source in source_dir.iterdir():
        if source.name == "config.json":
            continue
        destination = output_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
            copied += 1
            continue
        try:
            os.link(source, destination)
            linked += 1
        except OSError:
            shutil.copy2(source, destination)
            copied += 1

    with (source_dir / "config.json").open() as f:
        config = json.load(f)
    config["gemma3_moe_canonical_dense_init"] = True
    with (output_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        f"Created canonical dense-init view at {output_dir} ({linked} hard links, {copied} copies)",
        flush=True,
    )


def create(args: argparse.Namespace) -> None:
    if not math.isfinite(args.router_aux_loss_coef) or args.router_aux_loss_coef <= 0:
        raise ValueError("--router-aux-loss-coef must be positive so the top-1 router can train")
    if args.router_init_std is not None and (not math.isfinite(args.router_init_std) or args.router_init_std <= 0):
        raise ValueError(f"--router-init-std must be positive and finite; got {args.router_init_std}")
    if args.canonical_dense_init and args.canonical_output_dir is not None:
        raise ValueError("Use either --canonical-dense-init or --canonical-output-dir, not both")

    dense_model = resolve_dense_model(args.dense_model, args.revision, args.cache_dir)
    output_dir = args.output_dir
    if not (dense_model / "config.json").is_file():
        raise FileNotFoundError(f"{dense_model} does not contain config.json")
    prepare_empty_output_dir(output_dir)
    copy_support_files(dense_model, output_dir)

    dtype = dtype_from_name(args.dtype)
    config = build_config(
        dense_model,
        output_dir,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        router_aux_loss_coef=args.router_aux_loss_coef,
        dtype_name=args.dtype,
        canonical_dense_init=args.canonical_dense_init,
    )
    router_std = args.router_init_std
    if router_std is None:
        router_std = float(config.get("initializer_range", 0.02))
    if not math.isfinite(router_std) or router_std <= 0:
        raise ValueError(f"--router-init-std must be positive and finite; got {router_std}")

    router_dtype = dtype or dtype_from_name(config.get("dtype")) or torch.bfloat16
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
            tensors[dst_prefix + "mlp.router.weight"] = cast_tensor(router, router_dtype)

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

    if args.verify:
        verify_checkpoint(
            dense_model,
            output_dir,
            config,
            num_experts=args.num_experts,
            cast_dtype=dtype,
            router_dtype=router_dtype,
        )
    print(f"Wrote fresh Gemma3-MoE checkpoint to {output_dir}", flush=True)

    if args.canonical_output_dir is not None:
        create_canonical_view(output_dir, args.canonical_output_dir)


def main() -> None:
    create(parse_args())


if __name__ == "__main__":
    main()
