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

"""Convert Gemma 3 MoE Megatron dist checkpoints to HF safetensors.

The source checkpoints produced by the local SFT runs store text weights as
stacked tensors, for example:

    language_model.decoder.layers.mlp.experts.experts.linear_fc1.weight
        [num_layers, num_experts, 2 * intermediate_size, hidden_size]

This script splits those tensors into a Hugging Face remote-code checkpoint
with per-layer/per-expert keys understood by ``modeling_gemma3_moe.py``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import save_file
from torch.distributed.checkpoint import load as dcp_load

LANG_PREFIX = "language_model."
SOURCE_KEYS = {
    "embed": LANG_PREFIX + "embedding.word_embeddings.weight",
    "qkv": LANG_PREFIX + "decoder.layers.self_attention.linear_qkv.weight",
    "qkv_ln": LANG_PREFIX + "decoder.layers.self_attention.linear_qkv.layer_norm_weight",
    "q_norm": LANG_PREFIX + "decoder.layers.self_attention.q_layernorm.weight",
    "k_norm": LANG_PREFIX + "decoder.layers.self_attention.k_layernorm.weight",
    "o": LANG_PREFIX + "decoder.layers.self_attention.linear_proj.weight",
    "post_attn_ln": LANG_PREFIX + "decoder.layers.self_attention.linear_proj.post_layernorm.weight",
    "pre_mlp_ln": LANG_PREFIX + "decoder.layers.pre_mlp_layernorm.weight",
    "router": LANG_PREFIX + "decoder.layers.mlp.router.weight",
    "fc1": LANG_PREFIX + "decoder.layers.mlp.experts.experts.linear_fc1.weight",
    "fc2": LANG_PREFIX + "decoder.layers.mlp.experts.experts.linear_fc2.weight",
    "expert_post_ln": LANG_PREFIX + "decoder.layers.mlp.experts.experts.linear_fc2.post_layernorm.weight",
    "final_ln": LANG_PREFIX + "decoder.final_layernorm.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        help="Snapshot root, global_step dir, or dist_ckpt dir. Use --hf-repo-id instead to download.",
    )
    parser.add_argument("--hf-repo-id", help="HF repo containing global_step_250/dist_ckpt.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/hf-gemma3-moe-cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-experts", type=int)
    parser.add_argument("--num-experts-per-tok", type=int, default=1)
    parser.add_argument("--router-aux-loss-coef", type=float, default=1e-3)
    parser.add_argument("--router-pre-softmax", action="store_true")
    parser.add_argument("--router-score-function", default="softmax")
    parser.add_argument("--router-dtype", choices=["fp32", "fp64"])
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo-id", help="Destination HF repo id for --push-to-hub.")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.hf_repo_id:
        root = Path(
            snapshot_download(
                repo_id=args.hf_repo_id,
                revision=args.revision,
                cache_dir=args.cache_dir,
                allow_patterns=[
                    "global_step_250/dist_ckpt/*",
                    "global_step_250/huggingface/*",
                    "global_step_250/transformer_config.json",
                ],
            )
        )
    elif args.input_path:
        root = args.input_path
    else:
        raise SystemExit("Provide either --input-path or --hf-repo-id.")

    if (root / ".metadata").is_file():
        dist_ckpt = root
        step_dir = root.parent
    elif (root / "dist_ckpt" / ".metadata").is_file():
        step_dir = root
        dist_ckpt = root / "dist_ckpt"
    elif (root / "global_step_250" / "dist_ckpt" / ".metadata").is_file():
        step_dir = root / "global_step_250"
        dist_ckpt = step_dir / "dist_ckpt"
    else:
        raise FileNotFoundError(f"Could not find a dist_ckpt/.metadata below {root}")

    hf_template = step_dir / "huggingface"
    if not (hf_template / "config.json").is_file():
        raise FileNotFoundError(f"Could not find Hugging Face template config under {hf_template}")
    return dist_ckpt, hf_template


def read_metadata(dist_ckpt: Path):
    with (dist_ckpt / ".metadata").open("rb") as f:
        return pickle.load(f)


def load_tensor(dist_ckpt: Path, metadata: Any, key: str) -> torch.Tensor:
    tensor_meta = metadata.state_dict_metadata[key]
    tensor = torch.empty(tuple(tensor_meta.size), dtype=tensor_meta.properties.dtype, device="cpu")
    dcp_load({key: tensor}, checkpoint_id=str(dist_ckpt))
    return tensor


def split_qkv(qkv: torch.Tensor, *, num_attention_heads: int, num_query_groups: int, head_dim: int):
    heads_per_group = num_attention_heads // num_query_groups
    total_heads_per_group = heads_per_group + 2
    qkv_total_dim = num_attention_heads + 2 * num_query_groups
    qkv = qkv.view(qkv_total_dim, head_dim, qkv.shape[-1])

    q_slice = torch.cat(
        [
            torch.arange(total_heads_per_group * i, total_heads_per_group * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(total_heads_per_group - 2, qkv_total_dim, total_heads_per_group)
    v_slice = torch.arange(total_heads_per_group - 1, qkv_total_dim, total_heads_per_group)
    return (
        qkv[q_slice].reshape(-1, qkv.shape[-1]).contiguous(),
        qkv[k_slice].reshape(-1, qkv.shape[-1]).contiguous(),
        qkv[v_slice].reshape(-1, qkv.shape[-1]).contiguous(),
    )


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def write_shard(output_dir: Path, filename: str, tensors: dict[str, torch.Tensor], weight_map: dict[str, str]) -> int:
    tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
    save_file(tensors, output_dir / filename, metadata={"format": "pt"})
    for name in tensors:
        weight_map[name] = filename
    return sum(tensor_nbytes(tensor) for tensor in tensors.values())


def build_config(
    hf_template: Path,
    output_dir: Path,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    router_aux_loss_coef: float,
    router_pre_softmax: bool,
    router_score_function: str,
    router_dtype: str | None,
) -> dict[str, Any]:
    with (hf_template / "config.json").open() as f:
        base_config = json.load(f)

    text_config = dict(base_config.get("text_config", base_config))
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key in base_config:
            text_config[key] = base_config[key]

    text_config.update(
        {
            "architectures": ["Gemma3MoeForCausalLM"],
            "auto_map": {
                "AutoConfig": "configuration_gemma3_moe.Gemma3MoeConfig",
                "AutoModel": "modeling_gemma3_moe.Gemma3MoeModel",
                "AutoModelForCausalLM": "modeling_gemma3_moe.Gemma3MoeForCausalLM",
            },
            "dtype": base_config.get("dtype", base_config.get("torch_dtype", "bfloat16")),
            "model_type": "gemma3_moe",
            "num_experts": num_experts,
            "num_local_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "router_aux_loss_coef": router_aux_loss_coef,
            "router_pre_softmax": router_pre_softmax,
            "router_score_function": router_score_function,
            "router_dtype": router_dtype,
            "tie_word_embeddings": True,
            "transformers_version": "4.56.1",
        }
    )

    with (output_dir / "config.json").open("w") as f:
        json.dump(text_config, f, indent=2, sort_keys=True)
        f.write("\n")
    return text_config


def copy_support_files(hf_template: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in hf_template.iterdir():
        if src.name == "config.json" or src.is_dir():
            continue
        shutil.copy2(src, output_dir / src.name)

    code_dir = Path(__file__).resolve().parent
    for filename in ("configuration_gemma3_moe.py", "modeling_gemma3_moe.py"):
        shutil.copy2(code_dir / filename, output_dir / filename)


def convert(args: argparse.Namespace) -> None:
    dist_ckpt, hf_template = resolve_paths(args)
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} already exists and is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading metadata from {dist_ckpt}")
    metadata = read_metadata(dist_ckpt)
    missing = [key for key in SOURCE_KEYS.values() if key not in metadata.state_dict_metadata]
    if missing:
        raise KeyError(f"Checkpoint is missing expected text tensors: {missing}")

    print("Copying tokenizer, generation config, and remote-code files")
    copy_support_files(hf_template, output_dir)

    print("Loading stacked text tensors from dist checkpoint")
    src = {name: load_tensor(dist_ckpt, metadata, key) for name, key in SOURCE_KEYS.items()}
    num_layers = int(src["qkv"].shape[0])
    num_experts = int(src["router"].shape[1])
    if args.num_experts is not None and args.num_experts != num_experts:
        raise ValueError(f"--num-experts={args.num_experts} but checkpoint has {num_experts}")

    config = build_config(
        hf_template,
        output_dir,
        num_experts=num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        router_aux_loss_coef=args.router_aux_loss_coef,
        router_pre_softmax=args.router_pre_softmax,
        router_score_function=args.router_score_function,
        router_dtype=args.router_dtype,
    )

    num_attention_heads = int(config["num_attention_heads"])
    num_query_groups = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim", config["hidden_size"] // config["num_attention_heads"]))
    weight_map: dict[str, str] = {}
    total_size = 0
    shard_count = num_layers + 1

    print("Writing embedding/final-norm shard")
    total_size += write_shard(
        output_dir,
        f"model-00001-of-{shard_count:05d}.safetensors",
        {
            "model.embed_tokens.weight": src["embed"],
            "model.norm.weight": src["final_ln"],
        },
        weight_map,
    )

    for layer_idx in range(num_layers):
        print(f"Writing layer {layer_idx + 1}/{num_layers}")
        q, k, v = split_qkv(
            src["qkv"][layer_idx],
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            head_dim=head_dim,
        )
        tensors = {
            f"model.layers.{layer_idx}.input_layernorm.weight": src["qkv_ln"][layer_idx],
            f"model.layers.{layer_idx}.self_attn.q_norm.weight": src["q_norm"][layer_idx],
            f"model.layers.{layer_idx}.self_attn.k_norm.weight": src["k_norm"][layer_idx],
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": q,
            f"model.layers.{layer_idx}.self_attn.k_proj.weight": k,
            f"model.layers.{layer_idx}.self_attn.v_proj.weight": v,
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": src["o"][layer_idx],
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": src["post_attn_ln"][layer_idx],
            f"model.layers.{layer_idx}.pre_feedforward_layernorm.weight": src["pre_mlp_ln"][layer_idx],
            f"model.layers.{layer_idx}.mlp.router.weight": src["router"][layer_idx],
        }
        for expert_idx in range(num_experts):
            gate, up = torch.chunk(src["fc1"][layer_idx, expert_idx], 2, dim=0)
            expert_prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            tensors[f"{expert_prefix}.gate_proj.weight"] = gate
            tensors[f"{expert_prefix}.up_proj.weight"] = up
            tensors[f"{expert_prefix}.down_proj.weight"] = src["fc2"][layer_idx, expert_idx]
            tensors[f"{expert_prefix}.post_layernorm.weight"] = src["expert_post_ln"][layer_idx, expert_idx]

        total_size += write_shard(
            output_dir,
            f"model-{layer_idx + 2:05d}-of-{shard_count:05d}.safetensors",
            tensors,
            weight_map,
        )

    with (output_dir / "model.safetensors.index.json").open("w") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote HF Gemma3-MoE checkpoint to {output_dir}")
    if args.push_to_hub:
        if not args.hub_repo_id:
            raise ValueError("--hub-repo-id is required with --push-to-hub")
        api = HfApi()
        api.create_repo(args.hub_repo_id, private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.hub_repo_id, folder_path=str(output_dir))
        print(f"Uploaded to https://huggingface.co/{args.hub_repo_id}")


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
