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

"""Make Gemma 4 Hugging Face checkpoints self-contained for serving."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers.utils import cached_file


@dataclass(frozen=True)
class Gemma4SharedKVAlias:
    """A materialized shared-KV tensor and the logical tensor it aliases."""

    target: str
    source: str


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _text_config(config: Any) -> Any:
    return _config_value(config, "text_config", config)


def is_gemma4_config(config: Any) -> bool:
    return _config_value(config, "model_type") == "gemma4" or _config_value(_text_config(config), "model_type") in {
        "gemma4",
        "gemma4_text",
    }


def _gemma4_shared_kv_layout(config: Any) -> tuple[int, int, list[str]]:
    text_config = _text_config(config)
    num_layers = int(_config_value(text_config, "num_hidden_layers", 0))
    num_shared = int(_config_value(text_config, "num_kv_shared_layers", 0) or 0)
    layer_types = list(_config_value(text_config, "layer_types", []) or [])

    if num_shared < 0 or num_shared >= num_layers:
        raise ValueError(
            "Gemma 4 num_kv_shared_layers must be non-negative and smaller than num_hidden_layers; "
            f"got {num_shared=} and {num_layers=}"
        )
    if num_shared > 0 and len(layer_types) != num_layers:
        raise ValueError("Gemma 4 shared-KV expansion requires one layer_types entry per hidden layer")
    return num_layers, num_shared, layer_types


def _find_language_layer_prefix(tensor_names: set[str]) -> str:
    prefixes = (
        "model.language_model.layers",
        "language_model.model.layers",
        "model.layers",
    )
    for prefix in prefixes:
        if f"{prefix}.0.self_attn.q_proj.weight" in tensor_names:
            return prefix
    raise ValueError("Could not identify the Gemma 4 language-model layer prefix in the checkpoint state dict")


def gemma4_shared_kv_aliases(config: Any, tensor_names: set[str]) -> list[Gemma4SharedKVAlias]:
    """Return every KV tensor name required to make a Gemma 4 checkpoint conventional.

    Transformers does not instantiate K/V projection modules for the final
    ``num_kv_shared_layers`` layers. Current vLLM Gemma 4 loading still expects
    those parameter names, even though inference reuses the logical source
    layer. The conventional checkpoint representation therefore carries a
    cloned tensor under each shared-layer name.
    """

    num_layers, num_shared, layer_types = _gemma4_shared_kv_layout(config)
    if num_shared == 0:
        return []

    layer_prefix = _find_language_layer_prefix(tensor_names)
    first_shared = num_layers - num_shared
    aliases: list[Gemma4SharedKVAlias] = []
    suffixes = ("k_norm.weight", "k_proj.weight", "v_proj.weight")
    for layer_index in range(first_shared, num_layers):
        layer_type = layer_types[layer_index]
        source_candidates = [index for index in range(first_shared) if layer_types[index] == layer_type]
        if not source_candidates:
            raise ValueError(f"No non-shared Gemma 4 KV source layer exists for layer type {layer_type!r}")
        source_layer = source_candidates[-1]
        for suffix in suffixes:
            source = f"{layer_prefix}.{source_layer}.self_attn.{suffix}"
            if source not in tensor_names:
                raise ValueError(f"Missing Gemma 4 shared-KV source tensor {source}")
            aliases.append(
                Gemma4SharedKVAlias(
                    target=f"{layer_prefix}.{layer_index}.self_attn.{suffix}",
                    source=source,
                )
            )
    return aliases


def expand_gemma4_shared_kv_state_dict(
    state_dict: dict[str, torch.Tensor],
    config: Any,
) -> tuple[dict[str, torch.Tensor], list[Gemma4SharedKVAlias]]:
    """Clone omitted Gemma 4 shared-KV parameters into a save-only state dict."""

    if not is_gemma4_config(config):
        return state_dict, []
    aliases = gemma4_shared_kv_aliases(config, set(state_dict))
    if not aliases:
        return state_dict, []

    expanded = state_dict.copy()
    for alias in aliases:
        if alias.target not in expanded:
            # A clone is required: shared storage can be deduplicated again by
            # save_pretrained's safe-serialization pass.
            expanded[alias.target] = expanded[alias.source].clone()
    verify_gemma4_shared_kv_tensor_names(config, set(expanded))
    return expanded, aliases


def verify_gemma4_shared_kv_tensor_names(config: Any, tensor_names: set[str]) -> list[Gemma4SharedKVAlias]:
    """Fail if a Gemma 4 serving artifact omits any shared-KV tensor name."""

    if not is_gemma4_config(config):
        return []
    aliases = gemma4_shared_kv_aliases(config, tensor_names)
    missing = [alias.target for alias in aliases if alias.target not in tensor_names]
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(f"Gemma 4 checkpoint is missing {len(missing)} shared-KV serving tensors: {preview}")
    return aliases


def checkpoint_tensor_names(checkpoint_dir: str | os.PathLike[str]) -> set[str]:
    """Read tensor names from a safetensors checkpoint without loading weights."""

    checkpoint_path = Path(checkpoint_dir)
    index_paths = sorted(checkpoint_path.glob("*.safetensors.index.json"))
    if index_paths:
        tensor_names: set[str] = set()
        for index_path in index_paths:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError(f"Invalid safetensors weight map: {index_path}")
            tensor_names.update(weight_map)
        return tensor_names

    tensor_names = set()
    for model_path in sorted(checkpoint_path.glob("*.safetensors")):
        with safe_open(model_path, framework="pt", device="cpu") as checkpoint:
            tensor_names.update(checkpoint.keys())
    if not tensor_names:
        raise FileNotFoundError(f"No safetensors model files found in {checkpoint_path}")
    return tensor_names


def verify_gemma4_shared_kv_checkpoint(
    checkpoint_dir: str | os.PathLike[str],
    config: Any,
) -> list[Gemma4SharedKVAlias]:
    """Verify that a serialized Gemma 4 artifact retained all expanded aliases."""

    if not is_gemma4_config(config):
        return []
    return verify_gemma4_shared_kv_tensor_names(config, checkpoint_tensor_names(checkpoint_dir))


def _validate_processor_config(path: Path) -> None:
    try:
        processor_config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Gemma 4 processor metadata: {path}") from exc
    if processor_config.get("processor_class") != "Gemma4Processor":
        raise ValueError(f"Gemma 4 processor_config.json has an unexpected processor_class: {path}")


def _resolve_processor_config(config: Any) -> Path | None:
    model_source = _config_value(config, "name_or_path") or _config_value(config, "_name_or_path")
    if not model_source:
        return None

    local_source = Path(str(model_source)).expanduser()
    local_processor_config = local_source / "processor_config.json"
    if local_processor_config.is_file():
        return local_processor_config.resolve()

    try:
        resolved = cached_file(
            str(model_source),
            "processor_config.json",
            revision=_config_value(config, "_commit_hash"),
            local_files_only=True,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
    except (OSError, ValueError):
        return None
    return Path(resolved).resolve() if resolved else None


def ensure_gemma4_processor_metadata(config: Any, checkpoint_dir: str | os.PathLike[str]) -> Path | None:
    """Copy pinned Gemma 4 processor metadata into an HF checkpoint directory."""

    if not is_gemma4_config(config):
        return None

    destination = Path(checkpoint_dir) / "processor_config.json"
    source = _resolve_processor_config(config)
    if source is not None:
        _validate_processor_config(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination.resolve():
            shutil.copy2(source, destination)

    if not destination.is_file():
        raise FileNotFoundError(
            "Gemma 4 HF checkpoint is missing processor_config.json and it could not be recovered from the "
            f"pinned model source {_config_value(config, 'name_or_path')!r}"
        )
    _validate_processor_config(destination)
    return destination
