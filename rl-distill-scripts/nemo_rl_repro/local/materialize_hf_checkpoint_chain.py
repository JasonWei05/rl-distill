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

"""Validate and materialize a pinned Hugging Face sparse-delta checkpoint chain."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from hf_base_canonicalizer import canonicalize_base, sha256_file
from hf_checkpoint_delta import CODEC, LEGACY_CODEC, reconstruct_delta
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_LOCK = Path(__file__).resolve().parents[1] / "config/e4b_step100_delta_chain.lock.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_locked_chain(lock_path: Path) -> dict[str, Any]:
    lock = read_json(lock_path)
    if lock.get("schema_version") != 1:
        raise ValueError(f"Unsupported chain-lock schema: {lock.get('schema_version')}")
    steps = lock.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Chain lock must contain a non-empty steps list")
    if int(lock["target_step"]) != int(steps[-1]["step"]):
        raise ValueError("Chain lock target_step does not match its final step")
    metadata_files = lock.get("base", {}).get("metadata_files", {})
    if not isinstance(metadata_files, dict):
        raise ValueError("base.metadata_files must be a filename-to-SHA256 object")
    for filename, expected_sha256 in metadata_files.items():
        if Path(filename).name != filename:
            raise ValueError(f"Base metadata filename must not contain directories: {filename!r}")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError(f"Base metadata SHA256 must be a 64-character hex string: {filename!r}")
        try:
            int(expected_sha256, 16)
        except ValueError as error:
            raise ValueError(f"Base metadata SHA256 is not hexadecimal: {filename!r}") from error
    output = lock.get("output", {})
    if not isinstance(output, dict):
        raise ValueError("output must be an object")
    if output.get("expand_gemma4_shared_kv_aliases"):
        expected_output_sha256 = output.get("sha256")
        if expected_output_sha256 is not None:
            if not isinstance(expected_output_sha256, str) or len(expected_output_sha256) != 64:
                raise ValueError("output.sha256 must be a 64-character hexadecimal SHA256")
            try:
                int(expected_output_sha256, 16)
            except ValueError as error:
                raise ValueError("output.sha256 is not hexadecimal") from error
    return lock


def download_manifests(lock: dict[str, Any], *, token: str | None = None) -> dict[int, tuple[Path, dict[str, Any]]]:
    downloaded: dict[int, tuple[Path, dict[str, Any]]] = {}
    for locked_step in lock["steps"]:
        step = int(locked_step["step"])
        manifest_path = Path(
            hf_hub_download(
                repo_id=str(lock["repo_id"]),
                repo_type="model",
                revision=str(lock["repo_revision"]),
                filename=f"checkpoints/step_{step}/delta/delta_manifest.json",
                token=token,
            )
        )
        downloaded[step] = (manifest_path, read_json(manifest_path))
    return downloaded


def validate_chain(lock: dict[str, Any], manifests: dict[int, dict[str, Any]]) -> None:
    locked_steps = lock["steps"]
    expected_steps = [int(item["step"]) for item in locked_steps]
    if sorted(manifests) != expected_steps:
        raise ValueError(f"Expected manifests for steps {expected_steps}, got {sorted(manifests)}")

    base = lock["base"]
    previous: dict[str, Any] | None = None
    for locked, step in zip(locked_steps, expected_steps, strict=True):
        manifest = manifests[step]
        exact_fields = (
            "base_sha256",
            "codec",
            "delta_bytes",
            "target_bytes",
            "target_sha256",
        )
        for field in exact_fields:
            if manifest.get(field) != locked.get(field):
                raise ValueError(
                    f"Step {step} {field} differs from the lock: "
                    f"expected {locked.get(field)!r}, got {manifest.get(field)!r}"
                )
        if int(manifest.get("target_step", -1)) != step:
            raise ValueError(f"Step {step} manifest has target_step={manifest.get('target_step')!r}")
        if manifest.get("codec") not in {CODEC, LEGACY_CODEC}:
            raise ValueError(f"Step {step} uses unsupported codec {manifest.get('codec')!r}")
        if int(manifest["mask_bytes"]) + int(manifest["value_bytes"]) != int(manifest["delta_bytes"]):
            raise ValueError(f"Step {step} delta byte counts are inconsistent")
        if int(manifest["word_count"]) * 2 != int(manifest["target_bytes"]):
            raise ValueError(f"Step {step} word_count is inconsistent with target_bytes")

        if previous is None:
            for manifest_field, lock_field in (
                ("base_model_id", "model_id"),
                ("base_revision", "revision"),
                ("base_canonical_sha256", "canonical_sha256"),
            ):
                if manifest.get(manifest_field) != base.get(lock_field):
                    raise ValueError(
                        f"Anchor {manifest_field} differs from the lock: "
                        f"expected {base.get(lock_field)!r}, got {manifest.get(manifest_field)!r}"
                    )
            if manifest["base_sha256"] != base["canonical_sha256"]:
                raise ValueError("Anchor base_sha256 does not match the canonical base hash")
        else:
            if int(manifest.get("base_step", -1)) != int(previous["target_step"]):
                raise ValueError(f"Step {step} does not point to the preceding locked step")
            if int(locked.get("base_step", -1)) != int(previous["target_step"]):
                raise ValueError(f"Step {step} lock has an invalid base_step")
            if manifest["base_sha256"] != previous["target_sha256"]:
                raise ValueError(f"Step {step} base hash does not match step {previous['target_step']} target hash")
            if int(manifest["target_bytes"]) != int(previous["target_bytes"]):
                raise ValueError(f"Step {step} changes the checkpoint byte length")
        previous = manifest

    if int(lock["target_step"]) != int(previous["target_step"]):
        raise ValueError("Final manifest does not match the locked target step")


def verify_payloads(snapshot: Path, lock: dict[str, Any], manifests: dict[int, dict[str, Any]]) -> None:
    for item in lock["steps"]:
        step = int(item["step"])
        delta_dir = snapshot / f"checkpoints/step_{step}/delta"
        expected = {
            "changed_mask.bitset.zst": int(manifests[step]["mask_bytes"]),
            "add_values.u16.zst": int(manifests[step]["value_bytes"]),
        }
        for filename, expected_bytes in expected.items():
            path = delta_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing delta payload: {path}")
            if path.stat().st_size != expected_bytes:
                raise ValueError(f"Unexpected size for {path}: expected {expected_bytes}, got {path.stat().st_size}")


def copy_hf_metadata(snapshot: Path, anchor_step: int, target_step: int, output_dir: Path) -> None:
    anchor_root = snapshot / f"checkpoints/step_{anchor_step}"
    target_root = snapshot / f"checkpoints/step_{target_step}"
    model_root = anchor_root / "policy/weights/model/consolidated"
    tokenizer_root = anchor_root / "policy/tokenizer"
    output_dir.mkdir(parents=True, exist_ok=False)
    for source in sorted(model_root.glob("*.json")):
        shutil.copy2(source, output_dir / source.name)
    for source in sorted(tokenizer_root.iterdir()):
        if source.is_file():
            shutil.copy2(source, output_dir / source.name)
    for filename in ("config.yaml", "training_info.json"):
        source = target_root / filename
        if not source.is_file():
            raise FileNotFoundError(f"Target checkpoint {target_step} does not contain {filename}")
        shutil.copy2(source, output_dir / filename)
    if not (output_dir / "config.json").is_file():
        raise FileNotFoundError(f"Anchor checkpoint {anchor_step} does not contain config.json")
    if not (output_dir / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"Anchor checkpoint {anchor_step} does not contain a model index")


def copy_locked_base_metadata(base: dict[str, Any], output_dir: Path, *, token: str | None = None) -> None:
    """Copy processor metadata pinned to the immutable base-model revision.

    NeMo checkpoints contain tokenizer and model metadata, but Gemma 4's vLLM
    loader also constructs ``Gemma4Processor`` and therefore requires
    ``processor_config.json``. Keep such files in the chain lock so a
    materialized checkpoint is reproducible and directly vLLM-loadable.
    """

    for filename, expected_sha256 in sorted(base.get("metadata_files", {}).items()):
        source = Path(
            hf_hub_download(
                repo_id=str(base["model_id"]),
                repo_type="model",
                revision=str(base["revision"]),
                filename=filename,
                token=token,
            )
        )
        actual_sha256 = sha256_file(source)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Pinned base metadata hash mismatch for {filename}: expected {expected_sha256}, got {actual_sha256}"
            )
        destination = output_dir / filename
        if destination.exists():
            if sha256_file(destination) != expected_sha256:
                raise ValueError(f"Checkpoint metadata conflicts with pinned base file {filename}")
            continue
        shutil.copy2(source, destination)


def expand_gemma4_shared_kv_aliases(
    source_model: Path,
    config_path: Path,
    output_model: Path,
) -> list[dict[str, Any]]:
    """Materialize weights omitted by Transformers' shared-tensor saving.

    Gemma 4 E4B shares the KV cache of its final layers with the last
    non-shared layer of the same attention type. Transformers understands the
    tied/shared parameters and may omit their duplicate tensors when saving.
    vLLM 0.25.1 correctly skips K/V projection use for those layers, but still
    requires every shared layer's unused ``k_norm`` parameter to be present at
    load time. Expanding all three omitted KV tensors from the logical source
    layer yields a conventional, deterministic checkpoint while preserving
    model behavior.
    """

    config = read_json(config_path)
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise ValueError(f"Expected text_config object in {config_path}")
    num_layers = int(text_config.get("num_hidden_layers", 0))
    num_shared = int(text_config.get("num_kv_shared_layers", 0))
    layer_types = text_config.get("layer_types")
    if num_shared <= 0:
        shutil.copy2(source_model, output_model)
        return []
    if not isinstance(layer_types, list) or len(layer_types) != num_layers:
        raise ValueError("Gemma 4 shared-KV expansion requires one layer_types entry per hidden layer")
    first_shared = num_layers - num_shared
    if first_shared <= 0:
        raise ValueError("num_kv_shared_layers must be smaller than num_hidden_layers")

    with safe_open(source_model, framework="pt", device="cpu") as source:
        metadata = source.metadata()
        tensors = {name: source.get_tensor(name) for name in source.keys()}

    prefixes = (
        "model.language_model.layers",
        "language_model.model.layers",
        "model.layers",
    )
    layer_prefix = next(
        (prefix for prefix in prefixes if any(name.startswith(f"{prefix}.0.") for name in tensors)),
        None,
    )
    if layer_prefix is None:
        raise ValueError("Could not identify Gemma 4 language-model layer prefix")

    aliases: list[dict[str, Any]] = []
    suffixes = ("k_norm.weight", "k_proj.weight", "v_proj.weight")
    for layer_index in range(first_shared, num_layers):
        layer_type = layer_types[layer_index]
        candidates = [index for index in range(first_shared) if layer_types[index] == layer_type]
        if not candidates:
            raise ValueError(f"No non-shared source layer exists for layer type {layer_type!r}")
        source_layer = candidates[-1]
        for suffix in suffixes:
            target_name = f"{layer_prefix}.{layer_index}.self_attn.{suffix}"
            if target_name in tensors:
                continue
            source_name = f"{layer_prefix}.{source_layer}.self_attn.{suffix}"
            if source_name not in tensors:
                raise ValueError(f"Missing shared-KV source tensor {source_name}")
            tensors[target_name] = tensors[source_name].clone()
            aliases.append(
                {
                    "target": target_name,
                    "source": source_name,
                    "layer_type": layer_type,
                }
            )

    save_file(dict(sorted(tensors.items())), output_model, metadata=metadata)
    return aliases


def update_single_file_index(index_path: Path, model_path: Path) -> None:
    index = read_json(index_path)
    with safe_open(model_path, framework="pt", device="cpu") as model:
        tensor_names = list(model.keys())
        total_size = sum(
            model.get_tensor(name).numel() * model.get_tensor(name).element_size() for name in tensor_names
        )
    index["metadata"] = dict(index.get("metadata", {}), total_size=total_size)
    index["weight_map"] = {name: model_path.name for name in tensor_names}
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def materialize(
    lock: dict[str, Any],
    manifests: dict[int, dict[str, Any]],
    *,
    snapshot: Path,
    output_dir: Path,
    work_dir: Path,
    token: str | None = None,
    keep_intermediates: bool = False,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if partial_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing partial output directory: {partial_dir}")
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty work directory: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    verify_payloads(snapshot, lock, manifests)

    anchor_step = int(lock["steps"][0]["step"])
    anchor_model_root = snapshot / f"checkpoints/step_{anchor_step}/policy/weights/model/consolidated"
    target_index = anchor_model_root / "model.safetensors.index.json"
    if not target_index.is_file():
        raise FileNotFoundError(f"Missing anchor model index: {target_index}")

    base = lock["base"]
    source_base = Path(
        hf_hub_download(
            repo_id=str(base["model_id"]),
            repo_type="model",
            revision=str(base["revision"]),
            filename="model.safetensors",
            token=token,
        )
    )
    current = work_dir / "canonical_base.safetensors"
    metadata = canonicalize_base(source_base, target_index, current)
    if metadata["canonical_sha256"] != base["canonical_sha256"]:
        raise ValueError(
            "Canonical base SHA-256 differs from the lock: "
            f"expected {base['canonical_sha256']}, got {metadata['canonical_sha256']}"
        )

    for item in lock["steps"]:
        step = int(item["step"])
        target = work_dir / f"step_{step}.safetensors"
        reconstruct_delta(
            current,
            snapshot / f"checkpoints/step_{step}/delta",
            target,
            manifests[step],
        )
        if not keep_intermediates and current.parent == work_dir:
            current.unlink()
        current = target

    try:
        target_step = int(lock["target_step"])
        copy_hf_metadata(snapshot, anchor_step, target_step, partial_dir)
        copy_locked_base_metadata(base, partial_dir, token=token)
        chain_target_sha256 = sha256_file(current)
        if chain_target_sha256 != lock["steps"][-1]["target_sha256"]:
            raise ValueError("Reconstructed model differs from the locked chain target")
        final_model = partial_dir / "model.safetensors"
        output_config = lock.get("output", {})
        if output_config.get("expand_gemma4_shared_kv_aliases"):
            aliases = expand_gemma4_shared_kv_aliases(current, partial_dir / "config.json", final_model)
            if not keep_intermediates:
                current.unlink()
            update_single_file_index(partial_dir / "model.safetensors.index.json", final_model)
        else:
            aliases = []
            try:
                os.replace(current, final_model)
            except OSError:
                shutil.copy2(current, final_model)
                if not keep_intermediates:
                    current.unlink()
        materialized_sha256 = sha256_file(final_model)
        expected_output_sha256 = output_config.get("sha256", lock["steps"][-1]["target_sha256"])
        if expected_output_sha256 is not None and materialized_sha256 != expected_output_sha256:
            raise ValueError(
                "Final materialized model hash differs from the lock: "
                f"expected {expected_output_sha256}, got {materialized_sha256}"
            )
        provenance = {
            "base": lock["base"],
            "checkpoint_repo_id": lock["repo_id"],
            "checkpoint_repo_revision": lock["repo_revision"],
            "chain_target_sha256": chain_target_sha256,
            "materialized_sha256": materialized_sha256,
            "materialized_step": lock["target_step"],
            "base_metadata_files": lock["base"].get("metadata_files", {}),
            "shared_kv_aliases": aliases,
            "steps": [
                {
                    "step": int(item["step"]),
                    "target_sha256": item["target_sha256"],
                }
                for item in lock["steps"]
            ],
        }
        (partial_dir / "materialization_manifest.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        os.replace(partial_dir, output_dir)
    except Exception:
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise
    if not keep_intermediates:
        work_dir.rmdir()
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep-intermediates", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    load_dotenv(repo_root / ".env")
    token = (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    )
    lock = load_locked_chain(args.lock.resolve())
    downloaded = download_manifests(lock, token=token)
    manifests = {step: manifest for step, (_, manifest) in downloaded.items()}
    validate_chain(lock, manifests)
    total_delta_bytes = sum(int(item["delta_bytes"]) for item in lock["steps"])
    print(
        f"Validated {lock['repo_id']}@{lock['repo_revision']} through step {lock['target_step']}; "
        f"target_sha256={lock['steps'][-1]['target_sha256']}; delta_bytes={total_delta_bytes}"
    )
    if args.validate_only:
        return 0
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --validate-only is used")

    patterns = [f"checkpoints/step_{int(item['step'])}/delta/*" for item in lock["steps"]]
    anchor_step = int(lock["steps"][0]["step"])
    patterns.extend(
        [
            f"checkpoints/step_{anchor_step}/policy/tokenizer/*",
            f"checkpoints/step_{anchor_step}/policy/weights/model/consolidated/*.json",
            f"checkpoints/step_{int(lock['target_step'])}/config.yaml",
            f"checkpoints/step_{int(lock['target_step'])}/training_info.json",
        ]
    )
    snapshot = Path(
        snapshot_download(
            repo_id=str(lock["repo_id"]),
            repo_type="model",
            revision=str(lock["repo_revision"]),
            allow_patterns=patterns,
            token=token,
        )
    )
    output_dir = args.output_dir.resolve()
    work_dir = (
        args.work_dir.resolve()
        if args.work_dir is not None
        else output_dir.with_name(f".{output_dir.name}.materialize-work")
    )
    result = materialize(
        lock,
        manifests,
        snapshot=snapshot,
        output_dir=output_dir,
        work_dir=work_dir,
        token=token,
        keep_intermediates=args.keep_intermediates,
    )
    print(f"Materialized and verified {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
