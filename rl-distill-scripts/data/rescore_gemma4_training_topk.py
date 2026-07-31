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

"""Post-hoc Gemma 4 HF teacher-forced top-k target rescorer.

The source vLLM trace bundle is immutable. This writes a separate one-to-one,
order-preserving target overlay whose rows are cryptographically bound to the
validated source dataset, shards, and trace-ID sets.

Operator instructions: ``GEMMA4_TRAINING_TOPK_RESCORER.md``. GPU execution
requires separate, explicit approval.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OVERLAY_SCHEMA_VERSION = "gemma4-hf-bf16-sdpa-topk-overlay-v1"
OVERLAY_MANIFEST_VERSION = 1
TOPK_WIDTH = 128
FP16_MASS_TOLERANCE = 2.5e-3
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_RECEIPT_NAME = "parity_receipt.json"
COPIED_SOURCE_FIELDS = {
    "trace_id": "trace_id",
    "direction": "direction",
    "split": "split",
    "source_dataset": "source_dataset",
    "source_dataset_sha256": "source_dataset_sha256",
    "source_uid": "source_uid",
    "question_sha256": "question_sha256",
    "prompt_index": "prompt_index",
    "sample_index": "sample_index",
    "question_text": "question_text",
    "gold_answer": "gold_answer",
    "strict_grade": "strict_grade",
    "strict_correct": "strict_correct",
    "strict_prediction": "strict_prediction",
    "response_text": "response_text",
    "vllm_response_text": "vllm_response_text",
    "prompt_token_ids": "prompt_token_ids",
    "response_token_ids": "response_token_ids",
    "input_ids": "input_ids",
    "response_mask": "response_mask",
    "prompt_length": "prompt_length",
    "response_length": "response_length",
    "shard_id": "shard_id",
    "row_within_shard": "row_within_shard",
    "source_generation_config_sha256": "generation_config_sha256",
    "source_teacher_model": "teacher_model",
    "source_teacher_revision": "teacher_revision",
    "source_teacher_content_sha256": "teacher_content_sha256",
    "source_sampling_parameters_json": "sampling_parameters_json",
    "source_environment_versions_json": "environment_versions_json",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _repo_helpers(repo_root: Path):
    data_dir = repo_root / "rl-distill-scripts" / "data"
    if not data_dir.is_dir():
        raise ValueError(f"not an rl-distill checkout: {repo_root}")
    sys.path.insert(0, str(data_dir))
    import gemma4_distill_trace_schema as trace_schema
    import gemma4_model_identity as model_identity

    return trace_schema, model_identity


def overlay_schema(topk_width: int = TOPK_WIDTH) -> pa.Schema:
    ids = pa.list_(pa.list_(pa.int32(), topk_width))
    logprobs = pa.list_(pa.list_(pa.float16(), topk_width))
    return pa.schema(
        [
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("rescore_config_sha256", pa.string(), nullable=False),
            pa.field("source_dataset_index_sha256", pa.string(), nullable=False),
            pa.field("source_experiment_sha256", pa.string(), nullable=False),
            pa.field("source_parquet_sha256", pa.string(), nullable=False),
            pa.field("source_generation_config_sha256", pa.string(), nullable=False),
            pa.field("trace_id", pa.string(), nullable=False),
            pa.field("direction", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("source_dataset", pa.string(), nullable=False),
            pa.field("source_dataset_sha256", pa.string(), nullable=False),
            pa.field("source_uid", pa.string(), nullable=False),
            pa.field("question_sha256", pa.string(), nullable=False),
            pa.field("prompt_index", pa.int64(), nullable=False),
            pa.field("sample_index", pa.int8(), nullable=False),
            pa.field("question_text", pa.large_string(), nullable=False),
            pa.field("gold_answer", pa.large_string(), nullable=False),
            pa.field("strict_grade", pa.float32(), nullable=False),
            pa.field("strict_correct", pa.bool_(), nullable=False),
            pa.field("strict_prediction", pa.large_string(), nullable=False),
            pa.field("response_text", pa.large_string(), nullable=False),
            pa.field("vllm_response_text", pa.large_string(), nullable=False),
            pa.field("prompt_token_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("response_token_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("input_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("response_mask", pa.list_(pa.int8()), nullable=False),
            pa.field("teacher_topk_token_ids", ids, nullable=False),
            pa.field("teacher_topk_logprobs", logprobs, nullable=False),
            pa.field("teacher_sampled_token_logprobs", pa.list_(pa.float16()), nullable=False),
            pa.field("teacher_topk_rank_order", pa.string(), nullable=False),
            pa.field("prompt_length", pa.int32(), nullable=False),
            pa.field("response_length", pa.int32(), nullable=False),
            pa.field("shard_id", pa.int32(), nullable=False),
            pa.field("row_within_shard", pa.int32(), nullable=False),
            pa.field("source_target_engine", pa.string(), nullable=False),
            pa.field("source_teacher_model", pa.string(), nullable=False),
            pa.field("source_teacher_revision", pa.string()),
            pa.field("source_teacher_content_sha256", pa.string()),
            pa.field("source_sampling_parameters_json", pa.large_string(), nullable=False),
            pa.field("source_environment_versions_json", pa.large_string(), nullable=False),
            pa.field("target_engine", pa.string(), nullable=False),
            pa.field("target_model_identity_sha256", pa.string(), nullable=False),
            pa.field("target_dtype", pa.string(), nullable=False),
            pa.field("target_attention_implementation", pa.string(), nullable=False),
            pa.field("target_final_logit_softcapping", pa.float32(), nullable=False),
            pa.field("rescoring_timestamp", pa.string(), nullable=False),
        ],
        metadata={
            b"schema_version": OVERLAY_SCHEMA_VERSION.encode(),
            b"topk_width": str(topk_width).encode(),
            b"causal_alignment": b"target token at input index i uses hidden/logits at i-1",
        },
    )


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain an object: {path}")
    return value


def _resolve_under(root: Path, value: Any, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{description} must be relative to the source dataset root")
    resolved = (root / candidate).resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{description} escapes the source dataset root")
    return resolved


def _require_sha256(value: Any, description: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{description} must be a 64-character SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{description} must be hexadecimal") from error
    return value.lower()


def load_source_index(index_path: Path, trace_schema: Any) -> tuple[Path, dict[str, Any]]:
    index_path = index_path.resolve(strict=True)
    source_root = index_path.parent
    index = _load_json(index_path, "source dataset index")
    claimed = index.get("dataset_index_sha256")
    unhashed = dict(index)
    unhashed.pop("dataset_index_sha256", None)
    actual = trace_schema.hash_json(unhashed)
    if claimed != actual:
        raise ValueError(f"source dataset index self-hash mismatch: {claimed} != {actual}")
    if index.get("schema_version") != trace_schema.SCHEMA_VERSION:
        raise ValueError(f"unsupported source schema: {index.get('schema_version')!r}")
    if index.get("topk_width") != TOPK_WIDTH:
        raise ValueError("source bundle is not top-128")
    if index.get("decode_check_performed") is not True:
        raise ValueError("source bundle lacks a completed decode validation")
    if not isinstance(index.get("splits"), dict) or set(index["splits"]) != {"train", "validation"}:
        raise ValueError("source index must contain complete train and validation splits")
    return source_root, index


def load_source_manifest(
    source_root: Path,
    source_entry: Mapping[str, Any],
    trace_schema: Any,
) -> tuple[Path, dict[str, Any], str]:
    path = _resolve_under(source_root, source_entry["manifest_path"], "source shard manifest")
    manifest = _load_json(path, "source shard manifest")
    expected = {
        "parquet_sha256": source_entry["sha256"],
        "shard_id": source_entry["shard_id"],
        "row_count": source_entry["rows"],
        "parquet_file": Path(str(source_entry["path"])).name,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"source shard manifest {key} mismatch")
    _require_sha256(manifest.get("trace_ids_sha256"), "source shard manifest trace_ids_sha256")
    return path, manifest, trace_schema.sha256_file(path)


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("text_config", config)
    if not isinstance(value, Mapping):
        raise ValueError("model text_config is malformed")
    return value


def build_run_config(
    *,
    args: argparse.Namespace,
    source_index: Mapping[str, Any],
    model_identity: Any,
    trace_schema: Any,
) -> dict[str, Any]:
    model_config = _load_json(Path(args.model_path) / "config.json", "model config")
    text_config = _text_config(model_config)
    if model_config.get("model_type") != "gemma4":
        raise ValueError(f"expected a Gemma 4 model, got {model_config.get('model_type')!r}")
    softcap = text_config.get("final_logit_softcapping")
    if not isinstance(softcap, int | float) or float(softcap) <= 0:
        raise ValueError(f"Gemma 4 final_logit_softcapping must be positive, got {softcap!r}")
    source_teacher_identity = source_index.get("teacher", {}).get("model_identity_sha256")
    if source_teacher_identity != model_identity.model_identity_sha256:
        raise ValueError(
            "rescoring model is not the exact source teacher: "
            f"{model_identity.model_identity_sha256} != {source_teacher_identity}"
        )
    source_code_sha256 = trace_schema.sha256_file(Path(__file__))
    semantic = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "source_dataset_index_sha256": source_index["dataset_index_sha256"],
        "source_experiment_sha256": source_index["experiment_sha256"],
        "source_direction": source_index["direction"],
        "source_teacher": source_index["teacher"],
        "target_model_identity": model_identity.manifest(),
        "loader": "verl class resolver + unsharded transformers.from_pretrained",
        "target_engine": "hf_bf16_sdpa_full_forward",
        "dtype": args.dtype,
        "attention_implementation": args.attention_implementation,
        "topk_width": args.topk_width,
        "vocab_size": int(source_index["tokenizer"]["vocab_size"]),
        "lm_head_chunk_tokens": args.lm_head_chunk_tokens,
        "max_sequence_tokens": args.max_sequence_tokens,
        "final_logit_softcapping": float(softcap),
        "causal_alignment": "response token at input index i is scored by hidden/logits at i-1",
        "normalization": "full-vocabulary logsumexp in FP32 after BF16 LM head and softcap",
        "storage": {"token_ids": "int32", "logprobs": "float16"},
        "rescorer_source_sha256": source_code_sha256,
        "environment_versions": {
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "pyarrow": _package_version("pyarrow"),
            "verl": _package_version("verl"),
        },
    }
    return {
        "manifest_version": OVERLAY_MANIFEST_VERSION,
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "rescore_config_sha256": trace_schema.hash_json(semantic),
        "semantic_config": semantic,
        "runtime": {
            "source_dataset_index": str(Path(args.source_dataset_index).resolve()),
            "model_path": str(Path(args.model_path).resolve()),
        },
        "created_at": utc_now(),
    }


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def ensure_run_config(output_root: Path, run_config: Mapping[str, Any], trace_schema: Any) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "rescore_config.json"
    with file_lock(output_root / ".rescore_config.lock"):
        if path.exists():
            existing = _load_json(path, "existing rescore config")
            if existing.get("rescore_config_sha256") != run_config["rescore_config_sha256"]:
                raise ValueError("output root already belongs to a different rescoring configuration")
            if existing.get("semantic_config") != run_config["semantic_config"]:
                raise ValueError("rescore configuration hash collision/non-canonical configuration")
            return existing
        if any(output_root.rglob("*.parquet")) or (output_root / "dataset_index.json").exists():
            raise ValueError("refusing to adopt outputs without rescore_config.json")
        trace_schema.atomic_write_json(path, dict(run_config))
        return dict(run_config)


def require_disjoint_source_and_output(source_root: Path, output_root: Path) -> None:
    source_root = source_root.resolve(strict=True)
    output_root = output_root.resolve()
    if output_root == source_root or output_root.is_relative_to(source_root) or source_root.is_relative_to(output_root):
        raise ValueError(
            f"output root must be disjoint from the immutable source bundle; source={source_root}, output={output_root}"
        )


def write_parity_receipt(
    output_root: Path,
    *,
    run_config: Mapping[str, Any],
    source_index: Mapping[str, Any],
    checked_rows: int,
    parity_max_response_tokens: int,
    trace_schema: Any,
) -> dict[str, Any]:
    receipt = {
        "schema_version": 1,
        "passed_at": utc_now(),
        "rescore_config_sha256": run_config["rescore_config_sha256"],
        "source_dataset_index_sha256": source_index["dataset_index_sha256"],
        "target_model_identity_sha256": run_config["semantic_config"]["target_model_identity"]["model_identity_sha256"],
        "target_engine": run_config["semantic_config"]["target_engine"],
        "checked_rows": checked_rows,
        "eligibility_response_token_cap": parity_max_response_tokens,
        "comparison": "exact top-k IDs, FP16 top-k logprobs, and FP16 sampled-token logprobs",
    }
    receipt["parity_receipt_sha256"] = trace_schema.hash_json(receipt)
    trace_schema.atomic_write_json(output_root / PARITY_RECEIPT_NAME, receipt)
    return receipt


def validate_parity_receipt(
    output_root: Path,
    *,
    run_config: Mapping[str, Any],
    source_index: Mapping[str, Any],
    trace_schema: Any,
) -> dict[str, Any]:
    path = output_root / PARITY_RECEIPT_NAME
    receipt = _load_json(path, "parity receipt")
    claimed = _require_sha256(receipt.get("parity_receipt_sha256"), "parity receipt self-hash")
    unhashed = dict(receipt)
    del unhashed["parity_receipt_sha256"]
    if trace_schema.hash_json(unhashed) != claimed:
        raise ValueError("parity receipt self-hash mismatch")
    expected = {
        "rescore_config_sha256": run_config["rescore_config_sha256"],
        "source_dataset_index_sha256": source_index["dataset_index_sha256"],
        "target_model_identity_sha256": run_config["semantic_config"]["target_model_identity"]["model_identity_sha256"],
        "target_engine": run_config["semantic_config"]["target_engine"],
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"parity receipt {key} mismatch")
    if not isinstance(receipt.get("checked_rows"), int) or int(receipt["checked_rows"]) <= 0:
        raise ValueError("parity receipt checked_rows must be positive")
    if (
        not isinstance(receipt.get("eligibility_response_token_cap"), int)
        or int(receipt["eligibility_response_token_cap"]) <= 0
    ):
        raise ValueError("parity receipt eligibility_response_token_cap must be positive")
    return receipt


def shifted_prediction_positions(input_ids: Sequence[int], response_mask: Sequence[int]) -> list[int]:
    if len(input_ids) != len(response_mask):
        raise ValueError("input_ids/response_mask length mismatch")
    if len(input_ids) < 2:
        raise ValueError("a trace must contain at least two tokens")
    if any(int(value) not in (0, 1) for value in response_mask):
        raise ValueError("response_mask must be binary")
    response_positions = [index for index, value in enumerate(response_mask) if int(value) == 1]
    if not response_positions:
        raise ValueError("a trace must contain at least one response token")
    if response_positions[0] == 0:
        raise ValueError("cannot score a response token without a preceding context token")
    if response_positions != list(range(response_positions[0], len(response_mask))):
        raise ValueError("response_mask must be an exact zero-prefix/one-suffix")
    return [index - 1 for index in response_positions]


def _dtype(name: str):
    import torch

    values = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype {name!r}") from error


def load_training_model(args: argparse.Namespace):
    import torch
    from transformers import AutoConfig

    from verl.utils.model import get_hf_auto_model_class

    config = AutoConfig.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation=args.attention_implementation,
    )
    auto_class = get_hf_auto_model_class(config)
    model = auto_class.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=_dtype(args.dtype),
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        raise ValueError(f"unsupported training model class: {type(model).__name__}")
    model.eval()
    model.requires_grad_(False)
    model.to(torch.device(args.device))
    return model, type(model).__name__


def score_topk(
    *,
    model: Any,
    input_ids: Sequence[int],
    response_mask: Sequence[int],
    topk_width: int,
    chunk_tokens: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full-vocabulary-normalized top-k at response prediction positions."""

    import torch

    positions = shifted_prediction_positions(input_ids, response_mask)
    response_token_ids = [int(input_ids[position + 1]) for position in positions]
    token_tensor = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(token_tensor, dtype=torch.bool)
    position_ids = torch.arange(token_tensor.shape[1], device=device, dtype=torch.long).unsqueeze(0)
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ),
    ):
        outputs = model.model(
            input_ids=token_tensor,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        if hidden.shape[:2] != token_tensor.shape:
            raise RuntimeError(f"unexpected hidden shape {tuple(hidden.shape)}")
        active_hidden = hidden.index_select(1, torch.tensor(positions, dtype=torch.long, device=device)).squeeze(0)
        text_config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        softcap = getattr(text_config, "final_logit_softcapping", None)
        if softcap is None or float(softcap) <= 0:
            raise RuntimeError(f"invalid final_logit_softcapping: {softcap!r}")
        ids_parts: list[torch.Tensor] = []
        logprob_parts: list[torch.Tensor] = []
        sampled_logprob_parts: list[torch.Tensor] = []
        for start in range(0, active_hidden.shape[0], chunk_tokens):
            chunk = active_hidden[start : start + chunk_tokens]
            # Training-shaped operation order: LM head + softcap in the model
            # dtype, then FP32 top-k and full-vocabulary logsumexp.
            logits = model.lm_head(chunk)
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
            logits = logits.float()
            values, ids = torch.topk(logits, k=topk_width, dim=-1, largest=True, sorted=True)
            denominator = torch.logsumexp(logits, dim=-1, keepdim=True)
            logprobs = values - denominator
            sampled_ids = torch.tensor(
                response_token_ids[start : start + chunk_tokens], dtype=torch.long, device=device
            ).unsqueeze(1)
            sampled_logprobs = torch.gather(logits, dim=-1, index=sampled_ids).squeeze(1) - denominator.squeeze(1)
            ids_parts.append(ids.to(dtype=torch.int32, device="cpu"))
            logprob_parts.append(logprobs.to(dtype=torch.float16, device="cpu"))
            sampled_logprob_parts.append(sampled_logprobs.to(dtype=torch.float16, device="cpu"))
        topk_ids = torch.cat(ids_parts, dim=0).numpy()
        topk_logprobs = torch.cat(logprob_parts, dim=0).numpy()
        sampled_logprobs = torch.cat(sampled_logprob_parts, dim=0).numpy()
    if topk_ids.shape != (len(positions), topk_width) or topk_logprobs.shape != topk_ids.shape:
        raise RuntimeError("top-k output shape mismatch")
    return topk_ids, topk_logprobs, sampled_logprobs


def score_topk_native_forward(
    *,
    model: Any,
    input_ids: Sequence[int],
    response_mask: Sequence[int],
    topk_width: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference path through native HF forward for the pre-run parity gate."""

    import torch

    positions = shifted_prediction_positions(input_ids, response_mask)
    response_token_ids = [int(input_ids[position + 1]) for position in positions]
    token_tensor = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(token_tensor, dtype=torch.bool)
    position_ids = torch.arange(token_tensor.shape[1], device=device, dtype=torch.long).unsqueeze(0)
    position_tensor = torch.tensor(positions, dtype=torch.long, device=device)
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(device).type == "cuda",
        ),
    ):
        outputs = model(
            input_ids=token_tensor,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=position_tensor,
            return_dict=True,
        )
        logits = outputs.logits.squeeze(0).float()
        if logits.shape[0] != len(positions):
            raise RuntimeError(f"native forward returned unexpected logits shape {tuple(logits.shape)}")
        denominator = torch.logsumexp(logits, dim=-1, keepdim=True)
        values, ids = torch.topk(logits, k=topk_width, dim=-1, largest=True, sorted=True)
        logprobs = values - denominator
        sampled_ids = torch.tensor(response_token_ids, dtype=torch.long, device=device).unsqueeze(1)
        sampled_logprobs = torch.gather(logits, dim=-1, index=sampled_ids).squeeze(1) - denominator.squeeze(1)
    return (
        ids.to(dtype=torch.int32, device="cpu").numpy(),
        logprobs.to(dtype=torch.float16, device="cpu").numpy(),
        sampled_logprobs.to(dtype=torch.float16, device="cpu").numpy(),
    )


def validate_stored_targets(
    ids: np.ndarray,
    logprobs: np.ndarray,
    *,
    vocab_size: int,
    response_token_ids: Sequence[int] | None = None,
    sampled_logprobs: np.ndarray | None = None,
) -> np.ndarray:
    if ids.dtype != np.int32 or logprobs.dtype != np.float16:
        raise ValueError(f"target storage dtypes must be int32/float16, got {ids.dtype}/{logprobs.dtype}")
    if ids.ndim != 2 or ids.shape[1] != TOPK_WIDTH or logprobs.shape != ids.shape:
        raise ValueError(f"invalid target shapes: {ids.shape}, {logprobs.shape}")
    ids64 = ids.astype(np.int64, copy=False)
    if ids64.size and (ids64.min() < 0 or ids64.max() >= vocab_size):
        raise ValueError("top-k token ID outside vocabulary")
    if any(len(set(row.tolist())) != TOPK_WIDTH for row in ids64):
        raise ValueError("duplicate token ID in top-k row")
    lp32 = logprobs.astype(np.float32)
    if not np.isfinite(lp32).all() or (lp32 > 5e-4).any():
        raise ValueError("invalid top-k log probabilities")
    if (lp32[:, 1:] > lp32[:, :-1] + 5e-4).any():
        raise ValueError("stored top-k is not rank sorted")
    masses = np.exp(lp32).sum(axis=1, dtype=np.float64)
    if (masses > 1.0 + FP16_MASS_TOLERANCE).any():
        raise ValueError(f"stored top-k probability mass exceeds one: {masses.max()}")
    if (response_token_ids is None) != (sampled_logprobs is None):
        raise ValueError("response_token_ids and sampled_logprobs must be supplied together")
    if sampled_logprobs is not None:
        sampled = np.asarray(sampled_logprobs)
        if sampled.dtype != np.float16 or sampled.shape != (ids.shape[0],):
            raise ValueError("sampled-token log probabilities must be FP16 with one value per response token")
        sampled32 = sampled.astype(np.float32)
        if not np.isfinite(sampled32).all() or (sampled32 > 5e-4).any():
            raise ValueError("invalid sampled-token log probabilities")
        for position, token_id in enumerate(response_token_ids):
            matches = np.nonzero(ids[position] == int(token_id))[0]
            if matches.size and abs(float(lp32[position, matches[0]]) - float(sampled32[position])) > 5e-4:
                raise ValueError("sampled-token log probability disagrees with ranked target")
    return masses


def output_paths(output_root: Path, split: str, shard_id: int) -> tuple[Path, Path]:
    directory = output_root / split
    parquet = directory / f"targets-{split}-{shard_id:06d}.parquet"
    return parquet, parquet.with_suffix(".manifest.json")


class ShardStats:
    def __init__(self):
        self.rows = 0
        self.response_tokens = 0
        self.top1_agree = 0
        self.topk_overlap = 0
        self.mass_sum = 0.0
        self.mass_min = float("inf")
        self.mass_max = float("-inf")

    def update(self, masses: np.ndarray, new_ids: np.ndarray, old_ids: Any) -> None:
        old = np.asarray(old_ids, dtype=np.int64)
        if old.shape != new_ids.shape:
            raise ValueError(f"source/new top-k shape mismatch: {old.shape} != {new_ids.shape}")
        self.rows += 1
        self.response_tokens += int(new_ids.shape[0])
        self.top1_agree += int((old[:, 0] == new_ids[:, 0]).sum())
        for lhs, rhs in zip(old, new_ids, strict=True):
            self.topk_overlap += len(set(lhs.tolist()).intersection(rhs.tolist()))
        self.mass_sum += float(masses.sum())
        if masses.size:
            self.mass_min = min(self.mass_min, float(masses.min()))
            self.mass_max = max(self.mass_max, float(masses.max()))

    def as_dict(self) -> dict[str, Any]:
        tokens = self.response_tokens
        return {
            "row_count": self.rows,
            "response_token_count": tokens,
            "training_vs_vllm_top1_agreement": self.top1_agree / tokens if tokens else None,
            "training_vs_vllm_top128_overlap_fraction": (self.topk_overlap / (tokens * TOPK_WIDTH) if tokens else None),
            "training_top128_mass_mean": self.mass_sum / tokens if tokens else None,
            "training_top128_mass_min": self.mass_min if tokens else None,
            "training_top128_mass_max": self.mass_max if tokens else None,
        }


def make_overlay_record(
    source: Mapping[str, Any],
    *,
    topk_ids: np.ndarray,
    topk_logprobs: np.ndarray,
    sampled_logprobs: np.ndarray,
    source_index: Mapping[str, Any],
    source_parquet_sha256: str,
    run_config: Mapping[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    semantic = run_config["semantic_config"]
    return {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "rescore_config_sha256": run_config["rescore_config_sha256"],
        "source_dataset_index_sha256": source_index["dataset_index_sha256"],
        "source_experiment_sha256": source_index["experiment_sha256"],
        "source_parquet_sha256": source_parquet_sha256,
        "source_generation_config_sha256": source["generation_config_sha256"],
        "trace_id": source["trace_id"],
        "direction": source["direction"],
        "split": source["split"],
        "source_dataset": source["source_dataset"],
        "source_dataset_sha256": source["source_dataset_sha256"],
        "source_uid": source["source_uid"],
        "question_sha256": source["question_sha256"],
        "prompt_index": source["prompt_index"],
        "sample_index": source["sample_index"],
        "question_text": source["question_text"],
        "gold_answer": source["gold_answer"],
        "strict_grade": source["strict_grade"],
        "strict_correct": source["strict_correct"],
        "strict_prediction": source["strict_prediction"],
        "response_text": source["response_text"],
        "vllm_response_text": source["vllm_response_text"],
        "prompt_token_ids": source["prompt_token_ids"],
        "response_token_ids": source["response_token_ids"],
        "input_ids": source["input_ids"],
        "response_mask": source["response_mask"],
        "teacher_topk_token_ids": topk_ids.tolist(),
        "teacher_topk_logprobs": topk_logprobs.tolist(),
        "teacher_sampled_token_logprobs": sampled_logprobs.tolist(),
        "teacher_topk_rank_order": f"1..{TOPK_WIDTH}",
        "prompt_length": source["prompt_length"],
        "response_length": source["response_length"],
        "shard_id": source["shard_id"],
        "row_within_shard": source["row_within_shard"],
        "source_target_engine": "vllm",
        "source_teacher_model": source["teacher_model"],
        "source_teacher_revision": source["teacher_revision"],
        "source_teacher_content_sha256": source["teacher_content_sha256"],
        "source_sampling_parameters_json": source["sampling_parameters_json"],
        "source_environment_versions_json": source["environment_versions_json"],
        "target_engine": semantic["target_engine"],
        "target_model_identity_sha256": semantic["target_model_identity"]["model_identity_sha256"],
        "target_dtype": semantic["dtype"],
        "target_attention_implementation": semantic["attention_implementation"],
        "target_final_logit_softcapping": semantic["final_logit_softcapping"],
        "rescoring_timestamp": timestamp,
    }


def _flush_records(writer: pq.ParquetWriter, records: list[dict[str, Any]]) -> None:
    if records:
        writer.write_table(pa.Table.from_pylist(records, schema=overlay_schema()), row_group_size=len(records))
        records.clear()


def validate_source_trace_id_binding(
    trace_ids: Sequence[str],
    *,
    source_manifest: Mapping[str, Any],
    trace_schema: Any,
) -> str:
    if len(set(trace_ids)) != len(trace_ids):
        raise ValueError("overlay contains duplicate trace IDs")
    actual = trace_schema.hash_json(sorted(trace_ids))
    expected = _require_sha256(
        source_manifest.get("trace_ids_sha256"),
        "source shard manifest trace_ids_sha256",
    )
    if actual != expected:
        raise ValueError(f"overlay trace-ID set does not match source manifest: {actual} != {expected}")
    return actual


def validate_output_shard(
    parquet_path: Path,
    manifest_path: Path,
    *,
    source_parquet_path: Path,
    source_entry: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    source_manifest_sha256: str,
    run_config: Mapping[str, Any],
    trace_schema: Any,
) -> dict[str, Any]:
    if not parquet_path.is_file() or not manifest_path.is_file():
        raise ValueError("missing output parquet/manifest pair")
    if trace_schema.sha256_file(source_parquet_path) != source_entry["sha256"]:
        raise ValueError("source parquet SHA mismatch")
    manifest = _load_json(manifest_path, "overlay manifest")
    expected = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "rescore_config_sha256": run_config["rescore_config_sha256"],
        "source_parquet_sha256": source_entry["sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        "source_trace_ids_sha256": source_manifest["trace_ids_sha256"],
        "shard_id": source_entry["shard_id"],
        "row_count": source_entry["rows"],
        "parquet_file": parquet_path.name,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"manifest {key} mismatch")
    if not isinstance(manifest.get("created_at"), str) or not manifest["created_at"]:
        raise ValueError("manifest created_at must be a non-empty string")
    if manifest.get("parquet_sha256") != trace_schema.sha256_file(parquet_path):
        raise ValueError("output parquet SHA mismatch")
    parquet_file = pq.ParquetFile(parquet_path)
    if not parquet_file.schema_arrow.equals(overlay_schema(), check_metadata=False):
        raise ValueError("output parquet schema mismatch")
    source_file = pq.ParquetFile(source_parquet_path)
    ordered_trace_ids: list[str] = []
    response_tokens = 0
    row_index = 0
    vocab_size = int(run_config["semantic_config"]["vocab_size"])
    semantic = run_config["semantic_config"]
    expected_constants = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "rescore_config_sha256": run_config["rescore_config_sha256"],
        "source_dataset_index_sha256": semantic["source_dataset_index_sha256"],
        "source_experiment_sha256": semantic["source_experiment_sha256"],
        "source_parquet_sha256": source_entry["sha256"],
        "source_target_engine": "vllm",
        "target_engine": semantic["target_engine"],
        "target_model_identity_sha256": semantic["target_model_identity"]["model_identity_sha256"],
        "target_dtype": semantic["dtype"],
        "target_attention_implementation": semantic["attention_implementation"],
        "target_final_logit_softcapping": np.float32(semantic["final_logit_softcapping"]).item(),
        "teacher_topk_rank_order": f"1..{TOPK_WIDTH}",
        "rescoring_timestamp": manifest["created_at"],
    }

    def iter_rows(file: pq.ParquetFile, *, columns: Sequence[str] | None = None):
        # One long response carries two [response_len, 128] target tensors.
        # Converting a multi-row target batch to Python can transiently consume
        # several GiB, so resume/finalize validation stays deliberately
        # row-bounded even when an operator chose larger Parquet row groups.
        for batch in file.iter_batches(batch_size=1, columns=columns):
            yield from batch.to_pylist()

    missing = object()
    source_columns = list(dict.fromkeys(COPIED_SOURCE_FIELDS.values()))
    for source, row in zip_longest(
        iter_rows(source_file, columns=source_columns),
        iter_rows(parquet_file),
        fillvalue=missing,
    ):
        if source is missing or row is missing:
            raise ValueError("source/output row count mismatch")
        if int(row["row_within_shard"]) != row_index:
            raise ValueError("output row order is not contiguous")
        for output_field, source_field in COPIED_SOURCE_FIELDS.items():
            if row[output_field] != source[source_field]:
                raise ValueError(
                    f"output copied field {output_field} does not match source field {source_field} at row {row_index}"
                )
        for field, expected_value in expected_constants.items():
            if row[field] != expected_value:
                raise ValueError(f"output constant {field} mismatch at row {row_index}")
        if len(row["response_token_ids"]) != int(row["response_length"]):
            raise ValueError("response_token_ids length does not match response_length")
        ids = np.asarray(row["teacher_topk_token_ids"], dtype=np.int32)
        lp = np.asarray(row["teacher_topk_logprobs"], dtype=np.float16)
        sampled_lp = np.asarray(row["teacher_sampled_token_logprobs"], dtype=np.float16)
        validate_stored_targets(
            ids,
            lp,
            vocab_size=vocab_size,
            response_token_ids=row["response_token_ids"],
            sampled_logprobs=sampled_lp,
        )
        if ids.shape[0] != int(row["response_length"]):
            raise ValueError("output target length does not match response_length")
        ordered_trace_ids.append(row["trace_id"])
        response_tokens += ids.shape[0]
        row_index += 1
    if row_index != int(source_entry["rows"]):
        raise ValueError("output row count mismatch")
    if manifest.get("ordered_trace_ids_sha256") != trace_schema.hash_json(ordered_trace_ids):
        raise ValueError("ordered trace-ID hash mismatch")
    source_trace_ids_sha256 = validate_source_trace_id_binding(
        ordered_trace_ids,
        source_manifest=source_manifest,
        trace_schema=trace_schema,
    )
    if manifest.get("source_trace_ids_sha256") != source_trace_ids_sha256:
        raise ValueError("manifest source trace-ID hash mismatch")
    if manifest.get("stats", {}).get("response_token_count") != response_tokens:
        raise ValueError("output response-token count mismatch")
    return manifest


def score_shard(
    *,
    model: Any,
    model_class_name: str,
    source_root: Path,
    source_index: Mapping[str, Any],
    split: str,
    source_entry: Mapping[str, Any],
    output_root: Path,
    run_config: Mapping[str, Any],
    args: argparse.Namespace,
    trace_schema: Any,
) -> None:
    shard_id = int(source_entry["shard_id"])
    source_path = _resolve_under(source_root, source_entry["path"], "source shard")
    if trace_schema.sha256_file(source_path) != source_entry["sha256"]:
        raise ValueError(f"source shard SHA mismatch: {source_path}")
    _source_manifest_path, source_manifest, source_manifest_sha256 = load_source_manifest(
        source_root,
        source_entry,
        trace_schema,
    )
    source_run_config_path = _resolve_under(
        source_root, source_index["splits"][split]["run_config_path"], "source run config"
    )
    source_run_config = _load_json(source_run_config_path, "source run config")
    parquet_path, manifest_path = output_paths(output_root, split, shard_id)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(parquet_path.with_name(f".{parquet_path.name}.lock")):
        try:
            validate_output_shard(
                parquet_path,
                manifest_path,
                source_parquet_path=source_path,
                source_entry=source_entry,
                source_manifest=source_manifest,
                source_manifest_sha256=source_manifest_sha256,
                run_config=run_config,
                trace_schema=trace_schema,
            )
            print(f"[resume] valid {parquet_path}", flush=True)
            return
        except (OSError, KeyError, TypeError, ValueError):
            pass

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{parquet_path.name}.", suffix=".tmp.parquet", dir=parquet_path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        stats = ShardStats()
        trace_ids: list[str] = []
        records: list[dict[str, Any]] = []
        timestamp = utc_now()
        vocab_size = int(source_index["tokenizer"]["vocab_size"])
        try:
            writer = pq.ParquetWriter(
                temporary,
                overlay_schema(),
                compression="zstd",
                compression_level=3,
                write_statistics=True,
                data_page_version="2.0",
            )
            try:
                source_file = pq.ParquetFile(source_path)
                row_index = 0
                for row_group_index in range(source_file.metadata.num_row_groups):
                    # Read the complete source row so the repository's strict
                    # validator can re-check every original vLLM field before
                    # this derived target is accepted.
                    table = source_file.read_row_group(row_group_index)
                    for batch in table.to_batches(max_chunksize=1):
                        source = batch.to_pylist()[0]
                        # Reuse the existing source validator, including the original
                        # vLLM targets/provenance, before deriving a replacement target.
                        trace_schema.validate_trace_record(
                            source,
                            decoder=None,
                            expected_config_sha256=source_run_config["generation_config_sha256"],
                            expected_direction=source_index["direction"],
                            expected_split=split,
                            expected_shard_id=shard_id,
                            expected_row_within_shard=row_index,
                            expected_semantic_config=source_run_config["semantic_config"],
                            max_prompt_tokens=int(
                                source_run_config["semantic_config"]["sampling"]["max_prompt_tokens"]
                            ),
                            max_response_tokens=int(
                                source_run_config["semantic_config"]["sampling"]["max_response_tokens"]
                            ),
                        )
                        if len(source["input_ids"]) > args.max_sequence_tokens:
                            raise ValueError(
                                f"trace {source['trace_id']} has {len(source['input_ids'])} tokens, "
                                f"above {args.max_sequence_tokens}"
                            )
                        topk_ids, topk_logprobs, sampled_logprobs = score_topk(
                            model=model,
                            input_ids=source["input_ids"],
                            response_mask=source["response_mask"],
                            topk_width=args.topk_width,
                            chunk_tokens=args.lm_head_chunk_tokens,
                            device=args.device,
                        )
                        masses = validate_stored_targets(
                            topk_ids,
                            topk_logprobs,
                            vocab_size=vocab_size,
                            response_token_ids=source["response_token_ids"],
                            sampled_logprobs=sampled_logprobs,
                        )
                        stats.update(masses, topk_ids, source["teacher_topk_token_ids"])
                        trace_ids.append(source["trace_id"])
                        records.append(
                            make_overlay_record(
                                source,
                                topk_ids=topk_ids,
                                topk_logprobs=topk_logprobs,
                                sampled_logprobs=sampled_logprobs,
                                source_index=source_index,
                                source_parquet_sha256=source_entry["sha256"],
                                run_config=run_config,
                                timestamp=timestamp,
                            )
                        )
                        if len(records) >= args.output_row_group_rows:
                            _flush_records(writer, records)
                        row_index += 1
                _flush_records(writer, records)
            finally:
                writer.close()
            if stats.rows != int(source_entry["rows"]):
                raise ValueError(f"source/output row count mismatch: {stats.rows} != {source_entry['rows']}")
            source_trace_ids_sha256 = validate_source_trace_id_binding(
                trace_ids,
                source_manifest=source_manifest,
                trace_schema=trace_schema,
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            output_sha = trace_schema.sha256_file(temporary)
            output_metadata = pq.ParquetFile(temporary).metadata
            manifest = {
                "manifest_version": OVERLAY_MANIFEST_VERSION,
                "schema_version": OVERLAY_SCHEMA_VERSION,
                "rescore_config_sha256": run_config["rescore_config_sha256"],
                "source_dataset_index_sha256": source_index["dataset_index_sha256"],
                "source_experiment_sha256": source_index["experiment_sha256"],
                "source_parquet_file": source_entry["path"],
                "source_parquet_sha256": source_entry["sha256"],
                "source_manifest_sha256": source_manifest_sha256,
                "source_trace_ids_sha256": source_trace_ids_sha256,
                "split": split,
                "shard_id": shard_id,
                "row_count": stats.rows,
                "ordered_trace_ids_sha256": trace_schema.hash_json(trace_ids),
                "parquet_file": parquet_path.name,
                "parquet_sha256": output_sha,
                "parquet_size_bytes": temporary.stat().st_size,
                "parquet_row_groups": output_metadata.num_row_groups,
                "stats": stats.as_dict(),
                "model_class": model_class_name,
                "created_at": timestamp,
            }
            os.replace(temporary, parquet_path)
            trace_schema.atomic_write_json(manifest_path, manifest)
            validate_output_shard(
                parquet_path,
                manifest_path,
                source_parquet_path=source_path,
                source_entry=source_entry,
                source_manifest=source_manifest,
                source_manifest_sha256=source_manifest_sha256,
                run_config=run_config,
                trace_schema=trace_schema,
            )
            print(f"[saved] {parquet_path} rows={stats.rows} response_tokens={stats.response_tokens}", flush=True)
        finally:
            temporary.unlink(missing_ok=True)


def selected_shards(
    source_index: Mapping[str, Any], args: argparse.Namespace
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    emitted = 0
    splits = ("train", "validation") if args.split == "all" else (args.split,)
    for split in splits:
        for entry in source_index["splits"][split]["shards"]:
            shard_id = int(entry["shard_id"])
            if shard_id % args.num_workers != args.worker_id:
                continue
            if args.max_shards > 0 and emitted >= args.max_shards:
                return
            yield split, entry
            emitted += 1


def run_native_parity_gate(
    *,
    model: Any,
    source_root: Path,
    source_index: Mapping[str, Any],
    args: argparse.Namespace,
    trace_schema: Any,
) -> int:
    checked = 0
    for split, source_entry in selected_shards(source_index, args):
        source_path = _resolve_under(source_root, source_entry["path"], "source shard")
        if trace_schema.sha256_file(source_path) != source_entry["sha256"]:
            raise ValueError(f"source shard SHA mismatch: {source_path}")
        source_file = pq.ParquetFile(source_path)
        for batch in source_file.iter_batches(batch_size=1):
            row = batch.to_pylist()[0]
            if int(row["response_length"]) > args.parity_max_response_tokens:
                continue
            chunked = score_topk(
                model=model,
                input_ids=row["input_ids"],
                response_mask=row["response_mask"],
                topk_width=args.topk_width,
                chunk_tokens=args.lm_head_chunk_tokens,
                device=args.device,
            )
            native = score_topk_native_forward(
                model=model,
                input_ids=row["input_ids"],
                response_mask=row["response_mask"],
                topk_width=args.topk_width,
                device=args.device,
            )
            labels = ("top-k IDs", "FP16 top-k logprobs", "FP16 sampled-token logprobs")
            for label, actual, expected in zip(labels, chunked, native, strict=True):
                if not np.array_equal(actual, expected):
                    mismatch_count = int(np.count_nonzero(actual != expected))
                    if np.issubdtype(actual.dtype, np.floating):
                        max_abs = float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))
                    else:
                        max_abs = None
                    raise ValueError(
                        f"native parity failed for trace {row['trace_id']} ({label}): "
                        f"mismatches={mismatch_count}, max_abs={max_abs}"
                    )
            checked += 1
            print(f"[parity] {checked}/{args.parity_rows} trace={row['trace_id']} exact", flush=True)
            if checked >= args.parity_rows:
                return checked
    raise ValueError(
        f"native parity gate found only {checked} eligible rows; requested {args.parity_rows} "
        f"with response_length <= {args.parity_max_response_tokens}"
    )


def finalize(
    *,
    output_root: Path,
    source_root: Path,
    source_index: Mapping[str, Any],
    run_config: Mapping[str, Any],
    trace_schema: Any,
) -> dict[str, Any]:
    split_indexes: dict[str, Any] = {}
    total_rows = 0
    total_response_tokens = 0
    for split in ("train", "validation"):
        shards = []
        for source_entry in source_index["splits"][split]["shards"]:
            parquet_path, manifest_path = output_paths(output_root, split, int(source_entry["shard_id"]))
            source_path = _resolve_under(source_root, source_entry["path"], "source shard")
            _source_manifest_path, source_manifest, source_manifest_sha256 = load_source_manifest(
                source_root,
                source_entry,
                trace_schema,
            )
            manifest = validate_output_shard(
                parquet_path,
                manifest_path,
                source_parquet_path=source_path,
                source_entry=source_entry,
                source_manifest=source_manifest,
                source_manifest_sha256=source_manifest_sha256,
                run_config=run_config,
                trace_schema=trace_schema,
            )
            shards.append(
                {
                    "shard_id": manifest["shard_id"],
                    "path": parquet_path.relative_to(output_root).as_posix(),
                    "manifest_path": manifest_path.relative_to(output_root).as_posix(),
                    "sha256": manifest["parquet_sha256"],
                    "size_bytes": manifest["parquet_size_bytes"],
                    "rows": manifest["row_count"],
                    "response_tokens": manifest["stats"]["response_token_count"],
                    "source_parquet_sha256": manifest["source_parquet_sha256"],
                    "source_manifest_sha256": manifest["source_manifest_sha256"],
                    "source_trace_ids_sha256": manifest["source_trace_ids_sha256"],
                    "ordered_trace_ids_sha256": manifest["ordered_trace_ids_sha256"],
                }
            )
        rows = sum(item["rows"] for item in shards)
        tokens = sum(item["response_tokens"] for item in shards)
        if rows != int(source_index["splits"][split]["row_count"]):
            raise ValueError(f"{split} overlay is incomplete: {rows} rows")
        split_indexes[split] = {
            "row_count": rows,
            "response_token_count": tokens,
            "source_generation_config_sha256": source_index["splits"][split]["generation_config_sha256"],
            "shards": shards,
        }
        total_rows += rows
        total_response_tokens += tokens
    index = {
        "manifest_version": OVERLAY_MANIFEST_VERSION,
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "created_at": run_config["created_at"],
        "rescore_config_sha256": run_config["rescore_config_sha256"],
        "source_dataset_index_sha256": source_index["dataset_index_sha256"],
        "source_experiment_sha256": source_index["experiment_sha256"],
        "direction": source_index["direction"],
        "target_model_identity": run_config["semantic_config"]["target_model_identity"],
        "target_engine": run_config["semantic_config"]["target_engine"],
        "topk_width": TOPK_WIDTH,
        "total_rows": total_rows,
        "total_response_tokens": total_response_tokens,
        "splits": split_indexes,
    }
    index["dataset_index_sha256"] = trace_schema.hash_json(index)
    trace_schema.atomic_write_json(output_root / "dataset_index.json", index)
    return index


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "parity", "score", "finalize"))
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--source-dataset-index", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--topk-width", type=int, default=TOPK_WIDTH)
    parser.add_argument("--lm-head-chunk-tokens", type=int, default=16)
    parser.add_argument("--max-sequence-tokens", type=int, default=12288)
    parser.add_argument("--output-row-group-rows", type=int, default=1)
    parser.add_argument("--split", choices=("all", "train", "validation"), default="all")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-shards", type=int, default=-1)
    parser.add_argument("--parity-rows", type=int, default=8)
    parser.add_argument("--parity-max-response-tokens", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.topk_width != TOPK_WIDTH:
        raise ValueError(f"this experiment requires top-k width {TOPK_WIDTH}")
    if args.dtype != "bfloat16" or args.attention_implementation != "sdpa":
        raise ValueError("the registered production rescoring contract requires BF16 + SDPA")
    if args.lm_head_chunk_tokens <= 0 or args.output_row_group_rows <= 0:
        raise ValueError("chunk/row-group sizes must be positive")
    if args.parity_rows <= 0 or args.parity_max_response_tokens <= 0:
        raise ValueError("parity row/count limits must be positive")
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError("worker-id must lie in [0, num-workers)")
    repo_root = Path(args.repo_root).resolve(strict=True)
    trace_schema, model_identity_module = _repo_helpers(repo_root)
    source_root, source_index = load_source_index(Path(args.source_dataset_index), trace_schema)
    model_identity = model_identity_module.inspect_local_hf_model(args.model_path)
    run_config = build_run_config(
        args=args,
        source_index=source_index,
        model_identity=model_identity,
        trace_schema=trace_schema,
    )
    output_root = Path(args.output_root).resolve()
    require_disjoint_source_and_output(source_root, output_root)
    run_config = ensure_run_config(output_root, run_config, trace_schema)
    plan = {
        "mode": args.mode,
        "source_index_sha256": source_index["dataset_index_sha256"],
        "model_identity_sha256": model_identity.model_identity_sha256,
        "rescore_config_sha256": run_config["rescore_config_sha256"],
        "selected_shards": sum(1 for _ in selected_shards(source_index, args)),
        "output_root": str(output_root),
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.mode == "inspect":
        return 0
    if args.mode == "finalize":
        validate_parity_receipt(
            output_root,
            run_config=run_config,
            source_index=source_index,
            trace_schema=trace_schema,
        )
        index = finalize(
            output_root=output_root,
            source_root=source_root,
            source_index=source_index,
            run_config=run_config,
            trace_schema=trace_schema,
        )
        print(json.dumps(index, indent=2, sort_keys=True), flush=True)
        return 0
    if args.mode == "score":
        validate_parity_receipt(
            output_root,
            run_config=run_config,
            source_index=source_index,
            trace_schema=trace_schema,
        )
    model, model_class_name = load_training_model(args)
    if args.mode == "parity":
        checked_rows = run_native_parity_gate(
            model=model,
            source_root=source_root,
            source_index=source_index,
            args=args,
            trace_schema=trace_schema,
        )
        receipt = write_parity_receipt(
            output_root,
            run_config=run_config,
            source_index=source_index,
            checked_rows=checked_rows,
            parity_max_response_tokens=args.parity_max_response_tokens,
            trace_schema=trace_schema,
        )
        print(f"[parity] PASS model_class={model_class_name}", flush=True)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        return 0
    for split, source_entry in selected_shards(source_index, args):
        score_shard(
            model=model,
            model_class_name=model_class_name,
            source_root=source_root,
            source_index=source_index,
            split=split,
            source_entry=source_entry,
            output_root=output_root,
            run_config=run_config,
            args=args,
            trace_schema=trace_schema,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
