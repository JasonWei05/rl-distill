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

"""Audit an unsharded-HF Gemma 4 target overlay with the real verl FSDP2 train forward.

Run this script with ``torchrun``. Every rank loads the same immutable Gemma 4
checkpoint through verl's production ``TrainingWorker``. The audit first
compares selected stored sequences with the FP16 overlay, then reconstructs the
exact seed-42 first production train batch and backpropagates the real
distillation objective through the configured fixed-microbatch policy.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import subprocess
import sys
from bisect import bisect_right
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
for import_root in (str(SCRIPT_PATH.parent), str(SCRIPTS_ROOT), str(REPO_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

import audit_gemma4_cross_engine_topk as cross_engine  # noqa: E402
from gemma4_distill_trace_schema import TOPK_WIDTH, atomic_write_json, hash_json, sha256_file  # noqa: E402
from gemma4_model_identity import inspect_local_hf_model  # noqa: E402

REPORT_VERSION = 2
EXPECTED_SCHEMA = "gemma4-hf-bf16-sdpa-topk-overlay-v1"
EXPECTED_TARGET_ENGINE = "hf_bf16_sdpa_full_forward"
AUDITED_SOURCE_PATHS = (
    "rl-distill-scripts/data/audit_gemma4_fsdp2_training_topk.py",
    "rl-distill-scripts/data/audit_gemma4_cross_engine_topk.py",
    "rl-distill-scripts/data/gemma4_model_identity.py",
    "rl-distill-scripts/full_vocab_kl_loss.py",
    "verl/utils/dataset/dataset_utils.py",
    "verl/utils/fsdp_utils.py",
    "verl/workers/engine/utils.py",
    "verl/workers/engine/fsdp/transformer_impl.py",
    "verl/workers/engine_workers.py",
)
REQUIRED_COLUMNS = (
    "trace_id",
    "split",
    "source_uid",
    "sample_index",
    "input_ids",
    "response_mask",
    "teacher_topk_token_ids",
    "teacher_topk_logprobs",
    "teacher_sampled_token_logprobs",
    "prompt_length",
    "response_length",
    "target_model_identity_sha256",
)


class FSDP2TopKAuditError(ValueError):
    """Raised when the audit contract or its numerical gate fails."""


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FSDP2TopKAuditError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise FSDP2TopKAuditError(f"{description} {path} must contain a JSON object")
    return value


def verify_index(index_path: Path, model_path: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index = _load_json(index_path, "overlay dataset index")
    if index.get("schema_version") != EXPECTED_SCHEMA:
        raise FSDP2TopKAuditError(f"expected overlay schema {EXPECTED_SCHEMA!r}, got {index.get('schema_version')!r}")
    if index.get("target_engine") != EXPECTED_TARGET_ENGINE:
        raise FSDP2TopKAuditError(
            f"expected target engine {EXPECTED_TARGET_ENGINE!r}, got {index.get('target_engine')!r}"
        )
    claimed_hash = index.get("dataset_index_sha256")
    unhashed = dict(index)
    unhashed.pop("dataset_index_sha256", None)
    actual_hash = hash_json(unhashed)
    if claimed_hash != actual_hash:
        raise FSDP2TopKAuditError(f"overlay dataset-index self-hash mismatch: {actual_hash} != {claimed_hash}")

    model_identity = inspect_local_hf_model(model_path)
    expected_identity = index.get("target_model_identity", {}).get("model_identity_sha256")
    if model_identity.model_identity_sha256 != expected_identity:
        raise FSDP2TopKAuditError(
            "audit model does not match the overlay target identity: "
            f"{model_identity.model_identity_sha256} != {expected_identity}"
        )

    split_value = index.get("splits", {}).get(split)
    if not isinstance(split_value, dict) or not isinstance(split_value.get("shards"), list):
        raise FSDP2TopKAuditError(f"overlay index is missing split {split!r}")
    root = index_path.parent
    shards: list[dict[str, Any]] = []
    for entry in split_value["shards"]:
        relative = Path(entry.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise FSDP2TopKAuditError(f"unsafe overlay shard path: {relative}")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise FSDP2TopKAuditError(f"overlay shard escapes its dataset root: {path}")
        if sha256_file(path) != entry.get("sha256"):
            raise FSDP2TopKAuditError(f"overlay shard hash mismatch: {path}")
        rows = int(entry.get("rows", -1))
        if rows <= 0:
            raise FSDP2TopKAuditError(f"overlay shard has invalid row count: {path}: {rows}")
        shards.append({"path": path, "rows": rows, "sha256": entry["sha256"]})
    if not shards:
        raise FSDP2TopKAuditError(f"overlay split {split!r} has no shards")
    return index, shards


def select_global_rows(shards: list[dict[str, Any]], count: int) -> list[tuple[Path, int, str]]:
    """Select deterministic, evenly spaced rows across all registered shards."""
    total_rows = sum(int(shard["rows"]) for shard in shards)
    if count <= 0 or count > total_rows:
        raise FSDP2TopKAuditError(f"selected row count must be in [1, {total_rows}], got {count}")
    indices = np.linspace(0, total_rows - 1, num=count, dtype=np.int64).tolist()
    if len(set(indices)) != count:
        raise FSDP2TopKAuditError("could not select distinct audit rows")
    return locate_global_rows(shards, indices)


def locate_global_rows(shards: list[dict[str, Any]], indices: list[int]) -> list[tuple[Path, int, str]]:
    total_rows = sum(int(shard["rows"]) for shard in shards)
    if any(index < 0 or index >= total_rows for index in indices):
        raise FSDP2TopKAuditError(f"row index outside [0, {total_rows}): {indices}")
    stops = np.cumsum([int(shard["rows"]) for shard in shards]).tolist()
    selected: list[tuple[Path, int, str]] = []
    for global_index in indices:
        shard_index = bisect_right(stops, int(global_index))
        shard_start = 0 if shard_index == 0 else stops[shard_index - 1]
        shard = shards[shard_index]
        selected.append((Path(shard["path"]), int(global_index) - shard_start, str(shard["sha256"])))
    return selected


def load_rows(selected: list[tuple[Path, int, str]], expected_identity: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, row_index, shard_sha256 in selected:
        parquet_file = pq.ParquetFile(path)
        try:
            missing = set(REQUIRED_COLUMNS).difference(parquet_file.schema_arrow.names)
            if missing:
                raise FSDP2TopKAuditError(f"overlay shard {path} is missing columns: {sorted(missing)}")
            table = parquet_file.read(columns=list(REQUIRED_COLUMNS), use_threads=False)
        finally:
            parquet_file.close()
        if row_index >= table.num_rows:
            raise FSDP2TopKAuditError(f"selected row {row_index} is outside {path}")
        row = table.slice(row_index, 1).to_pylist()[0]
        if row["target_model_identity_sha256"] != expected_identity:
            raise FSDP2TopKAuditError(f"row target identity mismatch at {path}:{row_index}")
        row["path"] = str(path)
        row["row_index"] = row_index
        row["registered_shard_sha256"] = shard_sha256
        rows.append(row)
    return rows


def first_distributed_train_batch_indices(
    *,
    dataset_size: int,
    world_size: int,
    rank: int,
    global_batch_size: int,
    seed: int,
) -> list[int]:
    from torch.utils.data import DistributedSampler

    if global_batch_size % world_size != 0:
        raise FSDP2TopKAuditError("production train batch size must be divisible by world size")
    local_batch_size = global_batch_size // world_size
    sampler = DistributedSampler(
        range(dataset_size),
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    sampler.set_epoch(0)
    indices = list(sampler)
    if len(indices) < local_batch_size:
        raise FSDP2TopKAuditError("production dataset is smaller than one distributed train batch")
    return indices[:local_batch_size]


def validate_row(row: dict[str, Any]) -> None:
    input_ids = [int(value) for value in row["input_ids"]]
    response_mask = [int(value) for value in row["response_mask"]]
    prompt_length = int(row["prompt_length"])
    response_length = int(row["response_length"])
    if len(input_ids) != prompt_length + response_length:
        raise FSDP2TopKAuditError(f"input length mismatch for trace {row['trace_id']}")
    if response_mask != [0] * prompt_length + [1] * response_length:
        raise FSDP2TopKAuditError(f"response mask mismatch for trace {row['trace_id']}")
    topk_ids = np.asarray(row["teacher_topk_token_ids"], dtype=np.int64)
    topk_logprobs = np.asarray(row["teacher_topk_logprobs"], dtype=np.float32)
    sampled_logprobs = np.asarray(row["teacher_sampled_token_logprobs"], dtype=np.float32)
    if topk_ids.shape != (response_length, TOPK_WIDTH):
        raise FSDP2TopKAuditError(f"top-k ID shape mismatch for trace {row['trace_id']}: {topk_ids.shape}")
    if topk_logprobs.shape != topk_ids.shape or sampled_logprobs.shape != (response_length,):
        raise FSDP2TopKAuditError(f"top-k log-probability shape mismatch for trace {row['trace_id']}")
    if not np.isfinite(topk_logprobs).all() or not np.isfinite(sampled_logprobs).all():
        raise FSDP2TopKAuditError(f"non-finite stored log probability for trace {row['trace_id']}")


def repository_provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "repository_root": str(REPO_ROOT),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
        "script_path": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "source_sha256s": {relative: sha256_file(REPO_ROOT / relative) for relative in AUDITED_SOURCE_PATHS},
    }


def evaluate_gate(
    aggregate: dict[str, Any],
    exact: dict[str, Any],
    grad_norm: float | None,
    production_batches: list[dict[str, Any] | None],
    args,
) -> dict[str, Any]:
    observations = {
        "top1_tie_safe_mean": float(aggregate["top1_tie_safe"]["mean"]),
        "top10_overlap_fraction_mean": float(aggregate["top10_overlap_fraction"]["mean"]),
        "topk_overlap_fraction_mean": float(aggregate["topk_overlap_fraction"]["mean"]),
        "weighted_abs_logprob_delta_mean": float(aggregate["stored_support_weighted_abs_logprob_delta"]["mean"]),
        "support_probability_l1_mean": float(aggregate["stored_support_probability_l1"]["mean"]),
        "sampled_token_abs_logprob_delta_p95": float(aggregate["sampled_token_abs_logprob_delta"]["p95"]),
        "stored_only_topk_mass_p99": float(aggregate["stored_only_topk_mass"]["p99"]),
        "reference_only_topk_mass_p99": float(aggregate["reference_only_topk_mass"]["p99"]),
        "ordered_topk_exact_mean": float(exact["ordered_topk_exact"]["mean"]),
        "fp16_support_logprob_exact_fraction_mean": float(exact["fp16_support_exact_fraction"]["mean"]),
        "fp16_sampled_logprob_exact_mean": float(exact["fp16_sampled_exact"]["mean"]),
    }
    requirements = {
        "top1_tie_safe_mean": (">=", args.min_top1_tie_safe),
        "top10_overlap_fraction_mean": (">=", args.min_top10_overlap),
        "topk_overlap_fraction_mean": (">=", args.min_topk_overlap),
        "weighted_abs_logprob_delta_mean": ("<=", args.max_weighted_abs_logprob_delta),
        "support_probability_l1_mean": ("<=", args.max_support_probability_l1),
        "sampled_token_abs_logprob_delta_p95": ("<=", args.max_sampled_token_abs_delta_p95),
        "stored_only_topk_mass_p99": ("<=", args.max_membership_delta_mass_p99),
        "reference_only_topk_mass_p99": ("<=", args.max_membership_delta_mass_p99),
    }
    if grad_norm is not None:
        complete_batches = [batch for batch in production_batches if batch is not None]
        microbatch_counts = [int(batch["microbatch_count"]) for batch in complete_batches]
        all_global_indices = [index for batch in complete_batches for index in batch["global_indices"]]
        all_microbatches = [microbatch for batch in complete_batches for microbatch in batch["microbatches"]]
        multi_sequence_padded_tokens = [
            int(microbatch["padded_tokens"]) for microbatch in all_microbatches if len(microbatch["indices"]) > 1
        ]
        observations.update(
            {
                "production_rank_count": float(len(complete_batches)),
                "production_row_count": float(len(all_global_indices)),
                "production_unique_row_count": float(len(set(all_global_indices))),
                "production_microbatch_count_min": float(min(microbatch_counts, default=-1)),
                "production_microbatch_count_max": float(max(microbatch_counts, default=-1)),
                "production_max_microbatch_size": float(
                    max((len(microbatch["indices"]) for microbatch in all_microbatches), default=-1)
                ),
                "production_max_multi_sequence_padded_tokens": float(max(multi_sequence_padded_tokens, default=0)),
            }
        )
        requirements.update(
            {
                "production_rank_count": ("==", len(production_batches)),
                "production_row_count": ("==", args.train_batch_size),
                "production_unique_row_count": ("==", args.train_batch_size),
                "production_microbatch_count_min": ("==", observations["production_microbatch_count_max"]),
                "production_max_microbatch_size": ("<=", args.micro_batch_size_per_gpu),
                "production_max_multi_sequence_padded_tokens": (
                    "<=",
                    args.max_padded_tokens_per_microbatch,
                ),
            }
        )
    if grad_norm is not None:
        observations["backward_grad_norm"] = float(grad_norm)
        requirements["backward_grad_norm"] = ("<=", args.max_grad_norm)
    checks: dict[str, Any] = {}
    failures: list[str] = []
    for name, (operator, threshold) in requirements.items():
        observed = observations[name]
        passed = math.isfinite(observed) and (
            observed >= threshold
            if operator == ">="
            else observed <= threshold
            if operator == "<="
            else observed == threshold
        )
        checks[name] = {
            "observed": observed,
            "operator": operator,
            "threshold": float(threshold),
            "passed": passed,
        }
        if not passed:
            failures.append(f"{name}={observed:.12g} does not satisfy {operator} {threshold:.12g}")
    return {"status": "pass" if not failures else "fail", "checks": checks, "failure_reasons": failures}


def run_distributed_audit(args) -> int:
    import torch
    import torch.distributed as dist
    from full_vocab_kl_loss import FullVocabKLLoss

    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode, SFTTensorCollator
    from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
    from verl.utils.fsdp_utils import fsdp2_grad_norm_diagnostics
    from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
    from verl.workers.engine.utils import prepare_micro_batches
    from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig

    initialize_global_process_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    try:
        model_path = args.model.resolve(strict=True)
        student_model_path = args.student_model.resolve(strict=True)
        student_identity = inspect_local_hf_model(student_model_path)
        index_path = args.dataset_index.resolve(strict=True)
        output_path = args.output.resolve()
        if output_path.exists() and not args.overwrite:
            raise FSDP2TopKAuditError(f"output already exists: {output_path}; pass --overwrite to replace it")
        if (
            output_path.is_relative_to(model_path)
            or output_path.is_relative_to(student_model_path)
            or output_path.is_relative_to(index_path.parent)
        ):
            raise FSDP2TopKAuditError("audit output must be outside the immutable model and dataset trees")

        index, shards = verify_index(index_path, model_path, args.split)
        train_index, train_shards = verify_index(index_path, model_path, "train")
        if train_index["dataset_index_sha256"] != index["dataset_index_sha256"]:
            raise FSDP2TopKAuditError("train and parity selections resolved different overlay identities")
        expected_identity = index["target_model_identity"]["model_identity_sha256"]
        selected = select_global_rows(shards, world_size * args.traces_per_rank)
        local_selection = selected[rank * args.traces_per_rank : (rank + 1) * args.traces_per_rank]
        rows = load_rows(local_selection, expected_identity)
        for row in rows:
            validate_row(row)

        train_total_rows = sum(int(shard["rows"]) for shard in train_shards)
        local_train_indices = first_distributed_train_batch_indices(
            dataset_size=train_total_rows,
            world_size=world_size,
            rank=rank,
            global_batch_size=args.train_batch_size,
            seed=args.train_seed,
        )
        local_train_selection = locate_global_rows(train_shards, local_train_indices)
        train_rows = load_rows(local_train_selection, expected_identity)
        for global_index, row in zip(local_train_indices, train_rows, strict=True):
            validate_row(row)
            row["global_index"] = global_index

        os.environ.setdefault("VERL_FSDP2_LOCAL_LOAD", "1")
        if not args.fsdp_wrap:
            if world_size != 1:
                raise FSDP2TopKAuditError("--no-fsdp-wrap is a single-rank diagnostic only")
            from verl.workers.engine.fsdp import transformer_impl as fsdp_impl

            fsdp_impl.apply_fsdp2 = lambda _module, _fsdp_kwargs, _engine_config: None
            fsdp_impl.fsdp2_load_full_state_dict = lambda _module, _full_state, _fsdp_mesh, _offload_policy: None
        backward_exercised = args.execution_mode == "train"

        def build_worker(checkpoint_path: Path):
            model_config = HFModelConfig(
                path=str(checkpoint_path),
                use_remove_padding=False,
                override_config={"attn_implementation": "sdpa"},
                enable_gradient_checkpointing=args.gradient_checkpointing,
            )
            engine_config = FSDPEngineConfig(
                forward_only=False,
                strategy="fsdp2",
                fsdp_size=-1,
                model_dtype=args.model_dtype,
                use_remove_padding=False,
                use_dynamic_bsz=False,
                use_torch_compile=False,
                mixed_precision={
                    "param_dtype": args.fsdp_param_dtype,
                    "reduce_dtype": args.fsdp_reduce_dtype,
                    "buffer_dtype": args.fsdp_buffer_dtype,
                    "cast_forward_inputs": True,
                },
                wrap_policy={"transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer"]},
            )
            optimizer_config = FSDPOptimizerConfig(
                lr=0.0,
                total_training_steps=1,
                lr_warmup_steps=0,
                lr_scheduler_type="constant",
                clip_grad=1.0,
            )
            built_worker = TrainingWorker(
                TrainingWorkerConfig(
                    model_type="language_model",
                    model_config=model_config,
                    engine_config=engine_config,
                    optimizer_config=optimizer_config,
                    checkpoint_config=None,
                    profiler_config=None,
                )
            )
            built_worker.reset()
            return built_worker

        worker = build_worker(model_path)
        engine = worker.engine
        if not args.fsdp_wrap:
            engine.module.to("cuda")
        module = getattr(engine.module, "module", engine.module)
        helper = FullVocabKLLoss(
            precomputed_topk=True,
            top_k=TOPK_WIDTH,
            chunk_size=args.kl_chunk_size,
            checkpoint_student_chunks=True,
        )
        local_records: list[dict[str, Any]] = []
        local_trace_records: list[dict[str, Any]] = []
        local_production_batch: dict[str, Any] | None = None

        grad_diagnostics = None
        mode_context = engine.train_mode() if backward_exercised else engine.eval_mode()
        with mode_context:
            for row in rows:
                prompt_length = int(row["prompt_length"])
                response_length = int(row["response_length"])
                response_positions = cross_engine.selected_positions(response_length, args.positions_per_trace)
                prediction_positions = [prompt_length - 1 + position for position in response_positions]
                input_ids = torch.tensor(row["input_ids"], dtype=torch.long, device="cuda").unsqueeze(0)
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device="cuda").unsqueeze(0)
                prediction_indices = torch.tensor(prediction_positions, dtype=torch.long, device="cuda")
                batch_indices = torch.zeros_like(prediction_indices)

                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    common_forward_kwargs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "use_cache": False,
                        "return_dict": True,
                    }
                    if args.forward_path == "native_logits":
                        raw_output = engine.module(
                            **common_forward_kwargs,
                            logits_to_keep=prediction_indices,
                        )
                        logits = raw_output.logits.squeeze(0)
                        softcap = engine._logit_softcap
                        if softcap is not None:
                            logits = torch.tanh(logits / softcap) * softcap
                    else:
                        compact_hidden = args.forward_path == "compact_hidden"
                        raw_output = engine.module(
                            **common_forward_kwargs,
                            logits_to_keep=prediction_indices if compact_hidden else 0,
                            logits_to_keep_batch_indices=batch_indices if compact_hidden else None,
                            skip_lm_head=True,
                        )
                        active_hidden = raw_output.logits.squeeze(0)
                        if not compact_hidden:
                            active_hidden = active_hidden.index_select(0, prediction_indices)
                        with helper._lm_head_forward_context(module.lm_head) as (full_weight, full_bias):
                            logits = helper._apply_lm_head(
                                module.lm_head,
                                active_hidden,
                                module.config,
                                full_weight=full_weight,
                                full_bias=full_bias,
                                logit_softcap_override=engine._logit_softcap,
                            )
                    logits_fp32 = logits.float()
                    log_denominator = torch.logsumexp(logits_fp32, dim=-1, keepdim=True)
                    reference_top_logits, reference_top_ids = torch.topk(logits_fp32, k=TOPK_WIDTH, dim=-1)
                    reference_top_logprobs = reference_top_logits - log_denominator

                    stored_ids = torch.tensor(
                        np.asarray(row["teacher_topk_token_ids"], dtype=np.int64)[response_positions],
                        dtype=torch.long,
                        device="cuda",
                    )
                    stored_logprobs = torch.tensor(
                        np.asarray(row["teacher_topk_logprobs"], dtype=np.float32)[response_positions],
                        dtype=torch.float32,
                        device="cuda",
                    )
                    sampled_ids = input_ids[0, prediction_indices + 1]
                    stored_sampled_logprobs = torch.tensor(
                        np.asarray(row["teacher_sampled_token_logprobs"], dtype=np.float32)[response_positions],
                        dtype=torch.float32,
                        device="cuda",
                    )
                    reference_on_stored = logits_fp32.gather(-1, stored_ids) - log_denominator
                    reference_sampled = logits_fp32.gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1)
                    reference_sampled = reference_sampled - log_denominator.squeeze(-1)

                stored_ids_cpu = stored_ids.detach().cpu().numpy()
                stored_logprobs_cpu = stored_logprobs.detach().cpu().numpy()
                reference_top_ids_cpu = reference_top_ids.detach().cpu().numpy()
                reference_top_logprobs_cpu = reference_top_logprobs.detach().cpu().numpy()
                reference_on_stored_cpu = reference_on_stored.detach().cpu().numpy()
                stored_sampled_cpu = stored_sampled_logprobs.detach().cpu().numpy()
                reference_sampled_cpu = reference_sampled.detach().cpu().numpy()
                for local_index, response_position in enumerate(response_positions):
                    metrics = cross_engine.compare_topk_position(
                        stored_ids=stored_ids_cpu[local_index].tolist(),
                        stored_logprobs=stored_logprobs_cpu[local_index].tolist(),
                        reference_top_ids=reference_top_ids_cpu[local_index].tolist(),
                        reference_top_logprobs=reference_top_logprobs_cpu[local_index].tolist(),
                        reference_logprobs_on_stored=reference_on_stored_cpu[local_index].tolist(),
                        stored_sampled_logprob=float(stored_sampled_cpu[local_index]),
                        reference_sampled_logprob=float(reference_sampled_cpu[local_index]),
                        top1_tie_logprob_tolerance=args.top1_tie_logprob_tolerance,
                    )
                    fp16_stored = stored_logprobs_cpu[local_index].astype(np.float16)
                    fp16_reference = reference_on_stored_cpu[local_index].astype(np.float16)
                    local_records.append(
                        {
                            "rank": rank,
                            "trace_id": row["trace_id"],
                            "split": row["split"],
                            "source_uid": row["source_uid"],
                            "sample_index": int(row["sample_index"]),
                            "prompt_length": prompt_length,
                            "response_length": response_length,
                            "response_position": int(response_position),
                            "prediction_position": int(prediction_positions[local_index]),
                            "response_fraction": response_position / max(1, response_length - 1),
                            "ordered_topk_exact": int(
                                np.array_equal(stored_ids_cpu[local_index], reference_top_ids_cpu[local_index])
                            ),
                            "fp16_support_exact_fraction": float(np.mean(fp16_stored == fp16_reference)),
                            "fp16_sampled_exact": int(
                                np.float16(stored_sampled_cpu[local_index])
                                == np.float16(reference_sampled_cpu[local_index])
                            ),
                            **metrics,
                        }
                    )
                local_trace_records.append(
                    {
                        "rank": rank,
                        "trace_id": row["trace_id"],
                        "path": row["path"],
                        "row_index": int(row["row_index"]),
                        "registered_shard_sha256": row["registered_shard_sha256"],
                        "prompt_length": prompt_length,
                        "response_length": response_length,
                        "positions_scored": response_positions,
                    }
                )
                del raw_output, logits, logits_fp32, log_denominator

        del module, engine, worker
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()

        if backward_exercised:
            student_worker = build_worker(student_model_path)
            student_engine = student_worker.engine
            if not args.fsdp_wrap:
                student_engine.module.to("cuda")
            student_helper = FullVocabKLLoss(
                precomputed_topk=True,
                top_k=TOPK_WIDTH,
                chunk_size=args.kl_chunk_size,
                checkpoint_student_chunks=True,
            )
            samples = []
            for row in train_rows:
                input_ids_cpu = torch.tensor(row["input_ids"], dtype=torch.long)
                loss_mask_cpu = torch.tensor(row["response_mask"], dtype=torch.long)
                teacher_ids_cpu = torch.from_numpy(np.asarray(row["teacher_topk_token_ids"], dtype=np.int32)).flatten()
                teacher_logprobs_cpu = torch.from_numpy(
                    np.asarray(row["teacher_topk_logprobs"], dtype=np.float16)
                ).flatten()
                samples.append(
                    {
                        "input_ids": input_ids_cpu,
                        "position_ids": torch.arange(input_ids_cpu.numel(), dtype=torch.long),
                        "loss_mask": loss_mask_cpu,
                        "teacher_topk_token_ids": teacher_ids_cpu,
                        "teacher_topk_logprobs": teacher_logprobs_cpu,
                    }
                )

            collated = SFTTensorCollator(DatasetPadMode.NO_PADDING)(samples)
            production_data = tu.get_tensordict(
                tensor_dict=collated,
                non_tensor_dict={
                    "use_remove_padding": False,
                    "use_dynamic_bsz": False,
                    "max_token_len_per_gpu": args.max_length,
                    "micro_batch_size_per_gpu": args.micro_batch_size_per_gpu,
                    "max_padded_tokens_per_microbatch": args.max_padded_tokens_per_microbatch,
                    "temperature": 1.0,
                    "global_batch_size": args.train_batch_size,
                    "pad_mode": DatasetPadMode.NO_PADDING,
                    "pad_token_id": 0,
                    "use_logits_processor": True,
                    "skip_lm_log_probs": True,
                    "use_hidden_logits_processor": True,
                },
            )
            tu.assign_non_tensor(production_data, sp_size=1)
            audit_micro_batches, audit_indices = prepare_micro_batches(
                data=production_data,
                dp_group=student_engine.get_data_parallel_group(),
                same_micro_num_in_dp=True,
            )
            sequence_lengths = production_data["input_ids"].offsets().diff().tolist()
            microbatch_layout = []
            for partition in audit_indices:
                lengths = [int(sequence_lengths[index]) for index in partition]
                microbatch_layout.append(
                    {
                        "indices": partition,
                        "sequence_lengths": lengths,
                        "padded_tokens": max(lengths) * len(lengths),
                    }
                )
            local_production_batch = {
                "rank": rank,
                "global_indices": local_train_indices,
                "trace_ids": [row["trace_id"] for row in train_rows],
                "sequence_lengths": sequence_lengths,
                "microbatch_count": len(audit_micro_batches),
                "microbatches": microbatch_layout,
            }

            with student_engine.train_mode():
                student_engine.optimizer_zero_grad()
                student_engine.forward_backward_batch(
                    data=production_data,
                    loss_function=student_helper,
                    forward_only=False,
                )
                # The train-mode context clears gradients on exit, so measure
                # the exact production batch while it remains active.
                grad_diagnostics = fsdp2_grad_norm_diagnostics(
                    student_engine.module.named_parameters(),
                    group=student_engine.get_data_parallel_group(),
                    top_k=20,
                )
        gathered_records: list[Any] | None = [None] * world_size if rank == 0 else None
        gathered_traces: list[Any] | None = [None] * world_size if rank == 0 else None
        gathered_production_batches: list[Any] | None = [None] * world_size if rank == 0 else None
        dist.gather_object(local_records, gathered_records, dst=0)
        dist.gather_object(local_trace_records, gathered_traces, dst=0)
        dist.gather_object(local_production_batch, gathered_production_batches, dst=0)

        exit_code = 0
        if rank == 0:
            records = [record for rank_records in gathered_records for record in rank_records]
            traces = [record for rank_records in gathered_traces for record in rank_records]
            production_batches = list(gathered_production_batches)
            aggregate = cross_engine.aggregate(records)
            exact = {
                name: cross_engine.summary([float(record[name]) for record in records])
                for name in (
                    "ordered_topk_exact",
                    "fp16_support_exact_fraction",
                    "fp16_sampled_exact",
                )
            }
            grad_norm = float(grad_diagnostics["total_norm"]) if grad_diagnostics is not None else None
            gate = evaluate_gate(aggregate, exact, grad_norm, production_batches, args)
            checkpointing_label = "enabled" if args.gradient_checkpointing else "disabled"
            report = {
                "report_version": REPORT_VERSION,
                "created_at": datetime.now(UTC).isoformat(),
                "status": gate["status"],
                "gate": gate,
                "contract": {
                    "reference": (
                        "E2B target parity plus E4B student exact first-batch backward through "
                        f"verl TrainingWorker FSDP2 {args.execution_mode} mode, activation checkpointing "
                        f"{checkpointing_label}, {args.fsdp_param_dtype} FSDP parameters, "
                        f"{args.fsdp_reduce_dtype} reductions, {args.fsdp_buffer_dtype} buffers, "
                        "BF16 autocast, SDPA, use_cache=False"
                    ),
                    "candidate": "precomputed unsharded-HF BF16 SDPA top-128 overlay",
                    "causal_alignment": "response token j is scored by hidden position prompt_length - 1 + j",
                    "top_k": TOPK_WIDTH,
                    "execution_mode": args.execution_mode,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "forward_path": args.forward_path,
                    "fsdp_wrap": args.fsdp_wrap,
                    "backward_exercised": backward_exercised,
                    "model_dtype": args.model_dtype,
                    "fsdp_param_dtype": args.fsdp_param_dtype,
                    "fsdp_reduce_dtype": args.fsdp_reduce_dtype,
                    "fsdp_buffer_dtype": args.fsdp_buffer_dtype,
                    "fsdp_cast_forward_inputs": True,
                    "train_seed": args.train_seed,
                    "train_batch_size": args.train_batch_size,
                    "micro_batch_size_per_gpu": args.micro_batch_size_per_gpu,
                    "max_padded_tokens_per_microbatch": args.max_padded_tokens_per_microbatch,
                    "kl_chunk_size": args.kl_chunk_size,
                    "max_length": args.max_length,
                },
                "dataset": {
                    "index_path": str(index_path),
                    "dataset_index_sha256": index["dataset_index_sha256"],
                    "source_dataset_index_sha256": index["source_dataset_index_sha256"],
                    "target_model_identity_sha256": expected_identity,
                    "split": args.split,
                },
                "selection": {
                    "world_size": world_size,
                    "traces_per_rank": args.traces_per_rank,
                    "positions_per_trace": args.positions_per_trace,
                    "trace_count": len(traces),
                    "position_count": len(records),
                    "traces": traces,
                    "production_first_train_batch": production_batches,
                },
                "model": {
                    "path": str(model_path),
                    "model_identity_sha256": expected_identity,
                    "dtype_load": args.model_dtype,
                    "autocast": "bfloat16",
                    "attention_implementation": "sdpa",
                },
                "student_model": {
                    "path": str(student_model_path),
                    "model_identity_sha256": student_identity.model_identity_sha256,
                    "dtype_load": args.model_dtype,
                    "autocast": "bfloat16",
                    "attention_implementation": "sdpa",
                },
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "pyarrow": pa.__version__,
                    "gpu": torch.cuda.get_device_name(),
                },
                "implementation": repository_provenance(),
                "backward": grad_diagnostics,
                "aggregate": aggregate,
                "exact_serialization": exact,
                "positions": records,
            }
            atomic_write_json(output_path, report)
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "output": str(output_path),
                        "position_count": len(records),
                        "gate": gate,
                        "exact_serialization": exact,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
            exit_code = 0 if gate["status"] == "pass" else 2
        exit_tensor = torch.tensor(exit_code, dtype=torch.int32, device="cuda")
        dist.broadcast(exit_tensor, src=0)
        return int(exit_tensor.item())
    finally:
        destroy_global_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--traces-per-rank", type=int, default=2)
    parser.add_argument("--positions-per-trace", type=int, default=32)
    parser.add_argument("--top1-tie-logprob-tolerance", type=float, default=0.0025)
    parser.add_argument("--model-dtype", choices=("fp32", "bfloat16"), default="fp32")
    parser.add_argument("--execution-mode", choices=("train", "eval"), default="train")
    parser.add_argument(
        "--forward-path",
        choices=("compact_hidden", "full_hidden", "native_logits"),
        default="compact_hidden",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fsdp-wrap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fsdp-param-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--fsdp-reduce-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--fsdp-buffer-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size-per-gpu", type=int, default=2)
    parser.add_argument("--max-padded-tokens-per-microbatch", type=int, default=5120)
    parser.add_argument("--kl-chunk-size", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=12288)
    parser.add_argument("--min-top1-tie-safe", type=float, default=0.999)
    parser.add_argument("--min-top10-overlap", type=float, default=0.995)
    parser.add_argument("--min-topk-overlap", type=float, default=0.995)
    parser.add_argument("--max-weighted-abs-logprob-delta", type=float, default=0.003)
    parser.add_argument("--max-support-probability-l1", type=float, default=0.003)
    parser.add_argument("--max-sampled-token-abs-delta-p95", type=float, default=0.01)
    parser.add_argument("--max-membership-delta-mass-p99", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name in (
        "traces_per_rank",
        "positions_per_trace",
        "train_batch_size",
        "micro_batch_size_per_gpu",
        "max_padded_tokens_per_microbatch",
        "kl_chunk_size",
        "max_length",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


if __name__ == "__main__":
    sys.exit(run_distributed_audit(parse_args()))
