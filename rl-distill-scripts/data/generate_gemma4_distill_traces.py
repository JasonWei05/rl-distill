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

"""Generate resumable Gemma 4 off-policy traces with precomputed top-128.

The module keeps vLLM behind ``main``/``run_generation`` so schema helpers and
rank extraction can be tested on hosts where vLLM is not installed.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from gemma4_distill_trace_schema import (
    MANIFEST_VERSION,
    RESPONSE_TEXT_NORMALIZATION,
    SCHEMA_VERSION,
    TOPK_WIDTH,
    TraceValidationError,
    atomic_write_json,
    derive_sampling_seed,
    hash_json,
    normalized_decode,
    parquet_manifest_path,
    publish_parquet_temporary,
    sha256_file,
    sha256_text,
    tokenizer_fingerprint,
    validate_parquet_shard,
    validate_shard_bundle,
    write_parquet_temporary,
)
from gemma4_model_identity import (
    generation_teacher_identity,
    inspect_local_hf_model,
    require_sha256,
)

DEFAULT_CHAT_TEMPLATE = Path(__file__).with_name("gemma3_it_fewshot_math.jinja")
DEFAULT_STOP_STRINGS = ("<end_of_turn>", "<start_of_turn>")
DIRECTIONS = (
    "e4b_rl100_to_e2b",
    "e2b_base_to_e4b",
    "e4b_easy_to_e2b",
    "e4b_medium_to_e2b",
    "e4b_hard_to_e2b",
    "12b_easy_to_e2b",
    "12b_medium_to_e2b",
    "26b_easy_to_e2b",
)
SPLITS = ("train", "validation")


@dataclass(frozen=True)
class SourcePrompt:
    prompt_index: int
    messages: list[dict[str, Any]]
    question_text: str
    gold_answer: str
    source_uid: str
    source_uid_original: str | None
    question_sha256: str


@dataclass(frozen=True)
class PreparedRequest:
    source: SourcePrompt
    sample_index: int
    sampling_seed: int
    prompt_token_ids: list[int]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_versions(vllm_version: str | None = None) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pyarrow": _package_version("pyarrow"),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "vllm": vllm_version or _package_version("vllm"),
    }


def repository_state(repo_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def generator_source_hash() -> str:
    source_paths = [Path(__file__), Path(__file__).with_name("gemma4_distill_trace_schema.py")]
    payload = []
    for path in source_paths:
        payload.append({"name": path.name, "sha256": sha256_file(path)})
    return hash_json(payload)


def _extract_question(messages: Any) -> tuple[list[dict[str, Any]], str]:
    if hasattr(messages, "as_py"):
        messages = messages.as_py()
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if not isinstance(messages, list | tuple) or not messages:
        raise ValueError("source prompt must be a non-empty chat-message list")
    normalized_messages = [dict(message) for message in messages]
    user_messages = [message for message in normalized_messages if message.get("role") == "user"]
    if not user_messages:
        raise ValueError("source prompt does not contain a user message")
    content = user_messages[-1].get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("the final user message must contain non-empty string content")
    return normalized_messages, content


def _extract_gold(reward_model: Any) -> str:
    if hasattr(reward_model, "as_py"):
        reward_model = reward_model.as_py()
    if not isinstance(reward_model, Mapping) or reward_model.get("ground_truth") is None:
        raise ValueError("reward_model.ground_truth is required")
    return str(reward_model["ground_truth"])


def load_unique_source_prompts(path: str | Path, *, max_questions: int = -1) -> tuple[list[SourcePrompt], int]:
    parquet_file = pq.ParquetFile(path)
    available = set(parquet_file.schema_arrow.names)
    required = {"prompt", "reward_model"}
    if not required.issubset(available):
        raise ValueError(f"source parquet is missing columns: {sorted(required.difference(available))}")
    columns = ["prompt", "reward_model"]
    if "uid" in available:
        columns.append("uid")

    prompts: list[SourcePrompt] = []
    by_source_uid: dict[str, SourcePrompt] = {}
    source_rows = 0
    for batch in parquet_file.iter_batches(columns=columns, batch_size=1024):
        for row in batch.to_pylist():
            prompt_index = source_rows
            source_rows += 1
            messages, question = _extract_question(row["prompt"])
            gold = _extract_gold(row["reward_model"])
            question_hash = sha256_text(question)
            original_uid = str(row["uid"]) if row.get("uid") not in (None, "") else None
            # Validation repeats are identified by their shared UID and must be
            # collapsed.  The train parquet has no UID and contains some
            # duplicate question strings, so each source row remains a distinct
            # training question to preserve the registered 9,723 x 5 schedule.
            source_uid = original_uid or f"row:{prompt_index:08d}:sha256:{question_hash}"
            candidate = SourcePrompt(
                prompt_index=prompt_index,
                messages=messages,
                question_text=question,
                gold_answer=gold,
                source_uid=source_uid,
                source_uid_original=original_uid,
                question_sha256=question_hash,
            )
            previous = by_source_uid.get(source_uid)
            if previous is not None:
                if previous.messages != messages or previous.question_text != question or previous.gold_answer != gold:
                    raise ValueError(f"source UID {source_uid!r} has conflicting rows at index {prompt_index}")
                continue
            by_source_uid[source_uid] = candidate
            prompts.append(candidate)
            if max_questions > 0 and len(prompts) >= max_questions:
                return prompts, source_rows
    return prompts, source_rows


def _logprob_field(entry: Any, field_name: str) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(field_name)
    return getattr(entry, field_name, None)


def extract_ranked_topk(
    position_logprobs: Mapping[int, Any], sampled_token_id: int, *, topk_width: int = TOPK_WIDTH
) -> tuple[list[int], list[float], float]:
    """Extract ranks 1..K, excluding vLLM's possible extra sampled token."""

    if sampled_token_id not in position_logprobs:
        raise ValueError(f"sampled token {sampled_token_id} is absent from vLLM logprobs")
    sampled_logprob = float(_logprob_field(position_logprobs[sampled_token_id], "logprob"))
    if not math.isfinite(sampled_logprob):
        raise ValueError("sampled-token log probability is not finite")

    by_rank: dict[int, tuple[int, float]] = {}
    for raw_token_id, entry in position_logprobs.items():
        rank = _logprob_field(entry, "rank")
        if rank is None:
            continue
        rank = int(rank)
        if rank < 1 or rank > topk_width:
            continue
        if rank in by_rank:
            raise ValueError(f"vLLM returned duplicate teacher rank {rank}")
        token_id = int(raw_token_id)
        logprob = float(_logprob_field(entry, "logprob"))
        if not math.isfinite(logprob):
            raise ValueError(f"vLLM returned a non-finite log probability at rank {rank}")
        by_rank[rank] = (token_id, logprob)
    expected_ranks = set(range(1, topk_width + 1))
    if set(by_rank) != expected_ranks:
        missing = sorted(expected_ranks.difference(by_rank))
        raise ValueError(f"vLLM did not return exact ranks 1..{topk_width}; missing {missing[:8]}")
    ranked = [by_rank[rank] for rank in range(1, topk_width + 1)]
    token_ids = [token_id for token_id, _ in ranked]
    logprobs = [logprob for _, logprob in ranked]
    if len(set(token_ids)) != topk_width:
        raise ValueError("vLLM returned duplicate token IDs in ranks 1..K")
    if any(later > earlier + 1e-6 for earlier, later in zip(logprobs, logprobs[1:], strict=False)):
        raise ValueError("vLLM rank metadata disagrees with log-probability ordering")
    if sampled_token_id in token_ids:
        ranked_logprob = logprobs[token_ids.index(sampled_token_id)]
        if abs(ranked_logprob - sampled_logprob) > 1e-6:
            raise ValueError("sampled-token log probability disagrees with its ranked value")
    return token_ids, logprobs, sampled_logprob


def _render_prompt(tokenizer: Any, messages: Sequence[Mapping[str, Any]], chat_template: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        list(messages), chat_template=chat_template, tokenize=False, add_generation_prompt=True
    )
    try:
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    except AttributeError:
        token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    return [int(token_id) for token_id in token_ids]


def _matched_stop(stop_reason: Any, stop_strings: Sequence[str], tokenizer: Any) -> str | None:
    if isinstance(stop_reason, str) and stop_reason in stop_strings:
        return stop_reason
    if isinstance(stop_reason, int):
        decoded = normalized_decode(tokenizer, [stop_reason])
        if decoded in stop_strings:
            return decoded
    return None


def _stop_reason_string(stop_reason: Any) -> str | None:
    if stop_reason is None:
        return None
    return json.dumps(stop_reason, ensure_ascii=False, separators=(",", ":"))


def build_trace_record(
    *,
    request: PreparedRequest,
    output: Any,
    shard_id: int,
    row_within_shard: int,
    run_config: Mapping[str, Any],
    tokenizer: Any,
    strict_grade: float,
    strict_prediction: str,
    generation_timestamp: str,
) -> dict[str, Any]:
    if len(output.outputs) != 1:
        raise ValueError(f"expected one vLLM completion, got {len(output.outputs)}")
    if list(output.prompt_token_ids) != request.prompt_token_ids:
        raise ValueError("vLLM conditioned on prompt_token_ids different from the captured request")
    completion = output.outputs[0]
    response_ids = [int(token_id) for token_id in completion.token_ids]
    if completion.logprobs is None or len(completion.logprobs) != len(response_ids):
        raise ValueError("vLLM did not return one logprob mapping per generated response token")

    topk_ids: list[list[int]] = []
    topk_logprobs: list[list[float]] = []
    sampled_logprobs: list[float] = []
    for position_logprobs, sampled_token_id in zip(completion.logprobs, response_ids, strict=True):
        if position_logprobs is None:
            raise ValueError("vLLM returned null logprobs for a response position")
        ranked_ids, ranked_logprobs, sampled_logprob = extract_ranked_topk(position_logprobs, sampled_token_id)
        topk_ids.append(ranked_ids)
        topk_logprobs.append(ranked_logprobs)
        sampled_logprobs.append(sampled_logprob)

    semantic = run_config["semantic_config"]
    sampling = semantic["sampling"]
    prompt_ids = request.prompt_token_ids
    response_text = normalized_decode(tokenizer, response_ids)
    stop_reason = getattr(completion, "stop_reason", None)
    finish_reason = getattr(completion, "finish_reason", None)
    source = request.source
    trace_id = hash_json(
        {
            "generation_config_sha256": run_config["generation_config_sha256"],
            "source_uid": source.source_uid,
            "question_sha256": source.question_sha256,
            "sample_index": request.sample_index,
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generation_config_sha256": run_config["generation_config_sha256"],
        "trace_id": trace_id,
        "direction": semantic["direction"],
        "split": semantic["split"],
        "source_dataset": semantic["source_dataset"],
        "source_dataset_sha256": semantic["source_dataset_sha256"],
        "source_uid": source.source_uid,
        "source_uid_original": source.source_uid_original,
        "question_sha256": source.question_sha256,
        "prompt_index": source.prompt_index,
        "sample_index": request.sample_index,
        "question_text": source.question_text,
        "gold_answer": source.gold_answer,
        "strict_grade": float(strict_grade),
        "strict_correct": bool(strict_grade > 0.5),
        "strict_prediction": strict_prediction,
        "teacher_model": semantic["teacher"]["model"],
        "teacher_revision": semantic["teacher"]["revision"],
        "teacher_content_sha256": semantic["teacher"]["content_sha256"],
        "tokenizer_model": semantic["tokenizer"]["model"],
        "tokenizer_revision": semantic["tokenizer"]["revision"],
        "tokenizer_sha256": semantic["tokenizer"]["sha256"],
        "tokenizer_vocab_size": semantic["tokenizer"]["vocab_size"],
        "chat_template_path": semantic["chat_template"]["path"],
        "chat_template_sha256": semantic["chat_template"]["sha256"],
        "global_seed": semantic["global_seed"],
        "sampling_seed": request.sampling_seed,
        "sampling_parameters_json": json.dumps(sampling, sort_keys=True, separators=(",", ":")),
        "prompt_token_ids": prompt_ids,
        "response_token_ids": response_ids,
        "input_ids": prompt_ids + response_ids,
        "response_mask": [0] * len(prompt_ids) + [1] * len(response_ids),
        "teacher_topk_token_ids": topk_ids,
        "teacher_topk_logprobs": topk_logprobs,
        "sampled_token_ids": response_ids,
        "sampled_token_logprobs": sampled_logprobs,
        "teacher_topk_rank_order": f"1..{TOPK_WIDTH}",
        "prompt_length": len(prompt_ids),
        "response_length": len(response_ids),
        "finish_reason": finish_reason,
        "stop_reason": _stop_reason_string(stop_reason),
        "matched_stop_string": _matched_stop(stop_reason, sampling["stop"], tokenizer),
        "reached_max_response_tokens": bool(
            len(response_ids) == sampling["max_response_tokens"] and finish_reason == "length"
        ),
        "response_text": response_text,
        "vllm_response_text": completion.text,
        "response_text_normalization": RESPONSE_TEXT_NORMALIZATION,
        "shard_id": shard_id,
        "row_within_shard": row_within_shard,
        "generation_timestamp": generation_timestamp,
        "generator_commit": semantic["generator"]["commit"],
        "generator_source_sha256": semantic["generator"]["source_sha256"],
        "environment_versions_json": json.dumps(
            semantic["environment_versions"], sort_keys=True, separators=(",", ":")
        ),
    }


def _is_hex_revision(value: str | None) -> bool:
    if value is None or len(value) not in (40, 64):
        return False
    try:
        int(value, 16)
        return True
    except ValueError:
        return False


def _resolve_teacher_identity(args: argparse.Namespace) -> dict[str, Any]:
    local_teacher = Path(args.teacher_model).expanduser()
    if local_teacher.exists():
        if args.teacher_revision:
            raise ValueError("a local --teacher-model must use content identity, not --teacher-revision")
        expected_content_sha256 = require_sha256(
            args.teacher_content_sha256,
            "--teacher-content-sha256 for a local teacher",
        )
        identity = inspect_local_hf_model(local_teacher)
        if identity.weight_content_sha256 != expected_content_sha256:
            raise ValueError(
                "local teacher weights do not match --teacher-content-sha256: "
                f"{identity.weight_content_sha256} != {expected_content_sha256}"
            )
        teacher_identity = generation_teacher_identity(args.teacher_model)
    else:
        if args.teacher_content_sha256:
            raise ValueError("a remote --teacher-model must be pinned by revision, not an unverifiable content hash")
        teacher_identity = generation_teacher_identity(args.teacher_model, args.teacher_revision)

    source_values = (
        args.teacher_source_repo,
        args.teacher_source_revision,
        args.teacher_source_subfolder,
    )
    if any(value is not None for value in source_values):
        if not all(source_values):
            raise ValueError(
                "--teacher-source-repo, --teacher-source-revision, and --teacher-source-subfolder "
                "must be supplied together"
            )
        if not _is_hex_revision(args.teacher_source_revision):
            raise ValueError("--teacher-source-revision must be an immutable 40/64-hex revision")
        teacher_identity = {
            **teacher_identity,
            "source_repo": args.teacher_source_repo,
            "source_revision": args.teacher_source_revision,
            "source_subfolder": args.teacher_source_subfolder.strip("/"),
        }

    tokenizer_is_local = Path(args.tokenizer_model or args.teacher_model).exists()
    tokenizer_revision = args.tokenizer_revision or args.teacher_revision
    if not tokenizer_is_local and not _is_hex_revision(tokenizer_revision):
        raise ValueError("a remote tokenizer requires an immutable 40/64-hex --tokenizer-revision")
    return teacher_identity


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def ensure_run_config(output_dir: Path, run_config: Mapping[str, Any]) -> None:
    config_path = output_dir / "run_config.json"
    with file_lock(output_dir / ".run_config.lock"):
        if config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing.get("generation_config_sha256") != run_config["generation_config_sha256"]:
                raise RuntimeError(
                    f"{output_dir} already contains a different generation configuration: "
                    f"{existing.get('generation_config_sha256')} != {run_config['generation_config_sha256']}"
                )
            if existing.get("semantic_config") != run_config["semantic_config"]:
                raise RuntimeError(f"{config_path} has a hash collision or non-canonical semantic configuration")
            return
        if any(output_dir.glob("*.parquet")) or any(output_dir.glob("*.manifest.json")):
            raise RuntimeError(f"refusing to adopt existing shards without {config_path}")
        atomic_write_json(config_path, run_config)


def _load_grader() -> Any:
    import math_verify.grader  # noqa: F401 -- fail before generation if the dependency is absent

    from verl.utils.reward_score import math_verify as grader

    os.environ["VERL_MATH_VERIFY_STRICT_BOXED"] = "1"
    return grader


def _build_run_config(
    args: argparse.Namespace,
    *,
    source_sha256: str,
    source_rows: int,
    prompts: Sequence[SourcePrompt],
    tokenizer_sha256: str,
    tokenizer_vocab_size: int,
    chat_template_sha256: str,
    generator_commit: str,
    repository_dirty: bool,
    source_hash: str,
    versions: Mapping[str, Any],
    teacher_identity: Mapping[str, Any],
) -> dict[str, Any]:
    tokenizer_model = args.tokenizer_model or args.teacher_model
    tokenizer_revision = args.tokenizer_revision or args.teacher_revision
    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "sampling_top_k": args.sampling_top_k,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_response_tokens": args.max_response_tokens,
        "max_model_len": args.max_model_len,
        "stop": list(DEFAULT_STOP_STRINGS),
        "include_stop_str_in_output": False,
        "skip_special_tokens": False,
        "logprobs": TOPK_WIDTH,
    }
    engine = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "distributed_executor_backend": args.distributed_executor_backend,
        "mm_encoder_attn_backend": args.mm_encoder_attn_backend,
    }
    roster = [
        {
            "prompt_index": prompt.prompt_index,
            "source_uid": prompt.source_uid,
            "question_sha256": prompt.question_sha256,
            "gold_answer": prompt.gold_answer,
        }
        for prompt in prompts
    ]
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "direction": args.direction,
        "split": args.split,
        "source_dataset": args.source_dataset or Path(args.input_parquet).name,
        "source_dataset_sha256": source_sha256,
        "prompt_roster_sha256": hash_json(roster),
        "source_row_count": source_rows,
        "unique_question_count": len(prompts),
        "samples_per_question": args.samples_per_question,
        "topk_width": TOPK_WIDTH,
        "global_seed": args.global_seed,
        "prompts_per_shard": args.prompts_per_shard,
        "row_group_rows": args.row_group_rows,
        "total_shards": math.ceil(len(prompts) / args.prompts_per_shard),
        "teacher": dict(teacher_identity),
        "tokenizer": {
            "model": tokenizer_model,
            "revision": tokenizer_revision,
            "sha256": tokenizer_sha256,
            "vocab_size": tokenizer_vocab_size,
        },
        "chat_template": {
            "path": str(Path(args.chat_template).resolve()),
            "sha256": chat_template_sha256,
        },
        "sampling": sampling,
        "engine": engine,
        "generator": {
            "commit": generator_commit,
            "repository_dirty": repository_dirty,
            "source_sha256": source_hash,
        },
        "environment_versions": dict(versions),
    }
    return {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generation_config_sha256": hash_json(semantic),
        "semantic_config": semantic,
        "input_parquet": str(Path(args.input_parquet).resolve()),
        "created_at": utc_now(),
    }


def _make_llm(args: argparse.Namespace) -> tuple[Any, Any, Any, str]:
    from vllm import LLM, SamplingParams
    from vllm import __version__ as vllm_version

    revision = args.teacher_revision
    tokenizer_revision = args.tokenizer_revision or revision
    llm_kwargs: dict[str, Any] = {
        "model": args.teacher_model,
        "tokenizer": args.tokenizer_model or args.teacher_model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_logprobs": TOPK_WIDTH,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "seed": args.global_seed,
    }
    if revision and not Path(args.teacher_model).exists():
        llm_kwargs["revision"] = revision
    if tokenizer_revision and not Path(args.tokenizer_model or args.teacher_model).exists():
        llm_kwargs["tokenizer_revision"] = tokenizer_revision
    if args.distributed_executor_backend:
        llm_kwargs["distributed_executor_backend"] = args.distributed_executor_backend
    if args.mm_encoder_attn_backend:
        llm_kwargs["mm_encoder_attn_backend"] = args.mm_encoder_attn_backend
    return LLM(**llm_kwargs), SamplingParams, llm_kwargs, vllm_version


def _sampling_params(SamplingParams: Any, sampling: Mapping[str, Any], seed: int) -> Any:
    return SamplingParams(
        n=1,
        temperature=float(sampling["temperature"]),
        top_p=float(sampling["top_p"]),
        top_k=int(sampling["sampling_top_k"]),
        max_tokens=int(sampling["max_response_tokens"]),
        stop=list(sampling["stop"]),
        include_stop_str_in_output=bool(sampling["include_stop_str_in_output"]),
        skip_special_tokens=bool(sampling["skip_special_tokens"]),
        logprobs=int(sampling["logprobs"]),
        seed=seed,
    )


def _shard_filename(split: str, shard_id: int) -> str:
    return f"traces-{split}-{shard_id:06d}.parquet"


def _valid_completed_shard(
    parquet_path: Path, run_config: Mapping[str, Any], tokenizer: Any, expected_shard_id: int
) -> bool:
    try:
        manifest, validation = validate_shard_bundle(
            parquet_path,
            run_config=run_config,
            decoder=lambda ids: normalized_decode(tokenizer, ids),
        )
        if int(manifest["shard_id"]) != expected_shard_id:
            raise TraceValidationError("sidecar shard ID does not match its filename")
        if validation.stats["empty_response_count"]:
            raise TraceValidationError("completed shard contains an empty response")
        return True
    except (OSError, KeyError, TypeError, ValueError, TraceValidationError) as error:
        if parquet_path.exists() or parquet_manifest_path(parquet_path).exists():
            print(f"[resume] shard {expected_shard_id} is not valid and will be regenerated: {error}", flush=True)
        return False


def _write_validated_shard(
    records: Sequence[Mapping[str, Any]],
    *,
    parquet_path: Path,
    shard_id: int,
    prompt_start: int,
    prompt_end: int,
    run_config: Mapping[str, Any],
    tokenizer: Any,
    row_group_rows: int,
) -> None:
    temporary = write_parquet_temporary(records, parquet_path, row_group_size=row_group_rows)
    try:
        semantic = run_config["semantic_config"]
        validation = validate_parquet_shard(
            temporary,
            decoder=lambda ids: normalized_decode(tokenizer, ids),
            expected_config_sha256=run_config["generation_config_sha256"],
            expected_direction=semantic["direction"],
            expected_split=semantic["split"],
            expected_shard_id=shard_id,
            max_prompt_tokens=int(semantic["sampling"]["max_prompt_tokens"]),
            max_response_tokens=int(semantic["sampling"]["max_response_tokens"]),
        )
        if validation.stats["empty_response_count"]:
            raise TraceValidationError("refusing to publish a shard containing an empty response")
        parquet_file = pq.ParquetFile(temporary)
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "schema_version": SCHEMA_VERSION,
            "generation_config_sha256": run_config["generation_config_sha256"],
            "direction": semantic["direction"],
            "split": semantic["split"],
            "shard_id": shard_id,
            "prompt_start": prompt_start,
            "prompt_end": prompt_end,
            "row_count": validation.stats["row_count"],
            "source_uid_count": len(validation.source_uid_to_question),
            "source_uids_sha256": hash_json(sorted(validation.source_uid_to_question)),
            "trace_ids_sha256": hash_json(sorted(validation.trace_ids)),
            "parquet_file": parquet_path.name,
            "parquet_sha256": sha256_file(temporary),
            "parquet_size_bytes": temporary.stat().st_size,
            "parquet_row_groups": parquet_file.metadata.num_row_groups,
            "row_group_rows": row_group_rows,
            "stats": validation.stats,
            "created_at": utc_now(),
        }
        publish_parquet_temporary(temporary, parquet_path)
        atomic_write_json(parquet_manifest_path(parquet_path), manifest)
    finally:
        temporary.unlink(missing_ok=True)


def run_generation(args: argparse.Namespace) -> None:
    teacher_identity = _resolve_teacher_identity(args)
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError("--worker-id must be in [0, --num-workers)")
    if args.samples_per_question <= 0 or args.prompts_per_shard <= 0 or args.row_group_rows <= 0:
        raise ValueError("sample/shard/row-group sizes must be positive")
    if args.max_num_seqs <= 0 or args.max_num_batched_tokens <= 0:
        raise ValueError("vLLM concurrency limits must be positive")
    if args.max_prompt_tokens + args.max_response_tokens > args.max_model_len:
        raise ValueError("max prompt + response tokens exceeds --max-model-len")
    if args.temperature != 1.0 or args.top_p != 1.0 or args.sampling_top_k != -1:
        raise ValueError("the registered experiment requires temperature=1, top_p=1, and sampling top_k disabled")

    input_path = Path(args.input_parquet)
    template_path = Path(args.chat_template)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mirror = None
    if args.s3_mirror_uri:
        from gemma4_trace_s3 import TraceS3Mirror

        mirror = TraceS3Mirror(args.s3_mirror_uri)
        # Two-GPU fallback runs use independent data-parallel processes with a
        # shared output directory.  Serialize S3 restoration so they cannot
        # download/replace the same shard concurrently at startup.
        with file_lock(output_dir / ".s3_restore.lock"):
            restored = mirror.restore_directory(output_dir)
        print(f"[s3] restored {restored} files before generation", flush=True)
    source_sha256 = sha256_file(input_path)
    chat_template = template_path.read_text(encoding="utf-8")
    template_sha256 = sha256_text(chat_template)
    prompts, source_rows = load_unique_source_prompts(input_path, max_questions=args.max_questions)
    if not prompts:
        raise ValueError("source parquet produced no unique questions")
    print(json.dumps(vars(args), indent=2, sort_keys=True, default=str), flush=True)
    print(f"[source] {source_rows} rows -> {len(prompts)} unique questions; sha256={source_sha256}", flush=True)

    llm, SamplingParams, llm_kwargs, vllm_version = _make_llm(args)
    tokenizer = llm.get_tokenizer()
    tokenizer.chat_template = chat_template
    tokenizer_sha256, tokenizer_vocab_size = tokenizer_fingerprint(tokenizer)
    repo_root = Path(__file__).resolve().parents[2]
    generator_commit, repository_dirty = repository_state(repo_root)
    versions = environment_versions(vllm_version)
    run_config = _build_run_config(
        args,
        source_sha256=source_sha256,
        source_rows=source_rows,
        prompts=prompts,
        tokenizer_sha256=tokenizer_sha256,
        tokenizer_vocab_size=tokenizer_vocab_size,
        chat_template_sha256=template_sha256,
        generator_commit=generator_commit,
        repository_dirty=repository_dirty,
        source_hash=generator_source_hash(),
        versions=versions,
        teacher_identity=teacher_identity,
    )
    run_config["runtime"] = {"llm_kwargs": llm_kwargs, "creator_worker_id": args.worker_id}
    ensure_run_config(output_dir, run_config)
    if mirror is not None:
        mirror.upload_file(output_dir / "run_config.json", root=output_dir)
    print(json.dumps(run_config, indent=2, sort_keys=True), flush=True)
    grader = _load_grader()

    semantic = run_config["semantic_config"]
    sampling = semantic["sampling"]
    generated_shards = 0
    total_shards = int(semantic["total_shards"])
    pending_uploads: list[Future[Any]] = []
    executor_context = ThreadPoolExecutor(max_workers=1) if mirror is not None else nullcontext(None)
    with executor_context as upload_executor:
        for shard_id in range(total_shards):
            if shard_id % args.num_workers != args.worker_id:
                continue
            if args.max_shards > 0 and generated_shards >= args.max_shards:
                break
            parquet_path = output_dir / _shard_filename(args.split, shard_id)
            with file_lock(output_dir / f".{parquet_path.name}.lock"):
                if _valid_completed_shard(parquet_path, run_config, tokenizer, shard_id):
                    print(f"[resume] shard {shard_id}/{total_shards} is complete", flush=True)
                    continue
                prompt_start = shard_id * args.prompts_per_shard
                prompt_end = min(prompt_start + args.prompts_per_shard, len(prompts))
                shard_prompts = prompts[prompt_start:prompt_end]
                prepared: list[PreparedRequest] = []
                prompt_requests: list[dict[str, list[int]]] = []
                sampling_params: list[Any] = []
                for source in shard_prompts:
                    prompt_ids = _render_prompt(tokenizer, source.messages, chat_template)
                    if len(prompt_ids) > args.max_prompt_tokens:
                        raise ValueError(
                            f"source UID {source.source_uid} renders to {len(prompt_ids)} tokens, above "
                            f"the {args.max_prompt_tokens}-token contract; prompts are never truncated"
                        )
                    for sample_index in range(args.samples_per_question):
                        seed = derive_sampling_seed(args.global_seed, args.split, source.source_uid, sample_index)
                        request = PreparedRequest(source, sample_index, seed, prompt_ids)
                        prepared.append(request)
                        prompt_requests.append({"prompt_token_ids": prompt_ids})
                        sampling_params.append(_sampling_params(SamplingParams, sampling, seed))
                print(
                    f"[generate] shard {shard_id}/{total_shards}: prompts [{prompt_start}:{prompt_end}], "
                    f"requests={len(prompt_requests)}",
                    flush=True,
                )
                outputs = llm.generate(prompt_requests, sampling_params, use_tqdm=True)
                if len(outputs) != len(prepared):
                    raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(prepared)} requests")
                timestamp = utc_now()
                records: list[dict[str, Any]] = []
                for row_within_shard, (request, output) in enumerate(zip(prepared, outputs, strict=True)):
                    if len(output.outputs) != 1:
                        raise RuntimeError("vLLM returned an unexpected number of completions")
                    response_text = normalized_decode(tokenizer, list(output.outputs[0].token_ids))
                    strict_grade = float(grader.compute_score(response_text, request.source.gold_answer))
                    strict_prediction = str(grader.extract_prediction(response_text))
                    records.append(
                        build_trace_record(
                            request=request,
                            output=output,
                            shard_id=shard_id,
                            row_within_shard=row_within_shard,
                            run_config=run_config,
                            tokenizer=tokenizer,
                            strict_grade=strict_grade,
                            strict_prediction=strict_prediction,
                            generation_timestamp=timestamp,
                        )
                    )
                _write_validated_shard(
                    records,
                    parquet_path=parquet_path,
                    shard_id=shard_id,
                    prompt_start=prompt_start,
                    prompt_end=prompt_end,
                    run_config=run_config,
                    tokenizer=tokenizer,
                    row_group_rows=args.row_group_rows,
                )
                generated_shards += 1
                print(f"[saved] {parquet_path} ({len(records)} rows)", flush=True)
                if mirror is not None and upload_executor is not None:
                    pending_uploads.append(upload_executor.submit(mirror.upload_shard, parquet_path, root=output_dir))
                    if len(pending_uploads) >= 2:
                        pending_uploads.pop(0).result()
        for pending_upload in pending_uploads:
            pending_upload.result()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-revision", default=None)
    parser.add_argument("--teacher-content-sha256", default=None)
    parser.add_argument("--teacher-source-repo", default=None)
    parser.add_argument("--teacher-source-revision", default=None)
    parser.add_argument("--teacher-source-subfolder", default=None)
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--source-dataset", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--direction", choices=DIRECTIONS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--chat-template", default=str(DEFAULT_CHAT_TEMPLATE))
    parser.add_argument("--samples-per-question", type=int, default=5)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sampling-top-k", type=int, default=-1)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-response-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--prompts-per-shard", type=int, default=8)
    parser.add_argument("--row-group-rows", type=int, default=2)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=-1)
    parser.add_argument("--max-shards", type=int, default=-1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--distributed-executor-backend", default=None)
    parser.add_argument("--mm-encoder-attn-backend", default=None)
    parser.add_argument("--s3-mirror-uri", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_generation(parse_args(argv))
    except TraceValidationError as error:
        # A deterministic trace-contract violation (for example, an empty
        # completion produced with a fixed seed) cannot be repaired by
        # restarting the worker with the same configuration.  Give the
        # supervisor a distinct status so it can fail closed immediately.
        print(f"ERROR: {error}", file=sys.stderr)
        return 3
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
