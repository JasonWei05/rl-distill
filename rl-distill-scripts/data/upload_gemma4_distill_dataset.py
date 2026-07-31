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

"""Validate and atomically upload a complete Gemma 4 trace dataset.

Only a validator-produced dataset index containing complete ``train`` and
``validation`` splits is accepted.  Every Parquet row, shard manifest, run
configuration, and index hash is checked before any Hugging Face mutation.
The exact validated bundle is copied to content-verified staging files, then
committed only to an otherwise empty destination branch with the observed
parent commit pinned. Uploads are private by default. Public visibility
requires an explicit ``--public`` opt-in and is verified before and after the
commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gemma4_distill_trace_schema as trace_schema
import preflight_gemma4_topk_distill as preflight
import validate_gemma4_distill_traces as dataset_validator

_INDEX_KEYS = frozenset(
    {
        "manifest_version",
        "schema_version",
        "created_at",
        "experiment_sha256",
        "direction",
        "topk_width",
        "recommended_training_topk_validation_tolerance",
        "samples_per_question",
        "decode_check_performed",
        "teacher",
        "tokenizer",
        "chat_template",
        "sampling",
        "total_rows",
        "total_response_tokens",
        "cross_split_question_text_overlap_count",
        "cross_split_question_text_overlap_sha256s",
        "splits",
        "dataset_index_sha256",
    }
)
_SPLIT_KEYS = frozenset(
    {
        "source_dataset",
        "source_dataset_sha256",
        "generation_config_sha256",
        "run_config_path",
        "run_config_sha256",
        "question_count",
        "row_count",
        "complete",
        "missing_shard_ids",
        "stats",
        "parquet_files",
        "shards",
    }
)
_SHARD_KEYS = frozenset(
    {
        "shard_id",
        "path",
        "manifest_path",
        "sha256",
        "size_bytes",
        "rows",
        "row_groups",
        "stats",
    }
)
_DIRECTIONS = frozenset({"e4b_rl100_to_e2b", "e2b_base_to_e4b"})
_ALLOWED_HUB_INFRA_FILES = frozenset({".gitattributes"})
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


class DatasetUploadError(ValueError):
    """Raised when validation or upload cannot be completed safely."""


@dataclass(frozen=True)
class UploadFile:
    local_path: Path
    path_in_repo: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ValidatedUploadBundle:
    dataset_root: Path
    index_path: Path
    dataset_index_sha256: str
    experiment_sha256: str
    direction: str
    total_rows: int
    total_response_tokens: int
    files: tuple[UploadFile, ...]


@dataclass(frozen=True)
class UploadResult:
    repo_id: str
    requested_revision: str
    commit_oid: str
    dataset_index_sha256: str
    file_count: int
    private: bool

    def lines(self) -> list[str]:
        return [
            f"REPO_ID={self.repo_id}",
            f"REQUESTED_REVISION={self.requested_revision}",
            f"COMMIT_OID={self.commit_oid}",
            f"DATASET_INDEX_SHA256={self.dataset_index_sha256}",
            f"FILES_UPLOADED={self.file_count}",
            f"PRIVATE={'true' if self.private else 'false'}",
        ]


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetUploadError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise DatasetUploadError(f"{description} {path} must contain a JSON object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], description: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise DatasetUploadError(f"{description} keys do not match the schema: missing={missing}, extra={extra}")


def _resolve_dataset_index(dataset_path: str | Path) -> tuple[Path, Path]:
    supplied = Path(dataset_path).expanduser()
    if supplied.is_dir():
        dataset_root = supplied.resolve()
        candidate = supplied / "dataset_index.json"
        if not candidate.is_file():
            raise DatasetUploadError(
                f"dataset directory must contain dataset_index.json; partial split/smoke directory refused: {supplied}"
            )
    elif supplied.is_file():
        if supplied.suffix != ".json" or supplied.name.endswith(".manifest.json"):
            raise DatasetUploadError(
                "dataset path must be a validator-produced dataset index JSON, "
                "not a direct shard/manifest smoke artifact"
            )
        candidate = supplied
        dataset_root = supplied.resolve().parent
    else:
        raise DatasetUploadError(f"dataset path does not exist: {supplied}")
    try:
        index_path = candidate.resolve(strict=True)
    except OSError as error:
        raise DatasetUploadError(f"cannot resolve dataset index {candidate}: {error}") from error
    if not index_path.is_relative_to(dataset_root):
        raise DatasetUploadError(f"dataset index escapes the dataset directory: {candidate}")
    return dataset_root, index_path


def _bundle_relative_path(path: Path, dataset_root: Path, description: str) -> str:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise DatasetUploadError(f"cannot resolve {description} {path}: {error}") from error
    try:
        relative = resolved.relative_to(dataset_root)
    except ValueError as error:
        raise DatasetUploadError(f"{description} escapes the dataset directory: {path}") from error
    if not resolved.is_file():
        raise DatasetUploadError(f"{description} is not a regular file: {path}")
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise DatasetUploadError(f"{description} has an unsafe repository path: {relative}")
    return relative.as_posix()


def _resolve_index_reference(index_path: Path, value: Any, field_name: str) -> Path:
    try:
        return preflight._resolve_index_path(index_path, value, field_name)
    except preflight.PreflightError as error:
        raise DatasetUploadError(str(error)) from error


def _validate_created_at(value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise DatasetUploadError("dataset index created_at must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise DatasetUploadError("dataset index created_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise DatasetUploadError("dataset index created_at must include a timezone")


def _assert_equal(actual: Any, expected: Any, description: str) -> None:
    if actual != expected:
        raise DatasetUploadError(f"{description} does not match a fresh full validation")


def _compare_regenerated_index(
    *,
    original: Mapping[str, Any],
    original_path: Path,
    regenerated: Mapping[str, Any],
    regenerated_path: Path,
) -> None:
    ignored_top_level = {
        "created_at",
        "decode_check_performed",
        "experiment_sha256",
        "dataset_index_sha256",
        "splits",
    }
    for key in sorted(_INDEX_KEYS.difference(ignored_top_level)):
        _assert_equal(original[key], regenerated[key], f"dataset index {key}")

    original_splits = original["splits"]
    regenerated_splits = regenerated["splits"]
    for split in ("train", "validation"):
        original_split = original_splits[split]
        regenerated_split = regenerated_splits[split]
        _require_exact_keys(original_split, _SPLIT_KEYS, f"splits.{split}")
        ignored_split = {"run_config_path", "parquet_files", "shards"}
        for key in sorted(_SPLIT_KEYS.difference(ignored_split)):
            _assert_equal(original_split[key], regenerated_split[key], f"splits.{split}.{key}")

        original_run_config = _resolve_index_reference(
            original_path, original_split["run_config_path"], f"splits.{split}.run_config_path"
        )
        regenerated_run_config = _resolve_index_reference(
            regenerated_path, regenerated_split["run_config_path"], f"regenerated splits.{split}.run_config_path"
        )
        _assert_equal(original_run_config, regenerated_run_config, f"splits.{split}.run_config_path")

        original_files = tuple(
            _resolve_index_reference(original_path, value, f"splits.{split}.parquet_files")
            for value in original_split["parquet_files"]
        )
        regenerated_files = tuple(
            _resolve_index_reference(
                regenerated_path,
                value,
                f"regenerated splits.{split}.parquet_files",
            )
            for value in regenerated_split["parquet_files"]
        )
        _assert_equal(original_files, regenerated_files, f"splits.{split}.parquet_files")

        if len(original_split["shards"]) != len(regenerated_split["shards"]):
            raise DatasetUploadError(f"splits.{split}.shards count does not match a fresh full validation")
        for position, (original_shard, regenerated_shard) in enumerate(
            zip(original_split["shards"], regenerated_split["shards"], strict=True)
        ):
            if not isinstance(original_shard, Mapping):
                raise DatasetUploadError(f"splits.{split}.shards[{position}] must be an object")
            _require_exact_keys(original_shard, _SHARD_KEYS, f"splits.{split}.shards[{position}]")
            ignored_shard = {"path", "manifest_path"}
            for key in sorted(_SHARD_KEYS.difference(ignored_shard)):
                _assert_equal(
                    original_shard[key],
                    regenerated_shard[key],
                    f"splits.{split}.shards[{position}].{key}",
                )
            original_parquet = _resolve_index_reference(
                original_path, original_shard["path"], f"splits.{split}.shards[{position}].path"
            )
            regenerated_parquet = _resolve_index_reference(
                regenerated_path,
                regenerated_shard["path"],
                f"regenerated splits.{split}.shards[{position}].path",
            )
            _assert_equal(original_parquet, regenerated_parquet, f"splits.{split}.shards[{position}].path")
            original_manifest = _resolve_index_reference(
                original_path,
                original_shard["manifest_path"],
                f"splits.{split}.shards[{position}].manifest_path",
            )
            expected_manifest = trace_schema.parquet_manifest_path(regenerated_parquet).resolve()
            _assert_equal(
                original_manifest,
                expected_manifest,
                f"splits.{split}.shards[{position}].manifest_path",
            )


def _validate_teacher_identity(teacher: Any) -> None:
    if not isinstance(teacher, Mapping):
        raise DatasetUploadError("dataset teacher identity must be an object")
    try:
        preflight._require_sha256(teacher.get("model_identity_sha256"), "teacher.model_identity_sha256")
        if teacher.get("content_sha256") is None:
            revision = preflight._require_immutable_revision(teacher.get("revision"), "teacher.revision")
            if teacher.get("content_sha256_kind") is not None:
                raise DatasetUploadError("remote teacher content_sha256_kind must be null")
            expected_identity = trace_schema.hash_json({"model": teacher.get("model"), "revision": revision})
            if teacher.get("model_identity_sha256") != expected_identity:
                raise DatasetUploadError("remote teacher model_identity_sha256 is inconsistent")
        else:
            preflight._require_sha256(teacher.get("content_sha256"), "teacher.content_sha256")
            if not isinstance(teacher.get("content_sha256_kind"), str) or not teacher["content_sha256_kind"]:
                raise DatasetUploadError("local teacher content_sha256_kind must be a non-empty string")
            if teacher.get("revision") is not None:
                raise DatasetUploadError("local teacher revision must be null")
    except preflight.PreflightError as error:
        raise DatasetUploadError(str(error)) from error


def validate_upload_bundle(
    dataset_path: str | Path,
    *,
    allow_question_overlap: bool = False,
    _expected_questions: Mapping[str, int] | None = None,
    _expected_samples_per_question: int | Mapping[str, int] = preflight.EXPECTED_SAMPLES_PER_QUESTION,
) -> ValidatedUploadBundle:
    """Perform a full, read-only validation and return the exact upload set.

    The expected question and sample counts are explicit upload contracts. The
    CLI defaults preserve the original 9,723x5 train and 200x5 validation roster.
    """

    dataset_root, index_path = _resolve_dataset_index(dataset_path)
    index = _load_json(index_path, "dataset index")
    _require_exact_keys(index, _INDEX_KEYS, "dataset index")
    _validate_created_at(index["created_at"])
    try:
        dataset_index_sha256 = preflight._verify_index_self_hash(index)
    except preflight.PreflightError as error:
        raise DatasetUploadError(str(error)) from error
    if index["manifest_version"] != trace_schema.MANIFEST_VERSION:
        raise DatasetUploadError("dataset index has an unsupported manifest version")
    if index["schema_version"] != trace_schema.SCHEMA_VERSION:
        raise DatasetUploadError("dataset index has an unsupported trace schema version")
    if index["decode_check_performed"] is not True:
        raise DatasetUploadError("dataset index must record a completed tokenizer decode check")
    if index["direction"] not in _DIRECTIONS:
        raise DatasetUploadError(f"unsupported dataset direction: {index['direction']!r}")
    if index["topk_width"] != trace_schema.TOPK_WIDTH:
        raise DatasetUploadError(f"dataset top-k width must be exactly {trace_schema.TOPK_WIDTH}")
    expected_samples = preflight._normalize_split_counts(
        _expected_samples_per_question, field_name="expected samples_per_question"
    )
    index_samples = preflight._normalize_split_counts(
        index["samples_per_question"], field_name="dataset samples_per_question"
    )
    if index_samples != expected_samples:
        raise DatasetUploadError("dataset samples_per_question does not match the upload contract")
    if index["recommended_training_topk_validation_tolerance"] != trace_schema.FP16_TOPK_MASS_TOLERANCE:
        raise DatasetUploadError("dataset top-k validation tolerance does not match the trace schema")
    if not isinstance(index["splits"], Mapping) or set(index["splits"]) != {"train", "validation"}:
        raise DatasetUploadError("dataset index must contain exactly complete train and validation splits")
    _validate_teacher_identity(index["teacher"])

    split_dirs: dict[str, Path] = {}
    for split in ("train", "validation"):
        split_index = index["splits"][split]
        if not isinstance(split_index, Mapping):
            raise DatasetUploadError(f"splits.{split} must be an object")
        _require_exact_keys(split_index, _SPLIT_KEYS, f"splits.{split}")
        run_config_path = _resolve_index_reference(
            index_path, split_index["run_config_path"], f"splits.{split}.run_config_path"
        )
        _bundle_relative_path(run_config_path, dataset_root, f"{split} run config")
        if run_config_path.name != "run_config.json":
            raise DatasetUploadError(f"{split} run config must be named run_config.json")
        split_dirs[split] = run_config_path.parent

    expected_questions = dict(_expected_questions or preflight.EXPECTED_QUESTIONS)
    if set(expected_questions) != {"train", "validation"}:
        raise DatasetUploadError("expected question counts must contain exactly train and validation")
    with tempfile.TemporaryDirectory(prefix="gemma4-trace-upload-validation-") as temporary_directory:
        regenerated_path = Path(temporary_directory) / "dataset_index.json"
        try:
            regenerated = dataset_validator.validate_dataset(
                split_dirs,
                output_index=regenerated_path,
                decoder=None,
                expected_questions=expected_questions,
                expected_samples_per_question=expected_samples,
                allow_incomplete=False,
                allow_empty_responses=False,
                fail_on_question_overlap=not allow_question_overlap,
            )
        except (OSError, RuntimeError, ValueError, trace_schema.TraceValidationError) as error:
            raise DatasetUploadError(f"full trace validation failed: {error}") from error
        _compare_regenerated_index(
            original=index,
            original_path=index_path,
            regenerated=regenerated,
            regenerated_path=regenerated_path,
        )

    run_configs = {split: dataset_validator.load_run_config(split_dir) for split, split_dir in split_dirs.items()}
    common_configs = {
        split: preflight._common_generation_config(config["semantic_config"]) for split, config in run_configs.items()
    }
    if common_configs["train"] != common_configs["validation"]:
        raise DatasetUploadError("train and validation use mixed generation/teacher configurations")
    common = common_configs["train"]
    try:
        preflight._verify_generation_contract(common)
    except preflight.PreflightError as error:
        raise DatasetUploadError(str(error)) from error
    if index["experiment_sha256"] not in preflight._experiment_hash_candidates(common, expected_samples):
        raise DatasetUploadError("dataset experiment_sha256 does not match the split run configurations")

    upload_files: dict[str, Path] = {}

    def add_file(path: Path, description: str) -> None:
        path_in_repo = _bundle_relative_path(path, dataset_root, description)
        prior = upload_files.setdefault(path_in_repo, path.resolve())
        if prior != path.resolve():
            raise DatasetUploadError(f"multiple files map to repository path {path_in_repo!r}")

    add_file(index_path, "dataset index")
    seen_local_paths: set[Path] = {index_path}
    for split in ("train", "validation"):
        split_index = index["splits"][split]
        run_config_path = split_dirs[split] / "run_config.json"
        add_file(run_config_path, f"{split} run config")
        seen_local_paths.add(run_config_path.resolve())
        for position, shard in enumerate(split_index["shards"]):
            parquet_path = _resolve_index_reference(
                index_path, shard["path"], f"splits.{split}.shards[{position}].path"
            )
            manifest_path = _resolve_index_reference(
                index_path, shard["manifest_path"], f"splits.{split}.shards[{position}].manifest_path"
            )
            expected_manifest = trace_schema.parquet_manifest_path(parquet_path).resolve()
            if manifest_path != expected_manifest:
                raise DatasetUploadError(f"{split} shard {position} manifest path does not match its Parquet path")
            for path, description in (
                (parquet_path, f"{split} shard {position}"),
                (manifest_path, f"{split} shard {position} manifest"),
            ):
                resolved = path.resolve()
                if resolved in seen_local_paths:
                    raise DatasetUploadError(f"dataset bundle references a file more than once: {path}")
                seen_local_paths.add(resolved)
                add_file(path, description)

    files = tuple(
        UploadFile(
            local_path=path,
            path_in_repo=path_in_repo,
            size_bytes=path.stat().st_size,
            sha256=trace_schema.sha256_file(path),
        )
        for path_in_repo, path in sorted(upload_files.items())
    )
    return ValidatedUploadBundle(
        dataset_root=dataset_root,
        index_path=index_path,
        dataset_index_sha256=dataset_index_sha256,
        experiment_sha256=index["experiment_sha256"],
        direction=index["direction"],
        total_rows=index["total_rows"],
        total_response_tokens=index["total_response_tokens"],
        files=files,
    )


def _validate_repo_id(repo_id: str) -> str:
    repo_id = repo_id.strip()
    if repo_id.count("/") != 1:
        raise DatasetUploadError("repo_id must be explicit namespace/name")
    try:
        from huggingface_hub.utils import HFValidationError, validate_repo_id

        validate_repo_id(repo_id)
    except HFValidationError as error:
        raise DatasetUploadError(f"invalid Hugging Face repo_id: {error}") from error
    return repo_id


def _validate_revision(revision: str) -> str:
    revision = revision.strip()
    if (
        not _REVISION_RE.fullmatch(revision)
        or ".." in revision
        or "//" in revision
        or "@{" in revision
        or revision.endswith((".", "/"))
    ):
        raise DatasetUploadError(f"invalid target revision: {revision!r}")
    return revision


def _redact_secret(message: str, secret: str | None) -> str:
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return message


def _new_hf_api(token: str) -> Any:
    from huggingface_hub import HfApi

    return HfApi(token=token)


def _verify_upload_files_unchanged(files: Sequence[UploadFile]) -> None:
    for item in files:
        if not item.local_path.is_file():
            raise DatasetUploadError(f"validated upload file disappeared: {item.local_path}")
        if item.local_path.stat().st_size != item.size_bytes:
            raise DatasetUploadError(f"validated upload file size changed: {item.local_path}")
        if trace_schema.sha256_file(item.local_path) != item.sha256:
            raise DatasetUploadError(f"validated upload file content changed: {item.local_path}")


def _verify_bundle_unchanged(bundle: ValidatedUploadBundle) -> None:
    _verify_upload_files_unchanged(bundle.files)


@contextmanager
def _stage_validated_files(bundle: ValidatedUploadBundle):
    """Yield private, content-verified snapshots for the duration of one commit.

    ``CommitOperationAdd`` streams path contents after it is constructed.  The
    validated source tree can therefore change between the initial hash check
    and the actual upload unless the operation points at a disjoint snapshot.
    Staging beside the dataset keeps large Parquet copies on the same storage
    volume while the random, mode-0700 temporary directory keeps the paths
    private until the commit finishes.
    """

    _verify_bundle_unchanged(bundle)
    prefix = f".{bundle.dataset_root.name}-hf-upload-stage-"
    with tempfile.TemporaryDirectory(prefix=prefix, dir=bundle.dataset_root.parent) as temporary_directory:
        staging_root = Path(temporary_directory)
        staged_files: list[UploadFile] = []
        for item in bundle.files:
            relative = Path(item.path_in_repo)
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                raise DatasetUploadError(f"unsafe staged repository path: {item.path_in_repo!r}")
            staged_path = staging_root / relative
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(item.local_path, staged_path)
            except OSError as error:
                raise DatasetUploadError(f"cannot stage validated upload file {item.local_path}: {error}") from error
            if staged_path.stat().st_size != item.size_bytes:
                raise DatasetUploadError(f"staged upload file size mismatch: {item.path_in_repo}")
            staged_sha256 = trace_schema.sha256_file(staged_path)
            if staged_sha256 != item.sha256:
                raise DatasetUploadError(f"staged upload file content mismatch: {item.path_in_repo}")
            staged_files.append(
                UploadFile(
                    local_path=staged_path,
                    path_in_repo=item.path_in_repo,
                    size_bytes=item.size_bytes,
                    sha256=item.sha256,
                )
            )
        yield tuple(staged_files)


def _require_commit_oid(value: Any, description: str) -> str:
    if not isinstance(value, str) or len(value) not in (40, 64):
        raise DatasetUploadError(f"{description} must be an immutable 40/64-character commit OID")
    try:
        int(value, 16)
    except ValueError as error:
        raise DatasetUploadError(f"{description} must be hexadecimal") from error
    return value.lower()


def _git_blob_sha1(path: Path, size_bytes: int) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {size_bytes}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_lfs_sha256(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        digest = value.get("sha256")
    else:
        digest = getattr(value, "sha256", None)
    return digest if isinstance(digest, str) else None


def _verify_remote_file_content(
    api: Any,
    *,
    repo_id: str,
    revision: str,
    token: str,
    files: Sequence[UploadFile],
    expected_git_blob_sha1s: Mapping[str, str],
) -> None:
    remote_files: dict[str, Any] = {}
    for item in api.list_repo_tree(
        repo_id,
        repo_type="dataset",
        revision=revision,
        recursive=True,
        expand=True,
        token=token,
    ):
        path = getattr(item, "path", None)
        blob_id = getattr(item, "blob_id", None)
        if blob_id is None:
            continue
        if not isinstance(path, str) or not path:
            raise DatasetUploadError("Hugging Face returned a repository file without a valid path")
        if path in remote_files:
            raise DatasetUploadError(f"Hugging Face returned duplicate repository path {path!r}")
        remote_files[path] = item

    expected_paths = {item.path_in_repo for item in files}
    missing_paths = expected_paths.difference(remote_files)
    stale_paths = set(remote_files).difference(expected_paths, _ALLOWED_HUB_INFRA_FILES)
    if missing_paths or stale_paths:
        raise DatasetUploadError(
            "Hugging Face commit does not contain exactly the validated dataset files: "
            f"missing={sorted(missing_paths)[:8]}, stale={sorted(stale_paths)[:8]}"
        )

    for expected in files:
        remote = remote_files[expected.path_in_repo]
        remote_size = getattr(remote, "size", None)
        if remote_size != expected.size_bytes:
            raise DatasetUploadError(f"remote file size mismatch: {expected.path_in_repo}")
        lfs_sha256 = _remote_lfs_sha256(getattr(remote, "lfs", None))
        if lfs_sha256 is not None:
            if lfs_sha256.lower() != expected.sha256:
                raise DatasetUploadError(f"remote LFS SHA256 mismatch: {expected.path_in_repo}")
            continue
        remote_blob_id = getattr(remote, "blob_id", None)
        expected_blob_id = expected_git_blob_sha1s.get(expected.path_in_repo)
        if expected_blob_id is None:
            raise DatasetUploadError(f"missing pre-commit Git blob identity: {expected.path_in_repo}")
        if remote_blob_id != expected_blob_id:
            raise DatasetUploadError(f"remote Git blob SHA1 mismatch: {expected.path_in_repo}")


def upload_validated_bundle(
    bundle: ValidatedUploadBundle,
    *,
    repo_id: str,
    revision: str,
    token: str,
    private: bool = True,
    api: Any | None = None,
) -> UploadResult:
    repo_id = _validate_repo_id(repo_id)
    revision = _validate_revision(revision)
    if not isinstance(token, str) or not token:
        raise DatasetUploadError("a Hugging Face token must be supplied from the configured environment variable")
    if type(private) is not bool:
        raise DatasetUploadError("private must be a boolean")
    with _stage_validated_files(bundle) as staged_files:
        failure_message: str | None = None
        try:
            if api is None:
                api = _new_hf_api(token)
            repo_existed = api.repo_exists(repo_id, repo_type="dataset", token=token)
            if type(repo_existed) is not bool:
                raise DatasetUploadError(f"Hugging Face did not report whether repository {repo_id!r} exists")
            api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
            repo_info = api.repo_info(repo_id, repo_type="dataset", revision=revision, token=token)
            actual_private = getattr(repo_info, "private", None)
            if type(actual_private) is not bool:
                raise DatasetUploadError(f"Hugging Face did not report repository visibility for {repo_id!r}")
            if private and actual_private is not True:
                raise DatasetUploadError(f"refusing to upload because dataset repository {repo_id!r} is not private")
            if not private and actual_private is not False:
                existing = "existing " if repo_existed else "newly created "
                raise DatasetUploadError(
                    f"refusing public upload because {existing}dataset repository {repo_id!r} is private; "
                    "the generic uploader never changes repository visibility"
                )
            parent_commit = _require_commit_oid(
                getattr(repo_info, "sha", None),
                f"Hugging Face head for {repo_id!r} revision {revision!r}",
            )

            existing_paths = set(
                api.list_repo_files(
                    repo_id,
                    repo_type="dataset",
                    revision=revision,
                    token=token,
                )
            )
            existing_dataset_paths = existing_paths.difference(_ALLOWED_HUB_INFRA_FILES)
            if existing_dataset_paths:
                raise DatasetUploadError(
                    "refusing to overwrite an existing Hub dataset bundle; "
                    f"non-infrastructure files={sorted(existing_dataset_paths)[:8]}"
                )

            from huggingface_hub import CommitOperationAdd

            expected_git_blob_sha1s = {
                item.path_in_repo: _git_blob_sha1(item.local_path, item.size_bytes) for item in staged_files
            }
            operations = [
                CommitOperationAdd(path_in_repo=item.path_in_repo, path_or_fileobj=item.local_path)
                for item in staged_files
            ]
            for item, operation in zip(staged_files, operations, strict=True):
                operation_sha256 = operation.upload_info.sha256.hex()
                if operation.upload_info.size != item.size_bytes or operation_sha256 != item.sha256:
                    raise DatasetUploadError(f"Hugging Face upload operation changed staged bytes: {item.path_in_repo}")
            commit_info = api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                parent_commit=parent_commit,
                operations=operations,
                commit_message=f"Upload validated Gemma 4 traces {bundle.dataset_index_sha256[:12]}",
                commit_description=(
                    f"direction={bundle.direction}\n"
                    f"dataset_index_sha256={bundle.dataset_index_sha256}\n"
                    f"experiment_sha256={bundle.experiment_sha256}"
                ),
                token=token,
            )
            commit_oid = _require_commit_oid(
                getattr(commit_info, "oid", None),
                "Hugging Face returned commit OID",
            )
            _verify_upload_files_unchanged(staged_files)
            _verify_remote_file_content(
                api,
                repo_id=repo_id,
                revision=commit_oid,
                token=token,
                files=staged_files,
                expected_git_blob_sha1s=expected_git_blob_sha1s,
            )
            final_repo_info = api.repo_info(repo_id, repo_type="dataset", token=token)
            final_private = getattr(final_repo_info, "private", None)
            if final_private is not private:
                requested_visibility = "private" if private else "public"
                raise DatasetUploadError(
                    f"dataset repository {repo_id!r} is not {requested_visibility} after the upload commit"
                )
        except DatasetUploadError as error:
            failure_message = _redact_secret(str(error), token)
        except Exception as error:
            sanitized = _redact_secret(str(error), token)
            failure_message = f"Hugging Face dataset upload failed: {sanitized}"
        if failure_message is not None:
            # Raise after leaving the except suite so the raw provider
            # exception is not retained as __cause__/__context__ and cannot
            # leak the token through formatted tracebacks.
            raise DatasetUploadError(failure_message) from None
        return UploadResult(
            repo_id=repo_id,
            requested_revision=revision,
            commit_oid=commit_oid,
            dataset_index_sha256=bundle.dataset_index_sha256,
            file_count=len(bundle.files),
            private=private,
        )


def upload_dataset(
    dataset_path: str | Path,
    *,
    repo_id: str,
    revision: str,
    token: str,
    allow_question_overlap: bool = False,
    private: bool = True,
    expected_questions: Mapping[str, int] | None = None,
    expected_samples_per_question: int | Mapping[str, int] = preflight.EXPECTED_SAMPLES_PER_QUESTION,
    api: Any | None = None,
) -> UploadResult:
    bundle = validate_upload_bundle(
        dataset_path,
        allow_question_overlap=allow_question_overlap,
        _expected_questions=expected_questions,
        _expected_samples_per_question=expected_samples_per_question,
    )
    return upload_validated_bundle(
        bundle,
        repo_id=repo_id,
        revision=revision,
        token=token,
        private=private,
        api=api,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Completed dataset root containing dataset_index.json, or the dataset index JSON itself.",
    )
    parser.add_argument("--repo-id", required=True, help="Explicit HF dataset repo in namespace/name form.")
    parser.add_argument("--revision", default="main", help="Existing target branch/revision (default: main).")
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing the HF token; token values are never accepted as CLI arguments.",
    )
    parser.add_argument("--allow-question-overlap", action="store_true")
    parser.add_argument(
        "--public",
        action="store_true",
        help="Explicitly make and verify the dataset repository public. The default is private.",
    )
    parser.add_argument("--expected-train-questions", type=int, default=preflight.EXPECTED_QUESTIONS["train"])
    parser.add_argument("--expected-validation-questions", type=int, default=preflight.EXPECTED_QUESTIONS["validation"])
    parser.add_argument(
        "--expected-train-samples-per-question", type=int, default=preflight.EXPECTED_SAMPLES_PER_QUESTION
    )
    parser.add_argument(
        "--expected-validation-samples-per-question", type=int, default=preflight.EXPECTED_SAMPLES_PER_QUESTION
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    token = os.environ.get(args.token_env)
    if not token:
        print(f"ERROR: required token environment variable {args.token_env!r} is not set", file=sys.stderr)
        return 2
    try:
        result = upload_dataset(
            args.dataset_path,
            repo_id=args.repo_id,
            revision=args.revision,
            token=token,
            allow_question_overlap=args.allow_question_overlap,
            private=not args.public,
            expected_questions={
                "train": args.expected_train_questions,
                "validation": args.expected_validation_questions,
            },
            expected_samples_per_question={
                "train": args.expected_train_samples_per_question,
                "validation": args.expected_validation_samples_per_question,
            },
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {_redact_secret(str(error), token)}", file=sys.stderr)
        return 2
    sys.stdout.write("\n".join(result.lines()) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
