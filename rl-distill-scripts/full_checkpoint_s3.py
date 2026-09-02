#!/usr/bin/env python3
"""Persist and restore complete verl checkpoints through an S3 prefix.

The normal Hugging Face export contains weights only.  This helper preserves
the optimizer, scheduler/RNG extras, and dataloader cursor as one recoverable
checkpoint.  Both PPO-style ``actor/`` checkpoints and SPMD SFT checkpoints
with root-level rank shards are supported.  A remote checkpoint is considered
usable only after its size manifest is uploaded as ``_REMOTE_COMPLETE.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

_MANIFEST_NAME = "remote_checkpoint_manifest.json"
_COMPLETE_NAME = "_REMOTE_COMPLETE.json"
_RUN_COMPLETE_NAME = "run_complete.json"
_RUN_COMPLETE_PROTOCOL = "gemma4_rl_run_complete_v1"
_RUN_OUTCOME_NAME = "run_outcome.json"
_BEST_HF_PREFIX = "best_hf"
_BEST_HF_COMPLETE_NAME = "_REMOTE_COMPLETE.json"
_BEST_HF_PROTOCOL = "gemma4_rl_best_hf_v1"
_ROLLING_PREFIX = "rolling"
_LATEST_TRACKER_NAME = "latest_checkpointed_iteration.txt"
INCOMPLETE_EXIT_CODE = 3


def _require_complete_rank_shards(state_dir: Path, kind: str) -> int:
    pattern = re.compile(rf"{re.escape(kind)}_world_size_(\d+)_rank_(\d+)\.pt")
    shards = []
    for path in state_dir.glob(f"{kind}_world_size_*_rank_*.pt"):
        match = pattern.fullmatch(path.name)
        if match is not None:
            shards.append((int(match.group(1)), int(match.group(2))))
    if not shards:
        raise FileNotFoundError(f"checkpoint is missing {kind} shards in {state_dir}")
    world_sizes = {world_size for world_size, _ in shards}
    if len(world_sizes) != 1:
        raise ValueError(f"checkpoint has inconsistent {kind} world sizes: {sorted(world_sizes)}")
    world_size = world_sizes.pop()
    ranks = {rank for _, rank in shards}
    expected_ranks = set(range(world_size))
    if ranks != expected_ranks:
        raise ValueError(f"checkpoint has incomplete {kind} shards: world_size={world_size} ranks={sorted(ranks)}")
    return world_size


def _require_sft_dataloader_shards(step_dir: Path, world_size: int) -> None:
    pattern = re.compile(r"data_(\d+)\.pt")
    ranks = {
        int(match.group(1))
        for path in step_dir.glob("data_*.pt")
        if (match := pattern.fullmatch(path.name)) is not None
    }
    expected_ranks = set(range(world_size))
    if ranks != expected_ranks:
        raise ValueError(
            f"checkpoint has incomplete SFT dataloader shards: world_size={world_size} ranks={sorted(ranks)}"
        )


def _checkpoint_layout(step_dir: Path) -> tuple[str, int]:
    actor_dir = step_dir / "actor"
    if actor_dir.is_dir():
        if not (step_dir / "data.pt").is_file():
            raise FileNotFoundError(f"checkpoint is missing dataloader state: {step_dir / 'data.pt'}")
        state_dir = actor_dir
        layout = "ppo_actor"
    else:
        state_dir = step_dir
        layout = "sft_spmd"

    shard_world_sizes = {
        kind: _require_complete_rank_shards(state_dir, kind) for kind in ("model", "optim", "extra_state")
    }
    if len(set(shard_world_sizes.values())) != 1:
        raise ValueError(f"checkpoint shard world sizes disagree: {shard_world_sizes}")
    world_size = next(iter(shard_world_sizes.values()))
    if layout == "sft_spmd":
        _require_sft_dataloader_shards(step_dir, world_size)
    return layout, world_size


def _normalize_s3_uri(uri: str) -> str:
    normalized = uri.rstrip("/")
    if not normalized.startswith("s3://") or normalized == "s3://":
        raise ValueError(f"expected an s3:// checkpoint prefix, got {uri!r}")
    return normalized


def _split_s3_uri(uri: str) -> tuple[str, str]:
    normalized = _normalize_s3_uri(uri)
    bucket_and_key = normalized.removeprefix("s3://")
    bucket, separator, prefix = bucket_and_key.partition("/")
    if not bucket or not separator or not prefix:
        raise ValueError(f"checkpoint prefix must include a bucket and key prefix, got {uri!r}")
    return bucket, prefix.rstrip("/")


def _s3_client():
    import boto3

    return boto3.client("s3", region_name="us-west-2")


def build_size_manifest(step_dir: Path, step: int) -> dict[str, object]:
    if not step_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {step_dir}")
    layout, world_size = _checkpoint_layout(step_dir)

    files: list[dict[str, object]] = []
    for path in sorted(step_dir.rglob("*")):
        if not path.is_file() or path.name in {_MANIFEST_NAME, _COMPLETE_NAME}:
            continue
        files.append({"path": path.relative_to(step_dir).as_posix(), "size": path.stat().st_size})
    if not files:
        raise ValueError(f"checkpoint has no files: {step_dir}")
    return {
        "schema_version": 1,
        "global_step": step,
        "layout": layout,
        "world_size": world_size,
        "file_count": len(files),
        "total_bytes": sum(int(item["size"]) for item in files),
        "files": files,
    }


def validate_size_manifest(step_dir: Path, manifest: dict[str, object]) -> None:
    expected_files = manifest.get("files")
    if not isinstance(expected_files, list) or not expected_files:
        raise ValueError("remote checkpoint manifest has no files")

    expected: dict[str, int] = {}
    for item in expected_files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("remote checkpoint manifest contains an invalid file entry")
        relative_path = Path(str(item["path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.parts:
            raise ValueError(f"remote checkpoint manifest contains an unsafe path: {item['path']!r}")
        normalized = relative_path.as_posix()
        if normalized in expected:
            raise ValueError(f"remote checkpoint manifest repeats a path: {normalized}")
        expected[normalized] = int(item["size"])

    actual: dict[str, int] = {}
    for path in sorted(step_dir.rglob("*")):
        if not path.is_file() or path.name in {_MANIFEST_NAME, _COMPLETE_NAME}:
            continue
        actual[path.relative_to(step_dir).as_posix()] = path.stat().st_size
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        wrong_size = sorted(path for path in set(actual) & set(expected) if actual[path] != expected[path])
        raise ValueError(
            "downloaded checkpoint does not match its completion manifest: "
            f"missing={missing[:10]} unexpected={unexpected[:10]} wrong_size={wrong_size[:10]}"
        )


def _upload_checkpoint_step(
    *,
    checkpoint_root: Path,
    step: int,
    remote_root: str,
    bucket: str,
    prefix: str,
    label: str,
) -> dict[str, object]:
    step_dir = checkpoint_root / f"global_step_{step}"
    manifest = build_size_manifest(step_dir, step)
    manifest_path = step_dir / _MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    remote_step = f"{remote_root}/global_step_{step}"
    print(
        f"[{label}] uploading step={step} files={manifest['file_count']} "
        f"bytes={manifest['total_bytes']} to {remote_step}",
        flush=True,
    )
    client = _s3_client()
    for item in manifest["files"]:
        relative_path = str(item["path"])
        client.upload_file(
            str(step_dir / relative_path),
            bucket,
            f"{prefix}/global_step_{step}/{relative_path}",
        )
    client.upload_file(
        str(manifest_path),
        bucket,
        f"{prefix}/global_step_{step}/{_MANIFEST_NAME}",
    )

    manifest_bytes = manifest_path.read_bytes()
    client.put_object(
        Bucket=bucket,
        Key=f"{prefix}/global_step_{step}/{_COMPLETE_NAME}",
        Body=manifest_bytes,
        ContentType="application/json",
    )
    _verify_remote_checkpoint_step(client, bucket, prefix, step)
    return manifest


def upload_checkpoint(checkpoint_root: Path, step: int, s3_uri: str) -> None:
    """Upload one permanent checkpoint and advance its durable tracker."""

    remote_root = _normalize_s3_uri(s3_uri)
    bucket, prefix = _split_s3_uri(remote_root)
    _upload_checkpoint_step(
        checkpoint_root=checkpoint_root,
        step=step,
        remote_root=remote_root,
        bucket=bucket,
        prefix=prefix,
        label="FullCheckpointS3",
    )
    client = _s3_client()
    client.put_object(
        Bucket=bucket,
        Key=f"{prefix}/{_LATEST_TRACKER_NAME}",
        Body=f"{step}\n".encode(),
        ContentType="text/plain",
    )
    observed_step = _read_remote_step(client, bucket, prefix)
    if observed_step != step:
        raise RuntimeError(
            f"permanent checkpoint tracker verification failed: expected={step} observed={observed_step}"
        )
    print(f"[FullCheckpointS3] completed step={step}", flush=True)


def _list_s3_keys(client, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    continuation_token: str | None = None
    while True:
        kwargs: dict[str, object] = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token is not None:
            kwargs["ContinuationToken"] = continuation_token
        response = client.list_objects_v2(**kwargs)
        keys.extend(str(item["Key"]) for item in response.get("Contents", []))
        if not response.get("IsTruncated", False):
            return keys
        continuation_token = response.get("NextContinuationToken")
        if not isinstance(continuation_token, str) or not continuation_token:
            raise RuntimeError(f"S3 listing for s3://{bucket}/{prefix} was truncated without a continuation token")


def _delete_s3_keys(client, bucket: str, keys: list[str]) -> None:
    for start in range(0, len(keys), 1000):
        batch = keys[start : start + 1000]
        if not batch:
            continue
        response = client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
        )
        errors = response.get("Errors", [])
        if errors:
            raise RuntimeError(f"failed to delete rolling checkpoint objects: {errors[:5]}")


def _best_effort_cleanup(action: str, callback) -> bool:
    """Run storage reclamation without making a valid checkpoint fatal.

    Uploads, manifests, and pointer verification remain fail-closed. Deleting
    an older rolling checkpoint is only a space-reclamation operation, and the
    EKS training role is intentionally not guaranteed to have DeleteObject.
    """

    try:
        callback()
    except Exception as error:
        print(
            f"[RollingCheckpointS3] cleanup_deferred action={action} error={type(error).__name__}: {error}",
            flush=True,
        )
        return False
    return True


def _prune_rolling_steps(client, bucket: str, rolling_prefix: str, *, keep_step: int | None) -> None:
    step_pattern = re.compile(rf"^{re.escape(rolling_prefix)}/global_step_(\d+)/")
    stale_keys: list[str] = []
    for key in _list_s3_keys(client, bucket, f"{rolling_prefix}/global_step_"):
        match = step_pattern.match(key)
        if match is None:
            continue
        if keep_step is None or int(match.group(1)) != keep_step:
            stale_keys.append(key)
    _delete_s3_keys(client, bucket, stale_keys)


def upload_rolling_checkpoint(checkpoint_root: Path, step: int, s3_uri: str) -> None:
    """Atomically replace the single rolling, fully resumable S3 checkpoint.

    The completion marker and every object are verified before the rolling
    pointer advances. Old rolling objects are pruned only after that commit, so
    an interrupted upload cannot invalidate the previously committed step.
    """

    permanent_root = _normalize_s3_uri(s3_uri)
    rolling_root = f"{permanent_root}/{_ROLLING_PREFIX}"
    bucket, rolling_prefix = _split_s3_uri(rolling_root)
    client = _s3_client()
    previous_step = _read_remote_step(client, bucket, rolling_prefix)
    if previous_step is not None and previous_step > step:
        raise ValueError(
            "refusing to move the rolling checkpoint backward: "
            f"previous_step={previous_step} new_step={step} uri={rolling_root}"
        )

    _upload_checkpoint_step(
        checkpoint_root=checkpoint_root,
        step=step,
        remote_root=rolling_root,
        bucket=bucket,
        prefix=rolling_prefix,
        label="RollingCheckpointS3",
    )
    client = _s3_client()
    client.put_object(
        Bucket=bucket,
        Key=f"{rolling_prefix}/{_LATEST_TRACKER_NAME}",
        Body=f"{step}\n".encode(),
        ContentType="text/plain",
    )
    observed_step = _read_remote_step(client, bucket, rolling_prefix)
    if observed_step != step:
        raise RuntimeError(f"rolling checkpoint pointer verification failed: expected={step} observed={observed_step}")

    cleanup_complete = _best_effort_cleanup(
        f"prune-before-step-{step}",
        lambda: _prune_rolling_steps(client, bucket, rolling_prefix, keep_step=step),
    )
    print(
        f"[RollingCheckpointS3] committed step={step} previous_step={previous_step} "
        f"cleanup={'complete' if cleanup_complete else 'deferred'}",
        flush=True,
    )


def retire_rolling_checkpoint(s3_uri: str, permanent_step: int) -> bool:
    """Retire rolling state once an equal-or-newer permanent checkpoint exists."""

    permanent_root = _normalize_s3_uri(s3_uri)
    bucket, permanent_prefix = _split_s3_uri(permanent_root)
    client = _s3_client()
    permanent_pointer = _read_remote_step(client, bucket, permanent_prefix)
    if permanent_pointer is None or permanent_pointer < permanent_step:
        raise ValueError(
            "cannot retire rolling state before the permanent checkpoint is committed: "
            f"expected_at_least={permanent_step} observed={permanent_pointer}"
        )
    rolling_prefix = f"{permanent_prefix}/{_ROLLING_PREFIX}"
    rolling_step = _read_remote_step(client, bucket, rolling_prefix)
    if rolling_step is not None and rolling_step > permanent_step:
        raise ValueError(
            "refusing to retire a rolling checkpoint newer than the permanent checkpoint: "
            f"rolling_step={rolling_step} permanent_step={permanent_step}"
        )
    pointer_deleted = _best_effort_cleanup(
        f"retire-pointer-after-permanent-{permanent_step}",
        lambda: client.delete_object(Bucket=bucket, Key=f"{rolling_prefix}/{_LATEST_TRACKER_NAME}"),
    )
    steps_deleted = _best_effort_cleanup(
        f"retire-steps-after-permanent-{permanent_step}",
        lambda: _prune_rolling_steps(client, bucket, rolling_prefix, keep_step=None),
    )
    cleanup_complete = pointer_deleted and steps_deleted
    if cleanup_complete:
        print(
            f"[RollingCheckpointS3] retired rolling_step={rolling_step} after permanent_step={permanent_step}",
            flush=True,
        )
    else:
        print(
            f"[RollingCheckpointS3] retirement_deferred rolling_step={rolling_step} "
            f"after permanent_step={permanent_step}",
            flush=True,
        )
    return cleanup_complete


def cleanup_rolling_checkpoint(s3_uri: str, *, finalize: bool = False) -> dict[str, object]:
    """Delete stale rolling objects using an external identity with S3 delete access.

    The non-finalizing mode is safe to run while training: it never deletes a
    step newer than the committed rolling pointer and leaves the pointer in
    place. Finalizing is for a terminal job; it requires an equal-or-newer,
    fully verified permanent checkpoint before removing the complete rolling
    namespace.
    """

    permanent_root = _normalize_s3_uri(s3_uri)
    bucket, permanent_prefix = _split_s3_uri(permanent_root)
    rolling_prefix = f"{permanent_prefix}/{_ROLLING_PREFIX}"
    client = _s3_client()

    permanent_step = _read_remote_step(client, bucket, permanent_prefix)
    if permanent_step is not None:
        _verify_remote_checkpoint_step(client, bucket, permanent_prefix, permanent_step)

    rolling_step = _read_remote_step(client, bucket, rolling_prefix)
    rolling_valid = False
    if rolling_step is not None:
        try:
            _verify_remote_checkpoint_step(client, bucket, rolling_prefix, rolling_step)
            rolling_valid = True
        except Exception as error:
            print(
                f"[RollingCheckpointJanitor] rolling_pointer_invalid step={rolling_step} "
                f"error={type(error).__name__}: {error}",
                flush=True,
            )

    if finalize:
        if permanent_step is None:
            raise ValueError(f"cannot finalize rolling cleanup without a permanent checkpoint: {permanent_root}")
        if rolling_valid and rolling_step is not None and rolling_step > permanent_step:
            raise ValueError(
                "refusing to finalize a rolling checkpoint newer than the permanent checkpoint: "
                f"rolling_step={rolling_step} permanent_step={permanent_step}"
            )
        keys = _list_s3_keys(client, bucket, f"{rolling_prefix}/")
        _delete_s3_keys(client, bucket, keys)
        print(
            f"[RollingCheckpointJanitor] finalized permanent_step={permanent_step} "
            f"rolling_step={rolling_step} deleted={len(keys)}",
            flush=True,
        )
        return {
            "mode": "finalize",
            "permanent_step": permanent_step,
            "rolling_step": rolling_step,
            "deleted": len(keys),
        }

    step_pattern = re.compile(rf"^{re.escape(rolling_prefix)}/global_step_(\d+)/")
    stale_keys: list[str] = []
    for key in _list_s3_keys(client, bucket, f"{rolling_prefix}/global_step_"):
        match = step_pattern.match(key)
        if match is None:
            continue
        candidate_step = int(match.group(1))
        if rolling_valid and rolling_step is not None:
            if candidate_step < rolling_step:
                stale_keys.append(key)
            elif permanent_step is not None and permanent_step >= rolling_step and candidate_step == rolling_step:
                stale_keys.append(key)
        elif permanent_step is not None and candidate_step <= permanent_step:
            stale_keys.append(key)
    _delete_s3_keys(client, bucket, stale_keys)
    print(
        f"[RollingCheckpointJanitor] pruned permanent_step={permanent_step} "
        f"rolling_step={rolling_step} deleted={len(stale_keys)} pointer_retained=true",
        flush=True,
    )
    return {
        "mode": "active",
        "permanent_step": permanent_step,
        "rolling_step": rolling_step,
        "deleted": len(stale_keys),
    }


def _read_local_run_outcome(checkpoint_root: Path) -> dict[str, object]:
    outcome_path = checkpoint_root / _RUN_OUTCOME_NAME
    if not outcome_path.is_file():
        raise FileNotFoundError(f"run outcome does not exist: {outcome_path}")
    payload = json.loads(outcome_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"run outcome must be a JSON object: {outcome_path}")
    if payload.get("protocol") != "gemma4_rl_run_outcome_v1":
        raise ValueError(f"unexpected run outcome protocol: {payload.get('protocol')!r}")
    if payload.get("status") != "complete":
        raise ValueError(f"run outcome is not complete: {payload}")
    best_step = int(payload.get("best_step", -1))
    final_step = int(payload.get("final_step", -1))
    if best_step < 0 or final_step < 1 or best_step > final_step:
        raise ValueError(f"run outcome has invalid step selection: best_step={best_step} final_step={final_step}")
    return payload


def _directory_size_manifest(source_dir: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        files.append({"path": path.relative_to(source_dir).as_posix(), "size": path.stat().st_size})
    return files


def publish_best_hf_export(checkpoint_root: Path, s3_uri: str) -> dict[str, object]:
    """Publish the selected all-time-best HF export and outcome receipt.

    The completion marker is uploaded last. A best step of zero means the base
    checkpoint remained the all-time high; in that case the outcome receipt is
    still durable and no redundant copy of the immutable base model is made.
    """

    outcome = _read_local_run_outcome(checkpoint_root)
    best_step = int(outcome["best_step"])
    remote_root = _normalize_s3_uri(s3_uri)
    bucket, prefix = _split_s3_uri(remote_root)
    client = _s3_client()

    files: list[dict[str, object]] = []
    source_dir: Path | None = None
    if best_step > 0:
        source_dir = checkpoint_root / f"global_step_{best_step}" / "actor" / "huggingface"
        if not source_dir.is_dir():
            raise FileNotFoundError(f"best HF export does not exist: {source_dir}")
        files = _directory_size_manifest(source_dir)
        if not files:
            raise ValueError(f"best HF export has no files: {source_dir}")
        print(
            f"[BestHFS3] uploading best_step={best_step} files={len(files)} "
            f"bytes={sum(int(item['size']) for item in files)} to {remote_root}/{_BEST_HF_PREFIX}",
            flush=True,
        )
        for item in files:
            relative_path = str(item["path"])
            client.upload_file(
                str(source_dir / relative_path),
                bucket,
                f"{prefix}/{_BEST_HF_PREFIX}/{relative_path}",
            )

    outcome_bytes = (json.dumps(outcome, sort_keys=True, indent=2) + "\n").encode()
    client.put_object(
        Bucket=bucket,
        Key=f"{prefix}/{_RUN_OUTCOME_NAME}",
        Body=outcome_bytes,
        ContentType="application/json",
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "protocol": _BEST_HF_PROTOCOL,
        "status": "complete",
        "checkpoint_s3_uri": remote_root,
        "best_step": best_step,
        "final_step": int(outcome["final_step"]),
        "stop_reason": outcome.get("stop_reason"),
        "file_count": len(files),
        "total_bytes": sum(int(item["size"]) for item in files),
        "files": files,
        "run_outcome": outcome,
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    client.put_object(
        Bucket=bucket,
        Key=f"{prefix}/{_BEST_HF_PREFIX}/{_BEST_HF_COMPLETE_NAME}",
        Body=manifest_bytes,
        ContentType="application/json",
    )
    verified = check_best_hf_export(remote_root)
    print(
        f"[BestHFS3] published verified outcome best_step={best_step} files={verified['file_count']} at {remote_root}",
        flush=True,
    )
    return verified


def check_best_hf_export(s3_uri: str) -> dict[str, object] | None:
    remote_root = _normalize_s3_uri(s3_uri)
    bucket, prefix = _split_s3_uri(remote_root)
    client = _s3_client()
    manifest = _read_json_object(
        client,
        bucket,
        f"{prefix}/{_BEST_HF_PREFIX}/{_BEST_HF_COMPLETE_NAME}",
        missing_ok=True,
    )
    if manifest is None:
        return None
    if manifest.get("protocol") != _BEST_HF_PROTOCOL or manifest.get("status") != "complete":
        raise ValueError(f"invalid best-HF completion marker at {remote_root}: {manifest}")
    if manifest.get("checkpoint_s3_uri") != remote_root:
        raise ValueError(f"best-HF completion marker points at the wrong root: {manifest}")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("best-HF completion marker has an invalid files list")
    expected_total = 0
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("best-HF completion marker has an invalid file entry")
        relative_path = Path(str(item["path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.parts:
            raise ValueError(f"best-HF completion marker has an unsafe path: {item['path']!r}")
        expected_size = int(item.get("size", -1))
        response = client.head_object(
            Bucket=bucket,
            Key=f"{prefix}/{_BEST_HF_PREFIX}/{relative_path.as_posix()}",
        )
        observed_size = int(response.get("ContentLength", -1))
        if observed_size != expected_size:
            raise ValueError(
                f"best-HF object size mismatch for {relative_path}: expected={expected_size} observed={observed_size}"
            )
        expected_total += expected_size
    if int(manifest.get("file_count", -1)) != len(files):
        raise ValueError("best-HF completion marker file_count mismatch")
    if int(manifest.get("total_bytes", -1)) != expected_total:
        raise ValueError("best-HF completion marker total_bytes mismatch")
    if int(manifest.get("best_step", -1)) > 0 and not files:
        raise ValueError("a trained best step must contain an HF export")
    outcome = _read_json_object(client, bucket, f"{prefix}/{_RUN_OUTCOME_NAME}")
    if outcome != manifest.get("run_outcome"):
        raise ValueError("remote run outcome does not match the best-HF completion marker")
    return manifest


def _read_remote_step(client, bucket: str, prefix: str) -> int | None:
    try:
        response = client.get_object(Bucket=bucket, Key=f"{prefix}/{_LATEST_TRACKER_NAME}")
    except Exception as error:
        response_payload = getattr(error, "response", {})
        error_code = response_payload.get("Error", {}).get("Code") if isinstance(response_payload, dict) else None
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    raw = response["Body"].read().decode().strip()
    if not raw:
        return None
    step = int(raw)
    if step < 1:
        raise ValueError(f"remote checkpoint tracker must be positive, got {step}")
    return step


def _read_json_object(client, bucket: str, key: str, *, missing_ok: bool = False) -> dict[str, object] | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except Exception as error:
        response_payload = getattr(error, "response", {})
        error_code = response_payload.get("Error", {}).get("Code") if isinstance(response_payload, dict) else None
        if missing_ok and error_code in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return None
        raise
    payload = json.loads(response["Body"].read())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object at s3://{bucket}/{key}")
    return payload


def _manifest_file_sizes(manifest: dict[str, object]) -> dict[str, int]:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("remote checkpoint completion marker has no files")
    sizes: dict[str, int] = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("remote checkpoint completion marker has an invalid file entry")
        path = Path(str(item["path"]))
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"remote checkpoint completion marker has an unsafe path: {item['path']!r}")
        normalized = path.as_posix()
        if normalized in sizes:
            raise ValueError(f"remote checkpoint completion marker repeats a path: {normalized}")
        size = int(item.get("size", -1))
        if size <= 0:
            raise ValueError(f"remote checkpoint completion marker has an invalid size for {normalized}: {size}")
        sizes[normalized] = size
    return sizes


def restore_hf_export(checkpoint_root: Path, s3_uri: str, step: int) -> Path:
    """Restore only one checkpoint's HF export for all-time-best publication."""

    if step < 1:
        raise ValueError(f"HF export step must be positive, got {step}")
    remote_root = _normalize_s3_uri(s3_uri)
    bucket, prefix = _split_s3_uri(remote_root)
    client = _s3_client()
    step_prefix = f"{prefix}/global_step_{step}"
    manifest = _read_json_object(client, bucket, f"{step_prefix}/{_COMPLETE_NAME}")
    assert manifest is not None
    if int(manifest.get("global_step", -1)) != step:
        raise ValueError(f"remote completion marker records the wrong global step: {manifest}")

    hf_prefix = "actor/huggingface/"
    hf_files = {
        path.removeprefix(hf_prefix): size
        for path, size in _manifest_file_sizes(manifest).items()
        if path.startswith(hf_prefix)
    }
    if not hf_files:
        raise FileNotFoundError(f"remote checkpoint step {step} has no HF export at {remote_root}")

    hf_dir = checkpoint_root / f"global_step_{step}" / "actor" / "huggingface"
    if hf_dir.is_dir():
        observed = {
            path.relative_to(hf_dir).as_posix(): path.stat().st_size
            for path in sorted(hf_dir.rglob("*"))
            if path.is_file()
        }
        if observed == hf_files:
            return hf_dir
        raise ValueError(
            f"local HF export does not match remote checkpoint step {step}: "
            f"expected_files={len(hf_files)} observed_files={len(observed)} path={hf_dir}"
        )

    partial_dir = hf_dir.with_name(".huggingface.partial")
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)
    print(
        f"[FullCheckpointS3] restoring best HF export step={step} files={len(hf_files)} from {remote_root}",
        flush=True,
    )
    try:
        for relative_path, expected_size in sorted(hf_files.items()):
            destination = partial_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(
                bucket,
                f"{step_prefix}/{hf_prefix}{relative_path}",
                str(destination),
            )
            if destination.stat().st_size != expected_size:
                raise ValueError(
                    f"restored HF object size mismatch for {relative_path}: "
                    f"expected={expected_size} observed={destination.stat().st_size}"
                )
        hf_dir.parent.mkdir(parents=True, exist_ok=True)
        partial_dir.rename(hf_dir)
    except BaseException:
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise
    return hf_dir


def _require_remote_rank_shards(
    file_sizes: dict[str, int],
    kind: str,
    world_size: int,
    *,
    directory: str,
) -> None:
    path_prefix = f"{directory.rstrip('/')}/" if directory else ""
    pattern = re.compile(rf"{re.escape(path_prefix)}{re.escape(kind)}_world_size_(\d+)_rank_(\d+)\.pt")
    observed = {
        (int(match.group(1)), int(match.group(2)))
        for path in file_sizes
        if (match := pattern.fullmatch(path)) is not None
    }
    expected = {(world_size, rank) for rank in range(world_size)}
    if observed != expected:
        raise ValueError(
            f"remote checkpoint has incomplete {kind} shards: expected={sorted(expected)} observed={sorted(observed)}"
        )


def _verify_remote_checkpoint_step(
    client,
    bucket: str,
    prefix: str,
    expected_step: int,
    *,
    expected_world_size: int | None = None,
    require_ppo: bool = False,
) -> dict[str, object]:
    step_prefix = f"{prefix}/global_step_{expected_step}"
    manifest = _read_json_object(client, bucket, f"{step_prefix}/{_COMPLETE_NAME}")
    assert manifest is not None
    if int(manifest.get("global_step", -1)) != expected_step:
        raise ValueError(f"remote completion marker records the wrong global step: {manifest}")

    layout = manifest.get("layout")
    if layout not in {"ppo_actor", "sft_spmd"}:
        raise ValueError(f"remote completion marker has unsupported checkpoint layout: {layout!r}")
    if require_ppo and layout != "ppo_actor":
        raise ValueError(f"expected PPO actor checkpoint layout, found {layout!r}")

    world_size = int(manifest.get("world_size", -1))
    if world_size < 1:
        raise ValueError(f"remote completion marker has invalid world_size={world_size}")
    if expected_world_size is not None and world_size != expected_world_size:
        raise ValueError(f"remote checkpoint world-size mismatch: expected {expected_world_size}, found {world_size}")

    file_sizes = _manifest_file_sizes(manifest)
    shard_directory = "actor" if layout == "ppo_actor" else ""
    if layout == "ppo_actor":
        if file_sizes.get("data.pt", 0) <= 0:
            raise ValueError("remote checkpoint is missing the StatefulDataLoader cursor data.pt")
    else:
        expected_data = {f"data_{rank}.pt" for rank in range(world_size)}
        observed_data = {path for path in file_sizes if re.fullmatch(r"data_\d+\.pt", path)}
        if observed_data != expected_data:
            raise ValueError(
                "remote checkpoint has incomplete SFT dataloader shards: "
                f"expected={sorted(expected_data)} observed={sorted(observed_data)}"
            )
    for kind in ("model", "optim", "extra_state"):
        _require_remote_rank_shards(file_sizes, kind, world_size, directory=shard_directory)
    if int(manifest.get("file_count", -1)) != len(file_sizes):
        raise ValueError("remote checkpoint file_count does not match its file list")
    if int(manifest.get("total_bytes", -1)) != sum(file_sizes.values()):
        raise ValueError("remote checkpoint total_bytes does not match its file list")

    for relative_path, expected_size in file_sizes.items():
        response = client.head_object(Bucket=bucket, Key=f"{step_prefix}/{relative_path}")
        observed_size = int(response.get("ContentLength", -1))
        if observed_size != expected_size:
            raise ValueError(
                f"remote checkpoint object size mismatch for {relative_path}: "
                f"expected {expected_size}, found {observed_size}"
            )
    return manifest


def verify_remote_checkpoint(
    s3_uri: str,
    expected_step: int,
    *,
    expected_world_size: int | None = None,
) -> dict[str, object]:
    """Verify the exact final PPO checkpoint and every remotely registered object."""

    remote_root = _normalize_s3_uri(s3_uri)
    bucket, prefix = _split_s3_uri(remote_root)
    client = _s3_client()
    step = _read_remote_step(client, bucket, prefix)
    if step is None:
        raise FileNotFoundError(f"no completed remote checkpoint at {remote_root}")
    if step != expected_step:
        raise ValueError(
            f"remote checkpoint tracker mismatch for {remote_root}: expected {expected_step}, found {step}"
        )

    return _verify_remote_checkpoint_step(
        client,
        bucket,
        prefix,
        expected_step,
        expected_world_size=expected_world_size,
        require_ppo=True,
    )


def _validate_run_completion_receipt(
    receipt: dict[str, object],
    *,
    s3_uri: str,
    expected_step: int,
    expected_world_size: int,
    model: str,
    difficulty: str,
    seed: int,
    wandb_run_id: str,
    hf_repo: str,
) -> None:
    expected = {
        "protocol": _RUN_COMPLETE_PROTOCOL,
        "status": "complete",
        "checkpoint_s3_uri": _normalize_s3_uri(s3_uri),
        "checkpoint_step": expected_step,
        "checkpoint_world_size": expected_world_size,
        "model": model,
        "difficulty": difficulty,
        "seed": seed,
        "wandb_run_id": wandb_run_id,
        "hf_repo": hf_repo,
    }
    mismatches = {
        key: {"expected": value, "observed": receipt.get(key)}
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if mismatches:
        raise ValueError(f"run completion receipt mismatch: {mismatches}")


def read_run_completion_receipt(s3_uri: str) -> dict[str, object] | None:
    remote_root = _normalize_s3_uri(s3_uri)
    bucket, prefix = _split_s3_uri(remote_root)
    return _read_json_object(
        _s3_client(),
        bucket,
        f"{prefix}/{_RUN_COMPLETE_NAME}",
        missing_ok=True,
    )


def check_run_completion_at_most(
    *,
    s3_uri: str,
    max_step: int,
    expected_world_size: int,
    model: str,
    difficulty: str,
    seed: int,
    wandb_run_id: str,
    hf_repo: str,
) -> dict[str, object] | None:
    """Validate a completed run whose terminal step may be an early stop."""

    receipt = read_run_completion_receipt(s3_uri)
    if receipt is None:
        return None
    checkpoint_step = receipt.get("checkpoint_step")
    if type(checkpoint_step) is not int or not 1 <= checkpoint_step <= max_step:
        raise ValueError(
            "run completion receipt has an invalid terminal step: "
            f"checkpoint_step={checkpoint_step!r} max_step={max_step}"
        )
    _validate_run_completion_receipt(
        receipt,
        s3_uri=s3_uri,
        expected_step=checkpoint_step,
        expected_world_size=expected_world_size,
        model=model,
        difficulty=difficulty,
        seed=seed,
        wandb_run_id=wandb_run_id,
        hf_repo=hf_repo,
    )
    verify_remote_checkpoint(
        s3_uri,
        checkpoint_step,
        expected_world_size=expected_world_size,
    )
    return receipt


def finalize_run_if_complete(
    *,
    s3_uri: str,
    expected_step: int,
    expected_world_size: int,
    model: str,
    difficulty: str,
    seed: int,
    wandb_run_id: str,
    hf_repo: str,
    receipt_path: Path | None = None,
) -> bool:
    """Publish/validate a durable receipt when the exact final checkpoint exists.

    Returns ``False`` only when no checkpoint exists yet or its latest complete
    step is lower than ``expected_step``. A checkpoint beyond the configured
    final step is an error rather than something the launcher may silently use.
    """

    remote_root = _normalize_s3_uri(s3_uri)
    existing = read_run_completion_receipt(remote_root)
    if existing is not None:
        _validate_run_completion_receipt(
            existing,
            s3_uri=remote_root,
            expected_step=expected_step,
            expected_world_size=expected_world_size,
            model=model,
            difficulty=difficulty,
            seed=seed,
            wandb_run_id=wandb_run_id,
            hf_repo=hf_repo,
        )
        manifest = verify_remote_checkpoint(
            remote_root,
            expected_step,
            expected_world_size=expected_world_size,
        )
        if existing.get("checkpoint_file_count") != int(manifest["file_count"]):
            raise ValueError("run completion receipt checkpoint_file_count no longer matches S3")
        if existing.get("checkpoint_total_bytes") != int(manifest["total_bytes"]):
            raise ValueError("run completion receipt checkpoint_total_bytes no longer matches S3")
        if receipt_path is not None:
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(json.dumps(existing, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(
            f"[FullCheckpointS3] verified existing receipt and full remote checkpoint at {remote_root}",
            flush=True,
        )
        return True

    bucket, prefix = _split_s3_uri(remote_root)
    client = _s3_client()
    step = _read_remote_step(client, bucket, prefix)
    if step is None or step < expected_step:
        print(
            f"[FullCheckpointS3] run incomplete at {remote_root}: latest_step={step} expected_step={expected_step}",
            flush=True,
        )
        return False
    if step > expected_step:
        raise ValueError(
            f"remote checkpoint advanced beyond the configured final step: "
            f"latest_step={step} expected_step={expected_step} uri={remote_root}"
        )

    manifest = verify_remote_checkpoint(
        remote_root,
        expected_step,
        expected_world_size=expected_world_size,
    )
    receipt: dict[str, object] = {
        "protocol": _RUN_COMPLETE_PROTOCOL,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_s3_uri": remote_root,
        "checkpoint_step": expected_step,
        "checkpoint_layout": manifest["layout"],
        "checkpoint_world_size": int(manifest["world_size"]),
        "checkpoint_file_count": int(manifest["file_count"]),
        "checkpoint_total_bytes": int(manifest["total_bytes"]),
        "model": model,
        "difficulty": difficulty,
        "seed": seed,
        "wandb_run_id": wandb_run_id,
        "hf_repo": hf_repo,
    }
    receipt_bytes = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    if receipt_path is not None:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_bytes(receipt_bytes)
    client.put_object(
        Bucket=bucket,
        Key=f"{prefix}/{_RUN_COMPLETE_NAME}",
        Body=receipt_bytes,
        ContentType="application/json",
    )
    response = client.head_object(Bucket=bucket, Key=f"{prefix}/{_RUN_COMPLETE_NAME}")
    if int(response.get("ContentLength", -1)) != len(receipt_bytes):
        raise RuntimeError(f"completion receipt upload verification failed: {remote_root}")
    print(f"[FullCheckpointS3] published verified run completion receipt at {remote_root}", flush=True)
    return True


def _latest_valid_remote_checkpoint(s3_uri: str) -> tuple[int, str, str, dict[str, object]] | None:
    """Select the newest valid permanent or rolling checkpoint.

    Each namespace is validated independently. A broken or interrupted rolling
    upload therefore falls back to the permanent 10-step history instead of
    making the run unresumable.
    """

    permanent_root = _normalize_s3_uri(s3_uri)
    bucket, permanent_prefix = _split_s3_uri(permanent_root)
    client = _s3_client()
    candidates: list[tuple[int, int, str, str, dict[str, object]]] = []
    errors: list[str] = []
    for source, source_root, source_prefix, tie_priority in (
        ("permanent", permanent_root, permanent_prefix, 1),
        ("rolling", f"{permanent_root}/{_ROLLING_PREFIX}", f"{permanent_prefix}/{_ROLLING_PREFIX}", 0),
    ):
        try:
            step = _read_remote_step(client, bucket, source_prefix)
            if step is None:
                continue
            manifest = _verify_remote_checkpoint_step(client, bucket, source_prefix, step)
            candidates.append((step, tie_priority, source, source_prefix, manifest))
        except Exception as error:
            errors.append(f"{source}={type(error).__name__}: {error}")
            print(
                f"[FullCheckpointS3] ignoring invalid {source} checkpoint at {source_root}: {error}",
                flush=True,
            )

    if not candidates:
        if errors:
            raise RuntimeError(f"no valid remote checkpoint at {permanent_root}; " + "; ".join(errors))
        return None

    step, _tie_priority, source, source_prefix, manifest = max(candidates, key=lambda item: (item[0], item[1]))
    return step, source, source_prefix, manifest


def restore_latest_checkpoint(checkpoint_root: Path, s3_uri: str) -> int | None:
    remote_root = _normalize_s3_uri(s3_uri)
    bucket, _prefix = _split_s3_uri(remote_root)
    client = _s3_client()
    selected = _latest_valid_remote_checkpoint(remote_root)
    if selected is None:
        print(f"[FullCheckpointS3] no completed remote checkpoint at {remote_root}", flush=True)
        return None
    step, source, source_prefix, selected_manifest = selected

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    step_dir = checkpoint_root / f"global_step_{step}"
    if step_dir.exists():
        shutil.rmtree(step_dir)
    partial_dir = checkpoint_root / f".global_step_{step}.partial"
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

    source_root = remote_root if source == "permanent" else f"{remote_root}/{_ROLLING_PREFIX}"
    remote_step = f"{source_root}/global_step_{step}"
    print(
        f"[FullCheckpointS3] restoring source={source} {remote_step} to {step_dir}",
        flush=True,
    )
    try:
        marker_response = client.get_object(
            Bucket=bucket,
            Key=f"{source_prefix}/global_step_{step}/{_COMPLETE_NAME}",
        )
        marker_bytes = marker_response["Body"].read()
        complete_path = partial_dir / _COMPLETE_NAME
        complete_path.write_bytes(marker_bytes)
        manifest = json.loads(marker_bytes)
        if manifest != selected_manifest:
            raise ValueError("remote checkpoint completion marker changed after selection")
        if int(manifest.get("global_step", -1)) != step:
            raise ValueError(f"remote completion marker records the wrong global step: {manifest}")
        for item in manifest["files"]:
            relative_path = str(item["path"])
            candidate = Path(relative_path)
            if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
                raise ValueError(f"remote completion marker contains an unsafe path: {relative_path!r}")
            destination = partial_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(
                bucket,
                f"{source_prefix}/global_step_{step}/{relative_path}",
                str(destination),
            )
        (partial_dir / _MANIFEST_NAME).write_bytes(marker_bytes)
        validate_size_manifest(partial_dir, manifest)
        layout, world_size = _checkpoint_layout(partial_dir)
        if manifest.get("layout") != layout or int(manifest.get("world_size", -1)) != world_size:
            raise ValueError(
                "downloaded checkpoint layout does not match its completion manifest: "
                f"layout={layout} world_size={world_size} manifest={manifest.get('layout')}/"
                f"{manifest.get('world_size')}"
            )
        partial_dir.rename(step_dir)
    except BaseException:
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise

    tracker_path = checkpoint_root / "latest_checkpointed_iteration.txt"
    tracker_path.write_text(f"{step}\n", encoding="utf-8")
    print(f"[FullCheckpointS3] restored complete source={source} step={step}", flush=True)
    return step


def upload_completion_receipt(checkpoint_root: Path, s3_uri: str) -> None:
    remote_root = _normalize_s3_uri(s3_uri)
    bucket, prefix = _split_s3_uri(remote_root)
    receipt = checkpoint_root / _RUN_COMPLETE_NAME
    if not receipt.is_file():
        raise FileNotFoundError(f"run completion receipt does not exist: {receipt}")
    client = _s3_client()
    client.upload_file(str(receipt), bucket, f"{prefix}/{_RUN_COMPLETE_NAME}")
    response = client.head_object(Bucket=bucket, Key=f"{prefix}/{_RUN_COMPLETE_NAME}")
    if int(response.get("ContentLength", -1)) != receipt.stat().st_size:
        raise RuntimeError(f"completion receipt upload verification failed: {remote_root}/{_RUN_COMPLETE_NAME}")
    print(f"[FullCheckpointS3] uploaded run completion receipt to {remote_root}", flush=True)


def _add_run_completion_args(parser: argparse.ArgumentParser, *, receipt_path: bool) -> None:
    parser.add_argument("--s3-uri", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard", "strict4of4"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--hf-repo", required=True)
    if receipt_path:
        parser.add_argument("--receipt-path", type=Path)


def _finalize_from_args(args: argparse.Namespace) -> bool:
    return finalize_run_if_complete(
        s3_uri=args.s3_uri,
        expected_step=args.expected_step,
        expected_world_size=args.expected_world_size,
        model=args.model,
        difficulty=args.difficulty,
        seed=args.seed,
        wandb_run_id=args.wandb_run_id,
        hf_repo=args.hf_repo,
        receipt_path=getattr(args, "receipt_path", None),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload = subparsers.add_parser("upload")
    upload.add_argument("--checkpoint-root", type=Path, required=True)
    upload.add_argument("--step", type=int, required=True)
    upload.add_argument("--s3-uri", required=True)

    upload_rolling = subparsers.add_parser("upload-rolling")
    upload_rolling.add_argument("--checkpoint-root", type=Path, required=True)
    upload_rolling.add_argument("--step", type=int, required=True)
    upload_rolling.add_argument("--s3-uri", required=True)

    retire_rolling = subparsers.add_parser("retire-rolling")
    retire_rolling.add_argument("--s3-uri", required=True)
    retire_rolling.add_argument("--permanent-step", type=int, required=True)

    cleanup_rolling = subparsers.add_parser("cleanup-rolling")
    cleanup_rolling.add_argument("--s3-uri", required=True)
    cleanup_rolling.add_argument("--finalize", action="store_true")

    restore = subparsers.add_parser("restore-latest")
    restore.add_argument("--checkpoint-root", type=Path, required=True)
    restore.add_argument("--s3-uri", required=True)

    completion = subparsers.add_parser("upload-completion")
    completion.add_argument("--checkpoint-root", type=Path, required=True)
    completion.add_argument("--s3-uri", required=True)

    publish_best = subparsers.add_parser("publish-best-hf")
    publish_best.add_argument("--checkpoint-root", type=Path, required=True)
    publish_best.add_argument("--s3-uri", required=True)

    check_best = subparsers.add_parser("check-best-hf")
    check_best.add_argument("--s3-uri", required=True)

    complete_run = subparsers.add_parser("complete-run")
    _add_run_completion_args(complete_run, receipt_path=True)

    finalize = subparsers.add_parser("finalize-if-complete")
    _add_run_completion_args(finalize, receipt_path=True)

    check = subparsers.add_parser("check-completion")
    _add_run_completion_args(check, receipt_path=False)

    check_max = subparsers.add_parser("check-completion-max")
    check_max.add_argument("--s3-uri", required=True)
    check_max.add_argument("--max-step", type=int, required=True)
    check_max.add_argument("--expected-world-size", type=int, required=True)
    check_max.add_argument("--model", required=True)
    check_max.add_argument("--difficulty", choices=("easy", "medium", "hard", "strict4of4"), required=True)
    check_max.add_argument("--seed", type=int, required=True)
    check_max.add_argument("--wandb-run-id", required=True)
    check_max.add_argument("--hf-repo", required=True)

    args = parser.parse_args()
    if args.command == "upload":
        upload_checkpoint(args.checkpoint_root, args.step, args.s3_uri)
    elif args.command == "upload-rolling":
        upload_rolling_checkpoint(args.checkpoint_root, args.step, args.s3_uri)
    elif args.command == "retire-rolling":
        retire_rolling_checkpoint(args.s3_uri, args.permanent_step)
    elif args.command == "cleanup-rolling":
        cleanup_rolling_checkpoint(args.s3_uri, finalize=args.finalize)
    elif args.command == "restore-latest":
        restore_latest_checkpoint(args.checkpoint_root, args.s3_uri)
    elif args.command == "upload-completion":
        upload_completion_receipt(args.checkpoint_root, args.s3_uri)
    elif args.command == "publish-best-hf":
        publish_best_hf_export(args.checkpoint_root, args.s3_uri)
    elif args.command == "check-best-hf":
        manifest = check_best_hf_export(args.s3_uri)
        if manifest is None:
            raise SystemExit(INCOMPLETE_EXIT_CODE)
        print(
            f"[BestHFS3] verified best_step={manifest['best_step']} "
            f"files={manifest['file_count']} at {_normalize_s3_uri(args.s3_uri)}",
            flush=True,
        )
    elif args.command == "complete-run":
        if not _finalize_from_args(args):
            raise SystemExit(f"final checkpoint step {args.expected_step} is not complete at {args.s3_uri}")
    elif args.command == "finalize-if-complete":
        if not _finalize_from_args(args):
            raise SystemExit(INCOMPLETE_EXIT_CODE)
    elif args.command == "check-completion-max":
        receipt = check_run_completion_at_most(
            s3_uri=args.s3_uri,
            max_step=args.max_step,
            expected_world_size=args.expected_world_size,
            model=args.model,
            difficulty=args.difficulty,
            seed=args.seed,
            wandb_run_id=args.wandb_run_id,
            hf_repo=args.hf_repo,
        )
        if receipt is None:
            raise SystemExit(INCOMPLETE_EXIT_CODE)
        print(
            f"[FullCheckpointS3] run completion receipt is valid at step "
            f"{receipt['checkpoint_step']} at {_normalize_s3_uri(args.s3_uri)}",
            flush=True,
        )
    else:
        receipt = read_run_completion_receipt(args.s3_uri)
        if receipt is None:
            raise SystemExit(INCOMPLETE_EXIT_CODE)
        _validate_run_completion_receipt(
            receipt,
            s3_uri=args.s3_uri,
            expected_step=args.expected_step,
            expected_world_size=args.expected_world_size,
            model=args.model,
            difficulty=args.difficulty,
            seed=args.seed,
            wandb_run_id=args.wandb_run_id,
            hf_repo=args.hf_repo,
        )
        print(f"[FullCheckpointS3] run completion receipt is valid at {args.s3_uri}", flush=True)


if __name__ == "__main__":
    main()
