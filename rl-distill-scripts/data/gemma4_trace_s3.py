#!/usr/bin/env python3
"""Durable S3 mirroring for resumable Gemma 4 trace shards."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from gemma4_distill_trace_schema import parquet_manifest_path, sha256_file


@dataclass(frozen=True)
class S3Location:
    bucket: str
    prefix: str


def parse_s3_uri(uri: str) -> S3Location:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected s3://bucket/prefix URI, got {uri!r}")
    return S3Location(parsed.netloc, parsed.path.strip("/"))


def _object_key(location: S3Location, relative_path: str) -> str:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe S3 relative path: {relative_path!r}")
    suffix = relative.as_posix()
    return f"{location.prefix}/{suffix}" if location.prefix else suffix


def _relative_local_path(root: Path, path: Path) -> str:
    root = root.resolve()
    path = path.resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"file escapes mirror root {root}: {path}") from error
    if not path.is_file():
        raise ValueError(f"mirror input is not a regular file: {path}")
    return relative.as_posix()


class TraceS3Mirror:
    def __init__(self, uri: str):
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.exceptions import ClientError

        self.uri = uri.rstrip("/")
        self.location = parse_s3_uri(uri)
        self.client = boto3.client("s3")
        self.client_error = ClientError
        self.transfer_config = TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
            max_concurrency=16,
            use_threads=True,
        )

    def _head_matches(self, key: str, *, size_bytes: int, sha256: str) -> bool:
        try:
            response = self.client.head_object(Bucket=self.location.bucket, Key=key)
        except self.client_error as error:
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status == 404 or error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
                return False
            raise
        return int(response.get("ContentLength", -1)) == size_bytes and response.get("Metadata", {}).get(
            "sha256"
        ) == sha256

    def upload_file(
        self,
        path: str | Path,
        *,
        root: str | Path,
        relative_path: str | None = None,
        sha256: str | None = None,
    ) -> dict[str, Any]:
        path = Path(path)
        root = Path(root)
        relative_path = relative_path or _relative_local_path(root, path)
        key = _object_key(self.location, relative_path)
        size_bytes = path.stat().st_size
        sha256 = sha256 or sha256_file(path)
        if self._head_matches(key, size_bytes=size_bytes, sha256=sha256):
            print(f"[s3] reuse s3://{self.location.bucket}/{key}", flush=True)
            return {"key": key, "size_bytes": size_bytes, "sha256": sha256, "reused": True}
        self.client.upload_file(
            str(path),
            self.location.bucket,
            key,
            ExtraArgs={"Metadata": {"sha256": sha256}},
            Config=self.transfer_config,
        )
        if not self._head_matches(key, size_bytes=size_bytes, sha256=sha256):
            raise RuntimeError(f"S3 upload verification failed: s3://{self.location.bucket}/{key}")
        print(f"[s3] uploaded s3://{self.location.bucket}/{key}", flush=True)
        return {"key": key, "size_bytes": size_bytes, "sha256": sha256, "reused": False}

    def upload_shard(self, parquet_path: str | Path, *, root: str | Path) -> list[dict[str, Any]]:
        parquet_path = Path(parquet_path)
        manifest_path = parquet_manifest_path(parquet_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha256 = manifest.get("parquet_sha256")
        if expected_sha256 != sha256_file(parquet_path):
            raise ValueError(f"cannot mirror shard with mismatched manifest SHA256: {parquet_path}")
        return [
            self.upload_file(parquet_path, root=root, sha256=expected_sha256),
            self.upload_file(manifest_path, root=root),
        ]

    def restore_directory(self, output_dir: str | Path) -> int:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.location.prefix}/" if self.location.prefix else ""
        paginator = self.client.get_paginator("list_objects_v2")
        restored = 0
        for page in paginator.paginate(Bucket=self.location.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = str(item["Key"])
                relative = key[len(prefix) :] if prefix else key
                if not relative or relative.startswith("."):
                    continue
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(f"unsafe object key below trace prefix: {key}")
                destination = output_dir / relative_path
                if destination.exists() and destination.stat().st_size == int(item["Size"]):
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".s3-download", delete=False) as handle:
                    temporary = Path(handle.name)
                try:
                    self.client.download_file(
                        self.location.bucket,
                        key,
                        str(temporary),
                        Config=self.transfer_config,
                    )
                    metadata = self.client.head_object(Bucket=self.location.bucket, Key=key).get("Metadata", {})
                    expected_sha256 = metadata.get("sha256")
                    if expected_sha256 and sha256_file(temporary) != expected_sha256:
                        raise RuntimeError(f"downloaded S3 object failed SHA256 verification: {key}")
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
                restored += 1
                print(f"[s3] restored s3://{self.location.bucket}/{key} -> {destination}", flush=True)
        return restored


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    restore = subparsers.add_parser("restore")
    restore.add_argument("--s3-uri", required=True)
    restore.add_argument("--output-dir", type=Path, required=True)

    upload = subparsers.add_parser("upload")
    upload.add_argument("--s3-uri", required=True)
    upload.add_argument("--root", type=Path, required=True)
    upload.add_argument("paths", nargs="+", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mirror = TraceS3Mirror(args.s3_uri)
    if args.command == "restore":
        count = mirror.restore_directory(args.output_dir)
        print(f"S3_TRACE_RESTORE_COMPLETE files={count}", flush=True)
        return 0
    results = [mirror.upload_file(path, root=args.root) for path in args.paths]
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)
    print(f"S3_TRACE_UPLOAD_COMPLETE files={len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
