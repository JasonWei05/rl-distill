#!/usr/bin/env python3
"""Download one immutable Hugging Face model snapshot into a plain directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def _has_model_files(path: Path) -> bool:
    return path.joinpath("config.json").is_file() and (
        path.joinpath("model.safetensors.index.json").is_file() or any(path.glob("*.safetensors"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    source = {
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "revision": args.revision,
    }
    source_path = output / ".hf_snapshot_source.json"
    if _has_model_files(output) and not args.overwrite:
        if not source_path.is_file():
            raise RuntimeError(f"existing model lacks pinned source provenance: {source_path}")
        if json.loads(source_path.read_text(encoding="utf-8")) != source:
            raise RuntimeError(f"existing model does not match the requested immutable snapshot: {output}")
        print(f"model already present: {output}")
        return

    staging = output.parent / f".{output.name}.download-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            local_dir=str(staging),
            token=os.environ.get("HF_TOKEN"),
        )
        if not _has_model_files(staging):
            raise FileNotFoundError(f"downloaded snapshot has no complete model files: {staging}")
        (staging / ".hf_snapshot_source.json").write_text(
            json.dumps(source, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"downloaded {args.repo_id}@{args.revision} -> {output}")


if __name__ == "__main__":
    main()
