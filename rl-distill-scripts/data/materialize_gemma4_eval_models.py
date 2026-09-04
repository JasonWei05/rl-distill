#!/usr/bin/env python3
"""Resolve the immutable Gemma 4 eval source registry into local HF models.

The default is a no-download command plan. ``--execute`` performs the downloads,
content-binds every local model, and writes the resolved registry consumed by
the math and OOD schedulers.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gemma4_eval_registry import (  # noqa: E402
    RegisteredModel,
    canonical_json_sha256,
    load_source_registry,
    select_models,
)
from gemma4_model_identity import inspect_local_hf_model  # noqa: E402

DEFAULT_SOURCE_REGISTRY = SCRIPTS_DIR / "config/gemma4_rl_distill_eval_sources.json"
DEFAULT_OUTPUT_ROOT = Path("/lambda/nfs/Jason-scale/rl-distill-evals/gemma4-rl-distill-base-v1/models")


def materialization_command(model: RegisteredModel, output: Path, python_executable: str) -> list[str]:
    source = model.source
    if source["type"] == "hf_snapshot":
        return [
            python_executable,
            str(SCRIPT_DIR / "download_hf_model_snapshot.py"),
            "--repo-id",
            source["repo_id"],
            "--revision",
            source["revision"],
            "--output-dir",
            str(output),
        ]
    if source["type"] == "hf_subfolder":
        return [
            python_executable,
            str(SCRIPT_DIR / "download_hf_subfolder.py"),
            "--repo-id",
            source["repo_id"],
            "--revision",
            source["revision"],
            "--subfolder",
            source["subfolder"],
            "--metadata-repo",
            source["metadata_repo"],
            "--metadata-revision",
            source["metadata_revision"],
            "--output-dir",
            str(output),
        ]
    return [
        "aws",
        "s3",
        "sync",
        source["uri"],
        str(output),
        "--only-show-errors",
    ]


def _read_s3_completion(uri: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["aws", "s3", "cp", uri, "-", "--no-progress"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError(f"S3 completion receipt must be an object: {uri}")
    return value


def _execute_one(model: RegisteredModel, output: Path, python_executable: str) -> dict[str, Any]:
    source = model.source
    receipt_path = output / ".eval_source.json"
    if output.is_dir():
        if not receipt_path.is_file():
            raise RuntimeError(f"existing model lacks source receipt: {output}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("source") != source:
            raise RuntimeError(f"existing model has different immutable source: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        if source["type"] == "s3_hf_export":
            completion = _read_s3_completion(source["completion_uri"])
            if completion.get("global_step") != source["expected_global_step"]:
                raise ValueError(f"unexpected completion global_step for {model.tag}: {completion.get('global_step')}")
            staging = output.parent / f".{output.name}.partial-{os.getpid()}"
            if staging.exists():
                raise FileExistsError(f"stale materialization directory exists: {staging}")
            staging.mkdir()
            try:
                subprocess.run(materialization_command(model, staging, python_executable), check=True)
                (staging / ".eval_source.json").write_text(
                    json.dumps(
                        {"source": source, "completion_receipt_sha256": canonical_json_sha256(completion)},
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(staging, output)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        else:
            subprocess.run(materialization_command(model, output, python_executable), check=True)
            (output / ".eval_source.json").write_text(
                json.dumps({"source": source}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    identity = inspect_local_hf_model(output)
    return {
        **asdict(model),
        "model": str(output.resolve()),
        "expected_model_identity_sha256": identity.model_identity_sha256,
        "weight_content_sha256": identity.weight_content_sha256,
        "weight_content_kind": identity.weight_content_kind,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", type=Path, default=DEFAULT_SOURCE_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_payload, all_models = load_source_registry(args.source_registry)
    models = select_models(all_models, args.models)
    output_root = args.output_root.expanduser().resolve()
    plan = {
        "schema_version": 1,
        "protocol": "gemma4_eval_model_materialization_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "source_registry": str(args.source_registry.resolve()),
        "source_registry_sha256": canonical_json_sha256(source_payload),
        "resolved_registry": str(output_root / "resolved_model_registry.json"),
        "models": [
            {
                "tag": model.tag,
                "output": str(output_root / model.tag),
                "command": materialization_command(model, output_root / model.tag, args.python_executable),
            }
            for model in models
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "materialization_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.execute:
        return 0

    resolved_models = []
    for model in models:
        command = materialization_command(model, output_root / model.tag, args.python_executable)
        print(f"[{model.tag}] {shlex.join(command)}", flush=True)
        resolved_models.append(_execute_one(model, output_root / model.tag, args.python_executable))
    resolved = {
        "schema_version": 1,
        "protocol": "gemma4_rl_distill_eval_models_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_registry": str(args.source_registry.resolve()),
        "source_registry_sha256": canonical_json_sha256(source_payload),
        "models": resolved_models,
    }
    (output_root / "resolved_model_registry.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
