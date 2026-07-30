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

"""Upload finalized NeMo-RL model checkpoints to Hugging Face.

Only the consolidated model, tokenizer, run config, and training metadata are
uploaded. Optimizer and dataloader state remain local for crash recovery.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from hf_base_canonicalizer import (
    BASE_MODEL_ID,
    BASE_REVISION,
    canonicalize_pinned_base,
    sha256_file,
)
from hf_checkpoint_delta import create_delta
from huggingface_hub import HfApi
from safetensors import safe_open
from safetensors.torch import save_file

STOP_REQUESTED = False


def log(message: str) -> None:
    print(
        f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {message}",
        flush=True,
    )


def request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def load_state(path: Path, repo_id: str) -> dict[str, Any]:
    if not path.exists():
        return {"repo_id": repo_id, "uploaded_steps": [], "commits": {}}
    with path.open() as state_file:
        state = json.load(state_file)
    if state.get("repo_id") != repo_id:
        raise ValueError(f"Upload state belongs to {state.get('repo_id')!r}, not {repo_id!r}")
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as state_file:
        json.dump(state, state_file, indent=2, sort_keys=True)
        state_file.write("\n")
    os.replace(tmp_path, path)


def checkpoint_model_root(step_dir: Path) -> Path | None:
    """Return the HF-loadable consolidated model directory, if finalized."""
    candidate = step_dir / "policy" / "weights" / "model" / "consolidated"
    if not candidate.is_dir():
        return None
    if not (candidate / "config.json").is_file():
        return None
    if not any(candidate.glob("*.safetensors")):
        return None
    return candidate


def checkpoint_is_ready(step_dir: Path) -> bool:
    tokenizer_dir = step_dir / "policy" / "tokenizer"
    return (
        (step_dir / "training_info.json").is_file()
        and (step_dir / "config.yaml").is_file()
        and tokenizer_dir.is_dir()
        and (tokenizer_dir / "tokenizer_config.json").is_file()
        and checkpoint_model_root(step_dir) is not None
    )


def _set_bfloat16_dtype(config: Any) -> None:
    if isinstance(config, dict):
        for key, value in config.items():
            if key in {"dtype", "torch_dtype"} and value in {"float32", "float"}:
                config[key] = "bfloat16"
            else:
                _set_bfloat16_dtype(value)
    elif isinstance(config, list):
        for value in config:
            _set_bfloat16_dtype(value)


def _staged_checkpoint_is_ready(staging_dir: Path) -> bool:
    model_root = staging_dir / "policy" / "weights" / "model" / "consolidated"
    return (
        (staging_dir / "upload_manifest.json").is_file()
        and (staging_dir / "training_info.json").is_file()
        and (staging_dir / "config.yaml").is_file()
        and (staging_dir / "policy" / "tokenizer" / "tokenizer_config.json").is_file()
        and (model_root / "config.json").is_file()
        and any(model_root.glob("*.safetensors"))
    )


def prepare_upload_staging(step_dir: Path) -> Path:
    """Create a BF16, HF-loadable staging tree for a finalized checkpoint.

    NeMo's FSDP2 consolidation writes FP32 weights even though policy forward
    and rollout use BF16. Uploading those FP32 weights doubles every checkpoint
    without adding inference fidelity. Keep the FP32 and optimizer checkpoint
    locally for exact recovery, and upload a BF16 model matching execution
    precision.
    """

    staging_dir = step_dir / ".hf_upload"
    if _staged_checkpoint_is_ready(staging_dir):
        return staging_dir

    source_root = checkpoint_model_root(step_dir)
    if source_root is None:
        raise FileNotFoundError(f"No finalized consolidated HF checkpoint under {step_dir}")

    source_files = sorted(source_root.glob("*.safetensors"))
    if not source_files:
        raise FileNotFoundError(f"No safetensors files under {source_root}")

    temp_dir = step_dir / ".hf_upload.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    model_root = temp_dir / "policy" / "weights" / "model" / "consolidated"
    tokenizer_target = temp_dir / "policy" / "tokenizer"
    model_root.mkdir(parents=True)

    shutil.copy2(step_dir / "training_info.json", temp_dir / "training_info.json")
    shutil.copy2(step_dir / "config.yaml", temp_dir / "config.yaml")
    shutil.copytree(step_dir / "policy" / "tokenizer", tokenizer_target)

    config = json.loads((source_root / "config.json").read_text())
    _set_bfloat16_dtype(config)
    (model_root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    generation_config = source_root / "generation_config.json"
    if generation_config.is_file():
        shutil.copy2(generation_config, model_root / generation_config.name)

    tensors: dict[str, torch.Tensor] = {}
    source_dtypes: set[str] = set()
    for source_file in source_files:
        with safe_open(source_file, framework="pt", device="cpu") as source:
            for name in source.keys():
                tensor = source.get_tensor(name)
                source_dtypes.add(str(tensor.dtype).removeprefix("torch."))
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=torch.bfloat16)
                tensors[name] = tensor.contiguous()

    output_file = model_root / "model.safetensors"
    save_file(tensors, output_file, metadata={"format": "pt"})
    del tensors

    with safe_open(output_file, framework="pt", device="cpu") as staged:
        staged_keys = list(staged.keys())
        staged_dtypes = {str(staged.get_slice(name).get_dtype()) for name in staged_keys}
    if staged_dtypes != {"BF16"}:
        raise RuntimeError(f"Expected an all-BF16 staged model, got {sorted(staged_dtypes)}")

    index = {
        "metadata": {"total_size": output_file.stat().st_size},
        "weight_map": {name: output_file.name for name in staged_keys},
    }
    (model_root / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    manifest = {
        "format": "hf_bfloat16",
        "source_model_root": str(source_root.relative_to(step_dir)),
        "source_dtypes": sorted(source_dtypes),
        "uploaded_dtype": "bfloat16",
        "tensor_count": len(staged_keys),
        "model_bytes": output_file.stat().st_size,
        "training_info": json.loads((step_dir / "training_info.json").read_text()),
    }
    (temp_dir / "upload_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    os.replace(temp_dir, staging_dir)
    return staging_dir


def _delta_staging_is_ready(staging_dir: Path) -> bool:
    delta_dir = staging_dir / "delta"
    return (
        (staging_dir / "training_info.json").is_file()
        and (staging_dir / "config.yaml").is_file()
        and (delta_dir / "delta_manifest.json").is_file()
        and (delta_dir / "changed_mask.bitset.zst").is_file()
        and (delta_dir / "add_values.u16.zst").is_file()
        and (delta_dir / "reconstruct_delta.py").is_file()
    )


def _anchor_delta_staging_is_ready(staging_dir: Path) -> bool:
    return _delta_staging_is_ready(staging_dir) and (staging_dir / "delta" / "canonicalize_base.py").is_file()


def prepare_anchor_delta_staging(
    step_dir: Path,
    step: int,
    *,
    canonical_base_path: Path | None = None,
) -> Path:
    """Create an exact step anchor delta against the pinned public base."""

    staging_dir = step_dir / ".hf_anchor_delta_upload"
    if _anchor_delta_staging_is_ready(staging_dir):
        return staging_dir

    target_staging = prepare_upload_staging(step_dir)
    model_relative = Path("policy/weights/model/consolidated/model.safetensors")
    model_root = target_staging / model_relative.parent
    target_model = target_staging / model_relative
    target_index = model_root / "model.safetensors.index.json"

    if canonical_base_path is None:
        base_dir = step_dir.parent / ".hf_base_anchor"
        canonical_base_path = base_dir / "model.safetensors"
        metadata_path = base_dir / "canonical_manifest.json"
        if not canonical_base_path.is_file() or not metadata_path.is_file():
            base_dir.mkdir(parents=True, exist_ok=True)
            metadata = canonicalize_pinned_base(target_index, canonical_base_path)
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        else:
            metadata = json.loads(metadata_path.read_text())
    else:
        metadata = {
            "base_model_id": BASE_MODEL_ID,
            "base_revision": BASE_REVISION,
            "canonical_bytes": canonical_base_path.stat().st_size,
            "canonical_sha256": sha256_file(canonical_base_path),
        }

    if canonical_base_path.stat().st_size != target_model.stat().st_size:
        raise ValueError("Canonical base and target checkpoint sizes differ")

    temp_dir = step_dir / ".hf_anchor_delta_upload.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    shutil.copytree(
        target_staging,
        temp_dir,
        ignore=shutil.ignore_patterns("model.safetensors"),
    )
    delta_dir = temp_dir / "delta"
    delta_dir.mkdir(parents=True)
    shutil.copy2(
        Path(__file__).with_name("hf_checkpoint_delta.py"),
        delta_dir / "reconstruct_delta.py",
    )
    shutil.copy2(
        Path(__file__).with_name("hf_base_canonicalizer.py"),
        delta_dir / "canonicalize_base.py",
    )

    manifest = create_delta(canonical_base_path, target_model, delta_dir)
    manifest.update(
        {
            "base_model_id": metadata["base_model_id"],
            "base_revision": metadata["base_revision"],
            "base_canonical_sha256": metadata["canonical_sha256"],
            "target_step": step,
            "target_model_path": model_relative.as_posix(),
            "training_info": json.loads((step_dir / "training_info.json").read_text()),
        }
    )
    (delta_dir / "delta_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    upload_manifest = json.loads((temp_dir / "upload_manifest.json").read_text())
    upload_manifest.update(
        {
            "format": "hf_bfloat16_public_base_delta",
            "base_model_id": metadata["base_model_id"],
            "base_revision": metadata["base_revision"],
            "base_canonical_sha256": metadata["canonical_sha256"],
            "delta_bytes": manifest["delta_bytes"],
        }
    )
    (temp_dir / "upload_manifest.json").write_text(json.dumps(upload_manifest, indent=2, sort_keys=True) + "\n")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    os.replace(temp_dir, staging_dir)
    return staging_dir


def prepare_delta_staging(step_dir: Path, step: int, interval: int) -> Path:
    """Create an exact compressed delta against the preceding checkpoint."""

    staging_dir = step_dir / ".hf_delta_upload"
    if _delta_staging_is_ready(staging_dir):
        return staging_dir
    base_step = step - interval
    if base_step <= 0:
        raise ValueError(f"No positive base checkpoint for step {step}")
    base_step_dir = step_dir.parent / f"step_{base_step}"
    if not checkpoint_is_ready(base_step_dir):
        raise FileNotFoundError(f"Base checkpoint step {base_step} is not finalized")

    base_staging = prepare_upload_staging(base_step_dir)
    target_staging = prepare_upload_staging(step_dir)
    model_relative = Path("policy/weights/model/consolidated/model.safetensors")
    base_model = base_staging / model_relative
    target_model = target_staging / model_relative

    temp_dir = step_dir / ".hf_delta_upload.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    delta_dir = temp_dir / "delta"
    delta_dir.mkdir(parents=True)
    shutil.copy2(step_dir / "training_info.json", temp_dir / "training_info.json")
    shutil.copy2(step_dir / "config.yaml", temp_dir / "config.yaml")
    shutil.copy2(Path(__file__).with_name("hf_checkpoint_delta.py"), delta_dir / "reconstruct_delta.py")

    manifest = create_delta(base_model, target_model, delta_dir)
    manifest.update(
        {
            "base_step": base_step,
            "target_step": step,
            "base_repo_model_path": f"reconstruct step {base_step} first",
            "target_model_path": model_relative.as_posix(),
            "training_info": json.loads((step_dir / "training_info.json").read_text()),
        }
    )
    (delta_dir / "delta_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    os.replace(temp_dir, staging_dir)
    return staging_dir


def discover_steps(checkpoint_dir: Path, interval: int) -> list[tuple[int, Path]]:
    checkpoints = []
    for step_dir in checkpoint_dir.glob("step_*"):
        try:
            step = int(step_dir.name.removeprefix("step_"))
        except ValueError:
            continue
        if step > 0 and step % interval == 0 and checkpoint_is_ready(step_dir):
            checkpoints.append((step, step_dir))
    return sorted(checkpoints)


def upload_step(
    api: HfApi,
    *,
    repo_id: str,
    step: int,
    step_dir: Path,
    interval: int,
) -> str:
    staging_dir = (
        prepare_anchor_delta_staging(step_dir, step)
        if step == interval
        else prepare_delta_staging(step_dir, step, interval)
    )
    path_in_repo = f"checkpoints/step_{step}"
    log(f"Uploading step {step} to {repo_id}/{path_in_repo}")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=staging_dir,
        path_in_repo=path_in_repo,
        commit_message=f"Upload NeMo-RL checkpoint step {step}",
    )
    commit_id = getattr(commit, "oid", None) or getattr(commit, "commit_id", None)
    log(f"Uploaded step {step}; commit={commit_id}")
    return str(commit_id or "unknown")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID"))
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--init-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    load_dotenv(repo_root / ".env")
    args = parse_args()
    if not args.repo_id:
        raise ValueError("Pass --repo-id or set HF_REPO_ID")
    if args.interval <= 0:
        raise ValueError("--interval must be positive")

    token = (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    )
    if not token:
        raise RuntimeError("No Hugging Face token found in the environment or .env")

    checkpoint_dir = args.checkpoint_dir.resolve()
    state_file = args.state_file or checkpoint_dir / ".hf_upload_state.json"
    api = HfApi(token=token)
    api.create_repo(
        args.repo_id,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
    )
    api.update_repo_settings(
        args.repo_id,
        repo_type="model",
        private=not args.public,
    )
    log(f"Hugging Face repo ready: {args.repo_id} (private={not args.public})")
    if args.init_only:
        return 0

    state = load_state(state_file, args.repo_id)
    uploaded_steps = {int(step) for step in state.get("uploaded_steps", [])}
    backoff_seconds = 30

    while not STOP_REQUESTED:
        pending = [
            (step, step_dir)
            for step, step_dir in discover_steps(checkpoint_dir, args.interval)
            if step not in uploaded_steps
        ]
        for step, step_dir in pending:
            try:
                commit_id = upload_step(
                    api,
                    repo_id=args.repo_id,
                    step=step,
                    step_dir=step_dir,
                    interval=args.interval,
                )
            except Exception as error:
                log(f"Upload failed for step {step}: {type(error).__name__}: {error}")
                if args.once:
                    return 1
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 600)
                break
            else:
                backoff_seconds = 30
                uploaded_steps.add(step)
                state["uploaded_steps"] = sorted(uploaded_steps)
                state.setdefault("commits", {})[str(step)] = commit_id
                state["updated_at"] = datetime.now(UTC).isoformat()
                save_state(state_file, state)
        if args.once:
            return 0
        time.sleep(args.poll_seconds)

    log("Stop requested")
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    sys.exit(main())
