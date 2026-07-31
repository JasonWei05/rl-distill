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

"""Run the pinned five-task Gemma 4 out-of-domain lm-eval matrix.

The continuity default is the repository's pinned lm-evaluation-harness
submodule commit (package version 0.4.13.dev0). A different exact package
version and/or git commit can be provided explicitly. The wrapper fails closed
on identity mismatches, never uses ``--limit``, and writes a command/identity
manifest before execution.

Initialize the submodule, install it in a dedicated uv environment, and pass
that environment's ``lm_eval`` executable with ``--lm-eval-executable``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from gemma4_model_identity import require_sha256, resolve_model_identity  # noqa: E402

PINNED_HARNESS_VERSION = "0.4.13.dev0"
PINNED_HARNESS_GIT_REVISION = "f4d4b3de3ee6741a7151a9fe74945ee515262f4c"
PINNED_HARNESS_REPO = REPO_ROOT / "lm-evaluation-harness"
IMMUTABLE_REVISION_PATTERN = re.compile(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}")
OOD_GROUPS = (
    (5, ("mmlu", "winogrande", "triviaqa")),
    (10, ("hellaswag",)),
    (25, ("arc_challenge",)),
)


@dataclass(frozen=True)
class OODEvalConfig:
    model: str
    model_revision: str | None
    output_dir: str
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.7
    max_model_len: int = 4096
    batch_size: str = "auto"
    seed: int = 0


def build_ood_commands(config: OODEvalConfig, *, lm_eval_executable: str) -> list[list[str]]:
    """Build shell-free commands for the exact five-task/shot matrix."""

    model_args = [
        f"pretrained={config.model}",
        f"dtype={config.dtype}",
        f"tensor_parallel_size={config.tensor_parallel_size}",
        f"gpu_memory_utilization={config.gpu_memory_utilization}",
        f"max_model_len={config.max_model_len}",
        "add_bos_token=True",
    ]
    if config.model_revision:
        model_args.append(f"revision={config.model_revision}")

    commands = []
    for shots, tasks in OOD_GROUPS:
        commands.append(
            [
                lm_eval_executable,
                "--model",
                "vllm",
                "--model_args",
                ",".join(model_args),
                "--tasks",
                ",".join(tasks),
                "--num_fewshot",
                str(shots),
                "--batch_size",
                config.batch_size,
                "--output_path",
                str(Path(config.output_dir) / f"{shots}shot"),
                "--log_samples",
                "--seed",
                str(config.seed),
            ]
        )
    return commands


def _executable_package_identity(executable: Path) -> dict[str, str]:
    sibling_python = executable.with_name("python")
    if not sibling_python.exists():
        sibling_python = Path(sys.executable)
    script = (
        "import importlib.metadata, json, lm_eval; "
        "print(json.dumps({'version': importlib.metadata.version('lm_eval'), 'module_path': lm_eval.__file__}))"
    )
    try:
        output = subprocess.run(
            [str(sibling_python), "-c", script],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"could not determine lm_eval package version using {sibling_python}; "
            "pass an executable from the selected harness environment"
        ) from error
    identity = json.loads(output)
    if not identity.get("module_path"):
        raise RuntimeError("the selected lm_eval environment did not expose lm_eval.__file__")
    return {"version": str(identity["version"]), "module_path": str(Path(identity["module_path"]).resolve())}


def resolve_harness_identity(
    *,
    lm_eval_executable: str,
    expected_version: str,
    harness_repo: str | None,
    expected_git_revision: str | None,
) -> dict[str, Any]:
    executable = Path(lm_eval_executable).expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(
            f"lm_eval executable not found: {executable}. Install the pinned harness in a uv environment first."
        )
    package_identity = _executable_package_identity(executable)
    version = package_identity["version"]
    if version != expected_version:
        raise RuntimeError(f"lm_eval version mismatch: expected {expected_version}, found {version}")

    identity: dict[str, Any] = {
        "executable": str(executable),
        "package_version": version,
        "expected_package_version": expected_version,
        "module_path": package_identity["module_path"],
        "git_repo": None,
        "git_revision": None,
        "git_dirty": None,
        "expected_git_revision": expected_git_revision,
    }
    if expected_git_revision and not harness_repo:
        raise ValueError("--harness-git-revision requires --harness-repo")
    if harness_repo:
        repo = Path(harness_repo).expanduser().resolve()
        try:
            git_output = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "--show-toplevel", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"cannot resolve harness git revision under {repo}") from error
        if len(git_output) != 2 or Path(git_output[0]).resolve() != repo:
            raise RuntimeError(
                f"{repo} is not an initialized standalone harness checkout; initialize the git submodule first"
            )
        revision = git_output[1].strip()
        if expected_git_revision and revision != expected_git_revision:
            raise RuntimeError(f"harness git revision mismatch: expected {expected_git_revision}, found {revision}")
        module_path = Path(package_identity["module_path"])
        if not module_path.is_relative_to(repo):
            raise RuntimeError(
                f"lm_eval imports from {module_path}, outside the pinned harness checkout {repo}; "
                "use --skip-harness-git-check only for an intentionally pinned wheel"
            )
        dirty_output = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if dirty_output:
            raise RuntimeError(f"pinned harness checkout is dirty: {dirty_output.splitlines()[:5]}")
        identity.update({"git_repo": str(repo), "git_revision": revision, "git_dirty": False})
    return identity


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--expected-model-identity-sha256", default=None)
    parser.add_argument(
        "--allow-unpinned-local-model",
        action="store_true",
        help="allow a local checkpoint without an externally supplied identity (smoke diagnostics only)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lm-eval-executable", default=shutil.which("lm_eval") or "lm_eval")
    parser.add_argument("--expected-harness-version", default=PINNED_HARNESS_VERSION)
    parser.add_argument("--harness-repo", default=str(PINNED_HARNESS_REPO))
    parser.add_argument("--harness-git-revision", default=PINNED_HARNESS_GIT_REVISION)
    parser.add_argument(
        "--skip-harness-git-check",
        action="store_true",
        help="use only the exact package-version check (for a non-git wheel installation)",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    model_is_local = Path(args.model).exists()
    if not model_is_local and not (args.model_revision and IMMUTABLE_REVISION_PATTERN.fullmatch(args.model_revision)):
        raise ValueError("a remote model requires an immutable 40/64-hex --model-revision")
    model_identity = resolve_model_identity(args.model, args.model_revision)
    expected_model_identity = args.expected_model_identity_sha256
    if model_is_local and expected_model_identity is None and not args.allow_unpinned_local_model:
        raise ValueError(
            "a local model requires --expected-model-identity-sha256; "
            "use --allow-unpinned-local-model only for an explicit smoke diagnostic"
        )
    if expected_model_identity is not None:
        expected_model_identity = require_sha256(
            expected_model_identity,
            "--expected-model-identity-sha256",
        )
        if model_identity["model_identity_sha256"] != expected_model_identity:
            raise ValueError(
                "model identity does not match --expected-model-identity-sha256: "
                f"{model_identity['model_identity_sha256']} != {expected_model_identity}"
            )
    harness_repo = None if args.skip_harness_git_check else args.harness_repo
    harness_git_revision = None if args.skip_harness_git_check else args.harness_git_revision
    identity = resolve_harness_identity(
        lm_eval_executable=args.lm_eval_executable,
        expected_version=args.expected_harness_version,
        harness_repo=harness_repo,
        expected_git_revision=harness_git_revision,
    )
    config = OODEvalConfig(
        model=args.model,
        model_revision=args.model_revision,
        output_dir=args.output_dir,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    commands = build_ood_commands(config, lm_eval_executable=identity["executable"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "harness": identity,
        "model_identity": model_identity,
        "config": asdict(config),
        "tasks": [{"num_fewshot": shots, "tasks": list(tasks)} for shots, tasks in OOD_GROUPS],
        "commands": commands,
        "dry_run": args.dry_run,
        "notes": [
            "direct lm_eval vllm path; no chat template",
            "full benchmark splits only; --limit is intentionally unsupported",
            "raw PT evaluation uses add_bos_token=True for continuity with the Gemma 3 replication",
        ],
    }
    manifest_path = output_dir / "ood_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
