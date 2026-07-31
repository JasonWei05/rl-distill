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

"""Run a pinned Gemma 4 out-of-domain lm-eval profile.

The default preserves the prior five-task Gemma 3 continuity matrix. The
``gemma4-report`` profile runs MMLU-Pro, GPQA-Diamond, and the registered
14,042-item MMMLU subset using the repository's pinned lm-evaluation-harness
submodule commit (package version 0.4.13.dev0). The wrapper fails closed on
identity/revision mismatches, never uses ``--limit``, and writes a
command/identity manifest before execution.

Initialize the submodule, install it in a dedicated uv environment, and pass
that environment's ``lm_eval`` executable with ``--lm-eval-executable``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
CONTINUITY_OOD_GROUPS = (
    ("5shot", 5, ("mmlu", "winogrande", "triviaqa")),
    ("10shot", 10, ("hellaswag",)),
    ("25shot", 25, ("arc_challenge",)),
)
OOD_GROUPS = tuple((shots, tasks) for _, shots, tasks in CONTINUITY_OOD_GROUPS)
GEMMA4_REPORT_GPQA_TASKS = (
    "gpqa_diamond_cot_n_shot",
    "gpqa_diamond_n_shot",
)
GEMMA4_MMMLU_TASK_GROUP = "gemma4_mmmlu14k"
GEMMA4_REPORT_BENCHMARKS = ("mmlu_pro", "gpqa", "mmmlu14k")
DEFAULT_GEMMA4_MMMLU_TASK_DIR = Path("/lambda/nfs/Jason-scale/rl-distill-evals/gemma4-three-model/data/mmmlu14k_tasks")
DEFAULT_GEMMA4_MMMLU_MANIFEST = DEFAULT_GEMMA4_MMMLU_TASK_DIR / "manifest.json"
GEMMA4_REPORT_DATASET_REVISIONS = {
    "TIGER-Lab/MMLU-Pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "Idavidrein/gpqa": "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
    "openai/MMMLU": "325a01dc3e173cac1578df94120499aaca2e2504",
}
MMMLU_LOCALES = (
    "AR_XY",
    "BN_BD",
    "DE_DE",
    "ES_LA",
    "FR_FR",
    "HI_IN",
    "ID_ID",
    "IT_IT",
    "JA_JP",
    "KO_KR",
    "PT_BR",
    "SW_KE",
    "YO_NG",
    "ZH_CN",
)
EXPECTED_MMMLU_TASK_FILES = 14 * 57 + 14 + 3


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
    profile: str = "gemma3-continuity"
    gpqa_task: str = "gpqa_diamond_cot_n_shot"
    mmmlu_task_dir: str | None = str(DEFAULT_GEMMA4_MMMLU_TASK_DIR)
    benchmarks: tuple[str, ...] = GEMMA4_REPORT_BENCHMARKS


def _profile_groups(config: OODEvalConfig) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
    if config.profile == "gemma3-continuity":
        return CONTINUITY_OOD_GROUPS
    if config.profile == "gemma4-report":
        if config.gpqa_task not in GEMMA4_REPORT_GPQA_TASKS:
            raise ValueError(f"unsupported GPQA task: {config.gpqa_task}")
        invalid = set(config.benchmarks) - set(GEMMA4_REPORT_BENCHMARKS)
        if invalid:
            raise ValueError(f"unsupported Gemma 4 report benchmarks: {sorted(invalid)}")
        groups = {
            "mmlu_pro": ("mmlu_pro_5shot_cot", 5, ("mmlu_pro",)),
            "gpqa": (f"{config.gpqa_task}_5shot", 5, (config.gpqa_task,)),
            "mmmlu14k": ("mmmlu14k_5shot", 5, (GEMMA4_MMMLU_TASK_GROUP,)),
        }
        return tuple(groups[name] for name in GEMMA4_REPORT_BENCHMARKS if name in config.benchmarks)
    raise ValueError(f"unsupported OOD profile: {config.profile}")


def build_ood_commands(config: OODEvalConfig, *, lm_eval_executable: str) -> list[list[str]]:
    """Build shell-free commands for the selected exact task/shot matrix."""

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
    for output_name, shots, tasks in _profile_groups(config):
        command = [
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
            str(Path(config.output_dir) / output_name),
            "--log_samples",
            "--seed",
            str(config.seed),
        ]
        if GEMMA4_MMMLU_TASK_GROUP in tasks:
            if config.mmmlu_task_dir is None:
                raise ValueError("the reduced-MMMLU task requires mmmlu_task_dir")
            command.extend(["--include_path", config.mmmlu_task_dir])
        commands.append(command)
    return commands


def resolve_mmmlu14k_manifest(manifest_path: str | Path, task_dir: str | Path) -> dict[str, Any]:
    """Validate the generated reduced-MMMLU task tree before lm-eval starts."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    root = Path(task_dir).expanduser().resolve()
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read reduced-MMMLU manifest {manifest_file}: {error}") from error
    if manifest.get("protocol") != "gemma4_mmmlu14k_v1":
        raise ValueError("unexpected reduced-MMMLU manifest protocol")
    if manifest.get("schema_version") != 1:
        raise ValueError("unexpected reduced-MMMLU manifest schema version")
    if manifest.get("task_group") != GEMMA4_MMMLU_TASK_GROUP:
        raise ValueError("unexpected reduced-MMMLU task group")
    if Path(str(manifest.get("task_dir", ""))).expanduser().resolve() != root:
        raise ValueError("reduced-MMMLU manifest task_dir does not match the selected task directory")
    if manifest.get("harness_revision") != PINNED_HARNESS_GIT_REVISION:
        raise ValueError("reduced-MMMLU manifest was not generated from the pinned harness revision")
    source = manifest.get("source")
    if source != {
        "repo_id": "openai/MMMLU",
        "revision": GEMMA4_REPORT_DATASET_REVISIONS["openai/MMMLU"],
        "rows_per_locale": 14_042,
    }:
        raise ValueError("reduced-MMMLU manifest has an unexpected pinned source")
    assignment = manifest.get("assignment", {})
    if assignment.get("total_evaluation_rows") != 14_042:
        raise ValueError("reduced-MMMLU must contain exactly 14,042 evaluation rows")
    locale_counts = assignment.get("locale_counts", {})
    if set(locale_counts) != set(MMMLU_LOCALES) or set(locale_counts.values()) != {1003}:
        raise ValueError("reduced-MMMLU must assign exactly 1,003 rows to each of 14 locales")
    subjects = assignment.get("subjects", [])
    if not isinstance(subjects, list) or len(subjects) != 57 or len(set(subjects)) != 57:
        raise ValueError("reduced-MMMLU must contain all 57 MMLU subjects")
    if assignment.get("locales") != list(MMMLU_LOCALES):
        raise ValueError("reduced-MMMLU locale order does not match the registered allocation")
    disagreements = assignment.get("answer_key_disagreements_vs_first_locale")
    if not isinstance(disagreements, dict) or set(disagreements) != set(MMMLU_LOCALES):
        raise ValueError("reduced-MMMLU manifest has incomplete answer-key agreement records")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 14_042
        for value in disagreements.values()
    ):
        raise ValueError("reduced-MMMLU manifest has invalid cross-locale answer-key disagreement counts")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != EXPECTED_MMMLU_TASK_FILES:
        raise ValueError(f"reduced-MMMLU manifest must register {EXPECTED_MMMLU_TASK_FILES} task files")
    registered_paths: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("reduced-MMMLU file manifest entries must be objects")
        relative = Path(str(entry.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or str(relative) in registered_paths:
            raise ValueError(f"invalid or duplicate reduced-MMMLU task path: {relative}")
        registered_paths.add(str(relative))
        if not isinstance(entry.get("size"), int) or entry["size"] < 0:
            raise ValueError(f"invalid reduced-MMMLU task file size: {relative}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
            raise ValueError(f"invalid reduced-MMMLU task file SHA256: {relative}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"missing reduced-MMMLU task file: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry.get("sha256") or path.stat().st_size != entry.get("size"):
            raise ValueError(f"reduced-MMMLU task file identity mismatch: {path}")
    actual_paths = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and "__pycache__" not in path.relative_to(root).parts
        and path.suffix != ".pyc"
    }
    if actual_paths != registered_paths:
        raise ValueError("reduced-MMMLU task tree contains unregistered or missing files")
    return {
        "manifest": str(manifest_file),
        "task_dir": str(root),
        "task_group": manifest["task_group"],
        "total_evaluation_rows": assignment["total_evaluation_rows"],
        "locale_counts": locale_counts,
        "subject_count": len(assignment["subjects"]),
        "registered_file_count": len(files),
    }


def resolve_dataset_revisions(
    expected: dict[str, str],
    *,
    require_current: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Fail closed if a harness dataset's current Hub revision has drifted."""

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    resolved = {}
    for repo_id, expected_revision in expected.items():
        actual_revision = str(api.dataset_info(repo_id, revision=expected_revision).sha)
        if actual_revision != expected_revision:
            raise RuntimeError(
                f"dataset revision mismatch for {repo_id}: expected {expected_revision}, found {actual_revision}"
            )
        if repo_id in require_current:
            current_revision = str(api.dataset_info(repo_id).sha)
            if current_revision != expected_revision:
                raise RuntimeError(
                    f"the native harness task for {repo_id} would load current revision {current_revision}, "
                    f"not pinned revision {expected_revision}"
                )
        if repo_id == "Idavidrein/gpqa":
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=expected_revision,
                filename="gpqa_diamond.csv",
            )
        resolved[repo_id] = actual_revision
    return resolved


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
    parser.add_argument(
        "--profile",
        choices=("gemma3-continuity", "gemma4-report"),
        default="gemma3-continuity",
    )
    parser.add_argument(
        "--gpqa-task",
        choices=GEMMA4_REPORT_GPQA_TASKS,
        default="gpqa_diamond_cot_n_shot",
        help="the CoT variant is recommended; the likelihood variant preserves direct-answer continuity",
    )
    parser.add_argument("--mmmlu-task-dir", type=Path, default=DEFAULT_GEMMA4_MMMLU_TASK_DIR)
    parser.add_argument("--mmmlu-manifest", type=Path, default=DEFAULT_GEMMA4_MMMLU_MANIFEST)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=GEMMA4_REPORT_BENCHMARKS,
        default=list(GEMMA4_REPORT_BENCHMARKS),
    )
    parser.add_argument(
        "--skip-dataset-revision-check",
        action="store_true",
        help="disable Hub revision checks (offline diagnostics only)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
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
        profile=args.profile,
        gpqa_task=args.gpqa_task,
        mmmlu_task_dir=str(args.mmmlu_task_dir.resolve()) if args.profile == "gemma4-report" else None,
        benchmarks=tuple(args.benchmarks),
    )
    commands = build_ood_commands(config, lm_eval_executable=identity["executable"])
    dataset_revisions = None
    mmmlu14k_identity = None
    if args.profile == "gemma4-report" and not args.skip_dataset_revision_check:
        repos_by_benchmark = {
            "mmlu_pro": "TIGER-Lab/MMLU-Pro",
            "gpqa": "Idavidrein/gpqa",
            "mmmlu14k": "openai/MMMLU",
        }
        selected_repos = {repos_by_benchmark[benchmark] for benchmark in args.benchmarks}
        expected_revisions = {
            repo_id: revision
            for repo_id, revision in GEMMA4_REPORT_DATASET_REVISIONS.items()
            if repo_id in selected_repos
        }
        native_repos = frozenset(selected_repos & {"TIGER-Lab/MMLU-Pro", "Idavidrein/gpqa"})
        dataset_revisions = resolve_dataset_revisions(expected_revisions, require_current=native_repos)
    if args.profile == "gemma4-report" and "mmmlu14k" in args.benchmarks:
        mmmlu14k_identity = resolve_mmmlu14k_manifest(args.mmmlu_manifest, args.mmmlu_task_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "harness": identity,
        "model_identity": model_identity,
        "config": asdict(config),
        "tasks": [
            {"output_name": output_name, "num_fewshot": shots, "tasks": list(tasks)}
            for output_name, shots, tasks in _profile_groups(config)
        ],
        "dataset_revisions": dataset_revisions,
        "mmmlu14k_identity": mmmlu14k_identity,
        "commands": commands,
        "dry_run": args.dry_run,
        "notes": [
            "direct lm_eval vllm path; no chat template",
            "full benchmark splits only; --limit is intentionally unsupported",
            "raw PT evaluation uses add_bos_token=True for continuity with the Gemma 3 replication",
            "task-native generation/likelihood settings are preserved; only num_fewshot is explicit",
            "MMMLU is the pinned 14,042-item subject/locale-balanced task, never the full 196,588-row group",
        ],
    }
    manifest_path = output_dir / "ood_eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    child_env = os.environ.copy()
    executable_bin = str(Path(identity["executable"]).parent)
    path_entries = child_env.get("PATH", "").split(os.pathsep)
    if executable_bin not in path_entries:
        child_env["PATH"] = os.pathsep.join([executable_bin, *path_entries])
    for command in commands:
        subprocess.run(command, check=True, env=child_env)


if __name__ == "__main__":
    main()
