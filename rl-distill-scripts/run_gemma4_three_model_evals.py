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

"""Build or run the pinned three-model Gemma 4 math/OOD evaluation matrix.

The default mode only writes and prints an auditable command manifest. Pass
``--preflight`` to validate all dataset/model identities and the lm-eval
configuration without model generation, or ``--execute`` to run the matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from eval_math_passk import load_dataset_protocol_manifest  # noqa: E402
from gemma4_model_identity import resolve_model_identity  # noqa: E402

DEFAULT_DATA_MANIFEST = Path("/lambda/nfs/Jason-scale/rl-distill-evals/gemma4-three-model/data/math_eval_manifest.json")
DEFAULT_OUTPUT_ROOT = Path("/lambda/nfs/Jason-scale/rl-distill-evals/gemma4-three-model/results")
DEFAULT_BASE_MODEL = Path(
    "/tmp/hf_cache/models--google--gemma-4-E2B/snapshots/d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
)
DEFAULT_DISTILLED_MODEL = Path(
    "/lambda/nfs/Jason-scale/rl-distill-checkpoints/e4b-rl100-to-e2b-topk128-750-seed42/global_step_750/huggingface"
)
DEFAULT_RL_MODEL = Path("/lambda/nfs/Jason-scale/rl-distill-evals/gemma4-three-model/models/nemorl-e2b-rl-step125-vllm")
PINNED_MODEL_IDENTITIES = {
    "base_e2b": "bde9e800223cdd62228ce39e0305398f6ada05b98adaf438b0b3d3d3c3015561",
    "distilled_e2b_step750": "beef12a146c3373a049467b42412520929228849570ba4cf495107e4597add03",
}
DEFAULT_MATH_DATASETS = ("math500", "gsm8k", "olympiadbench", "minervamath", "aime2025", "aime2026")
DEFAULT_K_VALUES = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    model: str
    expected_identity_sha256: str


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_model_spec(tag: str, model_path: Path, expected_identity: str | None) -> ModelSpec:
    identity = resolve_model_identity(str(model_path))
    actual = str(identity["model_identity_sha256"])
    if expected_identity is not None and actual != expected_identity:
        raise ValueError(f"model identity mismatch for {tag}: expected {expected_identity}, found {actual}")
    if tag == "rl_e2b_step125":
        materialization_path = model_path / "materialization_manifest.json"
        materialization = _load_json(materialization_path)
        registered = str(materialization.get("output", {}).get("model_identity", {}).get("model_identity_sha256"))
        if registered != actual:
            raise ValueError(
                f"RL materialization identity mismatch: manifest records {registered}, current model is {actual}"
            )
    return ModelSpec(tag=tag, model=str(model_path.resolve()), expected_identity_sha256=actual)


def resolve_models(args: argparse.Namespace) -> list[ModelSpec]:
    requested = set(args.models)
    candidates = (
        ("base_e2b", args.base_model, PINNED_MODEL_IDENTITIES["base_e2b"]),
        (
            "distilled_e2b_step750",
            args.distilled_model,
            PINNED_MODEL_IDENTITIES["distilled_e2b_step750"],
        ),
        ("rl_e2b_step125", args.rl_model, None),
    )
    return [_resolve_model_spec(tag, Path(model), expected) for tag, model, expected in candidates if tag in requested]


def select_named_math_datasets(
    manifest: dict[str, Any], names_to_select: Sequence[str]
) -> list[dict[str, Any]]:
    entries = manifest.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("math data manifest has no datasets list")
    names = [str(entry["name"]) for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError("math data manifest contains duplicate dataset names")
    by_name = dict(zip(names, entries, strict=True))
    missing = [name for name in names_to_select if name not in by_name]
    if missing:
        raise ValueError(f"math data manifest is missing datasets: {missing}")
    selected = [by_name[name] for name in names_to_select]
    for entry in selected:
        path = Path(str(entry["output_path"]))
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != entry.get("output_sha256"):
            raise ValueError(
                f"dataset SHA256 mismatch for {entry['name']}: expected {entry.get('output_sha256')}, "
                f"found {actual_sha256}"
            )
    return selected


def select_math_datasets(manifest: dict[str, Any], *, id_validation: str) -> list[dict[str, Any]]:
    id_names = {
        "full": ("id_validation_full",),
        "clean": ("id_validation_clean",),
        "both": ("id_validation_full", "id_validation_clean"),
    }[id_validation]
    selected_names = (*id_names, *DEFAULT_MATH_DATASETS)
    return select_named_math_datasets(manifest, selected_names)


def build_math_command(
    *,
    model: ModelSpec,
    data_manifest_path: Path,
    data_manifest: dict[str, Any],
    datasets: Sequence[dict[str, Any]],
    output_root: Path,
    python_executable: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    request_batch_size: int,
    questions_per_batch: int = 1,
    kv_cache_memory_gib: float | None = None,
    predictive_topk_width: int | None = None,  # None = the manifest value
) -> list[str]:
    sampling = data_manifest["sampling"]
    command = [
        python_executable,
        str(SCRIPT_DIR / "eval_math_passk.py"),
        "--model",
        model.model,
        "--expected_model_identity_sha256",
        model.expected_identity_sha256,
        "--tag",
        model.tag,
        "--chat_template",
        str(data_manifest["chat_template"]["path"]),
        "--datasets",
        *[str(entry["output_path"]) for entry in datasets],
        "--dataset_manifest",
        str(data_manifest_path),
        "--out",
        str(output_root / model.tag / "math" / "metrics.json"),
        "--trace_dir",
        str(output_root / model.tag / "math" / "traces"),
        "--ks",
        *[str(value) for value in DEFAULT_K_VALUES],
        "--temperature",
        str(sampling["temperature"]),
        "--top_k",
        str(sampling["top_k"]),
        "--top_p",
        str(sampling["top_p"]),
        "--max_tokens",
        str(sampling["max_response_tokens"]),
        "--max_prompt_tokens",
        str(sampling["max_prompt_tokens"]),
        "--max_model_len",
        str(sampling["max_model_len"]),
        "--predictive_topk_width",
        str(sampling["predictive_topk_width"] if predictive_topk_width is None else predictive_topk_width),
        "--tensor_parallel_size",
        str(tensor_parallel_size),
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
        "--request_batch_size",
        str(request_batch_size),
        "--questions_per_batch",
        str(questions_per_batch),
    ]
    if kv_cache_memory_gib is not None:
        command.extend(["--kv_cache_memory_gib", str(kv_cache_memory_gib)])
    return command


def build_ood_command(
    *,
    model: ModelSpec,
    output_root: Path,
    python_executable: str,
    lm_eval_executable: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    gpqa_task: str,
    dry_run: bool,
) -> list[str]:
    command = [
        python_executable,
        str(SCRIPT_DIR / "eval_gemma4_ood.py"),
        "--model",
        model.model,
        "--expected-model-identity-sha256",
        model.expected_identity_sha256,
        "--output-dir",
        str(output_root / model.tag / "ood"),
        "--profile",
        "gemma4-report",
        "--gpqa-task",
        gpqa_task,
        "--lm-eval-executable",
        lm_eval_executable,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        "8192",
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("base_e2b", "distilled_e2b_step750", "rl_e2b_step125"),
        default=["base_e2b", "distilled_e2b_step750", "rl_e2b_step125"],
    )
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--distilled-model", type=Path, default=DEFAULT_DISTILLED_MODEL)
    parser.add_argument("--rl-model", type=Path, default=DEFAULT_RL_MODEL)
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--suite", choices=("math", "ood", "all"), default="all")
    parser.add_argument("--id-validation", choices=("full", "clean", "both"), default="full")
    parser.add_argument(
        "--gpqa-task",
        choices=("gpqa_diamond_cot_n_shot", "gpqa_diamond_n_shot"),
        default="gpqa_diamond_cot_n_shot",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--lm-eval-executable", default="/tmp/.venv-gemma4-e2e/bin/lm_eval")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--request-batch-size", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive")
    if args.request_batch_size <= 0:
        raise ValueError("--request-batch-size must be positive")
    data_manifest_path = args.data_manifest.resolve()
    data_manifest: dict[str, Any] | None = None
    selected_datasets: list[dict[str, Any]] = []
    if args.suite in {"math", "all"}:
        data_manifest = _load_json(data_manifest_path)
        load_dataset_protocol_manifest(data_manifest_path)
        chat_template = data_manifest.get("chat_template", {})
        actual_template_sha256 = _sha256_file(str(chat_template.get("path", "")))
        if actual_template_sha256 != chat_template.get("sha256"):
            raise ValueError(
                f"chat template SHA256 mismatch: expected {chat_template.get('sha256')}, found {actual_template_sha256}"
            )
        selected_datasets = select_math_datasets(data_manifest, id_validation=args.id_validation)
    models = resolve_models(args)
    output_root = args.output_root.resolve()

    commands = []
    for model in models:
        if args.suite in {"math", "all"}:
            assert data_manifest is not None
            commands.append(
                {
                    "model": model.tag,
                    "suite": "math",
                    "command": build_math_command(
                        model=model,
                        data_manifest_path=data_manifest_path,
                        data_manifest=data_manifest,
                        datasets=selected_datasets,
                        output_root=output_root,
                        python_executable=args.python_executable,
                        tensor_parallel_size=args.tensor_parallel_size,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        request_batch_size=args.request_batch_size,
                    ),
                }
            )
        if args.suite in {"ood", "all"}:
            commands.append(
                {
                    "model": model.tag,
                    "suite": "ood",
                    "command": build_ood_command(
                        model=model,
                        output_root=output_root,
                        python_executable=args.python_executable,
                        lm_eval_executable=args.lm_eval_executable,
                        tensor_parallel_size=args.tensor_parallel_size,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        gpqa_task=args.gpqa_task,
                        dry_run=args.preflight,
                    ),
                }
            )

    run_manifest = {
        "schema_version": 1,
        "protocol": "gemma4_three_model_eval_matrix_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "preflight" if args.preflight else "dry_run",
        "models": [asdict(model) for model in models],
        "data_manifest": str(data_manifest_path) if data_manifest is not None else None,
        "selected_datasets": [
            {
                "name": entry["name"],
                "unique_questions": entry["unique_questions"],
                "samples_per_question": entry["samples_per_question"],
                "total_requests": entry["total_requests"],
            }
            for entry in selected_datasets
        ],
        "id_validation_selection": args.id_validation,
        "gpqa_task": args.gpqa_task,
        "commands": commands,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "eval_matrix_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)

    if not args.preflight and not args.execute:
        return 0
    for item in commands:
        if args.preflight and item["suite"] == "math":
            continue
        subprocess.run(item["command"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
