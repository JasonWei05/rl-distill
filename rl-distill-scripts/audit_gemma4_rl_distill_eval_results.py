#!/usr/bin/env python3
"""Audit completed packed Gemma 4 evaluations and emit a report-ready summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from gemma4_eval_registry import canonical_json_sha256, load_source_registry

COMPLETE_PROTOCOL = "gemma4_rl_distill_base_evals_v2"
PACKED_COMPLETE_PROTOCOL = "gemma4_rl_distill_packed_complete_v1"
EXPECTED_MATH = {
    "id_easy": (300, 16),
    "id_medium": (300, 16),
    "id_hard": (300, 16),
    "math500": (500, 16),
    "gsm8k": (1_319, 8),
}
EXPECTED_OOD = {
    "mmlu_pro": ("mmlu_pro", 12_032, 12_032),
    "gpqa": ("gpqa_diamond_cot_n_shot", 198, 396),
    "mmmlu14k": ("gemma4_mmmlu14k", 14_042, 14_042),
}
OOD_SCORE_KEYS = {
    "mmlu_pro": ("exact_match,custom-extract", "exact_match,none", "acc,none", "acc_norm,none"),
    "gpqa": ("exact_match,flexible-extract", "exact_match,none", "acc,none", "acc_norm,none"),
    "mmmlu14k": ("acc,none", "acc_norm,none", "exact_match,none"),
}
RESULT_ARCHITECTURE_GROUPS = (
    ("E2B models", "gemma-4-E2B"),
    ("E4B models", "gemma-4-E4B"),
    ("12B models", "gemma-4-12B"),
)


class AuditError(ValueError):
    """Raised when a production evaluation artifact violates its contract."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _close(observed: Any, expected: float, label: str, *, tolerance: float = 1e-10) -> None:
    _require(isinstance(observed, int | float) and not isinstance(observed, bool), f"{label} is not numeric")
    _require(math.isfinite(float(observed)), f"{label} is not finite")
    _require(
        math.isclose(float(observed), expected, rel_tol=tolerance, abs_tol=tolerance),
        f"{label} mismatch: expected {expected}, found {observed}",
    )


def _pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def audit_file_manifest(model_root: Path, marker: Mapping[str, Any], tag: str) -> dict[str, Any]:
    files = marker.get("files")
    _require(isinstance(files, list) and files, f"{tag}: completion marker has no file manifest")
    registered: dict[str, Mapping[str, Any]] = {}
    total_bytes = 0
    for index, raw in enumerate(files):
        _require(isinstance(raw, Mapping), f"{tag}: file manifest row {index} is not an object")
        relative = raw.get("path")
        _require(isinstance(relative, str) and relative, f"{tag}: file manifest row {index} has no path")
        relative_path = Path(relative)
        _require(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            f"{tag}: unsafe file manifest path {relative!r}",
        )
        _require(relative not in registered, f"{tag}: duplicate file manifest path {relative}")
        path = model_root / relative_path
        _require(path.is_file(), f"{tag}: registered result file is missing: {relative}")
        size = path.stat().st_size
        _require(raw.get("size") == size, f"{tag}: size mismatch for {relative}")
        _require(raw.get("sha256") == _sha256(path), f"{tag}: SHA256 mismatch for {relative}")
        registered[relative] = raw
        total_bytes += size

    actual = {
        str(path.relative_to(model_root))
        for path in model_root.rglob("*")
        if path.is_file() and path.name != "RUN_COMPLETE.json" and not path.name.endswith(".partial")
    }
    _require(
        actual == set(registered),
        f"{tag}: completion manifest paths differ from local files; "
        f"missing={sorted(set(registered) - actual)[:8]} extra={sorted(actual - set(registered))[:8]}",
    )
    return {"registered_files": len(registered), "registered_bytes": total_bytes}


def audit_math_trace(
    path: Path,
    *,
    tag: str,
    dataset: str,
    expected_questions: int,
    samples_per_question: int,
) -> dict[str, Any]:
    indices: dict[str, set[int]] = defaultdict(set)
    correct_counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    class_correctness: dict[tuple[str, str], bool] = {}
    prompt_gold: dict[str, tuple[str, str]] = {}
    rows = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            _require(line.strip() != "", f"{tag}/{dataset}: blank trace row {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AuditError(f"{tag}/{dataset}: invalid JSON at trace row {line_number}") from error
            _require(isinstance(row, dict), f"{tag}/{dataset}: trace row {line_number} is not an object")
            _require(row.get("dataset") == dataset, f"{tag}/{dataset}: wrong dataset at trace row {line_number}")
            uid = row.get("uid")
            sample_index = row.get("sample_index")
            _require(isinstance(uid, str) and uid, f"{tag}/{dataset}: invalid UID at trace row {line_number}")
            _require(
                isinstance(sample_index, int) and not isinstance(sample_index, bool),
                f"{tag}/{dataset}: invalid sample index at trace row {line_number}",
            )
            _require(
                0 <= sample_index < samples_per_question,
                f"{tag}/{dataset}: out-of-range sample index for UID {uid}",
            )
            _require(
                sample_index not in indices[uid], f"{tag}/{dataset}: duplicate UID/sample pair {uid}/{sample_index}"
            )
            indices[uid].add(sample_index)

            raw_correct = row.get("acc")
            _require(
                isinstance(raw_correct, bool)
                or (
                    isinstance(raw_correct, int | float) and not isinstance(raw_correct, bool) and raw_correct in {0, 1}
                ),
                f"{tag}/{dataset}: invalid correctness at trace row {line_number}",
            )
            correct = bool(raw_correct)
            if correct:
                correct_counts[uid] += 1

            answer_class = row.get("answer_class")
            _require(
                answer_class is None or (isinstance(answer_class, str) and answer_class),
                f"{tag}/{dataset}: invalid answer class at trace row {line_number}",
            )
            _require(not correct or answer_class is not None, f"{tag}/{dataset}: correct trace has no answer class")
            if answer_class is not None:
                key = (uid, answer_class)
                previous = class_correctness.setdefault(key, correct)
                _require(previous == correct, f"{tag}/{dataset}: answer class crosses correctness for UID {uid}")
                class_counts[uid][answer_class] += 1

            identity = (repr(row.get("prompt_text")), repr(row.get("gold")))
            previous_identity = prompt_gold.setdefault(uid, identity)
            _require(previous_identity == identity, f"{tag}/{dataset}: conflicting prompt/gold for UID {uid}")
            rows += 1

    expected_indices = set(range(samples_per_question))
    _require(
        len(indices) == expected_questions, f"{tag}/{dataset}: expected {expected_questions} UIDs, found {len(indices)}"
    )
    malformed = [uid for uid, found in indices.items() if found != expected_indices]
    _require(not malformed, f"{tag}/{dataset}: malformed sample-index coverage for UIDs {malformed[:8]}")
    expected_rows = expected_questions * samples_per_question
    _require(rows == expected_rows, f"{tag}/{dataset}: expected {expected_rows} rows, found {rows}")

    majority_correct = 0
    for uid in indices:
        votes = class_counts[uid]
        if not votes:
            continue
        maximum = max(votes.values())
        winners = [answer_class for answer_class, count in votes.items() if count == maximum]
        if len(winners) == 1 and class_correctness[(uid, winners[0])]:
            majority_correct += 1

    correct_by_question = [correct_counts[uid] for uid in indices]
    return {
        "trace_path": str(path),
        "trace_sha256": _sha256(path),
        "n_questions": expected_questions,
        "n_samples": rows,
        "samples_per_question": samples_per_question,
        "correct_samples": sum(correct_by_question),
        "mean_full": sum(correct_by_question) / rows,
        "pass_full": sum(count > 0 for count in correct_by_question) / expected_questions,
        "maj_full": majority_correct / expected_questions,
        "pass_by_k": {
            str(k): sum(_pass_at_k(samples_per_question, count, k) for count in correct_by_question)
            / expected_questions
            for k in (1, 2, 4, 8, 16)
            if k <= samples_per_question
        },
    }


def audit_math(model_root: Path, model: Any) -> tuple[dict[str, Any], str]:
    tag = model.tag
    metrics_path = model_root / "math/metrics.json"
    metrics = _load_json(metrics_path)
    _require(metrics.get("tag") == tag, f"{tag}: math metrics tag mismatch")
    config = metrics.get("config")
    _require(isinstance(config, Mapping), f"{tag}: math metrics config is missing")
    identity = config.get("model_identity_sha256")
    _require(isinstance(identity, str) and len(identity) == 64, f"{tag}: invalid math model identity")
    results = metrics.get("results")
    _require(isinstance(results, Mapping), f"{tag}: math results are missing")
    _require(set(results) == set(model.math_datasets), f"{tag}: math dataset roster mismatch")

    audited: dict[str, Any] = {}
    for dataset in model.math_datasets:
        expected_questions, samples = EXPECTED_MATH[dataset]
        trace_path = model_root / "math/traces" / f"{tag}__{dataset}.jsonl"
        _require(trace_path.is_file(), f"{tag}/{dataset}: trace file is missing")
        trace = audit_math_trace(
            trace_path,
            tag=tag,
            dataset=dataset,
            expected_questions=expected_questions,
            samples_per_question=samples,
        )
        result = results[dataset]
        _require(isinstance(result, Mapping), f"{tag}/{dataset}: metric result is not an object")
        _require(result.get("n_questions") == expected_questions, f"{tag}/{dataset}: n_questions mismatch")
        _require(result.get("n_samples") == expected_questions * samples, f"{tag}/{dataset}: n_samples mismatch")
        _require(result.get("samples_per_question") == [samples], f"{tag}/{dataset}: sample count mismatch")
        expected_ks = [k for k in (1, 2, 4, 8, 16) if k <= samples]
        _require(result.get("k_values") == expected_ks, f"{tag}/{dataset}: k-value roster mismatch")
        by_k = result.get("by_k")
        _require(
            isinstance(by_k, Mapping) and set(by_k) == {str(k) for k in expected_ks},
            f"{tag}/{dataset}: by-k roster mismatch",
        )
        for k in expected_ks:
            entry = by_k[str(k)]
            _require(isinstance(entry, Mapping), f"{tag}/{dataset}: k={k} metrics are missing")
            _close(entry.get("pass_at_k"), trace["pass_by_k"][str(k)], f"{tag}/{dataset} pass@{k}")
            for metric in ("mean_at_k", "maj_at_k"):
                value = entry.get(metric)
                _require(
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and 0 <= value <= 1,
                    f"{tag}/{dataset}: invalid {metric} at k={k}",
                )
        full = by_k[str(samples)]
        _close(full.get("mean_at_k"), trace["mean_full"], f"{tag}/{dataset} full mean")
        _close(full.get("pass_at_k"), trace["pass_full"], f"{tag}/{dataset} full pass")
        _close(full.get("maj_at_k"), trace["maj_full"], f"{tag}/{dataset} full maj")
        _close(
            result.get("mean@k"), round(100 * trace["mean_full"], 2), f"{tag}/{dataset} displayed mean", tolerance=1e-9
        )
        _close(
            result.get("pass@k"), round(100 * trace["pass_full"], 2), f"{tag}/{dataset} displayed pass", tolerance=1e-9
        )
        _close(result.get("maj@k"), round(100 * trace["maj_full"], 2), f"{tag}/{dataset} displayed maj", tolerance=1e-9)
        audited[dataset] = {
            "k": samples,
            "n_questions": expected_questions,
            "n_samples": expected_questions * samples,
            "mean_at_k": trace["mean_full"],
            "pass_at_k": trace["pass_full"],
            "maj_at_k": trace["maj_full"],
            "by_k": {
                str(k): {
                    "mean_at_k": by_k[str(k)]["mean_at_k"],
                    "pass_at_k": by_k[str(k)]["pass_at_k"],
                    "maj_at_k": by_k[str(k)]["maj_at_k"],
                }
                for k in expected_ks
            },
            "trace_sha256": trace["trace_sha256"],
        }
    return audited, identity


def _count_jsonl(path: Path) -> int:
    rows = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            _require(line.strip() != "", f"blank sample row in {path} at line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise AuditError(f"invalid sample JSON in {path} at line {line_number}") from error
            _require(isinstance(value, dict), f"sample row in {path} at line {line_number} is not an object")
            rows += 1
    return rows


def audit_ood(model_root: Path, model: Any, model_identity: str) -> dict[str, Any]:
    audited: dict[str, Any] = {}
    for benchmark, (result_key, expected_effective, expected_logged) in EXPECTED_OOD.items():
        output_dir = model_root / "ood" / benchmark
        completion = _load_json(output_dir / "complete.json")
        task_id = f"{model.tag}__{benchmark}"
        _require(
            completion.get("protocol") == "gemma4_three_model_ood_task_v1", f"{task_id}: wrong completion protocol"
        )
        _require(completion.get("task_id") == task_id, f"{task_id}: completion task mismatch")
        _require(completion.get("benchmark") == benchmark, f"{task_id}: completion benchmark mismatch")
        _require(completion.get("model_identity_sha256") == model_identity, f"{task_id}: model identity mismatch")
        artifacts = completion.get("artifacts")
        _require(isinstance(artifacts, Mapping), f"{task_id}: completion artifacts are missing")

        result_relative = artifacts.get("result_path")
        _require(isinstance(result_relative, str) and result_relative, f"{task_id}: result path is missing")
        result_path = output_dir / result_relative
        _require(result_path.is_file(), f"{task_id}: result JSON is missing")
        _require(artifacts.get("result_sha256") == _sha256(result_path), f"{task_id}: result SHA256 mismatch")
        result = _load_json(result_path)
        result_metrics = result.get("results")
        _require(
            isinstance(result_metrics, Mapping) and result_key in result_metrics, f"{task_id}: result key mismatch"
        )
        task_metrics = result_metrics[result_key]
        _require(isinstance(task_metrics, Mapping), f"{task_id}: task metrics are missing")

        n_samples = result.get("n-samples")
        _require(isinstance(n_samples, Mapping) and n_samples, f"{task_id}: n-samples are missing")
        try:
            effective = sum(int(value["effective"]) for value in n_samples.values())
        except (KeyError, TypeError, ValueError) as error:
            raise AuditError(f"{task_id}: malformed n-samples") from error
        _require(
            effective == expected_effective,
            f"{task_id}: expected {expected_effective} effective samples, found {effective}",
        )

        timestamp = result_path.stem.removeprefix("results_")
        sample_files = sorted(output_dir.rglob(f"samples_*_{timestamp}.jsonl"))
        _require(sample_files, f"{task_id}: no sample logs match result timestamp {timestamp}")
        sample_manifest = []
        logged = 0
        for path in sample_files:
            rows = _count_jsonl(path)
            logged += rows
            sample_manifest.append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "rows": rows,
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        sample_manifest_sha256 = hashlib.sha256(
            json.dumps(sample_manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        _require(logged == expected_logged, f"{task_id}: expected {expected_logged} logged rows, found {logged}")
        expected_artifacts = {
            "result_path": result_relative,
            "result_sha256": _sha256(result_path),
            "result_key": result_key,
            "effective_samples": effective,
            "sample_file_count": len(sample_files),
            "logged_sample_rows": logged,
            "sample_manifest_sha256": sample_manifest_sha256,
        }
        _require(dict(artifacts) == expected_artifacts, f"{task_id}: completion artifact receipt mismatch")

        wrapper = _load_json(output_dir / "ood_eval_manifest.json")
        wrapper_identity = wrapper.get("model_identity")
        _require(isinstance(wrapper_identity, Mapping), f"{task_id}: wrapper model identity is missing")
        _require(
            wrapper_identity.get("model_identity_sha256") == model_identity, f"{task_id}: wrapper identity mismatch"
        )

        score_key = next((key for key in OOD_SCORE_KEYS[benchmark] if key in task_metrics), None)
        _require(score_key is not None, f"{task_id}: no registered accuracy metric found")
        score = task_metrics[score_key]
        _require(
            isinstance(score, int | float)
            and not isinstance(score, bool)
            and math.isfinite(float(score))
            and 0 <= score <= 1,
            f"{task_id}: invalid score {score_key}={score!r}",
        )
        audited[benchmark] = {
            "result_key": result_key,
            "score_key": score_key,
            "score": float(score),
            "effective_samples": effective,
            "logged_sample_rows": logged,
            "result_sha256": expected_artifacts["result_sha256"],
            "sample_manifest_sha256": sample_manifest_sha256,
        }
    return audited


def _metric_cell(model: Mapping[str, Any], dataset: str) -> str:
    result = model["math"].get(dataset)
    if result is None:
        return "—"
    k = result["k"]
    return f"{100 * result['mean_at_k']:.2f} / {100 * result['maj_at_k']:.2f} / {100 * result['pass_at_k']:.2f} (@{k})"


def _result_row(model: Mapping[str, Any]) -> str:
    return (
        "| "
        + " | ".join(
            [
                f"`{model['tag']}`",
                _metric_cell(model, "id_easy"),
                _metric_cell(model, "id_medium"),
                _metric_cell(model, "math500"),
                _metric_cell(model, "gsm8k"),
                f"{100 * model['ood']['mmlu_pro']['score']:.2f}",
                f"{100 * model['ood']['gpqa']['score']:.2f}",
                f"{100 * model['ood']['mmmlu14k']['score']:.2f}",
            ]
        )
        + " |"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "<!-- Generated by audit_gemma4_rl_distill_eval_results.py; do not edit metric cells manually. -->",
        f"Audit completed: `{report['audited_at_utc']}`.",
        "",
        "Math cells are `mean@k / maj@k / pass@k` in percent. OOD cells are harness accuracy in percent.",
        "Distilled checkpoints are grouped by student architecture; base and direct-RL checkpoints are grouped by their own architecture.",
    ]
    grouped_tags = []
    for title, architecture in RESULT_ARCHITECTURE_GROUPS:
        models = [model for model in report["models"] if model["architecture"] == architecture]
        if not models:
            continue
        grouped_tags.extend(model["tag"] for model in models)
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Model | Easy ID | Medium ID | MATH500 | GSM8K | MMLU-Pro | GPQA-Diamond | MMMLU-14K |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                *(_result_row(model) for model in models),
            ]
        )
    expected_tags = [model["tag"] for model in report["models"]]
    _require(
        sorted(grouped_tags) == sorted(expected_tags) and len(grouped_tags) == len(expected_tags),
        "result table architecture groups do not cover the model roster exactly",
    )
    return "\n".join(lines) + "\n"


def audit_results(results_root: Path, source_registry: Path) -> dict[str, Any]:
    source_payload, models = load_source_registry(source_registry)
    _require(len(models) == 15, f"production audit requires 15 models, found {len(models)}")
    packed = _load_json(results_root / "_packed/PACKED_RUN_COMPLETE.json")
    _require(packed.get("protocol") == PACKED_COMPLETE_PROTOCOL, "packed completion protocol mismatch")
    expected_tags = [model.tag for model in models]
    _require(packed.get("model_tags") == expected_tags, "packed completion model roster mismatch")
    _require(packed.get("permanently_failed") == [], "packed completion records permanent failures")
    _require(packed.get("missing_remote_completions") == [], "packed completion records missing models")
    _require(
        packed.get("source_registry_sha256") == canonical_json_sha256(source_payload),
        "packed completion source-registry hash mismatch",
    )

    audited_models = []
    total_registered_files = 0
    total_registered_bytes = 0
    total_math_samples = 0
    total_ood_effective = 0
    for model in models:
        model_root = results_root / model.tag
        marker = _load_json(model_root / "RUN_COMPLETE.json")
        _require(marker.get("protocol") == COMPLETE_PROTOCOL, f"{model.tag}: completion protocol mismatch")
        _require(marker.get("model_tag") == model.tag, f"{model.tag}: completion tag mismatch")
        manifest = audit_file_manifest(model_root, marker, model.tag)
        artifact_root = model_root / model.tag
        math_results, identity = audit_math(artifact_root, model)
        ood_results = audit_ood(artifact_root, model, identity)
        total_registered_files += manifest["registered_files"]
        total_registered_bytes += manifest["registered_bytes"]
        total_math_samples += sum(result["n_samples"] for result in math_results.values())
        total_ood_effective += sum(result["effective_samples"] for result in ood_results.values())
        audited_models.append(
            {
                "tag": model.tag,
                "display_name": model.display_name,
                "category": model.category,
                "architecture": model.architecture,
                "trained_on": model.trained_on,
                "model_identity_sha256": identity,
                "completion_sha256": _sha256(model_root / "RUN_COMPLETE.json"),
                "file_manifest": manifest,
                "math": math_results,
                "ood": ood_results,
            }
        )

    _require(total_math_samples == 422_280, f"matrix math sample total mismatch: {total_math_samples}")
    _require(total_ood_effective == 394_080, f"matrix OOD item total mismatch: {total_ood_effective}")
    return {
        "schema_version": 1,
        "protocol": "gemma4_rl_distill_eval_final_audit_v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_registry": str(source_registry),
        "source_registry_sha256": canonical_json_sha256(source_payload),
        "packed_completion_sha256": _sha256(results_root / "_packed/PACKED_RUN_COMPLETE.json"),
        "status": "pass",
        "counts": {
            "models": len(audited_models),
            "math_samples": total_math_samples,
            "ood_effective_items": total_ood_effective,
            "registered_files": total_registered_files,
            "registered_bytes": total_registered_bytes,
        },
        "models": audited_models,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--source-registry",
        type=Path,
        default=Path(__file__).resolve().parent / "config/gemma4_rl_distill_eval_sources.json",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_results(args.results_root.resolve(), args.source_registry.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": "pass", "counts": report["counts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
