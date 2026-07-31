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

"""Create pinned lm-eval tasks for the registered 14,042-item MMMLU protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

MMMLU_REPO = "openai/MMMLU"
MMMLU_REVISION = "325a01dc3e173cac1578df94120499aaca2e2504"
HARNESS_REVISION = "f4d4b3de3ee6741a7151a9fe74945ee515262f4c"
TASK_GROUP = "gemma4_mmmlu14k"
EXPECTED_ROWS = 14_042
LOCALES = (
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
DEFAULT_OUTPUT_DIR = Path("/lambda/nfs/Jason-scale/rl-distill-evals/gemma4-three-model/data/mmmlu14k_tasks")
DEFAULT_HARNESS_DIR = Path(__file__).resolve().parents[2] / "lm-evaluation-harness"


class _HarnessYamlLoader(yaml.SafeLoader):
    pass


_HarnessYamlLoader.add_constructor("!function", lambda loader, node: loader.construct_scalar(node))


def _normalize_subject(name: Any) -> str:
    value = str(name)
    for marker in ("_test.csv", "_test-"):
        index = value.find(marker)
        if index != -1:
            return value[:index]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]


def _validate_source_rows(locale_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[list[str], dict[str, Any]]:
    reference = locale_rows[LOCALES[0]]
    if len(reference) != EXPECTED_ROWS:
        raise ValueError(f"MMMLU expected {EXPECTED_ROWS} rows per locale, found {len(reference)}")
    subjects = sorted({_normalize_subject(row["Subject"]) for row in reference})
    if len(subjects) != 57:
        raise ValueError(f"MMMLU expected 57 subjects, found {len(subjects)}")

    reference_identity = [(_normalize_subject(row["Subject"]), int(row["Unnamed: 0"])) for row in reference]
    answer_disagreements = {}
    for locale, rows in locale_rows.items():
        if len(rows) != EXPECTED_ROWS:
            raise ValueError(f"{locale} expected {EXPECTED_ROWS} rows, found {len(rows)}")
        identity = [(_normalize_subject(row["Subject"]), int(row["Unnamed: 0"])) for row in rows]
        if identity != reference_identity:
            raise ValueError(f"{locale} subject/source-row ordering is not aligned with {LOCALES[0]}")
        answer_disagreements[locale] = sum(
            str(row["Answer"]) != str(reference_row["Answer"])
            for row, reference_row in zip(rows, reference, strict=True)
        )
    locale_counts = Counter(index % len(LOCALES) for index in range(EXPECTED_ROWS))
    if set(locale_counts.values()) != {EXPECTED_ROWS // len(LOCALES)}:
        raise ValueError("global modulo allocation did not produce equal locale counts")
    subject_locale_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for index, row in enumerate(reference):
        subject_locale_counts[_normalize_subject(row["Subject"])][index % len(LOCALES)] += 1
    for subject, counts in subject_locale_counts.items():
        padded = [counts[index] for index in range(len(LOCALES))]
        if max(padded) - min(padded) > 1:
            raise ValueError(f"subject {subject} is not balanced across locales")
    return subjects, {
        "locale_counts": {LOCALES[index]: locale_counts[index] for index in range(len(LOCALES))},
        "subject_locale_counts": {
            subject: {LOCALES[index]: counts[index] for index in range(len(LOCALES))}
            for subject, counts in sorted(subject_locale_counts.items())
        },
        "answer_key_disagreements_vs_first_locale": answer_disagreements,
    }


def _utils_source(subjects: Sequence[str]) -> str:
    return f'''"""Generated filters for the pinned Gemma 4 reduced-MMMLU protocol."""

from functools import partial

LOCALES = {LOCALES!r}
SUBJECTS = {tuple(subjects)!r}
_SUBJECT_INDEX_CACHE = {{}}


def _normalize_subject(name):
    if not isinstance(name, str):
        return name
    for marker in ("_test.csv", "_test-"):
        index = name.find(marker)
        if index != -1:
            return name[:index]
    return name


def _subject_indices(dataset):
    fingerprint = getattr(dataset, "_fingerprint", None) or f"object:{{id(dataset)}}"
    key = (fingerprint, len(dataset))
    cached = _SUBJECT_INDEX_CACHE.get(key)
    if cached is None:
        cached = {{subject: [] for subject in SUBJECTS}}
        for index, raw_subject in enumerate(dataset["Subject"]):
            subject = _normalize_subject(raw_subject)
            if subject not in cached:
                raise ValueError(f"unexpected MMMLU subject: {{subject!r}}")
            cached[subject].append(index)
        _SUBJECT_INDEX_CACHE[key] = cached
    return cached


def _filter_subject(dataset, subject):
    return dataset.select(_subject_indices(dataset)[subject])


def _filter_reduced(dataset, subject, locale_index):
    indices = [index for index in _subject_indices(dataset)[subject] if index % len(LOCALES) == locale_index]
    return dataset.select(indices)


for _subject in SUBJECTS:
    globals()[f"fewshot_{{_subject}}"] = partial(_filter_subject, subject=_subject)
    for _locale_index, _locale in enumerate(LOCALES):
        globals()[f"reduced_{{_locale.lower()}}_{{_subject}}"] = partial(
            _filter_reduced,
            subject=_subject,
            locale_index=_locale_index,
        )
'''


def _write_template(path: Path) -> None:
    path.write_text(
        f"""dataset_path: {MMMLU_REPO}
dataset_kwargs:
  revision: {MMMLU_REVISION}
test_split: test
fewshot_split: test
output_type: multiple_choice
doc_to_text: "{{{{Question.strip()}}}}\\nA. {{{{A}}}}\\nB. {{{{B}}}}\\nC. {{{{C}}}}\\nD. {{{{D}}}}\\nAnswer:"
doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: Answer
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
""",
        encoding="utf-8",
    )


def _write_task(
    path: Path,
    *,
    locale: str,
    subject: str,
    native: Mapping[str, Any],
) -> str:
    task_name = f"{TASK_GROUP}_{locale.lower()}_{subject}"
    description = json.dumps(str(native["description"]), ensure_ascii=False)
    alias = json.dumps(str(native["task_alias"]), ensure_ascii=False)
    path.write_text(
        f"""include: _reduced_template_yaml
dataset_name: {locale}
process_docs: !function reduced_mmmlu_utils.reduced_{locale.lower()}_{subject}
fewshot_config:
  sampler: first_n
  process_docs: !function reduced_mmmlu_utils.fewshot_{subject}
task: {task_name}
task_alias: {alias}
description: {description}
""",
        encoding="utf-8",
    )
    return task_name


def prepare(
    *,
    output_dir: Path,
    harness_dir: Path,
    overwrite: bool,
    load_dataset_fn: Any,
) -> dict[str, Any]:
    native_dir = harness_dir / "lm_eval" / "tasks" / "openai-mmmlu" / "default"
    subjects_path = native_dir.parent / "subjects.json"
    if not subjects_path.is_file():
        raise FileNotFoundError(f"missing pinned harness MMMLU subjects: {subjects_path}")
    harness_revision = subprocess_revision(harness_dir)
    if harness_revision != HARNESS_REVISION:
        raise ValueError(f"harness revision mismatch: expected {HARNESS_REVISION}, found {harness_revision}")

    locale_rows = {
        locale: list(load_dataset_fn(MMMLU_REPO, locale, split="test", revision=MMMLU_REVISION)) for locale in LOCALES
    }
    subjects, allocation = _validate_source_rows(locale_rows)
    registered_subjects = json.loads(subjects_path.read_text(encoding="utf-8"))
    if subjects != sorted(registered_subjects):
        raise ValueError("dataset subjects do not match the pinned harness subject registry")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"refusing to replace {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "reduced_mmmlu_utils.py").write_text(_utils_source(subjects), encoding="utf-8")
    _write_template(output_dir / "_reduced_template_yaml")

    locale_groups = []
    for locale in LOCALES:
        locale_lower = locale.lower()
        native_group = yaml.safe_load((native_dir / f"_mmmlu_{locale_lower}.yaml").read_text(encoding="utf-8"))
        task_names = []
        for subject in subjects:
            native_task_path = native_dir / f"mmmlu_{locale_lower}_{subject}.yaml"
            native_task = yaml.load(native_task_path.read_text(encoding="utf-8"), Loader=_HarnessYamlLoader)
            task_names.append(
                _write_task(
                    output_dir / f"{TASK_GROUP}_{locale_lower}_{subject}.yaml",
                    locale=locale,
                    subject=subject,
                    native=native_task,
                )
            )
        group_name = f"{TASK_GROUP}_{locale_lower}"
        locale_groups.append(group_name)
        group_payload = {
            "group": group_name,
            "group_alias": native_group["group_alias"],
            "task": task_names,
            "aggregate_metric_list": [
                {"metric": "acc", "weight_by_size": True},
                {"metric": "acc_norm", "weight_by_size": True},
            ],
            "metadata": {"version": 1},
        }
        (output_dir / f"_{group_name}.yaml").write_text(
            yaml.safe_dump(group_payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    overall_payload = {
        "group": TASK_GROUP,
        "group_alias": "Gemma 4 MMMLU 14k",
        "task": locale_groups,
        "aggregate_metric_list": [
            {"metric": "acc", "weight_by_size": True},
            {"metric": "acc_norm", "weight_by_size": True},
        ],
        "metadata": {"version": 1},
    }
    (output_dir / f"_{TASK_GROUP}.yaml").write_text(
        yaml.safe_dump(overall_payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    manifest = {
        "schema_version": 1,
        "protocol": "gemma4_mmmlu14k_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_group": TASK_GROUP,
        "task_dir": str(output_dir.resolve()),
        "source": {"repo_id": MMMLU_REPO, "revision": MMMLU_REVISION, "rows_per_locale": EXPECTED_ROWS},
        "harness_revision": HARNESS_REVISION,
        "assignment": {
            "rule": "aligned global test-row index modulo 14",
            "total_evaluation_rows": EXPECTED_ROWS,
            "locales": list(LOCALES),
            "subjects": subjects,
            **allocation,
        },
        "files": _tree_manifest(output_dir),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def subprocess_revision(repo: Path) -> str:
    import subprocess

    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError(f"pinned harness checkout is dirty: {dirty.splitlines()[:5]}")
    return revision


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--harness-dir", type=Path, default=DEFAULT_HARNESS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from datasets import load_dataset

    manifest = prepare(
        output_dir=args.output_dir.resolve(),
        harness_dir=args.harness_dir.resolve(),
        overwrite=args.overwrite,
        load_dataset_fn=load_dataset,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
