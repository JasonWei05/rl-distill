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

"""Aggregate scored Gemma 4 math JSONL traces without loading a model.

Each row needs a question id (``uid`` or ``question_id``), correctness
(``acc``, ``correct``, or ``score``), and a parsed answer class (preferably
``answer_class``; legacy ``pred`` is also accepted).  Optional predictive
entropy fields are documented by ``gemma4_eval_metrics.trace_sample_from_mapping``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from gemma4_eval_metrics import DEFAULT_K_VALUES, MAJORITY_RULES, SUBSET_STRATEGIES, aggregate_math_traces


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True, help="scored JSONL trace files")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument(
        "--expected-samples-per-question",
        type=int,
        default=64,
        help="fail unless every question has exactly this many samples",
    )
    parser.add_argument(
        "--allow-variable-samples",
        action="store_true",
        help="disable the exact per-question sample-count check",
    )
    parser.add_argument(
        "--subset-strategy",
        choices=SUBSET_STRATEGIES,
        default="full_only",
        help=(
            "mean/maj policy for k below the sample count; full_only is the conservative default, "
            "prefix is diagnostic, and monte_carlo is a seeded expected-subset estimate"
        ),
    )
    parser.add_argument("--monte-carlo-resamples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--majority-rule", choices=MAJORITY_RULES, default="plurality")
    parser.add_argument(
        "--prediction-field",
        default=None,
        help="field containing a precomputed answer-equivalence class (recommended: answer_class)",
    )
    parser.add_argument("--question-id-field", default=None)
    parser.add_argument("--correctness-field", default=None)
    parser.add_argument(
        "--predictive-entropy-kind",
        default="unknown",
        help="metadata label, e.g. exact_full_vocab or topk_plus_residual_bucket_lower_bound",
    )
    parser.add_argument("--include-per-question", action="store_true")
    parser.add_argument(
        "--allow-lexical-majority",
        action="store_true",
        help="explicit diagnostic escape hatch for traces lacking semantic answer-class metadata",
    )
    return parser.parse_args(argv)


def _load_jsonl(paths: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path_str in paths:
        path = Path(path_str)
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"expected an object at {path}:{line_number}")
                dataset = str(row.get("dataset") or path.stem)
                by_dataset[dataset].append(row)
    if not by_dataset:
        raise ValueError("no non-empty trace rows were found")
    return dict(by_dataset)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    resolved_config = {
        "trace_files": [str(Path(path).resolve()) for path in args.traces],
        "k_values": sorted(set(args.ks)),
        "expected_samples_per_question": None if args.allow_variable_samples else args.expected_samples_per_question,
        "subset_strategy": args.subset_strategy,
        "monte_carlo_resamples": args.monte_carlo_resamples,
        "seed": args.seed,
        "majority_rule": args.majority_rule,
        "prediction_field": args.prediction_field,
        "question_id_field": args.question_id_field,
        "correctness_field": args.correctness_field,
        "predictive_entropy_kind": args.predictive_entropy_kind,
        "include_per_question": args.include_per_question,
        "allow_lexical_majority": args.allow_lexical_majority,
    }
    print(json.dumps({"resolved_config": resolved_config}, indent=2), flush=True)

    results = {}
    for dataset, traces in _load_jsonl(args.traces).items():
        prediction_keys = (
            (args.prediction_field,)
            if args.prediction_field
            else ("answer_class", "normalized_prediction", "pred", "prediction")
        )
        missing_semantic_method = []
        for row_index, trace in enumerate(traces):
            prediction = next(
                (
                    trace[key]
                    for key in prediction_keys
                    if key and key in trace and trace[key] is not None and str(trace[key]).strip()
                ),
                None,
            )
            if prediction is not None and str(prediction).strip() and not trace.get("answer_class_method"):
                missing_semantic_method.append(row_index)
        if missing_semantic_method and not args.allow_lexical_majority:
            raise ValueError(
                f"dataset {dataset!r} has valid predictions without semantic answer_class_method metadata "
                f"(first rows: {missing_semantic_method[:8]}). Refuse to label lexical voting maj@k; "
                "regenerate with eval_math_passk.py or explicitly pass --allow-lexical-majority for diagnostics."
            )
        results[dataset] = aggregate_math_traces(
            traces,
            k_values=args.ks,
            expected_samples_per_question=resolved_config["expected_samples_per_question"],
            subset_strategy=args.subset_strategy,
            monte_carlo_resamples=args.monte_carlo_resamples,
            seed=args.seed,
            majority_rule=args.majority_rule,
            predictive_entropy_kind=args.predictive_entropy_kind,
            include_per_question=args.include_per_question,
            question_id_field=args.question_id_field,
            correctness_field=args.correctness_field,
            prediction_field=args.prediction_field,
        )
        print(
            f"[{dataset}] questions={results[dataset]['n_questions']} samples={results[dataset]['n_samples']}",
            flush=True,
        )

    output = {"config": resolved_config, "results": results}
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
