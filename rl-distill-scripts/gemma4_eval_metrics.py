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

"""Pure metric helpers for the sampled Gemma 4 math evaluations.

The expensive evaluator should write one scored trace per sampled response.  This
module deliberately has no model or vLLM dependency: it turns those traces into
question-level and dataset-level metrics and makes every subset/voting convention
part of the returned metadata.

The conservative default is ``subset_strategy="full_only"``.  It reports the
requested unbiased pass@k curve from all available samples, but only reports
mean/majority metrics when ``k`` is the full sample count.  Prefix curves are
available as deterministic diagnostics, and seeded Monte Carlo curves are
available when the caller explicitly chooses that estimator.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_K_VALUES = (1, 2, 4, 8, 16, 32, 64)
SUBSET_STRATEGIES = ("full_only", "prefix", "monte_carlo")
MAJORITY_RULES = ("plurality", "strict_majority")
FP16_TOPK_MASS_TOLERANCE = 2.5e-3


@dataclass(frozen=True)
class TraceSample:
    """One already-scored sampled response.

    ``prediction`` must be the answer-class key used for voting.  ``None`` means
    that the response abstained (for example, it had no valid boxed answer).
    ``sequence_entropy`` is a response-level mean predictive entropy.  The token
    fields are optional and permit a separate token-weighted aggregate.
    """

    question_id: str
    correct: bool
    prediction: str | None
    sample_index: int | None = None
    source_order: int = 0
    sequence_entropy: float | None = None
    token_entropy_sum: float | None = None
    token_entropy_count: int = 0


def _validate_n_c_k(n: int, c: int, k: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if isinstance(c, bool) or not isinstance(c, int | np.integer) or not 0 <= c <= n:
        raise ValueError(f"c must be an integer in [0, n], got c={c!r}, n={n!r}")
    if isinstance(k, bool) or not isinstance(k, int | np.integer) or not 1 <= k <= n:
        raise ValueError(f"k must be an integer in [1, n], got k={k!r}, n={n!r}")


def pass_at_k(n: int, c: int, k: int) -> float:
    """User-requested unbiased pass@k estimator for one question.

    This is exactly the without-replacement estimator specified in the project
    prompt.  It equals ``1 - comb(n-c, k) / comb(n, k)``.
    """

    _validate_n_c_k(n, c, k)
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def pass_at_k_dataset(ns: Sequence[int], cs: Sequence[int], k: int) -> float:
    """Macro-average :func:`pass_at_k` over questions."""

    if len(ns) != len(cs):
        raise ValueError(f"ns and cs must have the same length, got {len(ns)} and {len(cs)}")
    if len(ns) == 0:
        raise ValueError("pass_at_k_dataset requires at least one question")
    return float(np.mean([pass_at_k(n, c, k) for n, c in zip(ns, cs, strict=True)]))


def mean_at_k_prefix(correctness: Sequence[bool], k: int) -> float:
    """Accuracy of the deterministic first-k prefix (diagnostic, not unbiased)."""

    if not 1 <= k <= len(correctness):
        raise ValueError(f"k must be in [1, {len(correctness)}], got {k}")
    return float(np.mean(np.asarray(correctness[:k], dtype=np.float64)))


def _normalized_prediction(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"<none>", "<invalid>", "<abstain>"}:
        return None
    return text


def topk_plus_residual_bucket_entropy(topk_logprobs: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Approximate token entropy using top-k probabilities plus one residual bucket.

    The residual vocabulary is represented as *one* outcome, so this is a lower
    bound on full-vocabulary entropy, not an exact predictive entropy.
    """

    logprobs = np.asarray(topk_logprobs, dtype=np.float64)
    if logprobs.ndim < 1 or logprobs.shape[-1] == 0:
        raise ValueError("top-k log probabilities must have a non-empty final dimension")
    if np.any(np.isnan(logprobs)) or np.any(logprobs > 1e-7):
        raise ValueError("top-k values must be normalized log probabilities (finite/-inf and <= 0)")
    probs = np.exp(logprobs)
    topk_mass = probs.sum(axis=-1)
    if np.any(topk_mass > 1.0 + FP16_TOPK_MASS_TOLERANCE):
        raise ValueError("top-k probability mass exceeds 1 by more than the FP16 tolerance")
    residual = np.clip(1.0 - topk_mass, 0.0, 1.0)
    finite_logprobs = np.where(probs > 0.0, logprobs, 0.0)
    topk_terms = (-probs * finite_logprobs).sum(axis=-1)
    safe_residual = np.where(residual > 0.0, residual, 1.0)
    residual_terms = np.where(residual > 0.0, -residual * np.log(safe_residual), 0.0)
    return topk_terms + residual_terms


def _masked_entropy_observation(
    token_entropies: Sequence[float] | np.ndarray,
    response_mask: Sequence[int | bool] | np.ndarray | None,
) -> tuple[float, float, int]:
    values = np.asarray(token_entropies, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("token entropies must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)) or np.any(values < -1e-7):
        raise ValueError("token entropies must be finite and non-negative")

    if response_mask is not None:
        mask = np.asarray(response_mask)
        if mask.ndim != 1:
            raise ValueError("response_mask must be one-dimensional")
        if not np.all(np.isin(mask, (0, 1, False, True))):
            raise ValueError("response_mask must contain only 0/1 values")
        mask = mask.astype(bool)
        if mask.size == values.size:
            values = values[mask]
        elif int(mask.sum()) != values.size:
            raise ValueError(
                "response_mask must either align with token entropies or have one active entry per response entropy"
            )
        # If mask.sum() == len(values), values are already response-only.

    if values.size == 0:
        raise ValueError("response_mask selects no entropy-bearing response tokens")
    return float(values.mean()), float(values.sum()), int(values.size)


def trace_sample_from_mapping(
    row: Mapping[str, Any],
    *,
    source_order: int,
    question_id_field: str | None = None,
    correctness_field: str | None = None,
    prediction_field: str | None = None,
) -> TraceSample:
    """Convert a JSON-like trace row into the strict internal representation."""

    question_keys = (question_id_field,) if question_id_field else ("uid", "question_id")
    question_id = next((row[key] for key in question_keys if key and key in row), None)
    if question_id is None:
        raise ValueError(f"trace row {source_order} has no question id in {question_keys}")

    if correctness_field:
        if correctness_field not in row:
            raise ValueError(f"trace row {source_order} has no {correctness_field!r} field")
        raw_correct = row[correctness_field]
    elif "acc" in row:
        raw_correct = row["acc"]
    elif "correct" in row:
        raw_correct = row["correct"]
    elif "score" in row:
        raw_correct = float(row["score"]) > 0.5
    else:
        raise ValueError(f"trace row {source_order} has no acc/correct/score field")

    if isinstance(raw_correct, bool | np.bool_):
        correct = bool(raw_correct)
    elif isinstance(raw_correct, int | float | np.number) and float(raw_correct) in {0.0, 1.0}:
        correct = bool(raw_correct)
    else:
        raise ValueError(f"trace row {source_order} correctness must be bool or 0/1, got {raw_correct!r}")

    prediction_keys = (
        (prediction_field,) if prediction_field else ("answer_class", "normalized_prediction", "pred", "prediction")
    )
    prediction = None
    for key in prediction_keys:
        if key and key in row:
            prediction = _normalized_prediction(row[key])
            if prediction is not None:
                break
    if correct and prediction is None:
        raise ValueError(f"trace row {source_order} is correct but has no valid prediction class")

    sample_index = row.get("sample_index")
    if sample_index is not None:
        if isinstance(sample_index, bool) or not isinstance(sample_index, int | np.integer):
            raise ValueError(f"trace row {source_order} sample_index must be an integer")
        sample_index = int(sample_index)

    sequence_entropy: float | None = None
    token_entropy_sum: float | None = None
    token_entropy_count = 0
    token_entropies = row.get("token_entropies", row.get("response_token_entropies"))
    if token_entropies is None:
        topk_logprobs = row.get("predictive_topk_logprobs", row.get("token_topk_logprobs"))
        if topk_logprobs is not None:
            token_entropies = topk_plus_residual_bucket_entropy(topk_logprobs)
    if token_entropies is not None:
        sequence_entropy, token_entropy_sum, token_entropy_count = _masked_entropy_observation(
            token_entropies, row.get("response_mask")
        )
    else:
        summary_sum = row.get("token_entropy_sum")
        summary_count = row.get("token_entropy_count")
        if summary_sum is not None or summary_count is not None:
            if summary_sum is None or summary_count is None:
                raise ValueError(f"trace row {source_order} must provide both token entropy sum and count")
            token_entropy_sum = float(summary_sum)
            if not math.isfinite(token_entropy_sum) or token_entropy_sum < 0.0:
                raise ValueError(f"trace row {source_order} token entropy sum must be finite and non-negative")
            if isinstance(summary_count, bool) or not isinstance(summary_count, int | np.integer):
                raise ValueError(f"trace row {source_order} token entropy count must be an integer")
            token_entropy_count = int(summary_count)
            if token_entropy_count < 0 or (token_entropy_count == 0) != (token_entropy_sum == 0.0):
                raise ValueError(f"trace row {source_order} token entropy summary is inconsistent")
        scalar_entropy = row.get("response_entropy", row.get("sequence_entropy"))
        if scalar_entropy is not None:
            sequence_entropy = float(scalar_entropy)
            if not math.isfinite(sequence_entropy) or sequence_entropy < 0.0:
                raise ValueError(f"trace row {source_order} response entropy must be finite and non-negative")
        elif token_entropy_count:
            sequence_entropy = token_entropy_sum / token_entropy_count
        if token_entropy_count and not math.isclose(
            sequence_entropy * token_entropy_count,
            token_entropy_sum,
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            raise ValueError(f"trace row {source_order} sequence and token entropy summaries disagree")

    return TraceSample(
        question_id=str(question_id),
        correct=correct,
        prediction=prediction,
        sample_index=sample_index,
        source_order=source_order,
        sequence_entropy=sequence_entropy,
        token_entropy_sum=token_entropy_sum,
        token_entropy_count=token_entropy_count,
    )


def _ordered_groups(samples: Sequence[TraceSample]) -> dict[str, list[TraceSample]]:
    grouped: dict[str, list[TraceSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.question_id].append(sample)
    if not grouped:
        raise ValueError("at least one trace sample is required")

    ordered: dict[str, list[TraceSample]] = {}
    for question_id in sorted(grouped):
        question_samples = grouped[question_id]
        has_indices = [sample.sample_index is not None for sample in question_samples]
        if any(has_indices) and not all(has_indices):
            raise ValueError(f"question {question_id!r} mixes explicit and implicit sample_index values")
        if all(has_indices):
            indices = [sample.sample_index for sample in question_samples]
            if any(index is not None and index < 0 for index in indices):
                raise ValueError(f"question {question_id!r} has a negative sample_index")
            if len(set(indices)) != len(indices):
                raise ValueError(f"question {question_id!r} has duplicate sample_index values")
            question_samples = sorted(question_samples, key=lambda sample: (sample.sample_index, sample.source_order))
        else:
            question_samples = sorted(question_samples, key=lambda sample: sample.source_order)

        class_correctness: dict[str, set[bool]] = defaultdict(set)
        for sample in question_samples:
            if sample.prediction is not None:
                class_correctness[sample.prediction].add(sample.correct)
        inconsistent = [key for key, values in class_correctness.items() if len(values) != 1]
        if inconsistent:
            raise ValueError(
                f"question {question_id!r} has prediction classes with inconsistent correctness: {inconsistent[:3]}"
            )
        ordered[question_id] = question_samples
    return ordered


def _answer_entropy(predictions: Sequence[str | None]) -> float:
    valid = [prediction for prediction in predictions if prediction is not None]
    if not valid:
        return 0.0
    counts = np.asarray(list(Counter(valid).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def _subset_metrics(
    samples: Sequence[TraceSample], indices: Sequence[int] | np.ndarray, *, majority_rule: str
) -> dict[str, float]:
    subset = [samples[int(index)] for index in indices]
    k = len(subset)
    predictions = [sample.prediction for sample in subset]
    valid_predictions = [prediction for prediction in predictions if prediction is not None]
    counts = Counter(valid_predictions)

    tied = False
    all_abstain = not counts
    no_strict_majority = False
    majority_correct = 0.0
    winner_fraction = 0.0
    if counts:
        max_count = max(counts.values())
        winners = [prediction for prediction, count in counts.items() if count == max_count]
        tied = len(winners) != 1
        no_strict_majority = max_count * 2 <= k
        if not tied and (majority_rule == "plurality" or not no_strict_majority):
            winner = winners[0]
            winner_fraction = max_count / k
            winner_correctness = {sample.correct for sample in subset if sample.prediction == winner}
            if len(winner_correctness) != 1:
                raise ValueError(f"prediction class {winner!r} has inconsistent correctness within a subset")
            majority_correct = float(winner_correctness.pop())

    entropy_values = [sample.sequence_entropy for sample in subset if sample.sequence_entropy is not None]
    return {
        "mean_correct": float(np.mean([sample.correct for sample in subset])),
        "majority_correct": majority_correct,
        "majority_tie": float(tied),
        "all_abstain": float(all_abstain),
        "no_strict_majority": float(no_strict_majority),
        "winner_vote_fraction_all_samples": winner_fraction,
        "abstention_rate": sum(prediction is None for prediction in predictions) / k,
        "valid_answer_entropy_nats": _answer_entropy(predictions),
        "predictive_entropy_sequence_mean_nats": float(np.mean(entropy_values)) if entropy_values else math.nan,
        "predictive_entropy_sequence_coverage": len(entropy_values) / k,
    }


def majority_at_k_prefix(
    correctness: Sequence[bool],
    predictions: Sequence[str | None],
    k: int,
    *,
    majority_rule: str = "plurality",
) -> float:
    """Majority accuracy on a deterministic prefix, with conservative ties.

    Invalid predictions abstain.  A tie or an all-abstain prefix scores zero.
    This helper intentionally never resolves a tie by sample order.
    """

    if len(correctness) != len(predictions):
        raise ValueError("correctness and predictions must have the same length")
    if majority_rule not in MAJORITY_RULES:
        raise ValueError(f"majority_rule must be one of {MAJORITY_RULES}, got {majority_rule!r}")
    if not 1 <= k <= len(correctness):
        raise ValueError(f"k must be in [1, {len(correctness)}], got {k}")
    samples = [
        TraceSample(question_id="question", correct=bool(correct), prediction=_normalized_prediction(prediction))
        for correct, prediction in zip(correctness, predictions, strict=True)
    ]
    return _subset_metrics(samples, np.arange(k), majority_rule=majority_rule)["majority_correct"]


def _question_seed(base_seed: int, question_id: str, k: int) -> int:
    digest = hashlib.sha256(f"{base_seed}\0{question_id}\0{k}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _subset_indices(
    *,
    n: int,
    k: int,
    strategy: str,
    question_id: str,
    seed: int,
    monte_carlo_resamples: int,
) -> Iterable[np.ndarray]:
    if strategy == "prefix":
        yield np.arange(k)
        return
    if strategy == "monte_carlo":
        if monte_carlo_resamples <= 0:
            raise ValueError("monte_carlo_resamples must be positive")
        if k == n:
            yield np.arange(n)
            return
        rng = np.random.default_rng(_question_seed(seed, question_id, k))
        random_keys = rng.random((monte_carlo_resamples, n))
        yield from np.argpartition(random_keys, kth=k - 1, axis=1)[:, :k]
        return
    raise ValueError(f"cannot draw subsets for strategy {strategy!r}")


def _average_subset_metrics(draws: Sequence[dict[str, float]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in draws[0]:
        values = np.asarray([draw[key] for draw in draws], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[key] = float(finite.mean()) if finite.size else None
    return result


def _predictive_entropy_summary(
    grouped: Mapping[str, Sequence[TraceSample]], *, source_kind: str
) -> dict[str, float | int | str | None]:
    sequence_values = [
        sample.sequence_entropy
        for samples in grouped.values()
        for sample in samples
        if sample.sequence_entropy is not None
    ]
    question_values = []
    for samples in grouped.values():
        values = [sample.sequence_entropy for sample in samples if sample.sequence_entropy is not None]
        if values:
            question_values.append(float(np.mean(values)))

    token_sum = sum(
        sample.token_entropy_sum or 0.0
        for samples in grouped.values()
        for sample in samples
        if sample.token_entropy_count > 0
    )
    token_count = sum(sample.token_entropy_count for samples in grouped.values() for sample in samples)
    total_samples = sum(len(samples) for samples in grouped.values())
    return {
        "source_kind": source_kind,
        "sequence_weighted_mean_nats": float(np.mean(sequence_values)) if sequence_values else None,
        "question_weighted_mean_nats": float(np.mean(question_values)) if question_values else None,
        "token_weighted_mean_nats": token_sum / token_count if token_count else None,
        "n_sequences_with_entropy": len(sequence_values),
        "n_response_tokens_with_entropy": token_count,
        "sequence_coverage": len(sequence_values) / total_samples,
        "question_coverage": len(question_values) / len(grouped),
    }


def aggregate_math_traces(
    traces: Sequence[TraceSample] | Sequence[Mapping[str, Any]],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    expected_samples_per_question: int | None = 64,
    subset_strategy: str = "full_only",
    monte_carlo_resamples: int = 4096,
    seed: int = 0,
    majority_rule: str = "plurality",
    predictive_entropy_kind: str = "unknown",
    include_per_question: bool = False,
    question_id_field: str | None = None,
    correctness_field: str | None = None,
    prediction_field: str | None = None,
) -> dict[str, Any]:
    """Aggregate scored response traces into sampled-math metrics.

    ``pass_at_k`` always uses all ``n`` observed samples for each question.  The
    interpretation of mean/majority metrics for ``k < n`` is controlled by
    ``subset_strategy`` and recorded in the returned policy metadata.
    """

    if subset_strategy not in SUBSET_STRATEGIES:
        raise ValueError(f"subset_strategy must be one of {SUBSET_STRATEGIES}, got {subset_strategy!r}")
    if majority_rule not in MAJORITY_RULES:
        raise ValueError(f"majority_rule must be one of {MAJORITY_RULES}, got {majority_rule!r}")
    if expected_samples_per_question is not None and expected_samples_per_question <= 0:
        raise ValueError("expected_samples_per_question must be positive or None")

    ks = sorted(set(k_values))
    if not ks or any(isinstance(k, bool) or not isinstance(k, int | np.integer) or k <= 0 for k in ks):
        raise ValueError("k_values must contain positive integers")
    ks = [int(k) for k in ks]

    parsed: list[TraceSample] = []
    identity_fields: dict[str, dict[str, str]] = defaultdict(dict)
    answer_class_methods: set[str] = set()
    for source_order, trace in enumerate(traces):
        if isinstance(trace, TraceSample):
            parsed.append(trace)
        else:
            if trace.get("answer_class_method"):
                answer_class_methods.add(str(trace["answer_class_method"]))
            question_keys = (question_id_field,) if question_id_field else ("uid", "question_id")
            question_id = next((trace[key] for key in question_keys if key and key in trace), None)
            if question_id is not None:
                question_id = str(question_id)
                for field_name in ("prompt_text", "gold"):
                    if field_name not in trace or trace[field_name] is None:
                        continue
                    canonical_value = repr(trace[field_name])
                    previous = identity_fields[question_id].get(field_name)
                    if previous is not None and previous != canonical_value:
                        raise ValueError(f"question {question_id!r} has conflicting {field_name} values across traces")
                    identity_fields[question_id][field_name] = canonical_value
            parsed.append(
                trace_sample_from_mapping(
                    trace,
                    source_order=source_order,
                    question_id_field=question_id_field,
                    correctness_field=correctness_field,
                    prediction_field=prediction_field,
                )
            )
    grouped = _ordered_groups(parsed)

    sample_counts = {question_id: len(samples) for question_id, samples in grouped.items()}
    if expected_samples_per_question is not None:
        mismatches = {
            question_id: count for question_id, count in sample_counts.items() if count != expected_samples_per_question
        }
        if mismatches:
            preview = dict(list(mismatches.items())[:5])
            raise ValueError(
                f"expected exactly {expected_samples_per_question} samples per question; mismatches: {preview}"
            )
        expected_indices = set(range(expected_samples_per_question))
        invalid_indices = {}
        for question_id, samples in grouped.items():
            indices = [sample.sample_index for sample in samples]
            if any(index is None for index in indices) or set(indices) != expected_indices:
                invalid_indices[question_id] = indices[:8]
        if invalid_indices:
            preview = dict(list(invalid_indices.items())[:5])
            raise ValueError(
                f"expected sample_index values exactly 0..{expected_samples_per_question - 1}; mismatches: {preview}"
            )
    min_samples = min(sample_counts.values())
    if ks[-1] > min_samples:
        raise ValueError(f"requested k={ks[-1]}, but the smallest question has only {min_samples} samples")

    per_question: dict[str, dict[str, Any]] = {}
    for question_id, samples in grouped.items():
        per_question[question_id] = {
            "n": len(samples),
            "c": sum(sample.correct for sample in samples),
            "by_k": {},
        }

    by_k: dict[str, dict[str, Any]] = {}
    for k in ks:
        ns = [len(samples) for samples in grouped.values()]
        cs = [sum(sample.correct for sample in samples) for samples in grouped.values()]
        question_pass = [pass_at_k(n, c, k) for n, c in zip(ns, cs, strict=True)]
        for (question_id, _samples), question_value in zip(grouped.items(), question_pass, strict=True):
            per_question[question_id]["by_k"][str(k)] = {"pass_at_k": question_value}
        result: dict[str, Any] = {
            "pass_at_k": float(np.mean(question_pass)),
            "mean_at_k": None,
            "maj_at_k": None,
            "valid_answer_entropy_nats": None,
            "abstention_rate": None,
            "majority_tie_rate": None,
            "all_abstain_rate": None,
            "no_strict_majority_rate": None,
            "winner_vote_fraction_all_samples": None,
            "predictive_entropy_sequence_mean_nats": None,
            "predictive_entropy_sequence_coverage": None,
            "subset_draws_per_question": 0,
        }

        full_for_every_question = all(len(samples) == k for samples in grouped.values())
        can_compute_subset_metrics = subset_strategy != "full_only" or full_for_every_question
        question_aggregates: list[dict[str, float | None]] = []
        if can_compute_subset_metrics:
            strategy = "prefix" if subset_strategy == "full_only" else subset_strategy
            for question_id, samples in grouped.items():
                draws = [
                    _subset_metrics(samples, indices, majority_rule=majority_rule)
                    for indices in _subset_indices(
                        n=len(samples),
                        k=k,
                        strategy=strategy,
                        question_id=question_id,
                        seed=seed,
                        monte_carlo_resamples=monte_carlo_resamples,
                    )
                ]
                averaged = _average_subset_metrics(draws)
                question_aggregates.append(averaged)
                per_question[question_id]["by_k"][str(k)].update(averaged)

            key_map = {
                "mean_at_k": "mean_correct",
                "maj_at_k": "majority_correct",
                "valid_answer_entropy_nats": "valid_answer_entropy_nats",
                "abstention_rate": "abstention_rate",
                "majority_tie_rate": "majority_tie",
                "all_abstain_rate": "all_abstain",
                "no_strict_majority_rate": "no_strict_majority",
                "winner_vote_fraction_all_samples": "winner_vote_fraction_all_samples",
                "predictive_entropy_sequence_mean_nats": "predictive_entropy_sequence_mean_nats",
                "predictive_entropy_sequence_coverage": "predictive_entropy_sequence_coverage",
            }
            for result_key, question_key in key_map.items():
                values = [aggregate[question_key] for aggregate in question_aggregates]
                finite = [value for value in values if value is not None and math.isfinite(value)]
                result[result_key] = float(np.mean(finite)) if finite else None
            result["subset_draws_per_question"] = (
                monte_carlo_resamples if subset_strategy == "monte_carlo" and not full_for_every_question else 1
            )
        else:
            result["subset_metrics_status"] = (
                "not_reported: mean@k/maj@k for k<n require an explicit prefix or monte_carlo subset strategy"
            )

        by_k[str(k)] = result

    policies = {
        "pass_at_k": "unbiased without-replacement estimator computed from all n observed samples",
        "mean_maj_subset_strategy": subset_strategy,
        "prefix_order": "ascending sample_index, or trace-file order when every sample_index is absent",
        "prefix_warning": "prefix metrics are deterministic diagnostics, not expected random-subset estimates",
        "monte_carlo_seed": seed if subset_strategy == "monte_carlo" else None,
        "monte_carlo_resamples": monte_carlo_resamples if subset_strategy == "monte_carlo" else None,
        "majority_rule": majority_rule,
        "majority_ties": "wrong; no order-based tie breaker",
        "invalid_predictions": (
            "abstain from answer selection, remain in the k-sample denominator; all-abstain is wrong"
        ),
        "answer_classes": (
            f"exact values from {prediction_field!r}"
            if prediction_field
            else "first available precomputed answer-class field"
        ),
        "answer_class_methods": sorted(answer_class_methods),
        "valid_answer_entropy": "Shannon entropy in nats, conditional on non-abstaining answer classes",
        "predictive_entropy_kind": predictive_entropy_kind,
    }
    ambiguities = [
        {
            "id": "answer_equivalence",
            "resolved": len(answer_class_methods) == 1,
            "selected_field": prediction_field,
            "selected_methods": sorted(answer_class_methods),
            "note": (
                "The field name alone cannot prove semantic math equivalence. Record how answer classes were built."
            ),
        },
        {
            "id": "mean_maj_subset_semantics",
            "resolved": subset_strategy != "full_only",
            "note": (
                "For k<n choose a preregistered estimator. full_only intentionally suppresses mean@k/maj@k; "
                "prefix is diagnostic; monte_carlo estimates expected random-subset performance."
            ),
        },
        {
            "id": "plurality_vs_strict_majority",
            "resolved": False,
            "selected_for_run": majority_rule,
            "note": (
                "The project has not yet preregistered whether a unique plurality must exceed half of all k samples."
            ),
        },
        {
            "id": "predictive_entropy_definition",
            "resolved": predictive_entropy_kind != "unknown",
            "note": "Declare exact full-vocabulary entropy versus a top-k-plus-residual-bucket approximation.",
        },
    ]

    output: dict[str, Any] = {
        "n_questions": len(grouped),
        "n_samples": len(parsed),
        "samples_per_question": sorted(set(sample_counts.values())),
        "k_values": ks,
        "policies": policies,
        "scientific_ambiguities": ambiguities,
        "by_k": by_k,
        "predictive_entropy": _predictive_entropy_summary(grouped, source_kind=predictive_entropy_kind),
    }
    if include_per_question:
        output["per_question"] = per_question
    return output
