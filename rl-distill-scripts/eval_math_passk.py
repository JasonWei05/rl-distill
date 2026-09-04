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

"""Generate and score deterministic 64-sample Gemma math evaluations.

The input parquet is first collapsed to unique questions. Explicit UIDs are
preferred; otherwise a stable SHA256 of question text and gold answer is used.
Every question receives one request per sample with a deterministic seed. The
script stores response-only top-k predictive-entropy diagnostics and delegates
all aggregate metrics to :mod:`gemma4_eval_metrics`.

Model libraries are imported only by :func:`main`, so the request/score/metric
pipeline can be tested with CPU-only mocks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
for import_path in (SCRIPT_DIR, DATA_DIR, REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from gemma4_eval_metrics import (  # noqa: E402
    DEFAULT_K_VALUES,
    FP16_TOPK_MASS_TOLERANCE,
    SUBSET_STRATEGIES,
    aggregate_math_traces,
)
from gemma4_model_identity import require_sha256, resolve_model_identity  # noqa: E402

REQUIRED_SAMPLES_PER_QUESTION = 64
DEFAULT_PREDICTIVE_TOPK_WIDTH = 128
PREDICTIVE_ENTROPY_KIND = "topk_plus_residual_bucket_lower_bound"
STOP_STRINGS = ("<end_of_turn>", "<start_of_turn>")
IMMUTABLE_REVISION_PATTERN = re.compile(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}")
EXPECTED_SAMPLING_PROTOCOL = {
    "temperature": 1.0,
    "top_k": -1,
    "top_p": 1.0,
    "max_response_tokens": 8192,
    "max_prompt_tokens": 4096,
    "max_model_len": 12288,
    "predictive_topk_width": 128,
}


@dataclass(frozen=True)
class EvalQuestion:
    question_id: str
    question_text: str
    gold_answer: str


@dataclass(frozen=True)
class EvalRequest:
    question: EvalQuestion
    sample_index: int
    sampling_seed: int
    prompt_token_ids: list[int]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _ensure_python_bin_on_path() -> None:
    """Expose venv console tools (notably ninja) to vLLM worker subprocesses."""

    python_bin = str((Path(sys.prefix) / "bin").resolve())
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if python_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join([python_bin, *path_entries])


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_protocol_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate the pinned per-dataset sampling plan."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read dataset protocol manifest {manifest_path}: {error}") from error
    supported_protocols = {
        "gemma4_three_model_math_eval_v1",
        "gemma4_rl_distill_math_eval_v1",
        "gemma4_rl_distill_math_eval_v2",  # 300-question easy/medium/hard band validation sets
    }
    if manifest.get("schema_version") != 1 or manifest.get("protocol") not in supported_protocols:
        raise ValueError(f"unsupported dataset protocol manifest: {manifest_path}")
    repetition_rule = manifest.get("repetition_rule")
    if not isinstance(repetition_rule, dict):
        raise ValueError("dataset protocol manifest has no repetition_rule")
    if repetition_rule.get("allowed_factors") != "powers_of_two":
        raise ValueError("dataset protocol repetition factors must be powers_of_two")
    fixed_counts = repetition_rule.get("samples_per_question")
    fixed_policy = repetition_rule.get("policy") == "fixed_by_dataset"
    if fixed_policy:
        if not isinstance(fixed_counts, dict):
            raise ValueError("fixed_by_dataset repetition policy requires samples_per_question")
        threshold = None
    else:
        threshold = repetition_rule.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
            raise ValueError("dataset protocol repetition threshold must be a non-negative integer")
        if repetition_rule.get("comparison") != "strictly_greater_than":
            raise ValueError("dataset protocol repetition comparison must be strictly_greater_than")
    if manifest.get("sampling") != EXPECTED_SAMPLING_PROTOCOL:
        raise ValueError("dataset protocol manifest does not use the registered Gemma 4 sampling settings")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("dataset protocol manifest must contain a non-empty datasets list")
    by_path: dict[str, dict[str, Any]] = {}
    names: set[str] = set()
    for entry in datasets:
        if not isinstance(entry, dict):
            raise ValueError("every dataset protocol entry must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"invalid or duplicate dataset name in protocol manifest: {name!r}")
        names.add(name)
        raw_output_path = entry.get("output_path")
        if not isinstance(raw_output_path, str) or not raw_output_path:
            raise ValueError(f"invalid output_path for dataset {name!r}")
        output_path = Path(raw_output_path).expanduser().resolve()
        sample_count = entry.get("samples_per_question")
        unique_questions = entry.get("unique_questions")
        expected_requests = entry.get("total_requests")
        expected_sha256 = entry.get("output_sha256")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
            raise ValueError(f"invalid samples_per_question for {output_path}")
        if sample_count & (sample_count - 1):
            raise ValueError(f"samples_per_question must be a power of two for {output_path}")
        if isinstance(unique_questions, bool) or not isinstance(unique_questions, int) or unique_questions <= 0:
            raise ValueError(f"invalid unique_questions for {output_path}")
        if fixed_policy:
            expected_sample_count = fixed_counts.get(name)
            if not isinstance(expected_sample_count, int) or expected_sample_count <= 0:
                raise ValueError(f"fixed sample count is missing or invalid for {name!r}")
        else:
            expected_sample_count = 1
            while unique_questions * expected_sample_count <= threshold:
                expected_sample_count *= 2
        if sample_count != expected_sample_count:
            raise ValueError(
                f"samples_per_question is not the smallest registered power of two for {output_path}: "
                f"expected {expected_sample_count}, found {sample_count}"
            )
        if expected_requests != unique_questions * sample_count:
            raise ValueError(f"total_requests is inconsistent for {output_path}")
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError(f"invalid output_sha256 for {output_path}")
        key = str(output_path)
        if key in by_path:
            raise ValueError(f"duplicate dataset path in protocol manifest: {output_path}")
        by_path[key] = dict(entry)
    return by_path


def qtext(prompt_col: Any) -> str:
    if hasattr(prompt_col, "tolist"):
        prompt_col = prompt_col.tolist()
    if hasattr(prompt_col, "__len__") and len(prompt_col) and isinstance(prompt_col[-1], Mapping):
        return str(prompt_col[-1]["content"])
    return str(prompt_col)


def _gold_answer(reward_model: Any) -> str:
    if hasattr(reward_model, "as_py"):
        reward_model = reward_model.as_py()
    if not isinstance(reward_model, Mapping) or reward_model.get("ground_truth") is None:
        raise ValueError("reward_model.ground_truth is required")
    return str(reward_model["ground_truth"])


def _present_uid(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def stable_question_id(question_text: str, uid: Any = None) -> str:
    """Return a stable grouping key, independent of parquet row position."""

    explicit_uid = _present_uid(uid)
    if explicit_uid is not None:
        return explicit_uid
    digest = hashlib.sha256(_canonical_json_bytes({"question_text": question_text})).hexdigest()
    return f"sha256:{digest}"


def prepare_eval_questions(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str,
    max_questions: int = -1,
    excluded_question_sha256s: frozenset[str] = frozenset(),
) -> list[EvalQuestion]:
    """Collapse repeated validation rows and return a stable question order."""

    del dataset_name  # The content hash intentionally stays stable across dataset aliases.
    by_id: dict[str, EvalQuestion] = {}
    for row_index, row in enumerate(rows):
        if "prompt" not in row or "reward_model" not in row:
            raise ValueError(f"row {row_index} is missing prompt or reward_model")
        question_text = qtext(row["prompt"])
        if hashlib.sha256(question_text.encode()).hexdigest() in excluded_question_sha256s:
            continue
        gold_answer = _gold_answer(row["reward_model"])
        question_id = stable_question_id(question_text, row.get("uid"))
        candidate = EvalQuestion(question_id, question_text, gold_answer)
        previous = by_id.get(question_id)
        if previous is not None and previous != candidate:
            raise ValueError(f"question UID {question_id!r} maps to conflicting question/gold rows")
        by_id[question_id] = candidate

    questions = [by_id[question_id] for question_id in sorted(by_id)]
    if max_questions > 0:
        questions = questions[:max_questions]
    if not questions:
        raise ValueError("dataset produced no unique evaluation questions")
    return questions


def load_overlap_hashes_from_dataset_index(path: str | Path) -> frozenset[str]:
    """Load the registered train/validation question-text overlap hashes."""

    index_path = Path(path)
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read dataset index {index_path}: {error}") from error
    values = index.get("cross_split_question_text_overlap_sha256s")
    count = index.get("cross_split_question_text_overlap_count")
    if not isinstance(values, list) or isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("dataset index must contain overlap hash list/count fields")
    if count != len(values):
        raise ValueError("dataset index overlap count does not match its hash list")
    normalized = frozenset(str(value).lower() for value in values)
    if len(normalized) != len(values) or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in normalized):
        raise ValueError("dataset index overlap hashes must be unique 64-character SHA256 values")
    return normalized


def derive_sampling_seed(global_seed: int, dataset_name: str, question_id: str, sample_index: int) -> int:
    payload = {
        "global_seed": global_seed,
        "dataset": dataset_name,
        "question_id": question_id,
        "sample_index": sample_index,
    }
    seed = int.from_bytes(hashlib.sha256(_canonical_json_bytes(payload)).digest()[:8], "big") % (2**31 - 1)
    return seed or 1


def _semantic_equivalent(left: str, right: str, grader: Any) -> tuple[bool, str | None]:
    if left == right:
        return True, None

    def score(prediction: str, gold: str) -> float | None:
        boxed_prediction = f"\\boxed{{{prediction}}}"
        try:
            value = float(grader.compute_score(boxed_prediction, gold, timeout_score=-1.0))
        except TypeError:
            value = float(grader.compute_score(boxed_prediction, gold))
        return None if value < 0 else value

    left_score = score(left, right)
    right_score = score(right, left)
    if left_score is None or right_score is None:
        return False, "timeout"
    left_to_right = left_score > 0.5
    right_to_left = right_score > 0.5
    if left_to_right != right_to_left:
        return False, "asymmetric"
    return left_to_right, None


def assign_semantic_answer_classes(traces: Sequence[dict[str, Any]], grader: Any) -> None:
    """Assign stable per-question equivalence classes using strict math_verify.

    Invalid/no-box predictions remain abstentions. Every non-empty prediction is
    compared against existing class representatives in sample-index order using
    the same ``compute_score`` semantics as correctness grading. Verification
    Timeout, asymmetric comparisons, or equivalence claims that conflict with
    the independently computed correctness label fail closed by treating the
    candidate as non-equivalent and recording the verifier condition on the
    trace. This prevents an uncertain comparison from merging two answer
    classes while allowing the registered evaluation to continue.
    """

    by_question: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        by_question.setdefault(str(trace["uid"]), []).append(trace)

    for question_id in sorted(by_question):
        representatives: list[tuple[str, bool]] = []
        ordered = sorted(by_question[question_id], key=lambda trace: int(trace["sample_index"]))
        for trace in ordered:
            prediction = trace.get("pred")
            if prediction is None or not str(prediction).strip():
                trace["answer_class"] = None
                trace["answer_class_representative"] = None
                continue
            prediction = str(prediction).strip()
            representative = None
            fail_closed_conditions = set()
            correct = bool(trace["acc"])
            for existing, existing_correct in representatives:
                if correct != existing_correct:
                    fail_closed_conditions.add("correctness_mismatch")
                    continue
                equivalent, condition = _semantic_equivalent(prediction, existing, grader)
                if condition is not None:
                    fail_closed_conditions.add(condition)
                if equivalent:
                    representative = existing
                    break
            if representative is None:
                representative = prediction
                representatives.append((representative, correct))
            class_digest = hashlib.sha256(
                _canonical_json_bytes({"representative": representative, "correct": correct})
            ).hexdigest()[:16]
            trace["answer_class"] = f"math_verify:{class_digest}"
            trace["answer_class_representative"] = representative
            trace["answer_class_method"] = (
                "correctness-stratified bidirectional pairwise strict math_verify.compute_score; "
                "timeout/asymmetry/correctness mismatch treated as non-equivalent"
            )
            trace["answer_class_fail_closed_conditions"] = sorted(fail_closed_conditions)


def _render_prompt(tokenizer: Any, question_text: str, chat_template: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": question_text}],
        chat_template=chat_template,
        add_generation_prompt=True,
        tokenize=False,
    )
    try:
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    except AttributeError:
        token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    return [int(token_id) for token_id in token_ids]


def _logprob_field(entry: Any, field_name: str) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(field_name)
    return getattr(entry, field_name, None)


def _extract_predictive_statistics(
    completion: Any,
    *,
    topk_width: int,
    store_topk_logprobs: bool,
    store_per_token_diagnostics: bool,
) -> dict[str, Any]:
    response_token_ids = [int(token_id) for token_id in completion.token_ids]
    position_logprobs = completion.logprobs
    if position_logprobs is None or len(position_logprobs) != len(response_token_ids):
        raise ValueError("vLLM must return one logprob mapping per generated response token")

    topk_ids: list[list[int]] = []
    topk_logprobs: list[list[float]] = []
    sampled_logprobs: list[float] = []
    topk_masses: list[float] = []
    token_entropies: list[float] = []
    for position, (raw_mapping, sampled_token_id) in enumerate(zip(position_logprobs, response_token_ids, strict=True)):
        if raw_mapping is None or sampled_token_id not in raw_mapping:
            raise ValueError(f"position {position} is missing sampled-token log probability")
        sampled_logprob = float(_logprob_field(raw_mapping[sampled_token_id], "logprob"))
        if not math.isfinite(sampled_logprob):
            raise ValueError(f"position {position} sampled-token log probability is not finite")

        by_rank: dict[int, tuple[int, float]] = {}
        for raw_token_id, entry in raw_mapping.items():
            rank = _logprob_field(entry, "rank")
            if rank is None:
                continue
            rank = int(rank)
            if not 1 <= rank <= topk_width:
                continue
            if rank in by_rank:
                raise ValueError(f"position {position} has duplicate predictive rank {rank}")
            logprob = float(_logprob_field(entry, "logprob"))
            if not math.isfinite(logprob) or logprob > 1e-7:
                raise ValueError(f"position {position} rank {rank} is not a normalized log probability")
            by_rank[rank] = (int(raw_token_id), logprob)
        if set(by_rank) != set(range(1, topk_width + 1)):
            missing = sorted(set(range(1, topk_width + 1)).difference(by_rank))
            raise ValueError(f"position {position} is missing exact predictive ranks 1..{topk_width}: {missing[:8]}")
        ranked = [by_rank[rank] for rank in range(1, topk_width + 1)]
        ids = [token_id for token_id, _ in ranked]
        logprobs = [logprob for _, logprob in ranked]
        if len(set(ids)) != topk_width:
            raise ValueError(f"position {position} has duplicate token IDs in its predictive top-k")
        if any(later > earlier + 1e-6 for earlier, later in zip(logprobs, logprobs[1:], strict=False)):
            raise ValueError(f"position {position} predictive ranks are not logprob-sorted")
        mass = math.fsum(math.exp(logprob) for logprob in logprobs)
        if mass > 1.0 + FP16_TOPK_MASS_TOLERANCE:
            raise ValueError(f"position {position} top-k mass exceeds one: {mass}")
        if sampled_token_id in ids:
            ranked_logprob = logprobs[ids.index(sampled_token_id)]
            if abs(ranked_logprob - sampled_logprob) > 1e-6:
                raise ValueError(f"position {position} sampled-token and ranked log probabilities disagree")
        if store_topk_logprobs:
            topk_ids.append(ids)
            topk_logprobs.append(logprobs)
        sampled_logprobs.append(sampled_logprob)
        topk_masses.append(mass)
        residual = max(0.0, 1.0 - mass)
        entropy = -math.fsum(math.exp(logprob) * logprob for logprob in logprobs)
        if residual > 0.0:
            entropy -= residual * math.log(residual)
        token_entropies.append(entropy)

    token_count = len(token_entropies)
    entropy_sum = math.fsum(token_entropies)
    topk_mass_sum = math.fsum(topk_masses)
    sampled_logprob_sum = math.fsum(sampled_logprobs)
    statistics: dict[str, Any] = {
        "predictive_entropy_kind": PREDICTIVE_ENTROPY_KIND,
        "predictive_topk_width": topk_width,
        "sequence_entropy": entropy_sum / token_count if token_count else None,
        "token_entropy_sum": entropy_sum,
        "token_entropy_count": token_count,
        "predictive_topk_mass_sum": topk_mass_sum,
        "predictive_topk_mass_count": token_count,
        "predictive_topk_mass_mean": topk_mass_sum / token_count if token_count else None,
        "sampled_token_logprob_sum": sampled_logprob_sum,
        "sampled_token_logprob_count": token_count,
        "sampled_token_logprob_mean": sampled_logprob_sum / token_count if token_count else None,
    }
    if topk_masses:
        statistics["predictive_topk_mass_min"] = min(topk_masses)
        statistics["predictive_topk_mass_max"] = max(topk_masses)
    if store_per_token_diagnostics:
        statistics["sampled_token_logprobs"] = sampled_logprobs
        statistics["predictive_topk_mass"] = topk_masses
        statistics["token_entropies"] = token_entropies
    if store_topk_logprobs:
        statistics["predictive_topk_token_ids"] = topk_ids
        statistics["predictive_topk_logprobs"] = topk_logprobs
    return statistics


def _make_sampling_params(
    sampling_params_class: Any,
    *,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    predictive_topk_width: int,
) -> Any:
    return sampling_params_class(
        n=1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop=list(STOP_STRINGS),
        logprobs=predictive_topk_width,
        seed=seed,
    )


def evaluate_questions(
    *,
    llm: Any,
    tokenizer: Any,
    sampling_params_class: Any,
    grader: Any,
    questions: Sequence[EvalQuestion],
    dataset_name: str,
    chat_template: str,
    samples_per_question: int = REQUIRED_SAMPLES_PER_QUESTION,
    global_seed: int = 0,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    subset_strategy: str = "full_only",
    monte_carlo_resamples: int = 4096,
    metric_seed: int = 0,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    max_tokens: int = 8192,
    max_prompt_tokens: int = 4096,
    predictive_topk_width: int = DEFAULT_PREDICTIVE_TOPK_WIDTH,
    store_topk_logprobs: bool = False,
    store_per_token_diagnostics: bool = False,
    request_batch_size: int = 8,
    questions_per_batch: int = 1,
    trace_callback: Callable[[Mapping[str, Any]], None] | None = None,
    retain_traces: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run generation through scoring while bounding retained response data.

    Generation is batched across ``questions_per_batch`` questions (every request carries its
    own deterministic seed, so batching does not change what is sampled), with at most
    ``request_batch_size`` requests per vLLM call. Scoring and semantic answer classes are
    still assigned one question at a time. Production callers can
    stream each finalized full trace through ``trace_callback`` and set
    ``retain_traces=False``; only compact metric rows then remain in memory.
    """

    if samples_per_question <= 0:
        raise ValueError("samples_per_question must be positive")
    if predictive_topk_width <= 0:
        raise ValueError("predictive_topk_width must be positive")
    if request_batch_size <= 0:
        raise ValueError("request_batch_size must be positive")
    if questions_per_batch <= 0:
        raise ValueError("questions_per_batch must be positive")
    if not questions:
        raise ValueError("at least one evaluation question is required")
    requested_ks = sorted(set(int(k) for k in k_values) | {samples_per_question})
    if requested_ks[0] <= 0 or requested_ks[-1] > samples_per_question:
        raise ValueError("all k values must lie in [1, samples_per_question]")

    retained_traces: list[dict[str, Any]] = []
    metric_traces: list[dict[str, Any]] = []
    for group_start in range(0, len(questions), questions_per_batch):
        question_group = questions[group_start : group_start + questions_per_batch]
        group_requests: list[tuple[int, EvalRequest]] = []  # (index within group, request)
        for group_index, question in enumerate(question_group):
            prompt_token_ids = _render_prompt(tokenizer, question.question_text, chat_template)
            if len(prompt_token_ids) > max_prompt_tokens:
                raise ValueError(
                    f"question {question.question_id!r} renders to {len(prompt_token_ids)} tokens, "
                    f"above max_prompt_tokens={max_prompt_tokens}"
                )
            group_requests.extend(
                (
                    group_index,
                    EvalRequest(
                        question,
                        sample_index,
                        derive_sampling_seed(global_seed, dataset_name, question.question_id, sample_index),
                        prompt_token_ids,
                    ),
                )
                for sample_index in range(samples_per_question)
            )
        traces_by_question: list[list[dict[str, Any]]] = [[] for _ in question_group]
        for batch_start in range(0, len(group_requests), request_batch_size):
            request_batch = group_requests[batch_start : batch_start + request_batch_size]
            prompt_requests = [{"prompt_token_ids": request.prompt_token_ids} for _, request in request_batch]
            sampling_params = [
                _make_sampling_params(
                    sampling_params_class,
                    seed=request.sampling_seed,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_tokens,
                    predictive_topk_width=predictive_topk_width,
                )
                for _, request in request_batch
            ]
            outputs = llm.generate(prompt_requests, sampling_params, use_tqdm=True)
            if len(outputs) != len(request_batch):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for {len(request_batch)} seeded requests "
                    f"(question group starting at {group_start}, batch starting at {batch_start})"
                )

            for (group_index, request), output in zip(request_batch, outputs, strict=True):
                if len(output.outputs) != 1:
                    raise RuntimeError("vLLM must return exactly one completion per seeded request")
                if getattr(output, "prompt_token_ids", request.prompt_token_ids) != request.prompt_token_ids:
                    raise RuntimeError("vLLM conditioned on prompt tokens different from the captured request")
                completion = output.outputs[0]
                text = str(completion.text)
                score = float(grader.compute_score(text, request.question.gold_answer))
                prediction = grader.extract_prediction(text)
                prediction = None if prediction is None or not str(prediction).strip() else str(prediction)
                predictive = _extract_predictive_statistics(
                    completion,
                    topk_width=predictive_topk_width,
                    store_topk_logprobs=store_topk_logprobs,
                    store_per_token_diagnostics=store_per_token_diagnostics,
                )
                traces_by_question[group_index].append(
                    {
                        "dataset": dataset_name,
                        "uid": request.question.question_id,
                        "sample_index": request.sample_index,
                        "sampling_seed": request.sampling_seed,
                        "gold": request.question.gold_answer,
                        "prompt_text": request.question.question_text,
                        "prompt_token_ids": request.prompt_token_ids,
                        "response_text": text,
                        "response_token_ids": [int(token_id) for token_id in completion.token_ids],
                        "response_length": len(completion.token_ids),
                        "finish_reason": getattr(completion, "finish_reason", None),
                        "stop_reason": getattr(completion, "stop_reason", None),
                        "score": score,
                        "acc": score > 0.5,
                        "pred": prediction,
                        "answer_class": None,
                        **predictive,
                    }
                )

        for question_traces in traces_by_question:
            assign_semantic_answer_classes(question_traces, grader)
            for trace in question_traces:
                metric_traces.append(
                    {
                        "uid": trace["uid"],
                        "sample_index": trace["sample_index"],
                        "acc": trace["acc"],
                        "answer_class": trace["answer_class"],
                        "sequence_entropy": trace["sequence_entropy"],
                        "token_entropy_sum": trace["token_entropy_sum"],
                        "token_entropy_count": trace["token_entropy_count"],
                    }
                )
                if trace_callback is not None:
                    trace_callback(trace)
                if retain_traces:
                    retained_traces.append(trace)

    aggregation = aggregate_math_traces(
        metric_traces,
        k_values=requested_ks,
        expected_samples_per_question=samples_per_question,
        subset_strategy=subset_strategy,
        monte_carlo_resamples=monte_carlo_resamples,
        seed=metric_seed,
        prediction_field="answer_class",
        predictive_entropy_kind=PREDICTIVE_ENTROPY_KIND,
    )
    return aggregation, retained_traces


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--expected_model_identity_sha256", default=None)
    parser.add_argument(
        "--allow_unpinned_local_model",
        action="store_true",
        help="allow a local checkpoint without an externally supplied identity (smoke diagnostics only)",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--chat_template", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--dataset_manifest",
        default=None,
        help=(
            "pinned math_eval_manifest.json containing per-dataset samples/question; "
            "this is the production path for the variable >2,000-request protocol"
        ),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--trace_dir", required=True, help="required audit JSONL output directory")
    parser.add_argument("--samples_per_question", type=int, default=REQUIRED_SAMPLES_PER_QUESTION)
    parser.add_argument(
        "--allow_nonstandard_sample_count",
        action="store_true",
        help="allow a count other than the registered 64 samples/question (smoke tests only)",
    )
    parser.add_argument("--max_questions", type=int, default=-1)
    parser.add_argument(
        "--exclude_overlap_hashes_from_index",
        default=None,
        help="dataset_index.json whose registered train/validation overlap hashes should be excluded",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ks", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--subset_strategy", choices=SUBSET_STRATEGIES, default="full_only")
    parser.add_argument("--monte_carlo_resamples", type=int, default=4096)
    parser.add_argument("--metric_seed", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--max_prompt_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=12288)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--predictive_topk_width", type=int, default=DEFAULT_PREDICTIVE_TOPK_WIDTH)
    parser.add_argument("--store_predictive_topk_logprobs", action="store_true")
    parser.add_argument(
        "--store_per_token_diagnostics",
        action="store_true",
        help="store per-token entropy/mass/logprob arrays instead of only bounded summary statistics",
    )
    parser.add_argument(
        "--questions_per_batch",
        type=int,
        default=1,
        help="questions whose seeded requests are generated together (1 = one question per vLLM call)",
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=8,
        help="maximum seeded requests submitted to vLLM in one generate call",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--enforce_eager", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _ensure_python_bin_on_path()
    if (
        args.dataset_manifest is None
        and args.samples_per_question != REQUIRED_SAMPLES_PER_QUESTION
        and not args.allow_nonstandard_sample_count
    ):
        raise ValueError(
            f"production evaluation requires exactly {REQUIRED_SAMPLES_PER_QUESTION} samples/question; "
            "use --dataset_manifest for the registered variable-count protocol, or "
            "--allow_nonstandard_sample_count only for an explicit smoke test"
        )
    if args.max_prompt_tokens + args.max_tokens > args.max_model_len:
        raise ValueError("max_prompt_tokens + max_tokens exceeds max_model_len")
    if args.predictive_topk_width <= 0:
        raise ValueError("predictive_topk_width must be positive")
    if args.request_batch_size <= 0:
        raise ValueError("request_batch_size must be positive")
    if args.questions_per_batch <= 0:
        raise ValueError("questions_per_batch must be positive")
    model_is_local = Path(args.model).exists()
    if not model_is_local and not (args.model_revision and IMMUTABLE_REVISION_PATTERN.fullmatch(args.model_revision)):
        raise ValueError("a remote model requires an immutable 40/64-hex --model_revision")
    model_identity = resolve_model_identity(args.model, args.model_revision)
    expected_model_identity = args.expected_model_identity_sha256
    if model_is_local and expected_model_identity is None and not args.allow_unpinned_local_model:
        raise ValueError(
            "a local model requires --expected_model_identity_sha256; "
            "use --allow_unpinned_local_model only for an explicit smoke diagnostic"
        )
    if expected_model_identity is not None:
        expected_model_identity = require_sha256(
            expected_model_identity,
            "--expected_model_identity_sha256",
        )
        if model_identity["model_identity_sha256"] != expected_model_identity:
            raise ValueError(
                "model identity does not match --expected_model_identity_sha256: "
                f"{model_identity['model_identity_sha256']} != {expected_model_identity}"
            )

    import pandas as pd
    from vllm import LLM, SamplingParams

    from verl.utils.reward_score import math_verify as grader

    excluded_question_sha256s = (
        load_overlap_hashes_from_dataset_index(args.exclude_overlap_hashes_from_index)
        if args.exclude_overlap_hashes_from_index
        else frozenset()
    )
    dataset_manifest_path = Path(args.dataset_manifest).expanduser().resolve() if args.dataset_manifest else None
    dataset_protocol = load_dataset_protocol_manifest(dataset_manifest_path) if dataset_manifest_path else None
    dataset_manifest_sha256 = None
    if dataset_protocol is not None:
        manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        dataset_manifest_sha256 = _sha256_file(dataset_manifest_path)
        registered_template = manifest.get("chat_template", {})
        expected_template_sha256 = registered_template.get("sha256")
        actual_template_sha256 = _sha256_file(args.chat_template)
        if expected_template_sha256 != actual_template_sha256:
            raise ValueError(
                "chat template does not match the dataset protocol manifest: "
                f"expected {expected_template_sha256}, found {actual_template_sha256}"
            )
        cli_sampling = {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_response_tokens": args.max_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_model_len": args.max_model_len,
            "predictive_topk_width": args.predictive_topk_width,
        }
        if cli_sampling != manifest["sampling"]:
            raise ValueError(
                "CLI sampling settings do not match the dataset protocol manifest: "
                f"expected {manifest['sampling']}, found {cli_sampling}"
            )
    resolved_config = vars(args).copy()
    resolved_config["predictive_entropy_kind"] = PREDICTIVE_ENTROPY_KIND
    resolved_config["model_identity"] = model_identity
    resolved_config["excluded_question_sha256s"] = sorted(excluded_question_sha256s)
    resolved_config["dataset_protocol_entries"] = dataset_protocol
    resolved_config["dataset_protocol_manifest_sha256"] = dataset_manifest_sha256
    print(json.dumps({"resolved_config": resolved_config}, indent=2, sort_keys=True), flush=True)
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_logprobs": args.predictive_topk_width,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "trust_remote_code": True,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
    }
    if args.model_revision and not Path(args.model).exists():
        llm_kwargs["revision"] = args.model_revision
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    chat_template = Path(args.chat_template).read_text()
    tokenizer.chat_template = chat_template

    Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    for dataset_path in args.datasets:
        resolved_dataset_path = str(Path(dataset_path).expanduser().resolve())
        protocol_entry = dataset_protocol.get(resolved_dataset_path) if dataset_protocol is not None else None
        if dataset_protocol is not None and protocol_entry is None:
            raise ValueError(f"dataset is not registered in --dataset_manifest: {resolved_dataset_path}")
        if protocol_entry is not None:
            actual_sha256 = _sha256_file(resolved_dataset_path)
            if actual_sha256 != protocol_entry["output_sha256"]:
                raise ValueError(
                    f"dataset SHA256 mismatch for {resolved_dataset_path}: "
                    f"expected {protocol_entry['output_sha256']}, found {actual_sha256}"
                )
            samples_per_question = int(protocol_entry["samples_per_question"])
            name = str(protocol_entry["name"])
        else:
            samples_per_question = args.samples_per_question
            name = Path(dataset_path).stem
        dataframe = pd.read_parquet(dataset_path)
        questions = prepare_eval_questions(
            dataframe.to_dict(orient="records"),
            dataset_name=name,
            max_questions=args.max_questions,
            excluded_question_sha256s=excluded_question_sha256s,
        )
        if protocol_entry is not None and len(questions) != int(protocol_entry["unique_questions"]):
            raise ValueError(
                f"dataset unique-question count mismatch for {name}: "
                f"expected {protocol_entry['unique_questions']}, found {len(questions)}"
            )
        print(
            f"[{args.tag}] {name}: {len(dataframe)} source rows -> {len(questions)} unique questions; "
            f"generating {len(questions) * samples_per_question} seeded samples",
            flush=True,
        )
        trace_path = Path(args.trace_dir) / f"{args.tag}__{name}.jsonl"
        partial_trace_path = trace_path.with_suffix(trace_path.suffix + ".partial")
        partial_trace_path.unlink(missing_ok=True)
        with partial_trace_path.open("w", encoding="utf-8") as trace_handle:

            def write_trace(trace: Mapping[str, Any]) -> None:
                trace_handle.write(json.dumps(trace, ensure_ascii=False) + "\n")

            aggregation, _ = evaluate_questions(
                llm=llm,
                tokenizer=tokenizer,
                sampling_params_class=SamplingParams,
                grader=grader,
                questions=questions,
                dataset_name=name,
                chat_template=chat_template,
                samples_per_question=samples_per_question,
                global_seed=args.seed,
                k_values=[k for k in args.ks if k <= samples_per_question],
                subset_strategy=args.subset_strategy,
                monte_carlo_resamples=args.monte_carlo_resamples,
                metric_seed=args.metric_seed,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                max_prompt_tokens=args.max_prompt_tokens,
                predictive_topk_width=args.predictive_topk_width,
                store_topk_logprobs=args.store_predictive_topk_logprobs,
                store_per_token_diagnostics=args.store_per_token_diagnostics,
                request_batch_size=args.request_batch_size,
                questions_per_batch=args.questions_per_batch,
                trace_callback=write_trace,
                retain_traces=False,
            )
        partial_trace_path.replace(trace_path)
        full_metrics = aggregation["by_k"][str(samples_per_question)]
        results[name] = {
            "k": samples_per_question,
            "n_questions": aggregation["n_questions"],
            "mean@k": round(100 * full_metrics["mean_at_k"], 2),
            "pass@k": round(100 * full_metrics["pass_at_k"], 2),
            "maj@k": round(100 * full_metrics["maj_at_k"], 2),
            **aggregation,
        }
        print(
            f"[{args.tag}] {name}: wrote {len(questions) * samples_per_question} traces -> {trace_path}",
            flush=True,
        )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"tag": args.tag, "model": args.model, "config": resolved_config, "results": results}, indent=2)
        + "\n"
    )
    print(f"[{args.tag}] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
