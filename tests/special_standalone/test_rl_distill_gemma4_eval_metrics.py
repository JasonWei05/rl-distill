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

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).parents[2] / "rl-distill-scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import eval_gemma4_ood as ood_eval  # noqa: E402
from eval_gemma4_ood import OODEvalConfig, build_ood_commands  # noqa: E402
from eval_math_passk import (  # noqa: E402
    PREDICTIVE_ENTROPY_KIND,
    assign_semantic_answer_classes,
    derive_sampling_seed,
    evaluate_questions,
    load_overlap_hashes_from_dataset_index,
    prepare_eval_questions,
)
from gemma4_eval_metrics import (  # noqa: E402
    aggregate_math_traces,
    majority_at_k_prefix,
    pass_at_k,
    pass_at_k_dataset,
    topk_plus_residual_bucket_entropy,
)


@pytest.mark.parametrize(
    ("n", "c", "k"),
    [
        (64, 0, 1),
        (64, 1, 1),
        (64, 1, 32),
        (64, 8, 16),
        (64, 63, 2),
        (64, 64, 64),
        (7, 3, 4),
    ],
)
def test_pass_at_k_matches_exact_combinatorial_form(n: int, c: int, k: int) -> None:
    expected = 1.0 - math.comb(n - c, k) / math.comb(n, k) if n - c >= k else 1.0
    assert pass_at_k(n, c, k) == pytest.approx(expected, abs=1e-15)


def test_pass_at_k_dataset_is_macro_average() -> None:
    assert pass_at_k_dataset([4, 4], [0, 2], 2) == pytest.approx((0.0 + 5.0 / 6.0) / 2.0)


@pytest.mark.parametrize(("n", "c", "k"), [(0, 0, 1), (4, -1, 1), (4, 5, 1), (4, 1, 0), (4, 1, 5)])
def test_pass_at_k_rejects_invalid_counts(n: int, c: int, k: int) -> None:
    with pytest.raises(ValueError):
        pass_at_k(n, c, k)


def test_majority_ties_and_all_abstain_are_wrong() -> None:
    # The first response is correct, so this specifically guards against the old
    # first-occurrence tie breaker.
    assert majority_at_k_prefix([True, False], ["42", "17"], 2) == 0.0
    assert majority_at_k_prefix([False, False], [None, "<none>"], 2) == 0.0


def test_abstentions_remain_in_denominator_and_majority_rule_is_explicit() -> None:
    correctness = [True, False, False]
    predictions = ["42", None, None]
    assert majority_at_k_prefix(correctness, predictions, 3, majority_rule="plurality") == 1.0
    assert majority_at_k_prefix(correctness, predictions, 3, majority_rule="strict_majority") == 0.0

    traces = [
        {"uid": "q", "sample_index": index, "acc": correct, "answer_class": prediction}
        for index, (correct, prediction) in enumerate(zip(correctness, predictions, strict=True))
    ]
    result = aggregate_math_traces(
        traces,
        k_values=[3],
        expected_samples_per_question=3,
        prediction_field="answer_class",
    )
    assert result["by_k"]["3"]["winner_vote_fraction_all_samples"] == pytest.approx(1 / 3)
    assert result["by_k"]["3"]["abstention_rate"] == pytest.approx(2 / 3)


def _make_64_sample_question(uid: str, correct_count: int) -> list[dict[str, object]]:
    rows = []
    for sample_index in range(64):
        correct = sample_index < correct_count
        rows.append(
            {
                "uid": uid,
                "sample_index": sample_index,
                "acc": correct,
                "answer_class": "gold" if correct else f"wrong-{sample_index}",
            }
        )
    return rows


def test_full_64_sample_aggregation_and_unbiased_pass_curve() -> None:
    traces = _make_64_sample_question("q1", 8) + _make_64_sample_question("q2", 0)
    result = aggregate_math_traces(
        traces,
        k_values=[1, 64],
        expected_samples_per_question=64,
        subset_strategy="full_only",
        prediction_field="answer_class",
        include_per_question=True,
    )

    assert result["n_questions"] == 2
    assert result["n_samples"] == 128
    assert result["by_k"]["1"]["pass_at_k"] == pytest.approx((8 / 64) / 2)
    assert result["by_k"]["1"]["mean_at_k"] is None
    assert "not_reported" in result["by_k"]["1"]["subset_metrics_status"]
    assert result["by_k"]["64"]["pass_at_k"] == 0.5
    assert result["by_k"]["64"]["mean_at_k"] == pytest.approx((8 / 64) / 2)
    assert result["per_question"]["q1"]["by_k"]["1"]["pass_at_k"] == pytest.approx(8 / 64)
    assert result["policies"]["majority_ties"] == "wrong; no order-based tie breaker"


def test_prefix_diagnostic_uses_sample_index_not_file_order() -> None:
    traces = [
        {"uid": "q", "sample_index": 3, "acc": False, "answer_class": "d"},
        {"uid": "q", "sample_index": 0, "acc": True, "answer_class": "a"},
        {"uid": "q", "sample_index": 2, "acc": False, "answer_class": "c"},
        {"uid": "q", "sample_index": 1, "acc": False, "answer_class": "b"},
    ]
    result = aggregate_math_traces(
        traces,
        k_values=[2],
        expected_samples_per_question=4,
        subset_strategy="prefix",
        prediction_field="answer_class",
    )
    assert result["by_k"]["2"]["mean_at_k"] == 0.5
    assert result["by_k"]["2"]["maj_at_k"] == 0.0
    assert "diagnostics" in result["policies"]["prefix_warning"]


def test_seeded_monte_carlo_subset_estimate_is_reproducible() -> None:
    traces = [
        {"uid": "q", "sample_index": index, "acc": index < 2, "answer_class": "gold" if index < 2 else str(index)}
        for index in range(6)
    ]
    kwargs = {
        "k_values": [3],
        "expected_samples_per_question": 6,
        "subset_strategy": "monte_carlo",
        "monte_carlo_resamples": 100,
        "seed": 123,
        "prediction_field": "answer_class",
    }
    first = aggregate_math_traces(traces, **kwargs)
    second = aggregate_math_traces(list(reversed(traces)), **kwargs)
    assert first["by_k"] == second["by_k"]
    assert first["by_k"]["3"]["subset_draws_per_question"] == 100


def test_entropy_aggregation_separates_sequence_and_token_weighting() -> None:
    traces = [
        {
            "uid": "q1",
            "sample_index": 0,
            "acc": True,
            "answer_class": "1",
            "token_entropies": [1.0, 2.0, 3.0],
            "response_mask": [0, 1, 1],
        },
        {
            "uid": "q2",
            "sample_index": 0,
            "acc": False,
            "answer_class": "2",
            "response_entropy": 1.5,
        },
    ]
    result = aggregate_math_traces(
        traces,
        k_values=[1],
        expected_samples_per_question=1,
        predictive_entropy_kind="exact_full_vocab",
        prediction_field="answer_class",
    )
    entropy = result["predictive_entropy"]
    assert entropy["sequence_weighted_mean_nats"] == 2.0
    assert entropy["question_weighted_mean_nats"] == 2.0
    assert entropy["token_weighted_mean_nats"] == 2.5
    assert entropy["n_response_tokens_with_entropy"] == 2
    assert result["by_k"]["1"]["predictive_entropy_sequence_mean_nats"] == 2.0


def test_entropy_aggregation_accepts_compact_token_summary() -> None:
    result = aggregate_math_traces(
        [
            {
                "uid": "q",
                "sample_index": 0,
                "acc": True,
                "answer_class": "1",
                "sequence_entropy": 2.0,
                "token_entropy_sum": 6.0,
                "token_entropy_count": 3,
            }
        ],
        k_values=[1],
        expected_samples_per_question=1,
        predictive_entropy_kind="topk_summary",
        prediction_field="answer_class",
    )
    entropy = result["predictive_entropy"]
    assert entropy["sequence_weighted_mean_nats"] == 2.0
    assert entropy["token_weighted_mean_nats"] == 2.0
    assert entropy["n_response_tokens_with_entropy"] == 3


def test_topk_plus_residual_bucket_entropy_has_explicit_lower_bound_semantics() -> None:
    logprobs = np.asarray([[math.log(0.5), math.log(0.25), -math.inf]], dtype=np.float64)
    entropy = topk_plus_residual_bucket_entropy(logprobs)
    expected = -(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25))
    np.testing.assert_allclose(entropy, [expected])


def test_topk_entropy_accepts_103_way_equal_probability_fp16_rounding() -> None:
    # Independently storing 103 copies of log(1/103) in FP16 reconstructs a
    # mass around 1.001918. This valid regression motivated the shared 0.0025
    # tolerance in the distillation trace schema and evaluation code.
    logprobs = np.full((1, 103), np.float16(-np.log(103)), dtype=np.float16)
    reconstructed_mass = float(np.exp(logprobs.astype(np.float64)).sum())
    assert 1.001 < reconstructed_mass < 1.0025
    entropy = topk_plus_residual_bucket_entropy(logprobs)
    assert entropy.shape == (1,)
    assert np.isfinite(entropy[0])


def test_inconsistent_correctness_for_one_answer_class_is_rejected() -> None:
    traces = [
        {"uid": "q", "sample_index": 0, "acc": True, "answer_class": "same"},
        {"uid": "q", "sample_index": 1, "acc": False, "answer_class": "same"},
    ]
    with pytest.raises(ValueError, match="inconsistent correctness"):
        aggregate_math_traces(
            traces,
            k_values=[2],
            expected_samples_per_question=2,
            prediction_field="answer_class",
        )


def test_expected_sample_indices_must_be_exact_contiguous_range() -> None:
    traces = [
        {"uid": "q", "sample_index": 10, "acc": True, "answer_class": "a"},
        {"uid": "q", "sample_index": 11, "acc": False, "answer_class": "b"},
    ]
    with pytest.raises(ValueError, match="exactly 0..1"):
        aggregate_math_traces(
            traces,
            k_values=[2],
            expected_samples_per_question=2,
            prediction_field="answer_class",
        )

    traces[0]["sample_index"] = -1
    with pytest.raises(ValueError, match="negative sample_index"):
        aggregate_math_traces(
            traces,
            k_values=[2],
            expected_samples_per_question=None,
            prediction_field="answer_class",
        )


def test_external_trace_prompt_and_gold_must_be_constant_per_uid() -> None:
    traces = [
        {"uid": "q", "sample_index": 0, "acc": True, "answer_class": "a", "prompt_text": "p", "gold": "1"},
        {"uid": "q", "sample_index": 1, "acc": False, "answer_class": "b", "prompt_text": "other", "gold": "1"},
    ]
    with pytest.raises(ValueError, match="conflicting prompt_text"):
        aggregate_math_traces(
            traces,
            k_values=[2],
            expected_samples_per_question=2,
            prediction_field="answer_class",
        )


def test_offline_evaluator_entry_point(tmp_path: Path) -> None:
    trace_path = tmp_path / "tiny.jsonl"
    rows = [
        {"dataset": "tiny", "uid": uid, "sample_index": index, "acc": index == 0, "answer_class": str(index)}
        for uid in ("q1", "q2")
        for index in range(2)
    ]
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output_path = tmp_path / "metrics.json"

    command = [
        sys.executable,
        str(SCRIPTS_DIR / "eval_gemma4_math.py"),
        "--traces",
        str(trace_path),
        "--out",
        str(output_path),
        "--ks",
        "1",
        "2",
        "--expected-samples-per-question",
        "2",
        "--prediction-field",
        "answer_class",
    ]
    blocked = subprocess.run(command, check=False, capture_output=True, text=True)
    assert blocked.returncode != 0
    assert "Refuse to label lexical voting maj@k" in blocked.stderr

    completed = subprocess.run(
        [*command, "--allow-lexical-majority"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "resolved_config" in completed.stdout
    output = json.loads(output_path.read_text())
    assert output["results"]["tiny"]["by_k"]["1"]["pass_at_k"] == 0.5
    assert output["results"]["tiny"]["by_k"]["2"]["mean_at_k"] == 0.5


def test_question_hash_grouping_is_stable_without_uid() -> None:
    rows = [
        {"prompt": [{"role": "user", "content": "one"}], "reward_model": {"ground_truth": "1"}},
        {"prompt": [{"role": "user", "content": "one"}], "reward_model": {"ground_truth": "1"}},
        {"prompt": [{"role": "user", "content": "two"}], "reward_model": {"ground_truth": "2"}},
    ]
    forward = prepare_eval_questions(rows, dataset_name="math")
    reverse = prepare_eval_questions(list(reversed(rows)), dataset_name="math")
    assert len(forward) == 2
    assert [question.question_id for question in forward] == [question.question_id for question in reverse]
    assert all(question.question_id.startswith("sha256:") for question in forward)

    conflicting = rows + [{"prompt": [{"role": "user", "content": "one"}], "reward_model": {"ground_truth": "999"}}]
    with pytest.raises(ValueError, match="conflicting question/gold"):
        prepare_eval_questions(conflicting, dataset_name="math")


def test_registered_overlap_hashes_produce_clean_validation_subset(tmp_path: Path) -> None:
    leaked_text = "leaked question"
    leaked_hash = hashlib.sha256(leaked_text.encode()).hexdigest()
    index_path = tmp_path / "dataset_index.json"
    index_path.write_text(
        json.dumps(
            {
                "cross_split_question_text_overlap_count": 1,
                "cross_split_question_text_overlap_sha256s": [leaked_hash],
            }
        ),
        encoding="utf-8",
    )
    excluded = load_overlap_hashes_from_dataset_index(index_path)
    rows = [
        {
            "uid": "leaked",
            "prompt": [{"role": "user", "content": leaked_text}],
            "reward_model": {"ground_truth": "1"},
        },
        {
            "uid": "clean",
            "prompt": [{"role": "user", "content": "clean question"}],
            "reward_model": {"ground_truth": "2"},
        },
    ]
    questions = prepare_eval_questions(
        rows,
        dataset_name="validation",
        excluded_question_sha256s=excluded,
    )
    assert [question.question_id for question in questions] == ["clean"]


class _FakeSamplingParams:
    def __init__(self, **kwargs: object):
        self.__dict__.update(kwargs)


class _FakeTokenizer:
    def apply_chat_template(self, messages: list[dict[str, str]], **_: object) -> str:
        return messages[-1]["content"]

    def encode(self, rendered: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [1 if rendered == "one" else 2]


class _FakeGrader:
    @staticmethod
    def extract_prediction(text: str) -> str | None:
        prefix = "\\boxed{"
        if prefix not in text:
            return None
        return text.split(prefix, 1)[1].split("}", 1)[0]

    @classmethod
    def compute_score(cls, text: str, gold: str) -> float:
        return float(cls.extract_prediction(text) == gold)


class _SemanticFakeGrader(_FakeGrader):
    @staticmethod
    def _normalize(value: str | None) -> str | None:
        if value in {"0.5", "1/2", r"\frac{1}{2}"}:
            return "half"
        return value

    @classmethod
    def compute_score(cls, text: str, gold: str, timeout_score: float = 0.0) -> float:
        del timeout_score
        prediction = text[len("\\boxed{") : -1] if text.startswith("\\boxed{") and text.endswith("}") else None
        return float(cls._normalize(prediction) == cls._normalize(gold))


class _FakeLLM:
    def __init__(self) -> None:
        self.sampling_params: list[_FakeSamplingParams] = []
        self.all_sampling_params: list[_FakeSamplingParams] = []
        self.generate_batch_sizes: list[int] = []
        self.per_question_count: dict[int, int] = {}

    def generate(
        self,
        prompt_requests: list[dict[str, list[int]]],
        sampling_params: list[_FakeSamplingParams],
        *,
        use_tqdm: bool,
    ) -> list[SimpleNamespace]:
        assert use_tqdm
        self.sampling_params = sampling_params
        self.all_sampling_params.extend(sampling_params)
        self.generate_batch_sizes.append(len(prompt_requests))
        outputs = []
        for request in prompt_requests:
            question_token = request["prompt_token_ids"][0]
            sample_index = self.per_question_count.get(question_token, 0)
            self.per_question_count[question_token] = sample_index + 1
            if question_token == 1:
                prediction = "1" if sample_index == 0 else "9"
            else:
                prediction = "2"
            logprobs = {
                10: SimpleNamespace(logprob=math.log(0.7), rank=1),
                11: SimpleNamespace(logprob=math.log(0.2), rank=2),
            }
            completion = SimpleNamespace(
                text=f"answer \\boxed{{{prediction}}}",
                token_ids=[10],
                logprobs=[logprobs],
                finish_reason="stop",
                stop_reason="<end_of_turn>",
            )
            outputs.append(SimpleNamespace(prompt_token_ids=request["prompt_token_ids"], outputs=[completion]))
        return outputs


def test_mocked_seeded_generation_score_and_aggregate_e2e() -> None:
    rows = [
        {
            "uid": "q-one",
            "prompt": [{"role": "user", "content": "one"}],
            "reward_model": {"ground_truth": "1"},
        },
        {
            "uid": "q-two",
            "prompt": [{"role": "user", "content": "two"}],
            "reward_model": {"ground_truth": "2"},
        },
    ]
    questions = prepare_eval_questions(rows, dataset_name="tiny")
    llm = _FakeLLM()
    aggregation, traces = evaluate_questions(
        llm=llm,
        tokenizer=_FakeTokenizer(),
        sampling_params_class=_FakeSamplingParams,
        grader=_FakeGrader(),
        questions=questions,
        dataset_name="tiny",
        chat_template="unused",
        samples_per_question=2,
        global_seed=17,
        k_values=[1, 2],
        predictive_topk_width=2,
        max_tokens=8,
        max_prompt_tokens=8,
    )

    assert len(traces) == 4
    assert aggregation["by_k"]["1"]["pass_at_k"] == pytest.approx(0.75)
    assert aggregation["by_k"]["2"]["mean_at_k"] == pytest.approx(0.75)
    assert aggregation["by_k"]["2"]["maj_at_k"] == pytest.approx(0.5)
    assert aggregation["predictive_entropy"]["sequence_coverage"] == 1.0
    assert aggregation["predictive_entropy"]["source_kind"] == PREDICTIVE_ENTROPY_KIND
    assert all(trace["predictive_topk_mass_mean"] == pytest.approx(0.9) for trace in traces)
    assert all(trace["predictive_topk_mass_count"] == 1 for trace in traces)
    assert all(trace["sampled_token_logprob_count"] == 1 for trace in traces)
    assert all("predictive_topk_mass" not in trace for trace in traces)
    assert all("sampled_token_logprobs" not in trace for trace in traces)
    assert [params.seed for params in llm.all_sampling_params] == [trace["sampling_seed"] for trace in traces]
    assert traces[0]["sampling_seed"] == derive_sampling_seed(17, "tiny", traces[0]["uid"], 0)
    assert len({trace["sampling_seed"] for trace in traces}) == 4


def test_seeded_generation_uses_bounded_request_batches() -> None:
    rows = [
        {
            "uid": "q-one",
            "prompt": [{"role": "user", "content": "one"}],
            "reward_model": {"ground_truth": "1"},
        },
        {
            "uid": "q-two",
            "prompt": [{"role": "user", "content": "two"}],
            "reward_model": {"ground_truth": "2"},
        },
    ]
    llm = _FakeLLM()
    _, traces = evaluate_questions(
        llm=llm,
        tokenizer=_FakeTokenizer(),
        sampling_params_class=_FakeSamplingParams,
        grader=_FakeGrader(),
        questions=prepare_eval_questions(rows, dataset_name="tiny"),
        dataset_name="tiny",
        chat_template="unused",
        samples_per_question=2,
        k_values=[1, 2],
        predictive_topk_width=2,
        max_tokens=8,
        max_prompt_tokens=8,
        request_batch_size=1,
    )

    assert llm.generate_batch_sizes == [1, 1, 1, 1]
    assert [params.seed for params in llm.all_sampling_params] == [trace["sampling_seed"] for trace in traces]


def test_seeded_generation_can_stream_full_traces_without_retaining_them() -> None:
    rows = [
        {
            "uid": "q-one",
            "prompt": [{"role": "user", "content": "one"}],
            "reward_model": {"ground_truth": "1"},
        }
    ]
    streamed = []
    aggregation, traces = evaluate_questions(
        llm=_FakeLLM(),
        tokenizer=_FakeTokenizer(),
        sampling_params_class=_FakeSamplingParams,
        grader=_FakeGrader(),
        questions=prepare_eval_questions(rows, dataset_name="tiny"),
        dataset_name="tiny",
        chat_template="unused",
        samples_per_question=2,
        k_values=[1, 2],
        predictive_topk_width=2,
        max_tokens=8,
        max_prompt_tokens=8,
        request_batch_size=1,
        trace_callback=streamed.append,
        retain_traces=False,
    )

    assert traces == []
    assert len(streamed) == 2
    assert all(trace["answer_class"] is not None for trace in streamed)
    assert aggregation["n_questions"] == 1


def test_semantic_answer_classes_prevent_equivalent_correct_forms_from_splitting_majority() -> None:
    predictions = ["0.5", "1/2", r"\frac{1}{2}", "2", "2"]
    traces = [
        {
            "uid": "q",
            "sample_index": index,
            "acc": index < 3,
            "pred": prediction,
            "answer_class": None,
        }
        for index, prediction in enumerate(predictions)
    ]
    assign_semantic_answer_classes(traces, _SemanticFakeGrader())
    assert len({trace["answer_class"] for trace in traces[:3]}) == 1
    assert traces[3]["answer_class"] == traces[4]["answer_class"]
    assert traces[0]["answer_class"] != traces[3]["answer_class"]

    result = aggregate_math_traces(
        traces,
        k_values=[5],
        expected_samples_per_question=5,
        prediction_field="answer_class",
    )
    assert result["by_k"]["5"]["maj_at_k"] == 1.0
    answer_ambiguity = next(item for item in result["scientific_ambiguities"] if item["id"] == "answer_equivalence")
    assert answer_ambiguity["resolved"] is True


def test_ood_wrapper_builds_exact_five_task_shot_matrix(tmp_path: Path) -> None:
    config = OODEvalConfig(
        model="google/gemma-4-E2B",
        model_revision="a" * 40,
        output_dir=str(tmp_path),
        tensor_parallel_size=2,
        seed=7,
    )
    commands = build_ood_commands(config, lm_eval_executable="/venv/bin/lm_eval")
    assert len(commands) == 3
    assert [command[command.index("--num_fewshot") + 1] for command in commands] == ["5", "10", "25"]
    assert [command[command.index("--tasks") + 1] for command in commands] == [
        "mmlu,winogrande,triviaqa",
        "hellaswag",
        "arc_challenge",
    ]
    assert all("--limit" not in command for command in commands)
    assert all("revision=" + "a" * 40 in command[command.index("--model_args") + 1] for command in commands)
    assert all("tensor_parallel_size=2" in command[command.index("--model_args") + 1] for command in commands)


def test_ood_identity_binds_import_to_clean_pinned_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "venv" / "bin" / "lm_eval"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n")
    repo = tmp_path / "harness"
    (repo / "lm_eval").mkdir(parents=True)
    module_path = repo / "lm_eval" / "__init__.py"
    module_path.write_text("")
    revision = "f" * 40

    monkeypatch.setattr(
        ood_eval,
        "_executable_package_identity",
        lambda _executable: {"version": "0.4.13.dev0", "module_path": str(module_path)},
    )

    dirty = False

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        if "rev-parse" in command:
            return SimpleNamespace(stdout=f"{repo}\n{revision}\n")
        if "status" in command:
            return SimpleNamespace(stdout="?? local.py\n" if dirty else "")
        raise AssertionError(command)

    monkeypatch.setattr(ood_eval.subprocess, "run", fake_run)
    identity = ood_eval.resolve_harness_identity(
        lm_eval_executable=str(executable),
        expected_version="0.4.13.dev0",
        harness_repo=str(repo),
        expected_git_revision=revision,
    )
    assert identity["module_path"] == str(module_path)
    assert identity["git_dirty"] is False

    dirty = True
    with pytest.raises(RuntimeError, match="checkout is dirty"):
        ood_eval.resolve_harness_identity(
            lm_eval_executable=str(executable),
            expected_version="0.4.13.dev0",
            harness_repo=str(repo),
            expected_git_revision=revision,
        )

    outside_module = tmp_path / "other" / "lm_eval" / "__init__.py"
    outside_module.parent.mkdir(parents=True)
    outside_module.write_text("")
    monkeypatch.setattr(
        ood_eval,
        "_executable_package_identity",
        lambda _executable: {"version": "0.4.13.dev0", "module_path": str(outside_module)},
    )
    dirty = False
    with pytest.raises(RuntimeError, match="outside the pinned harness checkout"):
        ood_eval.resolve_harness_identity(
            lm_eval_executable=str(executable),
            expected_version="0.4.13.dev0",
            harness_repo=str(repo),
            expected_git_revision=revision,
        )
