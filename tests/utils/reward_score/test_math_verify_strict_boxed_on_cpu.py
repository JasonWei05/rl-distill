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

"""Boxed-only scoring in verl.utils.reward_score.math_verify (fork change).

compute_score is strict by default (VERL_MATH_VERIFY_STRICT_BOXED, default on):
  * require at least one well-formed \\boxed{} or \\fbox{},
  * score only the last well-formed box so a model can correct itself,
  * accept if either math_verify or the bounded Miles SymPy grader succeeds,
  * otherwise return 0.0 without crediting bare-LaTeX echoes.

Motivation (see PROGRESS_LOG 2026-07-22): the old code parsed the *whole* output with
LatexExtractionConfig and took max over all extracted candidates, so a model could score
"correct" by echoing a bare expression with no box, or by placing the correct
answer in any box rather than its final answer.
"""

import time

import pytest

from verl.utils.reward_score import math_verify as mv


@pytest.fixture(scope="module", autouse=True)
def _warm_verify_pool():
    """Force the spawn verify-pool workers to import sympy before the timed tests.

    A cold spawn (worker re-imports Python + math-verify + sympy) can exceed the 30s per-call
    timeout, making the first sympy-backed compute_score() return 0 spuriously. Warm it once.
    """
    mv._get_pool().submit(
        mv._verify_in_subprocess,
        "\\boxed{15x - 80}",
        "\\boxed{(15)x - 80.}",
        "15x - 80",
        "(15)x - 80.",
    ).result()
    yield


# ---- box extraction (fast, no sympy) ------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("no box at all", []),
        (r"the answer is \boxed{42}.", ["42"]),
        (r"\boxed{41} and later \boxed{42}", ["41", "42"]),
        (r"\boxed{\frac{1}{2}}", [r"\frac{1}{2}"]),   # balanced nested braces
        (r"\fbox{7}", ["7"]),                          # \fbox is also a box
        (r"\boxed{a + b = c}", ["a + b = c"]),
    ],
)
def test_all_boxed_contents(text, expected):
    assert mv._all_boxed_contents(text) == expected


# ---- boxed-only scoring (uses sympy verify) -----------------------------------------
@pytest.mark.parametrize(
    "output,gold,expected",
    [
        (r"reasoning... The final answer is $\boxed{42}$.", "42", 1.0),
        (r"The final answer is $\boxed{20}$.", "17", 0.0),
        (r"The final answer is $\log_2\frac{1}{16}$.", "-4", 0.0),
        (r"First $\boxed{41}$, but correcting: $\boxed{42}$.", "42", 1.0),
        (r"First $\boxed{42}$, but final answer: $\boxed{41}$.", "42", 0.0),
        (r"Correct $\boxed{42}$, then malformed $\boxed{", "42", 1.0),
        (r"$\boxed{1729 + 867}$", "2596", 1.0),
        (r"$\boxed{\frac{1}{2}}$", "0.5", 1.0),
        (r"$\boxed{(15)x - 80.}$", "15x - 80", 1.0),  # SymPy fallback only
    ],
)
def test_strict_compute_score(output, gold, expected):
    assert mv.compute_score(output, gold) == expected


def test_zero_box_short_circuits_without_sympy(monkeypatch):
    # A response with no well-formed box returns 0.0 without submitting to the verify pool.
    def _boom(*a, **k):
        raise AssertionError("verify pool should not be reached for a zero-box output")

    monkeypatch.setattr(mv, "_get_pool", _boom)
    assert mv.compute_score("no box here, answer 42", "42") == 0.0


def test_extract_prediction_uses_last_well_formed_box():
    assert mv.extract_prediction(r"First \boxed{1}, finally \fbox{2}.") == "2"
    assert mv.extract_prediction(r"Correct \boxed{2}, then malformed \boxed{") == "2"


def test_strict_flag_off_restores_lenient(monkeypatch):
    # The key leniency strict removes: with strict OFF the old whole-output extraction scores a bare
    # (unboxed) LaTeX expression that evaluates to the gold as 1.0; strict (default) scores it 0.
    monkeypatch.setenv("VERL_MATH_VERIFY_STRICT_BOXED", "0")
    assert mv.compute_score(r"The final answer is $\log_2\frac{1}{16}$.", "-4") == 1.0   # lenient: 1.0


def test_sympy_fallback_has_hard_timeout(monkeypatch):
    from verl.utils.reward_score import miles_sympy

    def _hang(*_args):
        time.sleep(10)
        return True

    monkeypatch.setattr(miles_sympy, "grade_answer_sympy", _hang)
    monkeypatch.setenv("VERL_MATH_SYMPY_TIMEOUT", "0.1")
    started = time.monotonic()
    assert mv._grade_answer_sympy_with_timeout("1", "2") == 0.0
    assert time.monotonic() - started < 1.0
