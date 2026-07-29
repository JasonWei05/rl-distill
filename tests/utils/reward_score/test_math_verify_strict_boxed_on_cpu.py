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

"""STRICT boxed-only scoring in verl.utils.reward_score.math_verify (fork change).

compute_score is strict by default (VERL_MATH_VERIFY_STRICT_BOXED, default on):
  * score only a SINGLE \\boxed{} answer,
  * 0 boxes OR >=2 boxes -> 0.0 (no bare-LaTeX-echo credit, no multi-box hedging),
  * otherwise run math-verify (sympy) on just that one box vs the gold.

Motivation (see PROGRESS_LOG 2026-07-22): the old code parsed the *whole* output with
LatexExtractionConfig and took max over all extracted candidates, so a model could score
"correct" by echoing a bare expression with no box, or by hedging with several boxes.
"""

import pytest

from verl.utils.reward_score import math_verify as mv


@pytest.fixture(scope="module", autouse=True)
def _warm_verify_pool():
    """Force the spawn verify-pool workers to import sympy before the timed tests.

    A cold spawn (worker re-imports Python + math-verify + sympy) can exceed the 30s per-call
    timeout, making the first sympy-backed compute_score() return 0 spuriously. Warm it once.
    """
    mv._get_pool().submit(mv._verify_in_subprocess, "\\boxed{1}", "\\boxed{1}").result()
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


# ---- strict scoring (uses sympy verify) ---------------------------------------------
@pytest.mark.parametrize(
    "output,gold,expected",
    [
        (r"reasoning... The final answer is $\boxed{42}$.", "42", 1.0),   # genuine single box
        (r"The final answer is $\boxed{20}$.", "17", 0.0),                # single, wrong
        (r"The final answer is $\log_2\frac{1}{16}$.", "-4", 0.0),        # NO box -> 0 (was a false positive)
        (r"Answer is $\boxed{41}$ or $\boxed{42}$.", "42", 0.0),          # multi-box hedge -> 0
        (r"$\boxed{1729 + 867}$", "2596", 1.0),                           # single box, sympy-equal
        (r"$\boxed{\frac{1}{2}}$", "0.5", 1.0),                           # equivalent forms
    ],
)
def test_strict_compute_score(output, gold, expected):
    assert mv.compute_score(output, gold) == expected


def test_strict_zero_and_multi_box_short_circuit_without_sympy(monkeypatch):
    # The len(boxes)!=1 guard must return 0.0 without ever submitting to the verify pool.
    def _boom(*a, **k):
        raise AssertionError("verify pool should not be reached for 0/multi-box outputs")

    monkeypatch.setattr(mv, "_get_pool", _boom)
    assert mv.compute_score("no box here, answer 42", "42") == 0.0
    assert mv.compute_score(r"$\boxed{1}$ $\boxed{2}$", "2") == 0.0


def test_strict_flag_off_restores_lenient(monkeypatch):
    # The key leniency strict removes: with strict OFF the old whole-output extraction scores a bare
    # (unboxed) LaTeX expression that evaluates to the gold as 1.0; strict (default) scores it 0.
    monkeypatch.setenv("VERL_MATH_VERIFY_STRICT_BOXED", "0")
    assert mv.compute_score(r"The final answer is $\log_2\frac{1}{16}$.", "-4") == 1.0   # lenient: 1.0
    # (multi-box hedging is NOT a clean lenient win: math_verify.parse returns a single candidate — the
    # first box 41 here — so lenient also scores this 0; strict likewise returns 0 via the len!=1 guard.)
