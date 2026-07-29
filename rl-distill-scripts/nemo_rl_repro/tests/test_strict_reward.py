#!/usr/bin/env python3
"""CPU strict-reward gate for the math_strict NeMo-RL environment.

Runs the 6 canonical cases against rl_distill_nemo.strict_math_env.compute_score
(the port of verl/utils/reward_score/math_verify.py strict scoring), then
cross-checks every case against the verl source module loaded by file path
(its _all_boxed_contents + _verify_in_subprocess, i.e. compute_score with the
always-on strict path, run in-process — verl's pool can't pickle across a
path-loaded module).

Run (CPU; math-verify + ray live in .venv-gemma4):
    /mnt/efs/jasonwei/rl-distill/.venv-gemma4/bin/python \
        rl-distill-scripts/nemo_rl_repro/tests/test_strict_reward.py
"""

import importlib.util
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPRO_DIR = os.path.dirname(TESTS_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(REPRO_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "third_party", "nemo-rl"))
sys.path.insert(0, REPRO_DIR)

# One spawn child is enough for the serial test cases (prod default stays 4);
# each child re-imports the heavy nemo_rl chain.
os.environ.setdefault("VERL_MATH_VERIFY_POOL_WORKERS", "1")

VERL_SCORER_PATH = os.path.join(
    REPO_ROOT, "verl", "utils", "reward_score", "math_verify.py"
)

# (name, ground_truth, model_output, expected_score)
CASES = [
    ("correct boxed", "7", "The answer is \\boxed{7}.", 1.0),
    ("wrong boxed", "7", "The answer is \\boxed{8}.", 0.0),
    ("two boxed", "7", "First \\boxed{7}, so the answer is \\boxed{7}.", 0.0),
    ("no boxed", "7", "The answer is 7.", 0.0),
    (
        "equivalent-latex boxed",
        "1/2",
        "Thus the final answer is $\\boxed{\\frac{1}{2}}$.",
        1.0,
    ),
    ("malformed latex (no crash)", "7", "So \\boxed{\\frac{1}{}} is it.", 0.0),
]


def load_verl_scorer():
    """Load verl's math_verify.py by path (skips the verl package __init__ chain)."""
    spec = importlib.util.spec_from_file_location("verl_math_verify_ref", VERL_SCORER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verl_reference_score(verl_mod, model_output: str, ground_truth: str) -> float:
    """verl compute_score, strict path (default VERL_MATH_VERIFY_STRICT_BOXED=1),
    with _verify_in_subprocess called in-process instead of through its pool."""
    boxes = verl_mod._all_boxed_contents(model_output)
    if len(boxes) != 1:
        return 0.0
    try:
        return verl_mod._verify_in_subprocess(
            "\\boxed{" + ground_truth + "}", "\\boxed{" + boxes[0] + "}"
        )
    except Exception:
        return 0.0


def main() -> int:
    from rl_distill_nemo.strict_math_env import compute_score, warm_verify_pool

    warm_verify_pool()
    verl_mod = load_verl_scorer()

    failures = 0
    for name, ground_truth, model_output, expected in CASES:
        got = compute_score(model_output, ground_truth)
        verl_got = verl_reference_score(verl_mod, model_output, ground_truth)
        if got == expected and verl_got == expected:
            print(f"[PASS] {name}: score={got}")
        else:
            failures += 1
            print(
                f"[FAIL] {name}: expected {expected}, "
                f"strict_math_env={got}, verl_reference={verl_got}"
            )

    if failures:
        print(f"\nSTRICT REWARD PARITY: FAIL ({failures}/{len(CASES)})")
        return 1
    print(f"\nSTRICT REWARD PARITY: PASS ({len(CASES)}/{len(CASES)}, "
          "port and verl reference agree)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
