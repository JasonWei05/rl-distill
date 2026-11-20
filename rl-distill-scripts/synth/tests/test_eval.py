"""eval.py's pass@k: the unbiased estimator against brute force over subsets, and per-group summaries."""

import itertools
import os
import subprocess
import sys

import torch

import eval as ev


def test_pass_at_is_the_subset_average():
    """pass@k = the fraction of k-subsets of the n samples with a correct one."""
    for n, c, k in ((8, 0, 1), (8, 3, 1), (8, 3, 4), (8, 8, 8), (6, 1, 6), (7, 2, 3)):
        samples = [1] * c + [0] * (n - c)
        subsets = list(itertools.combinations(samples, k))
        assert abs(ev.pass_at(c, n, k) - sum(any(s) for s in subsets) / len(subsets)) < 1e-12


def test_summary_per_rows():
    """Float judge scores (as every judge returns) of 3 prompts x 4 samples, over all prompts and a slice of them."""
    score = torch.tensor([1., 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    out = {"score": score, "closed_think": torch.ones(12)}
    s = ev.summary(out, 4, [1, 4], slice(None))
    assert s["prompts"] == 3 and abs(s["pass@1"] - 5 / 12) < 1e-12 and abs(s["pass@4"] - 2 / 3) < 1e-12 and s["closed_think"] == 1.0
    s = ev.summary(out, 4, [1, 4], slice(1, 3))
    assert s["prompts"] == 2 and s["pass@1"] == 0.5 and s["pass@4"] == 0.5


def test_eval_curve_fails_when_no_checkpoint_lands(tmp_path):
    """eval_curve exits non-zero (pack retries, then fails the job) once no checkpoint has landed for --idle-hours."""
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_curve.py")
    r = subprocess.run([sys.executable, script, "--run", str(tmp_path / "run"), "--data", "x", "--out", str(tmp_path / "out"),
                        "--idle-hours", "0"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "no new checkpoint" in r.stderr and not (tmp_path / "out" / "DONE").exists()
