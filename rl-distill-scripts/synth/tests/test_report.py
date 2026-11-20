"""report.py renderers and plan.py --pods splitting, on hand-made inputs."""

import json
import os
import subprocess
import sys

import pytest

import report

HERE = os.path.dirname(os.path.abspath(__file__))


def fake():
    """A study with one tuned, one tuning and one training SFT run, and RL cells with 0 to 3 seeds (one 1-point history)."""
    hist = lambda v: [[s, v + s / 1e5] for s in range(0, 10001, 20)]
    bases = {(size, seed): None for size in report.SIZES for seed in report.SEEDS}
    bases["2XS", 0] = {"steps": 40000, "p1": 0.0101, "p3": "not reached", "p10": "not reached"}
    bases["2XS", 1] = {"steps": 40000, "p1": None, "p3": None, "p10": None}
    rl = {(size, level): {i: [] for i in report.POSITIONS} for size in report.SIZES for level in report.LEVELS}
    rl["2XS", "p1"] = {0: [(hist(0.01), True), (hist(0.02), False)], 1: [([[0, 0.011]], False)], 2: [], 3: [(hist(0.01), True)] * 3}
    batches = {"2XS": [64, 256, 1024, 4096], "XS": [32, 128, 512, 2048], "S": [16, 64, 256, 1024]}
    step_flops = {size: [report.step_flops(size, 20, 1126, 114, b, 16) for b in batches[size]] for size in report.SIZES}
    return {"bases": bases, "rl": rl, "batches": batches, "step_flops": step_flops, "variant": report.variant(4, 2.5e-5)}


def test_markdown_and_page():
    d = fake()
    md = "\n".join(report.markdown("maze33", d))
    assert "| 2XS | 0 | 1.01 | not reached | not reached | 40000 |" in md and "| 2XS | 1 | tuning | tuning | tuning | 40000 |" in md
    assert "| XS | 0 | training | training | training | … |" in md
    assert "0.115 [1/2]" in md and "0.011 [0/1]" in md and "| – |" in md and "0.110 [3/3]" in md
    assert md.count("| 2XS | p") == 1  # RL rows only where runs started
    assert "| 64 / 32 / 16 | 256 / 128 / 64 |" in md
    page = report.page(["maze33"], {"maze33": d})
    assert page.count("<svg") == 2 and "polyline class=\"b3\"" in page and "Maze 33×33" in page
    assert page.count('class="b0 seed"') == 2 and page.count('class="b3 seed"') == 3  # every started seed, not only means
    assert page.count('class="s0 thin"') == 2 and page.count('class="s0"') == 1  # positions with points past step 0, one envelope


def test_page_with_baseline():
    """A variant page against a baseline grid: each RL cell shows the baseline's score."""
    d = {**fake(), "baseline": fake()}
    page = report.page(["maze33"], {"maze33": d}, "Maze33 Rerun", "Early stop after 100 validations.")
    assert "<title>Maze33 Rerun</title>" in page and "Early stop after 100 validations." in page
    assert page.count('was 0.110') == 1


def test_step_flops():
    """FLOP-matched positions: about equal FLOPs per step across sizes (S / 2XS 2.5-3.7x per sample, 4x fewer prompts)."""
    f = {size: report.step_flops(size, 20, 1126, 114, 1, 16) for size in report.SIZES}
    assert 1.6 < f["XS"] / f["2XS"] < 2.2 and 2.4 < f["S"] / f["2XS"] < 3.8
    assert report.step_flops("2XS", 20, 1126, 114, 64, 16) == 64 * f["2XS"]
    # 4 x the prompts, 4 updates: 4 x the samples, each also one pi_old forward outside the first mini-batch (3 of 4)
    assert report.step_flops("2XS", 20, 1126, 114, 64, 16, 4) == pytest.approx(64 * 4 * f["2XS"] * (4 + 3 / 4) / 4)


def test_plan_pods_split(tmp_path):
    # seeds no study uses, so none of the runs is done on S3
    out = str(tmp_path / "sft")
    subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), "plan.py"), "sft", "--configs", "maze33",
                    "--sizes", "2XS", "XS", "S", "--seeds", "7", "8", "9", "--pods", "2", "--out", out], check=True)
    plans = [json.load(open(f"{out}-{i}.json")) for i in range(2)]
    runs = [j["run"] for p in plans for j in p["jobs"]]
    assert len(runs) == len(set(runs)) == 9 and all(p["data"] == ["maze33"] for p in plans)
    loads = [sum(j["share"] * j["gpus"] for j in p["jobs"]) for p in plans]
    assert abs(loads[0] - loads[1]) <= 0.34 + 1e-9
