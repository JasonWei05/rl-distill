"""tune.py on a synthetic run whose pass@1 is a known function of the weights, so the search is checked exactly."""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

import tune
from data import Dataset
from model import hf_config, init_weights, load_model

STEPS = range(0, 3001, 10)
KEY = "model.norm.weight"


def pass1_at(step):
    """The fake run's pass@1 after `step` steps: rises from 0 to 0.2, with a dip around 1500."""
    x = step / 3000
    return 0.2 * x ** 2 * (1 - 0.3 * (abs(step - 1500) < 60))


@pytest.fixture()
def fake_run(sudoku_data, tmp_path, monkeypatch):
    """A run dir whose checkpoint at step s has KEY's first entry = s, and a tuner whose pass@1 of weights w is
    pass1_at(w[KEY][0]) (exact for averages: the averaged entry is the mean step)."""
    d = Dataset(sudoku_data)
    hf = hf_config("2XS", len(d.task.WORDS), d.width)
    w = init_weights(hf, 0)
    os.makedirs(tmp_path / "ckpt")
    for s in STEPS:
        w[KEY][0] = s
        save_file(w, str(tmp_path / "ckpt" / f"{s}.safetensors"))
    with open(tmp_path / "run.json", "w") as f:
        json.dump({"config": d.cfg, "size": "2XS", "seed": 0, "hf": hf}, f)
    with open(tmp_path / "screen.jsonl", "w") as f:
        for s in range(0, 3001, 20):
            f.write(json.dumps({"step": s, "score": pass1_at(s) * 1.1}) + "\n")  # noisy screens
    monkeypatch.setattr(tune.Tuner, "pass1", lambda self, w, k: pass1_at(w[KEY][0].item()))
    return str(tmp_path)


def test_tune_finds_levels(fake_run, sudoku_data):
    t = tune.Tuner(fake_run, sudoku_data)
    t.t = {**t.t, "levels": {"p1": 0.01, "p3": 0.03, "p10": 0.10, "p30": 0.3, "early": 1e-5}, "window": 20, "tol": 0.01}
    t.run_all()
    log = json.load(open(os.path.join(fake_run, "tune.json")))
    assert not log["p30"]["reachable"] and not os.path.exists(os.path.join(fake_run, "tune", "p30"))
    for level, target in (("p1", 0.01), ("p3", 0.03), ("p10", 0.10), ("early", 1e-5)):  # early: crossed at step ~21
        r = log[level]
        assert r["reachable"] and abs(r["pass1"] / target - 1) <= 0.01
        assert r["hi"] - r["lo"] == 10 and pass1_at(r["lo"]) < target <= pass1_at(r["hi"])
        cfg, synth, w = load_model(os.path.join(fake_run, "tune", level))
        assert synth["level"] == level and synth["pass1"] == r["pass1"] and cfg == t.meta["hf"]
        assert abs(pass1_at(w[KEY][0].item()) - r["pass1"]) < 1e-6
    assert pass1_at(log["p1"]["hi"]) < 0.0115  # below the dip: the first crossing, 0.01 at step ~671


def test_tune_resumes(fake_run, sudoku_data, monkeypatch):
    t = tune.Tuner(fake_run, sudoku_data)
    t.t = {**t.t, "levels": {"p1": 0.01}}
    t.run_all()
    monkeypatch.setattr(tune.Tuner, "tune", lambda *a: pytest.fail("a finished level was tuned again"))
    t2 = tune.Tuner(fake_run, sudoku_data)
    t2.t = {**t2.t, "levels": {"p1": 0.01}}
    t2.run_all()


def test_average(sudoku_data, tmp_path):
    """average(center): the mean of the checkpoints within +-window steps of center."""
    d = Dataset(sudoku_data)
    hf = hf_config("2XS", len(d.task.WORDS), d.width)
    os.makedirs(tmp_path / "ckpt")
    ws = {s: init_weights(hf, s) for s in (0, 10, 20, 30)}
    for s, w in ws.items():
        save_file(w, str(tmp_path / "ckpt" / f"{s}.safetensors"))
    with open(tmp_path / "run.json", "w") as f:
        json.dump({"config": d.cfg, "size": "2XS", "seed": 0, "hf": hf}, f)
    open(tmp_path / "screen.jsonl", "w").close()
    t = tune.Tuner(str(tmp_path), sudoku_data)
    t.t = {**t.t, "window": 10}
    avg = t.average(20)
    for k in avg:
        assert torch.allclose(avg[k].cpu(), (ws[10][k] + ws[20][k] + ws[30][k]) / 3, atol=1e-7)
