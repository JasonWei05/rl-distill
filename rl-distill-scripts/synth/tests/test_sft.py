"""sft.py: the batches, the loss against HF, training, resume and the stop rule."""

import argparse
import json
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import Qwen2Config, Qwen2ForCausalLM

import sft
from data import Dataset
from tasks import PAD


def run(data, out, seed=0, **sft_cfg):
    """An SFT on the dataset, with its sft settings overridden by sft_cfg for this process only."""
    t = sft.SFT(argparse.Namespace(data=data, size="2XS", seed=seed, out=str(out), init=None, chunk=None, wandb_project="x", wandb_id="x"))
    t.s = {**t.s, **sft_cfg}
    return t


def test_batches(sudoku_data):
    """Same rows for any init seed, every row once per epoch, width = the longest row rounded up to ALIGN."""
    d = Dataset(sudoku_data)
    rows = d.split("train")
    b = sft.Batches(rows, d.prompt_len, 128, 0, 1)
    per_epoch = len(rows) // 128
    epoch = np.concatenate([b.indices(s) for s in range(1, per_epoch + 1)])
    assert len(set(epoch.tolist())) == per_epoch * 128
    assert not np.array_equal(b.indices(1), b.indices(per_epoch + 1))
    for s in range(1, 4):
        x = b.next().cpu().numpy()
        ref = rows[b.indices(s)]
        n_resp = int((ref[:, d.prompt_len:] != PAD).sum(1).max())
        assert x.shape[1] == d.prompt_len + -(-n_resp // sft.ALIGN) * sft.ALIGN
        assert np.array_equal(x, ref[:, :x.shape[1]]) and (ref[:, x.shape[1]:] == PAD).all()


def test_loss_matches_hf(sudoku_data, tmp_path):
    """The compiled bf16 loss against HF Qwen2 in fp32: cross entropy of the response tokens, mean over them."""
    t = run(sudoku_data, tmp_path)
    ids = torch.from_numpy(t.data.split("train")[:32].astype(np.int64)).cuda()
    hf = Qwen2ForCausalLM(Qwen2Config(**t.hf)).cuda()
    hf.load_state_dict({**{k: v.detach() for k, v in t.params.items()}, "lm_head.weight": t.params["model.embed_tokens.weight"].detach()})
    Lp = t.data.prompt_len
    with torch.no_grad():
        logits = hf(ids).logits[:, Lp - 1:-1]
        ref = F.cross_entropy(logits.flatten(0, 1), ids[:, Lp:].flatten(), ignore_index=PAD)
        mine = t.loss(t.params, ids)
    assert abs(mine.item() - ref.item()) < 2e-3 * ref.item()


def test_train_and_resume(sudoku_data, tmp_path):
    """60 steps in one go and as 30 + a resumed 30: the resume restores the weights, optimizer and screens exactly; both
    runs have the same screens and checkpoints, and end equal up to run-to-run noise (the bf16 backward is not
    deterministic: two fresh runs differ by ~0.01 after 60 steps at lr 5e-4); the loss falls."""
    run(sudoku_data, tmp_path / "a").run()
    run(sudoku_data, tmp_path / "b", max_steps=30).run()
    os.remove(tmp_path / "b" / "DONE")
    b = run(sudoku_data, tmp_path / "b")
    saved = load_file(tmp_path / "b" / "ckpt" / "30.safetensors")
    assert b.step == 30 and [r["step"] for r in b.screens] == [0, 20]
    assert all(torch.equal(b.params[k].detach().cpu(), saved[k]) for k in saved)
    assert b.opt.state_dict()["state"][0]["step"] == 30
    b.run()
    sa, sb = ([json.loads(line) for line in open(tmp_path / r / "screen.jsonl")] for r in "ab")
    for r, screens in (("a", sa), ("b", sb)):
        assert [s["step"] for s in screens] == [0, 20, 40, 60]
        assert sorted(os.listdir(tmp_path / r / "ckpt")) == sorted(f"{s}.safetensors" for s in range(0, 61, 10))
        assert json.load(open(tmp_path / r / "DONE")) == {"step": 60, "reason": "max_steps"}
        assert screens[-1]["val_loss"] < 0.8 * screens[0]["val_loss"]
    assert abs(sa[0]["val_loss"] / sb[0]["val_loss"] - 1) < 2e-3 and abs(sa[-1]["val_loss"] / sb[-1]["val_loss"] - 1) < 0.02
    wa, wb = (load_file(tmp_path / r / "ckpt" / "60.safetensors") for r in "ab")
    assert max((wa[k] - wb[k]).abs().max() for k in wa) < 0.03


def test_resume_from_step0_state_drops_stale_files(sudoku_data, tmp_path):
    """A relaunch right after the step-0 save does not screen step 0 again, and checkpoints / screens of steps past the
    state (restored from S3 after a lost trajectory) are dropped."""
    run(sudoku_data, tmp_path, max_steps=0).run()
    os.remove(tmp_path / "DONE")
    shutil.copy(tmp_path / "ckpt" / "0.safetensors", tmp_path / "ckpt" / "50.safetensors")
    with open(tmp_path / "screen.jsonl", "a") as f:
        f.write(json.dumps({"step": 40, "score": 0.5}) + "\n")
    t = run(sudoku_data, tmp_path, max_steps=20)
    assert t.step == 0 and not (tmp_path / "ckpt" / "50.safetensors").exists()
    t.run()
    assert [json.loads(line)["step"] for line in open(tmp_path / "screen.jsonl")] == [0, 20]


def test_extend_finished_run(sudoku_data, tmp_path):
    """A finished run resumed with a larger max_steps trains on from its last step (screens and checkpoints continue)."""
    run(sudoku_data, tmp_path, max_steps=20).run()
    t = run(sudoku_data, tmp_path, max_steps=40)
    assert t.step == 20
    t.run()
    assert [json.loads(line)["step"] for line in open(tmp_path / "screen.jsonl")] == [0, 20, 40]
    assert json.load(open(tmp_path / "DONE")) == {"step": 40, "reason": "max_steps"}


def test_stop_rule(sudoku_data, tmp_path):
    """Stops once the median of the last 4 screen scores reaches stop_at (here at the 4th screen)."""
    t = run(sudoku_data, tmp_path, stop_at=0.0, screen_every=10)
    t.run()
    assert json.load(open(tmp_path / "DONE")) == {"step": 30, "reason": "stop_at"}
    assert os.path.exists(tmp_path / "ckpt" / "30.safetensors")


def test_val_every_logs_and_resumes(sudoku_data, tmp_path):
    """sft.val_every: the validation loss every that many steps in val.jsonl (step 0 included); a relaunch resumes the
    list from the last save, dropping the entries past it."""
    run(sudoku_data, tmp_path, max_steps=20, val_every=5).run()
    vals = [json.loads(line) for line in open(tmp_path / "val.jsonl")]
    assert [v["step"] for v in vals] == [0, 5, 10, 15, 20] and vals[-1]["val_loss"] < vals[0]["val_loss"]
    screens = {r["step"]: r["val_loss"] for r in map(json.loads, open(tmp_path / "screen.jsonl"))}
    assert all(abs(v["val_loss"] - screens[v["step"]]) < 1e-6 for v in vals if v["step"] in screens)
    t = run(sudoku_data, tmp_path, max_steps=30, val_every=5)
    t.run()
    assert [json.loads(line)["step"] for line in open(tmp_path / "val.jsonl")] == [0, 5, 10, 15, 20, 25, 30]
