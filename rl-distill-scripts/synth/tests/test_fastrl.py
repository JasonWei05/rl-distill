"""fastrl.py on a synth task: training, resume, outputs, the loss against a plain-Python reference of the math recipe's,
the update's invariance to micro-batches and GPUs."""

import argparse
import json
import math
import os
import sys

import pytest
import torch

import fastrl
from data import Dataset
from model import hf_config, init_weights, load_model, save_model


@pytest.fixture(scope="module")
def base(sudoku_data, tmp_path_factory):
    d = Dataset(sudoku_data)
    hf = hf_config("2XS", len(d.task.WORDS), d.width)
    path = str(tmp_path_factory.mktemp("base") / "p1")
    save_model(path, init_weights(hf, 0), hf, {"config": d.cfg, "size": "2XS", "seed": 0, "level": "p1"})
    return path


def args(base, data, out, steps, minibatches=1):
    return argparse.Namespace(base=base, data=data, out=str(out), batch=4, seed=3, steps=steps, minibatches=minibatches,
                              lr=None, patience=None, n=None, micro=None, decode_mem_gb=None, wandb_project="x", wandb_id="t")


def test_train_and_resume(base, sudoku_data, tmp_path):
    """6 steps with validations at 2, 4, 6; the same run stopped at 2 and resumed to 6: same validation steps, same
    step-0 validation, weights restored exactly at the resume point. A run resumed right after its step-0 validation does
    not validate step 0 again."""
    fastrl.Trainer(args(base, sudoku_data, tmp_path / "a", 6)).run()
    fastrl.Trainer(args(base, sudoku_data, tmp_path / "b", 2)).run()
    saved = torch.load(tmp_path / "b" / "ckpt.pt")
    tr = fastrl.Trainer(args(base, sudoku_data, tmp_path / "b", 6))
    assert tr.step == 2 and all(torch.equal(tr.params[k].detach(), saved["params"][k]) for k in tr.params)
    assert tr.opt.state_dict()["state"][0]["step"] == 2
    tr.run()
    for run in ("a", "b"):
        best = json.load(open(tmp_path / run / "best.json"))
        done = json.load(open(tmp_path / run / "COMPLETE.json"))
        assert [h[0] for h in best["history"]] == [0, 2, 4, 6] and done["reason"] == "max_steps" and done["last_step"] == 6
        _, synth, _ = load_model(str(tmp_path / run / "best_hf"))
        assert synth["rl"]["step"] == best["best_step"] and synth["level"] == "p1"
    assert json.load(open(tmp_path / "a" / "best.json"))["history"][0] == json.load(open(tmp_path / "b" / "best.json"))["history"][0]
    fastrl.Trainer(args(base, sudoku_data, tmp_path / "c", 0)).run()
    fastrl.Trainer(args(base, sudoku_data, tmp_path / "c", 2)).run()
    assert [h[0] for h in json.load(open(tmp_path / "c" / "best.json"))["history"]] == [0, 2]


def test_minibatches_train_and_resume_guard(base, sudoku_data, tmp_path):
    """2 mini-batches of 2 prompts: two optimizer steps per training step; resuming with another --minibatches fails."""
    tr = fastrl.Trainer(args(base, sudoku_data, tmp_path, 2, minibatches=2))
    tr.run()
    assert tr.opt.state_dict()["state"][0]["step"] == 4
    with pytest.raises(AssertionError, match="same --minibatches"):
        fastrl.Trainer(args(base, sudoku_data, tmp_path, 4))


def test_one_minibatch_gradient_is_tis():
    """With pi_old = pi (one mini-batch) the clipped ratio is 1 and the gradient is -A min(pi / pi_rollout, cap) grad
    log pi summed over tokens / `tokens`; with pi_old != pi the clip binds (A > 0 above 1.28, A < 0 below 0.8)."""
    torch.manual_seed(0)
    B, R = 64, 30
    mask = (torch.arange(R) < torch.randint(1, R + 1, (B, 1))).float()
    logp = (-4 * torch.rand(B, R)).requires_grad_()
    rollout = (logp.detach() + 0.6 * torch.randn(B, R)) * mask
    adv = torch.randn(B, 1) * mask
    loss, st = fastrl.ppo_tis_loss(logp, logp.detach(), rollout, adv, mask, 123.0, 2.0)
    (g,) = torch.autograd.grad(loss, logp)
    ref = -adv * (logp.detach() - rollout).clamp(-20, 20).exp().clamp(max=2.0) * mask / 123.0
    torch.testing.assert_close(g, ref)
    assert st[1] == 0 and st[2] == 0 and 0.05 < st[4] * 123 / mask.sum() < 0.5
    old = logp.detach() + 0.5 * torch.randn(B, R)
    _, st = fastrl.ppo_tis_loss(logp, old, rollout, adv, mask, mask.sum(), 2.0)
    assert 0.05 < st[1] < 0.5 and st[2] != 0


def reference_token(lp, old, rollout, a, cap):
    """The math recipe's loss on one token in plain Python (docs/RECIPES.md 1.1) and its derivative in log pi:
    w = min(exp(clamp(old - rollout)), cap); rho = exp(clamp(lp - old));
    l = w max(-a rho, -a clip(rho, 0.8, 1.28)) for a >= 0, w min(-10 a, max(...)) for a < 0."""
    rho = math.exp(min(max(lp - old, -20.0), 20.0))
    w = min(math.exp(min(max(old - rollout, -20.0), 20.0)), cap)
    unclipped, clipped = -a * rho, -a * min(max(rho, 0.8), 1.28)
    l, d = (unclipped, -a * rho) if unclipped >= clipped else (clipped, 0.0)  # d: derivative of the chosen term
    if lp - old != min(max(lp - old, -20.0), 20.0):
        d = 0.0  # the ratio clamp is flat
    if a < 0 and -10 * a < l:
        l, d = -10 * a, 0.0
    return w * l, w * d


def test_train_tflops_uses_head_dim():
    """Qwen3 configs set head_dim apart from hidden / heads (chess 50M: 128, not 64); the FLOPs count follows it."""
    c = {"hidden_size": 512, "num_attention_heads": 8, "num_key_value_heads": 4, "num_hidden_layers": 2,
         "intermediate_size": 1536, "vocab_size": 85}
    s = torch.tensor([100, 300])
    assert fastrl.train_tflops({**c, "head_dim": 64}, s) == fastrl.train_tflops(c, s) < fastrl.train_tflops({**c, "head_dim": 128}, s)


def test_micro_batches_hold_whole_prompts_within_the_full_length_budget():
    """At full-length responses the micro-batches are rl.micro rows (2 prompts of 4); short responses fill a mini-batch
    in one; a long prompt starts a new micro-batch once the context would pass rl.micro full-length rows."""
    tr = argparse.Namespace(n=4, micro=8)
    mb = lambda lens: [[(s.start, s.stop) for s in m] for m in fastrl.Trainer.micro_batches(tr, 16, lens, 10, 100)]
    assert mb([100] * 32) == [[(0, 8), (8, 16)], [(16, 24), (24, 32)]]
    assert mb([2] * 32) == [[(0, 16)], [(16, 32)]]
    assert mb([2] * 12 + [100] * 4 + [2] * 16) == [[(0, 12), (12, 16)], [(16, 32)]]  # 16 rows x 110 > 8 x 110


def test_ppo_tis_loss_matches_reference():
    """ppo_tis_loss and its gradient against reference_token on every token, with ratios on both sides of both clips,
    below the dual clip, TIS weights above the cap and log-ratios beyond the +-20 clamp; stats as token means."""
    torch.manual_seed(0)
    B, R, tokens = 24, 12, 77.0
    mask = (torch.arange(R) < torch.randint(1, R + 1, (B, 1))).double()
    logp = (-5 * torch.rand(B, R, dtype=torch.float64)).requires_grad_()
    old = logp.detach() + 0.3 * torch.randn(B, R, dtype=torch.float64)
    old[1, :5] = logp[1, :5].detach() - 3  # ratio e^3 > 10 under a negative advantage: the dual clip
    old[2, :3] = logp[2, :3].detach() + 25  # beyond the clamp
    rollout = old + 0.8 * torch.randn(B, R, dtype=torch.float64)
    rollout[0, :8] = old[0, :8] - 25
    adv = torch.randn(B, 1, dtype=torch.float64)
    adv[1] = -1.0
    adv = adv.expand(B, R) * mask
    loss, st = fastrl.ppo_tis_loss(logp, old, rollout, adv, mask, tokens, 2.0)
    (g,) = torch.autograd.grad(loss, logp)
    ref = [[reference_token(logp[i, t].item(), old[i, t].item(), rollout[i, t].item(), adv[i, t].item(), 2.0)
            if mask[i, t] else (0.0, 0.0) for t in range(R)] for i in range(B)]
    ref = torch.tensor(ref, dtype=torch.float64)
    torch.testing.assert_close(loss, ref[..., 0].sum() / tokens)
    torch.testing.assert_close(g, ref[..., 1] / tokens)
    rho = (logp.detach() - old).clamp(-20, 20).exp()
    per = lambda x: (x * mask).sum() / tokens
    torch.testing.assert_close(st[1], per(((rho < 0.8) & (adv < 0) | (rho > 1.28) & (adv > 0)).double()))
    torch.testing.assert_close(st[4], per(((old - rollout).clamp(-20, 20).exp() > 2.0).double()))
    assert st[1] > 0 and st[3] > 0 and st[4] > 0  # every branch was exercised


def test_early_stop(base, sudoku_data, tmp_path):
    """Validation scores 0.5, 0.5, 0.6, 0.6, 0.6: only a strictly greater score is a new best (step 4), and with patience
    2 (--patience) the run stops at the second validation after it without one (step 8)."""
    tr = fastrl.Trainer(argparse.Namespace(**{**vars(args(base, sudoku_data, tmp_path, 20)), "patience": 2}))
    scores = iter([0.5, 0.5, 0.6, 0.6, 0.6])
    tr.validate = lambda: {"val/score": next(scores)}
    tr.run()
    best = json.load(open(tmp_path / "best.json"))
    assert best["history"] == [[0, 0.5], [2, 0.5], [4, 0.6], [6, 0.6], [8, 0.6]] and best["best_step"] == 4
    assert json.load(open(tmp_path / "COMPLETE.json"))["reason"] == "early_stop"


def test_rejects_other_dataset(base, maze_data, tmp_path):
    with pytest.raises(AssertionError, match="another dataset config"):
        fastrl.Trainer(args(base, maze_data, tmp_path, 1))


def test_prompt_order_is_one_permutation_per_epoch():
    """Each epoch is a permutation; the order depends only on the seed."""
    o = fastrl.PromptOrder(100, 16, 3)
    epoch = [i for s in range(1, 7) for i in o(s)]
    assert len(set(epoch)) == 96
    assert list(fastrl.PromptOrder(100, 16, 3)(7)) == list(o(7))


def dp_update(a):
    """One fastrl training step's update (a.minibatches optimizer steps, a.micro-row micro-batches) from the base on a
    fixed random batch of 8 prompts x 16 samples, this rank's share of it (the prompts of each mini-batch split over the
    ranks, as train_step does); rank 0 saves AdamW's first moment (a fixed linear combination of the steps' clipped
    gradients) and the metrics to a.out."""
    import torch.distributed as dist
    from tasks import EOS
    tr = object.__new__(fastrl.Trainer)
    tr.a, tr.rank, tr.world = a, dist.get_rank(), dist.get_world_size()
    tr.hf, tr.synth, w = load_model(a.base)
    tr.data = Dataset(a.data)
    tr.r, tr.micro = tr.data.cfg["rl"], a.micro
    tr.n = tr.r["n"]
    tr.model = fastrl.Qwen2(tr.hf)
    tr.params = {k: v.requires_grad_() for k, v in w.items()}
    tr.opt = torch.optim.AdamW(tr.params.values(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    tr.scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 16, growth_interval=400)
    if a.dtype == "float32":
        tr.weights16 = lambda: dict(tr.params)  # fp32 compute
    g = torch.Generator().manual_seed(0)
    G, n, R = 8, tr.r["n"], 40
    prompts = tr.data.prompts("dev", G)
    tok = torch.randint(len(tr.data.task.WORDS), (G * n, R), generator=g)
    lens = torch.randint(2, R, (G * n,), generator=g)
    tok[torch.arange(G * n), lens] = EOS
    mask = (torch.arange(R) <= lens[:, None])
    old = -3 * torch.rand(G * n, R, generator=g) * mask
    adv = torch.randn(G * n, 1, generator=g) * mask
    mine = torch.arange(G).reshape(a.minibatches, tr.world, -1)[:, tr.rank].reshape(-1)
    rows = (mine[:, None] * n + torch.arange(n)).reshape(-1)
    m = tr.update(prompts[mine].cuda(), tok[rows].cuda(), old[rows].cuda(), adv[rows].cuda(), mask[rows].cuda())
    assert tr.opt.state_dict()["state"][0]["step"] == a.minibatches
    if tr.rank == 0:
        torch.save({"moment": {k_: tr.opt.state[p]["exp_avg"] for k_, p in tr.params.items()}, "metrics": m}, a.out)


@pytest.mark.parametrize("minibatches", [1, 2])
@pytest.mark.parametrize("dtype,tol", [("float32", 1e-5), ("float16", 3e-3)])
def test_update_does_not_depend_on_micro_batches_or_gpus(base, sudoku_data, tmp_path, dtype, tol, minibatches):
    """The gradients of one training step (1 or 2 mini-batches of the 8 prompts x 16 samples; read back from AdamW's first
    moment) are the same with micro-batches of 32 or 64 samples and on 1 or 2 ranks: each mini-batch's loss is its token
    mean however it is split. Compared: the first moment and the loss stats; fp32 compute up to float rounding, fp16 up to
    fp16 rounding (the split changes which partial sums round). Needs 2 visible GPUs."""
    assert torch.cuda.device_count() >= 2, "needs 2 GPUs"
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    out = []
    for gpus, micro in ((1, 32), (1, 64), (2, 32)):
        a = argparse.Namespace(base=base, data=sudoku_data, out=str(tmp_path / f"w{gpus}_{micro}.pt"), batch=8, gpus=gpus,
                               dtype=dtype, micro=micro, minibatches=minibatches)
        fastrl.launch(a, dp_update)
        out.append(torch.load(a.out))
    flat = lambda o: torch.cat([o["moment"][k].flatten() for k in sorted(o["moment"])])
    for o in out[1:]:
        rel = ((flat(out[0]) - flat(o)).norm() / flat(out[0]).norm()).item()
        print(f"{dtype} minibatches {minibatches}: gradient vs 1 rank, micro 32: {rel:.2e}")
        assert rel < tol
        for key in ("actor/pg_loss", "actor/grad_norm", "actor/ppo_kl", "actor/pg_clipfrac"):
            assert abs(out[0]["metrics"][key] - o["metrics"][key]) <= tol * max(1.0, abs(out[0]["metrics"][key])), key
    if minibatches == 2:  # the second mini-batch is off-policy
        assert out[0]["metrics"]["actor/ppo_kl"] != 0


def test_two_rank_train_and_resume(base, sudoku_data, tmp_path):
    """A 2-rank run of 4 steps then resumed to 6 (validations every 2): both ranks take every collective path (validate,
    checkpoint, resume); rank 0 writes the files; the checkpoint holds each rank's RNG. Needs 2 visible GPUs."""
    assert torch.cuda.device_count() >= 2, "needs 2 GPUs"
    a = lambda steps: argparse.Namespace(**{**vars(args(base, sudoku_data, tmp_path, steps)), "gpus": 2})
    fastrl.launch(a(4), fastrl.train)
    assert len(torch.load(tmp_path / "ckpt.pt", map_location="cpu")["rng"]) == 2
    fastrl.launch(a(6), fastrl.train)
    best = json.load(open(tmp_path / "best.json"))
    assert [h[0] for h in best["history"]] == [0, 2, 4, 6]
    assert json.load(open(tmp_path / "COMPLETE.json"))["last_step"] == 6
