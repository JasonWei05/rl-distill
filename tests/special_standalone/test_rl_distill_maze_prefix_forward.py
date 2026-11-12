"""CPU tests for the maze shared-prompt training forward (TailRL actor/prefix_forward.py, from maze/tailrl_rl_tis.patch).

prefix_shared_logits must give the padded forward's response logits and parameter gradients (fp32: float noise), for any
row order, response lengths 0 / 1 / full, one group, all-distinct prompts, uneven left padding, identical prompt tokens
under different masks, and under 2-rank FSDP FULL_SHARD with accumulated micro-batches. A tiny random Qwen2 stands in
for the maze models. Needs a TailRL checkout with the patch applied (TAILRL_ROOT, default /tmp/maze/TailRL or
/tmp/jw_maze/TailRL) and its venv (transformers >= 4.56):

    TAILRL_ROOT=/tmp/jw_maze/TailRL /tmp/jw_maze/TailRL/.venv/bin/python -m pytest -q \\
        tests/special_standalone/test_rl_distill_maze_prefix_forward.py
"""

import functools
import os
import socket
import sys

import pytest
import torch

ROOTS = [os.environ.get("TAILRL_ROOT", ""), "/tmp/maze/TailRL", "/tmp/jw_maze/TailRL"]
ACTOR = next((os.path.join(r, "experiments/text_maze/verl/workers/actor") for r in ROOTS
              if r and os.path.exists(os.path.join(r, "experiments/text_maze/verl/workers/actor/prefix_forward.py"))), None)
pytestmark = pytest.mark.skipif(ACTOR is None, reason="needs a patched TailRL checkout (TAILRL_ROOT)")
if ACTOR:
    sys.path.insert(0, ACTOR)
    import prefix_forward
    from prefix_forward import prefix_shared_logits

PAD, V, P, R = 0, 32, 40, 16


def tiny_model(seed=0):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=V, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      intermediate_size=128, max_position_embeddings=128, tie_word_embeddings=True)
    return Qwen2ForCausalLM(cfg).float().train()


def make_batch(n_prompts, n_per, resp_lens, pads=None, seed=1):
    """Left-padded prompts (n_per rows each), right-padded responses with the given real lengths, rows shuffled."""
    g = torch.Generator().manual_seed(seed)
    pads = pads if pads is not None else [4] * n_prompts
    prompts = torch.stack([torch.cat([torch.full((p,), PAD), torch.randint(3, V, (P - p,), generator=g)]) for p in pads])
    pmask = torch.stack([torch.cat([torch.zeros(p, dtype=torch.long), torch.ones(P - p, dtype=torch.long)]) for p in pads])
    B = n_prompts * n_per
    rmask = (torch.arange(R)[None] < torch.tensor(resp_lens)[:, None]).long()
    resp = torch.where(rmask.bool(), torch.randint(3, V, (B, R), generator=g), torch.full((B, R), PAD))
    ids = torch.cat([prompts.repeat_interleave(n_per, 0), resp], 1)
    attn = torch.cat([pmask.repeat_interleave(n_per, 0), rmask], 1)
    pos = torch.clamp(torch.cumsum(attn, 1) - 1, min=0)
    perm = torch.randperm(B, generator=g)
    return ids[perm], attn[perm], pos[perm], resp[perm], rmask[perm]


def logp_and_grads(model, fwd, ids, attn, pos, resp, rmask, adv, micro_batches=1):
    model.zero_grad(set_to_none=True)
    for _ in range(micro_batches):
        if fwd == "standard":
            lg, n = model(input_ids=ids, attention_mask=attn, position_ids=pos, use_cache=False).logits[:, -R - 1:-1], R
        else:
            lg, n = prefix_shared_logits(model, ids, attn, pos, R)
        lp = torch.log_softmax(lg.float(), -1).gather(-1, resp[:, :n, None])[..., 0]
        lp = torch.cat([lp, lp.new_zeros(lp.shape[0], R - n)], 1)
        (-(adv * lp * rmask).sum() / rmask.sum().clamp(min=1)).backward()
    return lp.detach(), {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}, n


def assert_equivalent(model, batch, adv):
    ids, attn, pos, resp, rmask = batch
    ls, gs, _ = logp_and_grads(model, "standard", *batch, adv)
    lh, gh, n = logp_and_grads(model, "shared", *batch, adv)
    assert n == max(1, int(rmask.sum(1).max()))
    mk = rmask.bool()
    if mk.any():
        assert (ls - lh)[mk].abs().max() < 2e-5
    assert set(gs) == set(gh)
    for k in gs:
        assert (gs[k] - gh[k]).norm() <= 1e-4 * gs[k].norm() + 1e-9, k


CASES = {
    "mixed lengths, 4 prompts x 8": dict(n_prompts=4, n_per=8, resp_lens=[3, 16, 1, 9, 12, 5, 7, 2] * 4),
    "all length 0": dict(n_prompts=2, n_per=4, resp_lens=[0] * 8),
    "all length 1": dict(n_prompts=2, n_per=4, resp_lens=[1] * 8),
    "lengths 0 / 1 / 2": dict(n_prompts=2, n_per=4, resp_lens=[0, 1, 2, 1, 0, 2, 2, 1]),
    "one group": dict(n_prompts=1, n_per=8, resp_lens=[4, 16, 8, 1, 3, 11, 6, 2]),
    "every row distinct": dict(n_prompts=8, n_per=1, resp_lens=[4, 16, 8, 1, 3, 11, 6, 2]),
    "uneven left padding": dict(n_prompts=3, n_per=4, resp_lens=[5, 9, 2, 16] * 3, pads=[0, 3, 9]),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_matches_padded_forward(case):
    model = tiny_model()
    batch = make_batch(**CASES[case])
    assert_equivalent(model, batch, torch.randn(batch[0].shape[0], 1, generator=torch.Generator().manual_seed(3)))


def test_identical_tokens_different_masks_get_separate_caches():
    """A real leading token equal to the pad id: same prompt tokens, different masks -> must not share a cache."""
    model = tiny_model()
    real = torch.randint(3, V, (P - 1,), generator=torch.Generator().manual_seed(4))
    prompts = torch.stack([torch.cat([torch.tensor([PAD]), real])] * 2)
    pmask = torch.tensor([[0] + [1] * (P - 1), [1] * P])
    resp, rmask = torch.randint(3, V, (2, R), generator=torch.Generator().manual_seed(5)), torch.ones(2, R, dtype=torch.long)
    attn = torch.cat([pmask, rmask], 1)
    batch = (torch.cat([prompts, resp], 1), attn, torch.clamp(torch.cumsum(attn, 1) - 1, min=0), resp, rmask)
    assert_equivalent(model, batch, torch.ones(2, 1))


def test_wrong_grouping_is_detected(monkeypatch):
    """Sensitivity check: pairing rows with another group's prompt cache must change the log-probs a lot."""
    model = tiny_model()
    ids, attn, pos, resp, rmask = make_batch(**CASES["mixed lengths, 4 prompts x 8"])
    with torch.no_grad():
        good, n = prefix_shared_logits(model, ids, attn, pos, R)
        unique = torch.unique

        def rotated(x, **kw):
            u, inv = unique(x, **kw)
            return u, (inv + 1) % u.shape[0]
        monkeypatch.setattr(prefix_forward.torch, "unique", rotated)
        bad, _ = prefix_shared_logits(model, ids, attn, pos, R)
    gp = torch.log_softmax(good, -1).gather(-1, resp[:, :n, None])[..., 0]
    bp = torch.log_softmax(bad, -1).gather(-1, resp[:, :n, None])[..., 0]
    assert (gp - bp)[rmask[:, :n].bool()].abs().mean() > 0.05


def test_needs_kv_cache_under_gradient_checkpointing():
    model = tiny_model()
    model.gradient_checkpointing_enable()
    ids, attn, pos, _, _ = make_batch(**CASES["one group"])
    with pytest.raises(AssertionError, match="gradient checkpointing"):
        prefix_shared_logits(model, ids, attn, pos, R)


def _fsdp_worker(rank, world, port, use_orig_params, out):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    sys.path.insert(0, ACTOR)
    from prefix_forward import prefix_shared_logits as shared  # noqa: F401 (re-import in the spawned process)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    model = FSDP(tiny_model(), sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=use_orig_params,
                 auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen2DecoderLayer}),
                 device_id=torch.device("cpu"))
    batch = make_batch(n_prompts=2, n_per=4, resp_lens=[3, 16, 1, 9, 12, 5, 7, 2], seed=100 + rank)
    adv = torch.randn(8, 1, generator=torch.Generator().manual_seed(rank))
    _, gs, _ = logp_and_grads(model, "standard", *batch, adv, micro_batches=2)
    _, gh, _ = logp_and_grads(model, "shared", *batch, adv, micro_batches=2)
    out[rank] = max(((gs[k] - gh[k]).norm() / gs[k].norm().clamp(min=1e-12)).item() for k in gs) if set(gs) == set(gh) else 1.0
    dist.destroy_process_group()


@pytest.mark.parametrize("use_orig_params", [False, True])
def test_fsdp_full_shard_two_forwards(use_orig_params):
    """Two root forwards per micro-batch under FULL_SHARD, two micro-batches accumulated: same sharded gradients."""
    import torch.multiprocessing as mp
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    out = mp.Manager().dict()
    mp.spawn(_fsdp_worker, args=(2, port, use_orig_params, out), nprocs=2, join=True)
    assert max(out.values()) < 1e-5, dict(out)
