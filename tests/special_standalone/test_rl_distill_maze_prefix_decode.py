"""GPU tests for the maze shared-prompt CUDA-graph decoder (TailRL rollout/prefix_decode.py, from maze/tailrl_rl_tis.patch).

PrefixSharedDecoder must: give identical tokens and logits with and without CUDA graphs (same seed), with and without
finished-row compaction, and in the validation layout (repeated rows, uneven groups) vs the train layout; score its own
samples like a teacher-forced HF forward (fp16 level); handle every row finishing at once, segment edges, many eos ids
(no capture-breaking host sync), want_logits=False and temperature; and not leak memory across calls. A tiny random
fp16 Qwen2 stands in for the maze models. Needs a CUDA GPU (~1 GB) and a patched TailRL checkout (TAILRL_ROOT):

    TAILRL_ROOT=/tmp/jw_maze/TailRL CUDA_VISIBLE_DEVICES=0 /mnt/efs/jasonwei/rl-distill/.venv/bin/python -m pytest -q \\
        tests/special_standalone/test_rl_distill_maze_prefix_decode.py
"""

import os
import sys

import pytest
import torch

ROOTS = [os.environ.get("TAILRL_ROOT", ""), "/tmp/maze/TailRL", "/tmp/jw_maze/TailRL"]
ROLLOUT = next((os.path.join(r, "experiments/text_maze/verl/workers/rollout") for r in ROOTS
                if r and os.path.exists(os.path.join(r, "experiments/text_maze/verl/workers/rollout/prefix_decode.py"))), None)
pytestmark = pytest.mark.skipif(ROLLOUT is None or not torch.cuda.is_available(), reason="needs a GPU and TAILRL_ROOT")
if ROLLOUT:
    sys.path.insert(0, ROLLOUT)
    import prefix_decode
    from prefix_decode import PrefixSharedDecoder

V, L, EOS, PAD = 32, 40, [2, 7], 0


@pytest.fixture(scope="module")
def model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=V, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      intermediate_size=128, max_position_embeddings=512, tie_word_embeddings=True)
    return Qwen2ForCausalLM(cfg).half().cuda().eval()


def prompts(n, seed=1):
    return torch.randint(3, V, (n, L), generator=torch.Generator().manual_seed(seed)).cuda()


class Greedy(PrefixSharedDecoder):
    """Deterministic sampling (argmax) so layouts can be compared token by token."""

    @staticmethod
    def _sample(logits, T):
        return torch.argmax(logits, -1)


def gen(dec, x, n=16, eos=EOS, seed=0, want_logits=True, T=1.0):
    torch.manual_seed(seed)
    out = dec.generate(x, T, eos, PAD, n, want_logits=want_logits)
    return out.sequences, (torch.stack(out.logits, 1) if out.logits is not None else None)


def alive(resp, eos=EOS):
    ended = torch.cumsum(torch.isin(resp, torch.tensor(eos, device=resp.device)).long(), 1)
    m = torch.ones_like(resp, dtype=torch.bool)
    m[:, 1:] = ended[:, :-1] == 0
    return m


def test_graphs_equal_eager(model):
    x = prompts(4)
    s1, l1 = gen(PrefixSharedDecoder(model, 80), x)
    s2, l2 = gen(PrefixSharedDecoder(model, 80, use_graphs=False), x)
    assert torch.equal(s1, s2) and torch.equal(l1, l2)


def test_compaction_matches_no_compaction(model, monkeypatch):
    x = prompts(4)
    s1, l1 = gen(Greedy(model, 96), x)
    monkeypatch.setattr(prefix_decode, "COMPACT_KEEP_FRAC", None)
    s2, l2 = gen(Greedy(model, 96), x)
    mk = alive(s1[:, L:])
    assert torch.equal(s1, s2)
    assert (l1 - l2).abs()[mk].max() < 1e-2


def test_validation_layout_matches_train_layout(model):
    x = prompts(5)
    s_tr, l_tr = gen(Greedy(model, 64), x)  # 5 prompts x 16
    rows = torch.arange(8, 72)  # a chunk that splits the first and last groups: 8 / 16 / 16 / 16 / 8
    s_va, l_va = gen(Greedy(model, 64), x.repeat_interleave(16, 0)[rows], n=1)
    n = min(s_tr.shape[1], s_va.shape[1])
    assert torch.equal(s_va[:, :n], s_tr[rows][:, :n])
    mk = alive(s_va[:, L:n])
    assert (l_va[:, :n - L] - l_tr[rows][:, :n - L]).abs()[mk].max() < 1e-2


def test_scores_like_teacher_forced_hf(model):
    s, lg = gen(PrefixSharedDecoder(model, 80), prompts(4))
    resp = s[:, L:]
    with torch.no_grad():
        ref = model.float()(input_ids=s).logits[:, L - 1:L - 1 + resp.shape[1]]
        model.half()
    lp = torch.log_softmax(lg, -1).gather(-1, resp[..., None])[..., 0]
    lr = torch.log_softmax(ref, -1).gather(-1, resp[..., None])[..., 0]
    assert (lp - lr)[alive(resp)].abs().mean() < 3e-3


def test_every_row_finishes_at_once(model):
    class AllEos(PrefixSharedDecoder):
        @staticmethod
        def _sample(logits, T):
            return torch.full(logits.shape[:1], EOS[0], dtype=torch.long, device=logits.device)
    s, lg = gen(AllEos(model, 80), prompts(2))
    assert s.shape[1] == L + 8 and lg.shape[1] == 8  # stops at the first all-finished check
    assert (s[:, L] == EOS[0]).all() and (s[:, L + 1:] == PAD).all()
    s1, _ = gen(AllEos(model, 80), prompts(1), n=1)
    assert s1.shape == (1, L + 8)


@pytest.mark.parametrize("max_new", [1, 3, 33, 34, 35])
def test_segment_edges(model, max_new):
    x = prompts(2)
    s1, l1 = gen(PrefixSharedDecoder(model, max_new), x, eos=[V + 5])  # no eos: every row runs to max_new
    s2, l2 = gen(PrefixSharedDecoder(model, max_new, use_graphs=False), x, eos=[V + 5])
    assert s1.shape[1] == L + max_new and l1.shape[1] == max_new
    assert torch.equal(s1, s2) and torch.equal(l1, l2)


def test_many_eos_ids_stay_capturable(model):
    s, _ = gen(PrefixSharedDecoder(model, 64), prompts(1), eos=list(range(3, 43)))  # 40 eos ids
    assert s.shape[0] == 16
    torch.empty(4, device="cuda").exponential_()  # the CUDA generator is not left in capture mode


def test_want_logits_false_samples_the_same(model):
    x = prompts(3)
    s1, _ = gen(PrefixSharedDecoder(model, 64), x)
    s2, lg = gen(PrefixSharedDecoder(model, 64), x, want_logits=False)
    assert lg is None and torch.equal(s1, s2)


def test_temperature(model):
    x = prompts(1, seed=7)
    dec = PrefixSharedDecoder(model, 1)
    with torch.no_grad():
        p = torch.softmax(model(input_ids=x).logits[0, -1].float() / 2.0, -1)
    s, _ = gen(dec, x, n=8192, eos=[V + 5], T=2.0)
    freq = torch.bincount(s[:, L], minlength=V).float() / s.shape[0]
    z = (freq - p) / (p * (1 - p) / s.shape[0]).sqrt().clamp(min=1e-6)
    assert z.abs().max() < 4.5


def test_no_memory_growth(model):
    dec, x = PrefixSharedDecoder(model, 64), prompts(4)
    gen(dec, x)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for seed in range(3):
        gen(dec, x, seed=seed)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() - before < 2**20
