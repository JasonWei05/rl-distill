"""fastrl.py against the verl stack it replaces (TailRL experiments/text_maze + tailrl_rl_tis.patch).

Each test runs a fastrl piece and the verl / TailRL code it reproduces on the same inputs: prompt tokenization, maze
parsing and BFS, the shortest-path judge, the GRPO advantage, the TIS loss, the prompt order, the FLOPs formula, the policy
forward (against HF Qwen2 and verl's prefix-shared forward), the sampler (decode log-probs against the teacher-forced
forward, token frequencies, and validation statistics against verl's PrefixSharedDecoder), one full update against verl's
DataParallelPPOActor on FSDP (fp32: equal up to float rounding; fp16 mixed precision: close), and a train / resume smoke run.

    cd rl-distill-scripts/maze/fastrl
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:$TAILRL/experiments/text_maze:../rl $TAILRL/.venv/bin/python -m pytest -q test_fastrl.py
env: FASTRL_BASE (a 2XS base, HF dir), FASTRL_DATA (dir with eval500/ and main_parquet/), FASTRL_BASE_HI (a 10% base)
"""

import argparse
import os

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import fastrl as fr

BASE = os.environ.get("FASTRL_BASE", "/tmp/fastrl/bases/x2_p3_s3")
BASE_HI = os.environ.get("FASTRL_BASE_HI", "/tmp/fastrl/bases/x2_p10_s3")
DATA = os.environ.get("FASTRL_DATA", "/tmp/jw_maze/dfs25_eps05")
VAL, TRAIN = f"{DATA}/eval500/test.parquet", f"{DATA}/main_parquet/train.parquet"
N = 16


@pytest.fixture(scope="module")
def vocab():
    return fr.Vocab(BASE)


@pytest.fixture(scope="module")
def model():
    return fr.Qwen2(BASE)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(BASE)


@pytest.fixture(scope="module")
def val(vocab):
    return torch.from_numpy(np.array(fr.load_prompts(VAL, vocab))).cuda().long()


@pytest.fixture(scope="module")
def val_rows():
    return pq.read_table(VAL).to_pylist()


def master(base):
    return {k: v.float().cuda() for k, v in load_file(os.path.join(base, "model.safetensors")).items()}


@pytest.fixture(scope="module")
def rollouts(model, vocab, val):
    """N samples per validation prompt from the 3% and the 10% base (fp16 decode): tokens, log-probs, entropies, masks."""
    out = {}
    for name, base in (("lo", BASE), ("hi", BASE_HI)):
        w = {k: v.half() for k, v in master(base).items()}
        torch.manual_seed(0)
        tok, logp, ent = fr.Decoder(model, vocab, 300).generate(w, val, N)
        out[name] = (w, tok, logp, ent, fr.response_mask(tok, vocab.eos))
    return out


def test_prompt_tokens_match_hf_tokenizer(vocab, tokenizer, val_rows):
    """verl RLHFDataset: tokenizer("".join(contents), add_special_tokens=False) (apply_chat_template=False)."""
    toks = fr.load_prompts(VAL, vocab)
    for i, r in enumerate(val_rows):
        assert tokenizer(r["prompt"][0]["content"], add_special_tokens=False)["input_ids"] == toks[i].tolist()
    train, pf = fr.load_prompts(TRAIN, vocab), pq.ParquetFile(TRAIN)
    sizes = [pf.metadata.row_group(j).num_rows for j in range(pf.num_row_groups)]
    for g in (0, pf.num_row_groups // 2, pf.num_row_groups - 1):
        rows = pf.read_row_group(g, columns=["prompt"]).to_pylist()
        for j in range(0, len(rows), 97):
            assert tokenizer(rows[j]["prompt"][0]["content"], add_special_tokens=False)["input_ids"] == train[sum(sizes[:g]) + j].tolist()


def test_mazes_match_tailrl(vocab, val, val_rows):
    """Grid / start / goal against MazeEnv.from_sequence(ground_truth); BFS length against _bfs_optimal_length and the
    data's optimal_path_length, on the validation set and 4000 train rows."""
    from verl.utils.reward_score.maze import MazeEnv, _bfs_optimal_length
    pf = pq.ParquetFile(TRAIN)
    train_rows = pf.read_row_group(pf.num_row_groups // 3, columns=["reward_model", "extra_info"]).to_pylist()[:4000]
    train_ids = np.array(fr.load_prompts(TRAIN, vocab)[sum(pf.metadata.row_group(j).num_rows for j in range(pf.num_row_groups // 3)):][:4000])
    for rows, prompts in ((val_rows, val), (train_rows, torch.from_numpy(train_ids).cuda().long())):
        mz = fr.Mazes(prompts, vocab)
        free, start, goal, lstar = mz.free.cpu().numpy(), mz.start.tolist(), mz.goal.tolist(), mz.lstar.tolist()
        for i, r in enumerate(rows):
            env = MazeEnv.from_sequence(r["reward_model"]["ground_truth"])
            assert np.array_equal(env.grid == 0, free[i])
            assert env.start == tuple(start[i]) and env.goal == tuple(goal[i])
            assert _bfs_optimal_length(env.grid, env.start, env.goal) == lstar[i] == r["extra_info"]["optimal_path_length"]


def synthetic_responses(vocab, val_rows, R=300):
    """Hand-made and random responses per validation maze (rows of maze i at i), EOS-terminated and pad-filled as the
    decoder returns them: the optimal path, variations of it (short, overshooting, detours, extra / special / newline /
    unknown words, no DONE, words after DONE), random move strings and random token strings."""
    v, rng = vocab.ids, np.random.default_rng(0)
    moves = [v[a] for a in ("UP", "DOWN", "LEFT", "RIGHT")]
    undo = {v["UP"]: v["DOWN"], v["DOWN"]: v["UP"], v["LEFT"]: v["RIGHT"], v["RIGHT"]: v["LEFT"]}
    DONE, EOS = v["DONE"], v["<eos>"]
    rows, maze = [], []
    for i, r in enumerate(val_rows):
        path = r["extra_info"]["answer"].split()
        opt = [v[w] for w in path[path.index("PATH_START") + 1:path.index("DONE")]]
        k = int(rng.integers(1, len(opt)))
        m = moves[int(rng.integers(4))]
        cases = [opt + [DONE, EOS], opt[:-1] + [DONE, EOS], opt + [m, DONE, EOS], opt[:k] + [m, undo[m]] + opt[k:] + [DONE, EOS],
                 opt[:k] + [v["\n"]] + opt[k:] + [DONE, EOS], opt[:k] + [v["<bos>"], v["<unk>"]] + opt[k:] + [DONE, EOS],
                 opt[:k] + [v["RESERVED_0"]] + opt[k:] + [DONE, EOS], opt[:k] + [v["PATH"]] + opt[k:] + [DONE, EOS],
                 opt + [EOS], opt + [DONE, m, v["WALL"], EOS], [DONE, EOS], [EOS], opt + [DONE, DONE, EOS],
                 opt[:k] + [DONE] + opt[k:] + [EOS], opt[:k] + [EOS] + opt[k:] + [DONE, EOS]]
        cases += [list(rng.choice(moves, int(rng.integers(1, 260)))) + [DONE, EOS] for _ in range(6)]
        cases += [list(rng.integers(0, len(v), int(rng.integers(1, R + 1)))) for _ in range(6)]
        cases += [list(rng.choice(moves + [v["\n"], DONE], int(rng.integers(1, R)))) + [EOS] for _ in range(3)]
        rows += cases
        maze += [i] * len(cases)
    tok = torch.full((len(rows), R), v["<pad>"], dtype=torch.long)
    for j, c in enumerate(rows):
        c = c[:R]
        tok[j, :len(c)] = torch.tensor(c)
    tok = torch.where(fr.response_mask(tok, EOS), tok, v["<pad>"])  # nothing follows EOS
    return tok.cuda(), torch.tensor(maze).cuda()


def test_judge_matches_tailrl(vocab, tokenizer, val, val_rows, rollouts):
    """Every outcome field against maze_shortest_reward.compute_score(decode(response, skip_special_tokens=True)), as
    verl's PrimeRewardManager calls it, on 16000 real samples and 13500 synthetic responses."""
    from maze_shortest_reward import compute_score
    syn, syn_maze = synthetic_responses(vocab, val_rows)
    sets = [(rollouts[k][1], torch.arange(len(val), device="cuda").repeat_interleave(N)) for k in ("lo", "hi")] + [(syn, syn_maze)]
    mz = fr.Mazes(val, vocab)
    for tok, mi in sets:
        sub = object.__new__(fr.Mazes)
        sub.S, sub.free, sub.start, sub.goal, sub.lstar = mz.S, mz.free[mi], mz.start[mi], mz.goal[mi], mz.lstar[mi]
        mine = {k: x.tolist() for k, x in fr.judge(tok, sub, vocab).items()}
        texts = tokenizer.batch_decode(tok.tolist(), skip_special_tokens=True)
        for j, (s, i) in enumerate(zip(texts, mi.tolist())):
            ref = compute_score("maze_17", s, val_rows[i]["reward_model"]["ground_truth"])
            for key in ("score",) + fr.OUTCOMES:
                assert mine[key][j] == ref[key], (key, j, s, ref, {k: mine[k][j] for k in mine})
    assert sum(fr.judge(rollouts["hi"][1], mz.repeat(N), vocab)["score"].tolist()) > 500  # the 10% base scores


def test_grpo_advantage_matches_verl():
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    torch.manual_seed(0)
    G, R = 64, 40
    p = torch.rand(G).repeat_interleave(N)
    p[:N], p[N:2 * N] = 0.0, 1.0  # all-0 and all-1 groups
    scores = (torch.rand(G * N) < p).float()
    lens = torch.randint(1, R + 1, (G * N,))
    mask = (torch.arange(R) < lens[:, None]).float()
    rewards = torch.zeros(G * N, R)
    rewards[torch.arange(G * N), lens - 1] = scores
    ref, _ = compute_grpo_outcome_advantage(rewards, mask, np.array([f"uid{i // N}" for i in range(G * N)], dtype=object))
    torch.testing.assert_close(fr.grpo_advantage(scores, N)[:, None] * mask, ref, rtol=1e-6, atol=1e-6)


def test_tis_loss_matches_verl():
    from omegaconf import OmegaConf
    from verl.trainer.ppo.core_algos import compute_policy_loss_tis
    torch.manual_seed(0)
    B, R = 256, 50
    mask = (torch.arange(R) < torch.randint(1, R + 1, (B, 1))).float()
    logp = (-5 * torch.rand(B, R)).requires_grad_()
    old = logp.detach() + 0.8 * torch.randn(B, R)  # ratios well above 2 and below 1/2
    adv = torch.randn(B, 1) * mask
    ref = compute_policy_loss_tis(old, logp, adv, mask, "token-mean", OmegaConf.create({"policy_loss": {"tis_cap": 2.0}}))
    (g_ref,) = torch.autograd.grad(ref[0], logp)
    loss, stats = fr.tis_loss(logp, old, adv, mask, 2.0)
    (g,) = torch.autograd.grad(loss, logp)
    torch.testing.assert_close(stats, torch.stack([x.detach() for x in ref]))
    torch.testing.assert_close(g, g_ref)
    assert 0.05 < stats[1] < 0.5 and 0.05 < stats[3] < 0.5


@pytest.mark.parametrize("n,batch,workers", [(1000, 64, 0), (1000, 64, 2), (1300000, 256, 0)])
def test_prompt_order_matches_verl(n, batch, workers):
    """verl main_ppo.create_rl_sampler + the trainer's StatefulDataLoader (drop_last), over several epochs."""
    from omegaconf import OmegaConf
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.trainer.main_ppo import create_rl_sampler
    ds = np.arange(n)
    sampler = create_rl_sampler(OmegaConf.create({"shuffle": True, "seed": 3}), ds)
    loader = StatefulDataLoader(dataset=ds, batch_size=batch, num_workers=workers, drop_last=True, sampler=sampler)
    order, step = fr.PromptOrder(n, batch, 3), 0
    for _ in range(3 if n < 10000 else 1):
        for i, b in enumerate(loader):
            step += 1
            assert np.array_equal(b.numpy(), order(step))
            if i == 40:
                break


def test_flops_match_verl_ledger():
    from verl.utils.flops_ledger import FlopsLedger
    led = FlopsLedger.from_model_path(BASE)
    seqlens = torch.randint(655, 955, (4096,))
    import json
    assert abs(fr.train_tflops(json.load(open(f"{BASE}/config.json")), seqlens) - led.train_tflops(seqlens.tolist())) < 1e-9


def full_sequences(val, tok, mask, groups):
    G = len(groups)
    rows = (groups[:, None] * N + torch.arange(N, device="cuda")).flatten()
    r = max(int(mask[rows].sum(1).max()), 2)
    return val[groups], tok[rows, :r], mask[rows, :r]


def hf_logprobs(hf, ids, Lp, rows=16):
    """log-probs of the tokens after the first Lp of each row under the HF model, rows at a time."""
    return torch.cat([hf(input_ids=x).logits[:, Lp - 1:-1].log_softmax(-1).gather(-1, x[:, Lp:, None]).squeeze(-1)
                      for x in ids.split(rows)])


def test_forward_matches_hf_and_verl(model, val, rollouts):
    """Log-probs against HF Qwen2ForCausalLM in float64. fp32: about as close as HF's own fp32 forward. fp16 (verl's mixed
    precision): about as close as verl's prefix_shared_logits on the fp16 HF module under fp16 autocast, with verl's left
    padding and position ids."""
    from verl.utils.torch_functional import logprobs_from_logits
    from verl.workers.actor.prefix_forward import prefix_shared_logits
    w16, tok, _, _, mask = rollouts["hi"]
    prompts, resp, mk = full_sequences(val, tok, mask, torch.arange(0, 500, 17, device="cuda"))
    Lp, r = prompts.shape[1], resp.shape[1]
    hf = AutoModelForCausalLM.from_pretrained(BASE_HI, torch_dtype=torch.float32, attn_implementation="sdpa").cuda().eval()
    hf64 = AutoModelForCausalLM.from_pretrained(BASE_HI, torch_dtype=torch.float64, attn_implementation="eager").cuda().eval()
    with torch.no_grad():
        mine32 = model.train_logprobs(master(BASE_HI), prompts, resp, N)
        ids = torch.cat([prompts.repeat_interleave(N, 0), resp], 1)
        ref32, ref64 = hf_logprobs(hf, ids, Lp), hf_logprobs(hf64, ids, Lp)
    err32, err32_hf = (mine32.double() - ref64)[mk].abs(), (ref32.double() - ref64)[mk].abs()
    print(f"fp32 error vs fp64: mean fastrl {err32.mean():.2e} HF {err32_hf.mean():.2e}, max fastrl {err32.max():.2e} HF {err32_hf.max():.2e}")
    assert err32.mean() < 1e-5 and err32.mean() <= 1.5 * err32_hf.mean() and err32.max() <= 1.5 * err32_hf.max()
    with torch.no_grad():
        mine16 = model.train_logprobs(w16, prompts, resp, N)
        pad = 660 - Lp
        ids = torch.cat([torch.zeros(len(resp), pad, dtype=torch.long, device="cuda"), ids], 1)
        am = torch.cat([torch.zeros(len(resp), pad, dtype=torch.long, device="cuda"), torch.ones(len(resp), Lp, dtype=torch.long, device="cuda"), mk.long()], 1)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, r_eff = prefix_shared_logits(hf.half(), ids, am, torch.clip(am.cumsum(-1) - 1, min=0), r)
            ref16 = logprobs_from_logits(logits, resp[:, :r_eff])
    assert ref16.dtype == torch.float32 and r_eff == r
    err, err_verl = (mine16.double() - ref64)[mk].abs(), (ref16.double() - ref64)[mk].abs()
    for q in (0.5, 0.99, 0.999):
        print(f"fp16 error vs fp64, quantile {q}: fastrl {err.quantile(q):.2e}, verl {err_verl.quantile(q):.2e}")
        assert err.quantile(q) <= 1.1 * err_verl.quantile(q)
    assert err.mean() <= 1.1 * err_verl.mean() < 2e-3


def test_decode_logprobs_match_forward(model, val, rollouts, vocab):
    """The decoder's log-prob of every sampled token (rows compacted mid-decode included) against the teacher-forced
    forward on the same tokens: within fp16 rounding of the fp16 forward, and no further from the fp32 forward than the
    log-probs verl's PrefixSharedDecoder returns for its own samples. Entropies against the forward's; nothing after EOS."""
    from verl.workers.rollout.prefix_decode import PrefixSharedDecoder
    errors = {}
    for name, base in (("lo", BASE), ("hi", BASE_HI)):
        w16, tok, logp, ent, mask = rollouts[name]
        assert (tok[~mask] == vocab.pad).all() and (mask.sum(1) < 300).float().mean() > 0.9
        hf = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
        torch.manual_seed(0)
        out = PrefixSharedDecoder(hf, 300).generate(val[:100], 1.0, [vocab.eos], vocab.pad, N, want_logits=True)
        ref_tok = torch.full((100 * N, 300), vocab.pad, device="cuda")
        ref_tok[:, :len(out.logits)] = out.sequences[:, val.shape[1]:]
        ref_logp = torch.zeros(100 * N, 300, device="cuda")
        ref_logp[:, :len(out.logits)] = torch.stack(out.logits, 1).log_softmax(-1).gather(-1, ref_tok[:, :len(out.logits), None]).squeeze(-1)
        e = {"fastrl": [], "verl": []}
        with torch.no_grad():
            for g0 in range(0, len(val), 100):
                prompts, resp, mk = full_sequences(val, tok, mask, torch.arange(g0, g0 + 100, device="cuda"))
                rows = slice(g0 * N, (g0 + 100) * N)
                r = resp.shape[1]
                tf16 = model.train_logprobs(w16, prompts, resp, N)
                assert (logp[rows, :r] - tf16)[mk].abs().mean() < 1e-3
                e["fastrl"].append((logp[rows, :r] - model.train_logprobs(master(base), prompts, resp, N))[mk].abs())
            ref_mask = fr.response_mask(ref_tok, vocab.eos)
            prompts, resp, mk = full_sequences(val, ref_tok, ref_mask, torch.arange(100, device="cuda"))
            e["verl"].append((ref_logp[:, :resp.shape[1]] - model.train_logprobs(master(base), prompts, resp, N))[mk].abs())
        errors[name] = {k: torch.cat(x) for k, x in e.items()}
        for q in (0.5, 0.99, 0.999):
            a, b = (errors[name][k].quantile(q) for k in ("fastrl", "verl"))
            print(f"{name}: decode log-prob error vs fp32 forward, quantile {q}: fastrl {a:.2e}, verl {b:.2e}")
            assert a <= 1.1 * b
        print(f"{name}: mean error fastrl {errors[name]['fastrl'].mean():.2e}, verl {errors[name]['verl'].mean():.2e}")
        assert errors[name]["fastrl"].mean() <= 1.1 * errors[name]["verl"].mean()
        # E[-log pi(sampled token)] = entropy: over ~10^6 sampled tokens the two means agree
        assert (ent[mask] >= 0).all() and abs(ent[mask].mean() / -logp[mask].mean() - 1) < 0.01


def test_sampler_token_frequencies(model, vocab, val):
    """First response token: 2^15 draws from one prompt against softmax(logits) (chi-square)."""
    from scipy.stats import chisquare
    w = {k: v.half() for k, v in master(BASE).items()}
    torch.manual_seed(1)
    tok, logp, _ = fr.Decoder(model, vocab, 8).generate(w, val[3:4], 2 ** 15)
    with torch.no_grad():
        hidden = model.forward(w, val[3:4], torch.arange(val.shape[1], device="cuda"), model.causal([]))
        probs = model.logits(w, hidden[0, -1]).float().softmax(-1)
    counts = torch.bincount(tok[:, 0], minlength=len(vocab.ids)).float()
    keep = probs * 2 ** 15 > 5  # tokens expected fewer than 5 times share one bucket
    obs = torch.cat([counts[keep], counts[~keep].sum()[None]])
    exp = torch.cat([probs[keep], probs[~keep].sum()[None]]).double() * 2 ** 15
    assert keep.sum() >= 2
    assert chisquare(obs.cpu().numpy(), exp.cpu().numpy() * obs.sum().item() / exp.sum().item()).pvalue > 1e-3


def test_sampling_statistics_match_verl_decoder(model, vocab, val):
    """Validation outcomes and lengths from fastrl's decoder against verl's PrefixSharedDecoder (same fp16 weights, 3 x 16
    samples per prompt each): paired by prompt, every mean within 4 standard errors."""
    from verl.workers.rollout.prefix_decode import PrefixSharedDecoder
    hf = AutoModelForCausalLM.from_pretrained(BASE_HI, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    ref_dec, dec = PrefixSharedDecoder(hf, 300), fr.Decoder(model, vocab, 300)
    w = {k: v.half() for k, v in master(BASE_HI).items()}
    mz = fr.Mazes(val, vocab).repeat(N)
    stats = {"mine": [], "verl": []}
    for rep in range(3):
        torch.manual_seed(100 + rep)
        tok = dec.generate(w, val, N)[0]
        out = ref_dec.generate(val, 1.0, [vocab.eos], vocab.pad, N, want_logits=False).sequences[:, val.shape[1]:]
        ref = torch.full_like(tok, vocab.pad)
        ref[:, :out.shape[1]] = out
        for key, t in (("mine", tok), ("verl", ref)):
            j = fr.judge(t, mz, vocab)
            j["length"] = fr.response_mask(t, vocab.eos).sum(1).float()
            stats[key].append(torch.stack([j[k].view(-1, N).mean(1) for k in ("is_shortest", "goal_reached", "valid_format", "wall_collision", "length")]))
    a, b = torch.stack(stats["mine"]).mean(0), torch.stack(stats["verl"]).mean(0)  # (5, prompts)
    d = a - b
    se = d.std(1) / d.shape[1] ** 0.5
    print("mine", a.mean(1).tolist(), "verl", b.mean(1).tolist(), "z", (d.mean(1) / se).tolist())
    assert ((d.mean(1) / se).abs() < 4).all()


def verl_batch(val, tok, mask, groups, old_logp, adv):
    from tensordict import TensorDict
    from verl import DataProto
    prompts = val[groups].repeat_interleave(N, 0)
    rows = (groups[:, None] * N + torch.arange(N, device="cuda")).flatten()
    B, Lp = prompts.shape
    pad = torch.zeros(B, 660 - Lp, dtype=torch.long, device="cuda")
    am = torch.cat([pad, torch.ones_like(prompts), mask[rows].long()], 1)
    td = TensorDict({"responses": tok[rows], "input_ids": torch.cat([pad, prompts, tok[rows]], 1), "attention_mask": am,
                     "position_ids": torch.clip(am.cumsum(-1) - 1, min=0), "old_log_probs": old_logp[rows],
                     "advantages": adv[rows]}, batch_size=B)
    return DataProto(batch=td, meta_info={"temperature": 1.0}), rows


def verl_actor(base, dtype, micro, batch, lr):
    """verl's actor as fsdp_workers builds it for the t3 grids: fp32 HF module, FSDP (flat params, one unit per decoder
    layer) with param_dtype = actor.dtype, AdamW, prefix-shared forward, TIS loss."""
    import functools

    import torch.distributed as dist
    from omegaconf import OmegaConf
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    if not dist.is_initialized():
        dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29571", rank=0, world_size=1)
    hf = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, attn_implementation="sdpa")
    pd = {"float16": torch.float16, "float32": torch.float32}[dtype]
    module = FSDP(hf, mixed_precision=MixedPrecision(param_dtype=pd, reduce_dtype=torch.float32, buffer_dtype=torch.float32),
                  auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Qwen2DecoderLayer}),
                  device_id=torch.cuda.current_device(), use_orig_params=False)
    opt = torch.optim.AdamW(module.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    cfg = OmegaConf.create({
        "strategy": "fsdp", "dtype": dtype, "ppo_mini_batch_size": batch, "ppo_micro_batch_size_per_gpu": micro,
        "use_dynamic_bsz": False, "ppo_max_token_len_per_gpu": 16384, "grad_clip": 1.0, "clip_ratio": 0.2,
        "clip_ratio_low": 0.2, "clip_ratio_high": 0.2, "clip_ratio_c": 3.0, "loss_agg_mode": "token-mean",
        "entropy_coeff": 0, "use_kl_loss": False, "use_torch_compile": True, "ppo_epochs": 1, "shuffle": False,
        "ulysses_sequence_parallel_size": 1, "entropy_from_logits_with_chunking": False, "entropy_checkpointing": False,
        "prefix_shared_forward": True, "use_remove_padding": False, "policy_loss": {"loss_mode": "tis", "tis_cap": 2.0}})
    return DataParallelPPOActor(cfg, module, opt), module, opt


def fastrl_update(model, dtype, micro, lr, prompts, tok, old, adv, mask):
    """fastrl.Trainer.update from the 10% base: (metrics, AdamW first moment, weights before, weights after)."""
    tr = object.__new__(fr.Trainer)
    tr.a = argparse.Namespace(n=N, micro=micro, tis_cap=2.0)
    tr.model = model
    tr.params = {k: v.requires_grad_() for k, v in master(BASE_HI).items()}
    tr.opt = torch.optim.AdamW(tr.params.values(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    tr.scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 16, growth_interval=400, enabled=dtype == "float16")
    if dtype == "float32":
        tr.weights16 = lambda: dict(tr.params)  # fp32 compute: no cast
    w0 = {k: p.detach().clone() for k, p in tr.params.items()}
    metrics = tr.update(prompts, tok, old, adv, mask)
    return metrics, {k: tr.opt.state[p]["exp_avg"] for k, p in tr.params.items()}, w0, {k: p.detach() for k, p in tr.params.items()}


def fp64_update(prompts, tok, old, adv, mask, micro):
    """The same update loss on the HF model in float64 (eager attention): (gradient by HF name, pg_loss)."""
    hf = AutoModelForCausalLM.from_pretrained(BASE_HI, torch_dtype=torch.float64, attn_implementation="eager").cuda()
    Lp, losses = prompts.shape[1], []
    k = len(tok) // micro
    for j in range(k):
        s = slice(j * micro, (j + 1) * micro)
        r = int(mask[s].sum(1).max())
        mk = mask[s, :r].double()
        denom = mk.sum() + 1e-8
        total = 0.0
        for c in range(j * micro, (j + 1) * micro, 32):  # the micro-batch's token sum, 32 rows at a time
            cc = slice(c, c + 32)
            ids = torch.cat([prompts.repeat_interleave(N, 0)[cc], tok[cc, :r]], 1)
            lp = hf(input_ids=ids).logits[:, Lp - 1:-1].log_softmax(-1).gather(-1, tok[cc, :r, None]).squeeze(-1)
            ratio = (lp.detach() - old[cc, :r].double()).clamp(-20.0, 20.0).exp()
            part = (-adv[cc, :r].double() * ratio.clamp(max=2.0) * lp * mask[cc, :r].double()).sum() / denom
            (part / k).backward()
            total += part.item()
        losses.append(total)
    grads = {name.replace("lm_head.weight", "model.embed_tokens.weight"): p.grad for name, p in hf.named_parameters()}
    return grads, float(np.mean(losses))


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_update_matches_verl_actor(model, vocab, val, rollouts, dtype):
    """One optimizer step of fastrl.Trainer.update and of verl's DataParallelPPOActor.update_policy from the same weights
    on the same 512-sample batch (2 micro-batches of 256), for 6 batches: loss stats, grad norm, the clipped gradient
    (read back from AdamW's first moment) and the weight update. The gradient sums +-A weighted terms that largely cancel,
    so rounding shows in it even in fp32 and varies from batch to batch: both are compared with the same update computed
    in float64, and on average fastrl must be about as close to it as verl."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    _, tok, logp, _, mask = rollouts["hi"]
    micro, lr = 256, 1e-5
    adv = fr.grpo_advantage(fr.judge(tok, fr.Mazes(val, vocab).repeat(N), vocab)["score"], N)[:, None] * mask
    old = logp * mask
    errs, metric_errs, coss = [], [], []
    for b in range(6):
        groups = torch.arange(b, 32 * 15 + b, 15, device="cuda")
        data, rows = verl_batch(val, tok, mask, groups, old, adv)
        actor, module, vopt = verl_actor(BASE_HI, dtype, micro, len(rows), lr)
        vm = actor.update_policy(data)
        with FSDP.summon_full_params(module):
            v_new = {k.replace("_fsdp_wrapped_module.", ""): p.detach().clone() for k, p in module.named_parameters()}
        g_verl = {k: s["exp_avg"].cuda() / 0.1 for k, s in FSDP.optim_state_dict(module, vopt)["state"].items()}
        batch = (val[groups], tok[rows], old[rows], adv[rows], mask[rows])
        mm, m1, w0, w1 = fastrl_update(model, dtype, micro, lr, *batch)
        g_fast = {k: v / 0.1 for k, v in m1.items()}
        g64, loss64 = fp64_update(*batch, micro)
        flat = lambda d: torch.cat([d[k].double().flatten() for k in w0])
        norm64 = flat(g64).norm().item()
        clipped = flat(g64) * min(1.0, 1.0 / (norm64 + 1e-6))
        err = lambda g: ((flat(g) - clipped).norm() / clipped.norm()).item()
        errs.append((err(g_fast), err(g_verl)))
        metric_errs.append([(abs(mm[k] - t), abs(float(np.mean(vm[k])) - t)) for k, t in (("actor/pg_loss", loss64), ("actor/grad_norm", norm64))])
        delta = lambda w: flat({k: w[k] - w0[k] for k in w0})
        coss.append(F.cosine_similarity(delta(w1), delta(v_new), 0).item())
        print(f"{dtype} batch {b}: pg_loss fastrl {mm['actor/pg_loss']:.7g} verl {float(np.mean(vm['actor/pg_loss'])):.7g} fp64 {loss64:.7g}; "
              f"grad norm fastrl {mm['actor/grad_norm']:.6g} verl {float(np.mean(vm['actor/grad_norm'])):.6g} fp64 {norm64:.6g}; "
              f"gradient rel. error vs fp64: fastrl {errs[-1][0]:.2e}, verl {errs[-1][1]:.2e}; update cos {coss[-1]:.6f}")
        for key in ("actor/ppo_kl", "actor/pg_clipfrac", "actor/pg_clipfrac_lower"):
            assert abs(mm[key] - float(np.mean(vm[key]))) < (1e-7 if dtype == "float32" else 1e-4), key
    e = np.array(errs).mean(0)
    me = np.array(metric_errs).mean(0)  # (metric, [fastrl, verl])
    print(f"{dtype}: mean gradient error vs fp64 fastrl {e[0]:.2e}, verl {e[1]:.2e}; pg_loss / grad-norm errors {me.tolist()}")
    assert e[0] <= 1.5 * e[1] and (dtype == "float16" or e[0] < 1e-3)
    assert (me[:, 0] <= 1.5 * me[:, 1] + 1e-7).all()
    assert min(coss) > (0.9999 if dtype == "float32" else 0.95)  # fp16: both are 2-60% off the fp64 gradient


def test_train_and_resume(tmp_path, monkeypatch):
    """6 steps of the real Trainer with validations at 3 and 6; then the same run stopped at 3 and resumed to 6: same
    step count, validation history and weights restored exactly at the resume point."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    args = lambda out, steps: argparse.Namespace(base=BASE, train=TRAIN, val=VAL, out=str(out), batch=16, seed=3, n=N, micro=128,
                                                 lr=1e-5, warmup=2, tis_cap=2.0, steps=steps, test_freq=3, patience=100,
                                                 val_n=N, max_response=300, wandb_project="x", wandb_id="t")
    import json
    fr.Trainer(args(tmp_path / "a", 6)).run()
    fr.Trainer(args(tmp_path / "b", 3)).run()
    saved = torch.load(tmp_path / "b" / "ckpt.pt", weights_only=False)
    tr = fr.Trainer(args(tmp_path / "b", 6))
    assert tr.step == 3 and all(torch.equal(tr.params[k].detach(), saved["params"][k]) for k in tr.params)
    assert tr.opt.state_dict()["state"][0]["step"] == 3
    tr.run()
    for run in ("a", "b"):
        best = json.load(open(tmp_path / run / "best.json"))
        done = json.load(open(tmp_path / run / "COMPLETE.json"))
        assert [h[0] for h in best["history"]] == [0, 3, 6] and done["reason"] == "max_steps" and done["last_step"] == 6
        assert os.path.exists(tmp_path / run / "best_hf" / "model.safetensors")
        assert json.load(open(tmp_path / run / "best_hf" / "best_step.json"))["step"] == best["best_step"]
