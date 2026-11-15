"""Fast single-GPU GRPO + TIS for the text-maze models (Qwen2, tied embeddings, ~1M params).

The recipe of rl/rl_tis_run.sh as the t3 grids run it on verl (TailRL + tailrl_rl_tis.patch with FAST_DECODE, PREFIX_DECODE,
PREFIX_FORWARD), in one process: no Ray, FSDP, HF generate or reward workers.
- Prompts are drawn in verl's order: a RandomSampler seeded with --seed, drop_last batches.
- --n samples per prompt at temperature 1 from the fp16 policy. Each prompt's KV cache is computed once and shared by its
  samples; finished rows are compacted away at 32-token segment boundaries; each segment's decode step is a CUDA graph.
  The log-prob and entropy of every sampled token are kept (the behaviour policy for TIS).
- Reward 1 iff the response reaches the goal by a shortest path (TailRL's judge_maze_binary_shortest), computed on the GPU.
- GRPO advantage: (r - group mean) / (group unbiased std + 1e-6).
- One update per batch. Each token's -A log pi is weighted by min(pi_train / pi_rollout, --tis-cap) (detached), averaged
  over the tokens of each micro-batch and divided by the number of micro-batches. Master weights are fp32; the forward and
  backward run in fp16 (FSDP mixed precision) under a dynamic loss scale (2^16, growth every 400 steps). Grad-norm clip
  1.0, AdamW (0.9, 0.999), weight decay 0.01, lr * min(1, (step - 1) / --warmup).
- Validation at step 0 and every --test-freq steps on the --val prompts x 16 samples. Writes best.json, best_hf/ and
  COMPLETE.json as verl does, logs verl's W&B metric names, and checkpoints at every validation (resumed in place).
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.fx.experimental._config
import torch.nn.functional as F
import wandb
from safetensors.torch import load_file, save_file
from torch.nn.attention.bias import causal_lower_right

# The small elementwise / reduction steps (norms, rotary, attention merge, sampling) are compiled into one kernel each: the
# decode step is launch-bound (~280 kernels before, each costing GPU time and a graph node). Shapes are dynamic, a size-1
# dimension reuses the dynamic kernels and equal sizes get separate symbols, so nothing recompiles mid-decode (a compile
# inside graph capture fails).
fused = torch.compile(dynamic=True)
torch.fx.experimental._config.backed_size_oblivious = True
torch.fx.experimental._config.use_duck_shape = False

SEG = 32  # decode tokens per CUDA-graph segment
COMPACT_KEEP_FRAC = 0.75  # compact the decode layout when the unfinished rows fit in this fraction of it
ES_METRIC = "val-aux/maze_17/is_shortest/mean@16"
OUTCOMES = ("goal_reached", "wall_collision", "valid_format", "done_token_generated", "valid_path_no_goal", "is_shortest")


class Vocab:
    def __init__(self, path):
        tok = json.load(open(os.path.join(path, "tokenizer.json")))
        self.ids = tok["model"]["vocab"]
        self.pad, self.eos = self.ids["<pad>"], self.ids["<eos>"]
        self.moves = torch.tensor([self.ids[a] for a in ("UP", "DOWN", "LEFT", "RIGHT")], device="cuda")
        # ids that vanish from decode(skip_special_tokens=True).split(): the special tokens and the "\n" token
        self.skip = torch.tensor([t["id"] for t in tok["added_tokens"] if t["special"]] + [self.ids["\n"]], device="cuda")
        self.delta = torch.zeros(len(self.ids), 2, dtype=torch.long, device="cuda")
        self.delta[self.moves] = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], device="cuda")


def load_prompts(path, vocab):
    """(N, L) uint8 token ids of the parquet's prompts (verl: tokenizer(content, add_special_tokens=False); the word-level
    tokenizer splits on spaces), cached as <path>.tokens.npy and memory-mapped."""
    cache = path + ".tokens.npy"
    if not os.path.exists(cache):
        names = pa.array(list(vocab.ids))
        lut = np.array(list(vocab.ids.values()), dtype=np.uint8)
        parts = []
        for batch in pq.ParquetFile(path).iter_batches(columns=["prompt"], batch_size=65536):
            prompt = batch.column(0)
            assert pc.all(pc.equal(pc.list_value_length(prompt), 1)).as_py()
            words = pc.utf8_split_whitespace(pc.struct_field(pc.list_flatten(prompt), "content"))
            lens = pc.list_value_length(words).to_numpy()
            idx = pc.index_in(pc.list_flatten(words), value_set=names)
            assert idx.null_count == 0 and (lens == lens[0]).all(), "unknown words or prompts of different lengths"
            parts.append(lut[idx.to_numpy()].reshape(len(lens), lens[0]))
        np.save(cache + ".tmp.npy", np.concatenate(parts))
        os.replace(cache + ".tmp.npy", cache)
    return np.load(cache, mmap_mode="r")


class Mazes:
    """Grids (open cells), starts, goals and shortest-path lengths of prompts <bos> GRID_START (S x (S cells NEWLINE))
    GRID_END PATH_START."""

    def __init__(self, prompts, vocab):
        B, L = prompts.shape
        S = int(round(((4 * L - 15) ** 0.5 - 1) / 2))
        v = vocab.ids
        assert S * (S + 1) + 4 == L, L
        assert (prompts[:, :2] == torch.tensor([v["<bos>"], v["GRID_START"]], device=prompts.device)).all()
        assert (prompts[:, -2:] == torch.tensor([v["GRID_END"], v["PATH_START"]], device=prompts.device)).all()
        cells = prompts[:, 2:-2].reshape(B, S, S + 1)
        assert (cells[:, :, S] == v["NEWLINE"]).all()
        cells = cells[:, :, :S]
        assert torch.isin(cells, torch.tensor([v["WALL"], v["PATH"], v["START"], v["GOAL"]], device=cells.device)).all()
        assert ((cells == v["START"]).sum((1, 2)) == 1).all() and ((cells == v["GOAL"]).sum((1, 2)) == 1).all()
        self.S, self.free = S, cells != v["WALL"]
        flat = lambda token: (cells == token).flatten(1).int().argmax(1)
        self.start, self.goal = (torch.stack([i // S, i % S], 1) for i in (flat(v["START"]), flat(v["GOAL"])))
        assert (self.start != self.goal).any(1).all()
        self.lstar = self._bfs()

    def _bfs(self):
        """Shortest path length start -> goal per maze (TailRL's _bfs_optimal_length), -1 when unreachable."""
        B, S = self.free.shape[0], self.S
        free = F.pad(self.free, (1, 1, 1, 1))  # a closed border: shifts never wrap
        seen = torch.zeros_like(free)
        b = torch.arange(B, device=free.device)
        seen[b, self.start[:, 0] + 1, self.start[:, 1] + 1] = True
        dist = torch.full((B,), -1, device=free.device)
        for k in range(1, S * S):
            grow = seen.clone()
            grow[:, 1:] |= seen[:, :-1]
            grow[:, :-1] |= seen[:, 1:]
            grow[:, :, 1:] |= seen[:, :, :-1]
            grow[:, :, :-1] |= seen[:, :, 1:]
            seen = grow & free
            dist = torch.where((dist < 0) & seen[b, self.goal[:, 0] + 1, self.goal[:, 1] + 1], k, dist)
            if k % 16 == 0 and bool((dist >= 0).all()):
                break
        return dist



def judge(resp, maze, idx, vocab):
    """TailRL's judge_maze_binary_shortest on token ids (B, R), row i on maze idx[i]: the actions are the words of
    decode(resp, skip_special_tokens=True) before the first DONE (all of them moves, else invalid format), run from the start
    until the goal or a wall / the edge. Returns float (B,) tensors: score (= is_shortest) and the OUTCOMES."""
    R = resp.shape[1]
    done_tok = resp == vocab.ids["DONE"]
    has_done = done_tok.any(1)
    first_done = torch.where(has_done, done_tok.int().argmax(1), R)
    before = torch.arange(R, device=resp.device) < first_done[:, None]
    move = before & torch.isin(resp, vocab.moves)
    valid = has_done & ~(before & ~move & ~torch.isin(resp, vocab.skip)).any(1)
    pos = maze.start[idx, None] + (vocab.delta[resp] * move[..., None]).cumsum(1)  # (B, R, 2) after each token
    inside = ((pos >= 0) & (pos < maze.S)).all(-1)
    p = pos.clamp(0, maze.S - 1)
    blocked = move & (~inside | ~maze.free[idx[:, None], p[..., 0], p[..., 1]])
    at_goal = move & ~blocked & (pos == maze.goal[idx, None]).all(-1)
    first = lambda m: torch.where(m.any(1), m.int().argmax(1), R)
    first_block, first_goal = first(blocked), first(at_goal)
    reached = valid & (first_goal < first_block)
    hit = valid & (first_block < first_goal)
    n_moves = move.cumsum(1).gather(1, first_goal.clamp(max=R - 1)[:, None]).squeeze(1)
    shortest = reached & (n_moves <= maze.lstar[idx])
    out = {"goal_reached": reached, "wall_collision": hit, "valid_format": valid, "done_token_generated": has_done,
           "valid_path_no_goal": valid & ~reached & ~hit, "is_shortest": shortest, "score": shortest}
    return {k: x.float() for k, x in out.items()}


def response_mask(resp, eos):
    """1 up to and including the first EOS (verl get_response_mask)."""
    e = (resp == eos).int()
    return (e.cumsum(1) - e) == 0


@fused
def rms(x, w, eps):
    h = x.float()
    return w * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


@fused
def rotate(x, cos, sin):
    x1, x2 = x.chunk(2, -1)
    return x * cos + torch.cat([-x2, x1], -1) * sin


class Qwen2:
    """The HF Qwen2ForCausalLM forward (tied embeddings, no sliding window), on a dict of weights in HF names."""

    def __init__(self, path):
        c = json.load(open(os.path.join(path, "config.json")))
        assert c["model_type"] == "qwen2" and c["tie_word_embeddings"] and c["hidden_act"] == "silu"
        assert c["rope_scaling"] is None and not c["use_sliding_window"] and c.get("head_dim") is None
        self.c, self.nl, self.eps = c, c["num_hidden_layers"], c["rms_norm_eps"]
        self.h, self.kvh = c["num_attention_heads"], c["num_key_value_heads"]
        self.d, self.rep = c["hidden_size"] // self.h, self.h // self.kvh
        self.inv_freq = 1.0 / (c["rope_theta"] ** (torch.arange(0, self.d, 2, dtype=torch.int64).float() / self.d)).cuda()

    def forward(self, w, ids, pos, attend):
        """Final-norm hidden states (N, T, H) of ids (N, T) at positions pos (T,); attend(i, q, k, v) is layer i's attention
        on the rotated q (N, h, T, d) and k, v (N, kvh, T, d), returning (N, h, T, d)."""
        x = F.embedding(ids, w["model.embed_tokens.weight"])
        f = pos.float()[:, None] * self.inv_freq
        emb = torch.cat([f, f], -1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        for i in range(self.nl):
            p = f"model.layers.{i}."
            hs = rms(x, w[p + "input_layernorm.weight"], self.eps)
            proj = lambda name, heads: F.linear(hs, w[p + f"self_attn.{name}.weight"], w[p + f"self_attn.{name}.bias"]
                                                ).unflatten(-1, (heads, self.d)).transpose(1, 2)
            q, k, v = proj("q_proj", self.h), proj("k_proj", self.kvh), proj("v_proj", self.kvh)
            o = attend(i, rotate(q, cos, sin), rotate(k, cos, sin), v)
            x = x + F.linear(o.transpose(1, 2).flatten(2), w[p + "self_attn.o_proj.weight"])
            hs = rms(x, w[p + "post_attention_layernorm.weight"], self.eps)
            x = x + F.linear(F.silu(F.linear(hs, w[p + "mlp.gate_proj.weight"])) * F.linear(hs, w[p + "mlp.up_proj.weight"]),
                             w[p + "mlp.down_proj.weight"])
        return rms(x, w["model.norm.weight"], self.eps)

    def logits(self, w, hidden):
        return F.linear(hidden, w["model.embed_tokens.weight"])

    def heads(self, x):
        """kv heads repeated for the query heads (HF repeat_kv; SDPA's enable_gqa has a less precise backward)."""
        return x.repeat_interleave(self.rep, 1)

    def causal(self, kv):
        """Causal self-attention that also keeps each layer's (k, v) in kv."""
        def attend(i, q, k, v):
            kv.append((k, v))
            return F.scaled_dot_product_attention(q, self.heads(k), self.heads(v), is_causal=True)
        return attend

    def train_logprobs(self, w, prompts, resp, n):
        """log pi (float32, (G n, r)) of resp (G n, r) given the G prompts, n rows each (verl actor/prefix_forward.py: each
        prompt runs once, every row runs its response against its prompt's cache). The log-softmax of the fp16 logits is
        taken in fp32, as under verl's fp16 autocast."""
        G, Lp = prompts.shape
        r = resp.shape[1]
        kv = []
        hidden = self.forward(w, prompts, torch.arange(Lp, device=prompts.device), self.causal(kv))
        first = self.logits(w, hidden[:, -1:]).repeat_interleave(n, 0)  # predicts response token 0
        mask = causal_lower_right(r - 1, Lp + r - 1)  # response token i sees the prompt and response tokens <= i

        def attend(i, q, k, v):
            k = torch.cat([kv[i][0].repeat_interleave(n, 0), k], 2)
            v = torch.cat([kv[i][1].repeat_interleave(n, 0), v], 2)
            return F.scaled_dot_product_attention(q, self.heads(k), self.heads(v), attn_mask=mask)

        hidden = self.forward(w, resp[:, :-1], torch.arange(Lp, Lp + r - 1, device=prompts.device), attend)
        logits = torch.cat([first, self.logits(w, hidden)], 1)
        return logits.float().log_softmax(-1).gather(-1, resp[..., None]).squeeze(-1)


@fused
def sample(logits, noise, done, pad):
    """A draw from softmax(logits) (Gumbel-max on Exp(1) noise; pad for finished rows), its log-prob and the entropy."""
    logp = logits.log_softmax(-1)
    tok = torch.where(done, pad, torch.argmax(logits - noise.clamp_min(1e-30).log(), -1))
    return tok, logp.gather(1, tok[:, None]).squeeze(1), -(logp.exp() * logp).sum(-1)


@fused
def merge_scores(sr, lse_p, future):
    """Response-part attention weights and the prompt part's share, for one softmax over prompt and response scores."""
    sr = sr.float().masked_fill(future, float("-inf"))
    lse = torch.logaddexp(lse_p, sr.logsumexp(-1))
    return (sr - lse[..., None]).exp().to(torch.float16), (lse_p - lse).exp()


@fused
def merge_outputs(op, share, orr):
    return (op.float() * share[..., None] + orr.float()).to(torch.float16)


class Decoder:
    """Sampling at temperature 1 with one KV cache per prompt shared by its n rows (verl rollout/prefix_decode.py). Rows of
    a group read the prompt cache once per step; each row's own cache holds its generated tokens. Decoding runs in
    segments of SEG tokens with fixed shapes: the step is captured as a CUDA graph and replayed, sampling included
    (Gumbel-max); the first token runs eagerly as the warm-up. At each segment boundary the layout drops finished rows (and
    groups with none left) when that removes at least 1 - COMPACT_KEEP_FRAC of it. Attention to the prompt is a fused
    flash-attention call per group; it merges with the attention to the row's own tokens through the log-sum-exp (one
    softmax over both)."""

    def __init__(self, model, vocab, max_new):
        self.m, self.v, self.R = model, vocab, max_new
        self.slots = -(-max_new // SEG) * SEG  # response cache slots: segments end on multiples of SEG (aligned widths)
        self.side = torch.cuda.Stream()
        # one memory pool for every graph, so each capture reuses the memory the previous ones freed; a pool lives as long
        # as a graph in it, so the latest graph is kept
        self.pool, self.graph = torch.cuda.graph_pool_handle(), None

    @torch.no_grad()
    def generate(self, w, prompts, n):
        """Tokens (G n, R) (pad after EOS), and log pi and entropy (float32, (G n, R)) of each sampled token; row g n + j
        is sample j of prompt g."""
        m = self.m
        G, Lp = prompts.shape
        B, dev = G * n, prompts.device
        kv = []
        hidden = m.forward(w, prompts, torch.arange(Lp, device=dev), m.causal(kv))
        zeros = lambda: torch.zeros(G, m.kvh, n, self.slots, m.d, dtype=kv[0][0].dtype, device=dev)  # 0 x unfilled = 0
        st = {"G": G, "n": n, "rows": torch.arange(B, device=dev).view(G, n), "done": torch.zeros(G, n, dtype=torch.bool, device=dev),
              "Kp": [k.contiguous() for k, _ in kv], "Vp": [v.contiguous() for _, v in kv],
              "Kr": [zeros() for _ in kv], "Vr": [zeros() for _ in kv], "t": torch.zeros(1, dtype=torch.long, device=dev),
              "logits": m.logits(w, hidden[:, -1]).float().repeat_interleave(n, 0)}
        out = {"tok": torch.full((B, self.R), self.v.pad, device=dev), "logp": torch.zeros(B, self.R, device=dev),
               "ent": torch.zeros(B, self.R, device=dev)}
        t = 1
        self.side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side):  # token 0 eagerly, on the capture stream: warms up kernels and workspaces
            self._step(w, st, out, SEG, Lp)
        torch.cuda.current_stream().wait_stream(self.side)
        while t < self.R and not bool(st["done"].all()):
            self._compact(st)
            W = (t // SEG + 1) * SEG  # the segment's end: response slots [0, W) cover its steps
            graph = torch.cuda.CUDAGraph()  # raw capture: torch.cuda.graph() would gc + empty_cache per segment
            with torch.cuda.stream(self.side):
                graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")
                self._step(w, st, out, W, Lp)
                graph.capture_end()
            self.graph = graph
            while t < min(W, self.R):
                graph.replay()
                t += 1
                if t % 8 == 0 and bool(st["done"].all()):
                    break
        return out["tok"], out["logp"], out["ent"]

    def _step(self, w, st, out, W, Lp):
        """Sample every row's token at slot t from the current logits, record it, then compute the logits after it."""
        t, rows, done = st["t"], st["rows"].view(-1), st["done"].view(-1)
        tok, logp, ent = sample(st["logits"], torch.empty_like(st["logits"]).exponential_(), done, self.v.pad)
        idx = (rows, t.expand_as(rows))
        out["tok"].index_put_(idx, tok)
        out["logp"].index_put_(idx, logp)
        out["ent"].index_put_(idx, ent)
        done |= tok == self.v.eos
        hidden = self.m.forward(w, tok[:, None], Lp + t, self._attend(st, t, W))
        st["logits"].copy_(self.m.logits(w, hidden[:, 0]).float())
        t += 1

    def _attend(self, st, t, W):
        m, G, n = self.m, st["G"], st["n"]
        future = torch.arange(W, device=t.device) > t

        def attend(i, q, k, v):  # q (G n, h, 1, d); k, v (G n, kvh, 1, d); query head j reads kv head j // rep
            Kp, Vp, Kr, Vr = st["Kp"][i], st["Vp"][i], st["Kr"][i], st["Vr"][i]
            Kr.index_copy_(3, t, k.reshape(G, n, m.kvh, 1, m.d).transpose(1, 2))
            Vr.index_copy_(3, t, v.reshape(G, n, m.kvh, 1, m.d).transpose(1, 2))
            qg = (q * m.d ** -0.5).reshape(G, n, m.kvh, m.rep, m.d).transpose(1, 2)  # (G, kvh, n, rep, d)
            # prompt part: a group's n * rep queries against its prompt keys, fused, with the log-sum-exp of the scores
            op, lse_p = torch.ops.aten._scaled_dot_product_flash_attention(
                qg.reshape(G, m.kvh, n * m.rep, m.d), Kp, Vp, scale=1.0)[:2]
            lse_p = lse_p.view(G, m.kvh, n, m.rep)
            # response part: each row's own tokens up to slot t; the two parts merge into one softmax (fp32)
            pr, share = merge_scores(qg @ Kr[..., :W, :].transpose(-1, -2), lse_p, future)  # (G, kvh, n, rep, W)
            o = merge_outputs(op.view(G, m.kvh, n, m.rep, m.d), share, pr @ Vr[..., :W, :])
            return o.transpose(1, 2).reshape(G * n, m.h, 1, m.d)
        return attend

    def _compact(self, st):
        """Keep the unfinished rows (and as few finished ones as fill the layout) of the groups that have any."""
        done, G, n = st["done"], st["G"], st["n"]
        left = (~done).sum(1)
        keep = (left > 0).nonzero().squeeze(1)
        n_new = int(left.max())
        if keep.numel() * n_new > COMPACT_KEEP_FRAC * G * n:
            return
        sel = torch.argsort(done.int(), dim=1, stable=True)[keep, :n_new]  # unfinished rows first
        gi = keep[:, None]
        st["rows"], st["done"] = st["rows"][gi, sel], done[gi, sel]
        st["logits"] = st["logits"].view(G, n, -1)[gi, sel].flatten(0, 1)
        st["Kp"], st["Vp"] = [x[keep] for x in st["Kp"]], [x[keep] for x in st["Vp"]]
        for key in ("Kr", "Vr"):  # (G, kvh, n, slots, d)
            st[key] = [x.transpose(1, 2)[gi, sel].transpose(1, 2).contiguous() for x in st[key]]
        st["G"], st["n"] = keep.numel(), n_new


def grpo_advantage(scores, n):
    """verl compute_grpo_outcome_advantage for consecutive groups of n: (r - mean) / (unbiased std + 1e-6), (B,)."""
    g = scores.view(-1, n)
    return ((g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-6)).view(-1)


def tis_loss(logp, old_logp, adv, mask, cap):
    """verl compute_policy_loss_tis, token-mean: -A min(pi / pi_rollout, cap) log pi (weight detached), and the stats
    (pg_loss, fraction of tokens above cap, mean log(pi_rollout / pi), fraction below 1 / cap)."""
    log_ratio = (logp.detach() - old_logp).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    mean = lambda x: (x * mask).sum() / (mask.sum() + 1e-8)
    loss = mean(-adv * ratio.clamp(max=cap) * logp)
    return loss, torch.stack([loss.detach(), mean((ratio > cap).float()), mean(-log_ratio), mean((ratio < 1 / cap).float())])


class PromptOrder:
    """verl's train prompts per step: an epoch of RandomSampler(generator seeded with seed) is one torch.randperm, plus a
    second one it draws for the remainder slice [:0] when the epoch runs out; batches drop the remainder."""

    def __init__(self, n, batch, seed):
        self.n, self.batch, self.gen, self.perm, self.epoch = n, batch, torch.Generator().manual_seed(seed), None, -1

    def __call__(self, step):
        epoch, i = divmod(step - 1, self.n // self.batch)
        while self.epoch < epoch:
            self.perm, _ = torch.randperm(self.n, generator=self.gen), torch.randperm(self.n, generator=self.gen)
            self.epoch += 1
        return self.perm[i * self.batch:(i + 1) * self.batch].numpy()


def train_tflops(c, seqlens):
    """verl FlopsCounter._estimate_qwen2_flops (forward + backward) in TFLOPs for sequences of the given lengths."""
    H, h, kvh, nl = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["num_hidden_layers"]
    d = H // h
    dense = (H * c["intermediate_size"] * 3 + H * (h * d + 2 * kvh * d + h * d)) * nl + c["vocab_size"] * H * 2
    s = seqlens.double()
    return float(6 * dense * s.sum() + 12 * (s * s).sum() * d * h * nl) / 1e12


class Trainer:
    def __init__(self, a):
        self.a = a
        self.vocab = Vocab(a.base)
        self.model = Qwen2(a.base)
        self.params = {k: v.float().cuda().requires_grad_() for k, v in load_file(os.path.join(a.base, "model.safetensors")).items()}
        self.opt = torch.optim.AdamW(self.params.values(), lr=a.lr, betas=(0.9, 0.999), weight_decay=0.01)
        self.scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 16, growth_interval=400)
        self.decoder = Decoder(self.model, self.vocab, a.max_response)
        self.train = load_prompts(a.train, self.vocab)
        self.val = torch.from_numpy(np.array(load_prompts(a.val, self.vocab))).cuda().long()
        self.val_maze = Mazes(self.val, self.vocab)
        self.val_idx = torch.arange(len(self.val), device="cuda").repeat_interleave(a.val_n)
        self.order = PromptOrder(len(self.train), a.batch, a.seed)
        self.step, self.flops = 0, {"gen": 0.0, "update_actor": 0.0, "total": 0.0}
        self.best = {"metric": ES_METRIC, "best_value": None, "best_step": None, "since_best": 0, "history": []}
        os.makedirs(a.out, exist_ok=True)
        self.ckpt = os.path.join(a.out, "ckpt.pt")
        if os.path.exists(self.ckpt):
            s = torch.load(self.ckpt)
            with torch.no_grad():
                for k, p in self.params.items():
                    p.copy_(s["params"][k])
            self.opt.load_state_dict(s["opt"])
            self.scaler.load_state_dict(s["scaler"])
            torch.cuda.set_rng_state(s["rng"])
            self.step, self.flops, self.best = s["step"], s["flops"], s["best"]
        else:
            torch.manual_seed(a.seed)
        wandb.init(project=a.wandb_project, id=a.wandb_id, name=a.wandb_id, resume="allow", config=vars(a), dir=a.out)

    def weights16(self):
        """The fp16 compute copy of the fp32 master weights; differentiable, so gradients reach the masters in fp32."""
        return {k: p.half() for k, p in self.params.items()}

    def rollout_weights(self):
        return {k: p.detach().half() for k, p in self.params.items()}

    def validate(self):
        tok, _, _ = self.decoder.generate(self.rollout_weights(), self.val, self.a.val_n)
        out = judge(tok, self.val_maze, self.val_idx, self.vocab)
        metrics = {f"val-aux/maze_17/{k}/mean@{self.a.val_n}": out[k].mean().item() for k in OUTCOMES}
        metrics[f"val-core/maze_17/reward/mean@{self.a.val_n}"] = out["score"].mean().item()
        return metrics

    def update(self, prompts, tok, old_logp, adv, mask):
        """One optimizer step on the batch (rows grouped n per prompt); returns the actor metrics."""
        a, n = self.a, self.a.n
        rows = min(len(tok), a.micro)
        assert len(tok) % rows == 0 and rows % n == 0, (len(tok), a.micro, n)
        k = len(tok) // rows
        lens = mask.sum(1).tolist()
        stats = []
        for j in range(k):
            s = slice(j * rows, (j + 1) * rows)
            r = max(max(lens[s]), 2)  # >= 1 response query for train_logprobs
            logp = self.model.train_logprobs(self.weights16(), prompts[j * rows // n:(j + 1) * rows // n], tok[s, :r], n)
            loss, st = tis_loss(logp, old_logp[s, :r], adv[s, :r], mask[s, :r].float(), a.tis_cap)
            self.scaler.scale(loss / k).backward()
            stats.append(st)
        self.scaler.unscale_(self.opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params.values(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        pg_loss, clipfrac, ppo_kl, clipfrac_lower = torch.stack(stats).mean(0).tolist()
        return {"actor/pg_loss": pg_loss, "actor/pg_clipfrac": clipfrac, "actor/ppo_kl": ppo_kl,
                "actor/pg_clipfrac_lower": clipfrac_lower, "actor/grad_norm": grad_norm.item()}

    def train_step(self):
        a, n, c = self.a, self.a.n, self.model.c
        t0 = time.perf_counter()
        prompts = torch.from_numpy(self.train[self.order(self.step)]).cuda().long()
        tok, logp, ent = self.decoder.generate(self.rollout_weights(), prompts, n)
        mask = response_mask(tok, self.vocab.eos)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = judge(tok, Mazes(prompts, self.vocab), torch.arange(len(prompts), device="cuda").repeat_interleave(n), self.vocab)
        adv = grpo_advantage(out["score"], n)[:, None] * mask
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        lr = a.lr * min(1.0, (self.step - 1) / a.warmup)
        for group in self.opt.param_groups:
            group["lr"] = lr
        metrics = self.update(prompts, tok, logp * mask, adv, mask)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        seqlens = prompts.shape[1] + mask.sum(1)
        gen, upd = train_tflops(c, seqlens) / 3, train_tflops(c, seqlens)
        for key, x in (("gen", gen), ("update_actor", upd), ("total", gen + upd)):
            self.flops[key] += x
        maskf = mask.float()
        metrics.update({
            "actor/entropy": ((ent * maskf).sum() / maskf.sum()).item(), "actor/lr": lr,
            "critic/score/mean": out["score"].mean().item(), "response_length/mean": mask.sum(1).float().mean().item(),
            "response_length/clip_ratio": (~(tok == self.vocab.eos).any(1)).float().mean().item(),
            "train/maze_17/path/is_shortest/fraction": out["is_shortest"].mean().item(),
            "train/maze_17/trajectory/goal_reached": out["goal_reached"].mean().item(),
            "timing_s/gen": t1 - t0, "timing_s/reward": t2 - t1, "timing_s/update_actor": t3 - t2,
            "tflops/gen": gen, "tflops/update_actor": upd, "tflops/total": gen + upd,
            **{f"tflops/cumulative_{k}": x for k, x in self.flops.items()},
            "training/global_step": self.step, "perf/max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 2 ** 30})
        return metrics

    def write_json(self, path, obj):
        with open(path + ".tmp", "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(path + ".tmp", path)

    def save_hf(self, value):
        """best_hf/: the weights in HF format next to the base's config and tokenizer, plus best_step.json."""
        dst = os.path.join(self.a.out, "best_hf")
        shutil.rmtree(dst + ".tmp", ignore_errors=True)
        # *.txt: the base's AVERAGED_FROM / INTERPOLATED_FROM provenance notes
        shutil.copytree(self.a.base, dst + ".tmp", ignore=shutil.ignore_patterns("*.safetensors", "*.txt"))
        save_file({k: p.detach().cpu().contiguous() for k, p in self.params.items()}, os.path.join(dst + ".tmp", "model.safetensors"))
        self.write_json(os.path.join(dst + ".tmp", "best_step.json"), {"step": self.step, "metric": ES_METRIC, "value": value})
        shutil.rmtree(dst, ignore_errors=True)
        os.replace(dst + ".tmp", dst)

    def checkpoint(self, val_metrics):
        """Track the best validation (strictly greater = new best, saved to best_hf), save the training state; returns
        True when --patience validations in a row brought no new best."""
        value, b = val_metrics[ES_METRIC], self.best
        b["history"].append([self.step, value])
        if b["best_value"] is None or value > b["best_value"]:
            b.update(best_value=value, best_step=self.step, since_best=0)
            self.save_hf(value)
        else:
            b["since_best"] += 1
        self.write_json(os.path.join(self.a.out, "best.json"), b)
        torch.save({"params": {k: p.detach() for k, p in self.params.items()}, "opt": self.opt.state_dict(),
                    "scaler": self.scaler.state_dict(), "rng": torch.cuda.get_rng_state(), "step": self.step,
                    "flops": self.flops, "best": b}, self.ckpt + ".tmp")
        os.replace(self.ckpt + ".tmp", self.ckpt)
        return b["since_best"] >= self.a.patience

    def run(self):
        a = self.a
        if not self.best["history"]:
            t = time.perf_counter()
            metrics = self.validate()
            metrics["timing_s/testing"] = time.perf_counter() - t
            self.checkpoint(metrics)
            wandb.log(metrics, step=0)
        stop = self.best["since_best"] >= a.patience
        while self.step < a.steps and not stop:
            self.step += 1
            t = time.perf_counter()
            metrics = self.train_step()
            if self.step % a.test_freq == 0 or self.step == a.steps:
                tv = time.perf_counter()
                metrics.update(self.validate())
                metrics["timing_s/testing"] = time.perf_counter() - tv
                stop = self.checkpoint(metrics)
                metrics.update({f"early_stop/{k}": self.best[k] for k in ("best_value", "best_step", "since_best")})
            metrics["timing_s/step"] = time.perf_counter() - t
            wandb.log(metrics, step=self.step)
        reason = "early_stop" if stop else "max_steps"
        self.write_json(os.path.join(a.out, "COMPLETE.json"), {
            "reason": reason, "last_step": self.step, "best_step": self.best["best_step"], "best_value": self.best["best_value"],
            "metric": ES_METRIC, "flops_total_tflops": self.flops["total"]})
        wandb.finish()


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="HF dir of the starting policy (config.json, model.safetensors, tokenizer.json)")
    ap.add_argument("--train", required=True, help="train parquet (verl RL format)")
    ap.add_argument("--val", required=True, help="validation parquet")
    ap.add_argument("--out", required=True, help="run dir: ckpt.pt, best.json, best_hf/, COMPLETE.json")
    ap.add_argument("--batch", type=int, required=True, help="prompts per step")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=16, help="samples per prompt")
    ap.add_argument("--micro", type=int, default=512, help="cap on samples per update micro-batch (verl PPO_MICRO_SAMPLES)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--tis-cap", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--test-freq", type=int, default=20)
    ap.add_argument("--patience", type=int, default=2000, help="validations without a new best before stopping")
    ap.add_argument("--val-n", type=int, default=16)
    ap.add_argument("--max-response", type=int, default=300)
    ap.add_argument("--wandb-project", default="tailrl-text-maze-dfs25-eps05")
    ap.add_argument("--wandb-id", required=True)
    return ap


def main():
    Trainer(parser().parse_args()).run()


if __name__ == "__main__":
    main()
