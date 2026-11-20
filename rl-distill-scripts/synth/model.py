"""The policy: an HF Qwen2ForCausalLM or Qwen3ForCausalLM (tied embeddings) run functionally on a dict of weights in HF
names, and a sampler.

A model directory is an HF checkpoint (config.json, model.safetensors in fp32, tokenizer files) plus synth.json, the
experiment config it was trained on, so every consumer gets the task from the model (released checkpoints imported as
they are have no synth.json).
"""

import math
import os
import shutil

import torch
import torch.fx.experimental._config
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
from torch.nn.attention.bias import causal_lower_right

from files import read_json, write_json
from tasks import BOS, ENV, EOS, PAD, SPECIAL, make_task

# The small elementwise / reduction steps (norms, rotary, attention merge, sampling) are compiled into one kernel each: the
# decode step is launch-bound. Shapes are dynamic, a size-1 dimension reuses the dynamic kernels and equal sizes get
# separate symbols, so nothing recompiles mid-decode (a compile inside graph capture fails).
fused = torch.compile(dynamic=True)
torch.fx.experimental._config.backed_size_oblivious = True
torch.fx.experimental._config.use_duck_shape = False

ARCH = {  # hidden layers heads kv_heads ffn
    "3XS": (96, 4, 4, 2, 384),     # 0.55M
    "2XS": (128, 4, 4, 2, 512),    # 1.0M
    "XS": (192, 4, 4, 2, 768),     # 2.2M
    "S": (256, 4, 4, 2, 1024),     # 3.9M
    "L": (384, 6, 6, 3, 1536),     # 13.3M
}
SEG = 32  # decode tokens per CUDA-graph segment
GROW = 8 * SEG  # response cache slots added at a time as the decode layout's responses grow
SCRIPT = ("pos", "env", "env_len", "turns", "calls", "fpos", "fleft", "count")  # scripted rows' decode state
COMPACT_KEEP_FRAC = 0.75  # compact the decode layout when the unfinished rows fit in this fraction of it


def hf_config(arch, vocab_size, max_len):
    hidden, layers, heads, kv_heads, ffn = ARCH[arch]
    return {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "vocab_size": vocab_size, "hidden_size": hidden,
            "intermediate_size": ffn, "num_hidden_layers": layers, "num_attention_heads": heads,
            "num_key_value_heads": kv_heads, "max_position_embeddings": max_len, "hidden_act": "silu", "rms_norm_eps": 1e-6,
            "rope_theta": 1e6, "rope_scaling": None, "tie_word_embeddings": True, "use_sliding_window": False,
            "sliding_window": None, "attention_dropout": 0.0, "initializer_range": 0.02, "bos_token_id": BOS,
            "eos_token_id": EOS, "pad_token_id": PAD, "torch_dtype": "float32", "use_cache": True}


def init_weights(c, seed):
    """HF Qwen2 initialization: N(0, 0.02) matrices and embedding (the <pad> row zero), zero biases, unit norms (fp32)."""
    g = torch.Generator().manual_seed(seed)
    H, h, kvh, F_ = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["intermediate_size"]
    d = H // h
    normal = lambda *shape: torch.randn(*shape, generator=g) * c["initializer_range"]
    w = {"model.embed_tokens.weight": normal(c["vocab_size"], H)}
    w["model.embed_tokens.weight"][PAD] = 0
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        w.update({p + "input_layernorm.weight": torch.ones(H), p + "post_attention_layernorm.weight": torch.ones(H),
                  p + "self_attn.q_proj.weight": normal(h * d, H), p + "self_attn.q_proj.bias": torch.zeros(h * d),
                  p + "self_attn.k_proj.weight": normal(kvh * d, H), p + "self_attn.k_proj.bias": torch.zeros(kvh * d),
                  p + "self_attn.v_proj.weight": normal(kvh * d, H), p + "self_attn.v_proj.bias": torch.zeros(kvh * d),
                  p + "self_attn.o_proj.weight": normal(H, h * d), p + "mlp.gate_proj.weight": normal(F_, H),
                  p + "mlp.up_proj.weight": normal(F_, H), p + "mlp.down_proj.weight": normal(H, F_)})
    w["model.norm.weight"] = torch.ones(H)
    return w


def save_model(path, weights, hf_cfg, synth):
    """An HF model directory (written to path.tmp, then moved into place)."""
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    save_file({k: v.detach().float().cpu().contiguous() for k, v in weights.items()}, os.path.join(tmp, "model.safetensors"))
    task = make_task(synth["config"]["task"])
    files = [("config.json", hf_cfg), ("synth.json", synth)]
    if hasattr(task, "HF_FILES"):  # the task's own tokenizer files (a released checkpoint's)
        shutil.copytree(task.HF_FILES, tmp, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    else:
        tok = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(task.WORDS)}, unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.add_special_tokens([AddedToken(t, special=True) for t in SPECIAL])
        tok.save(os.path.join(tmp, "tokenizer.json"))
        special = {"bos_token": "<bos>", "eos_token": "<eos>", "unk_token": "<unk>", "pad_token": "<pad>"}
        files += [("special_tokens_map.json", special), ("tokenizer_config.json", {
            **special, "tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": hf_cfg["max_position_embeddings"]})]
    for name, obj in files:
        write_json(os.path.join(tmp, name), obj)
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)


def load_model(path):
    """(HF config, synth.json (None if absent), fp32 weights on the GPU) of a model directory."""
    weights = {k: v.float() for k, v in load_file(os.path.join(path, "model.safetensors"), device=torch.cuda.current_device()).items()}
    synth = os.path.join(path, "synth.json")
    return read_json(os.path.join(path, "config.json")), read_json(synth) if os.path.exists(synth) else None, weights


@fused
def rms(x, w, eps):
    h = x.float()
    return w * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


@fused
def rotate(x, cos, sin):
    x1, x2 = x.chunk(2, -1)
    return x * cos + torch.cat([-x2, x1], -1) * sin


def rope(c, d):
    """HF's rotary inverse frequencies (d / 2,) and cos / sin scale for rope_scaling None or YaRN (transformers
    _compute_yarn_parameters: beta_fast 32, beta_slow 1, truncated correction range, attention factor 0.1 ln(factor) + 1)."""
    theta, s = c["rope_theta"], c.get("rope_scaling")
    if s is None:
        return 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d)), 1.0
    assert s.get("rope_type", s.get("type")) == "yarn" and set(s) <= {"type", "rope_type", "factor", "original_max_position_embeddings"}, s
    factor, orig = s["factor"], s["original_max_position_embeddings"]
    dim = lambda rotations: d * math.log(orig / (rotations * 2 * math.pi)) / (2 * math.log(theta))
    low, high = max(math.floor(dim(32)), 0), min(math.ceil(dim(1)), d - 1)
    ramp = ((torch.arange(d // 2, dtype=torch.float32) - low) / ((high - low) or 0.001)).clamp(0, 1)
    pos_freqs = theta ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d)
    return 1.0 / (factor * pos_freqs) * ramp + 1.0 / pos_freqs * (1 - ramp), 0.1 * math.log(factor) + 1.0


class Qwen2:
    """The HF Qwen2ForCausalLM / Qwen3ForCausalLM forward (tied embeddings, no sliding window) on a dict of weights in HF
    names. Qwen3: no q/k/v biases, an RMSNorm over each head of q and k before the rotary, head_dim from the config, YaRN."""

    def __init__(self, c):
        assert c["model_type"] in ("qwen2", "qwen3") and c["tie_word_embeddings"] and c["hidden_act"] == "silu"
        assert not c["use_sliding_window"] and not c.get("attention_bias")
        assert c["model_type"] == "qwen3" or (c["rope_scaling"] is None and c.get("head_dim") is None)
        self.c, self.nl, self.eps, self.qk_norm = c, c["num_hidden_layers"], c["rms_norm_eps"], c["model_type"] == "qwen3"
        self.h, self.kvh = c["num_attention_heads"], c["num_key_value_heads"]
        self.d, self.rep = c.get("head_dim") or c["hidden_size"] // self.h, self.h // self.kvh
        inv_freq, self.att = rope(c, self.d)
        self.inv_freq = inv_freq.cuda()

    def forward(self, w, ids, pos, attend):
        """Final-norm hidden states (N, T, H) of ids (N, T) at positions pos (T,) (or (N, T): per row); attend(i, q, k, v)
        is layer i's attention on the rotated q (N, h, T, d) and k, v (N, kvh, T, d), returning (N, h, T, d)."""
        x = F.embedding(ids, w["model.embed_tokens.weight"])
        f = pos.float()[..., None] * self.inv_freq
        emb = torch.cat([f, f], -1)
        cos, sin = emb.cos(), emb.sin()
        if self.att != 1.0:
            cos, sin = cos * self.att, sin * self.att
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        # per-row positions: (N, 1, T, d), expanded to each q / k (the fused kernels do not broadcast a size-1 dimension)
        rot = (lambda y: rotate(y, cos[:, None].expand_as(y), sin[:, None].expand_as(y))) if pos.ndim == 2 else \
            (lambda y: rotate(y, cos, sin))
        for i in range(self.nl):
            p = f"model.layers.{i}."
            hs = rms(x, w[p + "input_layernorm.weight"], self.eps)
            proj = lambda name, heads: F.linear(hs, w[p + f"self_attn.{name}.weight"], w.get(p + f"self_attn.{name}.bias")
                                                ).unflatten(-1, (heads, self.d)).transpose(1, 2)
            q, k, v = proj("q_proj", self.h), proj("k_proj", self.kvh), proj("v_proj", self.kvh)
            if self.qk_norm:
                q, k = rms(q, w[p + "self_attn.q_norm.weight"], self.eps), rms(k, w[p + "self_attn.k_norm.weight"], self.eps)
            o = attend(i, rot(q), rot(k), v)
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

    def causal(self, kv=None):
        """Causal self-attention; also keeps each layer's (k, v) in the list kv, if given."""
        def attend(i, q, k, v):
            if kv is not None:
                kv.append((k, v))
            return F.scaled_dot_product_attention(q, self.heads(k), self.heads(v), is_causal=True)
        return attend

    def train_logprobs(self, w, prompts, resp, n, plen=None):
        """log pi (float32, (G n, r)) of resp (G n, r) given the G prompts, n rows each: each prompt runs once, every row
        runs its response against its prompt's cache (verl actor/prefix_forward.py). The log-softmax is taken in fp32.
        plen (G,): right-padded prompts of these lengths; the shared prefix is the shortest one, and each row's response
        follows the rest of its prompt."""
        if plen is not None:
            Lc, r = int(plen.min()), resp.shape[1]
            tail = (plen - Lc).repeat_interleave(n)
            j = torch.arange(int(tail.max()) + r, device=resp.device)[None]
            rows = prompts.repeat_interleave(n, 0)
            cont = torch.where(j < tail[:, None], rows.gather(1, (Lc + j).clamp(max=rows.shape[1] - 1).expand(len(rows), -1)),
                               resp.gather(1, (j - tail[:, None]).clamp(0, r - 1)))
            return self.train_logprobs(w, prompts[:, :Lc], cont, n).gather(1, tail[:, None] + torch.arange(r, device=resp.device))
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
    """Sampling at temperature 1 from fp16 weights with one KV cache per prompt shared by its n rows (verl
    rollout/prefix_decode.py). Rows of a group read the prompt cache once per step; each row's own cache holds its
    generated tokens. Decoding runs in segments of SEG tokens with fixed shapes: the step is captured as a CUDA graph and
    replayed, sampling included (Gumbel-max); each segment's first step runs eagerly as the warm-up. At each segment boundary the
    layout drops finished rows (and groups with none left) when that removes at least 1 - COMPACT_KEEP_FRAC of it.
    Attention to the prompt is a fused flash-attention call per group; it merges with the attention to the row's own tokens
    through the log-sum-exp (one softmax over both). The response caches grow GROW slots at a time as the layout's
    responses do, so every prompt decodes in one layout while the caches fit in mem_gb; when growing would pass it, the
    layout's last groups move to host memory and resume, from the same step, once the others finish.

    With a script (multi-turn tasks, see generate) prompts are right-padded to their lengths and every row follows
    verl's multi-turn rollout (generate_multi_turn_sequences): after each sampled call token (the config's env_token_id)
    the row is fed its next environment reply, one token per step, then samples again; it finishes at EOS, at its last
    call, at a sampled pad (dropped), or once its sampled tokens reach `budget` or its context (prompt and every response
    token) reaches `max_ctx`."""

    def __init__(self, model, max_new, mem_gb, budget=None, max_ctx=None):
        self.m, self.R, self.budget, self.max_ctx = model, max_new, budget, max_ctx
        self.eos, self.pad, self.call = model.c["eos_token_id"], model.c["pad_token_id"], model.c.get("env_token_id")
        self.slots = -(-max_new // SEG) * SEG  # response cache slots: segments end on multiples of SEG (aligned widths)
        self.kv_bytes = 2 * 2 * model.nl * model.kvh * model.d  # fp16 K and V of one token, all layers
        self.mem = mem_gb * 2 ** 30
        self.side = torch.cuda.Stream()
        # one memory pool for every graph, so each capture reuses the memory the previous ones freed; a pool lives as long
        # as a graph in it, so the latest graph is kept
        self.pool, self.graph = torch.cuda.graph_pool_handle(), None

    @torch.no_grad()
    def generate(self, w, prompts, n, script=None):
        """Tokens (G n, R) (pad after EOS), and log pi and entropy (float32, (G n, R)) of each sampled token; row g n + j
        is sample j of prompt g. script: the task's per-prompt tensors plen (G,) (prompt lengths), env (G, K, F) and
        env_len (G, K) (reply k's tokens), turns (G,) (calls in a full answer); environment tokens come back + ENV, with
        log-prob and entropy 0."""
        G = len(prompts)
        Lp = int(script["plen"].max()) if script else prompts.shape[1]
        chunk = self._fit(Lp, n, min(GROW, self.slots), G)  # the prompts whose caches fit at the first width
        cut = lambda i: None if script is None else {k: v[i:i + chunk] for k, v in script.items()}
        parts = [self._generate(w, prompts[i:i + chunk], n, cut(i)) for i in range(0, G, chunk)]
        return tuple(torch.cat(x) for x in zip(*parts))

    def _generate(self, w, prompts, n, script):
        m = self.m
        G, dev = len(prompts), prompts.device
        Lp = int(script["plen"].max()) if script else prompts.shape[1]
        prompts, B = prompts[:, :Lp], G * n
        kv = []
        hidden = m.forward(w, prompts, torch.arange(Lp, device=dev), m.causal(kv))
        last = script["plen"] - 1 if script else torch.full((G,), Lp - 1, device=dev)
        width = min(GROW, self.slots)
        zeros = lambda: torch.zeros(G, m.kvh, n, width, m.d, dtype=kv[0][0].dtype, device=dev)  # 0 x unfilled = 0
        st = {"G": G, "n": n, "width": width, "rows": torch.arange(B, device=dev).view(G, n),
              "done": torch.zeros(G, n, dtype=torch.bool, device=dev),
              "Kp": [k.contiguous() for k, _ in kv], "Vp": [v.contiguous() for _, v in kv],
              "Kr": [zeros() for _ in kv], "Vr": [zeros() for _ in kv], "t": torch.zeros(1, dtype=torch.long, device=dev),
              "logits": m.logits(w, hidden[torch.arange(G, device=dev), last]).float().repeat_interleave(n, 0)}
        del kv  # Kp / Vp hold contiguous copies
        if script:  # per row (G, n, ...): position of the first response token, its replies, calls made, reply progress
            rows = lambda x: x.repeat_interleave(n, 0).view(G, n, *x.shape[1:]).contiguous()
            zero = lambda: torch.zeros(G, n, dtype=torch.long, device=dev)
            st.update(pos=rows(script["plen"]), env=rows(script["env"]), env_len=rows(script["env_len"]),
                      turns=rows(script["turns"]), calls=zero(), fpos=zero(), fleft=zero(), count=zero(),
                      pmask=(torch.arange(Lp, device=dev) >= script["plen"][:, None])[:, None, None, :])
        out = {"tok": torch.full((B, self.R), self.pad, device=dev), "logp": torch.zeros(B, self.R, device=dev),
               "ent": torch.zeros(B, self.R, device=dev)}
        step = self._script_step if script else self._step
        queue = [(st, 0)]  # layouts to decode and the step each resumes from
        while queue:
            st, t = queue.pop()
            self._to(st, dev)
            while t < self.R and not bool(st["done"].all()):
                t = self._segment(w, st, out, t, Lp, step, queue)
        return out["tok"], out["logp"], out["ent"]

    def _segment(self, w, st, out, t, Lp, step, queue):
        """Decode one segment of the layout st from step t (compacted, its caches grown to cover the segment, groups
        that would not fit moved to the queue); returns the step it reached."""
        self._compact(st)
        W = (t // SEG + 1) * SEG  # the segment's end: response slots [0, W) cover its steps
        if W > st["width"]:
            width = min(self.slots, st["width"] + GROW)
            g = self._fit(Lp, st["n"], width, st["G"])
            if g < st["G"]:
                queue.append((self._split(st, g), t))
            self._grow(st, width)
        self.side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side):
            # the segment's first step eagerly, on the capture stream: compiles the kernels for the segment's shapes
            # (compaction can make a dimension 1, which recompiles) and warms up workspaces outside the capture
            step(w, st, out, W, Lp)
            t += 1
            graph = torch.cuda.CUDAGraph()  # raw capture: torch.cuda.graph() would gc + empty_cache per segment
            graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")
            step(w, st, out, W, Lp)
            graph.capture_end()
        torch.cuda.current_stream().wait_stream(self.side)
        self.graph = graph
        while t < min(W, self.R):
            graph.replay()
            t += 1
            if t % 8 == 0 and bool(st["done"].all()):
                break
        return t

    def _fit(self, Lp, n, width, G):
        """How many of G groups' caches (prompt and n rows of width response slots) fit in mem_gb (at least 1)."""
        return min(G, max(1, int(self.mem // (self.kv_bytes * (Lp + n * width)))))

    def _grow(self, st, width):
        """Widen the layout's response caches to width slots (a layer at a time)."""
        for key in ("Kr", "Vr"):
            for i, x in enumerate(st[key]):
                y = x.new_zeros(*x.shape[:3], width, x.shape[4])
                y[..., :x.shape[3], :] = x
                st[key][i] = y
        st["width"] = width

    def _split(self, st, g):
        """Move the layout's groups from g on into a new layout (its caches in host memory); st keeps the first g."""
        n, out = st["n"], {"G": st["G"] - g, "n": st["n"], "width": st["width"], "t": st["t"].clone()}
        for key, x in st.items():
            if key in ("Kp", "Vp", "Kr", "Vr"):  # a layer at a time: each old one is freed before the next copy
                out[key] = []
                for i, y in enumerate(x):
                    out[key].append(y[g:].cpu())
                    x[i] = y[:g].clone()
            elif key == "logits":
                out[key], st[key] = x[g * n:], x[:g * n]
            elif torch.is_tensor(x) and key != "t":  # (G, n, ...) row state, pmask (G, ...)
                out[key], st[key] = x[g:], x[:g]
        st["G"] = g
        return out

    def _to(self, st, dev):
        for key in ("Kp", "Vp", "Kr", "Vr"):
            st[key] = [x.to(dev) for x in st[key]]

    def _step(self, w, st, out, W, Lp):
        """Sample every row's token at slot t from the current logits, record it, then compute the logits after it."""
        t, rows, done = st["t"], st["rows"].view(-1), st["done"].view(-1)
        tok, logp, ent = sample(st["logits"], torch.empty_like(st["logits"]).exponential_(), done, self.pad)
        idx = (rows, t.expand_as(rows))
        out["tok"].index_put_(idx, tok)
        out["logp"].index_put_(idx, logp)
        out["ent"].index_put_(idx, ent)
        done |= tok == self.eos
        hidden = self.m.forward(w, tok[:, None], Lp + t, self._attend(st, t, W))
        st["logits"].copy_(self.m.logits(w, hidden[:, 0]).float())
        t += 1

    def _script_step(self, w, st, out, W, Lp):
        """_step for scripted rows: a row inside a reply takes its next reply token instead of a sample (every state
        update is in place: the step is replayed as a graph)."""
        t, rows, done = st["t"], st["rows"].view(-1), st["done"].view(-1)
        pos, env, env_len, turns, calls, fpos, fleft, count = (st[k].flatten(0, 1) for k in SCRIPT)
        r, K, F = torch.arange(len(rows), device=t.device), env.shape[1], env.shape[2]
        forcing = ~done & (fleft > 0)
        forced = env[r, (calls - 1).clamp(0, K - 1), fpos.clamp(max=F - 1)]  # the reply being fed
        done |= ~forcing & ((count >= self.budget) | (pos + t >= self.max_ctx))
        tok, logp, ent = sample(st["logits"], torch.empty_like(st["logits"]).exponential_(), done | forcing, self.pad)
        live = ~done & ~forcing & (tok != self.pad)
        done |= ~forcing & (tok == self.pad)
        call = live & (tok == self.call)
        calls += call
        last = call & (calls >= turns)
        inject = call & ~last
        fleft.copy_(torch.where(inject, env_len[r, (calls - 1).clamp(0, K - 1)], fleft - forcing.long()))  # the next reply
        fpos.copy_(torch.where(inject, 0, fpos + forcing.long()))
        count += live
        done |= (live & (tok == self.eos)) | last
        idx = (rows, t.expand_as(rows))
        out["tok"].index_put_(idx, torch.where(forcing, forced + ENV, tok))  # tok is the pad where not live
        out["logp"].index_put_(idx, torch.where(live, logp, 0.0))
        out["ent"].index_put_(idx, torch.where(live, ent, 0.0))
        hidden = self.m.forward(w, torch.where(forcing, forced, tok)[:, None], (pos + t)[:, None], self._attend(st, t, W))
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
            qp = qg.reshape(G, m.kvh, n * m.rep, m.d)
            if "pmask" in st:  # right-padded prompts: the scores of each group's padding masked out
                sp = (qp @ Kp.transpose(-1, -2)).float().masked_fill(st["pmask"], float("-inf"))
                lse_p = sp.logsumexp(-1)
                op = (sp - lse_p[..., None]).exp().to(Vp.dtype) @ Vp
            else:  # prompt part: a group's n * rep queries against its prompt keys, fused, with the log-sum-exp of the scores
                op, lse_p = torch.ops.aten._scaled_dot_product_flash_attention(qp, Kp, Vp, scale=1.0)[:2]
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
        for key in ("Kr", "Vr"):  # (G, kvh, n, slots, d), a layer at a time (each old one is freed before the next copy)
            for i, x in enumerate(st[key]):
                st[key][i] = x.transpose(1, 2)[gi, sel].transpose(1, 2).contiguous()
        if "pmask" in st:
            st["pmask"] = st["pmask"][keep]
            st.update({k: st[k][gi, sel] for k in SCRIPT})
        st["G"], st["n"] = keep.numel(), n_new


def response_mask(resp):
    """1 up to and including the first EOS (verl get_response_mask)."""
    e = (resp == EOS).int()
    return (e.cumsum(1) - e) == 0


def half(weights):
    return {k: p.detach().half() for k, p in weights.items()}


@torch.no_grad()
def judged(task, decoder, w, prompts, parsed, n, rows=16384):
    """Every judge outcome of n samples of each prompt ((len(prompts) n,) each; prompts parsed by task.parse, whose
    "script", if any, drives the decoder); prompts are sampled and judged about `rows` samples at a time (the judges'
    intermediates grow with the number of samples)."""
    outs, c, script = [], max(1, rows // n), parsed.get("script")
    for i in range(0, len(prompts), c):
        kw = {} if script is None else {"script": {k: v[i:i + c] for k, v in script.items()}}
        tok = decoder.generate(w, prompts[i:i + c], n, **kw)[0]
        idx = torch.arange(i, i + len(tok) // n, device=prompts.device).repeat_interleave(n)
        outs.append(task.judge(tok, parsed, idx))
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def evaluate(task, decoder, w, prompts, parsed, n, rows=16384):
    """Mean of every judge outcome over n samples of each prompt (see judged)."""
    return {k: v.mean().item() for k, v in judged(task, decoder, w, prompts, parsed, n, rows).items()}
