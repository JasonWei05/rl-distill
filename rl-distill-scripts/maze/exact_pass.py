#!/usr/bin/env python3
"""Exact shortest-path pass@k of a maze checkpoint, without sampling.

The judge scores a response as shortest iff its actions (everything before the first DONE) reach the goal in L* steps, i.e.
the first L* actions are one of the maze's shortest paths (moves after the goal are ignored) and a DONE follows. Shortest paths
are few (17x17 depth-first 0-20% dev mazes: 3.5 per maze on average, 25x25: 14), so per maze

    p_hi = sum over shortest paths of P(path | prompt)              (teacher-forced, one forward pass per path)
    p_lo = sum over shortest paths of P(path | prompt) * P(DONE | prompt, path)

bracket the probability that one temperature-1 sample is shortest (the gap is the mass of continuing with more moves before
DONE, or stopping without DONE). pass@1 = mean p; pass@k = mean over mazes of 1 - (1 - p)^k, exactly (samples are
independent). Mazes with more than --max-paths shortest paths are scored on the --max-paths most likely ones found by a
best-first expansion (a lower bound; flagged in the output).

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/exact_pass.py --ckpt <dir> [--parquet <dev test.parquet>]
"""
import argparse
import heapq
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import screen_sft_ckpts as ss  # noqa: E402

MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
KS = [2 ** i for i in range(11)]


def grid_of(seq):
    toks = seq.split()
    i, j = toks.index("GRID_START"), toks.index("GRID_END")
    rows = " ".join(toks[i + 1:j]).split(" NEWLINE")
    return np.array([[1 if t == "WALL" else 0 for t in r.split()] for r in rows if r.strip()], dtype=np.int8)


def bfs(g, src):
    d = np.full(g.shape, -1)
    d[src] = 0
    q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES.values():
            p = (r + dr, c + dc)
            if g[p] == 0 and d[p] < 0:
                d[p] = d[r, c] + 1
                q.append(p)
    return d


def shortest_paths(g, cap):
    """All shortest start->goal paths as move lists (None if there are more than cap)."""
    n = g.shape[0]
    S, T = (1, 1), (n - 2, n - 2)
    dt = bfs(g, T)
    out = []

    def rec(pos, acc):
        if len(out) > cap:
            return
        if pos == T:
            out.append(list(acc))
            return
        for name, (dr, dc) in MOVES.items():
            p = (pos[0] + dr, pos[1] + dc)
            if g[p] == 0 and dt[p] == dt[pos] - 1:
                acc.append(name)
                rec(p, acc)
                acc.pop()
    rec(S, [])
    return out if len(out) <= cap else None


class Scorer:
    def __init__(self, ckpt, device, batch_tokens):
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32).to(device).eval()
        self.device, self.batch_tokens = device, batch_tokens
        self.ids = {w: self.tok.convert_tokens_to_ids(w) for w in list(MOVES) + ["DONE"]}

    @torch.no_grad()
    def logprobs(self, prompt_ids, conts):
        """For each continuation (list of words): (sum log P(words), log P(DONE | prompt, words))."""
        res = []
        seqs = [prompt_ids + [self.ids[w] for w in c] for c in conts]
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        bs = max(1, self.batch_tokens // max(len(s) for s in seqs))
        out = [None] * len(seqs)
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            L = max(len(seqs[i]) for i in idx)
            x = torch.zeros(len(idx), L, dtype=torch.long)
            m = torch.zeros(len(idx), L, dtype=torch.long)
            for r, i in enumerate(idx):
                x[r, :len(seqs[i])] = torch.tensor(seqs[i])
                m[r, :len(seqs[i])] = 1
            lp = torch.log_softmax(self.model(input_ids=x.to(self.device), attention_mask=m.to(self.device)).logits.float(), -1)
            for r, i in enumerate(idx):
                n0, n1 = len(prompt_ids), len(seqs[i])
                tgt = torch.tensor(seqs[i][n0:n1], device=self.device)
                s = lp[r, n0 - 1:n1 - 1].gather(1, tgt[:, None]).sum().item()
                out[i] = (s, lp[r, n1 - 1, self.ids["DONE"]].item())
        return out

    def best_first(self, prompt_ids, g, cap):
        """The cap most likely shortest paths, expanding prefixes by model probability (for mazes with too many)."""
        n = g.shape[0]
        T = (n - 2, n - 2)
        dt = bfs(g, T)
        heap, found = [(0.0, [], (1, 1))], []
        while heap and len(found) < cap:
            nlp, acc, pos = heapq.heappop(heap)
            if pos == T:
                found.append(acc)
                continue
            kids = [(name, (pos[0] + dr, pos[1] + dc)) for name, (dr, dc) in MOVES.items()
                    if g[pos[0] + dr, pos[1] + dc] == 0 and dt[pos[0] + dr, pos[1] + dc] == dt[pos] - 1]
            lps = self.logprobs(prompt_ids, [acc + [k] for k, _ in kids])
            for (name, p), (s, _) in zip(kids, lps):
                heapq.heappush(heap, (-s, acc + [name], p))
        return found


def exact_screen(ckpt, prompts, gts, device="cuda", max_paths=500, batch_tokens=262144):
    t0 = time.time()
    sc = Scorer(ckpt, device, batch_tokens)
    p_hi, p_lo, truncated = [], [], 0
    for prompt, gt in zip(prompts, gts):
        g = grid_of(gt)
        pids = sc.tok.encode(prompt, add_special_tokens=False)
        paths = shortest_paths(g, max_paths)
        if paths is None:
            paths, truncated = sc.best_first(pids, g, max_paths), truncated + 1
        lps = sc.logprobs(pids, paths)
        p_hi.append(sum(math.exp(s) for s, _ in lps))
        p_lo.append(sum(math.exp(s + d) for s, d in lps))
    res = {"seconds": round(time.time() - t0, 1), "truncated_mazes": truncated, "per_maze": {"p_hi": p_hi, "p_lo": p_lo}}
    for name, p in (("hi", p_hi), ("lo", p_lo)):
        p = np.asarray(p)
        res[f"pass@1_{name}"] = float(p.mean())
        res[f"pass@k_{name}"] = {k: float(np.mean(1 - (1 - p) ** k)) for k in KS}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--parquet", default=None, help="default: dev_parquet()")
    ap.add_argument("--tailrl-root", default=ss.DEFAULT_TAILRL)
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--max-paths", type=int, default=500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-tokens", type=int, default=262144, help="tokens per forward batch (lower on a shared GPU)")
    args = ap.parse_args()
    prompts, gts = ss.load_prompts(args.parquet or ss.dev_parquet(args.tailrl_root), args.n_prompts)
    r = exact_screen(args.ckpt, prompts, gts, max_paths=args.max_paths, batch_tokens=args.batch_tokens)
    print(json.dumps({k: v for k, v in r.items() if k != "per_maze"}))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(r, f)


if __name__ == "__main__":
    main()
