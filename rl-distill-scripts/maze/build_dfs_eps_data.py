#!/usr/bin/env python3
"""Data for the depth-first, epsilon-noisy maze study (pretraining on noisy demonstrations, RL on a shortest-path reward).

Mazes: --size x --size (default 17x17) depth-first (recursive backtracker) mazes, then a fraction k ~ U(--k-min, --k-max) of the inter-cell walls
knocked out (TailRL's knockout_walls), so loops and tied shortest paths appear as k grows. Every split draws from this one
distribution; maze i comes from random.Random(--seed-base + i) and the splits are disjoint seed ranges (no grid repeats
across splits, checked).

Demonstrations (one per maze): an epsilon-noisy shortest-path solver. At every step it takes a move along a shortest path
to the goal; with probability --eps it instead takes a uniformly random move. Both choose only among moves that never
revisit a cell and still leave the goal reachable, so every demonstration reaches the goal.

Outputs under --out (the item / row formats of TailRL's to_sft_json.py and to_rl_parquet.py, data_source maze_17):
  sft/train.json        pretraining items, one noisy demonstration per maze (--n-train mazes)
  sft/test.json         256 held-out mazes with noisy demonstrations (pretraining validation loss)
  main_parquet/train.parquet   RL prompts: the pretraining mazes (ground truth = one shortest path; the judge only uses its grid)
  eval1000/test.parquet        1000 dev mazes: screening, level selection, base confirmation
  eval500/test.parquet         the first 500 dev mazes: RL validation
  test1000/test.parquet        1000 further held-out mazes for final numbers
  stats.json            optimal share and path-length stats of the demonstrations

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/build_dfs_eps_data.py --out /tmp/maze_data/dfs_eps05 --eps 0.5
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import deque
from multiprocessing import Pool

import numpy as np

SIZE = 17
START, GOAL = (1, 1), (SIZE - 2, SIZE - 2)
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))   # UP, DOWN, LEFT, RIGHT = action ids 0..3
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")
BD = None   # TailRL's build_dataset module (knockout_walls), set in each worker
CFG = None


def gen_dfs_maze(rng):
    g = np.ones((SIZE, SIZE), dtype=np.int8)
    g[START] = 0
    stack = [START]
    while stack:
        r, c = stack[-1]
        nb = [(r + dr, c + dc, r + dr // 2, c + dc // 2) for dr, dc in ((0, 2), (2, 0), (0, -2), (-2, 0))
              if 0 < r + dr < SIZE - 1 and 0 < c + dc < SIZE - 1 and g[r + dr, c + dc] == 1]
        if not nb:
            stack.pop()
            continue
        nr, nc, mr, mc = rng.choice(nb)
        g[mr, mc] = 0
        g[nr, nc] = 0
        stack.append((nr, nc))
    g[GOAL] = 0
    return g


def dist_to_goal(g, blocked):
    """BFS distance to the goal avoiding the blocked cells (-1 = goal unreachable)."""
    d = np.full(g.shape, -1)
    d[GOAL] = 0
    q = deque([GOAL])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES:
            p = (r + dr, c + dc)
            if g[p] == 0 and d[p] < 0 and p not in blocked:
                d[p] = d[r, c] + 1
                q.append(p)
    return d


def eps_path(g, eps, rng):
    pos, visited, actions = START, {START}, []
    while pos != GOAL:
        d = dist_to_goal(g, visited)
        opts = [(a, (pos[0] + dr, pos[1] + dc)) for a, (dr, dc) in enumerate(MOVES)
                if g[pos[0] + dr, pos[1] + dc] == 0 and d[pos[0] + dr, pos[1] + dc] >= 0]
        best = min(d[p] for _, p in opts)
        pool = opts if rng.random() < eps else [o for o in opts if d[o[1]] == best]
        a, pos = rng.choice(pool)
        visited.add(pos)
        actions.append(a)
    return actions


def shortest_actions(g):
    d = dist_to_goal(g, set())
    pos, actions = START, []
    while pos != GOAL:
        a, pos = next((a, (pos[0] + dr, pos[1] + dc)) for a, (dr, dc) in enumerate(MOVES)
                      if d[pos[0] + dr, pos[1] + dc] == d[pos] - 1 and g[pos[0] + dr, pos[1] + dc] == 0)
        actions.append(a)
    return actions, int(d[START])


def grid_tokens(g):
    toks = []
    for r in range(SIZE):
        for c in range(SIZE):
            toks.append("START" if (r, c) == START else "GOAL" if (r, c) == GOAL else "WALL" if g[r, c] else "PATH")
        toks.append("NEWLINE")
    return toks


def set_size(size):
    global SIZE, GOAL
    SIZE, GOAL = size, (size - 2, size - 2)


def init_worker(tailrl_src, cfg):
    global BD, CFG
    set_size(cfg["size"])
    sys.path.insert(0, tailrl_src)
    import build_dataset
    BD, CFG = build_dataset, cfg


def build(seed):
    """One maze: (seed, grid bytes, k, L*, shortest actions, noisy actions or None)."""
    rng = random.Random(seed)
    g = gen_dfs_maze(rng)
    k = rng.uniform(CFG["k_min"], CFG["k_max"])
    BD.knockout_walls(g, k, rng)
    short, L_star = shortest_actions(g)
    noisy = eps_path(g, CFG["eps"], random.Random(f"eps-{seed}")) if CFG["demos"] else None
    return seed, g.tobytes(), k, L_star, short, noisy


def sequence(g, actions):
    return " ".join(["<bos>", "GRID_START", *grid_tokens(g), "GRID_END", "PATH_START",
                     *(ACTION_NAMES[a] for a in actions), "DONE", "<eos>"])


def sft_item(m):
    seed, gb, k, L_star, _, noisy = m
    g = np.frombuffer(gb, dtype=np.int8).reshape(SIZE, SIZE)
    return {"sequence": sequence(g, noisy), "optimal_path_length": L_star, "path_length": len(noisy),
            "reward_continuous": float(len(noisy) == L_star), "prompt_id": seed, "sample_id": 0, "ub": 0, "k_frac": k}


def rl_row(m, idx):
    seed, gb, k, L_star, short, _ = m
    seq = sequence(np.frombuffer(gb, dtype=np.int8).reshape(SIZE, SIZE), short)
    prompt = seq[: seq.find("PATH_START") + len("PATH_START")]
    return {"data_source": "maze_17", "prompt": [{"role": "user", "content": prompt}],
            "reward_model": {"style": "rule", "ground_truth": seq},
            "extra_info": {"answer": seq, "question": prompt, "optimal_path_length": L_star, "index": idx, "prompt_id": seed}}


def write_parquet(path, rows):
    """Small row groups: one multi-GB row group of the nested prompt column cannot be read back by pyarrow / HF datasets
    ("Nested data conversions not implemented for chunked array outputs"; the 25x25 RL train parquet hit this)."""
    import pandas as pd
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, row_group_size=20000)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=17, help="maze side (odd); 17 = the original task")
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--k-min", type=float, default=0.0)
    ap.add_argument("--k-max", type=float, default=0.20)
    ap.add_argument("--n-train", type=int, default=1_300_000)
    ap.add_argument("--seed-base", type=int, default=50_000_000, help="maze seeds; disjoint from TailRL's Prim datasets (0..1.3M)")
    ap.add_argument("--tailrl-src", default=os.path.expanduser("~/TailRL/experiments/text_maze/src"))
    ap.add_argument("--workers", type=int, default=96)
    args = ap.parse_args()

    # seed ranges: dev 1000 | test 1000 | sft val 256 | train n
    b = args.seed_base
    splits = {"dev": range(b, b + 1000), "test": range(b + 1000, b + 2000), "val": range(b + 2000, b + 2256),
              "train": range(b + 10_000, b + 10_000 + args.n_train)}
    set_size(args.size)
    cfg = {"size": args.size, "eps": args.eps, "k_min": args.k_min, "k_max": args.k_max}
    t0 = time.time()
    out, seen = {}, {}
    for name, seeds in splits.items():
        with Pool(args.workers, initializer=init_worker, initargs=(args.tailrl_src, {**cfg, "demos": name in ("train", "val")})) as pool:
            out[name] = []
            for i, m in enumerate(pool.imap(build, seeds, chunksize=256)):
                if m[1] in seen:   # a repeated grid stays only in the split that drew it first
                    continue
                seen[m[1]] = name
                out[name].append(m)
                if (i + 1) % 100_000 == 0:
                    print(f"  {name} {i + 1}/{len(seeds)} ({time.time() - t0:.0f}s)", flush=True)
        print(f"{name}: {len(out[name])} mazes ({len(seeds) - len(out[name])} repeats dropped) {time.time() - t0:.0f}s", flush=True)

    for name, path in (("train", "sft/train.json"), ("val", "sft/test.json")):
        os.makedirs(os.path.join(args.out, "sft"), exist_ok=True)
        with open(os.path.join(args.out, path), "w") as f:
            json.dump([sft_item(m) for m in out[name]], f)
    write_parquet(os.path.join(args.out, "main_parquet/train.parquet"), [rl_row(m, i) for i, m in enumerate(out["train"])])
    dev = [rl_row(m, i) for i, m in enumerate(out["dev"])]
    write_parquet(os.path.join(args.out, "eval1000/test.parquet"), dev)
    write_parquet(os.path.join(args.out, "eval500/test.parquet"), dev[:500])
    write_parquet(os.path.join(args.out, "test1000/test.parquet"), [rl_row(m, i) for i, m in enumerate(out["test"])])

    extra = [len(m[5]) - m[3] for m in out["train"]]
    by_k = {f"{lo:.2f}-{lo + 0.05:.2f}": [e == 0 for (m, e) in zip(out["train"], extra) if lo <= m[2] < lo + 0.05]
            for lo in (0.0, 0.05, 0.10, 0.15)}
    s = sorted(extra)
    stats = {"size": args.size, "eps": args.eps, "k_range": [args.k_min, args.k_max], "seed_base": b, "n": {k: len(v) for k, v in out.items()},
             "optimal_share": sum(e == 0 for e in extra) / len(extra), "mean_extra": statistics.mean(extra),
             "p90_extra": s[int(0.9 * len(s))], "p99_extra": s[int(0.99 * len(s))],
             "max_path_len": max(len(m[5]) for m in out["train"]), "mean_L_star": statistics.mean(m[3] for m in out["train"]),
             "optimal_share_by_k": {k: sum(v) / len(v) for k, v in by_k.items()}}
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1), f"\ndone in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
