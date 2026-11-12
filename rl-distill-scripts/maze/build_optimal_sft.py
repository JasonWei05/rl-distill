#!/usr/bin/env python3
"""Shortest-path-only SFT data for the text-maze pretraining (shortest-path task).

Same mazes, test split and item format as TailRL's src/to_sft_json.py (--seed picks the --test-prompts held-out mazes
exactly as it does), but every target -- train and test -- is a BFS-optimal path, drawn uniformly at random among all
shortest paths of that maze (count the shortest paths into every cell, then walk back from the goal choosing each
predecessor with probability proportional to its count). The draw is seeded per maze, so the data is reproducible.

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/build_optimal_sft.py --in-path ~/tailrl_maze_data/main_1.3M.jsonl \
        --out-train ~/tailrl_maze_data/sft_optimal/train_optimal.json --out-test ~/tailrl_maze_data/sft_optimal/test.json
"""
import argparse
import json
import os
import sys
import time
from collections import deque
from multiprocessing import Pool

import numpy as np

MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))   # UP, DOWN, LEFT, RIGHT = action ids 0..3 (to_sft_json.ACTION_NAMES)
TS = None   # TailRL's to_sft_json module (grid_to_tokens, make_item, START, GOAL), set in each worker


def random_shortest_path(grid, start, goal, rng):
    """Uniformly random shortest path from start to goal on a 0 = open / 1 = wall grid, as action ids."""
    h, w = len(grid), len(grid[0])
    dist = [[-1] * w for _ in range(h)]
    count = [[0] * w for _ in range(h)]
    dist[start[0]][start[1]] = 0
    count[start[0]][start[1]] = 1
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0:
                if dist[nr][nc] == -1:
                    dist[nr][nc] = dist[r][c] + 1
                    q.append((nr, nc))
                if dist[nr][nc] == dist[r][c] + 1:
                    count[nr][nc] += count[r][c]
    if dist[goal[0]][goal[1]] < 0:
        raise ValueError("goal unreachable")
    actions = []
    r, c = goal
    while (r, c) != tuple(start):
        preds, weights = [], []
        for a, (dr, dc) in enumerate(MOVES):   # predecessor p with p + move(a) == (r, c)
            pr, pc = r - dr, c - dc
            if 0 <= pr < h and 0 <= pc < w and dist[pr][pc] == dist[r][c] - 1 and count[pr][pc] > 0:
                preds.append((pr, pc, a))
                weights.append(count[pr][pc])
        pr, pc, a = preds[rng.choice(len(preds), p=np.asarray(weights, dtype=float) / sum(weights))]
        actions.append(a)
        r, c = pr, pc
    return actions[::-1], dist[goal[0]][goal[1]]


def init_worker(tailrl_src):
    global TS
    sys.path.insert(0, tailrl_src)
    import to_sft_json
    TS = to_sft_json


def convert(args):
    line, seed = args
    obj = json.loads(line)
    rng = np.random.default_rng([seed, obj["prompt_id"]])
    actions, L = random_shortest_path(obj["grid"], TS.START, TS.GOAL, rng)
    assert L == obj["L_star"], (obj["prompt_id"], L, obj["L_star"])
    sample = {"actions": actions, "L": L, "reward_continuous": 1.0, "sample_id": -1}   # -1: not one of the 16 harvested
    return obj["prompt_id"], TS.make_item(TS.grid_to_tokens(obj["grid"]), obj, sample)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-path", required=True, help="JSONL from TailRL src/build_dataset.py (main_1.3M.jsonl)")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-test", required=True)
    ap.add_argument("--test-prompts", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0, help="test split (as to_sft_json.py) and per-maze path draws")
    ap.add_argument("--tailrl-src", default=os.path.expanduser("~/TailRL/experiments/text_maze/src"))
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    with open(args.in_path) as f:
        lines = [ln for ln in (l.strip() for l in f) if ln]
    prompt_ids = [json.loads(ln)["prompt_id"] for ln in lines]
    shuffled = list(prompt_ids)
    np.random.default_rng(args.seed).shuffle(shuffled)   # to_sft_json.py: same rng, first use = the test split
    test_set = set(shuffled[: args.test_prompts])
    print(f"read {len(lines)} mazes in {time.time() - t0:.0f}s; {len(test_set)} held out", flush=True)

    train, test = [], []
    with Pool(args.workers, initializer=init_worker, initargs=(args.tailrl_src,)) as pool:
        for i, (pid, item) in enumerate(pool.imap(convert, ((ln, args.seed) for ln in lines), chunksize=512)):
            (test if pid in test_set else train).append(item)
            if (i + 1) % 200000 == 0:
                print(f"  {i + 1} mazes ({time.time() - t0:.0f}s)", flush=True)
    for path, items in ((args.out_train, train), (args.out_test, test)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(items, f)
    print(f"wrote train={len(train)} test={len(test)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
