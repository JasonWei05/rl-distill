"""The tasks: word-level vocabularies, example generation (CPU) and the reward (GPU).

A task turns an integer seed into one example: the prompt token ids (a fixed length per task), the demonstration's
response token ids (ending in DONE <eos>) and a key (examples with equal keys are the same puzzle). On the GPU it parses
a batch of prompts once (`parse`) and scores responses against them (`judge`: float (B,) tensors, "score" plus one per
name in OUTCOMES). Token ids are positions in WORDS; the first four are <pad> <bos> <eos> <unk> in the synthetic tasks
(Maze, Sudoku). Chess (the released pre2post-chess testbed) keeps its own vocabulary and multi-turn layout (see Chess).
"""

import io
import json
import os
import random
import re

import torch

SPECIAL = ("<pad>", "<bos>", "<eos>", "<unk>")
PAD, BOS, EOS = 0, 1, 2
ENV = 128  # a response token id >= ENV is an environment token (id - ENV): fed to the model, never sampled or trained


def first_true(m):
    """Index of the first True per row of m (B, R); R where there is none."""
    return torch.where(m.any(1), m.int().argmax(1), m.shape[1])


class Maze:
    """S x S grid maze (S odd), start (1, 1), goal (S-2, S-2). A depth-first (recursive backtracker) maze on the odd cells,
    then a fraction k ~ U(k_min, k_max) of the inter-cell walls knocked out, so loops and tied shortest paths appear.

    Prompt: <bos> GRID_START (S rows of S cells (START GOAL WALL PATH) and NEWLINE) GRID_END PATH_START.
    Demonstration: an eps-noisy shortest-path walk. Every step moves along a shortest path to the goal, or with probability
    eps takes a uniformly random move; both choose among moves that never revisit a cell and leave the goal reachable.
    Response: the moves (UP DOWN LEFT RIGHT), DONE, <eos>.
    Reward (TailRL's judge_maze_binary_shortest): the words of decode(response, skip_special_tokens=True) before the first
    DONE must all be moves; walked from the start they must reach the goal, without hitting a wall first, in L* moves.

    The token layout and generator reproduce rl-distill-scripts/maze/data/build_dfs_eps_data.py (the 17x17 / 25x25 data)."""

    WORDS = SPECIAL + ("GRID_START", "GRID_END", "PATH_START", "DONE", "PATH", "WALL", "GOAL", "START", "NEWLINE",
                       "UP", "DOWN", "LEFT", "RIGHT", "\n") + tuple(f"RESERVED_{i}" for i in range(14))
    OUTCOMES = ("goal_reached", "wall_collision", "valid_format", "done_token_generated", "valid_path_no_goal", "is_shortest")
    DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # UP DOWN LEFT RIGHT

    def __init__(self, size, eps, k_min=0.0, k_max=0.2):
        assert size % 2 == 1 and size >= 5, size
        self.S, self.eps, self.k_range = size, eps, (k_min, k_max)
        self.ids = {w: i for i, w in enumerate(self.WORDS)}
        self.prompt_len = size * (size + 1) + 4
        self.start, self.goal = size + 1, (size - 2) * size + size - 2  # flat cell indices
        self.steps = (-size, size, -1, 1)

    # ---- generation (CPU); the grid is a flat bytearray, 1 = wall; the border is all wall, so no bounds checks
    def example(self, seed):
        rng = random.Random(seed)
        wall = self._dfs(rng)
        self._knockout(wall, rng.uniform(*self.k_range), rng)
        moves, lstar = self._walk(wall, random.Random(f"eps-{seed}"))
        v, S = self.ids, self.S
        cell = lambda i: v["START"] if i == self.start else v["GOAL"] if i == self.goal else v["WALL"] if wall[i] else v["PATH"]
        prompt = [v["<bos>"], v["GRID_START"]]
        for r in range(S):
            prompt += [cell(r * S + c) for c in range(S)] + [v["NEWLINE"]]
        prompt += [v["GRID_END"], v["PATH_START"]]
        response = [v["UP"] + a for a in moves] + [v["DONE"], EOS]
        return bytes(wall), prompt, response, {"lstar": lstar, "moves": len(moves), "optimal": len(moves) == lstar}

    def _dfs(self, rng):
        S = self.S
        wall = bytearray(b"\x01") * (S * S)
        wall[self.start] = 0
        stack = [(1, 1)]
        while stack:
            r, c = stack[-1]
            nb = [(r + dr, c + dc, r + dr // 2, c + dc // 2) for dr, dc in ((0, 2), (2, 0), (0, -2), (-2, 0))
                  if 0 < r + dr < S - 1 and 0 < c + dc < S - 1 and wall[(r + dr) * S + c + dc]]
            if not nb:
                stack.pop()
                continue
            nr, nc, mr, mc = rng.choice(nb)
            wall[mr * S + mc] = wall[nr * S + nc] = 0
            stack.append((nr, nc))
        return wall

    def _knockout(self, wall, k, rng):
        """TailRL build_dataset.knockout_walls: open a random round(k n) of the n inter-cell walls."""
        S = self.S
        cands = [r * S + c for r in range(1, S - 1) for c in range(1, S - 1) if wall[r * S + c] and (
            (not wall[(r - 1) * S + c] and not wall[(r + 1) * S + c]) or (not wall[r * S + c - 1] and not wall[r * S + c + 1]))]
        rng.shuffle(cands)
        for i in cands[:int(round(len(cands) * k))]:
            wall[i] = 0

    def _dist(self, wall, blocked):
        """BFS distance of every cell to the goal avoiding walls and blocked cells (-1: cannot reach it)."""
        d = [-1] * len(wall)
        d[self.goal] = 0
        queue = [self.goal]
        for i in queue:
            for st in self.steps:
                p = i + st
                if d[p] < 0 and not wall[p] and not blocked[p]:
                    d[p] = d[i] + 1
                    queue.append(p)
        return d

    def _walk(self, wall, rng):
        """The eps-noisy walk's move ids and the shortest path length L*."""
        pos, moves = self.start, []
        visited = bytearray(len(wall))
        visited[pos] = 1
        lstar = None
        while pos != self.goal:
            d = self._dist(wall, visited)
            opts = [(a, pos + st) for a, st in enumerate(self.steps) if d[pos + st] >= 0]
            best = min(d[p] for _, p in opts)
            lstar = best + 1 if lstar is None else lstar
            pool = opts if rng.random() < self.eps else [o for o in opts if d[o[1]] == best]
            a, pos = rng.choice(pool)
            visited[pos] = 1
            moves.append(a)
        return moves, lstar

    # ---- reward (GPU)
    def parse(self, prompts):
        """Open cells (B, S, S), start and goal (B, 2) and the BFS shortest path length L* (B,) of prompts (B, prompt_len)."""
        B, L = prompts.shape
        S, v = self.S, self.ids
        assert L == self.prompt_len, (L, self.prompt_len)
        assert (prompts[:, :2] == torch.tensor([BOS, v["GRID_START"]], device=prompts.device)).all()
        assert (prompts[:, -2:] == torch.tensor([v["GRID_END"], v["PATH_START"]], device=prompts.device)).all()
        cells = prompts[:, 2:-2].reshape(B, S, S + 1)
        assert (cells[:, :, S] == v["NEWLINE"]).all()
        cells = cells[:, :, :S]
        assert torch.isin(cells, torch.tensor([v["WALL"], v["PATH"], v["START"], v["GOAL"]], device=cells.device)).all()
        assert ((cells == v["START"]).sum((1, 2)) == 1).all() and ((cells == v["GOAL"]).sum((1, 2)) == 1).all()
        flat = lambda token: (cells == token).flatten(1).int().argmax(1)
        start, goal = (torch.stack([i // S, i % S], 1) for i in (flat(v["START"]), flat(v["GOAL"])))
        free = cells != v["WALL"]
        return {"free": free, "start": start, "goal": goal, "lstar": self._bfs(free, start, goal)}

    def _bfs(self, free, start, goal):
        """Shortest path length start -> goal per maze (TailRL's _bfs_optimal_length), -1 when unreachable."""
        B, S = free.shape[0], self.S
        free = torch.nn.functional.pad(free, (1, 1, 1, 1))  # a closed border: shifts never wrap
        seen = torch.zeros_like(free)
        b = torch.arange(B, device=free.device)
        seen[b, start[:, 0] + 1, start[:, 1] + 1] = True
        dist = torch.full((B,), -1, device=free.device)
        for k in range(1, S * S):
            grow = seen.clone()
            grow[:, 1:] |= seen[:, :-1]
            grow[:, :-1] |= seen[:, 1:]
            grow[:, :, 1:] |= seen[:, :, :-1]
            grow[:, :, :-1] |= seen[:, :, 1:]
            seen = grow & free
            dist = torch.where((dist < 0) & seen[b, goal[:, 0] + 1, goal[:, 1] + 1], k, dist)
            if k % 16 == 0 and bool((dist >= 0).all()):
                break
        return dist

    def judge(self, resp, st, idx):
        """Responses (B, R) token ids, row i against parsed maze idx[i]."""
        v, dev, R = self.ids, resp.device, resp.shape[1]
        moves = torch.arange(v["UP"], v["RIGHT"] + 1, device=dev)
        skip = torch.tensor([PAD, BOS, EOS, v["<unk>"], v["\n"]], device=dev)  # vanish from decode(skip_special_tokens).split()
        delta = torch.zeros(len(self.WORDS), 2, dtype=torch.long, device=dev)
        delta[moves] = torch.tensor(self.DELTAS, device=dev)
        has_done = (resp == v["DONE"]).any(1)
        before = torch.arange(R, device=dev) < first_true(resp == v["DONE"])[:, None]
        move = before & torch.isin(resp, moves)
        valid = has_done & ~(before & ~move & ~torch.isin(resp, skip)).any(1)
        pos = st["start"][idx, None] + (delta[resp] * move[..., None]).cumsum(1)  # (B, R, 2) after each token
        inside = ((pos >= 0) & (pos < self.S)).all(-1)
        p = pos.clamp(0, self.S - 1)
        blocked = move & (~inside | ~st["free"][idx[:, None], p[..., 0], p[..., 1]])
        at_goal = move & ~blocked & (pos == st["goal"][idx, None]).all(-1)
        first_block, first_goal = first_true(blocked), first_true(at_goal)
        reached = valid & (first_goal < first_block)
        hit = valid & (first_block < first_goal)
        n_moves = move.cumsum(1).gather(1, first_goal.clamp(max=R - 1)[:, None]).squeeze(1)
        shortest = reached & (n_moves <= st["lstar"][idx])
        out = {"goal_reached": reached, "wall_collision": hit, "valid_format": valid, "done_token_generated": has_done,
               "valid_path_no_goal": valid & ~reached & ~hit, "is_shortest": shortest, "score": shortest}
        return {k: x.float() for k, x in out.items()}


# sudoku cells are flat indices r * 9 + c; UNITS: the 9 rows, 9 columns, 9 boxes; PEERS[i]: the 20 cells sharing a unit with i
UNITS = tuple(tuple(r * 9 + c for c in range(9)) for r in range(9)) + tuple(tuple(r * 9 + c for r in range(9)) for c in range(9)) \
    + tuple(tuple((br + i) * 9 + bc + j for i in range(3) for j in range(3)) for br in (0, 3, 6) for bc in (0, 3, 6))
PEERS = tuple(tuple(sorted({p for u in UNITS if i in u for p in u} - {i})) for i in range(81))


class Sudoku:
    """9 x 9 sudoku. A random solved grid (backtracking fill, digits tried in a random order per cell), then a uniform
    n ~ U{clue_min..clue_max} of its 81 cells kept as clues (puzzles with fewer clues can have several solutions; any
    valid completion counts).

    Prompt: <bos> GRID_START (9 rows of 9 cells (0 = blank, 1-9) and NEWLINE) GRID_END ANSWER_START.
    Demonstration (always a correct solution): a solver's fills in order. Each step fills the first naked single (raster
    order), else the first hidden single (rows, then columns, then boxes; digits ascending), else guesses on the first
    cell with the fewest candidates (digits ascending) and backtracks on a contradiction; the trace keeps only the fills of
    the branch that solves the puzzle (the solver of github sudoku-transfomer generate_data.solve_step_by_step).
    Response: R<row> C<col> <digit> per fill, DONE, <eos>.
    Reward: 1 iff the tokens before the first DONE are such triples, they fill every blank cell exactly once and no clue,
    and the completed grid is a valid sudoku."""

    WORDS = SPECIAL + ("GRID_START", "GRID_END", "ANSWER_START", "DONE", "NEWLINE") + tuple(str(d) for d in range(10)) \
        + tuple(f"R{i}" for i in range(1, 10)) + tuple(f"C{i}" for i in range(1, 10))
    OUTCOMES = ("valid_format", "done_token_generated", "overwrote_clue", "duplicate_fill", "complete", "solved")

    def __init__(self, clue_min, clue_max):
        assert 17 <= clue_min <= clue_max <= 80, (clue_min, clue_max)
        self.clues = (clue_min, clue_max)
        self.ids = {w: i for i, w in enumerate(self.WORDS)}
        self.prompt_len = 2 + 9 * 10 + 2
        self.digit0, self.row1, self.col1 = self.ids["0"], self.ids["R1"], self.ids["C1"]

    # ---- generation (CPU); grids are flat lists of 81 digits, 0 = blank
    def example(self, seed):
        rng = random.Random(seed)
        solution = self._fill([0] * 81, rng)
        keep = set(rng.sample(range(81), rng.randint(*self.clues)))
        puzzle = [d if i in keep else 0 for i, d in enumerate(solution)]
        trace, guesses = solve(puzzle)
        v = self.ids
        prompt = [BOS, v["GRID_START"]]
        for r in range(9):
            prompt += [self.digit0 + d for d in puzzle[r * 9:r * 9 + 9]] + [v["NEWLINE"]]
        prompt += [v["GRID_END"], v["ANSWER_START"]]
        response = [t for i, d in trace for t in (self.row1 + i // 9, self.col1 + i % 9, self.digit0 + d)] + [v["DONE"], EOS]
        return bytes(puzzle), prompt, response, {"clues": len(keep), "guesses": guesses}

    def _fill(self, grid, rng):
        """A random completion of grid (filled in place, raster order)."""
        if 0 not in grid:
            return grid
        i = grid.index(0)
        digits = list(range(1, 10))
        rng.shuffle(digits)
        used = {grid[p] for p in PEERS[i]}
        for d in digits:
            if d not in used:
                grid[i] = d
                if self._fill(grid, rng):
                    return grid
        grid[i] = 0
        return None

    # ---- reward (GPU)
    def parse(self, prompts):
        """Clue digits (B, 81) (0 = blank) of prompts (B, prompt_len)."""
        B, L = prompts.shape
        v = self.ids
        assert L == self.prompt_len, (L, self.prompt_len)
        assert (prompts[:, :2] == torch.tensor([BOS, v["GRID_START"]], device=prompts.device)).all()
        assert (prompts[:, -2:] == torch.tensor([v["GRID_END"], v["ANSWER_START"]], device=prompts.device)).all()
        cells = prompts[:, 2:-2].reshape(B, 9, 10)
        assert (cells[:, :, 9] == v["NEWLINE"]).all()
        clues = cells[:, :, :9].reshape(B, 81) - self.digit0
        assert ((clues >= 0) & (clues <= 9)).all()
        return {"clues": clues}

    def judge(self, resp, st, idx):
        """Responses (B, R) token ids, row i against parsed puzzle idx[i]."""
        dev, (B, R) = resp.device, resp.shape
        has_done = (resp == self.ids["DONE"]).any(1)
        n = first_true(resp == self.ids["DONE"])  # tokens before DONE
        pos = torch.arange(R, device=dev)
        lo = torch.tensor([self.row1, self.col1, self.digit0 + 1], device=dev)[pos % 3]  # expected token range per slot
        ok = (resp >= lo) & (resp < lo + 9)
        valid = has_done & (n % 3 == 0) & ~((pos < n[:, None]) & ~ok).any(1)
        T = R // 3
        trip = resp[:, :3 * T].view(B, T, 3)
        use = valid[:, None] & (torch.arange(T, device=dev) < (n // 3)[:, None])
        cell = torch.where(use, (trip[..., 0] - self.row1) * 9 + trip[..., 1] - self.col1, 81)  # 81: a discarded slot
        clues = st["clues"][idx]
        count = torch.zeros(B, 82, device=dev).scatter_add_(1, cell, use.float())[:, :81]
        grid = torch.cat([clues, torch.zeros(B, 1, dtype=clues.dtype, device=dev)], 1)
        grid = grid.scatter(1, cell, torch.where(use, trip[..., 2] - self.digit0, 0))[:, :81]
        overwrote = ((count > 0) & (clues > 0)).any(1)
        duplicate = (count > 1).any(1)
        complete = ((clues > 0) | (count > 0)).all(1)
        units = torch.tensor(UNITS, device=dev)
        per_unit = torch.nn.functional.one_hot(grid[:, units], 10).sum(2)  # (B, 27, 10) digit counts
        consistent = (per_unit[..., 1:] == 1).all(-1).all(-1)
        solved = valid & ~overwrote & ~duplicate & complete & consistent
        out = {"valid_format": valid, "done_token_generated": has_done, "overwrote_clue": valid & overwrote,
               "duplicate_fill": valid & duplicate, "complete": valid & complete, "solved": solved, "score": solved}
        return {k: x.float() for k, x in out.items()}


def solve(puzzle):
    """The solver's trace [(cell, digit), ...] for a flat puzzle (see Sudoku) and its number of guesses on the solving
    branch."""
    grid, trace = list(puzzle), []

    def candidates():
        return {i: set(range(1, 10)) - {grid[p] for p in PEERS[i]} for i in range(81) if grid[i] == 0}

    def single(cands):
        for i, c in cands.items():
            if len(c) == 1:
                return i, next(iter(c))
        for unit in UNITS:  # rows, columns, boxes
            for d in range(1, 10):
                places = [i for i in unit if d in cands.get(i, ())]
                if len(places) == 1:
                    return places[0], d
        return None

    def search():
        while True:
            cands = candidates()
            if not cands:
                return 0
            if not all(cands.values()):
                return None
            step = single(cands)
            if step is None:
                break
            grid[step[0]] = step[1]
            trace.append(step)
        i = min(cands, key=lambda j: len(cands[j]))
        saved, n = list(grid), len(trace)
        for d in sorted(cands[i]):
            grid[i] = d
            trace.append((i, d))
            guesses = search()
            if guesses is not None:
                return guesses + 1
            grid[:] = saved
            del trace[n:]
        return None

    guesses = search()
    assert guesses is not None, "unsolvable puzzle"
    return trace, guesses


FILES, RANKS = "abcdefgh", "12345678"


class Chess:
    """Lichess puzzles with search-trace CoT: the testbed of "Understanding Reasoning from Pretraining to Post-Training"
    (arXiv 2607.16097, github.com/pavelslab-nyu/pre2post-chess), with its released Qwen3 checkpoints, tokenizer
    (chess_hf/, copied from a released checkpoint; MIT) and data.

    Words (their 85): <bos> (also the pad) <eos> <unk>, pieces, the 64 squares, x = + # O-O O-O-O . ..., <T> </T> <sep>
    <call_env>. A move is its LAN pieces: piece, from, [x], to, [= piece], [+ or #] (castling: O-O or O-O-O, [+ or #]).
    A puzzle is the game so far (the opponent's trigger move included) and the solution line m1 o1 m2 ... mH, H <= 6
    solver moves. An answer is a trace <T> line <sep> line ... </T>, then m1 <call_env>, the environment's o1, m2
    <call_env>, ..., mH <call_env>: as in their verl rollout, the environment replies with the solution's next opponent
    move after every call whatever the move was, and the episode ends at the H-th call.

    Rows: the prompt slot is [the model's prompt | pad | the script, from column `slot`]. The script lists, per solver
    move, its UCI squares (and promotion piece), <sep>, the environment's reply tokens (none after the last), <call_env>.
      puzzle rows (splits test, dev, rl): the prompt is the game's LAN tokens and <T> (no <bos>: their RL and evaluation
        prompts); the response is a minimal correct answer, </T> m1 <call_env> o1 ... mH <call_env>. Prompts over 512
        tokens are dropped (their max_prompt_length filter).
      trace rows (splits train, val; their SFT data): the prompt is <bos>, the game and <T>; the response is the rest of
        the released trace (fully label-stripped) and <eos>. Opponent moves after a <call_env> are + ENV (their SFT
        masks them: MultiTurnSFTDataset._mask_env_responses).
    Splits: test, the released benchmark (B1-B5, 1480 of 1484 puzzles); dev and val, their SFT holdout (the last 2 of
    the first 85 released trace shards in name order: 937 puzzles, 933 under 512 prompt tokens) as puzzle and trace rows;
    train, the other 83 shards (32,858: their existing_sft puzzles); rl, their RL puzzles (rl_dataset_easy_skewed.csv,
    _source existing_rl: 62,207 of 62,290).
    Reward (reward_function_multiturn.compute_score_batch, on the decoded model and environment tokens): split at
    <call_env>, dropping the last part; each part's move is the first complete LAN move after </T> if it has one, else its
    last complete move; 1 iff there are H moves and each is the solution's (as UCI; castling always maps to the white
    squares, their bug: a black castling solution never scores)."""

    WORDS = ("<bos>", "<eos>", "<unk>", *"KQRBNP", *(f + r for f in FILES for r in RANKS), "x", "=", "+", "#", "O-O", "O-O-O",
             ".", "...", "<T>", "</T>", "<sep>", "<call_env>")
    OUTCOMES = ("closed_think", "first_move", "all_calls")
    PAD, EOS, MAX_CALLS, MAX_PROMPT, BLOCK = 0, 1, 6, 512, 3072  # their max_env_calls, prompt filter, SFT block
    HF_FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chess_hf")
    SPLITS, STEP = ("test", "dev", "val", "train", "rl"), 10 ** 7  # example seeds: split k's i-th example is k * STEP + i

    def __init__(self, slot, bench=None, traces=None, rl=None):
        self.slot, self.prompt_len = slot, slot + 96
        self.sources, self._src, self._lan = {"bench": bench, "traces": traces, "rl": rl}, None, None
        self.ids = {w: i for i, w in enumerate(self.WORDS)}
        self.T, self.T_END, self.SEP, self.CALL = (self.ids[w] for w in ("<T>", "</T>", "<sep>", "<call_env>"))

    # ---- generation (CPU, needs python-chess): their tokenizer on their files
    def splits(self):
        return tuple((name, k * self.STEP, len(self._records()[name])) for k, name in enumerate(self.SPLITS))

    def _records(self):
        """Per split, its examples: (key, game text, solution line (UCI), group); trace rows add the trace."""
        if self._src is None:
            import pandas as pd
            src = self.sources
            bench = []
            for k in range(1, 6):
                for i, r in pd.read_parquet(os.path.join(src["bench"], f"test_B{k}.parquet")).iterrows():
                    moves = r["extra_info"]["Moves"].split()[1:]
                    assert str(moves[::2]) == r["reward_model"]["ground_truth"] and r["prompt"].endswith(" <T>")
                    bench.append((f"B{k}:{i}", r["prompt"][:-4], moves, f"B{k}"))
            # the first 85 trace shards in name order (their max_train_files; the last 2 held out), or all 180 (the 2
            # held out the same, the other 178 for training)
            shards = sorted(n for n in os.listdir(src["traces"]) if n.endswith(".json"))
            assert len(shards) in (85, 180), len(shards)
            shards = shards[:83] + shards[85:] + shards[83:85]
            trace = lambda names: [(f"{x['index']}", x["pgn"], x["moves_list"][1:], None,
                                    x["cot_by_method"]["trajectory_sep"]["cot_format_no_labels"])
                                   for n in names for x in json.load(open(os.path.join(src["traces"], n)))["results"]]
            held, train = trace(shards[-2:]), trace(shards[:-2])
            rl = pd.read_csv(src["rl"])
            rl = [(r.PuzzleId, (r.ctx, r.Moves.split()[0]), r.Moves.split()[1:], None)
                  for r in rl[rl["_source"] == "existing_rl"].itertuples()]
            self._src = {"test": bench, "dev": [h[:4] for h in held], "val": held, "train": train, "rl": rl}
        return self._src

    def lan(self):
        """Their LanTokenizerSFT (the released tokenizer.py, as their HF wrapper builds it)."""
        if self._lan is None:
            import importlib.util
            spec = importlib.util.spec_from_file_location("chess_hf_tokenizer", os.path.join(self.HF_FILES, "tokenizer.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            with open(os.path.join(self.HF_FILES, "tokenizer_config.json")) as f:
                self._lan = mod.LanTokenizerSFT(config=json.load(f)["lan_config"])
            assert self._lan._tok2id == self.ids
        return self._lan

    def example(self, seed):
        k, i = divmod(seed, self.STEP)
        name = self.SPLITS[k]
        rec = self._records()[name][i]
        key = f"{name}:{rec[0]}".encode()
        return self._trace(key, *rec) if name in ("val", "train") else self._puzzle(key, *rec)

    def _puzzle(self, key, _, game, line, group):
        """A puzzle row (None if the prompt is over MAX_PROMPT): the prompt, the minimal answer and the script."""
        import chess
        import chess.pgn
        lan = self.lan()
        if isinstance(game, tuple):  # (PGN before the trigger move, the trigger move's UCI): their RL prompt
            ctx, trigger = game
            board = chess.pgn.read_game(io.StringIO(ctx)).end().board()
            board.push_uci(trigger)
            game = re.sub(r"\s*(1-0|0-1|1/2-1/2|\*)\s*$", "", str(chess.pgn.Game.from_board(board).accept(
                chess.pgn.StringExporter(headers=False, variations=False, comments=False)))).strip()
        prompt = lan.encode(game + " <T>")[1:-1]
        if len(prompt) > self.MAX_PROMPT:
            return None
        board = chess.pgn.read_game(io.StringIO(game)).end().board()
        answer, script = [self.T_END], []
        for j, uci in enumerate(line):
            mv = chess.Move.from_uci(uci)
            word = _lan_word(board, mv)
            board.push(mv)
            if j % 2 == 0:  # solver move
                target = [self.ids[uci[:2]], self.ids[uci[2:4]]] + ([self.ids[uci[4].upper()]] if len(uci) > 4 else [])
                answer += [self.ids[t] for t in lan._lan_move_to_tokens(word)] + [self.CALL]
                script += target + [self.SEP]
            else:  # the environment's reply, tokenized as their rollout does (tokenizer(reply, add_special_tokens=False))
                reply = lan.encode(word)[1:-1]
                answer += [t + ENV for t in reply]
                script += reply + [self.CALL]
        script += [self.CALL]
        H, width = (len(line) + 1) // 2, self.prompt_len - self.slot
        assert H <= self.MAX_CALLS and len(line) % 2 == 1 and len(script) <= width, (key, line)
        return key, prompt + [self.PAD] * (self.slot - len(prompt)) + script + [self.PAD] * (width - len(script)), answer, {
            "prompt": len(prompt), "turns": H, **({"group": group} if group else {})}

    def _trace(self, key, _, pgn, line, group, cot):
        """A trace row (None if over their 3072-token SFT block): <bos> game <T> | trace <eos>, opponent moves + ENV."""
        lan = self.lan()
        prompt = lan.encode(pgn.replace("\n", " ").strip())[:-1] + [self.T]
        response = lan.encode(cot)[1:]
        assert response[0] == self.T and response[-1] == self.EOS and len(prompt) <= self.slot, key
        response = response[1:]
        if len(prompt) + len(response) > self.BLOCK:
            return None
        env = _opponent_moves(response, self.CALL, self.ids)
        return key, prompt + [self.PAD] * (self.prompt_len - len(prompt)), [t + ENV * e for t, e in zip(response, env)], {
            "prompt": len(prompt), "response": len(response)}

    # ---- reward (the script parsed on the CPU; judged on the CPU, as their reward is string code)
    def parse(self, prompts):
        """The decoder's script (plen, env (B, K, 8), env_len (B, K), turns (B,)) and the solution's UCI moves ("gt")."""
        rows = prompts.cpu().numpy()
        B, K = len(rows), self.MAX_CALLS - 1
        plen, turns, env, env_len, gt = [], [], torch.zeros(B, K, 8, dtype=torch.long), torch.zeros(B, K, dtype=torch.long), []
        for b, row in enumerate(rows):
            plen.append(int(row[:self.slot].nonzero()[0][-1]) + 1)
            parts = _split(list(row[self.slot:]), self.CALL)[:-1]  # per solver move: UCI <sep> reply (the pad after)
            moves = []
            for k, part in enumerate(parts):
                sep = part.index(self.SEP)
                moves.append("".join(self.WORDS[t] for t in part[:sep]).lower())
                if k < len(parts) - 1:
                    env_len[b, k] = len(part) - sep - 1
                    env[b, k, :env_len[b, k]] = torch.tensor(part[sep + 1:])
            gt.append(moves)
            turns.append(len(moves))
        dev = prompts.device
        script = {"plen": torch.tensor(plen, device=dev), "env": env.to(dev), "env_len": env_len.to(dev),
                  "turns": torch.tensor(turns, device=dev)}
        return {"script": script, "gt": gt}

    def judge(self, resp, st, idx):
        """Responses (B, R) token ids (+ ENV: environment tokens), row i against parsed puzzle idx[i]."""
        out = {k: [] for k in ("score", *self.OUTCOMES)}
        for row, i in zip(resp.tolist(), idx.tolist()):
            text = self.decode(row)
            moves = [_segment_move(seg) for seg in text.split("<call_env>")[:-1]]
            gt = st["gt"][i]
            uci = [None if m is None else _lan_to_uci(m) for m in moves]
            out["score"].append(len(moves) == len(gt) and uci == gt)
            out["closed_think"].append("</T>" in text)
            out["first_move"].append(bool(uci) and uci[0] == gt[0])
            out["all_calls"].append(len(moves) == len(gt))
        return {k: torch.tensor(v, dtype=torch.float32, device=resp.device) for k, v in out.items()}

    def decode(self, ids):
        """Their LanTokenizerSFT.decode: <bos> / <eos> dropped, the pieces of each move glued greedily into one word."""
        words = [self.WORDS[t % ENV] for t in ids if t % ENV > self.EOS]
        out, i, n = [], 0, len(words)
        while i < n:
            t = words[i]
            if t in ("O-O", "O-O-O"):
                suf = words[i + 1] if i + 1 < n and words[i + 1] in ("+", "#") else ""
                out.append(t + suf)
                i += 1 + bool(suf)
            elif t in ("K", "Q", "R", "B", "N", "P"):
                frm, j = (words[i + 1] if i + 1 < n else ""), i + 2
                cap = "x" if j < n and words[j] == "x" else ""
                j += bool(cap)
                to, j = (words[j] if j < n else ""), j + 1
                promo = "=" + words[j + 1] if j + 1 < n and words[j] == "=" and words[j + 1] in ("Q", "R", "B", "N") else ""
                j += 2 * bool(promo)
                suf = words[j] if j < n and words[j] in ("+", "#") else ""
                out.append(f"{t}{frm}{cap}{to}{promo}{suf}")
                i = j + bool(suf)
            else:
                out.append(t)
                i += 1
        return " ".join(out)


MOVE_RE = re.compile(r"^[PNBRQK]([a-h][1-8])x?([a-h][1-8])(?:=([QRBN]))?$")  # a LAN move (no castling), its squares


def _split(seq, sep):
    """seq split after every sep."""
    parts, cur = [], []
    for t in seq:
        if t == sep:
            parts.append(cur)
            cur = []
        else:
            cur.append(t)
    return parts + [cur]


def _complete(word):
    return word.rstrip("+#") in ("O-O", "O-O-O") or bool(MOVE_RE.match(word.rstrip("+#")))


def _segment_move(seg):
    """reward_function_multiturn: the first complete move after the segment's first </T>, else its last complete move
    (move numbers skipped)."""
    after = [w for w in seg[seg.find("</T>") + 4:].split() if _complete(w)] if "</T>" in seg else []
    last = [w for w in seg.split() if _complete(w)]
    return after[0] if after else last[-1] if last else None


def _lan_to_uci(move):
    """Their lan_to_uci of a complete move (castling with its default side_to_move "white")."""
    move = move.rstrip("+#")
    if move in ("O-O", "O-O-O"):
        return "e1g1" if move == "O-O" else "e1c1"
    frm, to, promo = MOVE_RE.match(move).groups()
    return frm + to + (promo or "").lower()


def _lan_word(board, mv):
    """A move's LAN word on the board before it (their move_to_lan)."""
    after = board.copy()
    after.push(mv)
    suf = "#" if after.is_checkmate() else "+" if after.is_check() else ""
    if board.is_castling(mv):
        return ("O-O" if mv.to_square % 8 == 6 else "O-O-O") + suf
    import chess
    promo = "=" + chess.piece_symbol(mv.promotion).upper() if mv.promotion else ""
    return (board.piece_at(mv.from_square).symbol().upper() + chess.square_name(mv.from_square)
            + ("x" if board.is_capture(mv) else "") + chess.square_name(mv.to_square) + promo + suf)


def _opponent_moves(seq, call, ids):
    """Their MultiTurnSFTDataset._mask_env_responses on a response: True for the opponent move after each <call_env>
    (up to the next move start that is not a promotion piece, or a structural token)."""
    starts = {ids[w] for w in ("K", "Q", "R", "B", "N", "P", "O-O", "O-O-O")}
    struct = {ids[w] for w in ("<T>", "</T>", "<sep>", "<bos>", "<eos>")}
    out, inside, started, eq = [False] * len(seq), False, False, False
    for j, t in enumerate(seq):
        if j and seq[j - 1] == call:
            inside, started, eq = True, False, False
        if not inside:
            eq = False
        elif t in struct or (t in starts and not eq and started):
            inside = started = eq = False
        elif t in starts and not eq:
            started, out[j] = True, True
        else:
            out[j], eq = True, t == ids["="]
    return out


TASKS = {"maze": Maze, "sudoku": Sudoku, "chess": Chess}


def make_task(spec):
    """A task from its config entry, e.g. {"name": "maze", "size": 33, "eps": 0.3}."""
    spec = dict(spec)
    return TASKS[spec.pop("name")](**spec)
