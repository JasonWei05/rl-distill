"""Task generation and reward: the sudoku generator, solver and judge, the maze parse, and the dataset files."""

import json
import os
import random

import numpy as np
import pytest
import torch

from data import Dataset, build
from tasks import EOS, PAD, PEERS, UNITS, Maze, Sudoku, solve
from conftest import small_config

SUDOKU_REF = os.environ.get("SUDOKU_REF", "/mnt/efs/jasonwei/sudoku-transfomer/data/val_9x9_100k.jsonl")


def test_sudoku_solver_matches_reference():
    """The trace of sudoku-transfomer's solve_step_by_step, on 2000 of its 9x9 validation puzzles."""
    n = 0
    with open(SUDOKU_REF) as f:
        for line in f:
            ex = json.loads(line)
            if ex["is_correct"]:
                trace, _ = solve([d for row in ex["puzzle"] for d in row])
                assert [[i // 9, i % 9, d] for i, d in trace] == ex["trace"]
                n += 1
                if n == 2000:
                    break


def grid_ok(grid):
    return all(sorted(grid[i] for i in u) == list(range(1, 10)) for u in UNITS)


def test_sudoku_examples():
    """Clue counts in range, demonstrations are triples that fill exactly the blanks with a valid solution."""
    task = Sudoku(40, 75)
    W = task.WORDS
    clues = []
    for seed in range(300):
        key, prompt, response, info = task.example(seed)
        puzzle = list(key)
        assert len(prompt) == task.prompt_len and response[-2:] == [task.ids["DONE"], EOS]
        assert [int(W[t]) for t in prompt[2:-2] if W[t].isdigit()] == puzzle
        grid = list(puzzle)
        for j in range(0, len(response) - 2, 3):
            r, c, d = (W[t] for t in response[j:j + 3])
            i = (int(r[1]) - 1) * 9 + int(c[1]) - 1
            assert grid[i] == 0
            grid[i] = int(d)
        assert grid_ok(grid) and sum(d > 0 for d in puzzle) == info["clues"]
        clues.append(info["clues"])
    assert min(clues) >= 40 and max(clues) <= 75 and len(set(clues)) > 25


def reference_judge(puzzle, words):
    """Plain-Python reward: the words before the first DONE are (R<r> C<c> <digit>) triples filling every blank of the
    puzzle exactly once, no clue, and the grid is valid."""
    broken = {"valid_format": 0, "overwrote_clue": 0, "duplicate_fill": 0, "complete": 0, "solved": 0}
    if "DONE" not in words:
        return broken
    w = words[:words.index("DONE")]
    ok = len(w) % 3 == 0 and all(
        a in [f"R{i}" for i in range(1, 10)] and b in [f"C{i}" for i in range(1, 10)] and c in "123456789" and len(c) == 1
        for a, b, c in zip(w[0::3], w[1::3], w[2::3]))
    if not ok:
        return broken
    grid, filled = list(puzzle), []
    for a, b, c in zip(w[0::3], w[1::3], w[2::3]):
        i = (int(a[1]) - 1) * 9 + int(b[1]) - 1
        filled.append(i)
        grid[i] = int(c)
    overwrote, duplicate = any(puzzle[i] for i in filled), len(set(filled)) < len(filled)
    complete = all(puzzle[i] or i in filled for i in range(81))
    solved = not overwrote and not duplicate and complete and grid_ok(grid)
    return {"valid_format": 1, "overwrote_clue": int(overwrote), "duplicate_fill": int(duplicate), "complete": int(complete),
            "solved": int(solved)}


def other_solution(puzzle, rng):
    """A random completion of the puzzle (digits tried in random order), or None."""
    grid = list(puzzle)

    def fill():
        if 0 not in grid:
            return True
        i = grid.index(0)
        for d in rng.sample(range(1, 10), 9):
            if all(grid[p] != d for p in PEERS[i]):
                grid[i] = d
                if fill():
                    return True
        grid[i] = 0
        return False
    return grid if fill() else None


def sudoku_responses(task, n_puzzles, R):
    """Per puzzle, 5 solved responses (the demonstration, its fills shuffled, another valid completion, words after
    DONE, a second DONE) and 12 broken ones (a wrong digit, a clue overwritten, a cell twice, a cell missing, no DONE, a
    cut triple, a leading pad, one fill only, random tokens); EOS-terminated and padded to R."""
    rng, v = random.Random(0), task.ids
    prompts, rows, which = [], [], []
    for p in range(n_puzzles):
        key, prompt, response, _ = task.example(10_000 + p)
        puzzle = list(key)
        prompts.append(prompt)
        demo = [response[j:j + 3] for j in range(0, len(response) - 2, 3)]
        trip = lambda i, d: [task.row1 + i // 9, task.col1 + i % 9, task.digit0 + d]
        flat = lambda ts: [t for x in ts for t in x] + [v["DONE"], EOS]
        shuffled = rng.sample(demo, len(demo))
        alt = other_solution(puzzle, rng)
        wrong = [list(x) for x in demo]
        wrong[0][2] = task.digit0 + (wrong[0][2] - task.digit0) % 9 + 1
        clue = next(i for i in range(81) if puzzle[i])
        solved = [flat(demo), flat(shuffled), flat([trip(i, alt[i]) for i in range(81) if not puzzle[i]]),
                  flat(demo)[:-1] + [task.row1, EOS], flat(demo)[:-1] + [v["DONE"], EOS]]
        cases = solved + [flat(wrong), flat(demo + [trip(clue, puzzle[clue])]), flat(demo + demo[:1]), flat(demo[1:]),
                          flat(demo)[:-2] + [EOS], flat(demo)[:4] + flat(demo)[5:], [PAD] + flat(demo), flat(demo[:1])]
        cases += [[rng.randrange(len(task.WORDS)) for _ in range(rng.randrange(1, R))] + [EOS] for _ in range(4)]
        rows += cases
        which += [p] * len(cases)
    tok = torch.full((len(rows), R), PAD, dtype=torch.long)
    for j, c in enumerate(rows):
        c = c[:R]
        tok[j, :len(c)] = torch.tensor(c)
    return torch.tensor(prompts).cuda(), tok.cuda(), torch.tensor(which).cuda()


def test_sudoku_judge_matches_reference():
    task = Sudoku(40, 75)
    prompts, tok, which = sudoku_responses(task, 150, 140)
    out = task.judge(tok, task.parse(prompts), which)
    puzzles = task.parse(prompts)["clues"].tolist()
    for j, (row, p) in enumerate(zip(tok.tolist(), which.tolist())):
        ref = reference_judge(puzzles[p], [task.WORDS[t] for t in row])
        for k in ref:
            assert out[k][j].item() == ref[k], (k, j, [task.WORDS[t] for t in row])
    assert out["solved"].view(150, -1)[:, :5].all()
    assert out["solved"].view(150, -1)[:, 5:].sum() == 0
    assert out["score"].equal(out["solved"])


def test_maze_parse_and_lstar():
    """Parsed grid, start, goal and BFS L* against the generator's grid and its walk's L*."""
    task = Maze(25, 0.5)
    ex = [task.example(s) for s in range(200)]
    st = task.parse(torch.tensor([e[1] for e in ex]).cuda())
    assert st["lstar"].tolist() == [e[3]["lstar"] for e in ex]
    walls = torch.tensor(np.frombuffer(b"".join(e[0] for e in ex), dtype=np.uint8).reshape(200, 25, 25)).cuda()
    assert torch.equal(st["free"], walls == 0)
    assert (st["start"] == 1).all() and (st["goal"] == 23).all()
    out = task.judge(torch.tensor([e[2] + [PAD] * (300 - len(e[2])) for e in ex]).cuda(), st, torch.arange(200).cuda())
    assert out["goal_reached"].all()
    assert out["is_shortest"].tolist() == [float(e[3]["optimal"]) for e in ex]


@pytest.mark.parametrize("fixture", ["sudoku_data", "maze_data"])
def test_dataset_files(fixture, request):
    d = Dataset(request.getfixturevalue(fixture))
    seen = set()
    for name, n in (("dev", 1000), ("test", 1000), ("val", 256), ("train", 3000)):
        a = d.split(name)
        assert a.dtype == np.uint8 and a.shape == (n, d.width) and d.meta["splits"][name]["n"] == n
        first = d.meta["splits"][name]["first_seed"]
        for i in (0, n // 2, n - 1):  # rows are the generator's examples of consecutive seeds
            key, prompt, response, _ = d.task.example(first + i)
            assert a[i].tolist() == prompt + response + [PAD] * (d.width - len(prompt) - len(response))
        keys = {a[i, :d.prompt_len].tobytes() for i in range(n)}
        assert len(keys) == n and not keys & seen
        seen |= keys


def test_build_drops_repeats_and_is_deterministic(tmp_path):
    """9x9 mazes repeat (a 4x4 cell grid): repeats are dropped and every grid is in exactly one split."""
    cfg = small_config("maze25", size=9)
    meta = build(cfg, str(tmp_path / "a"), workers=4)
    build(cfg, str(tmp_path / "b"), workers=2)
    assert sum(s["repeats_dropped"] for s in meta["splits"].values()) > 0
    total = sum(s["n"] for s in meta["splits"].values())
    rows = [np.load(tmp_path / "a" / f"{n}.npy") for n in ("dev", "test", "val", "train")]
    assert len({r[:meta["prompt_len"]].tobytes() for a in rows for r in a}) == total
    for n in ("dev", "test", "val", "train"):
        assert np.array_equal(np.load(tmp_path / "a" / f"{n}.npy"), np.load(tmp_path / "b" / f"{n}.npy"))


def test_refresh_meta(sudoku_data, tmp_path):
    """data.py --refresh takes new tune / decode / rl settings into meta.json and refuses a change of a frozen section."""
    import shutil
    import subprocess
    import sys
    import yaml
    out = tmp_path / "d"
    out.mkdir()
    shutil.copy(os.path.join(sudoku_data, "meta.json"), out / "meta.json")
    cfg = Dataset(sudoku_data).cfg
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data.py")

    def refresh(c):
        path = tmp_path / "sudoku9.yaml"
        path.write_text(yaml.safe_dump({k: v for k, v in c.items() if k != "name"}))
        return subprocess.run([sys.executable, script, "--config", str(path), "--out", str(out), "--refresh"], capture_output=True, text=True)
    r = refresh({**cfg, "rl": {**cfg["rl"], "lr": 3e-5}})
    assert r.returncode == 0, r.stderr
    assert json.load(open(out / "meta.json"))["config"]["rl"]["lr"] == 3e-5
    r = refresh({**cfg, "sft": {**cfg["sft"], "max_steps": 2 * cfg["sft"]["max_steps"], "stop_at": 0.5}})
    assert r.returncode == 0, r.stderr  # extends the SFT runs
    r = refresh({**cfg, "sft": {**cfg["sft"], "lr": 1e-3}})
    assert r.returncode != 0 and "frozen section" in r.stderr
