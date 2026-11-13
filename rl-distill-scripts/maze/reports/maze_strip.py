#!/usr/bin/env python3
"""Five representative validation mazes as an HTML strip for the maze report pages.

The mazes are taken from an RL validation parquet (verl format: the prompt is the token grid WALL / PATH / START / GOAL rows
separated by NEWLINE). They are picked at the 10/30/50/70/90th percentiles of shortest-path length, so the strip spans the
set from short to long. Each maze shows one shortest path, with every cell on some shortest path tinted.

    from maze_strip import render_strip; html = render_strip("eval500/test.parquet", "17×17 depth-first, 0–20% knockout")
    python3 maze_strip.py --parquet eval1000/test.parquet --note "..." --inject page.html   # insert after the page's </header>
"""
import argparse
import html
from collections import deque

QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))

CSS = """<style>
.mazestrip { --mz-wall:#2b3431; --mz-open:#f3f1e9; --mz-short:#f4d9a3; --mz-route:#c4572a; --mz-start:#2f7d4a; --mz-goal:#3a5fc8; --mz-mark:#ffffff; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .mazestrip { --mz-wall:#0b0e0d; --mz-open:#2a312e; --mz-short:#6b5226;
  --mz-route:#f08a5d; --mz-start:#5cc28a; --mz-goal:#7d9cf0; --mz-mark:#0b0e0d; } }
:root[data-theme="dark"] .mazestrip { --mz-wall:#0b0e0d; --mz-open:#2a312e; --mz-short:#6b5226; --mz-route:#f08a5d; --mz-start:#5cc28a;
  --mz-goal:#7d9cf0; --mz-mark:#0b0e0d; }
.mazestrip { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:16px; display:grid; gap:10px; min-width:0; }
.mazestrip h2 { margin:0; font-size:1.05rem; font-weight:600; } .mazestrip p.note { margin:0; color:var(--muted); font-size:13px; max-width:90ch; }
.mazestrip .mzgrid { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 150px), 1fr)); gap:12px; }
.mazestrip figure { margin:0; display:grid; gap:4px; min-width:0; }
.mazestrip svg { width:100%; max-width:100%; height:auto; display:block; border-radius:4px; }
.mazestrip figcaption { font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; }
.mazestrip figcaption b { color:var(--fg); font-weight:500; }
.mazestrip .mzkey { display:flex; flex-wrap:wrap; gap:4px 14px; font-size:12px; color:var(--muted); }
.mazestrip .mzkey i { width:11px; height:11px; border-radius:2px; display:inline-block; vertical-align:-1px; margin-right:5px; }
.mazestrip rect.w { fill:var(--mz-wall); } .mazestrip rect.o { fill:var(--mz-open); } .mazestrip rect.s { fill:var(--mz-short); }
.mazestrip polyline { fill:none; stroke:var(--mz-route); stroke-width:2.4; stroke-linejoin:round; stroke-linecap:round; }
.mazestrip circle.st { fill:var(--mz-start); } .mazestrip circle.gl { fill:var(--mz-goal); }
.mazestrip text { fill:var(--mz-mark); font:600 7px system-ui, sans-serif; text-anchor:middle; }
</style>"""


def parse(prompt):
    """token grid -> (walls[r][c] bool, start, goal)"""
    toks = prompt.split()
    toks = toks[toks.index("GRID_START") + 1:toks.index("GRID_END")]
    rows, row = [], []
    for t in toks:
        if t == "NEWLINE":
            rows.append(row); row = []
        else:
            row.append(t)
    if row:
        rows.append(row)
    start = next((r, c) for r, x in enumerate(rows) for c, t in enumerate(x) if t == "START")
    goal = next((r, c) for r, x in enumerate(rows) for c, t in enumerate(x) if t == "GOAL")
    return [[t == "WALL" for t in x] for x in rows], start, goal


def bfs(walls, src):
    n, m = len(walls), len(walls[0])
    d = [[-1] * m for _ in range(n)]; cnt = [[0] * m for _ in range(n)]
    d[src[0]][src[1]] = 0; cnt[src[0]][src[1]] = 1; q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < m and not walls[rr][cc]:
                if d[rr][cc] < 0:
                    d[rr][cc] = d[r][c] + 1; q.append((rr, cc))
                if d[rr][cc] == d[r][c] + 1:
                    cnt[rr][cc] += cnt[r][c]
    return d, cnt


def analyse(walls, start, goal):
    ds, cnt = bfs(walls, start); dg, _ = bfs(walls, goal); L = ds[goal[0]][goal[1]]
    on = [[ds[r][c] >= 0 and dg[r][c] >= 0 and ds[r][c] + dg[r][c] == L for c in range(len(walls[0]))] for r in range(len(walls))]
    path = [goal]
    while path[-1] != start:
        r, c = path[-1]
        path.append(next((r + dr, c + dc) for dr, dc in MOVES if 0 <= r + dr < len(walls) and 0 <= c + dc < len(walls[0])
                         and not walls[r + dr][c + dc] and ds[r + dr][c + dc] == ds[r][c] - 1))
    return dict(L=L, n=cnt[goal[0]][goal[1]], on=on, path=path[::-1])


def svg(walls, start, goal, a):
    cs = 10; n = len(walls); w = len(walls[0]) * cs; h = n * cs
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{len(walls[0])}x{n} maze, shortest path {a["L"]} steps">',
           f'<rect width="{w}" height="{h}" class="o"/>']
    for r in range(n):
        for c in range(len(walls[0])):
            if walls[r][c] or a["on"][r][c]:
                out.append(f'<rect x="{c * cs}" y="{r * cs}" width="{cs}" height="{cs}" class="{"w" if walls[r][c] else "s"}"/>')
    out.append(f'<polyline points="{" ".join(f"{c * cs + cs / 2:g},{r * cs + cs / 2:g}" for r, c in a["path"])}"/>')
    for (r, c), cls, lab in ((start, "st", "S"), (goal, "gl", "G")):
        out.append(f'<circle cx="{c * cs + cs / 2:g}" cy="{r * cs + cs / 2:g}" r="{cs * 0.45:g}" class="{cls}"/>'
                   f'<text x="{c * cs + cs / 2:g}" y="{r * cs + cs / 2 + 2.5:g}">{lab}</text>')
    return "".join(out) + "</svg>"


def render_strip(parquet, note):
    import pandas as pd
    df = pd.read_parquet(parquet)
    mazes = []
    for i, p in enumerate(df["prompt"]):
        walls, s, g = parse(p[0]["content"])
        mazes.append((i, walls, s, g, analyse(walls, s, g)))
    Ls = sorted(m[4]["L"] for m in mazes); uniq = sum(m[4]["n"] == 1 for m in mazes) / len(mazes)
    by_len = Ls[0] != Ls[-1]   # every maze can have the same shortest-path length (open mazes: the grid distance)
    order = sorted(mazes, key=(lambda m: (m[4]["L"], m[4]["n"], m[0])) if by_len else (lambda m: (m[4]["n"], m[0])))
    picks = [order[min(len(order) - 1, int(q * len(order)))] for q in QUANTILES]
    size = f"{len(picks[0][1][0])}×{len(picks[0][1])}"
    cards = []
    for q, (i, walls, s, g, a) in zip(QUANTILES, picks):
        paths = f"{a['n']:,} shortest paths" if a["n"] > 1 else "unique shortest path"
        cards.append(f'<figure>{svg(walls, s, g, a)}<figcaption><b>{a["L"]} steps</b> · {paths}<br>'
                     f'p{int(q * 100)} · val maze #{i}</figcaption></figure>')
    nmed = sorted(m[4]["n"] for m in mazes)[len(mazes) // 2]
    how = (f"Picked at the 10th, 30th, 50th, 70th and 90th percentile of shortest-path length, ties broken by the number of shortest paths (median {Ls[len(Ls) // 2]} steps; "
           f"the shortest path is unique in {100 * uniq:.0f}% of them)" if by_len else
           f"Every shortest path here is {Ls[0]} steps, the grid distance, so the five are picked at the 10th, 30th, 50th, 70th and "
           f"90th percentile of the number of shortest paths (median {nmed:,})")
    return (f'{CSS}<section class="mazestrip"><h2>Five representative {size} validation mazes</h2>'
            f'<p class="note">{html.escape(note)} {how}, among the {len(mazes)} validation mazes. Start top-left, goal bottom-right.</p>'
            '<div class="mzkey"><span><i style="background:var(--mz-wall)"></i>wall</span><span><i style="background:var(--mz-open)"></i>open</span>'
            '<span><i style="background:var(--mz-short)"></i>on some shortest path</span><span><i style="background:var(--mz-route)"></i>one shortest path</span></div>'
            f'<div class="mzgrid">{"".join(cards)}</div></section>')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--inject", required=True, help="HTML page: the strip goes right after its first </header> (replacing an earlier strip)")
    args = ap.parse_args()
    page = open(args.inject).read()
    strip = render_strip(args.parquet, args.note)
    a = page.find("<!--mazestrip-->")
    if a >= 0:
        page = page[:a] + page[page.find("<!--/mazestrip-->") + len("<!--/mazestrip-->"):]
    i = page.find("</header>") + len("</header>")
    page = page[:i] + "<!--mazestrip-->" + strip + "<!--/mazestrip-->" + page[i:]
    open(args.inject, "w").write(page)
    print("injected into", args.inject)


if __name__ == "__main__":
    main()
