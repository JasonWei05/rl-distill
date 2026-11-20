"""Results of a synth study from S3: the tuned bases (pass@1 per size x seed x level, SFT steps) and the RL grid (best
validation score, mean over seeds, per size x level x position in the size's prompts-per-step list, rl.batches: the same
position has about the same FLOPs per step for every size), as Markdown tables, or with --html as a page that adds the
mean validation curves against steps and, per starting level, against cumulative training FLOPs for the three sizes.

    python report.py --configs sudoku9 maze33 maze49 [--html synth.html] [--cache /tmp/synth/report_cache]
    python report.py --configs maze33 --minibatches 4 --lr 2.5e-5   # the runs plan.py named for that variant
"""

import argparse
import html
import math
import os
import statistics
import subprocess
import time

from files import read_json
from model import ARCH
from plan import config, rl_name
from tasks import make_task

S3 = "s3://scale-ml/genai/rl-distill/synth"
SIZES, SEEDS, LEVELS, POSITIONS = ("2XS", "XS", "S"), (0, 1, 2), ("p1", "p3", "p10"), range(4)
TITLES = {"sudoku9": "Sudoku 9×9, 40–75 clues", "sudoku9hard": "Sudoku 9×9, 30–55 clues", "maze33": "Maze 33×33, ε 0.3",
          "maze49": "Maze 49×49, ε 0.2"}
PARAMS = {"2XS": "1.0M", "XS": "2.2M", "S": "3.9M"}


def aws(cmd):
    # the aws CLI intermittently dies with SIGBUS on the devbox: retry, then fail
    assert any(subprocess.run(cmd).returncode == 0 for _ in range(3)), f"{' '.join(cmd[:3])} failed 3 times"


def step_flops(size, words, prompt_len, response_len, batch, n, minibatches=1):
    """Training FLOPs of one RL step (validation left out), from the model shape: batch x minibatches prompts are sampled,
    each runs once and each of its n responses against its cache (model.Decoder, train_logprobs), as one forward in the
    rollout, a forward + backward (3 forwards) in the update and, outside the first mini-batch, one more forward for
    pi_old; response_len is the demonstrations' mean. A token at context c costs 2 x the weights it multiplies
    + 4 L H c (QK^T and AV); prompt tokens skip the output layer."""
    H, L, heads, kv_heads, ffn = ARCH[size]
    weights = L * (2 * H * H + 2 * H * kv_heads * H // heads + 3 * H * ffn)
    prompt = prompt_len * (2 * weights + 4 * L * H * prompt_len / 2)
    response = response_len * (2 * (weights + words * H) + 4 * L * H * (prompt_len + response_len / 2))
    return (5 - 1 / minibatches) * batch * minibatches * (prompt + n * response)


def variant(minibatches, lr):
    return f"lr {lr:g}" + (f"; each step samples {minibatches} x the prompts per step and takes {minibatches} updates on"
                           f" mini-batches of them, PPO clip x token TIS (columns: prompts per mini-batch)" if minibatches > 1 else "")


def collect(cfg, cache, minibatches=1, lr=None):
    """{"bases": {(size, seed): None (training) | {"steps", level: pass1 | "not reached" | None (tuning)}},
        "rl": {(size, level): {position: [(history, done) per started seed]}}, "batches": rl.batches,
        "step_flops": {size: [training FLOPs per step per position]}, "variant": its description} from the run files mirrored under cache/<cfg>/ and
    the dataset's meta.json (prompt length, mean demonstration length)."""
    cmd = ["aws", "s3", "sync", "--delete", "--only-show-errors", f"{S3}/runs/{cfg}", os.path.join(cache, cfg), "--exclude", "*",
           "--include", "sft/*/tune.json", "--include", "sft/*/DONE", "--include", "rl/*/best.json", "--include", "rl/*/COMPLETE.json"]
    meta = os.path.join(cache, f"{cfg}.meta.json")
    aws(cmd)
    aws(["aws", "s3", "cp", "--only-show-errors", f"{S3}/data/{cfg}/meta.json", meta])
    c, m = config(cfg), read_json(meta)
    words, response_len = len(make_task(c["task"]).WORDS), m["splits"]["train"]["stats"]["response_len"]["mean"]
    out = {"bases": {}, "rl": {}, "batches": c["rl"]["batches"], "variant": variant(minibatches, lr or c["rl"]["lr"]), "step_flops": {
        size: [step_flops(size, words, m["prompt_len"], response_len, b, c["rl"]["n"], minibatches) for b in c["rl"]["batches"][size]]
        for size in SIZES}}
    for size in SIZES:
        for seed in SEEDS:
            run = os.path.join(cache, cfg, "sft", f"{size}_s{seed}")
            if os.path.exists(os.path.join(run, "DONE")):
                tune = read_json(os.path.join(run, "tune.json")) if os.path.exists(os.path.join(run, "tune.json")) else {}
                out["bases"][size, seed] = {"steps": read_json(os.path.join(run, "DONE"))["step"], **{
                    l: (tune[l]["pass1"] if tune[l]["reachable"] else "not reached") if l in tune else None for l in LEVELS}}
            else:
                out["bases"][size, seed] = None
        for level in LEVELS:
            for i, batch in enumerate(out["batches"][size]):
                runs = [os.path.join(cache, cfg, "rl", rl_name(size, level, s, batch, minibatches, lr)) for s in SEEDS]
                out["rl"].setdefault((size, level), {})[i] = [
                    (read_json(os.path.join(r, "best.json"))["history"], os.path.exists(os.path.join(r, "COMPLETE.json")))
                    for r in runs if os.path.exists(os.path.join(r, "best.json"))]
    return out


def base_cell(b, level):
    return "training" if b is None else "tuning" if b[level] is None else b[level] if isinstance(b[level], str) else f"{100 * b[level]:.2f}"


def rl_cell(runs):
    """Mean best validation score over seeds, and how many runs are complete of those started."""
    return f"{statistics.mean(max(v for _, v in h) for h, _ in runs):.3f} [{sum(d for _, d in runs)}/{len(runs)}]" if runs else "–"


def best_mean(runs):
    return f"{statistics.mean(max(v for _, v in h) for h, _ in runs):.3f}" if runs else "–"


def start(cell):
    return statistics.mean(h[0][1] for runs in cell.values() for h, _ in runs)


def heading(d, i):
    """Column i's prompts per step, 2XS / XS / S."""
    return " / ".join(str(d["batches"][s][i]) for s in SIZES)


def markdown(cfg, d):
    lines = [f"**{cfg} bases** (confirmed pass@1, %; SFT steps)", "", "| size | seed | " + " | ".join(LEVELS) + " | SFT steps |",
             "|---" * (len(LEVELS) + 3) + "|"]
    for (size, seed), b in d["bases"].items():
        lines.append(f"| {size} | {seed} | " + " | ".join(base_cell(b, l) for l in LEVELS) + f" | {b['steps'] if b else '…'} |")
    lines += ["", f"**{cfg} RL** (best validation score, mean of seeds [complete / started]; step 0 in parentheses; columns:"
              " prompts per step for 2XS / XS / S, about equal FLOPs per step)", "",
              "| size | level | start | " + " | ".join(heading(d, i) for i in POSITIONS) + " |", "|---" * (len(POSITIONS) + 3) + "|"]
    for (size, level), cell in d["rl"].items():
        if any(cell.values()):
            lines.append(f"| {size} | {level} | ({start(cell):.3f}) | " + " | ".join(rl_cell(cell[i]) for i in POSITIONS) + " |")
    return lines


def curve(runs):
    """The mean over seeds of the validation history, up to the shortest seed's last step."""
    n = min(len(h) for h, _ in runs)
    return [(runs[0][0][i][0], statistics.mean(h[i][1] for h, _ in runs)) for i in range(n)]


def chart(cell, title):
    """An SVG of the validation curves of the four prompts-per-step positions: every seed's (thin) and their mean (thick,
    up to the shortest seed's last step); x: steps 0-10k, y: 0 to the cell's max."""
    W, H, L, B, T, R = 300, 190, 40, 26, 22, 10
    curves = {i: curve(runs) for i, runs in cell.items() if runs}
    seeds = [(i, h) for i, runs in cell.items() for h, _ in runs]
    top = max(0.05, max(v for _, h in seeds for _, v in h) * 1.1)
    x = lambda s: L + (W - L - R) * s / 10000
    y = lambda v: H - B - (H - B - T) * v / top
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}"><text class="ct" x="{L}" y="14">{html.escape(title)}</text>']
    for v in (0, top / 2, top):
        parts.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/><text class="ax" x="{L - 4}" y="{y(v) + 3:.1f}" text-anchor="end">{v:.2f}</text>')
    for s in (0, 5000, 10000):
        parts.append(f'<text class="ax" x="{x(s):.1f}" y="{H - 8}" text-anchor="middle">{s // 1000}k</text>')
    pts = lambda c: " ".join(f"{x(s):.1f},{y(v):.1f}" for s, v in c[::5] + c[-1:])
    for i, h in seeds:
        parts.append(f'<polyline class="b{i} seed" fill="none" points="{pts(h)}"/>')
    for i, c in curves.items():
        parts.append(f'<polyline class="b{i}" fill="none" points="{pts(c)}"/>')
    return "".join(parts) + "</svg>"


def flops_lines(d, level):
    """{(size index, position): [(cumulative training FLOPs, mean validation score)]} past step 0."""
    lines = {(k, i): [(s * d["step_flops"][size][i], v) for s, v in curve(runs) if s > 0]
             for k, size in enumerate(SIZES) for i, runs in d["rl"][size, level].items() if runs}
    return {key: c for key, c in lines.items() if c}


def envelope(lines, k):
    """Size k's best score within each FLOPs budget, over its positions."""
    merged = sorted(p for (kk, _), c in lines.items() if kk == k for p in c)
    best = [(f, max(v for _, v in merged[:j + 1])) for j, (f, _) in enumerate(merged)]
    return best[::max(1, len(best) // 200)] + best[-1:]


def flops_chart(d, level, title):
    """An SVG of validation score vs cumulative training FLOPs (log x) for the three sizes from one starting level: a thin
    line per prompts-per-step position (mean of seeds) and a thick one for the size's best score within each budget."""
    W, H, L, B, T, R = 300, 190, 40, 26, 22, 10
    lines = flops_lines(d, level)
    if not lines:
        return ""
    every = [p for c in lines.values() for p in c]
    lo = math.floor(math.log10(min(f for f, _ in every)))
    hi = math.ceil(math.log10(max(f for f, _ in every)))
    top = max(0.05, max(v for _, v in every) * 1.1)
    x = lambda f: L + (W - L - R) * (math.log10(f) - lo) / (hi - lo)
    y = lambda v: H - B - (H - B - T) * v / top
    pts = lambda c: " ".join(f"{x(f):.1f},{y(v):.1f}" for f, v in c)
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}"><text class="ct" x="{L}" y="14">{html.escape(title)}</text>']
    for v in (0, top / 2, top):
        parts.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/><text class="ax" x="{L - 4}" y="{y(v) + 3:.1f}" text-anchor="end">{v:.2f}</text>')
    for e in range(lo, hi + 1, max(1, (hi - lo + 3) // 4)):
        parts.append(f'<text class="ax" x="{x(10 ** e):.1f}" y="{H - 8}" text-anchor="middle">1e{e}</text>')
    for (k, i), c in lines.items():
        parts.append(f'<polyline class="s{k} thin" fill="none" points="{pts(c[::5] + c[-1:])}"/>')
    for k in range(len(SIZES)):
        best = envelope(lines, k)
        if len(best) > 1:
            parts.append(f'<polyline class="s{k}" fill="none" points="{pts(best)}"/>')
    return "".join(parts) + "</svg>"


def page(configs, data, title="Synth RL Study", note=""):
    sections = []
    for cfg in configs:
        d = data[cfg]
        rows = "".join(f"<tr><td>{size} ({PARAMS[size]})</td><td>{seed}</td>" + "".join(f"<td>{base_cell(b, l)}</td>" for l in LEVELS)
                       + f"<td>{b['steps'] if b else '…'}</td></tr>" for (size, seed), b in d["bases"].items())
        was = lambda size, level, i: f'<br><span class="note">was {best_mean(d["baseline"]["rl"][size, level][i])}</span>' if "baseline" in d else ""
        rl_rows = "".join(f"<tr><td>{size}</td><td>{level}</td><td>{start(cell):.3f}</td>" + "".join(f"<td>{rl_cell(cell[i])}{was(size, level, i)}</td>" for i in POSITIONS)
                          + "</tr>" for (size, level), cell in d["rl"].items() if any(cell.values()))
        charts = "".join(f"<figure>{chart(cell, f'{size} · {level}')}</figure>" for (size, level), cell in d["rl"].items() if any(cell.values()))
        fcharts = "".join(f"<figure>{c}</figure>" for c in (flops_chart(d, level, f"start {level}") for level in LEVELS) if c)
        sections.append(f"""<section><h2>{TITLES[cfg]}</h2><p class="sub">RL: {html.escape(d["variant"])}</p>
<h3>Bases: confirmed pass@1 (%)</h3><div class="scroll"><table><tr><th>size</th><th>seed</th>{''.join(f'<th>{l}</th>' for l in LEVELS)}<th>SFT steps</th></tr>{rows}</table></div>
<h3>RL: best validation score, mean of seeds <span class="note">[complete / started]; columns: prompts per step for 2XS / XS / S (about equal FLOPs per step)</span></h3>
<div class="scroll"><table><tr><th>size</th><th>level</th><th>step 0</th>{''.join(f'<th>{heading(d, i)}</th>' for i in POSITIONS)}</tr>{rl_rows or '<tr><td colspan="7">not started</td></tr>'}</table></div>
{f'<h3>Validation score vs training FLOPs, by size <span class="note">thin: each prompts-per-step setting (mean of seeds); thick: best score within the budget</span></h3><div class="grid">{fcharts}</div>' if fcharts else ''}
{f'<h3>Validation score vs step <span class="note">thin: each seed; thick: mean of seeds (to the shortest seed\'s last step)</span></h3><div class="grid">{charts}</div>' if charts else ''}</section>""")
    size_legend = "".join(f'<span><i class="k s{k}"></i>{size} ({PARAMS[size]})</span>' for k, size in enumerate(SIZES))
    legend = "".join(f'<span><i class="k b{i}"></i>{["smallest", "2nd", "3rd", "largest"][i]} prompts per step (equal FLOPs across sizes)</span>' for i in POSITIONS)
    return f"""<meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
:root {{ --bg: #f7f6f2; --fg: #1f2328; --muted: #5d6470; --line: #d9d6cc; --card: #ffffff; --c16: #9ecae1; --c64: #4292c6; --c256: #08519c; --c1024: #e6550d; --z0: #1b9e77; --z1: #7570b3; --z2: #d95f02; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #16181c; --fg: #e8e6e1; --muted: #9aa1ab; --line: #343841; --card: #1e2126; --c16: #6baed6; --c64: #4f9bd9; --c256: #9ecae1; --c1024: #fd8d3c; --z0: #4fd1a5; --z1: #a9a4f0; --z2: #fd8d3c; color-scheme: dark }} }}
:root[data-theme="dark"] {{ --bg: #16181c; --fg: #e8e6e1; --muted: #9aa1ab; --line: #343841; --card: #1e2126; --c16: #6baed6; --c64: #4f9bd9; --c256: #9ecae1; --c1024: #fd8d3c; --z0: #4fd1a5; --z1: #a9a4f0; --z2: #fd8d3c; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.5 "IBM Plex Sans", system-ui, sans-serif; padding-inline: 16px; padding-block: 24px 48px; max-width: 1100px; margin: 0 auto; }}
h1 {{ font-size: 1.6rem; margin: 0 0 .25rem; }} h2 {{ font-size: 1.25rem; margin: 2rem 0 .5rem; border-bottom: 1px solid var(--line); padding-bottom: .25rem; }}
h3 {{ font-size: .95rem; color: var(--muted); font-weight: 600; margin: 1.2rem 0 .4rem; }} .note, .sub {{ color: var(--muted); font-weight: 400; }}
.sub {{ max-width: 70ch; }} .scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; background: var(--card); }}
th, td {{ border: 1px solid var(--line); padding: 3px 10px; text-align: right; white-space: nowrap; }} th {{ font-weight: 600; }}
td:first-child, th:first-child, td:nth-child(2) {{ text-align: left; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 8px; }}
figure {{ margin: 0; background: var(--card); border: 1px solid var(--line); min-width: 0; }} svg {{ width: 100%; height: auto; display: block; }}
.ct {{ fill: var(--fg); font-size: 12px; font-weight: 600; }} .ax {{ fill: var(--muted); font-size: 10px; }} .grid line, line.grid {{ stroke: var(--line); }}
polyline {{ stroke-width: 1.8; }} .b0 {{ stroke: var(--c16); background: var(--c16); }} .b1 {{ stroke: var(--c64); background: var(--c64); }}
.b2 {{ stroke: var(--c256); background: var(--c256); }} .b3 {{ stroke: var(--c1024); background: var(--c1024); }}
.s0 {{ stroke: var(--z0); background: var(--z0); }} .s1 {{ stroke: var(--z1); background: var(--z1); }} .s2 {{ stroke: var(--z2); background: var(--z2); }}
polyline.s0, polyline.s1, polyline.s2 {{ stroke-width: 2.4; }} polyline.thin {{ stroke-width: 1; opacity: .35; }}
polyline.seed {{ stroke-width: .9; opacity: .45; }}
polyline.b0:not(.seed), polyline.b1:not(.seed), polyline.b2:not(.seed), polyline.b3:not(.seed) {{ stroke-width: 2.2; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 14px; color: var(--muted); font-size: .9rem; }} .k {{ display: inline-block; width: 18px; height: 3px; margin-right: 6px; vertical-align: middle; }}
</style>
<h1>{html.escape(title)}</h1>
<p class="sub">Pretraining from scratch on demonstrations (1.0M / 2.2M / 3.9M Qwen2, 3 init seeds each), bases tuned to 1 / 3 / 10% pass@1 where
reachable, then GRPO + TIS (t3 recipe, 10k steps, run seed = base seed). Validation: 500 dev puzzles × 16 samples. Code: rl-distill-scripts/synth. Data from S3 at {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. {html.escape(note)}</p>
<p class="sub">Training FLOPs per RL step: each prompt's forward once and each of its 16 responses against its cache, in the
rollout (1 forward) and the update (forward + backward, 3 forwards), at the dataset's prompt length and mean demonstration
length; validation left out.</p>
<div class="legend">FLOPs charts: {size_legend}</div>
<div class="legend">Step charts: {legend}</div>
{''.join(sections)}"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--cache", default="/tmp/synth/report_cache")
    ap.add_argument("--html", help="write the page here instead of printing Markdown")
    ap.add_argument("--minibatches", type=int, default=1, help="the runs of plan.py rl --minibatches")
    ap.add_argument("--lr", type=float, help="the runs of plan.py rl --lr")
    ap.add_argument("--title", default="Synth RL Study")
    ap.add_argument("--note", default="", help="a sentence added to the page's introduction")
    a = ap.parse_args()
    data = {cfg: collect(cfg, a.cache, a.minibatches, a.lr) for cfg in a.configs}
    if a.minibatches > 1 or a.lr:  # compare with the config's own grid
        for cfg in a.configs:
            data[cfg]["baseline"] = collect(cfg, a.cache)
    if a.html:
        with open(a.html, "w") as f:
            f.write(page(a.configs, data, a.title, a.note))
    else:
        print("\n\n".join("\n".join(markdown(cfg, data[cfg])) for cfg in a.configs))
