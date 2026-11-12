#!/usr/bin/env python3
"""HTML page of the shortest-path maze study: bases at 5 is_shortest pass@1 levels for small / medium / large.

Reads a local mirror of the pipeline output (<root>/<size>/<run>/screen_*.json, tune-<level>/tune.json, <size>/status.json,
see run_maze_pretrain_st.sh) and draws, per level, the unbiased shortest-path pass@k (k = 1..1024) of every confirmed
base (its fresh-seed 1000-maze x 2048-sample confirmation screen), a level x seed status grid, and the pretraining
trajectories (is_shortest and goal_reached pass@1 by step from the stride screens).

    python3 rl-distill-scripts/maze/report_shortest.py --root /tmp/jw_maze/pt_shortest_v1 --out page.html
"""
import argparse
import glob
import html
import json
import math
import os

import numpy as np

from compare_matched_curves import pass_at_k
from report_matched_curves import line_chart, nice_top, rl_section

ALL_SIZES = [("3XS", "3XS", "0.55M", "var(--tiny)"), ("2XS", "2XS", "1.0M", "var(--mini)"), ("XS", "small", "2.2M", "var(--small)"),
             ("S", "medium", "3.9M", "var(--medium)"), ("L", "large", "13.3M", "var(--large)")]
SIZES = ALL_SIZES[2:]   # --sizes picks others
LEVELS = [("p01", "0.1%", 0.001), ("p03", "0.3%", 0.003), ("p1", "1%", 0.01), ("p3", "3%", 0.03), ("p10", "10%", 0.10)]
KS = [2 ** i for i in range(11)]
DASH = {0: "", 1: "5 3", 2: "1.5 3", 3: "8 2 1.5 2", 4: "3 1.5"}
SEEDS = (0, 1, 2)
SEED_COLORS = {i: f"var(--seed{i})" for i in range(5)}
W, H, PL, PR, PT, PB = 520, 300, 54, 14, 12, 36

LEDE = 'Reward = 1 only for a shortest path. All three sizes are pretrained from scratch on shortest paths only (a BFS-optimal path per maze, drawn uniformly among ties; same 1.3M 17×17 mazes), batch 128, constant LR 5e-4, 3 init seeds each. Bases are tuned to each is_shortest pass@1 level by checkpoint averaging + interpolation and confirmed with a fresh sampling seed on 1000 dev mazes × 2048 samples.'


def load_json(p):
    try:
        return json.load(open(p))
    except (OSError, json.JSONDecodeError):
        return None


def load_jsonl(p):
    try:
        with open(p) as f:
            return [json.loads(line) for line in f if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def aligned(run, lvl):
    """align_passk.py result (pass@1 on target and the whole pass@k curve matched to the small base), preferred over tune-<lvl>."""
    if not os.path.exists(os.path.join(run, f"align-{lvl}", "COMPLETE")):
        return None
    a = load_json(os.path.join(run, f"align-{lvl}", "align.json"))
    if not a or "best" not in a:
        return None
    return a["best"]["pass1_confirm"], {"is_shortest": {int(k): v for k, v in a["best"]["passk"].items()}, "goal_reached": {}}


def confirmed(tune):
    """(pass@1 confirm, {metric: {k: pass@k}}) from tune.json's confirmation screen, or None."""
    if not tune or "best" not in tune:
        return None
    if tune["best"].get("method") == "exact":   # exact_tune.py: pass@k from the per-maze shortest-path probabilities
        return tune["best"]["pass1_confirm"], {"is_shortest": {int(k): v for k, v in tune["best"]["passk"].items()}, "goal_reached": {}}
    key = next((k for k in tune["screens"] if k.startswith("confirm-")), None)
    if key is None:
        return None
    r = tune["screens"][key]
    n = r.get("k", 2048)
    curves = {}
    for m in ("is_shortest", "goal_reached"):
        counts = np.asarray(r["per_maze"][m])
        curves[m] = {k: float(np.mean(pass_at_k(counts, n, k))) for k in KS if k <= n}
    return tune["best"]["pass1_confirm"], curves


def xlog(k, kmax=1024):
    return PL + (W - PL - PR) * math.log2(k) / math.log2(kmax)


def chart(series, ylo, xlabel, xticks, xpos, height=H, ylabel="", linear=False):
    """series: (label, color, dash, [(x, y)], width); log10 y from ylo to 1, or linear y from 0 to 1 (linear=True)."""
    def ypos(v):
        if linear:
            return PT + (height - PT - PB) * (1 - v)
        v = max(v, ylo)
        return PT + (height - PT - PB) * (1 - (math.log10(v) - math.log10(ylo)) / (0 - math.log10(ylo)))
    out = [f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="{html.escape(ylabel)}">']
    ticks = [i / 5 for i in range(6)] if linear else [ylo * 10 ** i for i in range(round(-math.log10(ylo)) + 1)]
    for t in ticks:
        y = ypos(t)
        lab = f"{t:g}" if linear or t >= 0.01 else f"{t:.0e}".replace("e-0", "e-")
        out.append(f'<line x1="{PL}" x2="{W - PR}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                   f'<text x="{PL - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{lab}</text>')
    for xv, lab in xticks:
        out.append(f'<text x="{xpos(xv):.1f}" y="{height - PB + 16}" class="tick" text-anchor="middle">{lab}</text>')
    out.append(f'<text x="{(PL + W - PR) / 2:.0f}" y="{height - 4}" class="axis" text-anchor="middle">{html.escape(xlabel)}</text>')
    for label, color, dash, pts, width in series:
        if pts:
            d = f' stroke-dasharray="{dash}"' if dash else ""
            out.append(f'<polyline points="{" ".join(f"{xpos(x):.1f},{ypos(y):.1f}" for x, y in pts)}" fill="none" stroke="{color}" '
                       f'stroke-width="{width}"{d}><title>{html.escape(label)}</title></polyline>')
    out.append("</svg>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--updated", default="")
    ap.add_argument("--rl-grid", action="append", default=[],
                    help="one shortest-path RL grid section: <name>|<rl_grid_status.py --json>|<bases.json>|<lr>|<patience>|<s3 root>"
                         "[|<model label>|<tis|icepop>|<wandb project url>|<grpo|reinforce_batch>]")
    ap.add_argument("--flops-json", default=None, help="rl_flops_curves.py --out: FLOPs-to-performance charts before the RL grids")
    ap.add_argument("--flops-title-note", nargs=2, default=["Compute vs performance", ""])
    ap.add_argument("--flops-section", action="append", default=[], help="another FLOPs chart section: <json>|<title>|<note>")
    ap.add_argument("--flops-best", default=None,
                    help="best-vs-FLOPs scatter section: <json>=<method>;<json>=<method>|<title>|<note> (first method = GRPO marker)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="init seeds per size (up to 5)")
    ap.add_argument("--tune-name", default="tune", help="base dirs <tune-name>-<level> (e.g. tune3: every seed tuned at temperature 1)")
    ap.add_argument("--levels", default=None, help="comma-separated levels to show (default all five)")
    ap.add_argument("--maze-strip", default=None, help="<val parquet>|<note>: five representative mazes under the header (maze_strip.py)")
    ap.add_argument("--title", default="Shortest-Path Maze Bases")
    ap.add_argument("--heading", default="Shortest-path maze bases: small, medium and large at five pass@1 levels")
    ap.add_argument("--lede", default=LEDE)
    ap.add_argument("--demo-shortest", type=float, default=None,
                    help="shortest-path share of the pretraining demonstrations: a reference line on the trajectory charts")
    ap.add_argument("--loss-floor", type=float, nargs=2, default=None, metavar=("TRAIN", "VAL"),
                    help="irreducible loss of the demonstrations (the demonstrator's own NLL): reference lines on the loss charts")
    ap.add_argument("--sizes", default=None, help="comma list of size codes to show (default XS,S,L), e.g. 3XS,2XS,XS,S,L")
    args = ap.parse_args()

    global LEVELS, SEEDS, SIZES
    SEEDS = tuple(args.seeds)
    if args.sizes:
        SIZES = [x for x in ALL_SIZES if x[0] in args.sizes.split(",")]
    if args.levels:
        LEVELS = [x for x in LEVELS if x[0] in args.levels.split(",")]
    tn = args.tune_name
    runs = {}   # (size, seed) -> run dir (latest attempt with screens)
    for size, *_ in SIZES:
        for run in sorted(glob.glob(os.path.join(args.root, size, f"{size}_is*"))):
            seed = int(os.path.basename(run).split("_is")[1].split("_")[0])
            if glob.glob(os.path.join(run, "screen_*.json")) or (size, seed) not in runs:
                runs[(size, seed)] = run
    status = {size: load_json(os.path.join(args.root, size, "status.json")) or {} for size, *_ in SIZES}

    # status grid + per-level curves
    grid_rows, level_sections = [], []
    results = {}
    for size, name, params, color in SIZES:
        for seed in SEEDS:
            run = runs.get((size, seed))
            st = ((status[size].get("runs") or {}).get(os.path.basename(run)) if run else None) or {}
            cells = []
            for lvl, title, tgt in LEVELS:
                c = (((aligned(run, lvl) if tn == "tune" else None) or confirmed(load_json(os.path.join(run, f"{tn}-{lvl}", "tune.json"))))
                     if run else None)
                if c:
                    results[(size, seed, lvl)] = c
                    rel = (c[0] - tgt) / tgt
                    cells.append(f"<td class='{'good' if abs(rel) <= 0.02 else 'warn'}'>{c[0]:.5f}<span class='ci'> {100 * rel:+.1f}%</span></td>")
                else:
                    cells.append(f"<td class='ci'>{html.escape((st.get('levels') or {}).get(lvl, 'waiting'))}</td>")
            ls = st.get("last_screen")
            train = ("training" if st.get("training") else "stopped" if st.get("stopped") else "done" if st.get("finished") else "–") if st else "not started"
            grid_rows.append(f"<tr><td><span class='key'><i style='background:{color}'></i>{name}</span> seed {seed}</td><td>{train}</td>"
                             f"<td>{'–' if not ls else ls[0]}</td><td>{'–' if not ls else f'{ls[1]:.4f}'}</td>{''.join(cells)}</tr>")
    for lvl, title, tgt in LEVELS:
        series, rows = [], []
        for size, name, params, color in SIZES:
            for seed in SEEDS:
                c = results.get((size, seed, lvl))
                if not c:
                    continue
                cur = c[1]["is_shortest"]
                series.append((f"{name} seed {seed}: pass@1 {c[0]:.5f}", color, DASH[seed], sorted(cur.items()), 1.6))
                rows.append(f"<tr><td><span class='key'><i style='background:{color}'></i>{name}</span> s{seed}</td>"
                            + "".join(f"<td>{cur.get(k, float('nan')):.4f}</td>" for k in (1, 4, 16, 64, 256, 1024))
                            + f"<td class='ci'>{c[1]['goal_reached'].get(1, float('nan')):.4f}</td></tr>")
        body = (chart(series, 1e-4, "k (samples per maze)", [(k, str(k)) for k in (1, 4, 16, 64, 256, 1024)], xlog,
                      ylabel=f"shortest-path pass@k at {title}", linear=tn != "tune")
                if series else "<p class='note'>No confirmed base at this level yet.</p>")
        table = ("<div class='tablewrap'><table><thead><tr><th>base</th><th>pass@1</th><th>@4</th><th>@16</th><th>@64</th><th>@256</th>"
                 f"<th>@1024</th><th>goal pass@1</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>") if rows else ""
        curves = [results[(z, sd, lvl)][1]["is_shortest"] for z, *_ in SIZES for sd in SEEDS if (z, sd, lvl) in results]
        spread = ""
        if len(curves) > 1:
            gaps = {k: max(c[k] for c in curves) - min(c[k] for c in curves) for k in (1, 16, 64, 256, 1024) if all(k in c for c in curves)}
            spread = "; spread (max − min over bases): " + ", ".join(f"pass@{k} {v:.3f}" for k, v in gaps.items())
        how = ("exact (teacher-forced) pass@k, temperature 1, 1000 dev mazes" if tn != "tune"
               else "shortest-path pass@k, 1000 dev mazes × 2048 samples (fresh-seed confirmation)")
        level_sections.append(f"<section class='panel'><div class='head'><h2>Level {title}</h2><span class='note'>target shortest-path "
                              f"pass@1 {tgt:g}; {len(series)} of {len(SIZES) * len(SEEDS)} bases tuned{spread}</span></div><div class='charts'><figure>"
                              f"<figcaption>{how}; colour = size; line style = seed (solid, dashed, dotted, dash-dot, short dash = seed 0–4)</figcaption>{body}</figure>"
                              f"<div>{table}</div></div></section>")

    # pretraining trajectories
    traj = []
    for size, name, params, color in SIZES:
        series = []
        steps_max = 100
        for seed in SEEDS:
            run = runs.get((size, seed))
            if not run:
                continue
            d = {}
            for p in glob.glob(os.path.join(run, "screen_*.json")):
                d.update(load_json(p) or {})
            pts = sorted((int(k.split("-")[1]), v) for k, v in d.items() if k.startswith("ckpt-"))
            if not pts:
                continue
            steps_max = max(steps_max, pts[-1][0])
            series.append((f"{name} seed {seed} shortest", color, DASH[seed], [(s, v["is_shortest"]["pass@1"]) for s, v in pts], 1.5))
            series.append((f"{name} seed {seed} goal", "var(--muted)", DASH[seed], [(s, v["goal_reached"]["pass@1"]) for s, v in pts], 1.0))
        xmax = int(math.ceil(steps_max / 500) * 500)
        if series and args.demo_shortest:
            series.append((f"demonstrations: {args.demo_shortest:.0%} shortest", "var(--muted)", "4 4", [(0, args.demo_shortest), (xmax, args.demo_shortest)], 1.0))
        xpos = lambda x, xm=xmax: PL + (W - PL - PR) * x / xm
        ticks = [(t, f"{t / 1000:g}k" if t >= 1000 else str(t)) for t in range(0, xmax + 1, max(500, xmax // 5 // 500 * 500))]
        body = chart(series, 1e-4, "pretraining step (batch 128)", ticks, xpos, height=260, ylabel=f"{name} pretraining") if series else \
            "<p class='note'>Not started.</p>"
        traj.append(f"<figure><figcaption><span class='key'><i style='background:{color}'></i>{name} ({params})</span> · pass@1 by step, "
                    f"colored = shortest path, grey = goal reached; solid / dashed / dotted = seed 0 / 1 / 2"
                    f"{'; dashed grey line = shortest share of the demonstrations' if args.demo_shortest else ''}</figcaption>{body}</figure>")

    # train / val loss: train_metrics.jsonl (every 10 steps) and eval_metrics.jsonl (val loss every 50 steps) of each run
    loss_figs = []
    for size, name, params, color in SIZES:
        curves = {}
        for seed in SEEDS:
            run = runs.get((size, seed))
            if not run:
                continue
            tr = sorted((r["step"], r["train_loss"]) for r in load_jsonl(os.path.join(run, "train_metrics.jsonl")))
            tr = [(tr[i + 9][0], sum(v for _, v in tr[i:i + 10]) / 10) for i in range(0, len(tr) - 9, 10)]   # mean per 100 steps
            va = sorted((r["step"], r["eval/val_loss"]) for r in load_jsonl(os.path.join(run, "eval_metrics.jsonl")) if "eval/val_loss" in r)
            curves[seed] = (tr, va)
        for kind, idx, note in (("train", 0, "mean over each 100 steps"), ("validation", 1, "256 held-out mazes, every 50 steps")):
            pts_all = [(x, y) for c in curves.values() for x, y in c[idx]]
            if not pts_all:
                loss_figs.append(f"<figure><figcaption>{name} ({params}) · {kind} loss</figcaption><p class='note'>No data yet.</p></figure>")
                continue
            xmax = max(500, int(math.ceil(max(x for x, _ in pts_all) / 500) * 500))
            hi = nice_top(max([y for x, y in pts_all if x >= 100] or [y for _, y in pts_all]))
            lo = 0.0
            series = [(f"{name} seed {seed} {kind} loss", SEED_COLORS[seed], [(x, min(y, hi)) for x, y in c[idx]], 1.6)
                      for seed, c in curves.items() if c[idx]]
            loss_figs.append(f"<figure><figcaption>{name} ({params}) · {kind} loss ({note}); latest "
                             + ", ".join(f"s{seed} {c[idx][-1][1]:.4f} @ {c[idx][-1][0]}" for seed, c in curves.items() if c[idx])
                             + "</figcaption>" + line_chart(series, False, f"{name} {kind} loss", xmax, height=220,
                                                            hlines=[(args.loss_floor[idx], f"floor {args.loss_floor[idx]:.3f}")] if args.loss_floor else (),
                                                            xtick=max(500, xmax // 5 // 500 * 500), ylim=(lo, hi)) + "</figure>")
    loss_html = (f"<section class='panel'><h2>Loss curves</h2><p class='note'>Per-token cross-entropy on the noisy demonstrations; "
                 + " ".join(f"<span class='key'><i style='background:var(--seed{i})'></i>seed {i}</span>" for i in SEEDS) +
                 f". The y axis starts at 0 and is capped at the loss after the first "
                 f"100 steps (the first point or two are drawn at the top edge)."
                 f"{' Dashed line: the lowest loss any model can reach, i.e. the demonstrator&#39;s own per-token NLL on these demonstrations.' if args.loss_floor else ''}</p><div class='losses'>{''.join(loss_figs)}</div></section>")

    grids = [(g.split("|") + ["Small (2.2M)", "tis", "", "grpo"])[:10] for g in args.rl_grid]
    rl_html = "".join(rl_section(st, bj, name, lr, pat, root, metric="is-shortest",
                                 reward="binary shortest-path reward (1 only for a shortest path)", model=model, loss=loss, wandb=wb or None,
                                 adv=adv or "grpo")
                      for name, st, bj, lr, pat, root, model, loss, wb, adv in grids if os.path.exists(st))
    from rl_flops_curves import render_best_section, render_section
    flops = ([(args.flops_json, *args.flops_title_note)] if args.flops_json else []) + [tuple(f.split("|", 2)) for f in args.flops_section]
    rl_html = "".join(render_section(*f) for f in flops if os.path.exists(f[0])) + rl_html
    if args.flops_best:
        specs, title, note = args.flops_best.split("|", 2)
        rl_html = render_best_section([tuple(x.split("=", 1)) for x in specs.split(";")], title, note) + rl_html
    page = f"""<title>{html.escape(args.title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{ --bg:#f6f7f8; --panel:#fff; --fg:#1d2329; --muted:#6b7680; --line:#dde2e6; --tiny:#b8405e; --mini:#d1a12a; --small:#7a5bbf; --medium:#1b7f8c; --large:#c2681d; --good:#2f7d4a; --warn:#a8741a; --b4:#9aa3ab; --b16:#d1a12a; --b64:#2f8f5b; --b256:#3a5fc8; --seed0:#3a5fc8; --seed1:#c2681d; --seed2:#2f8f5b; --seed3:#9b4fb8; --seed4:#b8405e;
  --sans:"IBM Plex Sans", system-ui, sans-serif; --mono:"IBM Plex Mono", ui-monospace, monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#14181c; --panel:#1b2026; --fg:#e4e8eb; --muted:#9aa5ae; --line:#2e353c; --tiny:#ec8aa1; --mini:#e6bd55; --small:#a98ae6; --medium:#4fb6c2; --large:#e8944f; --good:#6cc28a; --warn:#e2b04f; --b4:#7d8790; --b16:#e6bd55; --b64:#5cc28a; --b256:#7d9cf0; --seed0:#7d9cf0; --seed1:#e8944f; --seed2:#5cc28a; --seed3:#c59be6; --seed4:#ec8aa1; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#14181c; --panel:#1b2026; --fg:#e4e8eb; --muted:#9aa5ae; --line:#2e353c; --tiny:#ec8aa1; --mini:#e6bd55; --small:#a98ae6; --medium:#4fb6c2; --large:#e8944f; --good:#6cc28a; --warn:#e2b04f; --b4:#7d8790; --b16:#e6bd55; --b64:#5cc28a; --b256:#7d9cf0; --seed0:#7d9cf0; --seed1:#e8944f; --seed2:#5cc28a; --seed3:#c59be6; --seed4:#ec8aa1; color-scheme:dark; }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--sans); font-size:15px; line-height:1.5; }}
main {{ max-width:1120px; margin:0 auto; padding-inline:16px; padding-block:28px 48px; display:grid; gap:20px; }}
h1 {{ font-size:1.5rem; font-weight:600; margin:0; text-wrap:balance; }} h2 {{ font-size:1.08rem; font-weight:600; margin:0; }}
.lede {{ color:var(--muted); max-width:78ch; margin:6px 0 0; }} .stamp {{ color:var(--muted); font:12px var(--mono); }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:16px; display:grid; gap:12px; min-width:0; }}
.head {{ display:flex; flex-wrap:wrap; gap:6px 16px; align-items:baseline; justify-content:space-between; }}
.charts {{ display:grid; grid-template-columns:minmax(0,1.2fr) minmax(0,1fr); gap:14px; align-items:start; }}
.losses {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:14px; }}
@media (max-width:760px) {{ .losses {{ grid-template-columns:minmax(0,1fr); }} }}
.traj {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 330px), 1fr)); gap:14px; }}
@media (max-width:760px) {{ .charts {{ grid-template-columns:minmax(0,1fr); }} }}
figure {{ margin:0; min-width:0; }} figcaption {{ font-size:12px; color:var(--muted); margin-bottom:4px; }}
svg {{ width:100%; height:auto; display:block; }} svg .grid {{ stroke:var(--line); }} svg .tick {{ fill:var(--muted); font:11px var(--mono); }} svg .axis {{ fill:var(--muted); font:12px var(--sans); }}
.key {{ display:inline-flex; gap:6px; align-items:center; }} .key i {{ width:14px; height:3px; display:inline-block; border-radius:2px; }}
.tablewrap {{ overflow-x:auto; min-width:0; }}
table {{ border-collapse:collapse; font:13px var(--mono); font-variant-numeric:tabular-nums; width:100%; }}
th, td {{ text-align:right; padding:4px 9px; border-bottom:1px solid var(--line); white-space:nowrap; }}
th {{ font:500 12px var(--sans); color:var(--muted); }} th:first-child, td:first-child {{ text-align:left; }}
.ci, .note {{ color:var(--muted); }} td.good {{ color:var(--good); }} td.warn {{ color:var(--warn); }}
h3 {{ font-size:1rem; font-weight:600; margin:6px 0 0; }}
.level {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:16px; display:grid; gap:12px; min-width:0; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 14px; font-size:13px; color:var(--muted); }}
svg .zero {{ stroke:var(--muted); stroke-width:1; }} svg .faint {{ stroke-dasharray:2 3; }} svg .refline {{ stroke:var(--muted); stroke-dasharray:4 4; stroke-width:1; }}
.rlcharts {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 330px), 1fr)); gap:14px; }}
details summary {{ cursor:pointer; color:var(--muted); font-size:13px; }} td b {{ font-weight:600; }}
</style>
<main>
<header><h1>{html.escape(args.heading)}</h1>
<p class="lede">{html.escape(args.lede)}</p>
<p class="stamp">{html.escape(args.updated)}</p></header>
<section class="panel"><h2>Status</h2><div class="tablewrap"><table><thead><tr><th>run</th><th>pretraining</th><th>last screened step</th>
<th>shortest p@1 there</th>{''.join(f'<th>{t}</th>' for _, t, _ in LEVELS)}</tr></thead><tbody>{''.join(grid_rows)}</tbody></table></div>
<p class="note">Level cells: confirmed shortest-path pass@1 and its difference from the target (green within ±2%), or the task state.</p></section>
{''.join(level_sections)}
{rl_html}
<section class="panel"><h2>Pretraining trajectories</h2><div class="traj">{''.join(traj)}</div></section>
{loss_html}
</main>"""
    if args.maze_strip:
        from maze_strip import render_strip
        i = page.find("</header>") + len("</header>")
        page = page[:i] + render_strip(*args.maze_strip.split("|", 1)) + page[i:]
    with open(args.out, "w") as f:
        f.write(page)
    print(f"wrote {args.out} ({sum(1 for _ in results)} confirmed bases)")


if __name__ == "__main__":
    main()
