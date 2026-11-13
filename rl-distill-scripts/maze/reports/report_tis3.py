#!/usr/bin/env python3
"""Page: 25x25 GRPO + TIS RL from the per-seed tune3 bases (grids t3s small / t3m medium, 3 seeds, run seed i from base seed i).

Reads rl_grid_status.py --json rows (per run: state, best validation is_shortest, history, entropy) for each size and the
bases' tune.json (exact shortest-path pass@1). One chart per level and size (validation is_shortest vs step; colour =
prompts per step, one line per seed), a best-value summary (mean over seeds, min-max) and a per-run status table.
  report_tis3.py --size "Small (2.2M)|<status json>|16 64 256|XS" --size "Medium (3.9M)|<status json>|8 32 128|S" \
      --work <exact_tune work dir> --maze-strip <parquet> --out page.html
"""
import argparse
import html
import json
import math
import os
import statistics as st

from maze_strip import render_strip
from report_matched_curves import line_chart

LEVELS = [("p1", "1%", 0.01), ("p3", "3%", 0.03), ("p10", "10%", 0.10)]
SHADES = ["var(--b0)", "var(--b1)", "var(--b2)", "var(--b3)"]
SIZE_SHADES = [["var(--s0)", "var(--s1)", "var(--s2)", "var(--s3)"], ["var(--m0)", "var(--m1)", "var(--m2)", "var(--m3)"]]  # small, medium
CAP = 10000


def base_pass1(work, size, seed, lvl):
    f = os.path.join(work, f"{size}_is{seed}_a11090115", f"tune3-{lvl}", "tune.json")
    return json.load(open(f))["best"]["passk"]["1"] if os.path.exists(f) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", action="append", required=True, help="label|status json|batches|size code")
    ap.add_argument("--work", required=True)
    ap.add_argument("--maze-strip", default=None)
    ap.add_argument("--flops", default=None, help="rl_flops_curves.py --out json (grids labelled as the --size labels)")
    ap.add_argument("--updated", default="")
    ap.add_argument("--wandb", default="https://wandb.ai/rl-distill/tailrl-text-maze-dfs25-eps05")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sizes = []
    for spec in args.size:
        label, path, batches, code = spec.split("|")
        rows = json.load(open(path)) if os.path.exists(path) else []
        for r in rows:  # 10k-step budget: curves and bests only use validations at steps <= CAP
            r["history"] = [(x, y) for x, y in (r.get("history") or []) if x <= CAP]
            if r["history"] and r["state"] == "running":
                r["best_step"], r["best_value"] = max(r["history"], key=lambda p: (p[1], -p[0]))
        sizes.append((label, {(r["level"], int(r["batch"]), int(r["seed"])): r for r in rows}, [int(b) for b in batches.split()], code))

    # progress
    prog = []
    for label, runs, _, _ in sizes:
        states = [r["state"] for r in runs.values()]
        done = sum(s not in ("running", "queued") for s in states)
        prog.append(f"<div class='stat'><b>{html.escape(label)}</b><span>{done} of {len(states) or 27} done · "
                    f"{states.count('running')} running · {states.count('queued')} queued</span></div>")

    # summary table: best is_shortest, mean over seeds (min-max), per level x size x batch
    head = "".join(f"<th colspan='{len(b)}'>{html.escape(lab)}</th>" for lab, _, b, _ in sizes)
    sub = "".join(f"<th>{b} prompts</th>" for _, _, bs, _ in sizes for b in bs)
    body = []
    for lvl, title, _ in LEVELS:
        cells = []
        for _, runs, bs, _ in sizes:
            for b in bs:
                rs = [runs.get((lvl, b, s)) for s in (0, 1, 2)]
                vals = [r["best_value"] for r in rs if r and r.get("best_value") is not None]
                live = sum(1 for r in rs if r and r["state"] == "running")
                capped = sum(1 for r in rs if r and r["state"] == "max_steps")
                if not vals:
                    cells.append("<td class='muted'>…</td>")
                    continue
                tag = (f" <span class='muted'>[{capped} at cap]</span>" if capped else "") + (" <span class='live'>●</span>" if live else "")
                cells.append(f"<td>{st.mean(vals):.3f} <span class='muted'>({min(vals):.3f}–{max(vals):.3f})</span>{tag}</td>")
        body.append(f"<tr><td>{title}</td>{''.join(cells)}</tr>")
    summary = (f"<div class='tablewrap'><table><thead><tr><th rowspan='2'>start</th>{head}</tr><tr>{sub}</tr></thead>"
               f"<tbody>{''.join(body)}</tbody></table></div>")

    # curves
    sections = []
    for lvl, title, tgt in LEVELS:
        figs = []
        for label, runs, bs, code in sizes:
            series, top = [], tgt * 1.5
            for bi, b in enumerate(bs):
                for s in (0, 1, 2):
                    r = runs.get((lvl, b, s))
                    if not r or not r.get("history"):
                        continue
                    pts = [(x, y) for x, y in r["history"]]
                    top = max(top, max(y for _, y in pts))
                    end = "running" if r["state"] == "running" else "stopped"
                    series.append((f"{b} prompts, seed {s}", SHADES[bi], pts, 1.4, end))
            xmax = max([CAP // 5] + [x for *_, pts, _, _ in series for x, _ in pts])
            xmax = min(CAP, -(-xmax // 1000) * 1000)
            starts = [base_pass1(args.work, code, s, lvl) for s in (0, 1, 2)]
            starts = ", ".join(f"{v:.4f}" for v in starts if v is not None)
            legend = " · ".join(f"<span class='key'><i style='background:{SHADES[i]}'></i>{b} prompts</span>" for i, b in enumerate(bs))
            chart = (line_chart(series, False, "validation is-shortest", xmax, hlines=[(tgt, f"start {tgt:g}")], height=240,
                                xlabel="RL step", xtick=max(500, xmax // 5), ylim=(0.0, top * 1.08)) if series
                     else "<p class='muted'>No validation yet.</p>")
            figs.append(f"<figure><figcaption><b>{html.escape(label)}</b> · {legend}<br>"
                        f"<span class='muted'>one line per seed; base exact pass@1 by seed: {starts}; open circle = running</span>"
                        f"</figcaption>{chart}</figure>")
        sections.append(f"<section class='panel'><h2>Start {title}</h2><div class='figs'>{''.join(figs)}</div></section>")

    # seed means on a log essential-FLOPs axis: one chart per level, both sizes, one line per prompts-per-step setting
    fl = json.load(open(args.flops)) if args.flops and os.path.exists(args.flops) else {}
    fruns = {(label, r["level"], r["batch"], r["seed"]): r for label, g in fl.items() for r in g["runs"]}
    mean_figs = []
    for lvl, title, tgt in LEVELS:
        series, keys, top, xlo, xhi = [], [], tgt * 1.5, None, None
        for si, (label, runs, bs, _) in enumerate(sizes):
            for bi, b in enumerate(bs):
                fr = [fruns.get((label, lvl, b, s)) for s in (0, 1, 2)]
                if not all(fr):
                    continue
                hs = [{t: (f, y) for t, (f, y) in zip(r["steps"], r["points"]) if t <= CAP} for r in fr]
                steps = sorted(set.intersection(*(set(h) for h in hs)))
                # x = geometric mean of the seeds' cumulative FLOPs at the step, y = mean validation
                pts = [(math.exp(st.mean(math.log(h[t][0]) for h in hs)), st.mean(h[t][1] for h in hs)) for t in steps]
                if len(pts) < 2:
                    continue
                live = any((runs.get((lvl, b, s)) or {}).get("state") == "running" for s in (0, 1, 2))
                color = SIZE_SHADES[min(si, 1)][bi]
                series.append((f"{label.split(' ')[0]} {b} prompts", color, pts, 1.8, "running" if live else "stopped"))
                keys.append(f"<span class='key'><i style='background:{color}'></i>{html.escape(label.split(' ')[0])} {b}</span>")
                top = max(top, max(y for _, y in pts))
                xlo = min(xlo or pts[0][0], pts[0][0]); xhi = max(xhi or pts[-1][0], pts[-1][0])
        chart = "<p class='muted'>No FLOPs data yet.</p>"
        if series:
            xmin, xmax = 10 ** math.floor(math.log10(xlo)), 10 ** math.ceil(math.log10(xhi))
            chart = line_chart(series, False, "validation is-shortest (seed mean)", xmax, hlines=[(tgt, f"start {tgt:g}")], height=260,
                               xlabel="essential RL FLOPs (log)", xmin=xmin, ylim=(0.0, top * 1.08))
        mean_figs.append(f"<figure><figcaption><b>Start {title}</b> · {' · '.join(keys)}</figcaption>{chart}</figure>")
    means = (f"<section class='panel'><h2>Seed-mean curves vs compute (log FLOPs)</h2><p class='muted' style='margin:0;font-size:13px'>"
             f"Each line is one configuration (size × prompts per step): at every validation step all 3 seeds have reached, the "
             f"mean validation is-shortest of the seeds against the geometric mean of their cumulative essential RL FLOPs. "
             f"Essential FLOPs: generation for every sample (prompt prefilled once per group) plus training (3 forward passes) "
             f"on samples of mixed-correctness groups only; attention included; validation, padding and recompute excluded "
             f"(same convention as the other maze pages). Lighter to darker = more prompts per step; green = small, purple = "
             f"medium. Open circle = still running.</p>"
             f"<div class='figs'>{''.join(mean_figs)}</div></section>")

    # per-run status
    trs = []
    for label, runs, bs, _ in sizes:
        for (lvl, b, s), r in sorted(runs.items(), key=lambda kv: ([l for l, _, _ in LEVELS].index(kv[0][0]), kv[0][1], kv[0][2])):
            fmt = lambda v, f="{:.4f}": "–" if v is None else f.format(v)
            trs.append(f"<tr><td>{html.escape(label.split(' ')[0])}</td><td><a href='{args.wandb}' target='_blank' rel='noopener'>"
                       f"{html.escape(r['suite'])}</a></td><td>{html.escape(r['state'])}</td><td>{fmt(r.get('last_val_step'), '{}')}</td>"
                       f"<td>{fmt(r.get('best_value'))}</td><td>{fmt(r.get('best_step'), '{}')}</td></tr>")
    status = (f"<div class='tablewrap tall'><table><thead><tr><th>size</th><th>run</th><th>state</th><th>last validated step</th>"
              f"<th>best is-shortest</th><th>best step</th></tr></thead><tbody>{''.join(trs)}</tbody></table></div>")

    strip = render_strip(args.maze_strip, "25×25 depth-first mazes with a uniform 0–20% wall knockout.") if args.maze_strip else ""
    page = f"""<title>Maze TIS RL Seeds</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{ --bg:#f5f6f4; --panel:#fff; --fg:#1c2421; --muted:#68746f; --line:#dce1de; --accent:#2f6f5e; --live:#c0571b;
  --b0:#a9cdc0; --b1:#5fa38c; --b2:#2c7560; --b3:#103a2e;
  --s0:#a9cdc0; --s1:#5fa38c; --s2:#2c7560; --s3:#103a2e; --m0:#cdbfe6; --m1:#9b80cf; --m2:#6a45ab; --m3:#3a1f6e;
  --sans:"IBM Plex Sans", system-ui, sans-serif; --mono:"IBM Plex Mono", ui-monospace, monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#121715; --panel:#1a201d; --fg:#e3e9e6; --muted:#97a39e; --line:#2b3330; --accent:#6cc0a6; --live:#f0915a;
  --b0:#35604f; --b1:#4f9a81; --b2:#86cdb5; --b3:#c8f0e2; --s0:#35604f; --s1:#4f9a81; --s2:#86cdb5; --s3:#c8f0e2; --m0:#4f3d75; --m1:#7d62b8; --m2:#ad95e6; --m3:#ddd0fa; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#121715; --panel:#1a201d; --fg:#e3e9e6; --muted:#97a39e; --line:#2b3330; --accent:#6cc0a6; --live:#f0915a;
  --b0:#35604f; --b1:#4f9a81; --b2:#86cdb5; --b3:#c8f0e2; --s0:#35604f; --s1:#4f9a81; --s2:#86cdb5; --s3:#c8f0e2; --m0:#4f3d75; --m1:#7d62b8; --m2:#ad95e6; --m3:#ddd0fa; color-scheme:dark; }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--sans); font-size:15px; line-height:1.5; }}
main {{ max-width:1180px; margin:0 auto; padding-inline:16px; padding-block:28px 48px; display:grid; gap:20px; }}
h1 {{ font-size:1.45rem; font-weight:600; margin:0; text-wrap:balance; }} h2 {{ font-size:1.08rem; font-weight:600; margin:0; }}
.lede {{ color:var(--muted); max-width:85ch; margin:6px 0 0; }} .stamp {{ color:var(--muted); font:12px var(--mono); }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:16px; display:grid; gap:12px; min-width:0; }}
.stats {{ display:flex; flex-wrap:wrap; gap:10px 28px; }} .stat {{ display:grid; }} .stat span {{ font:13px var(--mono); color:var(--muted); }}
.muted {{ color:var(--muted); }} .live {{ color:var(--live); }}
.figs {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap:16px; }}
figure {{ margin:0; min-width:0; }} figcaption {{ font-size:12.5px; margin-bottom:4px; }}
svg {{ width:100%; height:auto; display:block; }} svg .grid {{ stroke:var(--line); }} svg .tick {{ fill:var(--muted); font:11px var(--mono); }} svg .axis {{ fill:var(--muted); font:12px var(--sans); }}
.tablewrap {{ overflow-x:auto; min-width:0; }} .tall {{ max-height:520px; overflow-y:auto; }}
table {{ border-collapse:collapse; font:13px var(--mono); font-variant-numeric:tabular-nums; width:100%; }}
th, td {{ text-align:right; padding:5px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }}
th {{ font:500 12px var(--sans); color:var(--muted); position:sticky; top:0; background:var(--panel); }} th:first-child, td:first-child {{ text-align:left; font-family:var(--sans); }}
td a {{ color:var(--accent); }}
.key {{ display:inline-flex; align-items:center; gap:5px; }} .key i {{ width:14px; height:3px; display:inline-block; border-radius:2px; }}
dl {{ display:grid; grid-template-columns:max-content 1fr; gap:4px 16px; margin:0; font-size:13.5px; }} dt {{ color:var(--muted); }} dd {{ margin:0; }}
</style>
<main>
<header><h1>25×25 maze RL with TIS: small and medium, three separately pretrained seeds</h1>
<p class="lede">GRPO (16 samples per prompt) with token-level truncated importance sampling against the rollout engine
(w = min(π<sub>train</sub>/π<sub>rollout</sub>, 2), one update per batch, no PPO clip), reward 1 only for a shortest path.
Each run starts from its own seed's exact-tuned base (run seed i ↔ base seed i) at shortest-path pass@1 1%, 3% or 10%.
Validation: shortest-path rate (mean@16) on 500 dev mazes every 20 steps.</p>
<p class="stamp">{html.escape(args.updated)}</p></header>
{strip}
<section class="panel"><h2>Progress</h2><div class="stats">{''.join(prog)}</div></section>
<section class="panel"><h2>Best validation is-shortest</h2><p class="muted" style="margin:0;font-size:13px">Mean of 3 seeds (min–max) of each run's best validation so far; ● = at least one seed still running; [n at cap] = seeds that reached 20,000 steps (state max_steps).</p>{summary}</section>
{means}
{''.join(sections)}
<section class="panel"><h2>Setup</h2><dl>
<dt>Objective</dt><dd>−Â · min(π<sub>train</sub>/π<sub>rollout</sub>, 2) · log π<sub>train</sub> per token, weight detached; Â = group z-score (16 samples); no KL, no entropy bonus; token-mean</dd>
<dt>Optimizer</dt><dd>lr 1e-5 constant after 20 warm-up steps; one optimizer step per rollout batch (no old-log-prob pass)</dd>
<dt>Prompts per step</dt><dd>small 16 / 64 / 256 / 1024, medium 8 / 32 / 128 / 512 (× 16 samples); the 1024 / 512 rung was added on 11-12 at 16:30 UTC, one run per GPU (update micro-batch ≤ 1024 / 512, 12 reward workers)</dd>
<dt>Stopping</dt><dd>budget 10,000 steps per run (bests and curves use validations up to step 10,000). All RL was stopped on 11-12 at 18:40 UTC: the 9 small 256-prompt runs (at 5.8k–6.8k steps), 4 medium 128-prompt runs (at ~9.5k) and the 1,024 / 512-prompt rung are unfinished; the cells marked ● are their best so far. Early stopping (1,000 steps without a new best) was turned off on 11-12 at 07:35 UTC and the 4 runs it had stopped were resumed; a 20k extension started at 15:45 UTC was cancelled at 18:30 UTC (runs past 10k are reported at 10k)</dd>
<dt>Lengths</dt><dd>prompt 654 tokens, response cap 300; temperature 1 for training and validation</dd>
<dt>Throughput</dt><dd>shared-prompt CUDA-graph decoder; update micro-batch ≤ 512 (small) / 256 (medium) samples; 4 reward workers; 1-GPU pods, 2–5 runs per GPU</dd>
<dt>Bases</dt><dd><code>s3://scale-ml/genai/rl-distill/maze/rl_{{small,medium}}_dfs25_tis3s_v1/bases/&lt;level&gt;_s&lt;seed&gt;</code> (from <code>pt_dfs25_eps05_v1/&lt;size&gt;_is&lt;seed&gt;/tune3-&lt;level&gt;</code>)</dd>
<dt>Logs</dt><dd><a href="{args.wandb}" target="_blank" rel="noopener">W&amp;B tailrl-text-maze-dfs25-eps05</a> (runs st3-* / mt3-*)</dd>
</dl></section>
<section class="panel"><h2>Runs</h2>{status}</section>
</main>"""
    with open(args.out, "w") as f:
        f.write(page)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
