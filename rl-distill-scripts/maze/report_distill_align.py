#!/usr/bin/env python3
"""Page: does a short forward-KL distillation from the small base pull the medium and large bases' pass@k curves onto it?

For each seed and level, the small (XS) exact-tuned base is the teacher; the medium (S) and large (L) bases of the same seed
and level are distilled on its samples (distill_offpolicy.py, full-vocabulary forward KL). Pass@k is exact (exact_pass.py).
  --work   exact_tune.py work dir holding <size>_is<seed>_*/tune3-<level>/tune.json (teacher and students before)
  --res    dir of exact_pass.py results <size>_s<seed>_<level>.json for the distilled students
"""
import argparse
import glob
import html
import json
import os
import statistics as st

from report_shortest import chart, xlog

LEVELS = [("p1", "1%", 0.01), ("p3", "3%", 0.03), ("p10", "10%", 0.10)]
KS = [1, 4, 16, 64, 256, 1024]
SHOW_K = (16, 64, 256, 1024)


def tuned(work, size, seed, lvl):
    f = glob.glob(os.path.join(work, f"{size}_is{seed}_*", f"tune3-{lvl}", "tune.json"))
    if not f:
        return None
    return {int(k): v for k, v in json.load(open(f[0]))["best"]["passk"].items()}


def distilled(res, size, seed, lvl):
    f = os.path.join(res, f"{size}_s{seed}_{lvl}.json")
    if not os.path.exists(f):
        return None
    return {int(k): v for k, v in json.load(open(f))["pass@k_lo"].items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--res", required=True, help="exact_pass results of the 100-step students")
    ap.add_argument("--res1k", default=None, help="exact_pass results of the 1000-step students")
    ap.add_argument("--out1k", default=None, help="1000-step distill output dirs (<size>_s<seed>_<level>/train_metrics.jsonl: exact pass@1 every 100 steps)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--updated", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sections, summary = [], []
    for lvl, title, tgt in LEVELS:
        figs, rows = [], []
        gaps_before, gaps_after = {k: [] for k in SHOW_K}, {k: [] for k in SHOW_K}
        for seed in args.seeds:
            t = tuned(args.work, "XS", seed, lvl)
            if not t:
                continue
            series = [(f"small seed {seed} (teacher)", "var(--small)", "", sorted(t.items()), 2.2)]
            for size, name, color in (("S", "medium", "var(--medium)"), ("L", "large", "var(--large)")):
                b, a100 = tuned(args.work, size, seed, lvl), distilled(args.res, size, seed, lvl)
                a1k = distilled(args.res1k, size, seed, lvl) if args.res1k else None
                a = a1k or a100
                if b:
                    series.append((f"{name} seed {seed} before", color, "4 3", sorted(b.items()), 1.3))
                if a100:
                    series.append((f"{name} seed {seed} after 100 steps", color, "1.5 2.5", sorted(a100.items()), 1.3))
                if a1k:
                    series.append((f"{name} seed {seed} after 1000 steps", color, "", sorted(a1k.items()), 1.9))
                cells = []
                for k in SHOW_K:
                    gb = b[k] - t[k] if b else None
                    ga = a[k] - t[k] if a else None
                    if gb is not None:
                        gaps_before[k].append(abs(gb))
                    if ga is not None:
                        gaps_after[k].append(abs(ga))
                    cells.append(f"<td>{'–' if gb is None else f'{gb:+.3f}'}</td><td class='{'good' if ga is not None and gb is not None and abs(ga) < abs(gb) else ''}'>"
                                 f"{'…' if ga is None else f'{ga:+.3f}'}</td>")
                p1a = f"{a[1]:.4f}" if a else "…"
                rows.append(f"<tr><td><span class='key'><i style='background:{color}'></i>{name}</span> s{seed}</td>"
                            f"<td>{b[1]:.4f}</td><td>{p1a}</td>{''.join(cells)}</tr>" if b else "")
            figs.append(f"<figure><figcaption>seed {seed}: small (thick purple) · dashed = before · dotted = after 100 steps · solid = after 1000 steps</figcaption>"
                        f"{chart(series, 1e-4, 'k (samples per maze)', [(k, str(k)) for k in KS], xlog, ylabel=f'pass@k {title} seed {seed}', linear=True)}</figure>")
        traj = ""
        if args.out1k:
            tser = []
            for seed in args.seeds:
                for size, name, color in (("S", "medium", "var(--medium)"), ("L", "large", "var(--large)")):
                    f = os.path.join(args.out1k, f"{size}_s{seed}_{lvl}", "train_metrics.jsonl")
                    if os.path.exists(f):
                        pts = [(r["step"], r["exact/pass@1"]) for r in map(json.loads, open(f)) if "exact/pass@1" in r]
                        if pts:
                            tser.append((f"{name} seed {seed}", color, {0: "", 1: "5 3", 2: "1.5 3"}[seed], pts, 1.5))
            if tser:
                top = max(y for *_, pts, _ in tser for _, y in pts) * 1.15
                from report_matched_curves import line_chart
                traj = ("<figure><figcaption>exact shortest-path pass@1 during the 1000-step distillation (every 100 steps); "
                        f"target {tgt:g}; solid / dashed / dotted = seed 0 / 1 / 2</figcaption>"
                        + line_chart([(l, c, p, w) for l, c, d, p, w in tser], False, "pass@1", 1000, hlines=[(tgt, f"target {tgt:g}")],
                                     height=230, xlabel="distillation step", xtick=200, ylim=(0.0, top)) + "</figure>")
        n_a = len(gaps_after[1024])
        mean = lambda v: f"{st.mean(v):.3f}" if v else "…"
        summary.append(f"<tr><td>{title}</td>" + "".join(f"<td>{mean(gaps_before[k])}</td><td>{mean(gaps_after[k])}</td>" for k in SHOW_K)
                       + f"<td>{n_a} of {2 * len(args.seeds)}</td></tr>")
        head = "".join(f"<th>Δ@{k} before</th><th>after</th>" for k in SHOW_K)
        sections.append(f"<section class='panel'><h2>Level {title}</h2><p class='note'>Target shortest-path pass@1 {tgt:g}. "
                        f"Table: pass@1 before / after, and each student's pass@k minus the same-seed small base (green = closer after).</p>"
                        f"<div class='figs'>{''.join(figs)}{traj}</div><div class='tablewrap'><table><thead><tr><th>student</th><th>pass@1 before</th>"
                        f"<th>after</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>")
    shead = "".join(f"<th>|Δ@{k}| before</th><th>after</th>" for k in SHOW_K)
    page = f"""<title>Small-to-Large Distill Alignment</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{ --bg:#f6f7f8; --panel:#fff; --fg:#1d2329; --muted:#6b7680; --line:#dde2e6; --small:#7a5bbf; --medium:#1b7f8c; --large:#c2681d; --good:#2f7d4a;
  --sans:"IBM Plex Sans", system-ui, sans-serif; --mono:"IBM Plex Mono", ui-monospace, monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#14181c; --panel:#1b2026; --fg:#e4e8eb; --muted:#9aa5ae; --line:#2e353c; --small:#a98ae6; --medium:#4fb6c2; --large:#e8944f; --good:#6cc28a; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#14181c; --panel:#1b2026; --fg:#e4e8eb; --muted:#9aa5ae; --line:#2e353c; --small:#a98ae6; --medium:#4fb6c2; --large:#e8944f; --good:#6cc28a; color-scheme:dark; }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--sans); font-size:15px; line-height:1.5; }}
main {{ max-width:1180px; margin:0 auto; padding-inline:16px; padding-block:28px 48px; display:grid; gap:20px; }}
h1 {{ font-size:1.45rem; font-weight:600; margin:0; text-wrap:balance; }} h2 {{ font-size:1.08rem; font-weight:600; margin:0; }}
.lede {{ color:var(--muted); max-width:82ch; margin:6px 0 0; }} .stamp {{ color:var(--muted); font:12px var(--mono); }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:16px; display:grid; gap:12px; min-width:0; }}
.note {{ color:var(--muted); font-size:13px; margin:0; max-width:90ch; }}
.figs {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 320px), 1fr)); gap:14px; }}
figure {{ margin:0; min-width:0; }} figcaption {{ font-size:12px; color:var(--muted); margin-bottom:4px; }}
svg {{ width:100%; height:auto; display:block; }} svg .grid {{ stroke:var(--line); }} svg .tick {{ fill:var(--muted); font:11px var(--mono); }} svg .axis {{ fill:var(--muted); font:12px var(--sans); }}
.tablewrap {{ overflow-x:auto; min-width:0; }}
table {{ border-collapse:collapse; font:13px var(--mono); font-variant-numeric:tabular-nums; width:100%; }}
th, td {{ text-align:right; padding:5px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }}
th {{ font:500 12px var(--sans); color:var(--muted); }} th:first-child, td:first-child {{ text-align:left; font-family:var(--sans); }}
td.good {{ color:var(--good); font-weight:500; }}
.key {{ display:inline-flex; align-items:center; gap:6px; }} .key i {{ width:14px; height:3px; display:inline-block; border-radius:2px; }}
</style>
<main>
<header><h1>Distilling the small base into medium and large, per seed and level (25×25)</h1>
<p class="lede">For each seed and level, the small model's exact-tuned base is the teacher; the medium and large bases of the same
seed and level are distilled on 6,400 of its samples (one per prompt, RL training prompts) with full-vocabulary forward KL:
100 steps, batch 64, constant learning rate 1e-5 (the RL rate), no warmup. Pass@k is exact (teacher-forced, temperature 1)
on the 1000 dev mazes. Question: do the medium and large pass@k curves move onto the small one?</p>
<p class="stamp">{html.escape(args.updated)}</p></header>
<section class="panel"><h2>Summary: mean |pass@k − small| over students</h2><p class="note">{html.escape(args.note)}</p>
<div class="tablewrap"><table><thead><tr><th>level</th>{shead}<th>students done</th></tr></thead><tbody>{''.join(summary)}</tbody></table></div></section>
{''.join(sections)}
</main>"""
    with open(args.out, "w") as f:
        f.write(page)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
