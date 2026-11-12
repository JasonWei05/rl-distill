#!/usr/bin/env python3
"""HTML page of the goal_reached pass@k curves of matched bases, one panel per starting level (text-maze study).

Per level: pass@k for k = 1, 2, 4, ..., samples per maze (unbiased estimator) for every model, with a bootstrap 95% band
over mazes; the paired difference to the level's first model with its band against the +-tol band; the 1000-maze values
recorded when the bases were selected (Jiahao's manifests, k <= 1024) as hollow markers; a table of the main k.
Levels whose models are not all screened yet are drawn as pending.

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/report_matched_curves.py --screens <screen.json> \
        --level "p01=0.1%:medium_p01,large_p01" ... --ref-json <rl_curves_v1.json> --ref-manifest <extra/matched_manifest.json> \
        --out <page.html>
"""

import argparse
import glob
import html
import json
import math
import os

import numpy as np

from compare_matched_curves import pass_at_k

SET_VALUES = {"p01": 0.001, "p03": 0.003, "p1": 0.01, "p3": 0.03, "p5": 0.055, "p10": 0.10}
COLORS = {"small": "var(--small)", "medium": "var(--medium)", "large": "var(--large)"}
W, H, PAD_L, PAD_R, PAD_T, PAD_B = 520, 300, 52, 14, 14, 34
DH = 140


def size_of(label):
    return label.split("_")[0]


def curve_stats(counts, n, ks, boot_idx):
    out = {"pass": [], "lo": [], "hi": [], "per_k": {}}
    for k in ks:
        pk = pass_at_k(counts, n, k)
        out["per_k"][k] = pk
        lo, hi = np.percentile(pk[boot_idx].mean(axis=1), [2.5, 97.5])
        out["pass"].append(float(pk.mean()))
        out["lo"].append(float(lo))
        out["hi"].append(float(hi))
    return out


def scale_of(p, n_mazes):
    """Tolerance scale of a pass@k value: the success rate at the sparse end, the miss rate near saturation. The miss
    side is floored at one maze's worth (1 / n_mazes): a smaller miss rate is below what the maze set can resolve."""
    return max(min(p, max(1.0 - p, 1.0 / n_mazes)), 1e-12)


def sampling_se(counts_a, counts_b, n, ks, n_boot, rng):
    """SE of mean(pass@k_b - pass@k_a) from re-drawing every maze's successes (Binomial(n, (c + 0.5) / (n + 1))):
    the sampling noise of the comparison on this fixed set of mazes."""
    pa, pb = (np.asarray(counts_a) + 0.5) / (n + 1), (np.asarray(counts_b) + 0.5) / (n + 1)
    draws = {k: [] for k in ks}
    for _ in range(n_boot):
        ca, cb = rng.binomial(n, pa), rng.binomial(n, pb)
        for k in ks:
            draws[k].append(pass_at_k(cb, n, k).mean() - pass_at_k(ca, n, k).mean())
    return {k: float(np.std(v)) for k, v in draws.items()}


SMALL_COLORS = ["var(--small)", "var(--small2)", "var(--small3)"]


def small_candidates(select_root, level_key, target, max_k):
    """Per small seed: the averaged candidate closest to the set value in select_<level>_set.json, measured at the largest
    samples per maze available (the 2048-sample finalists once they exist). Returns [(label, seed, center, n, counts)]."""
    out = []
    for seed in range(3):
        tune = os.path.join(select_root, f"small_ds0_is{seed}", f"tune-{level_key}", "tune.json")
        if os.path.exists(tune):  # a finished pass@1 tuning run: its fresh-seed confirmation of the chosen interpolation
            try:
                tj = json.load(open(tune))
            except (json.JSONDecodeError, OSError):
                tj = {}
            if "best" in tj:
                conf = next(v for k, v in tj["screens"].items() if k.startswith(f"confirm-a{tj['best']['alpha']:.4f}"))
                out.append((f"small_s{seed}", seed, f"tuned α={tj['best']['alpha']:.3f}", conf["k"], np.asarray(conf["per_maze"]["goal_reached"])))
                continue
        path = os.path.join(select_root, f"small_ds0_is{seed}", f"select_{level_key}_set.json")
        if not os.path.exists(path):
            continue
        try:
            sel = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        cands = [(int(k.split("@k")[1]), int(k.split("-")[1]), v) for k, v in sel.items() if k.startswith("avg-") and "@k" in k]
        if not cands:
            continue
        n_best = max(n for n, _, _ in cands)
        n, center, v = min(((n, c, v) for n, c, v in cands if n == n_best), key=lambda t: abs(t[2]["goal_reached"]["pass@1"] - target))
        out.append((f"small_s{seed}", seed, f"avg-{center}-w60", n, np.asarray(v["per_maze"]["goal_reached"])))
    return out


def reference_points(args):
    """Level -> size -> {k: value} recorded at selection on all 1000 dev mazes."""
    ref = {}
    name_map = {"S0": "medium", "L0": "large"}
    if args.ref_json:
        starts = json.load(open(args.ref_json))["starts"]
        for lvl, sizes in starts.items():
            for s, m in sizes.items():
                ref.setdefault(lvl, {})[name_map[s]] = {int(k.split("@")[1]): v for k, v in m.items() if k.startswith("pass@")}
    if args.ref_manifest:
        pairs = json.load(open(args.ref_manifest))["pairs"]
        for lvl in ("p03", "p3"):
            for s in ("S0", "L0"):
                m = pairs.get(lvl, {}).get(s, {}).get("metrics_k1024", {}).get("goal_reached", {})
                if m:
                    ref.setdefault(lvl, {})[name_map[s]] = {int(k.split("@")[1]): v for k, v in m.items() if k.startswith("pass@")}
    return ref


def xpos(k, kmax):
    return PAD_L + (W - PAD_L - PAD_R) * math.log2(k) / math.log2(kmax)


def svg_curves(level, kmax):
    lows = [min(m["lo"]) for m in level["models"].values()] + [min(e["pass"]) for e in level.get("extra", {}).values()]
    ymin = 10 ** math.floor(math.log10(max(1e-4, min(lows) * 0.8)))
    def ypos(v):
        v = max(v, ymin)
        return PAD_T + (H - PAD_T - PAD_B) * (1 - (math.log10(v) - math.log10(ymin)) / (0 - math.log10(ymin)))
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="pass@k curves for {html.escape(level["title"])}">']
    decade = ymin
    while decade <= 1.0 + 1e-9:
        y = ypos(decade)
        parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        lab = f"{decade:g}" if decade >= 0.01 else f"{decade:.0e}".replace("e-0", "e-")
        parts.append(f'<text x="{PAD_L - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{lab}</text>')
        decade *= 10
    for k in level["ks"]:
        x = xpos(k, kmax)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{PAD_T}" y2="{H - PAD_B}" class="grid faint"/>')
        if int(math.log2(k)) % 2 == 0 or k == kmax:
            parts.append(f'<text x="{x:.1f}" y="{H - PAD_B + 16}" class="tick" text-anchor="middle">{k}</text>')
    parts.append(f'<text x="{(PAD_L + W - PAD_R) / 2:.0f}" y="{H - 4}" class="axis" text-anchor="middle">k (samples)</text>')
    for lab, m in level["models"].items():
        col = COLORS[size_of(lab)]
        band = [(xpos(k, kmax), ypos(h)) for k, h in zip(level["ks"], m["hi"])] + \
               [(xpos(k, kmax), ypos(l)) for k, l in reversed(list(zip(level["ks"], m["lo"])))]
        parts.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in band)}" fill="{col}" fill-opacity="0.16"/>')
        pts = " ".join(f"{xpos(k, kmax):.1f},{ypos(v):.1f}" for k, v in zip(level["ks"], m["pass"]))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        for k, v in zip(level["ks"], m["pass"]):
            parts.append(f'<circle cx="{xpos(k, kmax):.1f}" cy="{ypos(v):.1f}" r="2.6" fill="{col}"/>')
        for k, v in sorted(level["ref"].get(size_of(lab), {}).items()):
            if k <= kmax:
                parts.append(f'<circle cx="{xpos(k, kmax):.1f}" cy="{ypos(v):.1f}" r="5" fill="none" stroke="{col}" stroke-width="1.4"/>')
    for lab, e in level.get("extra", {}).items():
        pts = " ".join(f"{xpos(k, kmax):.1f},{ypos(v):.1f}" for k, v in zip(e["ks"], e["pass"]))
        dash = "" if e.get("chosen") else ' stroke-dasharray="5 3"'
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{e["color"]}" stroke-width="{2.2 if e.get("chosen") else 1.4}"{dash}/>')
        for k, v in zip(e["ks"], e["pass"]):
            parts.append(f'<rect x="{xpos(k, kmax) - 2.4:.1f}" y="{ypos(v) - 2.4:.1f}" width="4.8" height="4.8" fill="{e["color"]}"/>')
    parts.append("</svg>")
    return "".join(parts)


def svg_delta(level, kmax, tol_rel):
    """Relative difference (% of min(p, 1 - p)) with the +-tol band and the 2-sigma sampling-noise envelope."""
    d = level["delta"]
    rel, noise = [100 * x for x in d["rel"]], [100 * x for x in d["noise2"]]
    tol = 100 * tol_rel
    sm_rel = [100 * x for x in level["delta_small"]["rel"]] if level.get("delta_small") else []
    lim = min(max(tol * 1.8, max(abs(x) for x in rel + sm_rel) * 1.15, min(max(noise) * 1.1, 60)), 80)
    def ypos(v):
        v = max(-lim, min(lim, v))
        return PAD_T + (DH - PAD_T - PAD_B) * (1 - (v + lim) / (2 * lim))
    parts = [f'<svg viewBox="0 0 {W} {DH}" role="img" aria-label="relative difference, large vs medium">']
    parts.append(f'<rect x="{PAD_L}" y="{ypos(tol):.1f}" width="{W - PAD_L - PAD_R}" height="{ypos(-tol) - ypos(tol):.1f}" class="tolband"/>')
    step = 5 if lim <= 15 else (10 if lim <= 40 else 20)
    v = -(lim // step) * step
    while v <= lim:
        y = ypos(v)
        parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}" class="{"zero" if v == 0 else "grid"}"/>')
        parts.append(f'<text x="{PAD_L - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{v:+.0f}%</text>')
        v += step
    for k in level["ks"]:
        if int(math.log2(k)) % 2 == 0 or k == kmax:
            parts.append(f'<text x="{xpos(k, kmax):.1f}" y="{DH - PAD_B + 16}" class="tick" text-anchor="middle">{k}</text>')
    for sign in (1, -1):
        pts = " ".join(f"{xpos(k, kmax):.1f},{ypos(sign * nz):.1f}" for k, nz in zip(level["ks"], noise))
        parts.append(f'<polyline points="{pts}" fill="none" class="noiseline"/>')
    sm = level.get("delta_small")
    if sm:
        pts_s = " ".join(f"{xpos(k, kmax):.1f},{ypos(100 * v):.1f}" for k, v in zip(sm["ks"], sm["rel"]))
        parts.append(f'<polyline points="{pts_s}" fill="none" stroke="{sm["color"]}" stroke-width="2"/>')
        for k, v in zip(sm["ks"], sm["rel"]):
            parts.append(f'<rect x="{xpos(k, kmax) - 2.4:.1f}" y="{ypos(100 * v) - 2.4:.1f}" width="4.8" height="4.8" fill="{sm["color"] if abs(100 * v) <= tol else "var(--bad)"}"/>')
    pts = " ".join(f"{xpos(k, kmax):.1f},{ypos(v):.1f}" for k, v in zip(level["ks"], rel))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--large)" stroke-width="2"/>')
    for k, v in zip(level["ks"], rel):
        parts.append(f'<circle cx="{xpos(k, kmax):.1f}" cy="{ypos(v):.1f}" r="2.6" fill="{"var(--large)" if abs(v) <= tol else "var(--bad)"}"/>')
    parts.append("</svg>")
    return "".join(parts)


def nice_top(v):
    """Smallest 1 / 1.2 / 1.5 / 2 / 2.5 / 3 / 4 / 5 / 6 / 8 x 10^k at or above v (an axis maximum)."""
    mag = 10 ** math.floor(math.log10(v))
    return next(m * mag for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if m * mag >= v * (1 - 1e-9))


def line_chart(series, ylog, ylabel, xmax, hlines=(), height=240, xlabel="pretraining step", xtick=5000, ylim=None, xmin=None):
    """series: list of (label, color, [(x, y)]) or (label, color, [(x, y)], stroke width[, end]); hlines: list of (y, text).
    end = "stopped" draws a filled dot at the last point, "running" a dashed line ending in an open circle.
    xmin set: log10 x axis from xmin to xmax with a tick per decade (xtick ignored)."""
    ys = [y for s in series for _, y in s[2]] + [y for y, _ in hlines]
    if ylog:
        lo = 10 ** math.floor(math.log10(max(1e-4, min([y for y in ys if y > 0] or [1e-4]))))
        hi = 1.0
        def ypos(v):
            v = max(v, lo)
            return PAD_T + (height - PAD_T - PAD_B) * (1 - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)))
        ticks, t = [], lo
        while t <= hi * 1.0001:
            ticks.append(t); t *= 10
    else:
        lo, hi = ylim or (min(ys) * 0.95, max(ys) * 1.03)
        def ypos(v):
            return PAD_T + (height - PAD_T - PAD_B) * (1 - (v - lo) / (hi - lo))
        raw = (hi - lo) / 5   # ~5 gridlines at a 1 / 2 / 2.5 / 5 x 10^k spacing
        mag = 10 ** math.floor(math.log10(raw))
        step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw * (1 - 1e-9))
        ticks = [k * step for k in range(math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9) + 1)]
        decimals = max(0, -math.floor(math.log10(step) + 1e-9) + (1 if round(step / 10 ** math.floor(math.log10(step) + 1e-9), 3) == 2.5 else 0))
    def xpos_s(x):
        if xmin:
            return PAD_L + (W - PAD_L - PAD_R) * (math.log10(max(x, xmin)) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin))
        return PAD_L + (W - PAD_L - PAD_R) * x / xmax
    parts = [f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="{html.escape(ylabel)} by {html.escape(xlabel)}">']
    for t in ticks:
        y = ypos(t)
        lab = (f"{t:g}" if t >= 0.01 else f"{t:.0e}".replace("e-0", "e-")) if ylog else f"{t:.{decimals}f}"
        parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{PAD_L - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{lab}</text>')
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    xticks = ([(10.0 ** e, "10" + str(e).translate(sup)) for e in range(math.ceil(math.log10(xmin) - 1e-9), math.floor(math.log10(xmax) + 1e-9) + 1)]
              if xmin else [(xt, f"{xt / 1000:g}k" if xt >= 1000 else str(xt)) for xt in range(0, int(xmax) + 1, xtick)])
    for xt, lab in xticks:
        x = xpos_s(xt)
        if xmin:
            parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{PAD_T}" y2="{height - PAD_B}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - PAD_B + 16}" class="tick" text-anchor="middle">{lab}</text>')
    parts.append(f'<text x="{(PAD_L + W - PAD_R) / 2:.0f}" y="{height - 4}" class="axis" text-anchor="middle">{html.escape(xlabel)}</text>')
    for y, text in hlines:
        yy = ypos(y)
        parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{yy:.1f}" y2="{yy:.1f}" class="refline"/>')
        parts.append(f'<text x="{W - PAD_R - 4}" y="{yy - 4:.1f}" class="tick" text-anchor="end">{html.escape(text)}</text>')
    for label, color, pts, *rest in series:
        if pts:
            width, end = (rest + [1.6, None])[0], (rest + [None, None])[1]
            dash = ' stroke-dasharray="4 3"' if end == "running" else ""
            parts.append(f'<polyline points="{" ".join(f"{xpos_s(x):.1f},{ypos(y):.1f}" for x, y in pts)}" fill="none" stroke="{color}" stroke-width="{width}" stroke-opacity="0.9"{dash}><title>{html.escape(label)}</title></polyline>')
            if end:
                x, y = pts[-1]
                fill = color if end == "stopped" else "var(--panel)"
                parts.append(f'<circle cx="{xpos_s(x):.1f}" cy="{ypos(y):.1f}" r="3.2" fill="{fill}" stroke="{color}" stroke-width="1.4"><title>{html.escape(label)}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def pretrain_section(run_dirs, tag, ce_ref, max_steps):
    shades = ["var(--small)", "var(--small2)", "var(--small3)"]
    p1, vl, rows = [], [], []
    for i, rd in enumerate(run_dirs):
        name = os.path.basename(os.path.normpath(rd))
        col = shades[i % len(shades)]
        d = {}
        for sp in glob.glob(os.path.join(rd, f"screen_{tag}*.json")):  # screen_watch writes one file per worker
            try:
                d.update(json.load(open(sp)))
            except (json.JSONDecodeError, OSError):
                pass
        pts = sorted((int(k.split("-")[1]), v["goal_reached"]["pass@1"]) for k, v in d.items() if k.startswith("ckpt-"))
        ep = os.path.join(rd, "eval_metrics.jsonl")
        vpts = [(r["step"], r["eval/val_loss"]) for r in map(json.loads, open(ep))] if os.path.exists(ep) else []
        ckpts = [int(x.split("-")[1]) for x in os.listdir(rd) if x.startswith("ckpt-")] if os.path.isdir(rd) else []
        done = "done" if os.path.exists(os.path.join(rd, "DONE")) else ("failed" if os.path.exists(os.path.join(rd, "FAILED")) else "training")
        p1.append((name, col, pts)); vl.append((name, col, vpts))
        best = max(pts, key=lambda t: t[1]) if pts else None
        rows.append(f"<tr><td><span class='key'><i style='background:{col}'></i>{html.escape(name)}</span></td><td>{done}</td>"
                    f"<td>{max(ckpts) if ckpts else 0}</td><td>{vpts[-1][1]:.3f}</td>" if vpts else
                    f"<tr><td>{html.escape(name)}</td><td>{done}</td><td>{max(ckpts) if ckpts else 0}</td><td>–</td>")
        rows[-1] += (f"<td>{pts[-1][0] if pts else '–'}</td><td>{pts[-1][1]:.4f}</td><td>{best[1]:.4f} @ {best[0]}</td></tr>" if pts else "<td>–</td><td>–</td><td>–</td></tr>")
    hl = [(v, f"medium base, {k}") for k, v in ce_ref]
    return f"""<section class="level"><div class="head"><h2>Small (2.2M) pretraining</h2>
<div class="legend">3 init seeds, same data order and recipe as medium / large</div></div>
<div class="charts"><figure><figcaption>goal-reached pass@1 by step · every 100th checkpoint · 1000 mazes × 16 samples</figcaption>{line_chart(p1, True, "pass@1", max_steps)}</figure>
<figure><figcaption>validation loss by step · dashed = medium's matched bases (same metric)</figcaption>{line_chart(vl, False, "validation loss", max_steps, hl)}</figure></div>
<div class="tablewrap"><table><thead><tr><th>run</th><th>state</th><th>latest ckpt</th><th>val loss</th><th>last screened</th><th>pass@1 there</th><th>best pass@1 so far</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div></section>"""


RL_LEVELS = [("p01", "0.1%"), ("p03", "0.3%"), ("p1", "1%"), ("p3", "3%"), ("p10", "10%")]
RL_BATCHES = [4, 16, 64, 256]


def rl_section(status_path, bases_path, name, lr, patience, s3_root, metric="goal-reached", reward="binary goal-reached reward",
               model="Small (2.2M)", loss="tis", wandb=None, adv="grpo"):
    """Small-model RL grid (rl_grid_status.py --json): best-value grid, validation curves per level, per-run table.
    metric names the validation metric (the early-stop metric); bases_path is a bases.json with "bases" (goal grid) or
    "levels" (pick_shortest_bases.py) mapping level -> {seed, pass1_confirm, target}. loss: tis (min(ratio, 2)) or icepop
    (ratio kept inside [0.5, 5], zeroed outside). Prompts-per-step values come from the rows; colors go by rank."""
    rows = json.load(open(status_path))
    batches = sorted({r["batch"] for r in rows}) or RL_BATCHES
    n_seeds = len({r["seed"] for r in rows})
    bj = json.load(open(bases_path)) if bases_path else {}
    bases = bj.get("bases") or bj.get("levels") or {}
    color = {b: f"var(--b{c})" for b, c in zip(batches, RL_BATCHES)}
    n_done = sum(r["state"] not in ("running", "queued") for r in rows)
    legend = "".join(f"<span class='key'><i style='background:{color[b]}'></i>{b} prompts</span>" for b in batches) + (
        "<span class='key'><svg width='22' height='10' style='width:22px;height:10px;display:inline'><line x1='0' x2='15' y1='5' y2='5' stroke='currentColor' stroke-width='1.4'/>"
        "<circle cx='17' cy='5' r='3' fill='currentColor'/></svg>stopped</span>"
        "<span class='key'><svg width='22' height='10' style='width:22px;height:10px;display:inline'><line x1='0' x2='13' y1='5' y2='5' stroke='currentColor' stroke-width='1.4' stroke-dasharray='4 3'/>"
        "<circle cx='17' cy='5' r='3' fill='none' stroke='currentColor' stroke-width='1.3'/></svg>still running</span>")

    def end_kind(r):
        return "running" if r["state"] in ("running", "queued") else "stopped"

    def end_label(r):
        last = r["last_step"] if r["last_step"] is not None else r["last_val_step"]
        return "still running" if end_kind(r) == "running" else f"stopped at step {last} ({r['state'].replace('_', ' ')})"

    grid = []
    for lvl, title in RL_LEVELS:
        start = [r["history"][0][1] for r in rows if r["level"] == lvl and r["history"] and r["history"][0][0] == 0]
        base = bases.get(lvl, {})
        cells = [f"<td>{title}</td><td>s{base.get('seed', '?')}</td><td>{base.get('pass1_confirm', float('nan')):.4f}</td>"
                 f"<td>{f'{sum(start) / len(start):.4f}' if start else '–'}</td>"]
        for b in batches:
            rs = [r for r in rows if r["level"] == lvl and r["batch"] == b and r["best_value"] is not None]
            if not rs:
                cells.append("<td class='ci'>–</td>"); continue
            vals = [r["best_value"] for r in rs]
            steps = [r["best_step"] for r in rs]
            n_all = sum(r["level"] == lvl and r["batch"] == b for r in rows)
            stopped = sum(r["state"] not in ("running", "queued") for r in rs)
            cells.append(f"<td><b>{sum(vals) / len(vals):.3f}</b><span class='ci'> {min(vals):.3f}–{max(vals):.3f}</span>"
                         f"<br><span class='ci'>best @ {min(steps)}–{max(steps)} · {stopped}/{n_all} stopped</span></td>")
        grid.append("<tr>" + "".join(cells) + "</tr>")

    charts = []
    for lvl, title in RL_LEVELS:
        rs = [r for r in rows if r["level"] == lvl and r["history"]]
        if not rs:
            continue
        xmax = max(r["history"][-1][0] for r in rs)
        xmax = max(500, int(math.ceil(xmax / 500) * 500))
        series = [(f"{r['suite']} (best {r['best_value']:.3f} @ {r['best_step']}; {end_label(r)})", color[r["batch"]],
                   [tuple(h) for h in r["history"]], 1.3, end_kind(r)) for r in sorted(rs, key=lambda r: r["batch"])]
        base = bases.get(lvl, {}).get("target")
        top = nice_top(1.05 * max([y for r in rs for _, y in r["history"]] + ([base] if base else [])))
        charts.append(f"<figure><figcaption>start {title} · validation {metric} (500 mazes × 16) by RL step</figcaption>"
                      f"{line_chart(series, False, metric.replace('-', ' '), xmax, [(base, 'base pass@1')] if base else [], 230, 'RL step', 500 if xmax <= 3000 else 1000, (0.0, top))}</figure>")

    ent_charts = []
    for lvl, title in RL_LEVELS:
        rs = [r for r in rows if r["level"] == lvl and r.get("entropy")]
        if not rs:
            continue
        xmax = max(500, int(math.ceil(max(r["entropy"][-1][0] for r in rs) / 500) * 500))
        ymax = nice_top(1.05 * max(v for r in rs for _, v in r["entropy"]))
        series = [(f"{r['suite']} (entropy {r['entropy'][-1][1]:.3f} at step {r['entropy'][-1][0]}; {end_label(r)})", color[r["batch"]],
                   [tuple(e) for e in r["entropy"]], 1.1, end_kind(r)) for r in sorted(rs, key=lambda r: r["batch"])]
        ent_charts.append(f"<figure><figcaption>start {title} · rollout entropy (nats / response token) by RL step</figcaption>"
                          f"{line_chart(series, False, 'entropy', xmax, [], 230, 'RL step', 500 if xmax <= 3000 else 1000, (0.0, ymax))}</figure>")
    ent_block = (f'<h3>Entropy</h3><p class="note">actor/entropy from W&amp;B: mean entropy of the sampling distribution '
                 f'(temperature 1) per response token, from the rollout logits, every training step.</p>'
                 f'<div class="rlcharts">{"".join(ent_charts)}</div>') if ent_charts else ""

    def ent_at(r, step):
        pts = r.get("entropy") or []
        return min(pts, key=lambda e: abs(e[0] - step))[1] if pts and step is not None else None

    def run_row(r):
        lvl_title = dict(RL_LEVELS)[r["level"]]
        last = r["last_step"] if r["last_step"] is not None else r["last_val_step"]
        state = {"early_stop": "early stop", "max_steps": "step cap"}.get(r["state"], r["state"])
        best = "–" if r["best_value"] is None else f"{r['best_value']:.4f}"
        ents = [ent_at(r, s) for s in (0, r["best_step"], last)]
        ent = "–" if ents[0] is None else " → ".join(f"{e:.3f}" for e in ents)
        name = (f"<a href='{html.escape(wandb)}/runs/{r['suite']}-{r['wandb_suffix']}'>{r['suite']}</a>" if wandb and r.get("wandb_suffix") else r["suite"])
        return (f"<tr><td>{name}</td><td>{lvl_title}</td><td>{r['batch']}</td><td>{r['seed']}</td><td>{state}</td>"
                f"<td>{'–' if last is None else last}</td><td>{'–' if r['best_step'] is None else r['best_step']}</td>"
                f"<td>{best}</td><td>{ent}</td></tr>")

    loss_name = {"tis": "TIS", "icepop": "IcePop"}[loss]
    adv_name = {"grpo": "GRPO", "reinforce_batch": "REINFORCE"}[adv]
    adv_text = {"grpo": "GRPO (group 16, std-normalized advantages)",
                "reinforce_batch": "REINFORCE: one sample per prompt (prompts per step = samples per step, the same samples per step as the GRPO grid), advantage = (r - batch mean) / batch std over the whole update"}[adv]
    loss_text = {"tis": "token-level TIS with min(train/inference ratio, 2) on the rollout log-probs",
                 "icepop": "IcePop train/inference masking: each token's policy gradient is weighted by the train/inference ratio "
                           "when it lies in [0.5, 5] and dropped (weight 0) outside it"}[loss]
    return f"""<section class="level"><div class="head"><h2>{model} RL grid {html.escape(name)}: {adv_name} + {loss_name}, LR {lr}, patience {patience}</h2><div class="legend">{legend}</div></div>
<p class="note">{n_done} of {len(rows)} runs finished. {adv_text} on the {reward}, one update per rollout, {loss_text}, constant LR {lr} after a 20-step warmup. Validation every 10 steps on 500 dev mazes × 16 samples;
a run stops after {patience} validations without a new high (or at the 5k-step cap), and its best checkpoint (best_hf) is kept with its step.
Cells: mean best validation {metric} over {n_seeds} RL seed{"s" if n_seeds > 1 else ""}, then min–max, the range of best steps, and how many of the
seeds have stopped (the rest are still training, so their cell can still rise).</p>
<div class="tablewrap"><table><thead><tr><th>start level</th><th>small seed</th><th>base pass@1 (1000 × 2048)</th><th>step-0 val (500 × 16)</th>
{"".join(f"<th>{b} prompts / step</th>" for b in batches)}</tr></thead><tbody>{"".join(grid)}</tbody></table></div>
<div class="rlcharts">{"".join(charts)}</div>
{ent_block}
{f'<p class="note">W&amp;B: <a href="{html.escape(wandb)}">{html.escape(wandb.split("wandb.ai/")[-1])}</a> (run names link to each run).</p>' if wandb else ''}
<details><summary>All {len(rows)} runs (best checkpoint: {html.escape(s3_root)}/runs/&lt;run&gt;/state/best_hf)</summary>
<div class="tablewrap"><table><thead><tr><th>run</th><th>start</th><th>prompts / step</th><th>RL seed</th><th>state</th><th>last step</th><th>best step</th><th>best val</th><th>entropy start → best → last</th></tr></thead>
<tbody>{"".join(run_row(r) for r in rows)}</tbody></table></div></details></section>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screens", nargs="+", required=True)
    ap.add_argument("--level", action="append", required=True, help="<key>=<title>:<ref label>,<label>,...")
    ap.add_argument("--ref-json", default=None, help="rl_curves_v1.json (1000-maze start profiles of p10 / p1 / p01)")
    ap.add_argument("--ref-manifest", default=None, help="extra/matched_manifest.json (p03 / p3 1000-maze metrics)")
    ap.add_argument("--tol-rel", type=float, default=0.02, help="|delta pass@k| <= tol-rel * min(p, 1 - p), every k <= --max-k")
    ap.add_argument("--max-k", type=int, default=1024)
    ap.add_argument("--n-noise", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--override-screens", nargs="*", default=[], help="screens used instead for a level once all its models are in them (e.g. the 1000-maze re-measure)")
    ap.add_argument("--small-select", default=None, help="dir with small_ds0_is<seed>/select_<level>_set.json (candidate screens)")
    ap.add_argument("--pretrain-runs", nargs="*", default=[], help="small-model pretraining run dirs (progress panel)")
    ap.add_argument("--pretrain-tag", default="stride50_k16")
    ap.add_argument("--pretrain-max-steps", type=int, default=20000)
    ap.add_argument("--ce-ref", default="", help="'<level>=<val loss>,...' of the medium bases, drawn on the loss chart")
    ap.add_argument("--rl-grid", action="append", default=[], help="one RL grid section: <name>|<rl_grid_status.py --json>|<lr>|<patience>|<s3 root>")
    ap.add_argument("--rl-bases", default=None, help="the RL grid's bases.json (small seed and pass@1 per level)")
    ap.add_argument("--updated", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    screens = {}
    for path in args.screens:
        screens.update(json.load(open(path)))
    overrides = {}
    for path in args.override_screens:
        if os.path.exists(path):
            overrides.update(json.load(open(path)))
    refs = reference_points(args)
    rng = np.random.default_rng(0)
    sections, summary_rows, close_rows = [], [], []
    table_ks = [1, 4, 16, 64, 256, 1024]

    for spec in args.level:
        key, rest = spec.split("=", 1)
        target = SET_VALUES.get(key)
        title, labels = rest.split(":", 1)
        labels = labels.split(",")
        src = overrides if all(lab in overrides for lab in labels) else screens
        missing = [lab for lab in labels if lab not in src]
        if missing:
            sections.append(f'<section class="level pending"><h2>{html.escape(title)}</h2>'
                            f'<p class="note">Screening in progress: {", ".join(missing)} not finished yet.</p></section>')
            summary_rows.append(f"<tr><td>{html.escape(title)}</td><td colspan='7' class='note'>pending</td></tr>")
            continue
        n = src[labels[0]]["k"]
        counts = {lab: np.asarray(src[lab]["per_maze"]["goal_reached"]) for lab in labels}
        n_mazes = len(counts[labels[0]])
        ks = [2**i for i in range(int(math.log2(min(n, args.max_k))) + 1)]
        boot_idx = rng.integers(0, n_mazes, size=(args.n_boot, n_mazes))
        models = {lab: curve_stats(counts[lab], n, ks, boot_idx) for lab in labels}
        ref_lab, oth = labels[0], labels[1]
        se = sampling_se(counts[ref_lab], counts[oth], n, ks, args.n_noise, rng)
        delta = {"d": [], "rel": [], "noise2": []}
        for k in ks:
            d = models[oth]["pass"][ks.index(k)] - models[ref_lab]["pass"][ks.index(k)]
            sc = scale_of(0.5 * (models[oth]["pass"][ks.index(k)] + models[ref_lab]["pass"][ks.index(k)]), n_mazes)
            delta["d"].append(d); delta["rel"].append(d / sc); delta["noise2"].append(2 * se[k] / sc)
        level = {"title": title, "ks": ks, "models": models, "delta": delta, "ref": refs.get(key, {}), "extra": {}}
        small_rows, small_pool = [], []
        if args.small_select and target:
            for lab, seed, center, n_s, c_s in small_candidates(args.small_select, key, target, args.max_k):
                ks_s = [k for k in ks if k <= n_s]
                pass_s = [float(pass_at_k(c_s, n_s, k).mean()) for k in ks_s]
                level["extra"][lab] = {"ks": ks_s, "pass": pass_s, "color": SMALL_COLORS[seed]}
                same_mazes = len(c_s) == n_mazes  # only compare with medium on the same maze set
                rel_med = [(ps - models[ref_lab]["pass"][ks.index(k)]) / scale_of(0.5 * (ps + models[ref_lab]["pass"][ks.index(k)]), n_mazes)
                           for k, ps in zip(ks_s, pass_s)] if same_mazes else []
                worst = max(range(len(rel_med)), key=lambda i: abs(rel_med[i])) if rel_med else None
                cell = lambda k: (f"{100 * rel_med[ks_s.index(k)]:+.1f}%" if rel_med and k in ks_s else "–")
                if rel_med and n_s == n:
                    oth_lab = labels[1]
                    rel_lrg = [(ps - models[oth_lab]["pass"][ks.index(k)]) / scale_of(0.5 * (ps + models[oth_lab]["pass"][ks.index(k)]), n_mazes)
                               for k, ps in zip(ks_s, pass_s)]
                    score = max(max(abs(a), abs(b)) for k, a, b in zip(ks_s, rel_med, rel_lrg) if k <= 64)  # k <= 64: well measured at every level
                    small_pool.append((score, lab, seed, ks_s, pass_s, rel_med, rel_lrg, c_s))
                small_rows.append(f"<tr><td><span class='key'><i style='background:{SMALL_COLORS[seed]}'></i>seed {seed}</span></td>"
                                  f"<td>{html.escape(str(center))}</td><td>{n_s}</td><td>{pass_s[0]:.5f} <span class='ci'>({100 * (pass_s[0] - target) / target:+.1f}%)</span></td>"
                                  f"<td>{cell(16)}</td><td>{cell(64)}</td><td>{cell(256)}</td><td>{cell(1024)}</td>"
                                  + (f"<td>{100 * rel_med[worst]:+.1f}% (k={ks_s[worst]})</td></tr>" if rel_med else
                                     f"<td class='note'>medium on {n_mazes} mazes; compared once both are on {len(c_s)}</td></tr>"))

        def worst_upto(rel_list, ks_list, kmax_):
            vals = [(abs(r), r, k) for r, k in zip(rel_list, ks_list) if k <= kmax_]
            return max(vals)[1:] if vals else (None, None)
        chosen = min(small_pool) if small_pool else None
        if chosen:
            _, lab_c, seed_c, ks_c, pass_c, relm_c, rell_c, _ = chosen
            level["extra"][lab_c]["chosen"] = True
            level["delta_small"] = {"ks": ks_c, "rel": relm_c, "color": SMALL_COLORS[seed_c]}
            fmt = lambda t: f"{100 * t[0]:+.1f}% <span class='ci'>(k={t[1]})</span>" if t[0] is not None else "–"
            close_rows.append(f"<tr><td>{html.escape(title)}</td><td><span class='key'><i style='background:{SMALL_COLORS[seed_c]}'></i>seed {seed_c}</span></td>"
                              f"<td>{pass_c[0]:.5f}</td><td>{models[ref_lab]['pass'][0]:.5f}</td><td>{models[labels[1]]['pass'][0]:.5f}</td>"
                              + "".join(f"<td>{fmt(worst_upto(rl, kl, km))}</td>" for km in (64, 256, 1024)
                                        for rl, kl in ((relm_c, ks_c), (rell_c, ks_c), (delta["rel"], ks)))
                              + "</tr>")
        p1_ref, p1_oth = models[ref_lab]["pass"][0], models[oth]["pass"][0]
        judged = [i for i, k in enumerate(ks) if k <= args.max_k]
        worst_i = max(judged, key=lambda i: abs(delta["rel"][i]))
        ok = all(abs(delta["rel"][i]) <= args.tol_rel for i in judged)
        unresolved = [ks[i] for i in judged if delta["noise2"][i] > args.tol_rel]
        rows = []
        for k in table_ks:
            if k > n:
                continue
            i = ks.index(k)
            r1000 = " / ".join(f"{refs.get(key, {}).get(size_of(lab), {}).get(k):.4f}" if refs.get(key, {}).get(size_of(lab), {}).get(k) is not None else "–" for lab in labels)
            rel = delta["rel"][i]
            verdict = "–" if k > args.max_k else ("within" if abs(rel) <= args.tol_rel else "outside")
            rows.append(f"<tr><td>{k}</td>" + "".join(f"<td>{models[lab]['pass'][i]:.4f}</td>" for lab in labels) +
                        f"<td>{delta['d'][i]:+.4f}</td><td class='{'good' if verdict == 'within' else ('bad' if verdict == 'outside' else '')}'>{100 * rel:+.1f}% {verdict}</td>"
                        f"<td class='ci'>±{100 * delta['noise2'][i]:.1f}%</td><td class='ref'>{r1000}</td></tr>")
        legend = "".join(f'<span class="key"><i style="background:{COLORS[size_of(lab)]}"></i>{html.escape(lab)}</span>' for lab in labels)
        sections.append(f"""<section class="level"><div class="head"><h2>{html.escape(title)}</h2><div class="legend">{legend}
{"".join(f'<span class="key"><i style="background:{e["color"]}"></i>{html.escape(lab)}{" (best match)" if e.get("chosen") else " (dashed)"}</span>' for lab, e in level["extra"].items())}
<span class="key"><i class="hollow"></i>1000-maze value at selection</span></div></div>
<div class="charts"><figure><figcaption>pass@k up to k = {ks[-1]}, log scale · {n_mazes} mazes × {n} samples · band = bootstrap 95% over mazes</figcaption>{svg_curves(level, n)}</figure>
<figure><figcaption>vs medium: orange = large − medium, purple = best small seed − medium; Δ / min(p, 1 − p), miss side ≥ 1 maze · shaded = ±{100 * args.tol_rel:g}% · dashed = 2σ sampling noise · red = outside</figcaption>{svg_delta(level, n, args.tol_rel)}</figure></div>
<div class="tablewrap"><table><thead><tr><th>k</th>{"".join(f"<th>{html.escape(lab)}</th>" for lab in labels)}<th>Δ</th><th>Δ / min(p, 1−p)</th><th>noise (2σ)</th><th>1000-maze (sel.)</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
{(f'''<div class="tablewrap"><table><thead><tr><th>small (2.2M): tuned model, else candidate closest to the set value</th><th>candidate</th><th>samples / maze</th><th>pass@1 (vs set value)</th>
<th>p@16 vs medium</th><th>p@64 vs medium</th><th>p@256 vs medium</th><th>p@1024 vs medium</th><th>largest</th></tr></thead><tbody>{"".join(small_rows)}</tbody></table>
<p class="note">Differences are Δ / min(p, 1 − p) against medium on the same mazes; candidate screens have 64 samples per maze (256 at 0.1%), so their curves stop at that k until the 2048-sample finalists land.</p></div>''') if small_rows else ""}</section>""")
        vs_set = lambda v: f"<span class='ci'> ({100 * (v - target) / target:+.1f}%)</span>" if target else ""
        summary_rows.append(f"<tr><td>{html.escape(title)}</td><td>{n_mazes}</td><td>{p1_ref:.5f}{vs_set(p1_ref)}</td><td>{p1_oth:.5f}{vs_set(p1_oth)}</td>"
                            f"<td>{100 * delta['rel'][0]:+.1f}%</td><td>{100 * delta['rel'][worst_i]:+.1f}% (k={ks[worst_i]})</td>"
                            f"<td class='{'good' if ok else 'bad'}'>{'yes' if ok else 'no'}</td>"
                            f"<td class='ci'>{('k ≥ ' + str(unresolved[0])) if unresolved else 'none'}</td></tr>")

    page = f"""<title>Maze Base Curves</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one column of level panels, each = curves chart + difference strip + numbers table */
:root {{ --bg:#f6f7f8; --panel:#ffffff; --fg:#1d2329; --muted:#5d6872; --line:#dde2e6; --small:#7a5bbf; --small2:#b05a9e; --small3:#4d6fd1; --medium:#1b7f8c; --large:#c2681d; --tol:#e9eef1; --bad:#c23b3b; --good:#2f7d4a; --b4:#9aa3ab; --b16:#d1a12a; --b64:#2f8f5b; --b256:#3a5fc8;
  --sans:"IBM Plex Sans", system-ui, sans-serif; --mono:"IBM Plex Mono", ui-monospace, monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#14181c; --panel:#1b2026; --fg:#e4e8eb; --muted:#9aa5ae; --line:#2e353c; --small:#a98ae6; --small2:#d98cc6; --small3:#8aa4ee; --medium:#4fb6c2; --large:#e8944f; --tol:#222a31; --bad:#ef6f6f; --good:#6cc28a; --b4:#7d8790; --b16:#e6bd55; --b64:#5cc28a; --b256:#7d9cf0; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#14181c; --panel:#1b2026; --fg:#e4e8eb; --muted:#9aa5ae; --line:#2e353c; --small:#a98ae6; --small2:#d98cc6; --small3:#8aa4ee; --medium:#4fb6c2; --large:#e8944f; --tol:#222a31; --bad:#ef6f6f; --good:#6cc28a; --b4:#7d8790; --b16:#e6bd55; --b64:#5cc28a; --b256:#7d9cf0; color-scheme:dark; }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--sans); font-size:15px; line-height:1.5; }}
main {{ max-width:1120px; margin:0 auto; padding-inline:16px; padding-block:28px 48px; display:grid; gap:22px; }}
h1 {{ font-size:1.55rem; font-weight:600; margin:0; text-wrap:balance; }}
h2 {{ font-size:1.1rem; font-weight:600; margin:0; }} h3 {{ font-size:1rem; font-weight:600; margin:6px 0 0; }}
.lede {{ color:var(--muted); max-width:72ch; margin:6px 0 0; }}
.level {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:16px; display:grid; gap:12px; min-width:0; }}
.head {{ display:flex; flex-wrap:wrap; gap:8px 18px; align-items:baseline; justify-content:space-between; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 14px; font-size:13px; color:var(--muted); }}
.key {{ display:inline-flex; gap:6px; align-items:center; }}
.key i {{ width:14px; height:3px; display:inline-block; border-radius:2px; }}
.key i.hollow {{ width:9px; height:9px; border-radius:50%; border:1.4px solid var(--muted); background:none; }}
.charts {{ display:grid; grid-template-columns:minmax(0,1.35fr) minmax(0,1fr); gap:14px; align-items:start; }}
@media (max-width:760px) {{ .charts {{ grid-template-columns:minmax(0,1fr); }} }}
figure {{ margin:0; min-width:0; }}
figcaption {{ font-size:12px; color:var(--muted); margin-bottom:4px; }}
svg {{ width:100%; height:auto; display:block; }}
svg .grid {{ stroke:var(--line); stroke-width:1; }} svg .faint {{ stroke-dasharray:2 3; }}
svg .zero {{ stroke:var(--muted); stroke-width:1; }} svg .tolband {{ fill:var(--tol); }}
svg .tick {{ fill:var(--muted); font:11px var(--mono); }} svg .axis {{ fill:var(--muted); font:12px var(--sans); }}
.tablewrap {{ overflow-x:auto; min-width:0; }}
table {{ border-collapse:collapse; font:13px var(--mono); font-variant-numeric:tabular-nums; width:100%; }}
th, td {{ text-align:right; padding:4px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }}
th {{ font:500 12px var(--sans); color:var(--muted); }} th:first-child, td:first-child {{ text-align:left; }}
td.ci, td.ref {{ color:var(--muted); }} td.good {{ color:var(--good); }} td.bad {{ color:var(--bad); }} svg .noiseline {{ stroke:var(--muted); stroke-dasharray:3 3; stroke-width:1; }}
.note {{ color:var(--muted); }} .stamp {{ color:var(--muted); font:12px var(--mono); margin:4px 0 0; }} svg .refline {{ stroke:var(--muted); stroke-dasharray:4 4; stroke-width:1; }} .pending {{ opacity:.85; }}
.rlcharts {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 330px), 1fr)); gap:14px; }}
details summary {{ cursor:pointer; color:var(--muted); font-size:13px; }} td b {{ font-weight:600; }}
.summary {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:14px 16px; min-width:0; }}
</style>
<main>
<header><h1>Text-maze study: matched small / medium / large bases and small-model RL</h1>
<p class="lede">{html.escape(args.subtitle)}</p><p class="stamp">{html.escape(args.updated)}</p></header>
{"".join(rl_section(st, args.rl_bases, name, lr, pat, root) for name, st, lr, pat, root in (g.split("|") for g in args.rl_grid) if os.path.exists(st))}
{(f'''<section class="summary tablewrap"><h2>Small vs medium vs large</h2>
<p class="note">Largest relative difference Δ / min(p, 1 − p) over k ≤ 64, k ≤ 256 and k ≤ 1024 (the k where it occurs in brackets), all on the same 1000 mazes × 2048 samples. Small is the tuned seed with the smallest worst-case gap to both medium and large at k ≤ 64 (well measured at every level). Near-saturated k (pass@k ≈ 1) are noisy: see each level's noise column.</p>
<table><thead><tr><th rowspan="2">level</th><th rowspan="2">small seed</th><th colspan="3">pass@1</th><th colspan="3">k ≤ 64</th><th colspan="3">k ≤ 256</th><th colspan="3">k ≤ 1024</th></tr>
<tr><th>small</th><th>medium</th><th>large</th><th>S−M</th><th>S−L</th><th>L−M</th><th>S−M</th><th>S−L</th><th>L−M</th><th>S−M</th><th>S−L</th><th>L−M</th></tr></thead>
<tbody>{"".join(close_rows)}</tbody></table></section>''') if close_rows else ""}
<div class="summary tablewrap"><table><thead><tr><th>level</th><th>mazes</th><th>medium pass@1 (vs set value)</th><th>large pass@1 (vs set value)</th><th>pass@1 rel. diff</th><th>largest rel. diff, k ≤ {args.max_k}</th><th>all k ≤ {args.max_k} within ±{100 * args.tol_rel:g}%</th><th>noise above ±{100 * args.tol_rel:g}% at</th></tr></thead>
<tbody>{"".join(summary_rows)}</tbody></table></div>
{"".join(sections)}
{pretrain_section(args.pretrain_runs, args.pretrain_tag, [tuple((kv.split("=")[0], float(kv.split("=")[1]))) for kv in args.ce_ref.split(",") if kv], args.pretrain_max_steps) if args.pretrain_runs else ""}
</main>"""
    with open(args.out, "w") as f:
        f.write(page)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
