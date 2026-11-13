#!/usr/bin/env python3
"""Validation is_shortest vs cumulative *essential* RL FLOPs for maze RL grids (small vs medium), per start level.

Essential-FLOPs convention (same as analysis/qwen3_flops_matched_opd_vs_rl.py):
  forward over prompt p + response r = 2*Nb*p + 2*N*r + 2*A*(p+r)^2   (Nb = block params, N = Nb + LM head, A = layers*heads*head_dim)
  RL step = generation for every sample (prompt prefilled once per group: 2*Nb*p + 2*A*p^2 per group; decode 2*N*r + 2*A*((p+r)^2-p^2)
            per sample) + training = 3 forwards, only on samples of mixed-correctness groups (all-correct / all-wrong groups have
            zero GRPO advantage). Excluded: old log-prob recompute, validation, padding, recompute, LM head on prompt positions.
Per-step inputs are W&B scalars (prompt_length/mean, response_length/mean, advantage_per_group/num_zero_groups; mean lengths
stand in for the per-sample distribution); validation points come from each run's S3 state/best.json history.

    python3 rl_flops_curves.py --grid "small|XS|s3://.../rl_small_dfs_eps05_ice_v1|spi|epi|4 16 64 256|0 1 2" \\
        --grid "medium|S|s3://.../rl_medium_dfs_eps05_ice_v1|smi|mdi|2 8 32 128|0" --wandb-path rl-distill/tailrl-text-maze-dfs-eps05 \\
        --out /tmp/jw_maze/rl_flops_curves.json
"""
import argparse
import json
import os
import time

GROUP = 16
VOCAB = 32
ARCH = {"3XS": (96, 4, 4, 2, 384), "2XS": (128, 4, 4, 2, 512), "XS": (192, 4, 4, 2, 768), "S": (256, 4, 4, 2, 1024), "L": (384, 6, 6, 3, 1536)}  # hidden layers heads kv ffn
KEYS = ["prompt_length/mean", "response_length/mean", "advantage_per_group/num_zero_groups"]


def consts(size):
    h, L, nq, nkv, ffn = ARCH[size]
    dh = h // nq
    nb = L * (2 * h * nq * dh + 2 * h * nkv * dh + 3 * h * ffn)
    return dict(Nb=nb, N=nb + h * VOCAB, A=L * nq * dh)


def step_flops(m, groups, p, r, zero_groups, group=GROUP):
    fwd = 2 * m["Nb"] * p + 2 * m["N"] * r + 2 * m["A"] * (p + r) ** 2
    gen = groups * (2 * m["Nb"] * p + 2 * m["A"] * p ** 2) + groups * group * (2 * m["N"] * r + 2 * m["A"] * ((p + r) ** 2 - p ** 2))
    return gen + max(groups - zero_groups, 0) * group * 3 * fwd


def wandb_rows(api, path, run_id, cache_dir, done):
    f = os.path.join(cache_dir, f"{run_id}.json")
    if os.path.exists(f):
        c = json.load(open(f))
        if c.get("done") or time.time() - c["t"] < 1200:
            return {int(k): v for k, v in c["rows"].items()}
    rows = {}
    run = api.run(f"{path}/{run_id}")
    for key in KEYS:  # one sampled-history query per key, merged by _step (scan_history drops steps after a resume)
        for h in run.history(keys=[key], samples=20000, pandas=False):
            rows.setdefault(int(h["_step"]), {})[key] = h[key]
    json.dump({"t": time.time(), "done": done, "rows": rows}, open(f, "w"))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", action="append", required=True,
                    help="<label>|<arch XS/S/L>|<s3 root>|<suite prefix>|<wandb run-id suffix[;prompts:suffix...]>|<prompts per step>|<seeds>[|<samples per prompt, 16>]"
                         " (1 = REINFORCE: every prompt prefilled, every sample trained unless the whole batch has one reward)")
    ap.add_argument("--wandb-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/tmp/jw_maze/flops_cache")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    import boto3
    import wandb
    api = wandb.Api(timeout=120)
    s3 = boto3.Session(profile_name="ml-worker").client("s3")
    out = {}
    for g in args.grid:
        label, arch, root, sp, sfx, batches, seeds, group = (g.split("|") + ["16"])[:8]; group = int(group)
        m = consts(arch); bucket, prefix = root[len("s3://"):].split("/", 1)
        sfx, *over = sfx.split(";")   # "<suffix>[;<prompts>:<suffix>...]": a different W&B id suffix for some batch sizes
        sfx_of = {int(b): x for b, x in (o.split(":") for o in over)}
        runs = []
        for lvl in ["p01", "p03", "p1", "p3", "p10"]:
            for b in map(int, batches.split()):
                for s in map(int, seeds.split()):
                    suite = f"{sp}-{lvl}-b{b}-s{s}"
                    try:
                        bj = json.loads(s3.get_object(Bucket=bucket, Key=f"{prefix}/runs/{suite}/state/best.json")["Body"].read())
                    except s3.exceptions.NoSuchKey:
                        continue
                    try:
                        s3.head_object(Bucket=bucket, Key=f"{prefix}/runs/{suite}/COMPLETE.json"); done = True
                    except Exception:
                        done = False
                    try:
                        rows = wandb_rows(api, args.wandb_path, f"{suite}-{sfx_of.get(b, sfx)}", args.cache, done)
                    except Exception as e:  # run not on W&B yet
                        print(f"skip {suite}: {e}"); continue
                    steps = sorted(k for k, v in rows.items() if k >= 1 and all(key in v for key in KEYS))
                    if not steps:
                        continue
                    cum, total, last = {}, 0.0, None
                    for t in range(1, max(steps) + 1):   # a missing step costs what the nearest earlier logged step did
                        if t in rows and all(key in rows[t] for key in KEYS):
                            last = rows[t]
                        if last is None:
                            last = rows[steps[0]]
                        # REINFORCE (group 1): a "zero group" is a zero-advantage sample, i.e. every reward in the batch equal
                        total += step_flops(m, b, last[KEYS[0]], last[KEYS[1]], last[KEYS[2]], group)
                        cum[t] = total
                    pts = [(cum[t], v) for t, v in bj["history"] if t in cum]
                    runs.append(dict(suite=suite, level=lvl, batch=b, seed=s, done=done, points=pts,
                                     steps=[t for t, _ in bj["history"] if t in cum],
                                     flops_per_step=total / max(steps)))
        out[label] = dict(arch=arch, params=consts(arch)["N"], runs=runs)
        print(f"{label}: {len(runs)} runs")
    json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()


def smooth(pts, k=5):
    """Centered running mean over k validations (curves are noisy at 500 x 16)."""
    h = k // 2
    return [(x, sum(v for _, v in pts[max(0, i - h):i + h + 1]) / len(pts[max(0, i - h):i + h + 1])) for i, (x, _) in enumerate(pts)]


def seed_mean(curves):
    """Mean of several (x, y) curves on a shared log-x grid, over the x range every curve covers."""
    import math
    lo = max(c[0][0] for c in curves); hi = min(c[-1][0] for c in curves)
    if hi <= lo:
        return []
    grid = [min(max(10 ** (math.log10(lo) + (math.log10(hi) - math.log10(lo)) * i / 80), lo), hi) for i in range(81)]
    def at(c, x):
        if x <= c[0][0]:
            return c[0][1]
        for (x0, y0), (x1, y1) in zip(c, c[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1
        return c[-1][1]
    return [(x, sum(at(c, x) for c in curves) / len(curves)) for x in grid]


def render_section(path, title, note):
    """HTML: one chart per start level, val is_shortest vs cumulative essential FLOPs (log x), every (size, prompts/step)."""
    import html
    import math
    from report_matched_curves import line_chart, nice_top
    d = json.load(open(path))
    sizes = list(d)
    colors = {sizes[0]: "var(--small)", sizes[1]: "var(--medium)"} if len(sizes) > 1 else {sizes[0]: "var(--small)"}
    levels = [("p01", "0.1%"), ("p03", "0.3%"), ("p1", "1%"), ("p3", "3%"), ("p10", "10%")]
    shade = [38, 58, 78, 100]   # % of the size colour, smallest to largest prompts/step
    figs, legend = [], []
    for sz in sizes:
        bs = sorted({r["batch"] for r in d[sz]["runs"]})
        for i, b in enumerate(bs):
            c = f"color-mix(in srgb, {colors[sz]} {shade[i]}%, var(--panel))"
            legend.append(f"<span class='key'><i style='background:{c}'></i>{html.escape(sz)} {b}</span>")
    for lvl, ttl in levels:
        series, xs, ys = [], [], []
        for sz in sizes:
            runs = [r for r in d[sz]["runs"] if r["level"] == lvl and len(r["points"]) >= 3]
            for i, b in enumerate(sorted({r["batch"] for r in d[sz]["runs"]})):
                rs = [r for r in runs if r["batch"] == b]
                if not rs:
                    continue
                curve = seed_mean([smooth(r["points"]) for r in rs]) if len(rs) > 1 else smooth(rs[0]["points"])
                if not curve:
                    continue
                c = f"color-mix(in srgb, {colors[sz]} {shade[i]}%, var(--panel))"
                end = "stopped" if all(r["done"] for r in rs) else "running"
                series.append((f"{sz}, {b} prompts/step ({len(rs)} seed{'s' if len(rs) > 1 else ''}; {'all stopped' if end == 'stopped' else 'still running'})",
                               c, curve, 1.8 if i == 3 else 1.4, end))
                xs += [x for x, _ in curve]; ys += [y for _, y in curve]
        if not series:
            continue
        xmin = 10 ** math.floor(math.log10(min(xs))); xmax = 10 ** math.ceil(math.log10(max(xs)))
        figs.append(f"<figure><figcaption>start {ttl} · validation is-shortest (500 mazes × 16, 5-point running mean) by cumulative essential RL FLOPs</figcaption>"
                    f"{line_chart(series, False, 'is shortest', xmax, [], 250, 'essential RL FLOPs (log)', 0, (0.0, nice_top(1.05 * max(ys))), xmin=xmin)}</figure>")
    return (f'<section class="level"><div class="head"><h2>{html.escape(title)}</h2><div class="legend">{"".join(legend)}</div></div>'
            f'<p class="note">{note}</p><div class="rlcharts">{"".join(figs)}</div></section>')


def best_points(path):
    """Per run: (essential FLOPs spent up to the best validation, best is-shortest, run) from an rl_flops_curves.py --out json."""
    d = json.load(open(path))
    out = {}
    for sz, g in d.items():
        out[sz] = [(*max(r["points"], key=lambda p: p[1]), r) for r in g["runs"] if r["points"]]
    return out


def render_best_section(specs, title, note):
    """HTML: one scatter per start level, every run's best validation against the essential FLOPs spent to reach it (log x).
    specs: [(rl_flops_curves json, method label)]; colour = size, marker = method, shade = samples per step, filled = stopped.
    A line joins the seed means of each (method, size) across samples per step."""
    import html
    import math
    from report_matched_curves import W, PAD_L, PAD_R, PAD_T, PAD_B, nice_top
    levels = [("p01", "0.1%"), ("p03", "0.3%"), ("p1", "1%"), ("p3", "3%"), ("p10", "10%")]
    shade = [38, 58, 78, 100]
    groups = []   # (method, size, samples-per-step list, [(flops, best, run)])
    for i, (path, method) in enumerate(specs):
        if not os.path.exists(path):
            continue
        group = 1 if method.lower().startswith("reinforce") else GROUP
        for sz, pts in best_points(path).items():
            groups.append((i, method, sz, group, pts))
    if not groups:
        return ""
    sizes = list(dict.fromkeys(g[2] for g in groups))
    colors = {s: c for s, c in zip(sizes, ["var(--small)", "var(--medium)", "var(--large)"])}
    def marker(mi, x, y, fill, stroke, tip):
        t = f"<title>{html.escape(tip)}</title>"
        if mi == 0:
            return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{fill}" stroke="{stroke}" stroke-width="1.4">{t}</circle>'
        return (f'<path d="M{x:.1f},{y - 4.6:.1f} L{x + 4.2:.1f},{y + 3:.1f} L{x - 4.2:.1f},{y + 3:.1f} Z" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="1.4" stroke-linejoin="round">{t}</path>')
    legend = []
    for sz in sizes:
        legend.append(f"<span class='key'><i style='background:{colors[sz]}'></i>{html.escape(sz)}</span>")
    for mi, (_, method) in enumerate(specs):
        sym = "●" if mi == 0 else "▲"
        legend.append(f"<span class='key'>{sym} {html.escape(method)} ({'solid' if mi == 0 else 'dashed'} line = seed mean)</span>")
    legend.append("<span class='key'>lighter → darker = more samples per step</span>")
    height = 270
    figs = []
    for lvl, ttl in levels:
        pts_all = [(f, b) for _, _, _, _, pts in groups for f, b, r in pts if r["level"] == lvl]
        if not pts_all:
            continue
        xmin = 10 ** math.floor(math.log10(min(f for f, _ in pts_all))); xmax = 10 ** math.ceil(math.log10(max(f for f, _ in pts_all)))
        top = nice_top(1.08 * max(b for _, b in pts_all))
        X = lambda f: PAD_L + (W - PAD_L - PAD_R) * (math.log10(f) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin))
        Y = lambda v: PAD_T + (height - PAD_T - PAD_B) * (1 - v / top)
        parts = [f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="best is-shortest by essential RL FLOPs, start {ttl}">']
        raw = top / 5; mag = 10 ** math.floor(math.log10(raw))
        step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw * (1 - 1e-9))
        dec = max(0, -math.floor(math.log10(step) + 1e-9) + (1 if round(step / 10 ** math.floor(math.log10(step) + 1e-9), 3) == 2.5 else 0))
        for k in range(int(top / step + 1e-9) + 1):
            y = Y(k * step)
            parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                         f'<text x="{PAD_L - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{k * step:.{dec}f}</text>')
        sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        for e in range(round(math.log10(xmin)), round(math.log10(xmax)) + 1):
            x = X(10.0 ** e)
            parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{PAD_T}" y2="{height - PAD_B}" class="grid"/>'
                         f'<text x="{x:.1f}" y="{height - PAD_B + 16}" class="tick" text-anchor="middle">10{str(e).translate(sup)}</text>')
        parts.append(f'<text x="{(PAD_L + W - PAD_R) / 2:.0f}" y="{height - 4}" class="axis" text-anchor="middle">essential RL FLOPs to the best checkpoint (log)</text>')
        lines, dots = [], []
        for mi, method, sz, group, pts in groups:
            runs = [(f, b, r) for f, b, r in pts if r["level"] == lvl]
            samples = sorted({r["batch"] * group for _, _, r in pts})
            means = []
            for si, s in enumerate(samples):
                rs = [(f, b, r) for f, b, r in runs if r["batch"] * group == s]
                if not rs:
                    continue
                c = f"color-mix(in srgb, {colors[sz]} {shade[min(si, 3)]}%, var(--panel))"
                for f, b, r in rs:
                    tip = (f"{method}, {sz}, {s} samples/step ({r['batch']} prompts), seed {r['seed']}: best {b:.3f} at {f:.2e} FLOPs"
                           + ("" if r["done"] else " (still running)"))
                    dots.append(marker(mi, X(f), Y(b), c if r["done"] else "var(--panel)", c, tip))
                means.append((math.exp(sum(math.log(f) for f, _, _ in rs) / len(rs)), sum(b for _, b, _ in rs) / len(rs)))
            if len(means) > 1:
                dash = "" if mi == 0 else ' stroke-dasharray="5 3"'
                lines.append(f'<polyline points="{" ".join(f"{X(f):.1f},{Y(b):.1f}" for f, b in means)}" fill="none" stroke="{colors[sz]}" '
                             f'stroke-width="1.3" stroke-opacity="0.75"{dash}><title>{html.escape(f"{method}, {sz}: seed means")}</title></polyline>')
        parts += lines + dots + ["</svg>"]
        figs.append(f"<figure><figcaption>start {ttl} · best validation is-shortest (500 mazes × 16) of every run against the essential FLOPs "
                    f"spent to reach it</figcaption>{''.join(parts)}</figure>")
    return (f'<section class="level"><div class="head"><h2>{html.escape(title)}</h2><div class="legend">{"".join(legend)}</div></div>'
            f'<p class="note">{note}</p><div class="rlcharts">{"".join(figs)}</div></section>')
