#!/usr/bin/env python3
"""Distilled student vs its teacher (and its own starting base): pass@k for every k <= 32, mean@n, own-sample entropy and the
forward KL to the teacher, per validation band.

  A. 4B base -> 30B base (off-policy top-128 forward KL, per band): teacher Qwen3-4B-Base, students at steps 250..1000,
     reference Qwen3-30B-A3B-Base. pass@k from s3://.../qwen3-passk32-evals-v1/<model>/<band>/grades.json, entropy from
     s3://.../qwen3-own-nll-v1/<model>/<band>/nll.json, forward KL = W&B val/loss (top-128 forward KL on teacher val generations).
  B. base -> base forward-KL pre-distillation over the band union (q1p7b-base-all -> 4B', q4b-base-all -> 1.7B'): everything from
     s3://.../qwen3-4b-base-distill-evals-v1/<spec>-to-<student>-bs128-s1000-lr2e-6/step_<s>/<band>/{grades,summary}.json;
     the teacher base = step 0 of the opposite direction (same protocol). Sections with missing data are skipped.
pass@k = unbiased estimator 1 - C(n-c, k)/C(n, k) averaged over questions; entropy = own-sample NLL per token (token-weighted).
  python rl-distill-scripts/analysis/qwen3_distill_vs_teacher_report.py
"""
import io, json, os
from math import comb

import boto3
import numpy as np
import wandb

OUT = "rl-distill-scripts/figures/qwen3_distill_vs_teacher"
BUCKET = "scale-ml"
S3 = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", "ml-worker")).client("s3")
KS = list(range(1, 33))
SHOW_K = (1, 2, 4, 8, 16, 32)


def s3_json(key):
    try:
        return json.load(io.BytesIO(S3.get_object(Bucket=BUCKET, Key=key)["Body"].read()))
    except S3.exceptions.NoSuchKey:
        return None


def passk_from_grades(key):
    rows = s3_json(key)
    if rows is None:
        return None
    n = len(rows[0]["acc"]); c = [sum(r["acc"]) for r in rows]
    return {"n": n, "mean": 100 * sum(c) / (len(c) * n),
            "pass": {k: 100 * float(np.mean([1 - comb(n - ci, k) / comb(n, k) for ci in c])) for k in KS if k <= n}}


def wandb_kl(project, name):
    """val/loss by step, stitched across the run's W&B segments (later segments win)."""
    out = {}
    runs = sorted(wandb.Api(timeout=120).runs(f"rl-distill/{project}", filters={"display_name": name}), key=lambda r: r.created_at)
    for r in runs:
        h = r.history(keys=["val/loss"], samples=5000, pandas=True)
        if h.empty:   # crashed segments that never logged a validation
            continue
        out.update({int(s): float(v) for s, v in zip(h["_step"], h["val/loss"])})
    return out


def kl_at(kl, step):
    if not kl:
        return None
    s = min(kl, key=lambda x: abs(x - step))
    return kl[s] if abs(s - step) <= 10 else None


report = {"A": {}, "B": {}}
# ---- A. 4B base -> 30B base ----
P32, NLL = "genai/rl-distill/qwen3-passk32-evals-v1", "genai/rl-distill/qwen3-own-nll-v1"
for band in ("medium", "hard"):
    kl = wandb_kl("qwen3-4b-base-distill-v1", f"q4b-base-{band}-to-q30b-base-bs128-s1000-lr2e-6-g8")
    rows = []
    for label, name, step in [("teacher: Qwen3-4B-Base", "q4b-base", None), ("start: Qwen3-30B-A3B-Base", "q30b-base", 0)] + \
            [(f"30B student @{s}", f"q4b30b-{band}-s{s}", s) for s in (250, 500, 750, 1000)]:
        pk = passk_from_grades(f"{P32}/{name}/{band}/grades.json")
        nl = s3_json(f"{NLL}/{name}/{band}/nll.json")
        rows.append({"label": label, "step": step, "passk": pk, "nll": nl["nll_per_token"] if nl else None,
                     "kl": kl_at(kl, step) if step is not None else 0.0})
    report["A"][band] = rows

# ---- B. base -> base pre-distillation over the band union ----
EV = "genai/rl-distill/qwen3-4b-base-distill-evals-v1"
DIRS = {"4B' (1.7B base -> 4B base)": ("q1p7b-base-all-to-q4b", "q4b-base-all-to-q1p7b", "q1p7b-base-all-to-q4b-base"),
        "1.7B' (4B base -> 1.7B base)": ("q4b-base-all-to-q1p7b", "q1p7b-base-all-to-q4b", "q4b-base-all-to-q1p7b-base")}
for direction, (run, other, wname) in DIRS.items():
    kl = {}
    try:
        kl = wandb_kl("qwen3-4b-base-distill-v1", f"{wname}-bs128-s1000-lr2e-6-g4")
    except Exception:
        pass
    for band in ("easy", "medium", "hard"):
        def ev(r, s):
            pre = f"{EV}/{r}-bs128-s1000-lr2e-6/step_{s}/{band}"
            summ = s3_json(f"{pre}/summary.json")
            return passk_from_grades(f"{pre}/grades.json"), (summ or {}).get("own_nll", {}).get("nll_per_token")
        rows = []
        tp, tn = ev(other, 0)
        rows.append({"label": "teacher base", "step": None, "passk": tp, "nll": tn, "kl": 0.0})
        for s in (0, 250, 500, 750, 1000):
            p, n = ev(run, s)
            rows.append({"label": "start: student base" if s == 0 else f"student @{s}", "step": s, "passk": p, "nll": n, "kl": kl_at(kl, s)})
        if any(r["passk"] for r in rows[2:]):
            report["B"].setdefault(direction, {})[band] = rows

# ---- markdown ----
fmt = lambda v, f="{:.1f}": "-" if v is None else f.format(v)
lines = ["# Distilled student vs teacher", "",
         "pass@k (unbiased, n=32 samples per question on the band's 300-q val set, strict boxed grading), mean@32, entropy = own-sample",
         "NLL per token, forward KL = top-128 KL(teacher || student) per token on the teacher's held-out validation generations (W&B",
         "val/loss; at step 0 it is the KL between the two untouched bases). Ideal: student rows match the teacher row.", ""]
hdr = "| model | mean@32 | " + " | ".join(f"pass@{k}" for k in SHOW_K) + " | entropy (NLL/tok) | fwd KL |"
sep = "|" + "---|" * (len(SHOW_K) + 4)
for sec, title in (("A", "A. 4B base -> 30B base (per band)"), ("B", "B. base -> base over the band union")):
    if not report[sec]:
        continue
    lines += [f"## {title}", ""]
    groups = report[sec].items() if sec == "B" else [("", report["A"])]
    for gname, bands in groups:
        for band, rows in bands.items():
            lines += [f"### {gname + ', ' if gname else ''}{band}", "", hdr, sep]
            for r in rows:
                pk = r["passk"]
                cells = [fmt(pk["mean"]) if pk else "-"] + [fmt(pk["pass"].get(k)) if pk else "-" for k in SHOW_K]
                lab = f"**{r['label']}**" if r["label"].startswith("teacher") else r["label"]
                lines.append(f"| {lab} | " + " | ".join(cells) + f" | {fmt(r['nll'], '{:.3f}')} | {fmt(r['kl'], '{:.4f}')} |")
            lines.append("")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT + ".md", "w").write("\n".join(lines) + "\n")
json.dump(report, open(OUT + ".json", "w"), indent=1, default=str)

# ---- pass@k curves ----
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

panels = [("A", "", b, rows) for b, rows in report["A"].items()] + \
         [("B", d, b, rows) for d, bands in report["B"].items() for b, rows in bands.items()]
if panels:
    cols = 3; nrows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(5 * cols, 4 * nrows), squeeze=False)
    for ax, (sec, d, band, rows) in zip(axes.flat, panels):
        cmap = plt.get_cmap("viridis")
        studs = [r for r in rows if r["passk"] and r["label"].startswith(("30B student", "student @"))]
        for r in rows:
            if not r["passk"]:
                continue
            ks = sorted(r["passk"]["pass"]); ys = [r["passk"]["pass"][k] for k in ks]
            if r["label"].startswith("teacher"):
                ax.plot(ks, ys, color="black", lw=2.5, label=r["label"])
            elif r["label"].startswith("start"):
                ax.plot(ks, ys, color="gray", ls="--", label=r["label"])
            else:
                ax.plot(ks, ys, color=cmap(studs.index(r) / max(1, len(studs) - 1)), lw=1.3, label=r["label"])
        ax.set_xscale("log", base=2); ax.set_xticks(SHOW_K); ax.set_xticklabels(SHOW_K)
        ax.set_title(f"{'4B base -> 30B base' if sec == 'A' else d}: {band}", fontsize=9)
        ax.set_xlabel("k"); ax.set_ylabel("pass@k (%)"); ax.grid(alpha=0.3); ax.legend(fontsize=6)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT + ".png", dpi=130)
print("wrote", OUT + ".md")
