#!/usr/bin/env python3
"""Answer diversity of Qwen3 math-suite samples: per question, # distinct normalized last-\boxed{} answers among the n samples and
the Shannon entropy (bits) of that answer distribution, plus pass@16 - pass@1 from summary.json.

    python analysis/qwen3_answer_diversity.py <dir of <model>__<set>.parquet samples> <dir of <model>/<set>/summary.json>
"""
import glob, json, math, re, sys
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
S, MS = Path(sys.argv[1]), Path(sys.argv[2])

def last_boxed(t):
    i = max(t.rfind("\\boxed"), t.rfind("\\fbox"))
    if i < 0: return None
    j = t.find("{", i)
    if j < 0: return None
    d = 0
    for k in range(j, len(t)):
        d += t[k] == "{"; d -= t[k] == "}"
        if d == 0: return t[j + 1:k]
    return None

def norm(a):
    if a is None: return None
    a = re.sub(r"\\(left|right|displaystyle|!|,|;|quad)", "", a).replace("dfrac", "frac").replace("tfrac", "frac")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    return re.sub(r"\s+", "", a).rstrip(".").replace("^{\\circ}", "").replace("^\\circ", "")

res = {}
for f in sorted(S.glob("*.parquet")):
    model, s = f.stem.split("__")
    t = pq.read_table(f, columns=["response_texts"]).column("response_texts").to_pylist()
    distinct, ent, noans = [], [], 0
    for resp in t:
        ans = [norm(last_boxed(x)) for x in resp]
        noans += sum(a is None for a in ans)
        c = Counter(a if a is not None else "<none>" for a in ans); n = len(ans)
        distinct.append(len(c)); ent.append(-sum(v / n * math.log2(v / n) for v in c.values()))
    summ = json.load(open(MS / model / s / "summary.json"))
    res[(model, s)] = dict(distinct=sum(distinct) / len(distinct), ent=sum(ent) / len(ent), mean=summ["mean_at_n"],
                           p16=summ["pass_at_k"]["16"], p1=summ["pass_at_k"]["1"], p4=summ["pass_at_k"]["4"])
json.dump({f"{m}|{s}": v for (m, s), v in res.items()}, open(S.parent / "diversity.json", "w"), indent=1)
def show(label, m):
    cells = []
    for s in ("medium", "hard", "math500"):
        r = res[(m, s)]; cells.append(f"{r['distinct']:.2f} / {r['ent']:.2f} / {r['p16']-r['p1']:.1f}")
    print(f"| {label} | " + " | ".join(cells) + " |")
print("| model | medium: distinct ans / ans-entropy bits / pass@16-pass@1 | hard | math500 |"); print("|---|---|---|---|")
for size, teachers in (("q1p7b", ["q4bm","q4bh","q4bvh","q30bm","q30bh","q30bvh"]), ("q4b", ["q30bm","q30bh","q30bvh"])):
    for b in ("medium","hard","very_hard"): show(f"{size} RL {b}", f"{size}-rl-{b}-best")
    for t in teachers:
        show(f"{t}->{size} off-policy", f"{t}-to-{size}-s1000"); show(f"{t}->{size} OPD", f"{t}-to-{size}-opd120")
