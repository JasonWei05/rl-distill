# Aggregate band-eval grades: per (teacher set, model) mean@16, unbiased pass@k (>=1 of k correct) and pass^k
# (all k correct, tau-bench's pass-hat-k: C(c,k)/C(n,k)), plus the teacher's own mean@8.
import glob, json
from math import comb
import pandas as pd

def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n, c, k):
    return comb(c, k) / comb(n, k)

rows = []
for f in sorted(glob.glob("/tmp/qwen3_eval/band_evals/*/*/grades.json")):
    model, s = f.split("/")[-3], f.split("/")[-2]
    g = json.load(open(f)); n = len(g[0]["acc"])
    teacher = pd.read_parquet(f"/tmp/qwen3_eval/bands/{s}.parquet")
    r = {"set": s, "model": model.replace("Qwen3-", "").replace("-Base", ""), "teacher_mean@8": 100 * teacher.teacher_k.mean() / 8,
         "mean@16": 100 * sum(sum(x["acc"]) for x in g) / (len(g) * n),
         "boxed%": 100 * sum(x["boxed"] for x in g) / (len(g) * n), "trunc%": 100 * sum(x["trunc"] for x in g) / (len(g) * n)}
    for k in (1, 2, 4, 8, 16):
        r[f"pass@{k}"] = 100 * sum(pass_at_k(n, sum(x["acc"]), k) for x in g) / len(g)
    for k in (4, 16):
        r[f"pass^{k}"] = 100 * sum(pass_hat_k(n, sum(x["acc"]), k) for x in g) / len(g)
    rows.append(r)
df = pd.DataFrame(rows)
order = ["teacher8b_easy", "teacher8b_medium", "teacher8b_hard", "teacher30b_easy", "teacher30b_medium", "teacher30b_hard"]
df["o"] = df["set"].map(order.index); df = df.sort_values(["o", "model"]).drop(columns="o")
print(df.to_markdown(index=False, floatfmt=".1f"))
