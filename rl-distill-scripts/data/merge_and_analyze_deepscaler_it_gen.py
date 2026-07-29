#!/usr/bin/env python3
"""Merge the 4 DeepScaleR IT-gen shards and summarize response length vs accuracy.

Each shard row = one question with 4 samples (response_texts/response_lens/scores/accs + mean_len/
max_len/n_correct/any_correct). This merges them, buckets questions into response-length quintiles
(by per-question mean response length over the 4 samples), and reports accuracy per bucket — the
difficulty gradient we want for filtering/subsetting DeepScaleR.
"""
import glob
import numpy as np
import pandas as pd

OUT = "/mnt/efs/jasonwei/verl/data/deepscaler_it_gen"
files = sorted(glob.glob(f"{OUT}/shard_*.parquet"))
print("shards:", [f.split("/")[-1] for f in files])
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(f"merged questions: {len(df)}  (expect ~40315)")

df["mean_acc"] = df["accs"].apply(lambda a: float(np.mean(a)))     # avg correctness over 4 samples
df["pass4"] = df["any_correct"].astype(float)                       # solved at least once in 4

print("\n=== per-question mean response length (tokens) — distribution ===")
q = [0, .1, .25, .5, .75, .9, .95, 1.0]
print({f"p{int(p*100)}": int(df['mean_len'].quantile(p)) for p in q})
print(f"mean {df['mean_len'].mean():.0f}, capped-at-6144 fraction (max_len>=6100): "
      f"{100*(df['max_len']>=6100).mean():.1f}%")

print("\n=== overall accuracy (4B-IT, 2-shot, temp 1.0, 4 samples) ===")
print(f"mean@4 (avg per-sample acc): {100*df['mean_acc'].mean():.2f}%")
print(f"pass@4 (any of 4 correct):   {100*df['pass4'].mean():.2f}%")

print("\n=== accuracy by response-length quintile (bucket 1 = shortest responses) ===")
df["lenbucket"] = pd.qcut(df["mean_len"], 5, labels=[1, 2, 3, 4, 5])
g = df.groupby("lenbucket", observed=True).agg(
    n=("uid", "size"),
    len_lo=("mean_len", "min"), len_hi=("mean_len", "max"), len_med=("mean_len", "median"),
    mean_at4=("mean_acc", "mean"), pass_at4=("pass4", "mean"))
for b, r in g.iterrows():
    print(f"  bucket {b}: n={int(r['n'])}  len[{int(r['len_lo'])}-{int(r['len_hi'])}] med {int(r['len_med'])}"
          f"  | mean@4 {100*r['mean_at4']:.1f}%  pass@4 {100*r['pass_at4']:.1f}%")

merged = f"{OUT}/deepscaler_it_gen_merged.parquet"
df.to_parquet(merged, index=False)
print(f"\nwrote merged -> {merged}  ({len(df)} rows)")

# ---- the 5 accuracy subsets (by n_correct out of 4), saved RL-ready ----
print("\n=== 5 accuracy subsets (n_correct out of 4) — RL-ready parquets ===")
BOXED = " Please output the final answer within \\boxed{}."
for k in [0, 1, 2, 3, 4]:
    sub = df[df["n_correct"] == k]
    rows = [{"data_source": "math",
             "prompt": [{"content": (p if p.strip().endswith(BOXED.strip()) else p + BOXED), "role": "user"}],
             "reward_model": {"ground_truth": str(a), "style": "rule"},
             "extra_info": {"n_correct": k, "mean_len": float(ml)},
             "uid": u}
            for p, a, u, ml in zip(sub["problem"], sub["answer"], sub["uid"], sub["mean_len"])]
    path = f"{OUT}/deepscaler_acc{k}of4.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"  {k}/4 correct: {len(sub):>6} questions ({100*len(sub)/len(df):4.1f}%)  med_len {int(sub['mean_len'].median()) if len(sub) else 0}  -> deepscaler_acc{k}of4.parquet")
