# Usage: python grade_band_eval_samples.py <band_evals>/<model>/<set>/samples.parquet  (writes grades.json beside it)
# Strict-grade one band-eval samples.parquet -> grades.json (list of per-question lists of bools).
import json, os, sys
sys.path.insert(0, "/mnt/efs/jasonwei/rl-distill")
import pandas as pd

def main():
    from verl.utils.reward_score import math_verify
    # warm the verifier's worker pool first: its cold start (spawn + imports) can exceed the 30 s per-call timeout on a loaded
    # host, which would silently grade the first response 0 (seen locally: 28 s at load ~70)
    math_verify.compute_score("\\boxed{1}", "1", timeout=600)
    path = sys.argv[1]
    df = pd.read_parquet(path, columns=["uid", "answer", "response_texts", "response_lens"])
    out = [{"uid": r.uid, "acc": [float(math_verify.compute_score(t, r.answer)) > 0.5 for t in r.response_texts],
            "boxed": sum("\\boxed" in t for t in r.response_texts), "trunc": sum(int(l) >= 8192 for l in r.response_lens),
            "mean_len": float(sum(r.response_lens) / len(r.response_lens))} for r in df.itertuples()]
    json.dump(out, open(path.replace("samples.parquet", "grades.json"), "w"))
    sys.stdout.flush(); os._exit(0)

if __name__ == "__main__":
    main()
