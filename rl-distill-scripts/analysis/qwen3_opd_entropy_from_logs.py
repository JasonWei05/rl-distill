#!/usr/bin/env python3
"""Per-token entropy estimate of the Qwen OPD students over training: rollout_corr/training_log_ppl = mean NLL of the student's own
temperature-1 samples under the trainer (an unbiased entropy estimate), steps 1-10 (= the off-policy student) vs 111-120,
parsed from the supervisor pod logs; run from the repo root."""
import re, glob, sys
specs = "q4bm-to-q1p7b q4bh-to-q1p7b q4bvh-to-q1p7b q30bm-to-q1p7b q30bh-to-q1p7b q30bvh-to-q1p7b q30bm-to-q4b q30bh-to-q4b q30bvh-to-q4b".split()
keys = ["rollout_corr/training_log_ppl", "actor/distillation/k1", "response_length/mean"]
print("spec | NLL/token (entropy est.) steps 1-10 -> 111-120 | k1 | resp len")
for s in specs:
    rows = {}
    for f in glob.glob(f".scale_train_supervisors/q3-opd-{s}-pgk1-v1/pod-logs/*.log"):
        for line in open(f, errors="replace"):
            m = re.search(r" step:(\d+) - ", line)
            if not m: continue
            d = rows.setdefault(int(m.group(1)), {})
            for k in keys:
                mm = re.search(re.escape(k) + r":(?:np\.float64\()?([-0-9.e]+)", line)
                if mm: d.setdefault(k, float(mm.group(1)))
    def avg(a, b, k):
        v = [rows[i][k] for i in rows if a <= i <= b and k in rows[i]]; return sum(v) / len(v)
    print(s, " | ".join(f"{avg(1,10,k):.3f} -> {avg(111,120,k):.3f}" for k in keys))
