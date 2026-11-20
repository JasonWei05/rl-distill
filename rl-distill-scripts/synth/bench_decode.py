"""Decoder throughput on a task's RL prompts: seconds per generate() of --prompts x --n rollouts from --base, with the
response caches grown model.GROW slots at a time (--grow 1, the default layout) or allocated at full width up front
(--grow 0, the layout before 2026-11-19); the sampled lengths, the peak memory, and the sampled tokens' log-probs
against train_logprobs on the first 8 prompts.

    python bench_decode.py --base chess/bases/model_20m_16B@384 --data data/chess --prompts 256 --n 16 --mem-gb 60
"""

import argparse
import os
import statistics
import time

import torch

import model
from data import Dataset
from model import Decoder, Qwen2, half, load_model
from tasks import ENV

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--base", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--prompts", type=int, default=256)
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--mem-gb", type=float, default=60.0)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--grow", type=int, nargs="+", default=[0, 1])
a = ap.parse_args()
d = Dataset(a.data)
hf, _, weights = load_model(a.base)
w, rl, G = half(weights), d.prompts(d.cfg["rl"]["split"]), a.prompts
grow = model.GROW
for g in a.grow:
    model.GROW = grow if g else 10 ** 9  # full width: one allocation of every response slot, chunks fitted to it
    dec = Decoder(Qwen2(hf), **{**d.cfg["decode"], "mem_gb": a.mem_gb})
    secs, lens = [], []
    torch.cuda.reset_peak_memory_stats()
    for r in range(a.reps + 1):  # rep 0 compiles and captures
        prompts = rl[r * G:(r + 1) * G].cuda().long()
        script = d.task.parse(prompts)["script"]
        torch.manual_seed(r)
        torch.cuda.synchronize()
        t = time.perf_counter()
        tok, logp, _ = dec.generate(w, prompts, a.n, script)
        torch.cuda.synchronize()
        mask = (tok != hf["pad_token_id"]) & (tok < ENV)
        if r:
            secs.append(time.perf_counter() - t)
            lens.append(mask.sum(1).float())
    k = 8 * a.n  # the last rep's first 8 prompts against the teacher-forced forward
    with torch.no_grad():
        ref = Qwen2(hf).train_logprobs(w, prompts[:8], tok[:k] % ENV, a.n, script["plen"][:8])
    diff = (logp[:k] - ref).abs() * mask[:k]
    lens = torch.cat(lens)
    print(f"BENCH_DECODE {os.path.basename(a.base.rstrip('/'))} {'grow' if g else 'full'} {G}x{a.n}: "
          f"{statistics.median(secs):.2f} s (reps {', '.join(f'{s:.2f}' for s in secs)}), sampled tokens mean "
          f"{lens.mean():.0f} max {lens.max():.0f}, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB, "
          f"|logp - ref| mean {diff.sum() / mask[:k].sum():.1e} max {diff.max():.2f}", flush=True)
    del dec
    torch.cuda.empty_cache()
