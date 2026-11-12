#!/usr/bin/env python3
"""Check and time the maze rollout decoders (TailRL verl/workers/rollout): GraphDecoder vs PrefixSharedDecoder.

check: sample with PrefixSharedDecoder (train layout: P prompts x 16; validation layout: repeated rows with groups split at
the chunk edges), then teacher-force the sampled sequences through the HF model and compare per-token log-probs of the
sampled tokens (the GraphDecoder gap is printed as the baseline).
speed: wall time of one rollout in the production chunking (GraphDecoder: calls of <= CHUNK rows, as
ROLLOUT_MICRO_BATCH_SIZE=1000; PrefixSharedDecoder: one call) for train batches of P prompts x 16 and the 500 x 16
validation batch.
  bench_prefix_decode.py --tailrl <checkout> --model <hf dir> --val <eval500 test.parquet> [--prompts 16,64,256]
"""
import argparse
import sys
import time

import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tailrl", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--prompts", default="16,64,256")
    ap.add_argument("--chunk", type=int, default=1000, help="GraphDecoder rows per call (ROLLOUT_MICRO_BATCH_SIZE)")
    ap.add_argument("--max-new", type=int, default=300)
    args = ap.parse_args()
    sys.path.insert(0, f"{args.tailrl}/experiments/text_maze/verl/workers/rollout")
    from graph_decode import GraphDecoder
    from prefix_decode import PrefixSharedDecoder
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    m = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="sdpa", dtype=torch.float16).cuda().eval()
    df = pd.read_parquet(args.val)
    texts = [p if isinstance(p, str) else p[0]["content"] for p in df["prompt"]]
    enc = [tok(s, add_special_tokens=False, return_tensors="pt").input_ids[0] for s in texts]
    L = min(len(e) for e in enc)
    ids = torch.stack([e[:L] for e in enc]).cuda()
    eos, N = [2, 7], args.max_new
    print(f"model {args.model}: {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M params; {ids.shape[0]} prompts of {L} tokens", flush=True)
    gd, pdec = GraphDecoder(m, N), PrefixSharedDecoder(m, N)

    def sampled_logp(seq, logits):
        resp = seq[:, L:]
        ended = torch.cumsum(torch.isin(resp, torch.tensor(eos, device=resp.device)).long(), 1)
        alive = torch.ones_like(resp, dtype=torch.bool)
        alive[:, 1:] = ended[:, :-1] == 0
        lp = torch.log_softmax(torch.stack(logits, 1), -1).gather(-1, resp[..., None])[..., 0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            ref = m(input_ids=seq).logits[:, L - 1: L - 1 + resp.shape[1]].float()
        lr = torch.log_softmax(ref, -1).gather(-1, resp[..., None])[..., 0]
        return (lp - lr)[alive], alive.sum(1).float()

    for name, f in [("prefix", lambda x: pdec.generate(x, 1.0, eos, 0, 16)), ("graph", lambda x: gd.generate(x, 1.0, eos, 0, 16))]:
        torch.manual_seed(0)
        with torch.autocast("cuda", dtype=torch.float16):
            o = f(ids[:16])
        dlt, lens = sampled_logp(o.sequences, o.logits)
        print(f"check {name}: {dlt.numel()} tokens, response length mean {lens.mean():.1f}; |logp - logp_HF| mean {dlt.abs().mean():.2e} "
              f"max {dlt.abs().max():.2e}; k1 {(-dlt).mean():.2e}", flush=True)
        del o
    xv = ids[:5].repeat_interleave(16, 0)[8:72]
    with torch.autocast("cuda", dtype=torch.float16):
        o = pdec.generate(xv, 1.0, eos, 0, 1)
    dlt, _ = sampled_logp(o.sequences, o.logits)
    print(f"check prefix validation layout (groups of 8/16/16/16/8): |logp - logp_HF| mean {dlt.abs().mean():.2e} max {dlt.abs().max():.2e}", flush=True)

    def timed(fn, reps=2):
        best = None
        for _ in range(reps):
            torch.cuda.synchronize(); t0 = time.time()
            with torch.autocast("cuda", dtype=torch.float16):
                out = fn()
            torch.cuda.synchronize(); dt = time.time() - t0
            best = dt if best is None else min(best, dt)
        torch.cuda.empty_cache()
        return best, out

    for P in [int(p) for p in args.prompts.split(",")]:
        x = ids[:P]
        per = max(args.chunk // 16, 1)
        nch = max(P // per, 1)
        g_t, _ = timed(lambda: [gd.generate(c, 1.0, eos, 0, 16) for c in x.chunk(nch)])
        p_t, _ = timed(lambda: pdec.generate(x, 1.0, eos, 0, 16))
        print(f"train {P} prompts x 16 = {P * 16} rows: graph {g_t:.2f} s ({nch} calls) | prefix {p_t:.2f} s | {g_t / p_t:.1f}x", flush=True)
    xv = ids.repeat_interleave(16, 0)
    nch = max(xv.shape[0] // args.chunk, 1)
    g_t, _ = timed(lambda: [gd.generate(c, 1.0, eos, 0, 1) for c in xv.chunk(nch)], reps=1)
    p_t, _ = timed(lambda: pdec.generate(xv, 1.0, eos, 0, 1, want_logits=False), reps=1)
    print(f"validation {ids.shape[0]} x 16 = {xv.shape[0]} rows: graph {g_t:.2f} s ({nch} calls) | prefix {p_t:.2f} s (1 call) | "
          f"{g_t / p_t:.1f}x; peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
