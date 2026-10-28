#!/usr/bin/env python
"""Pretraining cross-entropy of the matched S0 / L0 checkpoints of the text-maze budget study.

The pairs were matched on goal-reached pass@1 only (make_pair.py). This script checks whether they are also matched on
the pretraining objective: it replicates sft_pretrain.py's validate_loss (CE over the path tokens after PATH_START,
micro-batches of 8 in file order, mean of batch means) on the SFT validation set, which is the 1000 dev mazes with one
reference random path each, so the numbers are directly comparable to the `eval/val_loss` curve in
`pretrain20k/<run>/eval_metrics.jsonl` (bf16 reproduces it exactly). It adds a token-weighted CE with a per-maze
bootstrap CI, the per-sequence NLL of the reference path, the predictive entropy at the path positions, and for every
pair a paired bootstrap of the S0 - L0 difference (same mazes for both models).

Optional extras: `--members` evaluates the 7 SWA members of each matched checkpoint (the noise floor of a single
checkpoint); `--controls` evaluates S at L's step and L at S's step (what step matching would have given).
CPU only: the models have 4M / 13M parameters and a run takes seconds per checkpoint.

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/ce_of_pairs.py --members --controls
    # -> rl-distill-scripts/maze/ce_of_pairs.json (+ a summary table on stdout)
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RUNS = os.path.expanduser(os.environ.get("TAILRL_MAZE_RUNS", "~/tailrl_maze_runs"))
VAL = os.path.expanduser(os.environ.get("TAILRL_MAZE_VAL", "~/tailrl_maze_data/sft_data/test.json"))
PAIRS = ["p10", "p1", "p01"]
# S at L's matched step and L at S's matched step (cross copies kept in pretrain20k for exactly this comparison)
CONTROLS = {"p10": (10040, 4020), "p1": (9160, 3240), "p01": (8080, 2680)}
S_RUN, L_RUN = "S_ds0_is4", "L_ds0_is0_rep"


def load_examples(tok, max_length=512):
    data = json.load(open(VAL))
    ps_id = tok.encode("PATH_START", add_special_tokens=False)[0]
    out = []
    for item in data:
        ids = tok.encode(item["sequence"], add_special_tokens=False)[:max_length]
        i = ids.index(ps_id)
        out.append((ids, [-100] * (i + 1) + ids[i + 1:]))
    return out


@torch.no_grad()
def evaluate(model, examples, bs=8, pad=0):
    model.eval()
    batch_means, seq_nll, seq_ntok, tok_ent = [], [], [], []
    for s in range(0, len(examples), bs):
        chunk = examples[s:s + bs]
        L = max(len(ids) for ids, _ in chunk)
        input_ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        attn = torch.zeros((len(chunk), L), dtype=torch.long)
        labels = torch.full((len(chunk), L), -100, dtype=torch.long)
        for i, (ids, lab) in enumerate(chunk):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            attn[i, :len(ids)] = 1
            labels[i, :len(lab)] = torch.tensor(lab)
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        batch_means.append(out.loss.item())
        logp = torch.log_softmax(out.logits[:, :-1].float(), -1)
        tgt = labels[:, 1:]
        mask = tgt != -100
        nll = -logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        ent = -(logp.exp() * logp).sum(-1)
        for i in range(len(chunk)):
            m = mask[i]
            seq_nll.append(nll[i][m].sum().item())
            seq_ntok.append(int(m.sum()))
            tok_ent.extend(ent[i][m].tolist())
    seq_nll, seq_ntok = np.array(seq_nll), np.array(seq_ntok)
    idx = np.random.default_rng(0).integers(0, len(seq_nll), size=(2000, len(seq_nll)))
    boots = seq_nll[idx].sum(1) / seq_ntok[idx].sum(1)
    return {
        "val_loss_sft": float(np.mean(batch_means)),
        "ce_token": float(seq_nll.sum() / seq_ntok.sum()),
        "ce_token_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "nll_seq_mean": float(seq_nll.mean()),
        "entropy_token": float(np.mean(tok_ent)),
        "n_seq": int(len(seq_nll)),
        "n_tok": int(seq_ntok.sum()),
        "seq_nll": seq_nll.tolist(),
        "seq_ntok": seq_ntok.tolist(),
    }


def paired_diff(a, b, n_boot=5000, seed=1):
    """Token-weighted CE difference a - b with a bootstrap CI over mazes (both evaluated on the same mazes)."""
    an, bn, nt = np.array(a["seq_nll"]), np.array(b["seq_nll"]), np.array(a["seq_ntok"])
    assert (nt == np.array(b["seq_ntok"])).all()
    idx = np.random.default_rng(seed).integers(0, len(an), size=(n_boot, len(an)))
    boots = (an[idx] - bn[idx]).sum(1) / nt[idx].sum(1)
    return {"diff": float((an - bn).sum() / nt.sum()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def averaged_from(path):
    txt = open(os.path.join(path, "AVERAGED_FROM.txt")).read()
    run = re.search(r"run=(\S+)", txt).group(1)
    members = [int(x) for x in re.search(r"members=\[([^\]]*)\]", txt).group(1).split(",")]
    return run, int(re.search(r"center=(\d+)", txt).group(1)), members


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ce_of_pairs.json"))
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="bfloat16 reproduces the logged eval/val_loss; float32 (default) is the precise number")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--members", action="store_true", help="also evaluate the 7 SWA members of each matched checkpoint")
    ap.add_argument("--controls", action="store_true", help="also evaluate S at L's step and L at S's step")
    ap.add_argument("--dump-seq", action="store_true", help="keep the per-maze NLL arrays in the JSON")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)

    jobs = []  # (pair, model, kind, label, path)
    for pair in PAIRS:
        for m in ["S0", "L0"]:
            p = os.path.join(RUNS, "matched", f"{pair}_{m}")
            run, center, members = averaged_from(p)
            jobs.append((pair, m, "matched", f"avg-{center}-w60", p))
            if args.members:
                for st in members:
                    jobs.append((pair, m, "member", f"ckpt-{st}", os.path.join(run, f"ckpt-{st}")))
    if args.controls:
        for pair, (s_step, l_step) in CONTROLS.items():
            jobs.append((pair, "S0", "control", f"ckpt-{l_step} (S at L's step)",
                         os.path.join(RUNS, "pretrain20k", S_RUN, f"ckpt-{l_step}")))
            jobs.append((pair, "L0", "control", f"ckpt-{s_step} (L at S's step)",
                         os.path.join(RUNS, "pretrain20k", L_RUN, f"ckpt-{s_step}")))

    tok = AutoTokenizer.from_pretrained(jobs[0][4])
    examples = load_examples(tok)
    results = []
    for pair, m, kind, label, path in jobs:
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
        r = evaluate(model, examples, pad=tok.pad_token_id)
        r.update(pair=pair, model=m, kind=kind, label=label, path=path, dtype=args.dtype)
        results.append(r)
        print(f"{pair:>4} {m} {label:<28} val_loss={r['val_loss_sft']:.4f} ce/tok={r['ce_token']:.4f} "
              f"[{r['ce_token_ci95'][0]:.4f},{r['ce_token_ci95'][1]:.4f}] nll/seq={r['nll_seq_mean']:.3f} "
              f"H/tok={r['entropy_token']:.4f} ({time.time() - t0:.1f}s)", flush=True)

    matched = {(r["pair"], r["model"]): r for r in results if r["kind"] == "matched"}
    paired = {}
    print(f"\npair   S0 ce/tok  L0 ce/tok  S0-L0     paired 95% CI         S0 H/tok  L0 H/tok   ({args.dtype})")
    for pair in PAIRS:
        s, l = matched[(pair, "S0")], matched[(pair, "L0")]
        d = paired_diff(s, l)
        paired[pair] = dict(d, s0=s["label"], l0=l["label"], s0_ce=s["ce_token"], l0_ce=l["ce_token"],
                            s0_entropy=s["entropy_token"], l0_entropy=l["entropy_token"])
        print(f"{pair:<5}  {s['ce_token']:.4f}     {l['ce_token']:.4f}    {d['diff']:+.4f}   "
              f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]   {s['entropy_token']:.3f}     {l['entropy_token']:.3f}")
    if not args.dump_seq:
        for r in results:
            r.pop("seq_nll")
            r.pop("seq_ntok")
    json.dump({"val_set": VAL, "dtype": args.dtype, "paired": paired, "checkpoints": results},
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
