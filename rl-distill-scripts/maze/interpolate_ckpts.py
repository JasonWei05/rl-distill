#!/usr/bin/env python3
"""Linear weight interpolation between two checkpoint directories (same architecture).

Used to hit a target starting capability between two averaged candidates when the 20-step center
grid is not fine enough: theta(alpha) = (1 - alpha) * theta_a + alpha * theta_b. Both candidates
should be close on the training trajectory (consecutive averaging windows), where interpolation
is well behaved. Configs and tokenizer are copied from --a.

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/interpolate_ckpts.py --a <run>/avg-5880-w60 --b <run>/avg-5900-w60 --alpha 0.5 --out <run>/interp-5880-5900-a0.50
"""

import argparse
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--alpha", type=float, required=True, help="0 = a, 1 = b")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sa = load_file(os.path.join(args.a, "model.safetensors"))
    sb = load_file(os.path.join(args.b, "model.safetensors"))
    assert sa.keys() == sb.keys(), "checkpoints have different parameter sets"
    out = {}
    for k in sa:
        assert sa[k].shape == sb[k].shape, f"shape mismatch at {k}"
        out[k] = ((1.0 - args.alpha) * sa[k].to(torch.float32) + args.alpha * sb[k].to(torch.float32)).to(sa[k].dtype).contiguous()
    os.makedirs(args.out, exist_ok=True)
    save_file(out, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
    for f in os.listdir(args.a):
        if f != "model.safetensors" and os.path.isfile(os.path.join(args.a, f)):
            shutil.copy(os.path.join(args.a, f), os.path.join(args.out, f))
    with open(os.path.join(args.out, "INTERPOLATED_FROM.txt"), "w") as f:
        f.write(f"a={args.a}\nb={args.b}\nalpha={args.alpha}\n")
    print(f"wrote {args.out} = (1-{args.alpha})*a + {args.alpha}*b")


if __name__ == "__main__":
    main()
