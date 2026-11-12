#!/usr/bin/env python3
"""Pick the two averaged candidates a pass@1 tuning run interpolates between, from a select_<level>_set.json.

The endpoints bracket the target. When the candidates re-measured at --final-k bracket it, the tightest such pair is
used (both endpoints then carry precise measurements); otherwise the candidate with the best measurement closest to the
target, plus its nearest neighbour on the other side of the target on the (noisier) grid screens. Prints JSON {a, b, pa, pb}; pa / pb are only given when measured at --final-k (tune_pass1.py measures the
others itself).

    python3 rl-distill-scripts/maze/pick_tune_endpoints.py --select <run>/select_p1_set.json --target 0.01
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", required=True)
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--final-k", type=int, default=2048)
    ap.add_argument("--metric", default="goal_reached", choices=["goal_reached", "is_shortest"])
    args = ap.parse_args()
    d = json.load(open(args.select))
    cands = {}
    for key, v in d.items():
        if not key.startswith("avg-") or "@k" not in key:
            continue
        name, k = key.split("@k")
        center, k = int(name.split("-")[1]), int(k)
        if center not in cands or k > cands[center][0]:
            cands[center] = (k, v[args.metric]["pass@1"], name)
    centers = sorted(cands)
    fin = [c for c in centers if cands[c][0] == args.final_k]
    lo = max((c for c in fin if cands[c][1] <= args.target), key=lambda c: cands[c][1], default=None)
    hi = min((c for c in fin if cands[c][1] > args.target), key=lambda c: cands[c][1], default=None)
    if lo is not None and hi is not None:
        print(json.dumps({"a": cands[lo][2], "b": cands[hi][2], "pa": cands[lo][1], "pb": cands[hi][1]}))
        return
    best = min(centers, key=lambda c: (cands[c][0] != args.final_k, abs(cands[c][1] - args.target)))
    above = cands[best][1] > args.target
    # pass@1 rises with step on these runs: the other side of the target is earlier if best is above it, later if below
    side = [c for c in centers if (c < best if above else c > best)]
    other = None
    for c in (reversed(side) if above else side):
        if (cands[c][1] <= args.target) == above:
            other = c
            break
    other = other if other is not None else (side[-1] if above and side else side[0] if side else best)
    lo, hi = sorted((best, other), key=lambda c: cands[c][1])
    if not cands[lo][1] <= args.target < cands[hi][1]:
        raise SystemExit(f"no candidate pair brackets {args.target}: closest {cands[lo][2]} {cands[lo][1]:.6f}, "
                         f"{cands[hi][2]} {cands[hi][1]:.6f} (widen the selection window)")
    out = {"a": cands[lo][2], "b": cands[hi][2]}
    if cands[lo][0] == args.final_k:
        out["pa"] = cands[lo][1]
    if cands[hi][0] == args.final_k:
        out["pb"] = cands[hi][1]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
