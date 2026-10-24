#!/usr/bin/env python3
"""FLOPs-matched 1.7B medium: direct 1.7B-base RL vs (4B RL to step k) + OPD 4B@k -> 1.7B-base.

Essential-FLOP accounting (no checkpoint recompute, padding, old-logprob recompute, validation, prompt-position LM head):
  linear: 2*N_mm per response token, 2*N_blk per prompt token, per forward (training = 3x forward)
  attention scores (QK^T + AV): 2*A*S^2 per causal forward over S tokens, A = layers*q_heads*head_dim
  RL step: generation for all 1024 samples (prompt prefill once per group) + training ONLY on samples from
           groups with mixed correctness (all-correct / all-wrong groups have zero GRPO advantage)
  OPD step: student generation + teacher scoring + student training, all 128 samples
Per-sample lengths come from the per-step train/generations W&B tables (100 random traces, retokenized), rescaled
so their prompt/response means match the exact logged batch means. Correctness per trace is recovered from its score
minus the DAPO overlong penalty.

  python rl-distill-scripts/analysis/qwen3_flops_matched_opd_vs_rl.py --download DIR   # fetch tables + history
  python rl-distill-scripts/analysis/qwen3_flops_matched_opd_vs_rl.py DIR              # ledger + plot
"""
import argparse, glob, json, os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

RUNS = {
    "rl1p7b": "q3-1.7b-medium-es5-s42-v1",
    "rl4b": "q3-4b-medium-es5-s42-v1",
    **{f"opd_s{k}": f"q3-opd-q4bm-s{k}-to-q1p7b-base-pgk1-base-v1-s42" for k in (20, 50, 100, 150, 200, 250, 510)},
}
SCALARS = ["response_length/mean", "prompt_length/mean", "train/acc_mean", "train/reward_group/all_zero/count",
           "train/reward_group/all_one/count", "train/reward_group/mixed/count", "train/num_responses_all_generated"]
VAL = "val-core/math/acc/mean@16"
OPD_STEPS = 200
RL_N, RL_GROUPS, OPD_BATCH = 16, 64, 128
OVERLONG_START, OVERLONG_LEN = 8192 - 2048, 2048


def download(root):
    import wandb

    api = wandb.Api(timeout=180)
    for tag, rid in RUNS.items():
        run = api.run(f"rl-distill/DAPO/{rid}")
        d = os.path.join(root, tag)
        os.makedirs(d, exist_ok=True)
        hist = [{k: row.get(k) for k in ["_step", *SCALARS, VAL]} for row in run.scan_history() if row.get("_step") is not None]
        json.dump(hist, open(os.path.join(d, "history.json"), "w"))
        files = [f for f in run.files(per_page=1000) if f.name.startswith("media/table/train/generations_")]
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(lambda f: f.download(root=d, replace=False, exist_ok=True), files))
        print(tag, rid, len(hist), "history rows,", len(files), "tables")


def model_consts(name):
    from transformers import AutoConfig

    c = AutoConfig.from_pretrained(name)
    h, L, I, nq, nkv, dh = c.hidden_size, c.num_hidden_layers, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim
    n_blk = L * (2 * h * nq * dh + 2 * h * nkv * dh + 3 * h * I)
    return dict(N=n_blk + h * c.vocab_size, Nb=n_blk, A=L * nq * dh)


def load_run(root, tag, tok):
    """Per-step dict: logged scalars + retokenized (p, r, correct) arrays from that step's trace table."""
    hist = {row["_step"]: row for row in json.load(open(os.path.join(root, tag, "history.json")))}
    tables = {}
    for f in glob.glob(os.path.join(root, tag, "media/table/train/generations_*.table.json")):
        t = json.load(open(f))
        step = t["data"][0][0]
        p = np.array([len(x) for x in tok([row[1] for row in t["data"]], add_special_tokens=False)["input_ids"]], float)
        r = np.array([len(x) for x in tok([row[2] for row in t["data"]], add_special_tokens=False)["input_ids"]], float)
        score = np.array([row[3] for row in t["data"]], float)
        penalty = -np.clip((r - OVERLONG_START) / OVERLONG_LEN, 0, 1) if tag.startswith("rl") else 0.0
        tables[step] = (p, r, score - penalty > 0.5)
    return hist, tables


def per_sample_lengths(hist, tables, step):
    """Table lengths for `step` (nearest logged table if missing) rescaled to the exact batch means."""
    nearest = min(tables, key=lambda s: abs(s - step))
    p, r, correct = tables[nearest]
    row = hist[step]
    return p * row["prompt_length/mean"] / p.mean(), r * row["response_length/mean"] / r.mean(), correct


def fwd(m, p, r):
    """Per-sample forward FLOPs over prompt p + response r (full causal sequence)."""
    return 2 * m["Nb"] * p + 2 * m["N"] * r + 2 * m["A"] * (p + r) ** 2


def rl_step(m, hist, tables, step):
    p, r, correct = per_sample_lengths(hist, tables, step)
    row = hist[step]
    total = row["train/num_responses_all_generated"]
    n_all_one, n_all_zero = RL_N * row["train/reward_group/all_one/count"], RL_N * row["train/reward_group/all_zero/count"]
    mixed_correct = row["train/acc_mean"] * total - n_all_one
    mixed_wrong = total - n_all_one - n_all_zero - mixed_correct

    def trained(x):  # summed over mixed-group samples, using the per-correctness-class mean of x
        cls = lambda mask: x[mask].mean() if mask.any() else x.mean()
        return mixed_correct * cls(correct) + mixed_wrong * cls(~correct)

    prefill_attn, decode_attn = 2 * m["A"] * p**2, 2 * m["A"] * ((p + r) ** 2 - p**2)
    gen_attn = RL_GROUPS * prefill_attn.mean() + total * decode_attn.mean()
    gen = RL_GROUPS * np.mean(2 * m["Nb"] * p) + total * np.mean(2 * m["N"] * r) + gen_attn
    train_attn = 6 * m["A"] * (p + r) ** 2
    train = 3 * fwd(m, p, r)
    return dict(gen=gen, train=trained(train), train_all=total * train.mean(), attn=gen_attn + trained(train_attn),
                frac_trained=(mixed_correct + mixed_wrong) / total)


def opd_step(s, t, hist, tables, step):
    p, r, _ = per_sample_lengths(hist, tables, step)
    gen, teacher, train = fwd(s, p, r).mean(), fwd(t, p, r).mean(), 3 * fwd(s, p, r).mean()
    attn = np.mean(2 * s["A"] * (p + r) ** 2 * 4 + 2 * t["A"] * (p + r) ** 2)
    return dict(gen=OPD_BATCH * gen, teacher=OPD_BATCH * teacher, train=OPD_BATCH * train, attn=OPD_BATCH * attn)


def cumulative(steps_flops):
    steps = sorted(steps_flops)
    return dict(zip(steps, np.cumsum([steps_flops[s] for s in steps])))


def main(root, out_png):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    s, t = model_consts("Qwen/Qwen3-1.7B-Base"), model_consts("Qwen/Qwen3-4B-Base")
    runs = {tag: load_run(root, tag, tok) for tag in RUNS}

    ledgers = {}
    for tag, m in (("rl1p7b", s), ("rl4b", t)):
        hist, tables = runs[tag]
        steps = sorted(k for k, v in hist.items() if k >= 1 and v.get("response_length/mean") is not None)
        led = {k: rl_step(m, hist, tables, k) for k in steps}
        ledgers[tag] = led
        tot = {key: sum(v[key] for v in led.values()) for key in ("gen", "train", "train_all", "attn")}
        print(f"{tag}: steps 1-{steps[-1]}  total {(tot['gen'] + tot['train']) / 1e18:.1f} EFLOP "
              f"(gen {tot['gen'] / 1e18:.1f}, train {tot['train'] / 1e18:.1f}; without zero-adv exclusion "
              f"{(tot['gen'] + tot['train_all']) / 1e18:.1f}); attention share {tot['attn'] / (tot['gen'] + tot['train']):.1%}; "
              f"mean fraction of samples trained {np.mean([v['frac_trained'] for v in led.values()]):.1%}")
    rl_cum = {tag: cumulative({k: v["gen"] + v["train"] for k, v in led.items()}) for tag, led in ledgers.items()}

    hist, _ = runs["rl1p7b"]
    rl_val = sorted((k, v[VAL]) for k, v in hist.items() if v.get(VAL) is not None and k >= 1)
    rl_curve = [(rl_cum["rl1p7b"][k], max(y for kk, y in rl_val if kk <= k), y) for k, y in rl_val]

    points = []
    for k in (20, 50, 100, 150, 200, 250, 510):
        hist, tables = runs[f"opd_s{k}"]
        steps = sorted(st for st, v in hist.items() if 1 <= st <= OPD_STEPS and v.get("response_length/mean") is not None)
        led = {st: opd_step(s, t, hist, tables, st) for st in steps}
        opd_total = sum(v["gen"] + v["teacher"] + v["train"] for v in led.values())
        attn_share = sum(v["attn"] for v in led.values()) / opd_total
        vals = [(st, v[VAL]) for st, v in hist.items() if 1 <= st <= OPD_STEPS and v.get(VAL) is not None]
        best_step, best = max(vals, key=lambda x: x[1])
        teacher = rl_cum["rl4b"][k]
        points.append(dict(k=k, x=teacher + opd_total, teacher=teacher, opd=opd_total, y=best, best_step=best_step,
                           last_step=steps[-1], complete=steps[-1] >= OPD_STEPS, attn_share=attn_share))
        print(f"4B@{k:>3} -> OPD: teacher {teacher / 1e18:6.1f} + OPD(1-{steps[-1]}) {opd_total / 1e18:5.1f} = "
              f"{(teacher + opd_total) / 1e18:6.1f} EFLOP (OPD attn share {attn_share:.1%}); best val "
              f"{100 * best:.1f} @ OPD step {best_step}")

    best_rl = max(rl_val, key=lambda x: x[1])
    print(f"1.7B RL best val {100 * best_rl[1]:.1f} @ step {best_rl[0]} = {rl_cum['rl1p7b'][best_rl[0]] / 1e18:.1f} EFLOP")
    json.dump(dict(rl_curve=rl_curve, points=points), open(os.path.splitext(out_png)[0] + ".json", "w"), indent=1)
    plot(rl_curve, points, out_png)


def plot(rl_curve, points, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    x, best, raw = (np.array(c) for c in zip(*rl_curve))
    ax.plot(x / 1e18, 100 * raw, color="tab:blue", alpha=0.25, lw=1)
    ax.plot(x / 1e18, 100 * best, color="tab:blue", lw=2, label="1.7B-base RL (running best val)")
    for pt in points:
        done = pt["complete"]
        ax.scatter(pt["x"] / 1e18, 100 * pt["y"], s=60, zorder=3, color="tab:red" if done else "white",
                   edgecolor="tab:red", label=None)
        note = f"4B@{pt['k']}" + ("" if done else f" (OPD {pt['last_step']}/{OPD_STEPS})")
        ax.annotate(note, (pt["x"] / 1e18, 100 * pt["y"]), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.scatter([], [], color="tab:red", s=60, label=f"4B RL to step k + OPD into 1.7B-base (best val, first {OPD_STEPS} steps)")
    ax.set_xlabel("Total training FLOPs (EFLOP = 1e18; incl. attention, zero-advantage RL groups untrained)")
    ax.set_ylabel("Val mean@16 (medium, %)")
    ax.set_title("Qwen3 medium: FLOPs-matched direct RL vs RL-teacher + OPD (1.7B student)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("wrote", out_png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out", default="rl-distill-scripts/figures/qwen3_flops_matched_opd_vs_rl_medium.png")
    a = ap.parse_args()
    download(a.root) if a.download else main(a.root, a.out)
