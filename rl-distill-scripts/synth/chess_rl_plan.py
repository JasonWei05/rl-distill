"""The chess RL grid's pack.py plans (pods chess-rl1, borrowing false, and chess-rl2, borrowing true): fastrl.py from
the matched SFT bases (chess/bases/model_<model>@384), 4 updates per step, 5k steps with the early-stop rule, one seed,
at each lr of LRS for its models. Runs finished (COMPLETE.json) or taken out by hand (STOPPED.json) on S3 are left
out. Order: by lr in LRS order, smallest batch first, with the runs sampling over 4,096 rollouts per step (20M P256, 50M
P128) last; pack.py backfills half-GPU jobs. chess-rl2 runs the 2.5e-5 group-size-16 runs, chess-rl1 the rest (pod).

    python chess_rl_plan.py --pod chess-rl1 --out /tmp/synth/plans/chess-rl1.json
"""

import argparse
import json

from plan import s3_keys

# model: its runs as (prompts per update P, group size n); a step samples MINIBATCHES x P prompts. The larger models get
# 1/2 and 1/4 of the 20M's prompts, about the same training FLOPs per step.
GRID = {"20m_16B": [(16, 4), (16, 16), (64, 16), (256, 16)], "50m_1.1B": [(8, 4), (8, 16), (32, 16), (128, 16)],
        "100m_0.53B": [(4, 4), (4, 16), (16, 16), (64, 16)]}
MICRO = {"20m_16B": 64, "50m_1.1B": 32, "100m_0.53B": 16}
# lr: its models (the 100M also at 1e-5: it failed at 5e-5 and 2.5e-5)
LRS = {5e-5: list(GRID), 2.5e-5: list(GRID), 1e-5: ["100m_0.53B"]}
MINIBATCHES, STEPS = 4, 5000


def pod(lr, n):
    return "chess-rl2" if lr == 2.5e-5 and n == 16 else "chess-rl1"


def run_name(model, prompts, n, lr):
    return f"{model}_s0_b{prompts}x{MINIBATCHES}_n{n}_lr{lr:g}"


def job(model, i, lr):
    prompts, n = GRID[model][i]
    name, base, share = run_name(model, prompts, n, lr), f"chess/bases/model_{model}@384", 0.5 if i < 2 else 1.0
    run = f"runs/chess/rl/{name}"
    return {"run": run, "done": "COMPLETE.json", "inputs": [base], "share": share, "gpus": 1, "cmds": [[
        "fastrl.py", "--base", base, "--data", "data/chess", "--out", run, "--batch", str(MINIBATCHES * prompts),
        "--minibatches", str(MINIBATCHES), "--n", str(n), "--micro", str(MICRO[model]), "--lr", f"{lr:g}",
        "--seed", "0", "--steps", str(STEPS), "--decode-mem-gb", "60" if share == 1.0 else "24",
        "--wandb-project", "synth-chess-rl", "--wandb-id", f"chess-{name}"]]}


def jobs(name):
    keys = s3_keys(["chess"])
    big = lambda model, i: MINIBATCHES * GRID[model][i][0] * GRID[model][i][1] > 4096
    order = sorted(((model, i, lr) for lr, models in LRS.items() for model in models for i in range(4)),
                   key=lambda t: (big(t[0], t[1]), list(LRS).index(t[2]), t[1], list(GRID).index(t[0])))
    out = [job(*t) for t in order if pod(t[2], GRID[t[0]][t[1]][1]) == name]
    return [j for j in out if not {f"{j['run']}/COMPLETE.json", f"{j['run']}/STOPPED.json"} & keys]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pod", required=True, choices=["chess-rl1", "chess-rl2"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plan = {"data": ["chess"], "jobs": jobs(a.pod)}
    with open(a.out, "w") as f:
        json.dump(plan, f, indent=1)
    for j in plan["jobs"]:
        print(j["run"], j["share"])
