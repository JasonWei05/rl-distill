"""The chess RL grid's page (configs/chess.yaml, chess_rl_plan.py): the SFT bases' benchmark scores (B1-B5) and every RL
run's validation pass@1 and response length against steps and rollouts sampled, from the run files on S3 and W&B. A
run taken out of the plan by hand has a STOPPED.json ({"reason": ..., "step": ...}) next to its files.

    python chess_report.py --html /tmp/synth/chess_rl.html [--cache /tmp/synth/report_cache] [--note "..."] [--findings FILE]
"""

import argparse
import json
import os
import re
import time

import wandb

from chess_rl_plan import GRID, LRS, MINIBATCHES, STEPS, run_name
from files import read_json
from report import S3, aws

SFT_STEPS = 896  # configs/chess_big.yaml sft.max_steps

BASE_EVALS = ("eval/test_n64/model_{}@384/EVAL.json", "sft/model_{}/EVAL384.json", "eval/test_n16/model_{}@384/EVAL.json")
WANDB = "rl-distill/synth-chess-rl"  # run ids chess-<run name>
STEP_LINE = re.compile(r"^step \d+ val ", re.M)


def base(root, model):
    """The base's highest-sample benchmark eval: pass@k and the per-band pass@1 (%), or None before it lands."""
    path = next((p for p in (os.path.join(root, f.format(model)) for f in BASE_EVALS) if os.path.exists(p)), None)
    if path is None:
        return None
    r = read_json(path)
    return {"n": r["n"], **{k: 100 * v for k, v in r["all"].items() if k != "prompts"},
            "bands": {g: 100 * v["pass@1"] for g, v in r["groups"].items()}}


def train_curves(name):
    """The run's training rollouts from W&B: [[step, mean sampled tokens, mean score %]]."""
    h = wandb.Api().run(f"{WANDB}/chess-{name}").history(keys=["response_length/mean", "train/score"], samples=400, pandas=False)
    return [[x["_step"], x["response_length/mean"], 100 * x["train/score"]] for x in h]


def run(root, model, prompts, n, lr):
    d = os.path.join(root, "rl", run_name(model, prompts, n, lr))
    log = open(os.path.join(d, "job.log")).read() if os.path.exists(os.path.join(d, "job.log")) else None
    best = read_json(os.path.join(d, "best.json")) if os.path.exists(os.path.join(d, "best.json")) else None
    done = read_json(os.path.join(d, "COMPLETE.json")) if os.path.exists(os.path.join(d, "COMPLETE.json")) else None
    stopped = read_json(os.path.join(d, "STOPPED.json")) if os.path.exists(os.path.join(d, "STOPPED.json")) else None
    status = ({"early_stop": "early stop", "max_steps": "max steps"}[done["reason"]] if done else
              f"stopped: {stopped['reason']}" if stopped else "running" if best else "starting" if log is not None else "queued")
    after = (log or "").rsplit("Traceback", 1)
    if len(after) == 2 and not STEP_LINE.search(after[1]) and not done:  # pack retries a failed job, resuming
        status = "error"
    return {"name": run_name(model, prompts, n, lr), "prompts": prompts, "n": n, "rollouts": MINIBATCHES * prompts * n,
            "status": status, "history": [[s, 100 * v] for s, v in best["history"]] if best else [],
            "best": [best["best_step"], 100 * best["best_value"]] if best else None,
            "train": train_curves(run_name(model, prompts, n, lr)) if best else [],
            "since_best": best["since_best"] if best else None}


def sft_big(root, model):
    """The bigger-trace SFT of a model (runs/chess/sft_big, configs/chess_big.yaml): train loss (every 4 steps), validation
    loss (every 10), and the B1-B5 pass@1 / pass@16 (%) of each checkpoint (eval_curve.py, n 16)."""
    run, ev = os.path.join(root, "sft_big", f"model_{model}"), os.path.join(root, "sft_big_eval", f"model_{model}")
    log = open(os.path.join(run, "job.log")).read() if os.path.exists(os.path.join(run, "job.log")) else ""
    val = os.path.join(run, "val.jsonl")
    evals = sorted((int(f[4:-5]), read_json(os.path.join(ev, f))) for f in os.listdir(ev)
                   if f.startswith("step") and f.endswith(".json")) if os.path.isdir(ev) else []
    conf = os.path.join(os.path.dirname(root), "chess_big", "eval", "test_n64")  # n 64 confirmations (plan.py eval)
    confirm = {int(d.split("@")[1]): read_json(os.path.join(conf, d, "EVAL.json"))["all"] for d in os.listdir(conf)
               if d.startswith(f"model_{model}@") and os.path.exists(os.path.join(conf, d, "EVAL.json"))} if os.path.isdir(conf) else {}
    return {"train": [[int(a), float(b)] for a, b in re.findall(r"^step (\d+) loss ([\d.e+-]+)", log, re.M)],
            "confirm": {k: [100 * a["pass@1"], 100 * a["pass@16"]] for k, a in confirm.items()},
            "val": [[r["step"], r["val_loss"]] for r in map(json.loads, open(val))] if os.path.exists(val) else [],
            "evals": [[k, 100 * e["all"]["pass@1"], 100 * e["all"]["pass@16"]] for k, e in evals],
            "done": os.path.exists(os.path.join(run, "DONE")), "started": bool(log)}


def collect(cache):
    root = os.path.join(cache, "chess")
    aws(["aws", "s3", "sync", "--delete", "--only-show-errors", f"{S3}/runs/chess", root, "--exclude", "*",
         "--include", "rl/*/best.json", "--include", "rl/*/COMPLETE.json", "--include", "rl/*/STOPPED.json", "--include", "rl/*/job.log",
         "--include", "eval/*/EVAL.json", "--include", "sft/*/EVAL384.json", "--include", "sft_big/*/job.log",
         "--include", "sft_big/*/val.jsonl", "--include", "sft_big/*/DONE", "--include", "sft_big_eval/*/step*.json"])
    aws(["aws", "s3", "sync", "--only-show-errors", f"{S3}/runs/chess_big/eval", os.path.join(cache, "chess_big", "eval"),
         "--exclude", "*", "--include", "*/EVAL.json"])
    label = {m: m.split("_")[0].upper() for m in GRID}
    return {"updated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "steps": STEPS,
            "bases": [{"key": m, "label": label[m], "tokens": m.split("_")[1], "base": base(root, m)} for m in GRID],
            "sft": {"steps": SFT_STEPS, "models": [{"key": m, "label": label[m], **sft_big(root, m)} for m in GRID]},
            "grids": [{"lr": f"{lr:g}".replace("e-0", "e-"), "models": [{"key": m, "label": label[m], "runs": [run(root, m, p, n, lr) for p, n in GRID[m]]}
                                                    for m in models]} for lr, models in LRS.items()]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="/tmp/synth/report_cache")
    ap.add_argument("--html", required=True)
    ap.add_argument("--note", default="", help="a sentence for the page's status line")
    ap.add_argument("--findings", help="a text file, one finding per line, listed under the introduction")
    a = ap.parse_args()
    data = {**collect(a.cache), "note": a.note,
            "findings": open(a.findings).read().splitlines() if a.findings else []}
    page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chess_report.html")).read()
    with open(a.html, "w") as f:
        f.write(page.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
