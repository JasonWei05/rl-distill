"""Write pack.py plans: the jobs of one or more pods (--pods: balanced by GPU share), heaviest first.

  sft: per config x size x seed, sft.py then tune.py (run dir runs/<config>/sft/<size>_s<seed>, done marker TUNED); with
       --inits (HF model dirs, relative to the S3 root), sft.py from each instead (runs/<config>/sft/<init name>, DONE)
  eval: per config x model (--models: HF model dirs, or <sft run>@<step>), eval.py on --split with --n samples per prompt
       (runs/<config>/eval/<split>_n<n>/<model name>, done marker EVAL.json)
  theirs: chess, their SFT trainer (theirs.py) from each of --inits on --gpus GPUs (runs/chess/theirs_sft/<init name>, DONE)
  rl:  per config x size x seed x level x the size's prompts per step (rl.batches[size]), fastrl.py from that SFT run's
       tuned base (runs/<config>/rl/<size>_<level>_s<seed>_b<batch>, done marker COMPLETE.json) on rl.gpus[batch] GPUs
       (data parallel); levels the tuning could not reach (tune.json on S3) are left out. --positions picks entries of
       rl.batches[size]. --minibatches k samples k x the batch per step and takes k updates on mini-batches of the batch
       (micro-batches as for the batch; GPUs and GPU share as for the k x batch prompts sampled per step); it and --lr
       name the run (rl_name)
GPU shares (the fraction of a GPU a job takes in pack.py) come from the config's pack section. Jobs whose done marker is
on S3 are left out; pack.py resumes the others from whatever S3 holds of them.

    python plan.py sft --configs sudoku9 maze33 --sizes 2XS XS S --seeds 0 1 2 --out sft.json
    python plan.py rl --configs maze33 --sizes 2XS --seeds 0 1 2 --pods 3 --out rl-maze33   # rl-maze33-{0,1,2}.json
    python plan.py eval --configs chess --models chess/ckpt/sft/model_20m_16B --n 64 --out eval-chess.json
"""

import argparse
import json
import os
import subprocess

from data import load_config

S3 = "s3://scale-ml/genai/rl-distill/synth"


def config(name):
    return load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{name}.yaml"))


def sft_jobs(a):
    jobs = []
    for cfg in a.configs:
        share = config(cfg)["pack"]["sft"]
        for init in a.inits:
            name, run = os.path.basename(init), f"runs/{cfg}/sft/{os.path.basename(init)}"
            jobs.append({"run": run, "done": "DONE", "inputs": [init], "share": share, "gpus": 1, "cmds": [
                ["sft.py", "--data", f"data/{cfg}", "--size", name, "--seed", "0", "--init", init, "--out", run,
                 *(["--chunk", str(a.chunk)] if a.chunk else []), "--wandb-project", f"synth-{cfg}-sft", "--wandb-id", f"{cfg}-{name}"]]})
        for size in a.sizes:
            for seed in a.seeds:
                run, data = f"runs/{cfg}/sft/{size}_s{seed}", f"data/{cfg}"
                jobs.append({"run": run, "done": "TUNED", "inputs": [], "share": share, "gpus": 1, "cmds": [
                    ["sft.py", "--data", data, "--size", size, "--seed", str(seed), "--out", run,
                     "--wandb-project", f"synth-{cfg}-sft", "--wandb-id", f"{cfg}-{size}-s{seed}"],
                    ["tune.py", "--run", run, "--data", data]]})
    return jobs


def eval_jobs(a):
    jobs = []
    for cfg in a.configs:
        share = config(cfg)["pack"]["eval"]
        for model in a.models:
            path, _, step = model.partition("@")
            name = os.path.basename(path) + (f"@{step}" if step else "")
            run = f"runs/{cfg}/eval/{a.split}_n{a.n}/{name}"
            jobs.append({"run": run, "done": "EVAL.json", "inputs": [path], "share": share, "gpus": 1, "cmds": [
                ["eval.py", *(["--run", path, "--step", step] if step else ["--model", path]), "--data", f"data/{cfg}",
                 "--split", a.split, "--n", str(a.n), "--out", f"{run}/EVAL.json"]]})
    return jobs


def theirs_jobs(a):
    inputs = ["chess/pre2post-chess", "chess/pylib", "chess/traces/fixed"]
    return [{"run": f"runs/chess/theirs_sft/{os.path.basename(init)}", "done": "DONE", "inputs": [*inputs, init], "share": 1.0,
             "gpus": a.gpus, "cmds": [["theirs.py", "--repo", inputs[0], "--pylib", inputs[1], "--model", init, "--data", inputs[2],
                                       "--out", f"runs/chess/theirs_sft/{os.path.basename(init)}", "--save-interval", "32"]]}
            for init in a.inits]


def s3_keys(configs):
    """Every key under runs/<config>/ on S3, relative to the S3 root."""
    out = set()
    for cfg in configs:
        ls = subprocess.run(["aws", "s3", "ls", "--recursive", f"{S3}/runs/{cfg}/"], capture_output=True, text=True)
        assert ls.returncode == 0 or not ls.stderr, ls.stderr  # an empty prefix exits 1 with no error
        out.update(key[key.index("runs/"):] for key in (line.split()[-1] for line in ls.stdout.splitlines()))
    return out


def rl_name(size, level, seed, batch, minibatches=1, lr=None):
    """The run name of an RL job: batch is the prompts per mini-batch, minibatches the updates per step (x<k>), lr the
    learning rate when not the config's (_lr<lr>)."""
    return f"{size}_{level}_s{seed}_b{batch}" + (f"x{minibatches}" if minibatches > 1 else "") + (f"_lr{lr:g}" if lr else "")


def tune_log(cfg, size, seed):
    uri = f"{S3}/runs/{cfg}/sft/{size}_s{seed}/tune.json"
    return json.loads(subprocess.run(["aws", "s3", "cp", uri, "-"], check=True, capture_output=True, text=True).stdout)


def rl_jobs(a):
    jobs = []
    for cfg in a.configs:
        c = config(cfg)
        gpus, share = c["rl"]["gpus"], c["pack"]["rl"]
        for size in a.sizes:
            for seed in a.seeds:
                log = tune_log(cfg, size, seed)
                for level in a.levels:
                    if not log[level]["reachable"]:
                        print(f"skip {cfg} {size} s{seed} {level}: not reached in pretraining")
                        continue
                    base = f"runs/{cfg}/sft/{size}_s{seed}/tune/{level}"
                    for batch in [c["rl"]["batches"][size][i] for i in a.positions]:
                        name = rl_name(size, level, seed, batch, a.minibatches, a.lr)
                        sampled = str(batch * a.minibatches)
                        jobs.append({"run": f"runs/{cfg}/rl/{name}", "done": "COMPLETE.json", "inputs": [base],
                                     "share": share[sampled], "gpus": gpus[sampled], "cmds": [
                                         ["fastrl.py", "--base", base, "--data", f"data/{cfg}", "--out", f"runs/{cfg}/rl/{name}",
                                          "--batch", str(batch * a.minibatches), "--seed", str(seed), "--gpus", str(gpus[sampled]),
                                          "--steps", str(a.steps), "--minibatches", str(a.minibatches),
                                          *(["--lr", str(a.lr)] if a.lr else []),
                                          *(["--patience", str(a.patience)] if a.patience else []), "--wandb-project", f"synth-{cfg}-rl",
                                          "--wandb-id", f"{cfg}-{name}"]]})
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["sft", "rl", "eval", "theirs"])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--sizes", nargs="+", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=[])
    ap.add_argument("--inits", nargs="+", default=[], help="sft: HF model dirs (relative to the S3 root) to start from")
    ap.add_argument("--chunk", type=int, help="sft --inits: rows per forward")
    ap.add_argument("--models", nargs="+", default=[], help="eval: HF model dirs or <sft run>@<step>")
    ap.add_argument("--split", default="test", help="eval: the dataset split")
    ap.add_argument("--n", type=int, default=64, help="eval: samples per prompt")
    ap.add_argument("--gpus", type=int, default=4, help="theirs: GPUs per run (their DDP; 4 as in the paper)")
    ap.add_argument("--levels", nargs="+", default=["p1", "p3", "p10"])
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--positions", type=int, nargs="+", default=[0, 1, 2, 3], help="rl: which of each size's rl.batches")
    ap.add_argument("--minibatches", type=int, default=1, help="rl: sample k x the batch, k updates per step (name x<k>)")
    ap.add_argument("--lr", type=float, help="rl: instead of rl.lr (name _lr<lr>)")
    ap.add_argument("--patience", type=int, help="rl: instead of rl.patience (validations without a new best)")
    ap.add_argument("--pods", type=int, default=1, help="split into this many plans <out>-<i>.json of about equal share")
    ap.add_argument("--out", required=True, help="plan file (with --pods: the prefix)")
    a = ap.parse_args()
    assert a.kind not in ("sft", "rl") or (a.sizes and a.seeds) or a.inits, "sft / rl: --sizes and --seeds (or sft --inits)"
    load = lambda j: j["share"] * j["gpus"]
    keys = s3_keys(a.configs)
    jobs = {"sft": sft_jobs, "rl": rl_jobs, "eval": eval_jobs, "theirs": theirs_jobs}[a.kind](a)
    jobs = sorted((j for j in jobs if f"{j['run']}/{j['done']}" not in keys), key=lambda j: -load(j))
    pods = [[] for _ in range(a.pods)]
    for job in jobs:  # heaviest first, each to the lightest pod
        min(pods, key=lambda p: sum(map(load, p))).append(job)
    for i, pod in enumerate(pods):
        out = a.out if a.pods == 1 else f"{a.out}-{i}.json"
        with open(out, "w") as f:
            json.dump({"data": a.configs, "jobs": pod}, f, indent=1)
        print(f"{len(pod)} jobs, {sum(map(load, pod)):.1f} GPUs of shares -> {out}")
