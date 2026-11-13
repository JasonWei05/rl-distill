#!/usr/bin/env python3
"""One model size of the shortest-path maze study, end to end on one multi-GPU machine, fully asynchronous.

Three kinds of work run at the same time and never wait for each other:
  * trainers   -- one sft_pretrain.py per init seed (from scratch, shared data order), checkpoints every --save-steps;
  * screeners  -- screen_watch.py workers spread over all GPUs, screening every --screen-stride-th checkpoint as it lands
                  (goal_reached and is_shortest pass@1 on the dev mazes at --screen-k samples);
  * level tasks -- as soon as a run's screens are clearly past a level's target, select_matched_init.py (averaged
                  candidates around the crossing) -> pick_tune_endpoints.py -> tune_pass1.py (interpolate to the target,
                  confirm with a fresh seed at --final-k samples on all dev mazes) for that (run, level), on whichever GPU
                  has the fewest tasks running.
A run is stopped (stop file) as soon as all 5 levels are queued -- its later checkpoints are not needed, and SCREEN_UNTIL
tells the screeners to skip them -- or once the median of its last 4 screens is >= --stop-at, or at --max-steps. Only every
100th checkpoint is mirrored to S3; a relaunch on another machine retrains a run whose levels are not all done (minutes). The pipeline ends when every trainer, screener and level task has finished. status.json is rewritten
every loop for monitoring.

    python3 maze_pt_pipeline.py --size XS --seeds 0 1 2 --gpus 2 --root /tmp/maze/pt --train-data .../train_optimal.json \
        --val-data .../test.json --tailrl-root /tmp/maze/TailRL --python /tmp/maze/TailRL/.venv/bin/python
"""
import argparse
import glob
import socket
import json
import os
import statistics
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = {  # hidden layers heads kv_heads ffn  (launch_pretrain_matrix.sh)
    "3XS": (96, 4, 4, 2, 384),     # 0.55M (same shape as XS / S, narrower; head dim 24)
    "2XS": (128, 4, 4, 2, 512),    # 1.0M (head dim 32)
    "XS": (192, 4, 4, 2, 768),     # small 2.2M
    "S": (256, 4, 4, 2, 1024),     # medium 3.9M
    "L": (384, 6, 6, 3, 1536),     # large 13.3M
}
LEVELS = {"p01": 0.001, "p03": 0.003, "p1": 0.01, "p3": 0.03, "p10": 0.10}


def screens(run, tag, metric):
    d = {}
    for p in glob.glob(os.path.join(run, f"screen_{tag}*.json")):
        try:
            d.update(json.load(open(p)))
        except (json.JSONDecodeError, OSError):
            pass
    return sorted((int(k.split("-")[1]), v[metric]["pass@1"]) for k, v in d.items() if k.startswith("ckpt-"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", required=True, choices=sorted(ARCH))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--gpu-ids", nargs="+", default=None, help="physical ids of the --gpus GPUs (default 0..gpus-1)")
    ap.add_argument("--root", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--train-cache", default=None, help="pre-tokenized --train-data (directory from tokenize_maze_sft.py); used if it exists")
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--tailrl-root", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--attempt", default="a0", help="run-name suffix; a relaunch retrains unfinished runs under a new one")
    ap.add_argument("--metric", default="is_shortest")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--save-steps", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--max-length", type=int, default=512, help="trainer sequence cap and max_position_embeddings (25x25 needs ~1024)")
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--screen-stride", type=int, default=25)
    ap.add_argument("--screen-k", type=int, default=16)
    ap.add_argument("--screeners-per-gpu", type=int, default=2)
    ap.add_argument("--stop-at", type=float, default=0.20)
    ap.add_argument("--tasks-per-gpu", type=int, default=4, help="level tasks per GPU (after training ends a GPU is ~40%% busy at 2)")
    ap.add_argument("--task-retries", type=int, default=5, help="requeue a failed level task this many times (e.g. after an OOM)")
    ap.add_argument("--retry-delay", type=int, default=180, help="seconds before the n-th retry starts (x n)")
    ap.add_argument("--center-step", type=int, default=10)
    ap.add_argument("--half-window", type=int, default=15)
    ap.add_argument("--margin", type=int, default=50)
    ap.add_argument("--margin-below", type=int, default=200, help="widened x2, x4 if no averaged candidate pair brackets the target")
    ap.add_argument("--select-k", type=int, default=64)
    ap.add_argument("--search-k", type=int, default=256, help="samples per maze while searching for a level (final screens use --final-k)")
    ap.add_argument("--final-k", type=int, default=2048)
    ap.add_argument("--fine-mode", default="fine", choices=["fine", "confirm"], help="tune_pass1.py --fine-mode")
    ap.add_argument("--wandb-project", default=None, help="log each pretraining run to W&B (entity from WANDB_ENTITY)")
    ap.add_argument("--no-tune", action="store_true", help="train + screen only (no level tasks; tune afterwards with exact_tune.py)")
    ap.add_argument("--past-factor", type=float, default=1.5, help="queue a level once the last 3 screens are >= factor x target")
    args = ap.parse_args()

    tag = f"stride{args.screen_stride}_k{args.screen_k}"
    gpu_ids = args.gpu_ids or [str(i) for i in range(args.gpus)]
    assert len(gpu_ids) == args.gpus
    os.makedirs(args.root, exist_ok=True)
    runs = {}
    for seed in args.seeds:
        prior = sorted(glob.glob(os.path.join(args.root, f"{args.size}_is{seed}_a*")))
        host = socket.gethostname()

        def usable(r):   # finished, and every level is done or its checkpoints are here (restored from S3 or this machine)
            if not os.path.exists(os.path.join(r, "DONE")):
                return False
            if args.no_tune:   # every 100th checkpoint is on S3, which is all exact_tune.py --ckpt-stride 100 needs
                return True
            if all(os.path.exists(os.path.join(r, f"tune-{l}", "COMPLETE")) for l in LEVELS):
                return True
            if os.path.exists(os.path.join(r, "CKPTS_SYNCED")):
                return True
            try:
                return open(os.path.join(r, "HOST")).read().strip() == host
            except OSError:
                return False
        done = [r for r in prior if usable(r)]
        runs[seed] = done[-1] if done else os.path.join(args.root, f"{args.size}_is{seed}_{args.attempt}")
    env0 = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": os.path.join(args.tailrl_root, "experiments/text_maze")}

    def proc(cmd, gpu, log):
        os.makedirs(os.path.dirname(log), exist_ok=True)
        return subprocess.Popen(cmd, env={**env0, "CUDA_VISIBLE_DEVICES": gpu_ids[gpu]}, stdout=open(log, "a"),
                                stderr=subprocess.STDOUT, cwd=os.path.join(args.tailrl_root, "experiments/text_maze"))

    # trainers: spread over the GPUs (they are tiny); a run with DONE is not retrained
    trainers = {}
    hid, lay, heads, kv, ffn = ARCH[args.size]
    for i, (seed, run) in enumerate(runs.items()):
        if os.path.exists(os.path.join(run, "DONE")):
            continue
        os.makedirs(run, exist_ok=True)
        with open(os.path.join(run, "HOST"), "w") as f:
            f.write(socket.gethostname())
        cmd = [args.python, os.path.join(args.tailrl_root, "experiments/text_maze/scripts/sft_pretrain.py"),
               "--train_data", args.train_data, "--val_data", args.val_data, "--output_dir", run,
               "--experiment_name", os.path.basename(run), "--hidden_size", str(hid), "--num_hidden_layers", str(lay),
               "--num_attention_heads", str(heads), "--num_key_value_heads", str(kv), "--intermediate_size", str(ffn),
               "--learning_rate", str(args.lr), "--batch_size", str(args.batch), "--micro_batch_size", str(args.batch),
               "--lr_scheduler", "constant", "--warmup_ratio", "0", "--max_steps", str(args.max_steps),
               "--save_steps", str(args.save_steps), "--eval_steps", str(args.eval_steps), "--seed", "0", "--data_seed", "0",
               "--init_seed", str(seed), "--stop_file", os.path.join(run, "STOP"),
               "--max_length", str(args.max_length), "--max_position_embeddings", str(args.max_length)]
        if args.wandb_project:
            cmd += ["--use_wandb", "--project_name", args.wandb_project]
        if args.train_cache:
            cmd += ["--train_cache", args.train_cache]
        trainers[seed] = proc(cmd, i % args.gpus, os.path.join(run, "train.log"))
        print(f"TRAIN_START {os.path.basename(run)} gpu {i % args.gpus}", flush=True)
        time.sleep(5)

    # screeners: stride-sharded across all GPUs; each exits when every run is DONE/FAILED and nothing is left
    n_scr = args.screeners_per_gpu * args.gpus
    screeners = [proc([args.python, os.path.join(HERE, "screen_watch.py"), "--run-dirs", *runs.values(),
                       "--stride", str(args.screen_stride), "--k", str(args.screen_k), "--worker", str(w), "--n-workers", str(n_scr),
                       "--tag", tag, "--tailrl-root", args.tailrl_root, "--poll", "15"],
                      w % args.gpus, os.path.join(args.root, "logs", f"screen_w{w}.log")) for w in range(n_scr)]

    tasks = {}       # (seed, level) -> (Popen, gpu)
    task_state = {}  # (seed, level) -> queued | running | done | failed
    task_fails = {}  # (seed, level) -> failed attempts
    retry_at = {}    # (seed, level) -> earliest start time of the next attempt
    for seed, run in runs.items():
        for lvl in LEVELS:
            if os.path.exists(os.path.join(run, f"tune-{lvl}", "COMPLETE")):
                task_state[(seed, lvl)] = "done"

    def gpu_for_task():
        load = [0] * args.gpus
        for _, g in tasks.values():
            load[g] += 1
        g = min(range(args.gpus), key=lambda i: load[i])
        return g if load[g] < args.tasks_per_gpu else None

    def task_cmd(run, lvl):
        return [args.python, os.path.join(HERE, "level_task.py"), "--run", run, "--level", lvl, "--target", str(LEVELS[lvl]),
                "--tag", tag, "--tailrl-root", args.tailrl_root, "--metric", args.metric, "--margin", str(args.margin),
                "--margin-below", str(args.margin_below), "--center-step", str(args.center_step),
                "--half-window", str(args.half_window), "--select-k", str(args.select_k), "--search-k", str(args.search_k),
                "--final-k", str(args.final_k), "--fine-mode", args.fine_mode]

    t_start = time.time()
    while True:
        status = {"size": args.size, "elapsed_s": int(time.time() - t_start), "runs": {}}
        for seed, run in runs.items():
            p = trainers.get(seed)
            if p is not None and p.poll() is not None:   # trainer exited: mark the run finished for the screeners
                open(os.path.join(run, "DONE" if p.returncode == 0 else "FAILED"), "w").close()
                print(f"TRAIN_END {os.path.basename(run)} rc={p.returncode}", flush=True)
                trainers[seed] = None
            sc = screens(run, tag, args.metric)
            vals = [v for _, v in sc]
            if (sc and not os.path.exists(os.path.join(run, "STOP")) and len(vals) >= 4
                    and statistics.median(vals[-4:]) >= args.stop_at):
                open(os.path.join(run, "STOP"), "w").close()
                print(f"STOP {os.path.basename(run)} at screened step {sc[-1][0]}: median of last 4 {args.metric} "
                      f"{statistics.median(vals[-4:]):.4f} >= {args.stop_at}", flush=True)
            finished = os.path.exists(os.path.join(run, "DONE")) or os.path.exists(os.path.join(run, "FAILED"))
            if sc and all((seed, l) in task_state for l in LEVELS) and not os.path.exists(os.path.join(run, "SCREEN_UNTIL")):
                with open(os.path.join(run, "SCREEN_UNTIL"), "w") as f:   # every level queued: later checkpoints are not needed
                    f.write(str(sc[-1][0]))
                if not os.path.exists(os.path.join(run, "STOP")):
                    open(os.path.join(run, "STOP"), "w").close()
                print(f"ALL_LEVELS_QUEUED {os.path.basename(run)}: stop training, screen up to step {sc[-1][0]}", flush=True)
            for lvl, tgt in LEVELS.items():
                key = (seed, lvl)
                past = len(vals) >= 3 and min(vals[-3:]) >= args.past_factor * tgt
                if not args.no_tune and key not in task_state and (past or (finished and vals and max(vals) >= tgt)):
                    task_state[key] = "queued"
                    print(f"QUEUE {os.path.basename(run)} {lvl}", flush=True)
            status["runs"][os.path.basename(run)] = {
                "seed": seed, "training": trainers.get(seed) is not None, "stopped": os.path.exists(os.path.join(run, "STOP")),
                "finished": finished, "last_screen": sc[-1] if sc else None, "n_screens": len(sc),
                "levels": {lvl: task_state.get((seed, lvl), "waiting") for lvl in LEVELS}}
        # reap and start level tasks
        for key, (p, g) in list(tasks.items()):
            if p.poll() is not None:
                task_state[key] = "done" if p.returncode == 0 else "failed"
                print(f"TASK_{task_state[key].upper()} {os.path.basename(runs[key[0]])} {key[1]} rc={p.returncode}", flush=True)
                del tasks[key]
                if p.returncode != 0 and task_fails.get(key, 0) < args.task_retries:
                    task_fails[key] = task_fails.get(key, 0) + 1
                    task_state[key] = "queued"
                    retry_at[key] = time.time() + args.retry_delay * task_fails[key]
                    print(f"TASK_RETRY {os.path.basename(runs[key[0]])} {key[1]} attempt {task_fails[key] + 1}", flush=True)
        ready = (k for k, s in task_state.items() if s == "queued" and time.time() >= retry_at.get(k, 0))
        for key in sorted(ready, key=lambda k: (LEVELS[k[1]], k[0])):
            g = gpu_for_task()
            if g is None:
                break
            run = runs[key[0]]
            tasks[key] = (proc(task_cmd(run, key[1]), g, os.path.join(run, "tasks", f"{key[1]}.log")), g)
            task_state[key] = "running"
            print(f"TASK_START {os.path.basename(run)} {key[1]} gpu {g}", flush=True)
        screeners_alive = sum(p.poll() is None for p in screeners)
        status.update(screeners_alive=screeners_alive, tasks_running=len(tasks))
        with open(os.path.join(args.root, "status.json.tmp"), "w") as f:
            json.dump(status, f, indent=1)
        os.replace(os.path.join(args.root, "status.json.tmp"), os.path.join(args.root, "status.json"))
        all_trained = all(trainers.get(s) is None for s in runs)
        if all_trained and screeners_alive == 0 and not tasks and not any(s == "queued" for s in task_state.values()):
            break
        time.sleep(30)
    missing = [] if args.no_tune else [f"{os.path.basename(runs[s])}:{l}" for s in runs for l in LEVELS if task_state.get((s, l)) != "done"]
    print(f"PIPELINE_END {args.size} missing={missing}", flush=True)
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
