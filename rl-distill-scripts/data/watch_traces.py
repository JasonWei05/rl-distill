#!/usr/bin/env python3
"""Monitor data-parallel trace generation (launch_teacher_gen.sh shards): progress + crash
detection, logged to its own wandb run (train and val stay separate). Exits 0 when every shard
has written its output parquet, non-zero if a shard dies without producing output."""
import argparse, glob, os, re, time
from pathlib import Path


def shard_progress(log_path):
    """Return (done, total) from the last vLLM 'Processed prompts: .. X/Y' line, else (0, 0)."""
    try:
        txt = Path(log_path).read_text(errors="ignore").replace("\r", "\n")
    except FileNotFoundError:
        return 0, 0
    m = re.findall(r"Processed prompts:.*?(\d+)/(\d+)", txt)
    if m:
        return int(m[-1][0]), int(m[-1][1])
    return 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)                 # train | val
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--total-prompts", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--project", default="gemma3-1b-traces")
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--stall-min", type=int, default=30)     # not-done + log idle this long = dead/hung
    ap.add_argument("--max-min", type=int, default=2880)     # 48h safety
    args = ap.parse_args()

    import wandb
    run = wandb.init(project=args.project, name=f"traces-1b-pt-{args.tag}",
                     config=vars(args), reinit=True)
    # Crash detection is stall-based, NOT traceback-based: torch emits benign line-start tracebacks
    # ("Exception ignored in __del__", inductor codecache warnings) that don't kill generation, so
    # matching them false-positives. A shard is crashed only if it produced no output parquet AND its
    # log has gone idle (mtime) for > stall-min (tqdm keeps the mtime fresh while alive).
    t0 = time.time()
    deadline = t0 + args.max_min * 60
    while time.time() < deadline:
        done_shards, tot_done, tot_all, crashed = 0, 0, 0, []
        for s in range(args.num_shards):
            lp = os.path.join(args.logdir, f"shard_{s}.log")
            outp = glob.glob(os.path.join(args.outdir, f"shard_{s:03d}*.parquet"))
            d, t = shard_progress(lp)
            tot_done += d; tot_all += t
            if outp:
                done_shards += 1
            elif os.path.exists(lp):
                idle_min = (time.time() - os.path.getmtime(lp)) / 60
                if idle_min > args.stall_min:
                    crashed.append(s)
        elapsed = time.time() - t0
        pct = (tot_done / tot_all * 100) if tot_all else 0.0
        wandb.log({"shards_done": done_shards, "prompts_done": tot_done,
                   "prompts_total_seen": tot_all, "pct": pct, "elapsed_min": elapsed / 60,
                   "crashed_shards": len(crashed)})
        print(f"[{args.tag}] {time.strftime('%H:%M:%S')} shards_done={done_shards}/{args.num_shards} "
              f"prompts={tot_done}/{tot_all} ({pct:.1f}%) crashed={crashed} elapsed={elapsed/60:.1f}m", flush=True)
        if done_shards == args.num_shards:
            wandb.summary["status"] = "complete"; run.finish()
            print(f"[{args.tag}] ALL {args.num_shards} SHARDS DONE"); return 0
        if crashed:
            wandb.summary["status"] = f"crashed:{crashed}"; run.finish()
            print(f"[{args.tag}] CRASH in shards {crashed}"); return 3
        time.sleep(args.poll)
    wandb.summary["status"] = "timeout"; run.finish()
    print(f"[{args.tag}] TIMEOUT"); return 2


if __name__ == "__main__":
    raise SystemExit(main())
