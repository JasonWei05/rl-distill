"""Benchmark every checkpoint of an SFT run as it lands: eval.py on each ckpt/<step>.safetensors in step order (waiting
for the next one while the run trains), into <out>/step<step>.json, until the run's DONE and its last checkpoint are
evaluated; then <out>/DONE. Run next to the SFT job on the same machine (pack.py: the SFT run is this job's input);
fails once no new checkpoint has landed for --idle-hours (the SFT job failed or runs elsewhere).

    python eval_curve.py --run runs/chess/sft_big/model_20m_16B --data data/chess_big --n 16 --out runs/chess/sft_big_eval/model_20m_16B
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--idle-hours", type=float, default=3.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ckpt, seen, last_new = os.path.join(a.run, "ckpt"), set(), time.time()
    while True:
        done = os.path.exists(os.path.join(a.run, "DONE"))  # read before the listing: DONE follows the last checkpoint
        steps = sorted(int(f.split(".")[0]) for f in os.listdir(ckpt) if f.endswith(".safetensors")) if os.path.isdir(ckpt) else []
        if set(steps) - seen:
            seen, last_new = set(steps), time.time()
        todo = [s for s in steps if not os.path.exists(os.path.join(a.out, f"step{s}.json"))]
        if todo:
            subprocess.run([sys.executable, os.path.join(HERE, "eval.py"), "--run", a.run, "--step", str(todo[0]), "--data", a.data,
                            "--split", a.split, "--n", str(a.n), "--out", os.path.join(a.out, f"step{todo[0]}.json")], check=True)
        elif done:
            break
        elif time.time() - last_new > a.idle_hours * 3600:
            sys.exit(f"no new checkpoint in {a.run} for {a.idle_hours} h and no DONE")
        else:
            time.sleep(60)
    open(os.path.join(a.out, "DONE"), "w").close()
