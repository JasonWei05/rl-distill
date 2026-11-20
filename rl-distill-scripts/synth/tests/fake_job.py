"""A pack.py test job: records its start / end time in its run dir; --fail-once fails the first attempt, --fail always
fails, --read checks an input file restored from S3."""
import argparse
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seconds", type=float, default=2)
ap.add_argument("--fail-once", action="store_true")
ap.add_argument("--fail", action="store_true")
ap.add_argument("--read")
a = ap.parse_args()
if a.read:
    assert open(a.read).read() == "input"
with open(os.path.join(a.out, "times"), "a") as f:
    f.write(f"{time.time()} start {os.environ['CUDA_VISIBLE_DEVICES']}\n")
time.sleep(a.seconds)
if a.fail or (a.fail_once and not os.path.exists(os.path.join(a.out, "tried"))):
    open(os.path.join(a.out, "tried"), "w").close()
    raise SystemExit(3)
with open(os.path.join(a.out, "times"), "a") as f:
    f.write(f"{time.time()} end\n")
open(os.path.join(a.out, "DONE"), "w").close()
