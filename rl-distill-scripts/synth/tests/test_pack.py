"""pack.py end to end on a scratch S3 prefix with tests/fake_job.py jobs (needs S3 write access: AWS_PROFILE=ml-worker)."""

import json
import os
import subprocess
import sys
import uuid

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(os.path.dirname(HERE), "pack.py")


@pytest.fixture()
def s3():
    uri = f"s3://scale-ml/genai/rl-distill/synth_test/{uuid.uuid4().hex[:12]}"
    yield uri
    subprocess.run(["aws", "s3", "rm", "--recursive", "--only-show-errors", uri], check=True)


def job(name, share, *args):
    return {"run": f"runs/{name}", "done": "DONE", "inputs": [], "share": share, "gpus": 1,
            "cmds": [["tests/fake_job.py", "--out", f"runs/{name}", *args]]}


def pack(tmp_path, s3, jobs, gpus):
    with open(tmp_path / "plan.json", "w") as f:
        json.dump({"data": ["d"], "jobs": jobs}, f)
    return subprocess.run([sys.executable, PACK, "--plan", str(tmp_path / "plan.json"), "--gpus", str(gpus), "--root",
                           str(tmp_path / "root"), "--s3", s3, "--sync-seconds", "1", "--retries", "1"],
                          capture_output=True, text=True)


def times(root, name):
    lines = [x.split() for x in open(root / "runs" / name / "times")]
    return [(float(t), kind, rest) for t, kind, *rest in lines]


def test_pack(tmp_path, s3):
    for rel, text in (("data/d/x", "data"), ("runs/done/DONE", ""), ("inputs/f", "input")):
        subprocess.run(["aws", "s3", "cp", "--only-show-errors", "-", f"{s3}/{rel}"], input=text, text=True, check=True)
    jobs = [job("a", 0.6), job("b", 0.6), job("c", 0.3, "--seconds", "1"), job("done", 0.1),
            {**job("flaky", 0.1, "--fail-once", "--read", "inputs/f"), "inputs": ["inputs"]}, job("bad", 0.1, "--fail")]
    r = pack(tmp_path, s3, jobs, gpus=2)
    root = tmp_path / "root"
    assert r.returncode == 1 and "JOB_FAILED runs/bad" in r.stdout and "SKIP_DONE runs/done" in r.stdout, r.stdout + r.stderr
    assert open(root / "data" / "d" / "x").read() == "data"
    assert not (root / "runs" / "done" / "times").exists()
    (a0, _, (ga,)), (a1, _, _) = times(root, "a")
    (b0, _, (gb,)), (b1, _, _) = times(root, "b")
    assert ga != gb  # 0.6 + 0.6 > 1: one per GPU
    assert len(times(root, "flaky")) == 3 and "JOB_RETRY runs/flaky" in r.stdout  # start, retried start, end
    for name in ("a", "b", "c", "flaky"):
        assert subprocess.run(["aws", "s3", "ls", f"{s3}/runs/{name}/DONE"], capture_output=True).returncode == 0
    r2 = pack(tmp_path, s3, jobs[:5], gpus=1)  # a relaunch: everything is done on S3
    assert r2.returncode == 0 and r2.stdout.count("SKIP_DONE") == 5, r2.stdout


def test_one_gpu_serializes_by_share(tmp_path, s3):
    r = pack(tmp_path, s3, [job("a", 0.6), job("b", 0.6), job("c", 0.4)], gpus=1)
    assert r.returncode == 0, r.stdout + r.stderr
    root = tmp_path / "root"
    (a0, _, _), (a1, _, _) = times(root, "a")
    (b0, _, _), (b1, _, _) = times(root, "b")
    (c0, _, _), (c1, _, _) = times(root, "c")
    assert b0 >= a1 - 0.5 and c0 < a1  # b waits for a; c (0.4) runs next to a


def test_sync_survives_replaced_files(tmp_path, s3):
    """A 24 MB file replaced (atomically, as state.pt is) while it uploads still lands on S3 whole: its trailing digest
    matches its contents."""
    import argparse
    import hashlib
    import threading
    sys.path.insert(0, os.path.dirname(PACK))
    import pack
    from files import atomic
    run = tmp_path / "root" / "runs" / "x"
    os.makedirs(run)
    stop = threading.Event()

    def churn():
        while not stop.is_set():
            body = os.urandom(24 * 2 ** 20)
            atomic(str(run / "state.bin"), lambda tmp: open(tmp, "wb").write(body + hashlib.sha256(body).digest()))
    atomic(str(run / "state.bin"), lambda tmp: open(tmp, "wb").write(b"x" + hashlib.sha256(b"x").digest()))
    t = threading.Thread(target=churn)
    t.start()
    p = pack.Packer.__new__(pack.Packer)
    p.a, p.sync_lock = argparse.Namespace(root=str(tmp_path / "root"), s3=s3), threading.Lock()
    for _ in range(3):
        assert p.sync_up({"run": "runs/x"}, tries=3)
        data = subprocess.run(["aws", "s3", "cp", f"{s3}/runs/x/state.bin", "-"], capture_output=True, check=True).stdout
        assert hashlib.sha256(data[:-32]).digest() == data[-32:]
    stop.set()
    t.join()
    assert not [n for n in os.listdir(tmp_path / "root" / "runs") if n.startswith(".snap")]


def test_multi_gpu_job(tmp_path, s3):
    """A 2-GPU job (share 1.0 on each) gets both GPUs; a 0.5-share job waits until it ends."""
    big = {**job("big", 1.0), "gpus": 2}
    r = pack(tmp_path, s3, [big, job("small", 0.5, "--seconds", "1")], gpus=2)
    assert r.returncode == 0, r.stdout + r.stderr
    root = tmp_path / "root"
    (b0, _, (bg,)), (b1, _, _) = times(root, "big")
    (s0, _, _), _ = times(root, "small")
    assert bg == "0,1" and s0 >= b1 - 0.5


def test_fractional_jobs_pack_together(tmp_path, s3):
    """Two 0.5-share jobs share one GPU (the fullest that fits), so a 1.0-share job starts on the other at once."""
    r = pack(tmp_path, s3, [job("a", 0.5, "--seconds", "2"), job("b", 0.5, "--seconds", "2"), job("c", 1.0)], gpus=2)
    assert r.returncode == 0, r.stdout + r.stderr
    root = tmp_path / "root"
    (a0, _, (ga,)), (a1, _, _) = times(root, "a")
    (b0, _, (gb,)), _ = times(root, "b")
    (c0, _, (gc,)), _ = times(root, "c")
    assert ga == gb != gc and c0 < a1


def test_job_that_cannot_fit_fails_at_once(tmp_path, s3):
    """A job asking for more GPUs than the pod has fails up front instead of waiting forever; the others run."""
    r = pack(tmp_path, s3, [{**job("big", 1.0), "gpus": 3}, job("small", 0.5)], gpus=2)
    assert r.returncode == 1 and "JOB_FAILED runs/big" in r.stdout and "JOB_DONE runs/small" in r.stdout, r.stdout + r.stderr
