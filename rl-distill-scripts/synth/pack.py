"""Run a plan (plan.py) on this machine's GPUs: jobs packed onto GPUs by their GPU share, run dirs mirrored to S3.

For each job, in plan order (heaviest first): if its done marker is on S3 it is skipped; otherwise its run dir and inputs
are restored from S3 and it waits for its number of GPUs ("gpus") with room for its share each (the fullest such GPUs,
so fractional jobs pack together and leave whole GPUs free), then runs its commands
in order with CUDA_VISIBLE_DEVICES set (a failed command is retried, resuming, up to --retries times; a retried job goes back to the
head of the queue). Every --sync-seconds and at the end of each job its run dir is mirrored to S3. Exits 1 if any job
failed (a relaunched pod resumes from S3).

    python pack.py --plan plan.json --gpus 4 --root /tmp/synth --s3 s3://scale-ml/genai/rl-distill/synth
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def aws(*args, check=True):
    return subprocess.run(["aws", "s3", *args, "--only-show-errors"], check=check).returncode == 0


def s3_exists(uri):
    return subprocess.run(["aws", "s3", "ls", uri], stdout=subprocess.DEVNULL).returncode == 0


class Packer:
    def __init__(self, a):
        self.a = a
        with open(a.plan) as f:
            self.plan = json.load(f)
        self.free = [1.0] * a.gpus
        self.running, self.failed, self.lock = {}, [], threading.Lock()
        self.sync_lock = threading.Lock()  # one upload at a time: a later snapshot never lands before an earlier one

    def path(self, rel):
        return os.path.join(self.a.root, rel)

    def uri(self, rel):
        return f"{self.a.s3}/{rel}"

    def sync_up(self, job, tries=1):
        """Mirror a run dir to S3; False if every try failed. The upload reads a hard-link snapshot of the run dir: a file
        the job replaces meanwhile (state.pt, every save) keeps its snapshot inode, so a multipart upload, which re-reads
        the file per part, cannot mix two versions."""
        run = self.path(job["run"])
        for i in range(tries):
            if i:
                time.sleep(30 * i)
            with self.sync_lock:
                snap = tempfile.mkdtemp(dir=os.path.dirname(run), prefix=".snap")
                ok = subprocess.run(["cp", "-al", run, os.path.join(snap, "run")]).returncode == 0 and aws(
                    "sync", os.path.join(snap, "run"), self.uri(job["run"]), "--exclude", "wandb/*", "--exclude", "*.tmp*", check=False)
                shutil.rmtree(snap)
            if ok:
                return True
        return False

    def syncer(self):
        while True:
            time.sleep(self.a.sync_seconds)
            with self.lock:
                jobs = [j for j, _ in self.running.values()]
            for job in jobs:
                if not self.sync_up(job):  # transient: the next round (and the job's final sync) catch up
                    print(f"SYNC_RETRY {job['run']}", flush=True)

    def prepare(self, job):
        """False if the job is done on S3; else restores its run dir and inputs."""
        if s3_exists(self.uri(f"{job['run']}/{job['done']}")):
            return False
        for rel in [job["run"], *job["inputs"]]:
            aws("sync", self.uri(rel), self.path(rel))
        return True

    def start(self, job, gpus):
        """The job's commands ([script, *args]: a synth script, paths relative to --root) in order, as one process."""
        cmd = " && ".join(shlex.join([sys.executable, os.path.join(HERE, c[0]), *c[1:]]) for c in job["cmds"])
        with open(os.path.join(self.path(job["run"]), "job.log"), "a") as log:
            return subprocess.Popen(["bash", "-c", cmd], cwd=self.a.root, env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus))},
                                    stdout=log, stderr=subprocess.STDOUT)

    def run(self):
        for d in self.plan["data"]:
            aws("sync", self.uri(f"data/{d}"), self.path(f"data/{d}"))
        threading.Thread(target=self.syncer, daemon=True).start()
        queue = []
        for job in self.plan["jobs"]:
            if self.prepare(job):
                os.makedirs(self.path(job["run"]), exist_ok=True)
                queue.append(job)
            else:
                print(f"SKIP_DONE {job['run']}", flush=True)
        for job in [j for j in queue if j["gpus"] > len(self.free) or j["share"] > 1]:  # would wait forever
            queue.remove(job)
            print(f"JOB_FAILED {job['run']}: {job['gpus']} GPUs at share {job['share']} on a {len(self.free)}-GPU pod", flush=True)
            self.failed.append(job["run"])
        tries = {j["run"]: 0 for j in queue}
        while queue or self.running:
            for name, (job, (p, gpus)) in list(self.running.items()):
                rc = p.poll()
                if rc is None:
                    continue
                with self.lock:
                    del self.running[name]
                for g in gpus:
                    self.free[g] += job["share"]
                if not self.sync_up(job, tries=4):  # its work stays here; the exit code makes the supervisor relaunch
                    print(f"SYNC_FAILED {name}", flush=True)
                    self.failed.append(name)
                elif rc == 0:
                    print(f"JOB_DONE {name}", flush=True)
                elif tries[name] < self.a.retries:
                    tries[name] += 1
                    print(f"JOB_RETRY {name} rc={rc} try {tries[name]}", flush=True)
                    queue.insert(0, job)
                else:
                    print(f"JOB_FAILED {name} rc={rc}", flush=True)
                    self.failed.append(name)
            for job in list(queue):
                gpus = sorted((g for g in range(len(self.free)) if self.free[g] + 1e-9 >= job["share"]), key=lambda g: self.free[g])
                if len(gpus) < job["gpus"]:
                    continue
                gpus = sorted(gpus[:job["gpus"]])
                queue.remove(job)
                for g in gpus:
                    self.free[g] -= job["share"]
                with self.lock:
                    self.running[job["run"]] = (job, (self.start(job, gpus), gpus))
                print(f"JOB_START {job['run']} gpus {gpus} share {job['share']} free {[round(self.free[g], 2) for g in gpus]}", flush=True)
            time.sleep(5)
        print(f"PACK_END failed={self.failed}", flush=True)
        return 1 if self.failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--root", required=True, help="local root: data/ and the run dirs")
    ap.add_argument("--s3", required=True, help="S3 root with the same layout")
    ap.add_argument("--sync-seconds", type=int, default=300)
    ap.add_argument("--retries", type=int, default=2)
    raise SystemExit(Packer(ap.parse_args()).run())
