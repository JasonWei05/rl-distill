"""A task's dataset, built from an experiment config (configs/<name>.yaml), as uint8 token arrays.

Every split is `<split>.npy`, shape (N, prompt_len + max_response): the prompt, then the demonstration's response
(ending DONE <eos>), then <pad>. Example i of a split comes from seed seed_base + offset + i; the splits are disjoint seed
ranges, and a puzzle drawn twice is kept only in the split that drew it first (dev, test, val, train).
  dev    1000  screening and base tuning; its first 500 are the RL validation prompts
  test   1000  held out for final numbers
  val     256  SFT validation loss
  train  data.n_train   SFT demonstrations and RL prompts
A task with its own splits (task.splits(): (name, offset, n), as Chess, built from released files) replaces these; its
example may be None (dropped), and examples whose info has a "group" are recorded per group (rows in order).
meta.json holds the config, the shapes and summary statistics of the demonstrations. Every stage reads its settings from
it; only the FROZEN sections must match what a model or run was built with, so tune / decode / rl settings (and an SFT
run's max_steps / stop_at, to extend it) can change later: --refresh rewrites meta.json's config from the yaml (after
checking the FROZEN sections are unchanged).

    python data.py --config configs/maze33.yaml --out /tmp/synth/data/maze33 --workers 64 [--refresh]
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import time

import numpy as np
import torch
import yaml

from files import read_json, write_json
from tasks import make_task

SPLITS = (("dev", 0, 1000), ("test", 1000, 1000), ("val", 2000, 256), ("train", 10_000, None))
FROZEN = ("task", "max_response", "data", "sft")  # the sections the data and the SFT runs (hence every base) were built with
EXTENDABLE = ("max_steps", "stop_at", "screen_split")  # sft settings that only decide when training stops (and what it
# screens on): a resumed run may change them
WORKER = {}


def frozen(cfg):
    return {k: {x: y for x, y in cfg[k].items() if x not in EXTENDABLE} if k == "sft" else cfg[k] for k in FROZEN}


def same_build(a, b):
    """Whether configs a and b agree on the FROZEN sections (sft up to its EXTENDABLE settings)."""
    return frozen(a) == frozen(b)


def load_config(path):
    """An experiment config, in its JSON form (as meta.json stores it: mapping keys are strings); its name is the file's
    stem."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return json.loads(json.dumps({"name": os.path.splitext(os.path.basename(path))[0], **cfg}))


def _init(task_spec, max_response):
    WORKER.update(task=make_task(task_spec), max_response=max_response)


def _example(seed):
    task, R = WORKER["task"], WORKER["max_response"]
    ex = task.example(seed)
    if ex is None:
        return None
    key, prompt, response, info = ex
    assert len(response) <= R, f"seed {seed}: a {len(response)}-token demonstration exceeds max_response {R}"
    row = bytes(prompt + response) + bytes(R - len(response))
    return hashlib.blake2b(key, digest_size=16).digest(), row, {**info, "response_len": len(response)}


def summary(infos):
    """Mean, p99 and max of every info field."""
    out = {}
    for k in infos[0]:
        x = np.sort(np.array([i[k] for i in infos], dtype=np.float64))
        out[k] = {"mean": round(float(x.mean()), 4), "p99": float(x[int(0.99 * (len(x) - 1))]), "max": float(x[-1])}
    return out


def build(cfg, out, workers):
    task = make_task(cfg["task"])
    width = task.prompt_len + cfg["max_response"]
    os.makedirs(out, exist_ok=True)
    meta = {"config": cfg, "prompt_len": task.prompt_len, "width": width, "splits": {}}
    seen, t0 = set(), time.time()
    # forkserver: the workers do not inherit the parent's threads (torch's), which fork() can deadlock
    with multiprocessing.get_context("forkserver").Pool(workers, initializer=_init, initargs=(cfg["task"], cfg["max_response"])) as pool:
        for name, offset, n in task.splits() if hasattr(task, "splits") else SPLITS:
            n = n or cfg["data"]["n_train"]
            first = cfg["data"]["seed_base"] + offset
            path = os.path.join(out, f"{name}.npy")
            arr = np.lib.format.open_memmap(path + ".tmp", mode="w+", dtype=np.uint8, shape=(n, width))
            kept, infos, groups, skipped = 0, [], {}, 0
            for ex in pool.imap(_example, range(first, first + n), chunksize=64):
                if ex is None or ex[0] in seen:
                    skipped += ex is None
                    continue
                key, row, info = ex
                seen.add(key)
                arr[kept] = np.frombuffer(row, dtype=np.uint8)
                if "group" in info:
                    groups.setdefault(info.pop("group"), [kept, 0])[1] += 1
                infos.append(info)
                kept += 1
            arr.flush()
            if kept < n:
                np.save(path, arr[:kept])
                os.remove(path + ".tmp")
            else:
                os.replace(path + ".tmp", path)
            meta["splits"][name] = {"n": kept, "repeats_dropped": n - kept - skipped, "first_seed": first, "stats": summary(infos),
                                    **({"skipped": skipped} if skipped else {}), **({"groups": groups} if groups else {})}
            print(f"{name}: {kept} examples ({n - kept - skipped} repeats, {skipped} skipped), {time.time() - t0:.0f}s", flush=True)
    write_json(os.path.join(out, "meta.json"), meta)
    return meta


class Dataset:
    """A built dataset: its task and memory-mapped splits."""

    def __init__(self, path):
        self.path = path
        self.meta = read_json(os.path.join(path, "meta.json"))
        self.cfg = self.meta["config"]
        self.task = make_task(self.cfg["task"])
        self.prompt_len, self.width = self.meta["prompt_len"], self.meta["width"]

    def split(self, name):
        """(N, width) uint8, memory-mapped."""
        return np.load(os.path.join(self.path, f"{name}.npy"), mmap_mode="r")

    def prompts(self, name, n=None):
        """The first n prompts of a split as a (n, prompt_len) int64 tensor."""
        return torch.from_numpy(np.ascontiguousarray(self.split(name)[:n, :self.prompt_len])).long()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--refresh", action="store_true", help="only rewrite meta.json's config from --config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.refresh:
        meta = read_json(os.path.join(a.out, "meta.json"))
        assert same_build(meta["config"], cfg), f"{a.config} changes a frozen section ({FROZEN}): build a new dataset"
        meta["config"] = cfg
        write_json(os.path.join(a.out, "meta.json"), meta)
    else:
        print(json.dumps(build(cfg, a.out, a.workers)["splits"], indent=1))


if __name__ == "__main__":
    main()
