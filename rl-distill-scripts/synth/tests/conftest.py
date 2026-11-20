import os
import sys

import pytest
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ["WANDB_MODE"] = "disabled"

from data import build, load_config  # noqa: E402


def small_config(name, **task):
    """A configs/<name>.yaml with a 3000-example train split and short SFT, optionally another task spec."""
    cfg = load_config(os.path.join(os.path.dirname(HERE), "configs", f"{name}.yaml"))
    cfg["task"].update(task)
    cfg["data"]["n_train"] = 3000
    cfg["sft"].update(max_steps=60, log_every=10, save_every=10, screen_every=20, screen_k=2)
    cfg["tune"].update(k=4, confirm_k=4, window=10)
    cfg["rl"].update(test_freq=2, val_prompts=8, val_n=4, micro={"2": 32, "4": 32})
    return cfg


@pytest.fixture(scope="session", autouse=True)
def one_rank():
    """fastrl runs in a process group; in-process tests are its only rank (no teardown: destroying it after failed tests
    can hang, and the process exit frees it)."""
    dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1)


@pytest.fixture(scope="session")
def sudoku_data(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("data") / "sudoku")
    build(small_config("sudoku9"), out, workers=8)
    return out


@pytest.fixture(scope="session")
def maze_data(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("data") / "maze")
    cfg = small_config("maze33", size=13)
    cfg["max_response"], cfg["decode"]["max_new"] = 96, 96
    build(cfg, out, workers=8)
    return out
