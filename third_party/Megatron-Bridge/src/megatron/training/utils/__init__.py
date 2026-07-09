# Minimal megatron.training namespace vendored from
# NVIDIA/Megatron-LM@64c3fb86 for megatron-core pip installs that do not ship
# it. Only the helpers the vendored submodules need are included.
import os

import torch


def print_rank_0(message):
    """Print only on global rank 0."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    elif int(os.getenv("RANK", "0")) == 0:
        print(message, flush=True)
