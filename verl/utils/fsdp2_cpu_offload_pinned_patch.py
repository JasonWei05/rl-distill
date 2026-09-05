# Copyright 2026 rl-distill fork. Licensed under the Apache License, Version 2.0.
"""Make FSDP2's CPU-offload gradient accumulation copy through pinned memory.

With ``CPUOffloadPolicy`` FSDP2 keeps the sharded fp32 gradient on the host. For every
micro-batch after the first one in an optimizer step, ``foreach_reduce`` accumulates the new
reduce-scattered shard into it. Upstream (torch 2.11) does that with a *blocking* copy into a
freshly allocated *pageable* CPU tensor::

    new_sharded_grad = new_sharded_grad.to(torch.device("cpu"), non_blocking=non_blocking)

with ``non_blocking=False`` whenever accumulating. A pageable D2H copy is driven by the CPU
through a staging buffer and pays page faults on the new allocation, so on a 2x H100 node the
E2B RL update spent ~4 s per micro-batch (about 100 per step) in that line at 100% CPU.

This patch swaps only that branch for: allocate the destination from PyTorch's caching *pinned*
host allocator (reused across micro-batches), issue the D2H copy as an asynchronous DMA on the
same stream, then synchronize that stream before the CPU add. The data movement is identical, so
gradients and the optimizer trajectory are unchanged; only the transfer mechanism differs.

Opt in with ``VERL_FSDP2_CPU_OFFLOAD_PINNED_ACCUM=1``. The patch rewrites the installed function
from its own source and fails loudly (leaving torch untouched) if the expected snippet is missing,
so a torch upgrade cannot silently apply a stale transformation.
"""

from __future__ import annotations

import inspect
import logging
import os
import textwrap

logger = logging.getLogger(__name__)

_ENV_FLAG = "VERL_FSDP2_CPU_OFFLOAD_PINNED_ACCUM"
_ORIGINAL_SNIPPET = """                new_sharded_grad = new_sharded_grad.to(
                    torch.device("cpu"), non_blocking=non_blocking
                )
"""
_PATCHED_SNIPPET = """                if non_blocking:
                    new_sharded_grad = new_sharded_grad.to(
                        torch.device("cpu"), non_blocking=True
                    )
                else:
                    # rl-distill: accumulate path. Pinned destination from the caching host
                    # allocator + async DMA + stream sync instead of a blocking pageable copy.
                    _pinned_dst = torch.empty_like(
                        new_sharded_grad, device=torch.device("cpu"), pin_memory=True
                    )
                    _pinned_dst.copy_(new_sharded_grad, non_blocking=True)
                    device_handle.current_stream().synchronize()
                    new_sharded_grad = _pinned_dst
"""
_APPLIED = False


def is_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "0").lower() in ("1", "true", "yes")


def apply() -> bool:
    """Rebind ``foreach_reduce`` in FSDP2 to the pinned-accumulate variant. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives, _fsdp_param_group

    original = _fsdp_collectives.foreach_reduce
    source = textwrap.dedent(inspect.getsource(original))
    # dedent() removed the common indent (zero for a module-level def), so the snippet's
    # 16-space indentation must be present verbatim.
    if source.count(_ORIGINAL_SNIPPET) != 1:
        raise RuntimeError(
            f"{__name__}: expected exactly one occurrence of the CPU-offload copy snippet in "
            f"torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_reduce, found "
            f"{source.count(_ORIGINAL_SNIPPET)}; refusing to patch this torch version"
        )
    patched_source = source.replace(_ORIGINAL_SNIPPET, _PATCHED_SNIPPET)
    # `@torch.no_grad()` decorator is part of the retrieved source; compile in the defining
    # module's namespace so every helper (_get_gradient_divide_factors, DTensor, ...) resolves.
    namespace = _fsdp_collectives.__dict__
    code = compile(patched_source, _fsdp_collectives.__file__ + " <rl-distill pinned-accum patch>", "exec")
    exec(code, namespace)  # noqa: S102 - re-defines foreach_reduce inside its own module
    patched = namespace["foreach_reduce"]
    if patched is original:
        raise RuntimeError(f"{__name__}: exec did not rebind foreach_reduce")
    patched.__wrapped_original__ = original
    _fsdp_collectives.foreach_reduce = patched
    # _fsdp_param_group imported the name at import time; rebind it there too (the call site).
    _fsdp_param_group.foreach_reduce = patched
    _APPLIED = True
    logger.warning("%s: FSDP2 CPU-offload gradient accumulation now copies through pinned memory", __name__)
    print(f"[{__name__}] FSDP2_CPU_OFFLOAD_PINNED_ACCUM_PATCH_APPLIED", flush=True)
    return True


def apply_if_enabled() -> bool:
    if not is_enabled():
        return False
    return apply()
