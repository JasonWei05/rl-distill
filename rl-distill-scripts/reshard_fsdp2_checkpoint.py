#!/usr/bin/env python3
"""Reshard a verl FSDP2 actor checkpoint (per-rank ``torch.save`` files) to a different world size.

verl saves ``model_world_size_{W}_rank_{r}.pt`` / ``optim_world_size_{W}_rank_{r}.pt`` /
``extra_state_world_size_{W}_rank_{r}.pt`` and loads them back by ``(world_size, rank)``, so a
checkpoint written by an 8-GPU run cannot be resumed on 4 GPUs as-is.  Every sharded value is a
``DTensor`` with ``Shard(0)`` on a 1-D ``fsdp`` mesh whose local piece follows ``torch.chunk``
semantics, so the full tensor is the rank-ordered concatenation of the local pieces and the new
shards are ``torch.chunk(full, D, dim=0)``.  Non-DTensor leaves (buffers, Adam ``step`` counters,
``param_groups``) are replicated and are copied from source rank 0.  ``extra_state`` (lr scheduler +
per-rank RNG) for destination rank ``r`` is source rank ``r``'s file (lr scheduler state is identical
on every rank).  Also rewrites ``fsdp_config.json`` and copies ``data.pt`` /
``validation_early_stopping.json`` from the step directory.

Runs on CPU with no GPUs: the destination DTensors are built on a ``_init_backend=False`` device
mesh under a 1-process gloo group, which pickles exactly like the training run's own shards.

    python reshard_fsdp2_checkpoint.py --src-step-dir CKPT/global_step_130 \
        --dst-step-dir NEW_CKPT/global_step_130 --dst-world-size 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _make_dtensor(local: torch.Tensor, full_shape: torch.Size, full_stride: tuple[int, ...], mesh: DeviceMesh) -> DTensor:
    spec = DTensorSpec(mesh=mesh, placements=(Shard(0),), tensor_meta=TensorMeta(shape=full_shape, stride=full_stride, dtype=local.dtype))
    return DTensor(local.contiguous(), spec, requires_grad=False)


def _reshard_leaf(values: list[Any], dst_world: int, mesh: DeviceMesh, path: str, stats: dict[str, int]) -> list[Any]:
    """values = the same leaf from every source rank (rank order). Returns one value per destination rank."""
    first = values[0]
    if isinstance(first, DTensor):
        assert all(isinstance(v, DTensor) for v in values), path
        assert tuple(first.placements) == (Shard(0),), f"{path}: unsupported placements {first.placements}"
        full = torch.cat([v._local_tensor for v in values], dim=0)
        assert full.shape == first.shape, f"{path}: concatenated {tuple(full.shape)} != global {tuple(first.shape)}"
        chunks = list(torch.chunk(full, dst_world, dim=0))
        while len(chunks) < dst_world:  # torch.chunk may return fewer pieces for tiny dim-0 sizes
            chunks.append(full.new_empty((0, *full.shape[1:])))
        stats["dtensors"] += 1
        stats["elements"] += full.numel()
        # clone: torch.chunk returns views of `full`, and torch.save would otherwise serialize the whole
        # underlying storage (the full tensor) into every shard file
        return [_make_dtensor(c.clone(), first.shape, first.stride(), mesh) for c in chunks]
    if isinstance(first, torch.Tensor):
        for v in values[1:]:
            if not torch.equal(v, first):
                raise RuntimeError(f"{path}: replicated tensor differs across source ranks")
        stats["tensors"] += 1
        return [first.clone() for _ in range(dst_world)]
    for v in values[1:]:
        if v != first:
            raise RuntimeError(f"{path}: non-tensor leaf differs across source ranks: {first!r} vs {v!r}")
    return [first for _ in range(dst_world)]


def _reshard_tree(values: list[Any], dst_world: int, mesh: DeviceMesh, path: str, stats: dict[str, int]) -> list[Any]:
    first = values[0]
    if isinstance(first, dict):
        keys = list(first.keys())
        for v in values[1:]:
            assert list(v.keys()) == keys, f"{path}: key order differs across source ranks"
        outs: list[Any] = [type(first)() for _ in range(dst_world)]
        for k in keys:
            for r, piece in enumerate(_reshard_tree([v[k] for v in values], dst_world, mesh, f"{path}.{k}", stats)):
                outs[r][k] = piece
        return outs
    if isinstance(first, (list, tuple)):
        n = len(first)
        assert all(len(v) == n for v in values), path
        cols = [_reshard_tree([v[i] for v in values], dst_world, mesh, f"{path}[{i}]", stats) for i in range(n)]
        return [type(first)(col[r] for col in cols) for r in range(dst_world)]
    return _reshard_leaf(values, dst_world, mesh, path, stats)


def _load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _reshard_component(name: str, src_actor: Path, dst_actor: Path, src_world: int, dst_world: int, mesh: DeviceMesh) -> None:
    t0 = time.time()
    srcs = [_load(src_actor / f"{name}_world_size_{src_world}_rank_{r}.pt") for r in range(src_world)]
    _log(f"{name}: loaded {src_world} source shards in {time.time() - t0:.0f}s")
    stats = {"dtensors": 0, "tensors": 0, "elements": 0}
    outs = _reshard_tree(srcs, dst_world, mesh, name, stats)
    _log(f"{name}: resharded {stats['dtensors']} DTensors ({stats['elements'] / 1e9:.2f}B elements), {stats['tensors']} replicated tensors")
    for r, out in enumerate(outs):
        dst = dst_actor / f"{name}_world_size_{dst_world}_rank_{r}.pt"
        torch.save(out, dst)
        _log(f"{name}: wrote {dst.name} ({dst.stat().st_size / 2**30:.2f} GiB)")
    # verify: the rank-ordered concatenation of the destination shards equals that of the source shards
    checked = 0
    backs = [_load(dst_actor / f"{name}_world_size_{dst_world}_rank_{r}.pt") for r in range(dst_world)]
    for key, src_vals, dst_vals in _iter_leaf_columns(srcs, backs):
        if isinstance(src_vals[0], DTensor):
            src_full = torch.cat([v._local_tensor for v in src_vals], 0)
            dst_full = torch.cat([v._local_tensor for v in dst_vals], 0)
            if not torch.equal(src_full, dst_full):
                raise RuntimeError(f"{name}.{key}: verification mismatch")
            assert dst_vals[0]._spec.mesh.size() == dst_world and dst_vals[0].shape == src_vals[0].shape, key
            checked += 1
    _log(f"{name}: verified {checked} DTensors bit-exact after round trip ({time.time() - t0:.0f}s total)")
    del srcs, outs, backs


def _iter_leaf_columns(srcs: list[Any], dsts: list[Any], path: str = ""):
    first = srcs[0]
    if isinstance(first, dict):
        for k in first:
            yield from _iter_leaf_columns([s[k] for s in srcs], [d[k] for d in dsts], f"{path}.{k}" if path else str(k))
    elif isinstance(first, (list, tuple)):
        for i in range(len(first)):
            yield from _iter_leaf_columns([s[i] for s in srcs], [d[i] for d in dsts], f"{path}[{i}]")
    else:
        yield path, srcs, dsts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-step-dir", type=Path, required=True, help="global_step_N directory holding actor/ + data.pt")
    ap.add_argument("--dst-step-dir", type=Path, required=True)
    ap.add_argument("--dst-world-size", type=int, required=True)
    ap.add_argument("--components", default="model,optim,extra_state")
    ap.add_argument("--device-type", default="cuda", help="device type recorded in the destination mesh (the training run's)")
    args = ap.parse_args()

    src_actor = args.src_step_dir / "actor"
    dst_actor = args.dst_step_dir / "actor"
    cfg = json.loads((src_actor / "fsdp_config.json").read_text())
    assert cfg.get("FSDP_version") == 2, cfg
    src_world = int(cfg["world_size"])
    dst_world = args.dst_world_size
    assert src_world % dst_world == 0 or dst_world % src_world == 0, (src_world, dst_world)
    dst_actor.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
    dist.init_process_group("gloo", rank=0, world_size=1)
    probe = _load(src_actor / f"model_world_size_{src_world}_rank_0.pt")
    src_mesh = next(v for v in probe.values() if isinstance(v, DTensor))._spec.mesh
    assert src_mesh.ndim == 1 and src_mesh.size() == src_world, src_mesh
    mesh = DeviceMesh(args.device_type, torch.arange(dst_world), mesh_dim_names=src_mesh.mesh_dim_names, _init_backend=False)
    _log(f"source mesh {src_mesh} -> destination mesh {mesh}")
    del probe

    for name in args.components.split(","):
        if name == "extra_state":
            for r in range(dst_world):
                src = src_actor / f"extra_state_world_size_{src_world}_rank_{r % src_world}.pt"
                shutil.copyfile(src, dst_actor / f"extra_state_world_size_{dst_world}_rank_{r}.pt")
            _log(f"extra_state: copied {dst_world} files (rank r <- source rank r)")
        else:
            _reshard_component(name, src_actor, dst_actor, src_world, dst_world, mesh)

    (dst_actor / "fsdp_config.json").write_text(json.dumps({**cfg, "world_size": dst_world}, indent=4) + "\n")
    for extra in ("data.pt", "validation_early_stopping.json"):
        if (args.src_step_dir / extra).exists():
            shutil.copyfile(args.src_step_dir / extra, args.dst_step_dir / extra)
    manifest = {
        "source_step_dir": str(args.src_step_dir.resolve()),
        "source_world_size": src_world,
        "destination_world_size": dst_world,
        "components": args.components.split(","),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.dst_step_dir / "reshard_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _log(f"RESHARD_OK {args.dst_step_dir} world_size {src_world} -> {dst_world}")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
