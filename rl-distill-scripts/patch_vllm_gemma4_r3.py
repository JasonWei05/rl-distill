#!/usr/bin/env python3
"""Patch vLLM 0.25.1's routed-expert manager for Gemma 4 R3.

The capturer already resolves Gemma 4's ``top_k_experts`` through
``_get_num_experts_per_tok`` when allocating its buffers.  A remaining logger
argument bypasses that helper and reads ``hf_config.num_experts_per_tok``
directly, which raises during engine startup for Gemma 4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

_BUGGY = """            hf_config.num_hidden_layers,
            hf_config.num_experts_per_tok,
            self.routed_experts_by_slot.dtype.name,
"""
_FIXED = """            hf_config.num_hidden_layers,
            num_experts_per_tok,
            self.routed_experts_by_slot.dtype.name,
"""


def patch_file(path: Path) -> str:
    source = path.read_text()
    buggy_count = source.count(_BUGGY)
    fixed_count = source.count(_FIXED)
    if buggy_count == 1:
        path.write_text(source.replace(_BUGGY, _FIXED, 1))
        return "patched"
    if buggy_count == 0 and fixed_count == 1:
        return "already-patched"
    raise RuntimeError(
        f"unexpected routed_experts_capturer source at {path}: buggy_count={buggy_count}, fixed_count={fixed_count}"
    )


def _installed_capturer_path() -> Path:
    import vllm

    version = getattr(vllm, "__version__", "")
    if version.split("+", 1)[0] != "0.25.1":
        raise RuntimeError(f"Gemma 4 R3 patch is validated only for vLLM 0.25.1, got {version!r}")
    return Path(vllm.__file__).resolve().parent / "model_executor/layers/fused_moe/routed_experts_capturer.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, help="Override target path for tests")
    args = parser.parse_args()
    path = args.path or _installed_capturer_path()
    result = patch_file(path)
    print(f"VLLM_GEMMA4_R3_PATCH={result} path={path}")


if __name__ == "__main__":
    main()
