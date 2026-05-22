# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Depth-upscale Gemma 3 4B PT by duplicating each complete 6-layer chunk in
place with SOLAR-style identity init. The 4-layer tail (positions 30..33) is
left untouched.

Layout: 5 cycles × 6 layers + 4-layer tail = 34 layers.
After surgery: 5 cycles × 12 layers + 4-layer tail = 64 layers.

Identity init per copied layer: zero `self_attn.o_proj.weight` and
`mlp.down_proj.weight` (Gemma3 layers have bias=False on both, so no bias to
handle). The double-norm block then computes `x + RMSNorm(0) = x`.

Usage:
    python upscale_gemma3_4b.py \
        --src /home/tiger/verl/models/gemma-3-4b-pt \
        --dst /home/tiger/verl/models/gemma-3-4b-pt-upscaled-64L
"""

import argparse
import copy
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

CYCLE = 6  # Gemma 3's sliding_window_pattern


def _zero_identity_init_(layer):
    """Zero a layer's two output projections so it computes f(x)=x."""
    torch.nn.init.zeros_(layer.self_attn.o_proj.weight)
    torch.nn.init.zeros_(layer.mlp.down_proj.weight)


def _duplicate_complete_cycles(layers):
    """Given a ModuleList of `n` decoder layers, return a new list where each
    complete CYCLE-sized chunk is followed by an identity-init deep-copy of
    itself. The trailing partial chunk (n % CYCLE layers) is appended unchanged.
    """
    out = []
    n = len(layers)
    full = (n // CYCLE) * CYCLE
    # Complete cycles: duplicate each one in place.
    for start in range(0, full, CYCLE):
        chunk = list(layers[start : start + CYCLE])
        out.extend(chunk)
        chunk_copy = []
        for src in chunk:
            cp = copy.deepcopy(src)
            _zero_identity_init_(cp)
            chunk_copy.append(cp)
        out.extend(chunk_copy)
    # Trailing partial cycle (if any): keep as-is.
    out.extend(list(layers[full:]))
    return out


def _double_complete_cycles(seq):
    """Same chunking logic for a plain list (e.g. layer_types). Doubles each
    complete CYCLE-sized chunk; trailing partial chunk is appended unchanged.
    """
    out = []
    n = len(seq)
    full = (n // CYCLE) * CYCLE
    for start in range(0, full, CYCLE):
        chunk = list(seq[start : start + CYCLE])
        out.extend(chunk)
        out.extend(chunk)
    out.extend(list(seq[full:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"[load] {args.src}")
    model = AutoModelForImageTextToText.from_pretrained(args.src, dtype=dtype, device_map="cpu")
    cfg = model.config

    # Multimodal Gemma3Config wraps a Gemma3TextConfig under .text_config.
    text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    text_model = model.language_model if hasattr(model, "language_model") else model.model

    n_orig = len(text_model.layers)
    print(f"[surgery] original num_hidden_layers = {n_orig}")
    assert n_orig == text_cfg.num_hidden_layers, (
        f"layers ({n_orig}) != text_config.num_hidden_layers ({text_cfg.num_hidden_layers})"
    )

    # 1. Layer surgery.
    new_layers = _duplicate_complete_cycles(text_model.layers)
    text_model.layers = torch.nn.ModuleList(new_layers)

    # 2. layer_types must be doubled in lockstep so per-layer attention type
    #    and RoPE base freq stay correct.
    assert text_cfg.layer_types is not None and len(text_cfg.layer_types) == n_orig, (
        "Gemma 3 config must have layer_types populated (transformers >= 5.0)."
    )
    new_layer_types = _double_complete_cycles(text_cfg.layer_types)
    assert len(new_layer_types) == len(new_layers)
    text_cfg.layer_types = new_layer_types

    # 3. num_hidden_layers and per-layer index bookkeeping.
    text_cfg.num_hidden_layers = len(new_layers)
    for idx, layer in enumerate(text_model.layers):
        layer.self_attn.layer_idx = idx
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = idx

    print(f"[surgery] new num_hidden_layers     = {text_cfg.num_hidden_layers}")
    full_pos = [i for i, t in enumerate(new_layer_types) if t == "full_attention"]
    print(f"[surgery] full_attention positions  = {full_pos}")

    # 4. Save.
    out = Path(args.dst)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[save] {out}")
    model.save_pretrained(out, safe_serialization=True)

    # 5. Tokenizer + processor.
    try:
        AutoTokenizer.from_pretrained(args.src).save_pretrained(out)
        print("[save] tokenizer ✓")
    except Exception as e:
        print(f"[save] tokenizer failed: {e}")
    try:
        AutoProcessor.from_pretrained(args.src).save_pretrained(out)
        print("[save] processor ✓")
    except Exception as e:
        print(f"[save] processor failed (non-fatal for text-only RL): {e}")

    # 6. Sanity reload.
    print("[reload] checking saved model ...")
    reloaded_cfg = AutoConfig.from_pretrained(out)
    reloaded_text_cfg = reloaded_cfg.text_config if hasattr(reloaded_cfg, "text_config") else reloaded_cfg
    print(
        f"[reload] num_hidden_layers={reloaded_text_cfg.num_hidden_layers}  "
        f"len(layer_types)={len(reloaded_text_cfg.layer_types)}"
    )
    assert reloaded_text_cfg.num_hidden_layers == len(new_layers)
    assert len(reloaded_text_cfg.layer_types) == len(new_layers)
    print("[ok]")


if __name__ == "__main__":
    main()
