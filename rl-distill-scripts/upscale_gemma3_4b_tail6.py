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

"""Depth-upscale Gemma 3 4B PT by duplicating ONLY the last complete 6-layer
cycle (layers 24..29) once, with SOLAR-style identity init (zeroed o_proj /
down_proj). The duplicate is inserted directly after layer 29, before the
4-layer tail (positions 30..33 in the original).

Original layout: 5 cycles x 6 layers + 4-layer tail = 34 layers.
After surgery:  5 cycles x 6 + 1 duplicated cycle (zero-init) + 4-layer tail = 40 layers.

Usage:
    python upscale_gemma3_4b_tail6.py \
        --src /home/tiger/verl/models/gemma-3-4b-pt \
        --dst /home/tiger/verl/models/gemma-3-4b-pt-upscaled-40L
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
TAIL = 4  # trailing partial cycle in 4B PT (34 = 5*6 + 4)


def _zero_identity_init_(layer):
    torch.nn.init.zeros_(layer.self_attn.o_proj.weight)
    torch.nn.init.zeros_(layer.mlp.down_proj.weight)


def _duplicate_last_full_cycle(layers):
    """Insert an identity-init deep-copy of the LAST complete CYCLE-sized
    chunk right after that chunk (i.e. before the trailing partial tail)."""
    n = len(layers)
    full = (n // CYCLE) * CYCLE
    last_chunk_start = full - CYCLE
    head = list(layers[:full])  # all complete cycles, unchanged
    tail = list(layers[full:])  # untouched 4-layer tail
    dup = []
    for src in layers[last_chunk_start:full]:
        cp = copy.deepcopy(src)
        _zero_identity_init_(cp)
        dup.append(cp)
    return head + dup + tail


def _insert_last_full_cycle_types(seq):
    """Mirror the layer-list surgery on a plain list (e.g. layer_types)."""
    n = len(seq)
    full = (n // CYCLE) * CYCLE
    last_chunk_start = full - CYCLE
    head = list(seq[:full])
    tail = list(seq[full:])
    dup = list(seq[last_chunk_start:full])
    return head + dup + tail


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

    text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    text_model = model.language_model if hasattr(model, "language_model") else model.model

    n_orig = len(text_model.layers)
    print(f"[surgery] original num_hidden_layers = {n_orig}")
    assert n_orig == text_cfg.num_hidden_layers
    assert n_orig % CYCLE == TAIL, f"expected layout n%CYCLE==TAIL ({TAIL}); got n={n_orig}, CYCLE={CYCLE}"

    new_layers = _duplicate_last_full_cycle(text_model.layers)
    text_model.layers = torch.nn.ModuleList(new_layers)
    assert len(new_layers) == n_orig + CYCLE

    assert text_cfg.layer_types is not None and len(text_cfg.layer_types) == n_orig
    new_layer_types = _insert_last_full_cycle_types(text_cfg.layer_types)
    assert len(new_layer_types) == len(new_layers)
    text_cfg.layer_types = new_layer_types

    text_cfg.num_hidden_layers = len(new_layers)
    for idx, layer in enumerate(text_model.layers):
        layer.self_attn.layer_idx = idx
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = idx

    print(f"[surgery] new num_hidden_layers     = {text_cfg.num_hidden_layers}")
    full_pos = [i for i, t in enumerate(new_layer_types) if t == "full_attention"]
    print(f"[surgery] full_attention positions  = {full_pos}")
    print(f"[surgery] layer_types               = {new_layer_types}")

    out = Path(args.dst)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[save] {out}")
    model.save_pretrained(out, safe_serialization=True)

    try:
        AutoTokenizer.from_pretrained(args.src).save_pretrained(out)
        print("[save] tokenizer ok")
    except Exception as e:
        print(f"[save] tokenizer failed: {e}")
    try:
        AutoProcessor.from_pretrained(args.src).save_pretrained(out)
        print("[save] processor ok")
    except Exception as e:
        print(f"[save] processor failed (non-fatal for text-only RL): {e}")

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
