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

"""Depth-upscale Gemma 3 4B PT by duplicating the LAST k complete 6-layer
cycles in place, each with SOLAR-style identity init (zeroed o_proj /
down_proj). The 4-layer tail is left untouched at the end.

Original layout: 5 cycles x 6 layers + 4-layer tail = 34 layers.

  --dup-last-k 1  -> 40 layers  (block 5 duplicated)
  --dup-last-k 2  -> 46 layers  (blocks 4, 5 duplicated)
  --dup-last-k 3  -> 52 layers  (blocks 3, 4, 5 duplicated)

Each duplicate is inserted directly after its source block, before the tail.

Usage:
    python upscale_gemma3_4b_tail_k.py \
        --src /tmp/verl/models/gemma-3-4b-pt \
        --dst /tmp/verl/models/gemma-3-4b-pt-upscaled-46L \
        --dup-last-k 2
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


def _duplicate_last_k_cycles(layers, k):
    """Insert an identity-init deep-copy of each of the last k complete
    CYCLE-sized chunks directly after that chunk. The trailing partial
    tail is appended unchanged."""
    n = len(layers)
    full = (n // CYCLE) * CYCLE
    n_cycles = full // CYCLE
    assert 1 <= k <= n_cycles, f"k={k} out of range [1, {n_cycles}]"
    dup_set = set(range(n_cycles - k, n_cycles))
    out = []
    for i in range(n_cycles):
        chunk = list(layers[i * CYCLE : (i + 1) * CYCLE])
        out.extend(chunk)
        if i in dup_set:
            for src in chunk:
                cp = copy.deepcopy(src)
                _zero_identity_init_(cp)
                out.append(cp)
    out.extend(list(layers[full:]))
    return out


def _duplicate_last_k_cycles_types(seq, k):
    """Mirror the surgery on a plain list (e.g. layer_types)."""
    n = len(seq)
    full = (n // CYCLE) * CYCLE
    n_cycles = full // CYCLE
    dup_set = set(range(n_cycles - k, n_cycles))
    out = []
    for i in range(n_cycles):
        chunk = list(seq[i * CYCLE : (i + 1) * CYCLE])
        out.extend(chunk)
        if i in dup_set:
            out.extend(chunk)
    out.extend(list(seq[full:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--dup-last-k", type=int, required=True)
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

    k = args.dup_last_k
    new_layers = _duplicate_last_k_cycles(text_model.layers, k)
    text_model.layers = torch.nn.ModuleList(new_layers)
    assert len(new_layers) == n_orig + k * CYCLE

    assert text_cfg.layer_types is not None and len(text_cfg.layer_types) == n_orig
    new_layer_types = _duplicate_last_k_cycles_types(text_cfg.layer_types, k)
    assert len(new_layer_types) == len(new_layers)
    text_cfg.layer_types = new_layer_types

    text_cfg.num_hidden_layers = len(new_layers)
    for idx, layer in enumerate(text_model.layers):
        layer.self_attn.layer_idx = idx
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = idx

    print(f"[surgery] dup_last_k                 = {k}")
    print(f"[surgery] new num_hidden_layers     = {text_cfg.num_hidden_layers}")
    full_pos = [i for i, t in enumerate(new_layer_types) if t == "full_attention"]
    print(f"[surgery] full_attention positions  = {full_pos}")

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
