#!/usr/bin/env python3
"""Convert an IFM "Towards Looped Models Done Right" Part I dense checkpoint (xLLM native format) to HF Llama.

Every IFM Part I dense layer is a pre-norm Llama layer (RMSNorm, GQA attention with RoPE, SwiGLU), so a checkpoint
whose forward pass is only a sequence of layer calls converts exactly into a `LlamaForCausalLM` with one HF layer per
executed layer: d28 -> 28 layers, d112 -> 112, and looped models are unrolled (dense-ouro -> 28 x 4, dense-middle ->
8 + 12 x 8 + 8), which is also the untied initialization. Recipes with anything between layers (input injection,
random/zero recurrent-state init, hierarchical H/L states, MoE) are refused.

Weight mapping (xLLM -> HF): embed -> embed_tokens; attention.norm -> input_layernorm; wq/wk/wv/wo -> q/k/v/o_proj;
nffn.norm -> post_attention_layernorm; fc1/fc3/fc2 -> gate/up/down_proj (xLLM: fc2(silu(fc1 x) * fc3 x));
output.final_norm -> norm; output.output -> lm_head. xLLM applies RoPE to interleaved (2i, 2i+1) pairs as complex
numbers; HF Llama rotates halves, so q/k rows are permuted per head (the standard Meta -> HF permutation).
xLLM's RMSNorm stores its gain as an offset (initialized at zero, applied as `weight + 1.0`); HF applies the gain
directly, so every norm weight gets +1, added in the stored dtype exactly as xLLM's forward does.

    python convert_ifm_dense_to_llama.py --src /tmp/ifm_eval/raw/dense-d28-336b --dst /tmp/ifm_eval/hf/dense-d28-336b
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def execution_schedule(m: dict) -> list[int]:
    """Physical layer index of every executed layer, mirroring xllm.models.looped (Part I recipes)."""
    n = m["num_layers"]
    if m.get("num_experts"):
        raise ValueError("MoE checkpoints are not supported")
    if m["arch"] == "transformer":
        if m.get("loop_input_injection", "none") != "none" or m.get("dense_prelude_input_injection", "none") != "none":
            raise ValueError("input injection is not a plain layer sequence")
        start, end, times = m["loop_start_layer"], m.get("loop_end_layers") or n, m["loop_times"]
        return list(range(start)) + list(range(start, end)) * times + list(range(end, n))
    if m["arch"] == "huginn":
        if m.get("huginn_input_injection", "none") != "none":
            raise ValueError("Huginn input injection is not a plain layer sequence")
        if m.get("huginn_state_init") != "input":
            raise ValueError(f"recurrent state init {m.get('huginn_state_init')!r} is not a plain layer sequence")
        if m.get("huginn_hierarchical_state", "none") != "none":
            raise ValueError("hierarchical H/L recurrent states are not supported")
        if m.get("huginn_depth_control"):
            raise ValueError("depth-controlled (Part II) models are not supported")
        p, r, c = m["huginn_prelude_layers"], m["huginn_recurrent_layers"], m["huginn_coda_layers"]
        if p + r + c != n:
            raise ValueError("prelude + recurrent + coda layers do not cover the model")
        return list(range(p)) + list(range(p, p + r)) * m["loop_times"] + list(range(p + r, n))
    raise ValueError(f"unsupported arch {m['arch']!r}")


def norm_gain(w: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """xLLM GroupRMSNorm applies `weight + 1.0` in the runtime dtype; HF LlamaRMSNorm applies `weight`."""
    return w.to(dtype) + 1.0


def rope_permute(w: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Rows (2i, 2i+1) of each head -> rows (i, i + head_dim/2): interleaved complex RoPE to HF rotate_half."""
    return w.view(n_heads, head_dim // 2, 2, w.shape[1]).transpose(1, 2).reshape(n_heads * head_dim, w.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="xLLM artifact dir (config.json + safetensors + tokenizer/)")
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()

    xcfg = json.loads((args.src / "config.json").read_text())
    m = xcfg["model"]
    if not m.get("swiglu") or not m.get("apply_rmsnorm") or m.get("qknorm") or m.get("apply_attn_gate") or m.get("scale_emb"):
        raise ValueError("only SwiGLU / RMSNorm / no-qk-norm / no-attn-gate / unscaled-embedding models are Llama layers")
    if m.get("residual_func", "base") not in ("base", "add") or m.get("rope_head_dim") != m["head_dim"]:
        raise ValueError("non-standard residual or partial RoPE")
    schedule = execution_schedule(m)

    state: dict[str, torch.Tensor] = {}
    for shard in sorted(args.src.glob("*.safetensors")):
        state.update(load_file(shard))
    dtype = getattr(torch, args.dtype)
    h, kv, hd = m["num_heads"], m["num_kv_heads"], m["head_dim"]
    out = {
        "model.embed_tokens.weight": state["embed.weight"],
        "model.norm.weight": norm_gain(state["output.final_norm.weight"], dtype),
        "lm_head.weight": state["output.output.weight"],
    }
    for j, i in enumerate(schedule):
        src, dst = f"layers.{i}.", f"model.layers.{j}."
        out[dst + "input_layernorm.weight"] = norm_gain(state[src + "attention.norm.weight"], dtype)
        out[dst + "self_attn.q_proj.weight"] = rope_permute(state[src + "attention.wq.weight"], h, hd)
        out[dst + "self_attn.k_proj.weight"] = rope_permute(state[src + "attention.wk.weight"], kv, hd)
        out[dst + "self_attn.v_proj.weight"] = state[src + "attention.wv.weight"]
        out[dst + "self_attn.o_proj.weight"] = state[src + "attention.wo.weight"]
        out[dst + "post_attention_layernorm.weight"] = norm_gain(state[src + "nffn.norm.weight"], dtype)
        out[dst + "mlp.gate_proj.weight"] = state[src + "nffn.fc1.weight"]
        out[dst + "mlp.up_proj.weight"] = state[src + "nffn.fc3.weight"]
        out[dst + "mlp.down_proj.weight"] = state[src + "nffn.fc2.weight"]
    used = {"embed.weight", "output.final_norm.weight", "output.output.weight"} | {
        f"layers.{i}.{name}" for i in set(schedule) for name in (
            "attention.norm.weight", "attention.wq.weight", "attention.wk.weight", "attention.wv.weight",
            "attention.wo.weight", "nffn.norm.weight", "nffn.fc1.weight", "nffn.fc2.weight", "nffn.fc3.weight")}
    if set(state) != used:
        raise ValueError(f"unmapped or missing tensors: {sorted(set(state) ^ used)[:10]}")

    tokenizer_dir = args.src / "tokenizer"
    special = json.loads((tokenizer_dir / "special_tokens_map.json").read_text())
    vocab = json.loads((tokenizer_dir / "tokenizer.json").read_text())["added_tokens"]
    token_id = {t["content"]: t["id"] for t in vocab}
    bos = token_id[special["bos_token"] if isinstance(special["bos_token"], str) else special["bos_token"]["content"]]
    eos = token_id[special["eos_token"] if isinstance(special["eos_token"], str) else special["eos_token"]["content"]]
    hf_config = {
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": m["model_dim"], "intermediate_size": m["ffn_hidden_dim"], "num_hidden_layers": len(schedule),
        "num_attention_heads": h, "num_key_value_heads": kv, "head_dim": hd, "hidden_act": "silu",
        "rms_norm_eps": m["rmsnorm_eps"], "rope_theta": m["rope_base"], "rope_scaling": None,
        "max_position_embeddings": 8192, "vocab_size": m["vocab_size"], "tie_word_embeddings": False,
        "attention_bias": False, "mlp_bias": False, "attention_dropout": 0.0, "initializer_range": 0.02,
        "bos_token_id": bos, "eos_token_id": eos, "torch_dtype": args.dtype, "use_cache": True,
        "ifm_source": {"artifact": str(args.src.name), "execution_schedule": schedule},
    }
    args.dst.mkdir(parents=True, exist_ok=True)
    (args.dst / "config.json").write_text(json.dumps(hf_config, indent=2) + "\n")
    (args.dst / "generation_config.json").write_text(json.dumps({"bos_token_id": bos, "eos_token_id": eos}) + "\n")
    save_file({k: v.to(dtype).contiguous() for k, v in out.items()}, str(args.dst / "model.safetensors"), metadata={"format": "pt"})
    for f in tokenizer_dir.iterdir():
        shutil.copy(f, args.dst / f.name)
    print(f"CONVERTED {args.src.name} -> {args.dst} layers={len(schedule)} distinct={len(set(schedule))} "
          f"params={sum(v.numel() for v in out.values()) / 1e9:.3f}B bos={bos} eos={eos}")


if __name__ == "__main__":
    main()
