#!/usr/bin/env python3
"""Parity gate: IFM xLLM native forward vs the converted HF Llama checkpoint (convert_ifm_dense_to_llama.py).

Loads the xLLM artifact exactly like `xllm.paper_part1.native_inference.load_native_model` (strict state-dict load,
plain non-fused modules, bf16) except that attention uses xLLM's pure-PyTorch backend (`causal_attn_backend=None`)
instead of FlashAttention 3. Tokenizes each probe with the xLLM tokenizer (BOS prepended, as native inference does),
checks the HF tokenizer gives identical ids, and compares next-token log-probs. The decisive check is --dtype float32
(both sides fp32; convert with --dtype float32): IFM_LLAMA_PARITY_OK needs >= 99.9% argmax agreement and mean
|delta logprob| < 1e-3. In bf16 the two stacks differ by kernel rounding, so the bar is 98% / 0.05.

    PYTHONPATH=<xllm-loop>:<deps>:<xllm_ext_build> python check_ifm_llama_parity.py --src <artifact> --hf <converted>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

PROBES = [
    "Problem: Find the sum of all positive divisors of 36.\nSolution: The divisors of 36 are 1, 2, 3, 4, 6, 9, 12, 18, "
    "and 36. Their sum is 91. The final answer is $\\boxed{91}$.",
    "Let $f(x) = 3x^2 - 5x + 2$. Compute $f(4)$. We have $f(4) = 3\\cdot 16 - 20 + 2 = 30$.",
    "The mitochondria is the powerhouse of the cell. In 1905, Albert Einstein published four papers that changed physics.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
]


def load_native(src: Path, dtype: torch.dtype):
    from xllm.config import ModelConf, TokenizerConf
    from xllm.data.dataset_streamer.tokenizer import build_tokenizer
    from xllm.models.build import get_model_cls
    from xllm.paper_part1 import native_inference as ni
    from xllm.paper_part1.artifacts import load_artifact

    ni.configure_native_runtime()
    config, state_dict = load_artifact(src)
    tokenizer = build_tokenizer(TokenizerConf.from_dict(config["tokenizer"]))
    model_config = ModelConf.from_dict(config["model"])
    model_config.init_mode = "none"
    model_config.fused_block = False
    model_config.fused_output_layer = False
    model_config.causal_attn_backend = None  # pure PyTorch attention instead of FlashAttention 3
    with ni._construction_defaults(torch.device("cuda"), dtype):
        model = get_model_cls(model_config.arch, model_config)(model_config, tokenizer)
    model.load_state_dict({k: v.to(dtype) for k, v in state_dict.items()}, strict=True)
    return model.eval(), tokenizer


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16",
                        help="both sides run in this dtype; convert with the same --dtype")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    native, xtok = load_native(args.src, dtype)
    hf = AutoModelForCausalLM.from_pretrained(args.hf, dtype=dtype, attn_implementation="eager").cuda().eval()
    hf_tok = AutoTokenizer.from_pretrained(args.hf)
    agree = total = 0
    abs_diffs, nll_native, nll_hf = [], [], []
    for text in PROBES:
        ids = xtok.encode(text, bos=True, eos=False)
        hf_ids = [hf.config.bos_token_id] + hf_tok(text, add_special_tokens=False)["input_ids"]
        if ids != hf_ids:
            raise SystemExit(f"TOKENIZER_MISMATCH on {text[:40]!r}: {ids[:12]} vs {hf_ids[:12]}")
        x = torch.tensor([ids], device="cuda")
        native_logits = native(x, multi_segments=False, cache=None)[0].float()
        hf_logits = hf(x).logits.float()
        lp_n, lp_h = native_logits.log_softmax(-1)[0, :-1], hf_logits.log_softmax(-1)[0, :-1]
        target = x[0, 1:]
        ref_n = lp_n.gather(-1, target[:, None])[:, 0]
        ref_h = lp_h.gather(-1, target[:, None])[:, 0]
        abs_diffs.append((ref_n - ref_h).abs())
        nll_native.append(-ref_n.mean().item())
        nll_hf.append(-ref_h.mean().item())
        agree += (lp_n.argmax(-1) == lp_h.argmax(-1)).sum().item()
        total += target.numel()
    diffs = torch.cat(abs_diffs)
    agreement, mean_diff = agree / total, diffs.mean().item()
    print(f"tokens={total} argmax_agreement={agreement:.4f} mean|dlogp|={mean_diff:.4f} max|dlogp|={diffs.max().item():.4f} "
          f"nll_native={sum(nll_native) / len(nll_native):.4f} nll_hf={sum(nll_hf) / len(nll_hf):.4f}")
    tol = 0.05 if dtype == torch.bfloat16 else 1e-3
    if agreement >= (0.98 if dtype == torch.bfloat16 else 0.999) and mean_diff < tol:
        print("IFM_LLAMA_PARITY_OK")
    else:
        raise SystemExit("IFM_LLAMA_PARITY_FAILED")


if __name__ == "__main__":
    main()
