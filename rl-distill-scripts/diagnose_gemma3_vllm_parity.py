# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Run one reproducible Gemma 3 dense/MoE rollout-parity probe.

Each invocation loads exactly one backend so vLLM can shut down cleanly when
the process exits. Run the following probes on otherwise idle GPUs and compare
the emitted ``PARITY_REPORT=`` records:

  # Native vLLM dense reference.
  CUDA_VISIBLE_DEVICES=0 python diagnose_gemma3_vllm_parity.py \\
      --model dense --backend native

  # The canonical upcycle through the registered native vLLM plugin.
  CUDA_VISIBLE_DEVICES=1 python diagnose_gemma3_vllm_parity.py \\
      --model moe --backend native --moe-model /checkpoints/moe-canonical

  # Direct Transformers reference for the same MoE checkpoint.
  CUDA_VISIBLE_DEVICES=2 python diagnose_gemma3_vllm_parity.py \\
      --model moe --backend hf --moe-model /checkpoints/moe-canonical

The prompt is rendered using the exact DAPO chat template.  The probe is
greedy by design: a difference in token ids then represents a model/backend
correctness difference rather than sampling noise. ``--backend transformers``
is retained only to diagnose the generic vLLM Transformers backend; it is not
the production rollout path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

DEFAULT_DENSE_MODEL = "google/gemma-3-4b-pt"
DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "data" / "gemma3_it_chat_template.jinja"
DEFAULT_PROMPT = "Solve the equation 3x + 7 = 22. Explain your reasoning."
STOP_TOKEN_IDS = [1, 106]  # Gemma EOS and <end_of_turn>.


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("dense", "moe"), required=True)
    parser.add_argument("--backend", choices=("hf", "native", "transformers"), required=True)
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--moe-model", default=None, help="Required when --model=moe.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Tokenizer source used for every model (default: --dense-model). Use the same value for every parity probe."
        ),
    )
    parser.add_argument("--data", default=None, help="Optional parquet containing a DAPO prompt column.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used when --data is omitted.")
    parser.add_argument("--chat-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument(
        "--attention-backend",
        help=(
            "Optional vLLM attention backend override (for example TRITON_ATTN). "
            "Use the same value on both probes when isolating model parity."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None, help="Also write the report as JSON.")
    parser.add_argument(
        "--explicit-stop-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass EOS and <end_of_turn> explicitly (default: true).",
    )
    return parser.parse_args()


def _render_prompt(args: argparse.Namespace) -> tuple[Any, str, list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if args.data is not None:
        import pandas as pd

        row = pd.read_parquet(args.data).iloc[args.prompt_index]
        messages = [dict(message) for message in row["prompt"]]
    else:
        messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=args.chat_template.read_text(),
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    return tokenizer, prompt, prompt_ids


def _distribution_summary(logits: torch.Tensor, tokenizer: Any) -> dict[str, Any]:
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = float(-(probs * log_probs).sum().item())
    values, indices = torch.topk(logits, k=8)
    return {
        "next_token_entropy": entropy,
        "next_token_top8": [
            {
                "id": int(token_id),
                "text": tokenizer.decode([int(token_id)]),
                "logit": float(logit),
            }
            for logit, token_id in zip(values.tolist(), indices.tolist(), strict=True)
        ],
    }


def _token_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()


def _hf_generate(
    model_path: str, tokenizer: Any, prompt_ids: list[int], max_tokens: int
) -> tuple[list[int], str, dict[str, Any]]:
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        .cuda()
        .eval()
    )
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        next_logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True).logits[0, -1]
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_tokens,
            eos_token_id=STOP_TOKEN_IDS,
            pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else STOP_TOKEN_IDS[0]),
            use_cache=True,
        )
    token_ids = generated[0, input_ids.shape[1] :].tolist()
    finish_reason = "stop" if token_ids and token_ids[-1] in STOP_TOKEN_IDS else "length"
    return token_ids, finish_reason, _distribution_summary(next_logits, tokenizer)


def _vllm_generate(
    model_path: str,
    tokenizer: Any,
    prompt_ids: list[int],
    args: argparse.Namespace,
) -> tuple[list[int], str, dict[str, Any]]:
    from vllm import LLM, SamplingParams

    model_kwargs: dict[str, Any] = {
        "model": model_path,
        "tokenizer": args.tokenizer,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "enforce_eager": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": len(prompt_ids) + args.max_tokens + 32,
        "disable_log_stats": True,
    }
    model_kwargs["model_impl"] = args.backend
    if args.attention_backend is not None:
        model_kwargs["attention_backend"] = args.attention_backend
    llm = LLM(**model_kwargs)
    sampling_kwargs: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "ignore_eos": False,
        "detokenize": True,
    }
    if args.explicit_stop_ids:
        sampling_kwargs["stop_token_ids"] = STOP_TOKEN_IDS
    # verl sends TokensPrompt(prompt_token_ids=...) to the server.  Do the same
    # here: passing the rendered string lets vLLM apply its own special-token
    # policy and can produce a different sequence before the model runs.
    output = llm.generate([{"prompt_token_ids": prompt_ids}], SamplingParams(**sampling_kwargs))[0].outputs[0]
    token_ids = list(output.token_ids)
    summary = {
        "next_token_entropy": None,
        "next_token_top8": None,
        "prompt_token_count": len(prompt_ids),
    }
    return token_ids, str(output.finish_reason), summary


def main() -> None:
    args = _parse_args()
    if args.model == "moe" and args.moe_model is None:
        raise ValueError("--moe-model is required when --model=moe")
    if args.backend == "hf" and args.model == "dense":
        # The official dense checkpoint is a multimodal Gemma3 container;
        # direct HF text-only parity is intentionally scoped to the MoE model.
        raise ValueError("Use --backend=native for the dense reference")
    args.tokenizer = args.tokenizer or args.dense_model
    model_path = args.dense_model if args.model == "dense" else args.moe_model
    tokenizer, _, prompt_ids = _render_prompt(args)
    if args.backend == "hf":
        token_ids, finish_reason, distribution = _hf_generate(model_path, tokenizer, prompt_ids, args.max_tokens)
    else:
        token_ids, finish_reason, distribution = _vllm_generate(model_path, tokenizer, prompt_ids, args)

    report = {
        "model": args.model,
        "backend": args.backend,
        "model_path": model_path,
        "tokenizer_path": args.tokenizer,
        "prompt_index": args.prompt_index,
        "prompt_token_count": len(prompt_ids),
        "prompt_token_sha256": _token_hash(prompt_ids),
        "explicit_stop_ids": args.explicit_stop_ids,
        "stop_token_ids": STOP_TOKEN_IDS,
        "max_tokens": args.max_tokens,
        "finish_reason": finish_reason,
        "output_token_count": len(token_ids),
        "output_token_sha256": _token_hash(token_ids),
        "output_prefix_sha256": {
            str(length): _token_hash(token_ids[:length])
            for length in (1, 8, 32, 128, 512, 1024, 2048)
            if len(token_ids) >= length
        },
        "output_token_ids_head": token_ids[:128],
        "output_token_ids_tail": token_ids[-128:],
        "output_ends_with_stop": bool(token_ids and token_ids[-1] in STOP_TOKEN_IDS),
        "output_head": tokenizer.decode(token_ids[:96]),
        "output_tail": tokenizer.decode(token_ids[-96:]),
        **distribution,
    }
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n")
    print("PARITY_REPORT=" + serialized)


if __name__ == "__main__":
    main()
