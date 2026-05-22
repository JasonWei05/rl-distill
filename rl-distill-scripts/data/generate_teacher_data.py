#!/usr/bin/env python3
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

"""Generate off-policy distillation dataset: teacher generates responses with per-token log probs.

Data-parallel: run multiple instances with --shard_id / --num_shards, each using --tp GPUs.
Example: 8 GPUs, TP=2 → 4 shards per node. Use launch_teacher_gen.sh to start all shards.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="JWei05/dapo-gemma3-27b-it")
    parser.add_argument("--revision", type=str, default="step_000040")
    parser.add_argument("--input_parquet", type=str, default=None, help="Default: ~/verl/data/dapo-math-17k.parquet")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory for output shards (default: ~/verl/data/teacher_gen)"
    )
    parser.add_argument("--n", type=int, default=4, help="Responses per prompt")
    parser.add_argument("--max_tokens", type=int, default=20480)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size per instance")
    parser.add_argument("--dp", type=int, default=1, help="Data parallel size per instance")
    parser.add_argument("--distributed_executor_backend", type=str, default=None)
    parser.add_argument("--max_model_len", type=int, default=22528)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument(
        "--mm_encoder_attn_backend",
        type=str,
        default=None,
        help="Optional vLLM VLM encoder attention backend, e.g. TORCH_SDPA",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="Optional path to a chat template jinja file to assign to the tokenizer",
    )
    parser.add_argument("--shard_id", type=int, default=0, help="This instance's shard index")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    if args.input_parquet is None:
        args.input_parquet = str(Path.home() / "verl/data/dapo-math-17k.parquet")
    if args.output_dir is None:
        args.output_dir = str(Path.home() / "verl/data/teacher_gen")

    df = pd.read_parquet(args.input_parquet)
    if args.max_samples > 0:
        df = df.head(args.max_samples)
    if "prompt_idx" in df.columns:
        source_prompt_indices = df["prompt_idx"].astype(int).tolist()
    else:
        source_prompt_indices = list(range(len(df)))

    total_prompts = len(df)

    # Shard the prompts across instances
    shard_size = (total_prompts + args.num_shards - 1) // args.num_shards
    start = args.shard_id * shard_size
    end = min(start + shard_size, total_prompts)
    df = df.iloc[start:end]
    print(f"Shard {args.shard_id}/{args.num_shards}: prompts [{start}:{end}] ({len(df)} prompts)")

    # Extract prompt text
    prompts = []
    for prompt_col in df["prompt"]:
        if isinstance(prompt_col, list):
            prompts.append(prompt_col[-1]["content"])
        else:
            prompts.append(str(prompt_col))

    # Treat empty string / "none" / "main" + local-path model as "no explicit revision".
    # vLLM's `revision=` is a git branch/tag/commit — not applicable when the teacher
    # is a pre-downloaded local path (e.g. a subfolder of a repo).
    revision = args.revision
    if not revision or revision.lower() == "none" or os.path.isdir(args.teacher_model):
        revision = None
    llm_kwargs = dict(
        model=args.teacher_model,
        tensor_parallel_size=args.tp,
        data_parallel_size=args.dp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    if args.distributed_executor_backend:
        llm_kwargs["distributed_executor_backend"] = args.distributed_executor_backend
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if revision is not None:
        llm_kwargs["revision"] = revision
    if args.mm_encoder_attn_backend:
        llm_kwargs["mm_encoder_attn_backend"] = args.mm_encoder_attn_backend
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    if args.chat_template:
        tokenizer.chat_template = Path(args.chat_template).read_text()

    # Build chat prompts — repeat each prompt N times for N responses
    chat_prompts = []
    prompt_indices = []
    prompt_token_ids_by_local_idx = []
    for idx, text in enumerate(prompts):
        messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_token_ids_by_local_idx.append(tokenizer.encode(formatted, add_special_tokens=False))
        for _ in range(args.n):
            chat_prompts.append(formatted)
            prompt_indices.append(idx)

    dp_rank = 0
    if args.dp > 1 and args.distributed_executor_backend == "external_launcher":
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dp_rank = global_rank // (world_size // args.dp)
        keep = [i for i in range(len(chat_prompts)) if i % args.dp == dp_rank]
        chat_prompts = [chat_prompts[i] for i in keep]
        prompt_indices = [prompt_indices[i] for i in keep]
        print(f"External DP rank {dp_rank}/{args.dp}: kept {len(chat_prompts)} local requests")

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        # logprobs=1 forces vLLM to compute proper log_softmax values for the
        # sampled token. logprobs=0 takes a fast-path that leaves the logprob
        # field as a 0.0 placeholder for ~40% of tokens (confirmed by audit on
        # the first generation run).
        logprobs=1,
    )

    print(f"Generating {len(chat_prompts)} responses ({len(prompts)} prompts x {args.n})...")
    print(f"Teacher: {args.teacher_model} @ {args.revision}, TP={args.tp}, DP={args.dp}")
    outputs = llm.generate(chat_prompts, sampling_params)

    records = []
    skipped = 0
    for output, local_idx in zip(outputs, prompt_indices, strict=False):
        completion = output.outputs[0]
        if not completion.token_ids or completion.logprobs is None:
            skipped += 1
            continue

        token_ids = list(completion.token_ids)
        log_probs = [lp[tid].logprob for tid, lp in zip(token_ids, completion.logprobs, strict=False)]

        assert len(log_probs) == len(token_ids), (
            f"log_probs length {len(log_probs)} != token_ids length {len(token_ids)}"
        )

        prompt_token_ids = prompt_token_ids_by_local_idx[local_idx]
        full_token_ids = list(prompt_token_ids) + token_ids
        response_mask = [0] * len(prompt_token_ids) + [1] * len(token_ids)

        records.append(
            {
                "messages": [
                    {"role": "user", "content": prompts[local_idx]},
                    {"role": "assistant", "content": completion.text},
                ],
                "input_ids": full_token_ids,
                "response_mask": response_mask,
                "teacher_log_probs": log_probs,
                "teacher_token_ids": token_ids,
                "prompt_idx": source_prompt_indices[start + local_idx],
            }
        )

    print(f"Shard {args.shard_id}: {len(records)} responses, {skipped} skipped")

    if args.dp > 1 and args.distributed_executor_backend == "external_launcher":
        out_name = f"shard_{args.shard_id:03d}_dp{dp_rank:03d}.parquet"
    else:
        out_name = f"shard_{args.shard_id:03d}.parquet"
    out_path = Path(args.output_dir) / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(records)
    out_df.to_parquet(str(out_path), index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
