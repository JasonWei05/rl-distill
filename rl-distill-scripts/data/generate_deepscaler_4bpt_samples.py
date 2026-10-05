#!/usr/bin/env python3
"""Generate multiple Gemma PT responses for deduplicated DeepScaleR.

The script is deliberately generation-only.  It checkpoints each chunk to a
separate parquet so a long local run can resume without discarding completed
GPU work.  Correctness grading is performed afterward by
``classify_deepscaler_samples.py`` using both historical DeepScaleR contracts:

* lenient: math-verify may extract a correct answer anywhere in the response;
* strict: exactly one well-formed ``\\boxed{}``/``\\fbox{}`` is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


BOXED_INSTRUCTION = "Please output the final answer within \\boxed{}."
DEFAULT_STOP_STRINGS = ("<end_of_turn>", "<start_of_turn>")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(temporary, index=False)
    temporary.replace(path)


def normalize_input(path: Path, append_boxed_instruction: bool, instruction: str = BOXED_INSTRUCTION) -> pd.DataFrame:
    df = pd.read_parquet(path).reset_index(drop=True)
    if {"problem", "answer"}.issubset(df.columns):
        out = df[["problem", "answer"]].copy()
        out["uid"] = df["uid"] if "uid" in df.columns else df.index.astype(str)
    elif {"prompt", "reward_model"}.issubset(df.columns):
        out = pd.DataFrame(
            {
                "uid": df["uid"] if "uid" in df.columns else df.index.astype(str),
                "problem": df["prompt"].map(lambda messages: messages[-1]["content"]),
                "answer": df["reward_model"].map(lambda reward: reward["ground_truth"]),
            }
        )
    else:
        raise ValueError(
            f"Unsupported input columns in {path}: expected problem/answer or prompt/reward_model, "
            f"got {sorted(df.columns)}"
        )

    out.insert(0, "source_index", range(len(out)))
    out["problem"] = out["problem"].astype(str).str.strip()
    out["answer"] = out["answer"].astype(str).str.strip()
    if append_boxed_instruction:
        out["prompt_text"] = out["problem"].map(
            lambda problem: problem
            if problem.rstrip().endswith(instruction)
            else f"{problem} {instruction}"
        )
    else:
        out["prompt_text"] = out["problem"]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--chat-template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument(
        "--stop-strings",
        default=",".join(DEFAULT_STOP_STRINGS),
        help="Comma-separated generation stop strings.",
    )
    parser.add_argument(
        "--stop-token-ids",
        default="",
        help="Comma-separated stop token ids (special tokens such as ChatML <|im_end|> never match stop strings).",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-rank", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--disable-allreduce-rms-fusion", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--boxed-instruction",
        default=BOXED_INSTRUCTION,
        help="Instruction appended to each problem when --append-boxed-instruction is on.",
    )
    parser.add_argument(
        "--append-boxed-instruction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0 or args.chunk_size <= 0:
        raise ValueError("--n and --chunk-size must be positive")
    if args.tensor_parallel_size <= 0 or args.data_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size and --data-parallel-size must be positive")
    if not 0 <= args.data_parallel_rank < args.data_parallel_size:
        raise ValueError("--data-parallel-rank must be in [0, --data-parallel-size)")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must be in [0, --num-shards)")
    stop_strings = [value for value in args.stop_strings.split(",") if value]
    stop_token_ids = [int(value) for value in args.stop_token_ids.split(",") if value]
    if not stop_strings:
        raise ValueError("--stop-strings must contain at least one non-empty value")

    input_path = Path(args.input).resolve()
    template_path = Path(args.chat_template).resolve()
    output_path = Path(args.output).resolve()
    part_dir = output_path.parent / f"{output_path.name}.parts"

    full = normalize_input(input_path, args.append_boxed_instruction, args.boxed_instruction)
    shard = full[full["source_index"] % args.num_shards == args.shard_id].reset_index(drop=True)
    if args.max_questions > 0:
        shard = shard.iloc[: args.max_questions].copy()
    print(
        f"SHARD_START shard={args.shard_id}/{args.num_shards} questions={len(shard)} n={args.n}",
        flush=True,
    )

    manifest = {
        "model": str(Path(args.model).resolve()) if Path(args.model).exists() else args.model,
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "chat_template": str(template_path),
        "chat_template_sha256": sha256_file(template_path),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "questions": len(shard),
        "n": args.n,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "max_prompt_tokens": args.max_prompt_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "stop": stop_strings,
        **({"stop_token_ids": stop_token_ids} if stop_token_ids else {}),
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "attention_backend": args.attention_backend,
        "chunk_size": args.chunk_size,
        "seed": args.seed + args.shard_id,
        "append_boxed_instruction": args.append_boxed_instruction,
        **({"boxed_instruction": args.boxed_instruction} if args.boxed_instruction != BOXED_INSTRUCTION else {}),
    }
    if (
        args.tensor_parallel_size != 1
        or args.data_parallel_size != 1
        or args.data_parallel_rank != 0
    ):
        manifest.update(
            {
                "tensor_parallel_size": args.tensor_parallel_size,
                "data_parallel_size": args.data_parallel_size,
                "data_parallel_rank": args.data_parallel_rank,
            }
        )
    if args.disable_allreduce_rms_fusion:
        manifest["disable_allreduce_rms_fusion"] = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Raising the length caps cannot change any prompt that already fit, so finished parts stay valid:
        # accept increases of max_prompt_tokens / max_model_len and record the new values.
        raisable = ("max_prompt_tokens", "max_model_len")
        if (
            existing_manifest != manifest
            and {k: v for k, v in existing_manifest.items() if k not in raisable}
            == {k: v for k, v in manifest.items() if k not in raisable}
            and all(manifest[k] >= existing_manifest[k] for k in raisable)
        ):
            print(f"MANIFEST_UPDATED shard={args.shard_id} raised caps "
                  f"{ {k: (existing_manifest[k], manifest[k]) for k in raisable} }", flush=True)
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            existing_manifest = manifest
        if existing_manifest != manifest:
            raise RuntimeError(
                f"Existing generation manifest does not match this run; refusing to reuse parts: "
                f"{manifest_path}"
            )
        print(f"MANIFEST_REUSED shard={args.shard_id} path={manifest_path}", flush=True)
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    llm = None
    tokenizer = None
    sampling = None
    chat_template = None
    expected_parts: list[Path] = []
    for start in range(0, len(shard), args.chunk_size):
        end = min(start + args.chunk_size, len(shard))
        chunk = shard.iloc[start:end]
        part_path = part_dir / f"part_{start:06d}_{end:06d}.parquet"
        expected_parts.append(part_path)
        if part_path.exists():
            existing = pd.read_parquet(
                part_path,
                columns=["source_index", "prompt_length", "response_texts", "response_lens"],
            )
            expected_indices = chunk["source_index"].astype(int).tolist()
            observed_indices = existing["source_index"].astype(int).tolist()
            valid_response_lengths = existing["response_lens"].map(
                lambda values: all(0 <= int(value) <= args.max_tokens for value in values)
            )
            if (
                len(existing) != end - start
                or observed_indices != expected_indices
                or existing["source_index"].duplicated().any()
                or not existing["response_texts"].map(len).eq(args.n).all()
                or not existing["response_lens"].map(len).eq(args.n).all()
                or not valid_response_lengths.all()
                or not existing["prompt_length"].astype(int).between(1, args.max_prompt_tokens).all()
            ):
                raise RuntimeError(f"Invalid existing checkpoint part: {part_path}")
            print(f"CHUNK_REUSED shard={args.shard_id} start={start} end={end}", flush=True)
            continue

        if llm is None:
            from vllm import LLM, SamplingParams

            compilation_config = None
            if args.disable_allreduce_rms_fusion:
                compilation_config = {"pass_config": {"fuse_allreduce_rms": False}}
            llm = LLM(
                model=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                dtype="bfloat16",
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_seqs=args.max_num_seqs,
                max_num_batched_tokens=args.max_num_batched_tokens,
                attention_backend=args.attention_backend,
                compilation_config=compilation_config,
                enable_prefix_caching=True,
                enforce_eager=args.enforce_eager,
                seed=args.seed + args.shard_id,
                trust_remote_code=True,
            )
            tokenizer = llm.get_tokenizer()
            chat_template = template_path.read_text()
            sampling = SamplingParams(
                n=args.n,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
                seed=args.seed + args.shard_id,
                stop=stop_strings,
                stop_token_ids=stop_token_ids or None,
            )

        assert tokenizer is not None and chat_template is not None and sampling is not None
        requests = []
        prompt_lengths = []
        for prompt_text in chunk["prompt_text"]:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                chat_template=chat_template,
                add_generation_prompt=True,
                tokenize=False,
            )
            token_ids = tokenizer.encode(rendered, add_special_tokens=False)
            if len(token_ids) > args.max_prompt_tokens:
                raise ValueError(
                    f"Prompt length {len(token_ids)} exceeds --max-prompt-tokens={args.max_prompt_tokens}; "
                    "refusing to truncate the few-shot prompt"
                )
            requests.append({"prompt_token_ids": token_ids})
            prompt_lengths.append(len(token_ids))

        outputs = llm.generate(requests, sampling, use_tqdm=True)
        records = []
        for row, prompt_length, output in zip(chunk.itertuples(index=False), prompt_lengths, outputs):
            if len(output.outputs) != args.n:
                raise RuntimeError(
                    f"source_index={row.source_index}: expected {args.n} outputs, got {len(output.outputs)}"
                )
            texts = [completion.text for completion in output.outputs]
            lengths = [len(completion.token_ids) for completion in output.outputs]
            records.append(
                {
                    "source_index": int(row.source_index),
                    "uid": row.uid,
                    "problem": row.problem,
                    "answer": row.answer,
                    "prompt_text": row.prompt_text,
                    "prompt_length": prompt_length,
                    "response_texts": texts,
                    "response_lens": lengths,
                    "mean_len": sum(lengths) / len(lengths),
                    "max_len": max(lengths),
                }
            )
        atomic_parquet(pd.DataFrame(records), part_path)
        print(
            f"CHUNK_COMPLETE shard={args.shard_id} start={start} end={end} "
            f"mean_response_tokens={sum(sum(r['response_lens']) for r in records) / ((end-start)*args.n):.1f}",
            flush=True,
        )

    merged = pd.concat([pd.read_parquet(path) for path in expected_parts], ignore_index=True)
    merged = merged.sort_values("source_index").reset_index(drop=True)
    expected_indices = sorted(shard["source_index"].astype(int).tolist())
    observed_indices = merged["source_index"].astype(int).tolist()
    if (
        len(merged) != len(shard)
        or observed_indices != expected_indices
        or merged["source_index"].duplicated().any()
    ):
        raise RuntimeError("Shard coverage validation failed")
    if not merged["response_texts"].map(len).eq(args.n).all():
        raise RuntimeError("At least one row does not contain the requested number of responses")
    if not merged["response_lens"].map(len).eq(args.n).all():
        raise RuntimeError("At least one row does not contain the requested number of response lengths")
    if not merged["response_lens"].map(
        lambda values: all(0 <= int(value) <= args.max_tokens for value in values)
    ).all():
        raise RuntimeError("At least one response length is outside the requested token bounds")
    if not merged["prompt_length"].astype(int).between(1, args.max_prompt_tokens).all():
        raise RuntimeError("At least one prompt length is outside the requested token bounds")
    atomic_parquet(merged, output_path)
    print(
        f"SHARD_COMPLETE shard={args.shard_id} rows={len(merged)} output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
