#!/usr/bin/env python3
"""Math evals for tomg-group-umd/huginn-0125 (recurrent-depth "Raven" model) on the DeepScaleR band sets.

Huginn has no vLLM support, so this drives the HF remote code (patched copy, see --model) with batched
``generate`` at a fixed ``num_steps`` (recurrence count). Prompts use Huginn's chat template and the math system
prompt from its paper (Fig. 14); ``--prompt boxinstr`` additionally asks for a ``\\boxed{}`` answer. Questions are
sharded across processes (``--shard i/N``); each finished batch is appended to the shard's JSONL immediately so
``--grade`` can score partial results. Grading is the repo's ``math_verify`` both boxed-only (= RL reward, our
protocol) and lenient whole-output extraction (``VERL_MATH_VERIFY_STRICT_BOXED=0``).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = "You are a helpful assistant that is capable of helping users with mathematical reasoning."
BOX_INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
EOS_ID, BOS_ID, PAD_ID = 65505, 65504, 65509
END_TURN = "<|end_turn|>"


def load_questions(dataset: Path) -> list[dict]:
    import pandas as pd

    rows = []
    for row in pd.read_parquet(dataset).to_dict(orient="records"):
        rows.append({"uid": str(row["uid"]), "question": row["prompt"][0]["content"], "gold": row["reward_model"]["ground_truth"]})
    return rows


def generate(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    shard, num_shards = (int(x) for x in args.shard.split("/"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    sample_kwargs = (
        {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p, "top_k": 0}
        if args.samples_per_question > 1 or args.temperature > 0
        else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    )
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens, use_cache=True, return_dict_in_generate=True,
        eos_token_id=EOS_ID, bos_token_id=BOS_ID, pad_token_id=PAD_ID, stop_strings=[END_TURN, "<|end_text|>"],
        **sample_kwargs,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        name = Path(dataset).stem
        out = out_dir / f"{name}.shard{shard}of{num_shards}.jsonl"
        done = {(r["uid"], r["sample_index"]) for r in map(json.loads, out.open())} if out.exists() else set()
        requests = []
        for question in load_questions(Path(dataset))[shard::num_shards]:
            user = question["question"] + (BOX_INSTRUCTION if args.prompt == "boxinstr" else "")
            text = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
            for sample_index in range(args.samples_per_question):
                if (question["uid"], sample_index) not in done:
                    requests.append({**question, "sample_index": sample_index, "text": text})
        requests.sort(key=lambda r: len(r["text"]))  # similar lengths per batch -> less padding
        print(f"[{name} shard {shard}/{num_shards}] {len(requests)} requests (skipped {len(done)} done)", flush=True)
        with out.open("a") as handle:
            for start in range(0, len(requests), args.batch_size):
                batch = requests[start : start + args.batch_size]
                enc = tokenizer([r["text"] for r in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                torch.manual_seed(args.seed + 1_000_003 * shard + start)
                t0 = time.time()
                result = model.generate(enc["input_ids"], config, tokenizer=tokenizer, num_steps=args.num_steps)
                new = result.sequences[:, enc["input_ids"].shape[1] :]
                for request, ids in zip(batch, new.tolist()):
                    ids = [t for t in ids if t != PAD_ID]
                    ended = bool(ids) and (ids[-1] == EOS_ID or tokenizer.decode(ids).rstrip().endswith((END_TURN, "<|end_text|>")))
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    handle.write(json.dumps({
                        "uid": request["uid"], "sample_index": request["sample_index"], "gold": request["gold"],
                        "response_text": text, "response_length": len(ids), "truncated": not ended,
                    }) + "\n")
                handle.flush()
                print(f"[{name} shard {shard}/{num_shards}] {start + len(batch)}/{len(requests)} batch_time={time.time() - t0:.0f}s "
                      f"max_len={max(len([t for t in s if t != PAD_ID]) for s in new.tolist())}", flush=True)


def grade(args: argparse.Namespace) -> None:
    """Score every finished sample under out_dir (cached per file) and print mean@k / pass@k per dataset."""
    sys.path.insert(0, str(REPO))
    from verl.utils.reward_score import math_verify

    summary = {}
    for name in sorted({Path(p).name.split(".shard")[0] for p in glob.glob(f"{args.out_dir}/*.shard*of*.jsonl")}):
        rows = []
        for path in sorted(glob.glob(f"{args.out_dir}/{name}.shard*of*.jsonl")):
            cache_path = Path(path).with_suffix(".graded.jsonl")
            cached = [json.loads(line) for line in cache_path.open()] if cache_path.exists() else []
            with open(path) as src, cache_path.open("a") as cache:
                for index, line in enumerate(src):
                    if index < len(cached) or not line.endswith("\n"):
                        continue
                    sample = json.loads(line)
                    os.environ["VERL_MATH_VERIFY_STRICT_BOXED"] = "1"
                    strict = float(math_verify.compute_score(sample["response_text"], sample["gold"])) > 0.5
                    os.environ["VERL_MATH_VERIFY_STRICT_BOXED"] = "0"
                    lenient = float(math_verify.compute_score(sample["response_text"], sample["gold"])) > 0.5
                    row = {"uid": sample["uid"], "sample_index": sample["sample_index"], "strict": strict, "lenient": lenient,
                           "boxed": "\\boxed" in sample["response_text"], "response_length": sample["response_length"],
                           "truncated": sample["truncated"]}
                    cache.write(json.dumps(row) + "\n")
                    cached.append(row)
            rows += cached
        by_question: dict[str, list[dict]] = {}
        for row in rows:
            by_question.setdefault(row["uid"], []).append(row)
        complete = [samples for samples in by_question.values() if len(samples) == args.samples_per_question]
        entry = {"samples": len(rows), "complete_questions": len(complete)}
        for key in ("strict", "lenient"):
            entry[f"mean_{key}"] = round(100 * sum(r[key] for r in rows) / max(1, len(rows)), 1)
            entry[f"pass@{args.samples_per_question}_{key}"] = (
                round(100 * sum(any(r[key] for r in s) for s in complete) / len(complete), 1) if complete else None
            )
        entry["boxed_rate"] = round(100 * sum(r["boxed"] for r in rows) / max(1, len(rows)), 1)
        entry["mean_len"] = round(sum(r["response_length"] for r in rows) / max(1, len(rows)))
        entry["truncated_rate"] = round(100 * sum(r["truncated"] for r in rows) / max(1, len(rows)), 1)
        summary[name] = entry
    print(json.dumps(summary, indent=1))
    Path(args.out_dir, "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/tmp/huginn_eval/huginn-0125-patched")
    parser.add_argument("--datasets", nargs="+", default=[f"/tmp/ouro_eval/data/id_{b}.parquet" for b in ("easy", "medium", "hard")])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--prompt", choices=("bare", "boxinstr"), default="bare")
    parser.add_argument("--samples_per_question", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0, help="0 with 1 sample = greedy")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grade", action="store_true", help="grade finished samples in --out_dir and exit")
    args = parser.parse_args()
    grade(args) if args.grade else generate(args)


if __name__ == "__main__":
    main()
