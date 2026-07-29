#!/usr/bin/env python3
"""Generate N samples per DeepScaleR question with Gemma-3-4B-IT (data-parallel by sharding).

Each shard handles rows where (index % num_shards == shard_id), runs on ONE GPU (TP=1). Feeds vLLM
token-id prompts built from the short 2-shot chat template (single BOS, add_special_tokens=False, same
as eval_math_passk). Sampling: temp 1.0 / top_p 1.0 / top_k -1, n samples, max_tokens 8192. Saves a
parquet per shard with, per question, the list of response texts, per-sample token lengths, and
math_verify scores/acc — so questions can later be bucketed by response length (and/or correctness).
"""
import argparse, sys
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams

sys.path.insert(0, "/mnt/efs/jasonwei/rl-distill")
from verl.utils.reward_score import math_verify as mv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--input", required=True)
    ap.add_argument("--chat_template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=6144)          # 6*1024 response
    ap.add_argument("--max_model_len", type=int, default=8192)       # 8k total context
    ap.add_argument("--max_prompt_len", type=int, default=2048)      # cap prompt (2-shot + question)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = ap.parse_args()

    df = pd.read_parquet(args.input).reset_index(drop=True)
    df = df[df.index % args.num_shards == args.shard_id].reset_index(drop=True)
    print(f"[shard {args.shard_id}/{args.num_shards}] {len(df)} questions x{args.n} samples", flush=True)

    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
    tok = llm.get_tokenizer()
    tok.chat_template = Path(args.chat_template).read_text()
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, max_tokens=args.max_tokens)

    reqs, golds, uids, probs = [], [], [], []
    for _, row in df.iterrows():
        q = row["prompt"][-1]["content"]
        ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": q}],
                         add_generation_prompt=True, tokenize=False), add_special_tokens=False)
        if len(ids) > args.max_prompt_len:
            ids = ids[-args.max_prompt_len:]   # left-truncate: keep the question + generation prompt
        reqs.append({"prompt_token_ids": ids})
        golds.append(row["reward_model"]["ground_truth"])
        uids.append(row["uid"])
        probs.append(row.get("problem", q))

    outs = llm.generate(reqs, sp)

    records = []
    for out, gold, uid, prob in zip(outs, golds, uids, probs):
        texts, lens, scores, accs = [], [], [], []
        for comp in out.outputs:                       # n completions
            t = comp.text
            s = float(mv.compute_score(t, gold))
            texts.append(t)
            lens.append(len(comp.token_ids))
            scores.append(s)
            accs.append(bool(s > 0.5))
        records.append({
            "uid": uid, "problem": prob, "answer": gold,
            "response_texts": texts, "response_lens": lens,
            "scores": scores, "accs": accs,
            "mean_len": sum(lens) / len(lens), "max_len": max(lens),
            "n_correct": int(sum(accs)), "any_correct": bool(any(accs)),
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(args.out, index=False)
    print(f"[shard {args.shard_id}] wrote {args.out} ({len(records)} rows)", flush=True)


if __name__ == "__main__":
    main()
