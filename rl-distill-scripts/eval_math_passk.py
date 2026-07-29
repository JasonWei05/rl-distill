#!/usr/bin/env python3
"""Sampled math eval -> mean@k / pass@k / maj@k over the RL-val repeat datasets.

For each dataset: 1 sample per parquet row (rows are the k repeats of a question, sharing `uid`),
same few-shot chat prompt + sampling as RL training (single BOS via add_special_tokens=False),
scored with the repo's math_verify. Grouped by uid:
  mean@k = mean accuracy over the k samples
  pass@k = 1 if ANY of the k samples is correct
  maj@k  = 1 if the majority-vote prediction (mode of \\boxed{} answers) is correct
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams

sys.path.insert(0, "/mnt/efs/jasonwei/rl-distill")
from verl.utils.reward_score import math_verify as mv


def qtext(prompt_col):
    if hasattr(prompt_col, "__len__") and len(prompt_col) and isinstance(prompt_col[-1], dict):
        return prompt_col[-1]["content"]
    return str(prompt_col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)               # rl_step350 | base
    ap.add_argument("--chat_template", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)  # parquet paths
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=20480)
    ap.add_argument("--max_model_len", type=int, default=24576)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--enforce_eager", action="store_true", help="skip torch.compile (avoids inductor cache issues)")
    ap.add_argument("--trace_dir", default=None, help="if set, write per-sample eval traces (jsonl per dataset)")
    args = ap.parse_args()
    if args.trace_dir:
        Path(args.trace_dir).mkdir(parents=True, exist_ok=True)

    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True,
              enforce_eager=args.enforce_eager)
    tok = llm.get_tokenizer()
    # Pass the template explicitly below (not just via the attribute): multimodal tokenizers/processors
    # (e.g. Gemma 4) don't honor a set .chat_template attribute, raising "chat_template is not set".
    chat_template_str = Path(args.chat_template).read_text()
    tok.chat_template = chat_template_str
    # Stop at the gemma turn terminator: gemma-3 auto-stops on <end_of_turn> (EOS), but gemma-4 emits
    # it as text and keeps rambling (into the next few-shot turn) unless we stop explicitly. Harmless
    # for gemma-3 (it already stops there). Keeps the response to one clean boxed answer for strict grading.
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        max_tokens=args.max_tokens, stop=["<end_of_turn>", "<start_of_turn>"])

    results = {}
    for ds in args.datasets:
        name = Path(ds).stem
        df = pd.read_parquet(ds)
        reqs, golds, uids, qtexts = [], [], [], []
        for _, row in df.iterrows():
            q = qtext(row["prompt"])
            ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": q}],
                             chat_template=chat_template_str, add_generation_prompt=True, tokenize=False),
                             add_special_tokens=False)
            reqs.append({"prompt_token_ids": ids})
            golds.append(row["reward_model"]["ground_truth"])
            uids.append(row["uid"] if "uid" in df.columns and row.get("uid") is not None else f"{name}-{_}")
            qtexts.append(q)
        print(f"[{args.tag}] {name}: generating {len(reqs)} samples ...", flush=True)
        outs = llm.generate(reqs, sp)

        by_uid = defaultdict(list)   # uid -> list[(acc_bool, pred_str)]
        traces = []
        for out, gold, uid, q in zip(outs, golds, uids, qtexts):
            text = out.outputs[0].text
            score = mv.compute_score(text, gold)
            acc = score > 0.5
            pred = mv.extract_prediction(text)
            by_uid[uid].append((bool(acc), pred))
            if args.trace_dir:
                traces.append({"dataset": name, "uid": uid, "gold": gold, "prompt_text": q,
                               "response_text": text, "response_token_ids": list(out.outputs[0].token_ids),
                               "score": float(score), "acc": bool(acc), "pred": pred})
        if args.trace_dir:
            tp = Path(args.trace_dir) / f"{args.tag}__{name}.jsonl"
            with tp.open("w") as f:
                for t in traces:
                    f.write(json.dumps(t) + "\n")
            print(f"[{args.tag}] {name}: wrote {len(traces)} traces -> {tp}", flush=True)

        means, passes, majs = [], [], []
        for lst in by_uid.values():
            accs = [a for a, _ in lst]
            means.append(sum(accs) / len(accs))
            passes.append(1.0 if any(accs) else 0.0)
            mode = Counter(p if p else "<none>" for _, p in lst).most_common(1)[0][0]
            maj_acc = next((a for a, p in lst if (p if p else "<none>") == mode), False)
            majs.append(1.0 if maj_acc else 0.0)
        k = int(round(sum(len(v) for v in by_uid.values()) / max(1, len(by_uid))))
        results[name] = {"k": k, "n_questions": len(by_uid),
                         "mean@k": round(100 * sum(means) / len(means), 2),
                         "pass@k": round(100 * sum(passes) / len(passes), 2),
                         "maj@k": round(100 * sum(majs) / len(majs), 2)}
        print(f"[{args.tag}] {name}: {results[name]}", flush=True)

    Path(args.out).write_text(json.dumps({"tag": args.tag, "model": args.model, "results": results}, indent=2))
    print(f"[{args.tag}] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
