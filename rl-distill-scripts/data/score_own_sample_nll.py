#!/usr/bin/env python3
"""Own-sample NLL per token (Monte-Carlo entropy estimate) of a model on its own band-eval samples.

For a samples.parquet written by generate_deepscaler_4bpt_samples.py (temperature 1, top-p 1, no top-k) BY THIS MODEL, re-renders
each prompt with the same chat template, appends the sampled response tokens and teacher-forces them through the same model with
vLLM prompt_logprobs. Mean -log p over response tokens = E_{y~pi}[-log pi(y)] = per-token entropy along the model's own
trajectories, i.e. the estimator behind rollout_corr/training_log_ppl in the RL / OPD logs. Responses are re-tokenized from text,
so a rare non-canonical sampled tokenization is scored canonically (negligible bias). Writes a JSON summary.
  python score_own_sample_nll.py --model DIR --samples samples.parquet --chat-template qwen3_base_native_chat.jinja \
      --per-question 8 --tensor-parallel-size 2 --output nll.json
"""
import argparse, json

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--samples", required=True)
ap.add_argument("--chat-template", required=True)
ap.add_argument("--per-question", type=int, default=8, help="score the first K responses of every question")
ap.add_argument("--tensor-parallel-size", type=int, default=1)
ap.add_argument("--max-model-len", type=int, default=12288)
ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
ap.add_argument("--output", required=True)
a = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

tok = AutoTokenizer.from_pretrained(a.model)
template = open(a.chat_template).read()
df = pd.read_parquet(a.samples)
requests, n_prompt = [], []
for row in df.itertuples():
    rendered = tok.apply_chat_template([{"role": "user", "content": row.prompt_text}], chat_template=template,
                                       add_generation_prompt=True, tokenize=False)
    p_ids = tok.encode(rendered, add_special_tokens=False)
    for text in list(row.response_texts)[: a.per_question]:
        r_ids = tok.encode(text, add_special_tokens=False)
        if not r_ids:
            continue
        ids = (p_ids + r_ids)[: a.max_model_len - 1]
        if len(ids) <= len(p_ids):
            continue
        requests.append(TokensPrompt(prompt_token_ids=ids))
        n_prompt.append(len(p_ids))

llm = LLM(model=a.model, tensor_parallel_size=a.tensor_parallel_size, max_model_len=a.max_model_len,
          gpu_memory_utilization=a.gpu_memory_utilization, enable_prefix_caching=False)
outs = llm.generate(requests, SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0))
seq_nll, seq_len = [], []
for out, n_p in zip(outs, n_prompt):
    ids = out.prompt_token_ids
    lps = [out.prompt_logprobs[i][ids[i]].logprob for i in range(n_p, len(ids))]
    seq_nll.append(-float(np.sum(lps))); seq_len.append(len(lps))
seq_nll, seq_len = np.array(seq_nll), np.array(seq_len)
summary = {"model": a.model, "samples": a.samples, "sequences": int(len(seq_len)), "tokens": int(seq_len.sum()),
           "nll_per_token": float(seq_nll.sum() / seq_len.sum()),                     # token-weighted (= training_log_ppl)
           "nll_per_token_seq_mean": float(np.mean(seq_nll / seq_len)),               # sequence-averaged variant
           "mean_response_tokens": float(seq_len.mean())}
json.dump(summary, open(a.output, "w"), indent=1)
print("OWN_NLL", json.dumps(summary))
