#!/usr/bin/env python3
"""Reverse KL of a student vs a teacher on the student's own samples, using the student's top-k token distribution.

Two phases (separate processes, so vLLM and the HF scoring model never share a GPU):

  generate: sample the *student* with vLLM on N questions per split (medium train / medium validation), the study's
            sampler (temp 1, top-p 1, top-k -1) and 12-shot prompt, recording per response position the sampled token's
            logprob and the student's top-k (token id, logprob) pairs.
  score:    run the *teacher* (HF Transformers, bf16, SDPA, final logit soft-capping applied) over the same token sequences
            and gather the teacher's log-probabilities of (a) the sampled token and (b) the student's top-k tokens.

Per response token this yields
  rkl_mc       = log p_s(x_t) - log p_t(x_t)            unbiased Monte-Carlo estimate of the full-vocab KL(student || teacher)
  rkl_topk     = sum_{i in topk_s} p_s(i) [log p_s(i) - log p_t(i)]      truncated to the student's top-k (training convention)
  rkl_topk_rn  = same with p_s renormalised over the top-k
plus the student's top-k mass. Aggregates are token means per split with a standard error from per-sequence means.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from eval_math_passk import _render_prompt, derive_sampling_seed, prepare_eval_questions  # noqa: E402


def select_questions(parquet: Path, count: int, seed: int, name: str):
    rows = pd.read_parquet(parquet).to_dict(orient="records")
    questions = prepare_eval_questions(rows, dataset_name=name)   # deduped, stable order
    if count < len(questions):
        questions = random.Random(seed).sample(questions, count)
        questions.sort(key=lambda q: q.question_id)
    return questions


def generate(args) -> None:
    from vllm import LLM, SamplingParams

    chat_template = Path(args.chat_template).read_text()
    llm = LLM(model=args.student, max_model_len=args.max_model_len, max_logprobs=args.topk, gpu_memory_utilization=args.gpu_memory_utilization,
              dtype="bfloat16", trust_remote_code=True, seed=args.seed, enforce_eager=False)
    tokenizer = llm.get_tokenizer()
    tokenizer.chat_template = chat_template
    out_dir = Path(args.trace_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for split, parquet in (("train", args.train_parquet), ("validation", args.val_parquet)):
        questions = select_questions(Path(parquet), args.questions_per_split, args.seed, f"medium_{split}")
        requests, params = [], []
        for q in questions:
            prompt_ids = _render_prompt(tokenizer, q.question_text, chat_template)
            if len(prompt_ids) > args.max_prompt_tokens:
                raise SystemExit(f"{q.question_id}: prompt has {len(prompt_ids)} tokens > {args.max_prompt_tokens}")
            for s in range(args.samples_per_question):
                requests.append((q, s, prompt_ids))
                # detokenize=False: we only need token ids and logprobs; skipping text/decoded-token work removes the CPU
                # bottleneck of top-128 logprob output processing and most of its memory.
                params.append(SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=args.max_tokens, logprobs=args.topk, detokenize=False,
                                             seed=derive_sampling_seed(args.seed, f"medium_{split}", q.question_id, s)))
        path = out_dir / f"{split}.jsonl"
        n_tokens = 0
        # Generate in small batches and convert each batch's Logprob objects to plain floats at once: a whole split held as
        # vLLM output objects (~7k tokens x 129 logprobs x 512 responses, with decoded strings) blew a 192 GiB pod (OOMKilled).
        with path.open("w") as fh:
            for start in range(0, len(requests), args.gen_batch):
                batch = requests[start : start + args.gen_batch]
                outputs = llm.generate([{"prompt_token_ids": p} for _, _, p in batch], params[start : start + args.gen_batch], use_tqdm=False)
                for (q, s, prompt_ids), output in zip(batch, outputs, strict=True):
                    comp = output.outputs[0]
                    resp_ids = [int(t) for t in comp.token_ids]
                    sampled_lp, topk_ids, topk_lps = [], [], []
                    for lp_map, tok in zip(comp.logprobs, resp_ids, strict=True):
                        items = sorted(((int(i), float(v.logprob)) for i, v in lp_map.items()), key=lambda x: -x[1])
                        sampled = next(v for i, v in items if i == tok)
                        top = items[: args.topk]      # includes the sampled token only if it is in the top-k
                        sampled_lp.append(sampled); topk_ids.append([i for i, _ in top]); topk_lps.append([v for _, v in top])
                    n_tokens += len(resp_ids)
                    fh.write(json.dumps({"split": split, "question_id": q.question_id, "sample_index": s, "prompt_token_ids": prompt_ids,
                                         "response_token_ids": resp_ids, "finish_reason": comp.finish_reason, "student_sampled_logprob": sampled_lp,
                                         "student_topk_ids": topk_ids, "student_topk_logprobs": topk_lps}) + "\n")
                del outputs
                fh.flush()
                print(f"[generate] {split}: {min(start + args.gen_batch, len(requests))}/{len(requests)} responses, {n_tokens} tokens so far", flush=True)
        print(f"[generate] {split}: {len(questions)} questions x {args.samples_per_question} samples -> {len(requests)} responses, {n_tokens} tokens -> {path}", flush=True)


def score(args) -> None:
    import torch
    from transformers import AutoConfig
    from verl.utils.model import get_hf_auto_model_class

    device = torch.device("cuda")
    config = AutoConfig.from_pretrained(args.teacher, attn_implementation="sdpa")
    model = get_hf_auto_model_class(config).from_pretrained(args.teacher, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to(device).eval().requires_grad_(False)
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    print(f"[score] teacher={args.teacher} class={type(model).__name__} softcap={softcap}", flush=True)
    results = {}
    for split in ("train", "validation"):
        rows = [json.loads(l) for l in (Path(args.trace_dir) / f"{split}.jsonl").open()]
        seq_records, tok_mc, tok_topk, tok_topk_rn, tok_mass, tok_tlp_sampled, tok_slp_sampled = [], [], [], [], [], [], []
        for row in rows:
            prompt_ids, resp_ids = row["prompt_token_ids"], row["response_token_ids"]
            if not resp_ids:
                continue
            ids = torch.tensor([prompt_ids + resp_ids], dtype=torch.long, device=device)
            P, R = len(prompt_ids), len(resp_ids)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = model.model(input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool),
                                     position_ids=torch.arange(ids.shape[1], device=device).unsqueeze(0), use_cache=False, return_dict=True).last_hidden_state[0, P - 1 : P - 1 + R]
            s_lp_sampled = torch.tensor(row["student_sampled_logprob"], dtype=torch.float32, device=device)
            s_topk_ids = torch.tensor(row["student_topk_ids"], dtype=torch.long, device=device)          # [R, k]
            s_topk_lp = torch.tensor(row["student_topk_logprobs"], dtype=torch.float32, device=device)   # [R, k]
            resp = torch.tensor(resp_ids, dtype=torch.long, device=device)
            t_lp_sampled, t_lp_topk = [], []
            for start in range(0, R, args.chunk):
                with torch.inference_mode():
                    logits = model.lm_head(hidden[start : start + args.chunk]).float()
                    if softcap: logits = torch.tanh(logits / float(softcap)) * float(softcap)
                    lp = torch.log_softmax(logits, dim=-1)
                t_lp_sampled.append(lp.gather(1, resp[start : start + args.chunk, None]).squeeze(1))
                t_lp_topk.append(lp.gather(1, s_topk_ids[start : start + args.chunk]))
            t_lp_sampled = torch.cat(t_lp_sampled); t_lp_topk = torch.cat(t_lp_topk)
            p_s = s_topk_lp.exp(); mass = p_s.sum(1)
            rkl_mc = s_lp_sampled - t_lp_sampled
            rkl_topk = (p_s * (s_topk_lp - t_lp_topk)).sum(1)
            p_rn = p_s / mass[:, None]
            rkl_topk_rn = (p_rn * ((p_rn.log()) - (t_lp_topk - torch.logsumexp(t_lp_topk, 1, keepdim=True)))).sum(1)
            for buf, vals in ((tok_mc, rkl_mc), (tok_topk, rkl_topk), (tok_topk_rn, rkl_topk_rn), (tok_mass, mass), (tok_tlp_sampled, t_lp_sampled), (tok_slp_sampled, s_lp_sampled)):
                buf.extend(vals.tolist())
            seq_records.append({"question_id": row["question_id"], "sample_index": row["sample_index"], "response_length": R, "finish_reason": row["finish_reason"],
                                "rkl_mc_mean": rkl_mc.mean().item(), "rkl_topk_mean": rkl_topk.mean().item(), "rkl_topk_renorm_mean": rkl_topk_rn.mean().item(),
                                "student_topk_mass_mean": mass.mean().item(), "rkl_mc_sum": rkl_mc.sum().item()})
        def agg(vals):
            a = np.asarray(vals, dtype=np.float64); return {"token_mean": float(a.mean()), "token_median": float(np.median(a)), "n_tokens": int(a.size)}
        seq_means = np.asarray([r["rkl_mc_mean"] for r in seq_records])
        results[split] = {
            "n_sequences": len(seq_records), "n_tokens": len(tok_mc), "mean_response_length": float(np.mean([r["response_length"] for r in seq_records])),
            "finish_reasons": {k: sum(1 for r in seq_records if r["finish_reason"] == k) for k in sorted({r["finish_reason"] for r in seq_records})},
            "rkl_mc": agg(tok_mc) | {"seq_mean_of_means": float(seq_means.mean()), "seq_se": float(seq_means.std(ddof=1) / math.sqrt(len(seq_means)))},
            "rkl_topk": agg(tok_topk), "rkl_topk_renorm": agg(tok_topk_rn), "student_topk_mass": agg(tok_mass),
            "student_logprob_sampled": agg(tok_slp_sampled), "teacher_logprob_sampled": agg(tok_tlp_sampled),
            "rkl_mc_per_sequence_sum_mean": float(np.mean([r["rkl_mc_sum"] for r in seq_records])),
        }
        (Path(args.trace_dir) / f"{split}.scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in seq_records))
        r = results[split]
        print(f"[score] {split}: seqs={r['n_sequences']} tokens={r['n_tokens']} mean_len={r['mean_response_length']:.0f} "
              f"rKL_mc/token={r['rkl_mc']['token_mean']:.4f}±{r['rkl_mc']['seq_se']:.4f} rKL_topk/token={r['rkl_topk']['token_mean']:.4f} "
              f"rKL_topk_renorm/token={r['rkl_topk_renorm']['token_mean']:.4f} topk_mass={r['student_topk_mass']['token_mean']:.4f}", flush=True)
    payload = {"student": args.student, "teacher": args.teacher, "topk": args.topk, "questions_per_split": args.questions_per_split,
               "samples_per_question": args.samples_per_question, "seed": args.seed, "results": results}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"REVERSE_KL_RESULT {json.dumps({s: {'rkl_mc': results[s]['rkl_mc']['token_mean'], 'rkl_topk': results[s]['rkl_topk']['token_mean'], 'tokens': results[s]['n_tokens']} for s in results})}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["generate", "score"])
    p.add_argument("--student", required=True); p.add_argument("--teacher", default=None)
    p.add_argument("--chat_template", default=str(SCRIPTS / "data/gemma3_it_fewshot_math.jinja"))
    p.add_argument("--train_parquet", required=True); p.add_argument("--val_parquet", required=True)
    p.add_argument("--questions_per_split", type=int, default=128); p.add_argument("--samples_per_question", type=int, default=4)
    p.add_argument("--topk", type=int, default=128); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_tokens", type=int, default=8192); p.add_argument("--max_prompt_tokens", type=int, default=4096); p.add_argument("--max_model_len", type=int, default=12288)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85); p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--gen_batch", type=int, default=128, help="requests per vLLM generate call (bounds host memory for top-k logprobs)")
    p.add_argument("--trace_dir", required=True); p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.phase == "generate":
        generate(args)
    else:
        if not args.teacher or not args.out:
            p.error("score needs --teacher and --out")
        score(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
