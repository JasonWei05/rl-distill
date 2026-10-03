# HF reference (CPU) for Ouro-1.4B on a real 12-shot eval prompt: next-token entropy + logprob of vLLM's sampled tokens,
# and a short sampled/greedy continuation, at total_ut_steps 1 and 4.
import json, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
torch.manual_seed(0); torch.set_num_threads(64)
path = "/tmp/ouro_eval/hf/Ouro-1.4B"
tok = AutoTokenizer.from_pretrained(path)
trace = json.loads(open("/tmp/ouro_eval/results_k4/ouro1p4b_ut1_fewshot12/traces/ouro1p4b_ut1_fewshot12__id_easy.jsonl").readline())
prompt, resp = trace["prompt_token_ids"], trace["response_token_ids"][:64]
for n in [int(x) for x in sys.argv[1].split(",")]:
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True); cfg.total_ut_steps = n
    m = AutoModelForCausalLM.from_pretrained(path, config=cfg, trust_remote_code=True, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        ids = torch.tensor([prompt + resp])
        lp = torch.log_softmax(m(ids, use_cache=False).logits.float(), -1)[0]
        P = len(prompt)
        first = lp[P - 1]
        ent = -(first.exp() * first).sum().item()
        resp_lp = [lp[P - 1 + i, t].item() for i, t in enumerate(resp)]
        top = torch.topk(first, 5)
        greedy = m.generate(torch.tensor([prompt]), max_new_tokens=40, do_sample=False)[0][P:]
        sampled = m.generate(torch.tensor([prompt]), max_new_tokens=40, do_sample=True, temperature=1.0, top_p=1.0, top_k=0)[0][P:]
    print(f"=== total_ut_steps={n}")
    print("first-token entropy (nats):", round(ent, 3), "top5:", [(tok.decode([i]), round(v.exp().item(), 3)) for v, i in zip(top.values, top.indices)])
    print("mean logprob of vLLM's first 64 sampled tokens:", round(sum(resp_lp) / len(resp_lp), 3))
    print("greedy :", repr(tok.decode(greedy)))
    print("sampled:", repr(tok.decode(sampled)), flush=True)
