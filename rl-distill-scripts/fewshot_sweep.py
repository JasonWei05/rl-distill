#!/usr/bin/env python3
"""Few-shot length sweep: find the SHORTEST few-shot math prompt that still reproduces
google/gemma-3-4b-pt GREEDY GSM8K (~38) and MATH500 (~24).

Builds N-shot prefix templates (first N of the 12 demos in gemma3_it_fewshot_math.jinja) for
N in NSHOTS, then evals base 4B greedy (temp 0) on GSM8K + MATH500 for each, one variant per
free GPU. max_tokens=2048 is safe: the doc verified greedy@1024 == greedy@20480 for base 4B.

  set -a && source .env; set +a ; .venv/bin/python rl-distill-scripts/fewshot_sweep.py
Writes per-variant JSON to the scratch dir and prints a summary table.
"""
import json, os, re, subprocess, time
from pathlib import Path

ROOT = "/mnt/efs/jasonwei/rl-distill"
SC = "/tmp/claude-1305/-mnt-efs-jasonwei-rl-distill/ef23df8c-77b2-46fa-b5c4-5819832c057a/scratchpad/fewshot_sweep"
PY = f"{ROOT}/.venv/bin/python"
FULL_TPL = f"{ROOT}/rl-distill-scripts/data/gemma3_it_fewshot_math.jinja"
DATA = os.path.expanduser("~/verl/data")
HFH = os.path.expanduser("~/.cache/huggingface")
MODEL = "google/gemma-3-4b-pt"
GPUS = [2, 3, 4, 5]                      # 0,1 = other user; 6,7 busy
NSHOTS = [0, 1, 2, 3, 4, 6, 8, 12]
DATASETS = [f"{DATA}/math__gsm8k_test.parquet", f"{DATA}/math__math_500_x2.parquet"]

HEAD = "{{ bos_token }}{% raw %}"


def parse_full():
    """Return (list[(q,a)] demos, tail_jinja_str) from the full 12-shot template."""
    txt = Path(FULL_TPL).read_text()
    raw = txt.split("{% raw %}", 1)[1].split("{% endraw %}", 1)[0]
    tail = "{% endraw %}" + txt.split("{% endraw %}", 1)[1]
    pairs = re.findall(
        r"<start_of_turn>user\n(.*?)<end_of_turn>\n<start_of_turn>model\n(.*?)<end_of_turn>",
        raw, flags=re.DOTALL)
    return pairs, tail


def build(n, demos, tail):
    body = "".join(
        f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n{a}<end_of_turn>\n"
        for q, a in demos[:n])
    return HEAD + body + tail


def main():
    Path(SC).mkdir(parents=True, exist_ok=True)
    demos, tail = parse_full()
    assert len(demos) == 12, f"expected 12 demos, got {len(demos)}"
    # sanity: rebuilt 12-shot must byte-match the original
    assert build(12, demos, tail) == Path(FULL_TPL).read_text(), "12-shot rebuild mismatch"

    tpl_paths = {}
    for n in NSHOTS:
        p = f"{SC}/tpl_{n}shot.jinja"
        Path(p).write_text(build(n, demos, tail))
        tpl_paths[n] = p

    # prompt token counts (one tokenizer load)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    print("prompt lengths (few-shot preamble + a short test question):")
    toklen = {}
    for n in NSHOTS:
        tok.chat_template = Path(tpl_paths[n]).read_text()
        s = tok.apply_chat_template([{"role": "user", "content": "What is 2+2?"}],
                                    add_generation_prompt=True, tokenize=False)
        toklen[n] = len(tok.encode(s, add_special_tokens=False))
        print(f"  {n:>2}-shot: {toklen[n]:>4} tokens")

    # orchestrate: one eval_math_passk per variant on a free GPU
    def free(g):
        try:
            u = subprocess.check_output(
                ["nvidia-smi", "-i", str(g), "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"], text=True).strip()
            return int(u) < 3000
        except Exception:
            return False

    def cmd(n):
        out = f"{SC}/res_{n}shot.json"
        return (f"{PY} {ROOT}/rl-distill-scripts/eval_math_passk.py --model {MODEL} "
                f"--tag base4b_{n}shot --chat_template {tpl_paths[n]} "
                f"--datasets {' '.join(DATASETS)} --out {out} "
                f"--temperature 0 --max_tokens 2048 --max_model_len 4096 "
                f"--gpu_memory_utilization 0.9")

    queue = list(NSHOTS)
    running = {}   # gpu -> (n, popen)
    log = open(f"{SC}/sweep_status.log", "a")

    def say(m):
        log.write(f"[{time.strftime('%H:%M:%S')}] {m}\n"); log.flush(); print(m, flush=True)

    say(f"sweep start: variants={NSHOTS} gpus={GPUS} toklen={toklen}")
    while queue or running:
        for g in list(running):
            n, p = running[g]
            if p.poll() is not None:
                say(f"DONE {n}-shot (gpu {g}, rc={p.returncode})")
                del running[g]
        for g in GPUS:
            if not queue:
                break
            if g in running or not free(g):
                continue
            n = queue.pop(0)
            lp = f"{SC}/eval_{n}shot.log"
            env = f"CUDA_VISIBLE_DEVICES={g} HF_HOME={HFH} HF_HUB_CACHE={HFH}/hub VLLM_WORKER_MULTIPROC_METHOD=spawn"
            full = f"cd {ROOT} && set -a && source .env 2>/dev/null; set +a; {env} {cmd(n)}"
            p = subprocess.Popen(["bash", "-c", full], stdout=open(lp, "w"),
                                 stderr=subprocess.STDOUT, start_new_session=True)
            running[g] = (n, p)
            say(f"LAUNCH {n}-shot on gpu {g} (pid {p.pid}) -> {lp}")
            time.sleep(15)
        time.sleep(20)

    # collect
    say("\n=== SWEEP RESULTS: base 4B greedy (temp 0) ===")
    say(f"{'variant':>9} | {'tokens':>6} | {'GSM8K':>7} | {'MATH500':>7}")
    rows = []
    for n in NSHOTS:
        rp = f"{SC}/res_{n}shot.json"
        if not os.path.exists(rp):
            say(f"{n:>7}-shot | {toklen[n]:>6} | (no result)"); continue
        r = json.load(open(rp))["results"]
        g = r.get("math__gsm8k_test", {}).get("mean@k")
        m = r.get("math__math_500_x2", {}).get("mean@k")
        rows.append((n, toklen[n], g, m))
        say(f"{n:>7}-shot | {toklen[n]:>6} | {g:>7} | {m:>7}")
    json.dump({"target": {"gsm8k": 38.06, "math500": 23.80}, "rows": rows},
              open(f"{SC}/summary.json", "w"), indent=2)
    say(f"target (12-shot doc): GSM8K 38.06, MATH500 23.80")
    say("SWEEP COMPLETE")


if __name__ == "__main__":
    main()
