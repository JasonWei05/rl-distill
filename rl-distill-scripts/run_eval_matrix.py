#!/usr/bin/env python3
"""Orchestrate the eval matrix across whatever GPUs are free.

Jobs = (model x {math, ood}). Runs each on a free GPU (polls a candidate pool, skips busy/other-user
GPUs), up to as many as there are free GPUs. math -> eval_math_passk.py (pass@k/mean@k/maj@k + traces);
ood -> lm_eval --model vllm on the 5 OOD benchmarks grouped by shot count. Results land in the
scratch dir; per-sample eval traces go under $HOME/verl/eval_traces/.
"""
import json, os, subprocess, time
from pathlib import Path

ROOT = "/mnt/efs/jasonwei/rl-distill"
SC = "/tmp/claude-1305/-mnt-efs-jasonwei-rl-distill/ef23df8c-77b2-46fa-b5c4-5819832c057a/scratchpad"
TRACES = os.path.expanduser("~/verl/eval_traces")
OOD_DIR = os.path.join(TRACES, "ood")
PY = f"{ROOT}/.venv/bin/python"
LM = f"{ROOT}/.venv/bin/lm_eval"
TPL = f"{ROOT}/rl-distill-scripts/data/gemma3_it_fewshot_math.jinja"
DATA = os.path.expanduser("~/verl/data")
HFH = os.path.expanduser("~/.cache/huggingface")
CANDIDATE_GPUS = [6, 7, 2, 3, 4, 5]           # never 0,1 (other user)
DATASETS = " ".join(f"{DATA}/{d}" for d in [
    "math__gsm8k_test.parquet", "math__math_500_x2.parquet", "math__olympiadbench_x2.parquet",
    "math__minervamath_x4.parquet", "math__beyondaime_x8.parquet", "math__aime2025_x32.parquet",
    "math__aime2026_x32.parquet", "dapo_rl_val100_x16.parquet"])

def _resolve(glob_pat):
    import glob
    hits = glob.glob(os.path.expanduser(glob_pat))
    return hits[0] if hits else None

RL1B = _resolve("~/.cache/huggingface/hub/models--JWei05--DAPO-Gemma3-1B-PT-FewShotMath-seed44/snapshots/*/step_000350")
RL4B = _resolve("~/.cache/huggingface/hub/models--JWei05--DAPO-Gemma3-4B-PT-FewShotMath/snapshots/*/step_000150")
MODELS = {"1b_base": "google/gemma-3-1b-pt", "1b_rl350": RL1B,
          "4b_base": "google/gemma-3-4b-pt", "4b_rl150": RL4B}

def math_cmd(tag, model):
    return (f"{PY} {ROOT}/rl-distill-scripts/eval_math_passk.py --model '{model}' --tag {tag} "
            f"--chat_template {TPL} --datasets {DATASETS} --out {SC}/passk_{tag}.json --trace_dir {TRACES}")

def ood_cmd(tag, model, shot, tasks):
    # one lm_eval call per shot-group so each gets a fresh GPU (0.7 util) — chaining calls on one
    # GPU failed because the prior vLLM didn't release memory before the next started.
    ma = f"pretrained={model},dtype=bfloat16,gpu_memory_utilization=0.7,max_model_len=4096,add_bos_token=True"
    return (f"{LM} --model vllm --model_args {ma} --tasks {tasks} --num_fewshot {shot} "
            f"--batch_size auto --output_path {OOD_DIR}/{tag}_{shot}shot --log_samples --seed 0")

OOD_GROUPS = [("5", "mmlu,winogrande,triviaqa"), ("10", "hellaswag"), ("25", "arc_challenge")]
# PASS 2: re-run the 2 crashed math jobs (1b_base CUDA, 4b_rl150 missing-processor now fixed) +
# ALL OOD (5-shot partials get overwritten cleanly), each OOD shot-group as its own job.
JOBS = [("math", t, MODELS[t], "", "") for t in ["1b_base", "4b_rl150"]] + \
       [("ood", t, MODELS[t], sh, tsk)
        for t in ["1b_base", "1b_rl350", "4b_base", "4b_rl150"] for sh, tsk in OOD_GROUPS]

def gpu_free(idx):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", str(idx), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True).strip()
        return int(out) < 3000
    except Exception:
        return False

def main():
    Path(OOD_DIR).mkdir(parents=True, exist_ok=True)
    status = open(f"{SC}/eval_matrix_status.log", "a")
    def log(m):
        status.write(f"[{time.strftime('%H:%M:%S')}] {m}\n"); status.flush()
        print(m, flush=True)
    log(f"matrix start: RL1B={RL1B} RL4B={RL4B}; jobs={[(j[0], j[1], j[3]) for j in JOBS]}")
    queue = list(JOBS)
    running = {}   # gpu_idx -> (name, popen, logpath)
    while queue or running:
        # reap finished
        for g in list(running):
            name, p, lp = running[g]
            if p.poll() is not None:
                log(f"DONE {name} (gpu {g}, rc={p.returncode})")
                del running[g]
        # launch onto free gpus
        for g in CANDIDATE_GPUS:
            if not queue:
                break
            if g in running or not gpu_free(g):
                continue
            kind, tag, model, shot, tasks = queue[0]
            if model is None:
                log(f"SKIP {kind}:{tag} (model path unresolved)"); queue.pop(0); continue
            queue.pop(0)
            suffix = f"_{shot}shot" if kind == "ood" else ""
            name = f"{kind}:{tag}{suffix}"
            cmd = math_cmd(tag, model) if kind == "math" else ood_cmd(tag, model, shot, tasks)
            env = f"CUDA_VISIBLE_DEVICES={g} HF_HOME={HFH} HF_HUB_CACHE={HFH}/hub VLLM_WORKER_MULTIPROC_METHOD=spawn"
            lp = f"{SC}/evalmx_{kind}_{tag}{suffix}.log"
            full = f"cd {ROOT} && set -a && source .env 2>/dev/null; set +a; {env} {cmd}"
            # NOTE: do NOT wrap in the `setsid` command — it forks+exits so Popen would track the
            # dead parent and report the job "done" instantly. start_new_session keeps the child in
            # its own group (survives terminal signals) while Popen still tracks the real bash proc.
            p = subprocess.Popen(["bash", "-c", full],
                                 stdout=open(lp, "w"), stderr=subprocess.STDOUT, start_new_session=True)
            running[g] = (name, p, lp)
            log(f"LAUNCH {name} on gpu {g} (pid {p.pid}) -> {lp}")
            time.sleep(20)   # stagger EFS loads
        time.sleep(30)
    log("matrix COMPLETE")

if __name__ == "__main__":
    main()
