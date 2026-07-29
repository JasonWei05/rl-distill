#!/usr/bin/env bash
# gemma-4-E2B few-shot ablation eval (1 GPU): 12-shot vs 0-shot prompt on 400 random
# questions from the DeepScaleR strict-4/4 TRAIN split (seed-42 sample, k=1 per question),
# temp 1.0, top_p 1.0, top_k -1, max 4096 new tokens. Success rate + response-length
# metrics print to stdout; full per-sample traces upload to a private HF dataset repo.
# Templates: rl-distill-scripts/data/gemma3_it_fewshot_math.jinja (the 12-shot prompt the
# RL runs train with) vs gemma3_it_0shot_math.jinja (same file with the example block removed).
set -euo pipefail
cd /workspace/rl-distill

MODEL="${GEMMA4_MODEL:-google/gemma-4-E2B}"

# HF_TOKEN arrives as a forwarded env var; .env may be absent.
if [ -f .env ]; then set -a; source .env; set +a; fi

export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
export VENV="${VENV:-/tmp/.venv-gemma4}"
export GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu129}"
if [ ! -x "${VENV}/bin/python" ]; then
  echo "### building ${VENV} via setup_env_gemma4.sh (${GEMMA4_CUDA_VARIANT})"
  VENV="${VENV}" bash rl-distill-scripts/setup_env_gemma4.sh
fi
export PATH="${VENV}/bin:${PATH}"; source "${VENV}/bin/activate"
python3 -c "import math_verify" || { echo "FATAL: math-verify missing from ${VENV}"; exit 1; }
echo "MATH_VERIFY_OK"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
CU13_LIB="${VENV}/lib/python3.12/site-packages/nvidia/cu13/lib"
if [ -d "${CU13_LIB}" ]; then export LD_LIBRARY_PATH="${CU13_LIB}:${LD_LIBRARY_PATH}"; fi
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_FLASHINFER_SAMPLER='0'

DATA=rl-distill-scripts/data/eval_ablation/ds4of4strict_train400.parquet
OUT=/tmp/eval_ablation
mkdir -p "$OUT/traces"

for arm in fewshot12 0shot; do
  tmpl=rl-distill-scripts/data/gemma3_it_fewshot_math.jinja
  [ "$arm" = "0shot" ] && tmpl=rl-distill-scripts/data/gemma3_it_0shot_math.jinja
  echo "### [$arm] eval starting (template: $tmpl)"
  python3 rl-distill-scripts/eval_math_passk.py \
    --model "$MODEL" --tag "$arm" \
    --chat_template "$tmpl" \
    --datasets "$DATA" \
    --out "$OUT/results_${arm}.json" \
    --max_tokens 4096 --max_model_len 8192 \
    --temperature 1.0 --top_p 1.0 --top_k -1 \
    --gpu_memory_utilization 0.9 --enforce_eager \
    --trace_dir "$OUT/traces"
done

echo "### summary (success rate + response-length metrics per arm)"
python3 - <<'PY'
import glob
import json
import statistics as st

for f in sorted(glob.glob("/tmp/eval_ablation/traces/*.jsonl")):
    rows = [json.loads(line) for line in open(f)]
    lens = sorted(len(r["response_token_ids"]) for r in rows)
    accs = [r["acc"] for r in rows]
    pct = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]
    print("ABLATION_RESULT " + json.dumps({
        "arm": f.rsplit("/", 1)[-1].split("__")[0],
        "n": len(rows),
        "success_rate_pct": round(100 * sum(accs) / len(accs), 2),
        "resp_len_mean": round(st.mean(lens), 1),
        "resp_len_median": pct(0.5),
        "resp_len_p90": pct(0.9),
        "resp_len_max": max(lens),
        "clip_at_4096_pct": round(100 * sum(l >= 4096 for l in lens) / len(lens), 2),
    }))
PY

python3 - <<'PY'
from huggingface_hub import HfApi

api = HfApi()
repo = "JWei05/gemma4-e2b-fewshot-ablation-traces"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path="/tmp/eval_ablation", repo_id=repo, repo_type="dataset")
print(f"TRACES_UPLOADED {repo}")
PY
echo "EVAL_ABLATION_DONE"
