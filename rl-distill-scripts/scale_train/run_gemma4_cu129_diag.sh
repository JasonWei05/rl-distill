#!/usr/bin/env bash
# ScaleTrain DIAGNOSTIC: localize the deterministic gemma-4 cu129 vLLM engine-init crash on p5 nodes
# (driver CUDA 12.8). Every verl RL attempt dies there: 1-2 of 8 ranks crash natively (no traceback),
# rest hang on NCCL broadcast -> watchdog at +30min. Devbox (CUDA-13 driver) engine tests pass BOTH
# eager+compile — but never exercised vLLM sleep mode (CuMemAllocator / cuMem* driver VMM APIs), which
# verl enables by default (free_cache_engine=True). Prime suspect: cu129 VMM calls on the 12.8 driver.
#
# Tests (each in its own python process; a native crash is captured as a signal exit code):
#   A: LLM(E2B) eager,   sleep OFF  -> baseline engine on p5
#   C: LLM(E2B) compile, sleep OFF  -> torch.compile/CUDA-graph path on p5
#   B: LLM(E2B) eager,   sleep ON + .sleep()/.wake_up() -> the verl-only code path
# Verdict matrix: A fails -> basic engine broken on p5; C fails -> compile path; B fails -> sleep mode
# (fix: rollout.free_cache_engine=False + rollout.enable_sleep_mode=False); all pass -> multi-rank issue.
set -uxo pipefail
cd /workspace/rl-distill
if [ -f .env ]; then set -a; source .env; set +a; fi

export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
export VENV=/tmp/.venv-gemma4
export GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu129}"
[ -x "${VENV}/bin/python" ] || VENV="${VENV}" bash rl-distill-scripts/setup_env_gemma4.sh
export PATH="${VENV}/bin:${PATH}"; source "${VENV}/bin/activate"
export HF_HOME=/tmp/hf_cache VLLM_CACHE_ROOT=/tmp/vllm_cache TRITON_CACHE_DIR=/tmp/triton_cache
export VLLM_ATTENTION_BACKEND=TRITON_ATTN VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0

echo "DIAG_DRIVER: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-E2B')"

cat > /tmp/diag_llm.py <<'PYEOF'
import sys
def main():
    mode = sys.argv[1]            # eager | compile
    sleep = sys.argv[2] == "sleep"
    from vllm import LLM, SamplingParams
    llm = LLM(model="google/gemma-4-E2B", tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=0.6, enforce_eager=(mode == "eager"),
              enable_sleep_mode=sleep, trust_remote_code=True)
    if sleep:
        llm.sleep(level=1); print("DIAG_SLEPT", flush=True)
        llm.wake_up();      print("DIAG_WOKE", flush=True)
    out = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=8))
    print("DIAG_GEN ::", out[0].outputs[0].text.strip()[:40], flush=True)
if __name__ == "__main__":
    main()
PYEOF

run_test() { # label mode sleepflag
  echo "===== DIAG_TEST_$1 START ====="
  python /tmp/diag_llm.py "$2" "$3"; rc=$?
  echo "===== DIAG_TEST_$1 RC=${rc} ====="
}
run_test A eager   nosleep
run_test C compile nosleep
run_test B eager   sleep
echo "DIAG_ALL_DONE"
exit 0
