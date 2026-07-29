#!/usr/bin/env bash
# ScaleTrain DIAGNOSTIC 2: capture the exact multi-rank deadlock in gemma-4 verl RL init.
# Evidence so far: all 6 full runs hang (not crash — worker .err files clean, no OOM) on the default
# NCCL PG with ranks at DIFFERENT collective seqnums (e.g. rank0 seq1521/Numel1 vs rank1-2 seq1513/
# Numel256) during vLLM engine bring-up => collective-order desync between verl's rollout integration
# (built for vllm<=0.12; setup.py pin) and vllm 0.25.1. Single-rank engines pass all tests (diag 1).
#
# This job: start the REAL RL run in the background, wait for FSDP init ("Total steps:"), give the
# deadlock 6 minutes to form, then py-spy dump EVERY python process — the per-rank stacks name the
# desynced call sites. Then kill the run and exit 0 (logs land in the pod stdout).
set -uxo pipefail
cd /workspace/rl-distill
if [ -f .env ]; then set -a; source .env; set +a; fi

export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
uv pip install --target /tmp/pyspy py-spy >/dev/null 2>&1 || pip install --target /tmp/pyspy py-spy
export PATH="/tmp/pyspy/bin:${PATH}"
py-spy --version || true

# real run, small knobs (the hang is at init; sizes irrelevant), no side effects
export GEMMA4_MODEL=google/gemma-4-E2B DATA_SEED=42
export TOTAL_TRAINING_STEPS=1 VAL_BEFORE_TRAIN=False HF_PUSH_ENABLE=False SAVE_FREQ=1000 TEST_FREQ=1000
export EXP_NAME="DIAG-deadlock-gemma4-e2b"
bash rl-distill-scripts/scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh > /tmp/rl_run.log 2>&1 &
RUN_PID=$!

# wait for FSDP init marker (venv build + setup ~15-20 min), then let the deadlock form
i=0; until grep -qa "Total steps:" /tmp/rl_run.log || [ $i -ge 240 ]; do sleep 10; i=$((i+1)); done
echo "DIAG2: 'Total steps:' seen after $((i*10))s; letting the hang form for 360s"
sleep 360

echo "===== DEADLOCK_STACKS_BEGIN ====="
# Ray retitles workers (ray::WorkerDict...) so match by executable, not cmdline: dump EVERY python proc.
for pid in $(ps -eo pid= | tr -d ' '); do
  exe=$(readlink -f "/proc/${pid}/exe" 2>/dev/null) || continue
  case "$exe" in *python*) ;; *) continue ;; esac
  CMD=$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | cut -c1-90)
  echo "--- PID ${pid} :: ${CMD} ---"
  py-spy dump --pid "$pid" --nonblocking 2>&1 | grep -vE "^Process |wandb_v1|hf_Yk" | head -45
done
echo "===== DEADLOCK_STACKS_END ====="

echo "===== RL RUN LOG TAIL ====="
tail -30 /tmp/rl_run.log
kill -TERM "$RUN_PID" 2>/dev/null; sleep 5
pkill -9 -f "main_dapo" 2>/dev/null || true
echo "DIAG2_DONE"
exit 0
