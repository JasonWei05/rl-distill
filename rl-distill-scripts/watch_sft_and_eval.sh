#!/usr/bin/env bash
# Wait for an SFT run to finish, then run eval for selected pushed HF steps.
set -xeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

: "${TRAIN_LOG:?TRAIN_LOG is required}"
: "${HF_PUSH_REPO:?HF_PUSH_REPO is required}"
: "${STUDENT_HF_REPO:?STUDENT_HF_REPO is required}"
: "${EXP_NAME:?EXP_NAME is required}"

DONE_STEP="${DONE_STEP:-750}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EVAL_STEPS="${EVAL_STEPS:-250 500}"
RAY_DATA_HOME="${RAY_DATA_HOME:-/tmp/verl}"
EVAL_OUT="${EVAL_OUT:-${RAY_DATA_HOME}/data/eval_results/${EXP_NAME}}"
WATCH_LOG="${WATCH_LOG:-/tmp/distill_runs/logs/watch-${EXP_NAME}.log}"
mkdir -p "$(dirname "${WATCH_LOG}")" "${EVAL_OUT}"

last_step() {
    grep -aoE 'step:[0-9]+' "${TRAIN_LOG}" 2>/dev/null | tail -1 | cut -d: -f2 || true
}

eval_done() {
    local repo_name="${HF_PUSH_REPO##*/}"
    local s
    for s in ${EVAL_STEPS}; do
        local sub
        sub="$(printf 'step_%06d' "${s}")"
        [ -f "${EVAL_OUT}/${repo_name}__${sub}__summary.json" ] || return 1
    done
    return 0
}

training_proc_count() {
    pgrep -af '[m]ain_distill_offpolicy.py|[t]orchrun .*main_distill_offpolicy.py|[d]istill_sft_eval.sh' | wc -l
}

kill_training_procs() {
    pkill -TERM -f '[m]ain_distill_offpolicy.py' 2>/dev/null || true
    pkill -TERM -f '[t]orchrun .*main_distill_offpolicy.py' 2>/dev/null || true
    pkill -TERM -f '[d]istill_sft_eval.sh' 2>/dev/null || true
    sleep 10
    pkill -KILL -f '[m]ain_distill_offpolicy.py' 2>/dev/null || true
    pkill -KILL -f '[t]orchrun .*main_distill_offpolicy.py' 2>/dev/null || true
    pkill -KILL -f '[d]istill_sft_eval.sh' 2>/dev/null || true
}

{
    echo "[watch] start exp=${EXP_NAME} repo=${HF_PUSH_REPO} done_step=${DONE_STEP} eval_steps=${EVAL_STEPS}"
    while true; do
        if eval_done; then
            echo "[watch] eval already complete for ${EXP_NAME}"
            python3 rl-distill-scripts/update_eval_results_md.py \
                --results-dir "${EVAL_OUT}" \
                --run-name "${EXP_NAME}" || true
            exit 0
        fi

        step="$(last_step)"
        step="${step:-0}"
        procs="$(training_proc_count)"
        echo "[watch] $(date -u +%Y-%m-%dT%H:%M:%SZ) step=${step} training_procs=${procs}"

        if [ "${step}" -ge "${DONE_STEP}" ]; then
            echo "[watch] done step reached; stopping any leftover training ranks before eval"
            kill_training_procs
            if pgrep -af '[d]istill_eval_steps.sh|[_]eval_model_on_math' >/dev/null; then
                echo "[watch] eval process already running; waiting for summaries"
                sleep "${POLL_SECONDS}"
                continue
            fi
            echo "[watch] launching eval"
            env \
                HF_PUSH_REPO="${HF_PUSH_REPO}" \
                STUDENT_HF_REPO="${STUDENT_HF_REPO}" \
                STUDENT_TAG="${STUDENT_TAG:-}" \
                EXP_NAME="${EXP_NAME}" \
                EVAL_STEPS="${EVAL_STEPS}" \
                RAY_DATA_HOME="${RAY_DATA_HOME}" \
                EVAL_DATA_DIR="${EVAL_DATA_DIR:-${HOME}/verl/data}" \
                HF_HOME="${HF_HOME:-/tmp/hf_cache}" \
                MM_ENCODER_ATTN_BACKEND="${MM_ENCODER_ATTN_BACKEND:-}" \
                WANDB_PROXY_CPU_IP="${WANDB_PROXY_CPU_IP:-}" \
                WANDB_PROXY_CPU_PORT="${WANDB_PROXY_CPU_PORT:-}" \
                WANDB_PROXY_LOCAL_PORT="${WANDB_PROXY_LOCAL_PORT:-18080}" \
                bash rl-distill-scripts/distill_eval_steps.sh
            python3 rl-distill-scripts/update_eval_results_md.py \
                --results-dir "${EVAL_OUT}" \
                --run-name "${EXP_NAME}" || true
            exit 0
        fi

        if [ "${procs}" -eq 0 ]; then
            echo "[watch] training exited before done_step=${DONE_STEP}; not launching eval"
            exit 1
        fi

        sleep "${POLL_SECONDS}"
    done
} >> "${WATCH_LOG}" 2>&1
