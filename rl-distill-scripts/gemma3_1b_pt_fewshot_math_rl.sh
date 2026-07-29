#!/usr/bin/env bash
set -euo pipefail
# Gemma 3 1B PT — few-shot math DAPO RL, LOCAL single-machine run on GPUs 5,7 (2xH100).
# Uses an isolated local Ray (its own _temp_dir, no dashboard) so it does NOT touch any other
# Ray cluster on the shared devbox. Thin wrapper over gemma3_pt_fewshot_math_rl.sh.
#
#   bash rl-distill-scripts/gemma3_1b_pt_fewshot_math_rl.sh
#   # quick smoke first:
#   TOTAL_TRAINING_STEPS=1 VAL_BEFORE_TRAIN=False HF_PUSH_ENABLE=False \
#     bash rl-distill-scripts/gemma3_1b_pt_fewshot_math_rl.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_TAG=${MODEL_TAG:-"gemma3-1b-pt"}
export MODEL_REPO=${MODEL_REPO:-"google/gemma-3-1b-pt"}
export MODEL_PATH=${MODEL_PATH:-"google/gemma-3-1b-pt"}   # from HF cache (gated; HF_TOKEN in .env)
export HF_PUSH_REPO=${HF_PUSH_REPO:-"JWei05/DAPO-Gemma3-1B-PT-FewShotMath"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,7}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
export NNODES=1
export OFFLOAD=${OFFLOAD:-False}
export SAVE_FREQ=${SAVE_FREQ:-25}
export RAY_ADDRESS=${RAY_ADDRESS:-local}                  # isolated local Ray, not the shared cluster
RAY_TMP=${RAY_TMP:-/tmp/ray_gemma1b_fewshot_${USER:-run}}
# Cap Ray CPUs: the raylet prestarts one Python worker per CPU (--num_prestart_python_workers=num_cpus),
# so on this 192-CPU box that's a 192-way import storm from the EFS venv at startup. A 2-GPU FSDP run
# needs only ~8 CPUs (max_colocate_count=3 -> {CPU:3,GPU:1} bundles), so 32 is ample and cuts the churn.
RAY_NUM_CPUS=${RAY_NUM_CPUS:-32}
# On a heavily-loaded shared box (load avg > #CPUs — routinely ~180 here) the raylet is CPU-starved and
# can't register with the GCS inside Ray's hardcoded 30s window -> ray.init "timed out during startup".
# We patched ray's node.py to read this env var (default 30); bump it so startup survives the
# slow-but-successful registration. Harmless no-op if that node.py patch is absent (e.g. after a fresh
# setup_env.sh) — the unpatched code just ignores it. See PROGRESS_LOG for the one-line patch.
export RAY_RAYLET_START_WAIT_TIME_S=${RAY_RAYLET_START_WAIT_TIME_S:-300}
# The dominant failure: the raylet aborts ("Timed out waiting for file .../metrics_agent_port",
# node_manager.cc) because the dashboard/metrics AGENT is a heavy Python process whose imports from the
# EFS venv take ~70-80s (per-file EFS metadata latency, not CPU-bound), while the raylet's default wait
# is only ~30s. This is STRUCTURAL on the EFS venv, not transient load. The governing knob is the C++
# RayConfig `agent_register_timeout_ms`; the RAY_<name> env var did NOT take effect for it, so we set it
# through the authoritative channel — ray.init(_system_config=...) — to give the slow agent time to
# register. worker_register_timeout_seconds is bumped the same way for the later worker-registration step.
RAY_AGENT_REGISTER_TIMEOUT_MS=${RAY_AGENT_REGISTER_TIMEOUT_MS:-300000}
RAY_WORKER_REGISTER_TIMEOUT_S=${RAY_WORKER_REGISTER_TIMEOUT_S:-600}

exec bash "${HERE}/gemma3_pt_fewshot_math_rl.sh" \
  +ray_kwargs.ray_init._temp_dir="${RAY_TMP}" \
  +ray_kwargs.ray_init.include_dashboard=False \
  +ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}" \
  +ray_kwargs.ray_init._system_config.agent_register_timeout_ms="${RAY_AGENT_REGISTER_TIMEOUT_MS}" \
  +ray_kwargs.ray_init._system_config.worker_register_timeout_seconds="${RAY_WORKER_REGISTER_TIMEOUT_S}" \
  "$@"
