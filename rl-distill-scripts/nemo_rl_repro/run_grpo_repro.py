#!/usr/bin/env python3
"""Wrapper around nemo-rl's examples/run_grpo.py that registers our adapters first.

Registration MUST precede setup_response_data because environments are created
before datasets (nemo_rl/data/utils.py setup_response_data: envs first). The
dataset needs no registration — the yaml references it by dotted path.

Run from the nemo-rl checkout's uv env (venv built separately; do not build here):

    source /mnt/efs/jasonwei/rl-distill/.env   # HF_TOKEN (gated google/gemma-4-E2B), WANDB_API_KEY
    cd /mnt/efs/jasonwei/rl-distill/third_party/nemo-rl
    uv run --locked --extra automodel \
        python /mnt/efs/jasonwei/rl-distill/rl-distill-scripts/nemo_rl_repro/run_grpo_repro.py \
        --config /mnt/efs/jasonwei/rl-distill/rl-distill-scripts/nemo_rl_repro/config/dapo_gemma4_e2b_pt_repro.yaml
"""

import importlib.util
import os
import sys

_REPRO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_REPRO_DIR))
_NEMO_RL_ROOT = os.environ.get(
    "NEMO_RL_ROOT", os.path.join(_REPO_ROOT, "third_party", "nemo-rl")
)

# Make rl_distill_nemo importable in the driver AND in Ray actors: create_env()
# forwards os.environ (including PYTHONPATH) into each actor's runtime_env, and
# the math_strict actor runs with PY_EXECUTABLES.SYSTEM (= this interpreter).
sys.path.insert(0, _REPRO_DIR)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (_REPRO_DIR, os.environ.get("PYTHONPATH", "")) if p
)

from nemo_rl.distributed.ray_actor_environment_registry import (  # noqa: E402
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES  # noqa: E402
from nemo_rl.environments.utils import register_env  # noqa: E402

register_env("math_strict", "rl_distill_nemo.strict_math_env.StrictMathEnvironment")
ACTOR_ENVIRONMENT_REGISTRY["rl_distill_nemo.strict_math_env.StrictMathEnvironment"] = (
    PY_EXECUTABLES.SYSTEM
)


def _force_local_ray() -> None:
    """Make nemo_rl's init_ray skip attaching to pre-existing Ray clusters.

    init_ray calls ray.init(address="auto") first, which on a shared devbox attaches
    to whatever stale/foreign cluster the discovery pointer names (and dies on version
    mismatch). Raising ConnectionError from that exact call routes init_ray into its
    own except-ConnectionError branch, which starts a fresh local cluster. Enabled via
    NEMORL_FORCE_LOCAL_RAY=1; pods (no stale clusters) keep the stock path.
    """
    import ray

    _orig_init = ray.init

    def _no_auto_init(*args, **kwargs):
        if kwargs.get("address") == "auto":
            raise ConnectionError("NEMORL_FORCE_LOCAL_RAY=1: not attaching to existing clusters")
        # the dashboard/API server is flaky on a heavily loaded shared box and the
        # local-start call hardcodes include_dashboard=True; we don't need it
        kwargs["include_dashboard"] = False
        ret = _orig_init(*args, **kwargs)
        # Pin RAY_ADDRESS to OUR cluster: later diagnostics (memory_tracker ->
        # ray.memory_summary) re-run bootstrap discovery, which dies with
        # "Found multiple active Ray instances" on a shared box.
        try:
            os.environ["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address
        except Exception:
            pass
        return ret

    ray.init = _no_auto_init


def main() -> None:
    if os.environ.get("NEMORL_FORCE_LOCAL_RAY") == "1":
        _force_local_ray()
    run_grpo_path = os.path.join(_NEMO_RL_ROOT, "examples", "run_grpo.py")
    spec = importlib.util.spec_from_file_location("nemo_rl_run_grpo", run_grpo_path)
    run_grpo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_grpo)
    run_grpo.main()


if __name__ == "__main__":
    main()
