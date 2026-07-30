"""Auto-imported by Python's site machinery in every interpreter that has this
directory on PYTHONPATH — which run_grpo_repro.py guarantees for the driver AND
all Ray actors (same mechanism that makes rl_distill_nemo importable there).
Also copy into the DTensor policy worker venv's site-packages as belt-and-suspenders.

Two version-gated patches enabling gemma-4 E2B training on transformers >= 5.5.2,
plus a fail-fast guard on the locked 5.5.0 stack (act-ckpt there silently corrupts
training — better to crash at setup than train garbage). Background (root-caused
2026-07-30, see PROGRESS_LOG):

gemma-4 E2B has KV-shared layers. On transformers < 5.5.2 they are only correct
with use_cache=True (nemo-rl PR #2224 / Automodel#1705), but HF force-disables
the cache in grad mode under gradient checkpointing, so with
activation_checkpointing the TRAINING forward silently computed garbage logits
(probs_ratio bulk < 0.8, clamped pinned at 0.80) while the no-grad logprob pass
stayed correct. transformers >= 5.5.2 (HF #45312) fixes KV sharing without the
cache — but introduces two new landmines this file defuses:

Patch 1 — remove the use_cache=True workaround (nemo-rl's own TODO): on 5.5.4
the cache path CRASHES (modeling_gemma4.py shared_kv_states KeyError) because
shared layers always read the shared dict; with the workaround removed, every
pass runs use_cache=False, which is correct on >= 5.5.2.

Patch 2 — MixedPrecisionPolicy(cast_forward_inputs=False): 5.5.2+ passes K/V
between layers by MUTATING a plain `shared_kv_states` dict kwarg. FSDP2's
pre-forward input cast (torch _fsdp_state.py: _apply_to_tensors over kwargs)
REBUILDS containers, so each fully-sharded decoder layer receives a fresh copy
— anchors write into copies, shared layers read empty dicts -> KeyError. The
cast is value-wise a no-op here (model and activations are already bf16;
_cast_fp_tensor returns same-dtype tensors unchanged) and the fp32 output cast
is gated separately on output_dtype, so disabling cast_forward_inputs only
removes the destructive container rebuild.
"""

import importlib.abc
import importlib.util
import sys

_TRAIN_MOD = "nemo_rl.models.automodel.train"
_SETUP_MOD = "nemo_rl.models.automodel.setup"


def _transformers_has_kv_sharing_fix() -> bool:
    try:
        import transformers
        from packaging.version import Version

        return Version(transformers.__version__) >= Version("5.5.2")
    except Exception:
        return False


def _patch_train(module):
    if not _transformers_has_kv_sharing_fix():
        return
    module._needs_kv_cache_for_shared_layers = lambda model: False
    print(
        "[rl-distill sitecustomize] transformers >= 5.5.2: disabled nemo-rl's "
        "use_cache KV-sharing workaround (per its own TODO)"
    )


def _patch_setup(module):
    real = module.MixedPrecisionPolicy

    def _no_input_cast_mp_policy(*args, **kwargs):
        kwargs["cast_forward_inputs"] = False
        return real(*args, **kwargs)

    module.MixedPrecisionPolicy = _no_input_cast_mp_policy
    print(
        "[rl-distill sitecustomize] transformers >= 5.5.2: forcing "
        "MixedPrecisionPolicy(cast_forward_inputs=False) so FSDP2 stops "
        "copying the gemma-4 shared_kv_states dict kwarg"
    )


def _guard_setup_old_transformers(module):
    """Fail fast instead of training garbage on the locked stack.

    On transformers < 5.5.2 (uv.lock pins 5.5.0), activation checkpointing makes
    HF disable the KV cache in the grad-mode forward, and gemma-4 E2B/E4B's
    KV-shared layers then silently compute garbage logits in train() only (the
    no-grad logprob pass stays correct, so val looks fine while the PPO loss is
    corrupted — probs_ratio bulk < 0.8, clamped pinned at 0.80). This launch dir
    is gemma-4-only, so raise unconditionally rather than let it train.
    """
    real = module.FSDP2Config

    def _checked_fsdp2_config(*args, **kwargs):
        if kwargs.get("activation_checkpointing"):
            raise RuntimeError(
                "[rl-distill sitecustomize] activation_checkpointing=true with "
                "transformers < 5.5.2 SILENTLY CORRUPTS gemma-4 E2B training "
                "(KV-shared layers vs grad-mode cache disable; PROGRESS_LOG "
                "2026-07-30). Upgrade the policy-worker venv to transformers "
                "5.5.4 (plus this sitecustomize) or set "
                "policy.dtensor_cfg.activation_checkpointing=false."
            )
        return real(*args, **kwargs)

    module.FSDP2Config = _checked_fsdp2_config


def _patch_setup_dispatch(module):
    if _transformers_has_kv_sharing_fix():
        _patch_setup(module)
    else:
        _guard_setup_old_transformers(module)


_PATCHES = {_TRAIN_MOD: _patch_train, _SETUP_MOD: _patch_setup_dispatch}


class _Gemma4KVSharingFixer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _PATCHES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module
        patch = _PATCHES[fullname]

        def exec_module(module):
            orig_exec(module)
            patch(module)  # each patch is version-gated internally

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Gemma4KVSharingFixer())
