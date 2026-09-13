# rl-distill fork: frozen HF teacher evaluated inside the actor update (one extra forward pass per micro-batch).
"""Teacher-in-actor forward for ``reverse_kl_student_topk``.

Each actor rank lazily loads the full teacher (bf16, sdpa, frozen) onto its GPU the first time the loss needs it and
keeps it resident. The teacher is run on the same padded ``input_ids``/``attention_mask``/``position_ids`` as the
student, so its logits line up position-for-position with the student's (softcapping is applied by the HF model itself).
Memory: Gemma 4 E4B is ~17 GB bf16 per rank plus a (bsz, seqlen, vocab) bf16 logits tensor (~2 GB at 4096 tokens).
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TEACHER: dict[str, torch.nn.Module] = {}


def get_in_actor_teacher(model_path: str, device: torch.device) -> torch.nn.Module:
    key = f"{model_path}@{device}"
    model = _TEACHER.get(key)
    if model is None:
        from transformers import AutoConfig

        from verl.utils.model import get_hf_auto_model_class

        config = AutoConfig.from_pretrained(model_path, attn_implementation="sdpa")
        model = get_hf_auto_model_class(config).from_pretrained(
            model_path, config=config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        model.to(device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
        print(
            f"TEACHER_IN_ACTOR_LOADED path={model_path} class={type(model).__name__} softcap={softcap} device={device}",
            flush=True,
        )
        _TEACHER[key] = model
    return model


@torch.no_grad()
def teacher_logits_padded(
    model_path: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Return (bsz, seqlen, vocab) bf16 logits of the frozen teacher for padded inputs."""
    model = get_in_actor_teacher(model_path, input_ids.device)
    output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
    logits = output.logits
    if logits.dtype != torch.bfloat16:
        logits = logits.to(torch.bfloat16)
    return logits
