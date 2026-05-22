# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Online full-vocabulary KL loss for Gemma off-policy distillation."""

from __future__ import annotations

import gc
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from transformers import AutoConfig

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.model import get_hf_auto_model_class
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import masked_sum


def _rank0_print(message: str):
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    ):
        print(message, flush=True)


def _shift_response_mask_no_cross_sample(loss_mask: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Align token-level response mask to next-token prediction positions."""
    shifted = torch.zeros_like(loss_mask, dtype=torch.bool)
    for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True):
        if end - start > 1:
            shifted[start : end - 1] = loss_mask[start + 1 : end].to(torch.bool)
    return shifted


@dataclass
class FullVocabKLLoss:
    teacher_model_path: str
    temperature: float = 1.0
    chunk_size: int = 64
    top_k: int = 0
    teacher_dtype: str = "bfloat16"
    trust_remote_code: bool = False
    attn_implementation: str = "flash_attention_2"
    use_teacher_hidden_states: bool = True

    def __post_init__(self):
        self.teacher_model = None
        self.teacher_config = None
        self._teacher_vocab_size: Optional[int] = None
        self._printed_vocab_check = False

    def _ensure_teacher(self, device: torch.device):
        if self.teacher_model is not None:
            return

        dtype = PrecisionType.to_dtype(self.teacher_dtype)
        self.teacher_config = AutoConfig.from_pretrained(
            self.teacher_model_path,
            trust_remote_code=self.trust_remote_code,
            attn_implementation=self.attn_implementation,
        )
        auto_class = get_hf_auto_model_class(self.teacher_config)
        mode = f"teacher top-k {self.top_k}" if self.top_k > 0 else "full vocab"
        _rank0_print(
            f"[FullVocabKLLoss] loading teacher {self.teacher_model_path} "
            f"as {auto_class.__name__} on {device} ({dtype}); loss mode={mode}"
        )
        self.teacher_model = auto_class.from_pretrained(
            self.teacher_model_path,
            torch_dtype=dtype,
            config=self.teacher_config,
            trust_remote_code=self.trust_remote_code,
        )
        self.teacher_model.to(device)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        self._teacher_vocab_size = int(self.teacher_model.lm_head.out_features)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def _teacher_hidden_states(self, input_ids, attention_mask, position_ids):
        if not self.use_teacher_hidden_states or not hasattr(self.teacher_model, "model"):
            return None
        outputs = self.teacher_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs[0]

    def _teacher_logits_for_positions(self, input_ids, attention_mask, position_ids, batch_idx, seq_idx):
        hidden_states = self._teacher_hidden_states(input_ids, attention_mask, position_ids)
        if hidden_states is not None:
            active_hidden = hidden_states[batch_idx, seq_idx]
            return active_hidden

        outputs = self.teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits[batch_idx, seq_idx]

    @contextmanager
    def _lm_head_forward_context(self, lm_head):
        """Prepare a possibly FSDP2-sharded LM head for repeated chunked calls."""
        did_unshard = False
        full_weight = None
        full_bias = None

        if hasattr(lm_head, "unshard") and hasattr(lm_head, "reshard"):
            lm_head.unshard()
            did_unshard = True

        weight = getattr(lm_head, "weight", None)
        bias = getattr(lm_head, "bias", None)
        if isinstance(weight, DTensor):
            full_weight = weight.full_tensor()
            if isinstance(bias, DTensor):
                full_bias = bias.full_tensor()
            else:
                full_bias = bias

        try:
            yield full_weight, full_bias
        finally:
            if did_unshard:
                lm_head.reshard()

    def _apply_lm_head(self, lm_head, active_hidden, config, full_weight=None, full_bias=None):
        if full_weight is not None:
            logits = F.linear(active_hidden, full_weight, full_bias)
        else:
            logits = lm_head(active_hidden)
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()
        softcap = getattr(config, "final_logit_softcapping", None)
        if softcap is None and hasattr(config, "get_text_config"):
            softcap = getattr(config.get_text_config(), "final_logit_softcapping", None)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def _compute_token_kl(
        self,
        data: TensorDict,
        student_logits: Optional[torch.Tensor] = None,
        student_active_flat_idx: Optional[torch.Tensor] = None,
        student_hidden: Optional[torch.Tensor] = None,
        student_lm_head=None,
        student_config=None,
    ) -> dict[str, torch.Tensor]:
        if student_logits is None:
            assert student_hidden is not None and student_lm_head is not None and student_config is not None
            assert student_hidden.dim() == 3 and student_hidden.shape[0] == 1, student_hidden.shape
        else:
            assert student_logits.dim() == 3 and student_logits.shape[0] == 1, student_logits.shape
        sp_size = tu.get_non_tensor_data(data=data, key="sp_size", default=1)
        if sp_size != 1:
            raise NotImplementedError("FullVocabKLLoss currently requires ulysses_sequence_parallel_size=1")

        device = student_logits.device if student_logits is not None else student_hidden.device
        self._ensure_teacher(device)

        input_ids_nt = data["input_ids"]
        loss_mask_nt = data["loss_mask"]
        position_ids_nt = data["position_ids"]
        offsets = input_ids_nt.offsets()
        lengths = offsets.diff()
        total_flat_positions = int(input_ids_nt.values().shape[0])

        if student_logits is not None:
            student_vocab_size = int(student_logits.shape[-1])
            total_positions = (
                total_flat_positions if student_active_flat_idx is not None else int(student_logits.shape[1])
            )
        else:
            student_vocab_size = int(student_lm_head.out_features)
            total_positions = (
                total_flat_positions if student_active_flat_idx is not None else int(student_hidden.shape[1])
            )
        if self._teacher_vocab_size != student_vocab_size:
            raise ValueError(f"Teacher/student vocab mismatch: {self._teacher_vocab_size=} {student_vocab_size=}")
        if not self._printed_vocab_check:
            _rank0_print(f"[FullVocabKLLoss] teacher/student vocab size verified: {student_vocab_size}")
            self._printed_vocab_check = True

        if student_active_flat_idx is not None:
            active_flat_idx = student_active_flat_idx.to(device=device, dtype=torch.long)
        else:
            flat_loss_mask = loss_mask_nt.values()
            active_mask = _shift_response_mask_no_cross_sample(flat_loss_mask, offsets)
            active_flat_idx = active_mask.nonzero(as_tuple=True)[0]

        if active_flat_idx.numel() == 0:
            output = student_logits if student_logits is not None else student_hidden
            return {"full_vocab_kl": output.new_zeros((1, total_positions), dtype=torch.float32)}

        batch_idx = torch.searchsorted(offsets[1:], active_flat_idx, right=True)
        seq_idx = active_flat_idx - offsets[batch_idx]

        batch_size = input_ids_nt.shape[0]
        max_seq_len = int(lengths.max().item())
        pad_token_id = tu.get_non_tensor_data(data=data, key="pad_token_id", default=0)
        input_ids = torch.nested.to_padded_tensor(
            input_ids_nt, padding=pad_token_id, output_size=(batch_size, max_seq_len)
        )
        position_ids = torch.nested.to_padded_tensor(position_ids_nt, padding=0, output_size=(batch_size, max_seq_len))
        attention_mask = torch.arange(max_seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        with torch.no_grad():
            teacher_active = self._teacher_logits_for_positions(
                input_ids, attention_mask, position_ids, batch_idx, seq_idx
            )

        if student_logits is not None:
            student_logits_flat = student_logits.squeeze(0)
            if student_active_flat_idx is not None:
                if student_logits_flat.shape[0] != active_flat_idx.numel():
                    raise ValueError(
                        "Compacted student logits must have one row per active response token: "
                        f"{student_logits_flat.shape[0]=} {active_flat_idx.numel()=}"
                    )
                active_student = student_logits_flat
            else:
                active_student = student_logits_flat.index_select(0, active_flat_idx)
        else:
            student_hidden_flat = student_hidden.squeeze(0)
            if student_active_flat_idx is not None:
                if student_hidden_flat.shape[0] != active_flat_idx.numel():
                    raise ValueError(
                        "Compacted student hidden states must have one row per active response token: "
                        f"{student_hidden_flat.shape[0]=} {active_flat_idx.numel()=}"
                    )
                active_student = student_hidden_flat
            else:
                active_student = student_hidden_flat.index_select(0, active_flat_idx)
        kl_chunks = []
        temperature = float(self.temperature)
        scale = temperature * temperature
        chunk_size = int(tu.get_non_tensor_data(data=data, key="teacher_chunk_size_override", default=self.chunk_size))
        top_k = int(tu.get_non_tensor_data(data=data, key="teacher_top_k_override", default=self.top_k))
        if chunk_size <= 0:
            raise ValueError(f"teacher KL chunk size must be positive, got {chunk_size}")

        student_head_context = (
            self._lm_head_forward_context(student_lm_head) if student_logits is None else nullcontext((None, None))
        )
        with student_head_context as (student_lm_weight, student_lm_bias):
            for start in range(0, active_flat_idx.numel(), chunk_size):
                end = min(start + chunk_size, active_flat_idx.numel())
                student_chunk = active_student[start:end]
                if student_logits is None:
                    student_chunk = self._apply_lm_head(
                        student_lm_head,
                        student_chunk,
                        student_config,
                        full_weight=student_lm_weight,
                        full_bias=student_lm_bias,
                    )
                student_chunk = student_chunk.float() / temperature
                teacher_chunk = teacher_active[start:end]
                if teacher_chunk.dim() == 2 and teacher_chunk.shape[-1] != student_vocab_size:
                    with torch.no_grad():
                        teacher_chunk = self._apply_lm_head(
                            self.teacher_model.lm_head, teacher_chunk, self.teacher_model.config
                        )
                teacher_chunk = teacher_chunk.float() / temperature

                if top_k > 0:
                    teacher_log_denominator = torch.logsumexp(teacher_chunk, dim=-1, keepdim=True)
                    teacher_topk_logits, teacher_topk_ids = torch.topk(
                        teacher_chunk, k=min(top_k, teacher_chunk.shape[-1]), dim=-1
                    )
                    teacher_topk_log_probs = teacher_topk_logits - teacher_log_denominator
                    student_topk_logits = torch.gather(student_chunk, dim=-1, index=teacher_topk_ids)
                    student_topk_log_probs = student_topk_logits - torch.logsumexp(student_chunk, dim=-1, keepdim=True)
                    teacher_topk_probs = teacher_topk_log_probs.exp()
                    kl = (teacher_topk_probs * (teacher_topk_log_probs - student_topk_log_probs)).sum(dim=-1) * scale
                else:
                    teacher_log_probs = F.log_softmax(teacher_chunk, dim=-1)
                    student_log_probs = F.log_softmax(student_chunk, dim=-1)
                    kl = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(dim=-1) * scale
                kl_chunks.append(kl)

        active_kl = torch.cat(kl_chunks, dim=0)
        output = student_logits if student_logits is not None else student_hidden
        per_token_kl = output.new_zeros((total_positions,), dtype=torch.float32)
        per_token_kl = per_token_kl.scatter(0, active_flat_idx, active_kl)
        return {"full_vocab_kl": per_token_kl.unsqueeze(0)}

    def _reduce_loss(self, model_output, data: TensorDict, dp_group=None):
        del dp_group
        pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        if pad_mode != DatasetPadMode.NO_PADDING:
            raise NotImplementedError("FullVocabKLLoss currently supports only pad_mode=no_padding")

        dp_size = data["dp_size"]
        batch_num_tokens = data["batch_num_tokens"]
        per_token_kl = model_output["full_vocab_kl"].values()
        loss_mask = _shift_response_mask_no_cross_sample(data["loss_mask"].values(), data["loss_mask"].offsets())
        token_sum = masked_sum(per_token_kl, loss_mask)
        loss = token_sum / batch_num_tokens * dp_size

        with torch.no_grad():
            active_kl = per_token_kl[loss_mask]
            if active_kl.numel() == 0:
                zero = torch.zeros((), device=per_token_kl.device)
                metrics = {
                    "full_vocab_kl/token_sum": zero.item(),
                    "full_vocab_kl/mean": zero.item(),
                    "full_vocab_kl/max": zero.item(),
                    "full_vocab_kl/min": zero.item(),
                    "full_vocab_kl/active_tokens": 0,
                    "full_vocab_kl/top_k": int(
                        tu.get_non_tensor_data(data=data, key="teacher_top_k_override", default=self.top_k)
                    ),
                    "full_vocab_kl/chunk_size": int(
                        tu.get_non_tensor_data(data=data, key="teacher_chunk_size_override", default=self.chunk_size)
                    ),
                }
            else:
                metrics = {
                    "full_vocab_kl/token_sum": token_sum.detach().item(),
                    "full_vocab_kl/mean": active_kl.mean().detach().item(),
                    "full_vocab_kl/max": active_kl.max().detach().item(),
                    "full_vocab_kl/min": active_kl.min().detach().item(),
                    "full_vocab_kl/active_tokens": int(active_kl.numel()),
                    "full_vocab_kl/top_k": int(
                        tu.get_non_tensor_data(data=data, key="teacher_top_k_override", default=self.top_k)
                    ),
                    "full_vocab_kl/chunk_size": int(
                        tu.get_non_tensor_data(data=data, key="teacher_chunk_size_override", default=self.chunk_size)
                    ),
                }
        return loss, metrics

    def __call__(
        self,
        model_output=None,
        data: Optional[TensorDict] = None,
        dp_group=None,
        student_logits: Optional[torch.Tensor] = None,
        student_active_flat_idx: Optional[torch.Tensor] = None,
        student_hidden: Optional[torch.Tensor] = None,
        student_lm_head=None,
        student_config=None,
        **kwargs,
    ):
        del kwargs
        if student_logits is not None or student_hidden is not None:
            return self._compute_token_kl(
                data=data,
                student_logits=student_logits,
                student_active_flat_idx=student_active_flat_idx,
                student_hidden=student_hidden,
                student_lm_head=student_lm_head,
                student_config=student_config,
            )
        return self._reduce_loss(model_output=model_output, data=data, dp_group=dp_group)
