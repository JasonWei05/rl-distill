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

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint
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
    teacher_model_path: str | None = None
    temperature: float = 1.0
    chunk_size: int = 64
    top_k: int = 0
    teacher_dtype: str = "bfloat16"
    trust_remote_code: bool = False
    attn_implementation: str = "flash_attention_2"
    use_teacher_hidden_states: bool = True
    precomputed_topk: bool = False
    clamp_min_kl: bool = False
    checkpoint_student_chunks: bool = True

    def __post_init__(self):
        if self.precomputed_topk:
            if self.top_k <= 0:
                raise ValueError("precomputed top-k distillation requires top_k > 0")
            if float(self.temperature) != 1.0:
                raise ValueError("precomputed normalized teacher log probabilities require temperature=1.0")
        elif not self.teacher_model_path:
            raise ValueError("teacher_model_path is required unless precomputed_topk=True")
        self.teacher_model = None
        self.teacher_config = None
        self._teacher_vocab_size: int | None = None
        self._printed_vocab_check = False

    def _ensure_teacher(self, device: torch.device):
        if self.teacher_model is not None:
            return
        if self.precomputed_topk:
            raise RuntimeError("online teacher loading is disabled for precomputed top-k distillation")

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

    def _apply_lm_head(
        self,
        lm_head,
        active_hidden,
        config,
        full_weight=None,
        full_bias=None,
        logit_softcap_override: float | None = None,
    ):
        if full_weight is not None:
            logits = F.linear(active_hidden, full_weight, full_bias)
        else:
            logits = lm_head(active_hidden)
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()
        softcap = logit_softcap_override
        if softcap is None:
            softcap = getattr(config, "final_logit_softcapping", None)
            if softcap is None and hasattr(config, "get_text_config"):
                softcap = getattr(config.get_text_config(), "final_logit_softcapping", None)
        if softcap is not None:
            if float(softcap) <= 0:
                raise ValueError(f"final_logit_softcapping must be positive, got {softcap}")
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def _precomputed_teacher_topk(
        self,
        data: TensorDict,
        *,
        active_mask: torch.Tensor,
        offsets: torch.Tensor,
        student_vocab_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids_key = "teacher_topk_token_ids"
        logprobs_key = "teacher_topk_logprobs"
        if ids_key not in data or logprobs_key not in data:
            raise KeyError("precomputed top-k data must contain teacher_topk_token_ids and teacher_topk_logprobs")

        teacher_ids_nt = data[ids_key]
        teacher_logprobs_nt = data[logprobs_key]
        if not teacher_ids_nt.is_nested or not teacher_logprobs_nt.is_nested:
            raise TypeError("precomputed teacher top-k tensors must be no-padding NestedTensors")

        top_k = int(self.top_k)
        response_counts = torch.stack(
            [active_mask[start:end].sum() for start, end in zip(offsets[:-1], offsets[1:], strict=True)]
        ).to(dtype=torch.long)
        expected_flat_lengths = response_counts * top_k
        ids_flat_lengths = teacher_ids_nt.offsets().diff().to(expected_flat_lengths.device)
        logprobs_flat_lengths = teacher_logprobs_nt.offsets().diff().to(expected_flat_lengths.device)
        if not torch.equal(ids_flat_lengths, expected_flat_lengths):
            raise ValueError(
                "teacher_topk_token_ids length mismatch: "
                f"expected {expected_flat_lengths.tolist()}, got {ids_flat_lengths.tolist()}"
            )
        if not torch.equal(logprobs_flat_lengths, expected_flat_lengths):
            raise ValueError(
                "teacher_topk_logprobs length mismatch: "
                f"expected {expected_flat_lengths.tolist()}, got {logprobs_flat_lengths.tolist()}"
            )

        teacher_ids = teacher_ids_nt.values().to(device=device, dtype=torch.long).reshape(-1, top_k)
        teacher_logprobs = teacher_logprobs_nt.values().to(device=device, dtype=torch.float32).reshape(-1, top_k)
        if teacher_ids.shape[0] != int(active_mask.sum().item()):
            raise ValueError(
                "precomputed teacher row count does not match active response predictions: "
                f"{teacher_ids.shape[0]} vs {int(active_mask.sum().item())}"
            )
        if teacher_ids.numel():
            min_teacher_id = int(teacher_ids.min().item())
            max_teacher_id = int(teacher_ids.max().item())
            if min_teacher_id < 0 or max_teacher_id >= student_vocab_size:
                raise ValueError(f"precomputed teacher token ids must be in [0, {student_vocab_size})")
        if not torch.isfinite(teacher_logprobs).all():
            raise ValueError("precomputed teacher top-k log probabilities contain non-finite values")
        return teacher_ids, teacher_logprobs

    def _compute_token_kl(
        self,
        data: TensorDict,
        student_logits: torch.Tensor | None = None,
        student_active_flat_idx: torch.Tensor | None = None,
        student_hidden: torch.Tensor | None = None,
        student_lm_head=None,
        student_config=None,
        student_logit_softcap: float | None = None,
    ) -> dict[str, torch.Tensor]:
        if student_logits is None:
            if student_hidden is None or student_lm_head is None or student_config is None:
                raise ValueError("student hidden-state distillation requires hidden states, LM head, and config")
            if student_hidden.dim() != 3 or student_hidden.shape[0] != 1:
                raise ValueError(
                    "hidden-state distillation currently requires one sequence per micro-batch; "
                    f"got student_hidden.shape={tuple(student_hidden.shape)}"
                )
        else:
            if student_logits.dim() != 3 or student_logits.shape[0] != 1:
                raise ValueError(
                    "logit distillation currently requires one sequence per micro-batch; "
                    f"got student_logits.shape={tuple(student_logits.shape)}"
                )
        sp_size = tu.get_non_tensor_data(data=data, key="sp_size", default=1)
        if sp_size != 1:
            raise NotImplementedError("FullVocabKLLoss currently requires ulysses_sequence_parallel_size=1")

        device = student_logits.device if student_logits is not None else student_hidden.device
        if not self.precomputed_topk:
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
        if not self.precomputed_topk and self._teacher_vocab_size != student_vocab_size:
            raise ValueError(f"Teacher/student vocab mismatch: {self._teacher_vocab_size=} {student_vocab_size=}")
        if not self._printed_vocab_check:
            source = "precomputed teacher token ids" if self.precomputed_topk else "online teacher"
            _rank0_print(f"[FullVocabKLLoss] {source}/student vocab size verified: {student_vocab_size}")
            self._printed_vocab_check = True

        flat_loss_mask = loss_mask_nt.values()
        active_mask = _shift_response_mask_no_cross_sample(flat_loss_mask, offsets)
        if student_active_flat_idx is not None:
            active_flat_idx = student_active_flat_idx.to(device=device, dtype=torch.long)
            expected_active_flat_idx = active_mask.nonzero(as_tuple=True)[0]
            if not torch.equal(active_flat_idx, expected_active_flat_idx):
                raise ValueError("student_active_flat_idx must exactly match shifted response prediction positions")
        else:
            active_flat_idx = active_mask.nonzero(as_tuple=True)[0]

        if active_flat_idx.numel() == 0:
            output = student_logits if student_logits is not None else student_hidden
            zeros = output.new_zeros((1, total_positions), dtype=torch.float32)
            return {
                "full_vocab_kl": zeros,
                "teacher_topk_mass": zeros.clone(),
                "student_topk_mass": zeros.clone(),
            }

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

        teacher_topk_ids = None
        teacher_topk_log_probs = None
        if self.precomputed_topk:
            teacher_topk_ids, teacher_topk_log_probs = self._precomputed_teacher_topk(
                data,
                active_mask=active_mask,
                offsets=offsets,
                student_vocab_size=student_vocab_size,
                device=device,
            )
            teacher_active = None
        else:
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
        teacher_mass_chunks = []
        student_mass_chunks = []
        temperature = float(self.temperature)
        scale = temperature * temperature
        chunk_size = int(tu.get_non_tensor_data(data=data, key="teacher_chunk_size_override", default=self.chunk_size))
        top_k = int(tu.get_non_tensor_data(data=data, key="teacher_top_k_override", default=self.top_k))
        if chunk_size <= 0:
            raise ValueError(f"teacher KL chunk size must be positive, got {chunk_size}")
        if self.precomputed_topk and top_k != self.top_k:
            raise ValueError(
                "teacher_top_k_override cannot change the width of precomputed targets: "
                f"stored={self.top_k}, override={top_k}"
            )

        student_head_context = (
            self._lm_head_forward_context(student_lm_head) if student_logits is None else nullcontext((None, None))
        )
        with student_head_context as (student_lm_weight, student_lm_bias):
            for start in range(0, active_flat_idx.numel(), chunk_size):
                end = min(start + chunk_size, active_flat_idx.numel())
                if self.precomputed_topk:
                    chunk_ids = teacher_topk_ids[start:end]
                    chunk_teacher_log_probs = teacher_topk_log_probs[start:end]
                else:
                    teacher_chunk = teacher_active[start:end]
                    if teacher_chunk.dim() == 2 and teacher_chunk.shape[-1] != student_vocab_size:
                        with torch.no_grad():
                            teacher_chunk = self._apply_lm_head(
                                self.teacher_model.lm_head, teacher_chunk, self.teacher_model.config
                            )
                    teacher_chunk = teacher_chunk.float() / temperature

                student_chunk_input = active_student[start:end]

                # Bind every per-chunk tensor as a default argument. The callable
                # is retained by activation checkpointing until backward, so a
                # normal closure over loop variables would incorrectly reuse the
                # final chunk during recomputation.
                def compute_chunk(
                    student_input,
                    *,
                    chunk_ids=chunk_ids if self.precomputed_topk else None,
                    chunk_teacher_log_probs=chunk_teacher_log_probs if self.precomputed_topk else None,
                    teacher_chunk=teacher_chunk if not self.precomputed_topk else None,
                    student_lm_weight=student_lm_weight,
                    student_lm_bias=student_lm_bias,
                ):
                    student_chunk = student_input
                    if student_logits is None:
                        student_chunk = self._apply_lm_head(
                            student_lm_head,
                            student_chunk,
                            student_config,
                            full_weight=student_lm_weight,
                            full_bias=student_lm_bias,
                            logit_softcap_override=student_logit_softcap,
                        )
                    student_chunk = student_chunk.float() / temperature

                    if self.precomputed_topk:
                        student_topk_logits = torch.gather(student_chunk, dim=-1, index=chunk_ids)
                        student_topk_log_probs = student_topk_logits - torch.logsumexp(
                            student_chunk, dim=-1, keepdim=True
                        )
                        teacher_topk_probs = chunk_teacher_log_probs.exp()
                        chunk_kl = (teacher_topk_probs * (chunk_teacher_log_probs - student_topk_log_probs)).sum(
                            dim=-1
                        ) * scale
                        chunk_teacher_mass = teacher_topk_probs.sum(dim=-1)
                        chunk_student_mass = student_topk_log_probs.exp().sum(dim=-1)
                    elif top_k > 0:
                        teacher_log_denominator = torch.logsumexp(teacher_chunk, dim=-1, keepdim=True)
                        chunk_teacher_topk_logits, chunk_teacher_topk_ids = torch.topk(
                            teacher_chunk, k=min(top_k, teacher_chunk.shape[-1]), dim=-1
                        )
                        online_teacher_log_probs = chunk_teacher_topk_logits - teacher_log_denominator
                        student_topk_logits = torch.gather(student_chunk, dim=-1, index=chunk_teacher_topk_ids)
                        student_topk_log_probs = student_topk_logits - torch.logsumexp(
                            student_chunk, dim=-1, keepdim=True
                        )
                        teacher_topk_probs = online_teacher_log_probs.exp()
                        chunk_kl = (teacher_topk_probs * (online_teacher_log_probs - student_topk_log_probs)).sum(
                            dim=-1
                        ) * scale
                        chunk_teacher_mass = teacher_topk_probs.sum(dim=-1)
                        chunk_student_mass = student_topk_log_probs.exp().sum(dim=-1)
                    else:
                        teacher_log_probs = F.log_softmax(teacher_chunk, dim=-1)
                        student_log_probs = F.log_softmax(student_chunk, dim=-1)
                        chunk_kl = (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(
                            dim=-1
                        ) * scale
                        chunk_teacher_mass = torch.ones_like(chunk_kl)
                        chunk_student_mass = torch.ones_like(chunk_kl)

                    if self.clamp_min_kl:
                        chunk_kl = chunk_kl.clamp_min(0.0)
                    return chunk_kl, chunk_teacher_mass.detach(), chunk_student_mass.detach()

                should_checkpoint = (
                    self.checkpoint_student_chunks
                    and self.precomputed_topk
                    and student_logits is None
                    and torch.is_grad_enabled()
                    and student_chunk_input.requires_grad
                )
                if should_checkpoint:
                    kl, teacher_mass, student_mass = checkpoint(
                        compute_chunk,
                        student_chunk_input,
                        use_reentrant=False,
                    )
                else:
                    kl, teacher_mass, student_mass = compute_chunk(student_chunk_input)
                kl_chunks.append(kl)
                teacher_mass_chunks.append(teacher_mass)
                student_mass_chunks.append(student_mass)

        active_kl = torch.cat(kl_chunks, dim=0)
        active_teacher_mass = torch.cat(teacher_mass_chunks, dim=0)
        active_student_mass = torch.cat(student_mass_chunks, dim=0)
        output = student_logits if student_logits is not None else student_hidden
        per_token_kl = output.new_zeros((total_positions,), dtype=torch.float32)
        per_token_kl = per_token_kl.scatter(0, active_flat_idx, active_kl)
        per_token_teacher_mass = output.new_zeros((total_positions,), dtype=torch.float32)
        per_token_teacher_mass = per_token_teacher_mass.scatter(0, active_flat_idx, active_teacher_mass)
        per_token_student_mass = output.new_zeros((total_positions,), dtype=torch.float32)
        per_token_student_mass = per_token_student_mass.scatter(0, active_flat_idx, active_student_mass)
        return {
            "full_vocab_kl": per_token_kl.unsqueeze(0),
            "teacher_topk_mass": per_token_teacher_mass.unsqueeze(0),
            "student_topk_mass": per_token_student_mass.unsqueeze(0),
        }

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
                    "full_vocab_kl/teacher_mass_sum": zero.item(),
                    "full_vocab_kl/student_mass_sum": zero.item(),
                    "full_vocab_kl/top_k": int(
                        tu.get_non_tensor_data(data=data, key="teacher_top_k_override", default=self.top_k)
                    ),
                    "full_vocab_kl/chunk_size": int(
                        tu.get_non_tensor_data(data=data, key="teacher_chunk_size_override", default=self.chunk_size)
                    ),
                }
            else:
                active_teacher_mass = model_output["teacher_topk_mass"].values()[loss_mask]
                active_student_mass = model_output["student_topk_mass"].values()[loss_mask]
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
                    "full_vocab_kl/teacher_mass": active_teacher_mass.mean().detach().item(),
                    "full_vocab_kl/teacher_mass_sum": active_teacher_mass.sum().detach().item(),
                    "full_vocab_kl/teacher_mass/min": active_teacher_mass.min().detach().item(),
                    "full_vocab_kl/teacher_mass/max": active_teacher_mass.max().detach().item(),
                    "full_vocab_kl/student_mass": active_student_mass.mean().detach().item(),
                    "full_vocab_kl/student_mass_sum": active_student_mass.sum().detach().item(),
                    "full_vocab_kl/student_mass/min": active_student_mass.min().detach().item(),
                    "full_vocab_kl/student_mass/max": active_student_mass.max().detach().item(),
                }
        return loss, metrics

    def __call__(
        self,
        model_output=None,
        data: TensorDict | None = None,
        dp_group=None,
        student_logits: torch.Tensor | None = None,
        student_active_flat_idx: torch.Tensor | None = None,
        student_hidden: torch.Tensor | None = None,
        student_lm_head=None,
        student_config=None,
        student_logit_softcap: float | None = None,
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
                student_logit_softcap=student_logit_softcap,
            )
        return self._reduce_loss(model_output=model_output, data=data, dp_group=dp_group)
