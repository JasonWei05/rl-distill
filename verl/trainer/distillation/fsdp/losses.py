# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def compute_reverse_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict:
    """Compute reverse KL distillation loss on teacher's top-k support.

    Reverse KL is KL(Q_student || P_teacher) = Σ_v Q(v) · (log Q(v) - log P(v)).
    Mode-seeking: the student is pushed to concentrate on teacher's peaks rather
    than spread mass over all teacher-supported tokens (the behavior of
    `forward_kl_topk`). This is the direction typically recommended for
    on-policy distillation (e.g., Agarwal et al. 2023, "On-Policy Distillation
    of Language Models").

    Support approximation: this computes a *partial sum* restricted to teacher's
    top-k indices: Σ_{v ∈ top-k(P)} Q(v) · (log Q(v) - log P(v)). It is NOT a
    renormalized exact reverse KL — student mass on tokens outside teacher's
    top-k contributes zero direct loss here. Softmax backprop still moves
    gradient toward teacher's support because Q(v) is a function of the full
    student logits.

    The partial sum can be negative early in training when student mass on
    teacher's top-k support (student_mass) is below teacher's mass (teacher_mass);
    callers clamp to 0 (mirrors the forward_kl_topk wrapper's treatment).

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd".

    Returns:
        dict with:
          - distillation_losses: (bsz, seqlen/sp_size) per-token partial-sum reverse KL
          - student_mass: (bsz, seqlen/sp_size) Σ_{v ∈ top-k(P)} Q(v)
          - teacher_mass: (bsz, seqlen/sp_size) Σ_{v ∈ top-k(P)} P(v)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)

    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)

    # kl_divergence(log_q, log_p) computes Σ exp(log_p) · (log_p - log_q).
    # For reverse KL, we want Σ Q(v) · (log Q(v) - log P(v)), so:
    #   log_p := student log-probs  (weight and "log Q" term)
    #   log_q := teacher log-probs  ("log P" term being subtracted)
    distillation_losses = kl_divergence(
        log_q=teacher_topk_log_probs, log_p=student_topk_log_probs
    )

    return {"distillation_losses": distillation_losses, "student_mass": student_mass, "teacher_mass": teacher_mass}


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    return {"distillation_losses": distillation_losses, "student_mass": student_mass, "teacher_mass": teacher_mass}


def compute_reverse_kl_student_topk(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    config: DistillationConfig,
) -> dict:
    """Reverse KL on the student's top-k support (rl-distill fork).

    Sum over v in top-k(student) of q(v) * (log q(v) - log p(v)), with q the student and p the teacher, both full-vocab
    softmaxes of the given logits (softcapped upstream). Unlike ``compute_reverse_kl_topk`` the support is chosen by the
    student, so mass the student moves anywhere is always inside the sum (>= 99 % of q by construction for k=128).

    Args:
        student_logits: (1, N, vocab) with grad, N packed positions.
        teacher_logits: (1, N, vocab), no grad, same positions/order as the student logits.
    Returns:
        distillation_losses, student_mass (student's top-k mass), teacher_mass (teacher's mass on the student's top-k),
        each (1, N).
    """
    k = int(config.distillation_loss.topk)
    assert student_logits.shape == teacher_logits.shape, (student_logits.shape, teacher_logits.shape)
    student_f = student_logits.float()
    student_lse = torch.logsumexp(student_f, dim=-1, keepdim=True)
    topk_ids = torch.topk(student_f.detach(), k=min(k, student_f.shape[-1]), dim=-1).indices
    student_topk_log_probs = torch.gather(student_f, dim=-1, index=topk_ids) - student_lse
    with torch.no_grad():
        teacher_f = teacher_logits.float()
        teacher_lse = torch.logsumexp(teacher_f, dim=-1, keepdim=True)
        teacher_topk_log_probs = torch.gather(teacher_f, dim=-1, index=topk_ids) - teacher_lse
        del teacher_f
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    q = student_topk_log_probs.exp()
    distillation_losses = (q * (student_topk_log_probs - teacher_topk_log_probs)).sum(dim=-1)
    student_mass = q.sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    return {"distillation_losses": distillation_losses, "student_mass": student_mass, "teacher_mass": teacher_mass}


def compute_reverse_kl_student_topk_padded(
    student_logits: torch.Tensor,
    seq_lengths: torch.Tensor,
    softcap: float | None,
    teacher_hidden: torch.Tensor,
    teacher_head,
    topk: int,
    chunk_rows: int = 1024,
    tail_bucket: bool = False,
) -> dict:
    """Memory-lean ``reverse_kl_student_topk`` on padded (bsz, seqlen, vocab) student logits (rl-distill fork).

    ``tail_bucket=True`` adds (1-Q_k)(log(1-Q_k) - log(1-P_k)) per position (the ``reverse_kl_student_topk_bucket`` mode):
    the KL between the (k+1)-bucket distributions, a proper lower bound of the full reverse KL that penalises the student
    for putting more mass outside its top-k than the teacher does there.

    Works sample by sample and chunk by chunk under activation checkpointing, so the peak extra memory is one chunk's
    fp32 student and teacher logits (~1 GB each at 1024 rows x 262k vocab) instead of full-vocab fp32 copies of the whole
    micro-batch (which OOMed at 6 GB per copy on 2026-09-13). The teacher is applied through its hidden states and LM
    head per chunk (``teacher_head``), never as a full logits tensor. Returns (total_nnz,) tensors in cu_seqlens order.

    Args:
        student_logits: (bsz, seqlen, vocab) model logits (raw if ``softcap`` is given, otherwise already capped).
        seq_lengths: (bsz,) real lengths.
        softcap: final-logit softcap to apply to the student logits, or None.
        teacher_hidden: (bsz, seqlen, H) teacher hidden states aligned with the student positions.
        teacher_head: callable (rows, H) -> (rows, vocab) fp32 softcapped teacher logits.
    """
    from torch.utils.checkpoint import checkpoint

    k = int(topk)

    def chunk_fn(s_chunk: torch.Tensor, h_chunk: torch.Tensor):
        s = s_chunk.float()
        if softcap:
            s = torch.tanh(s / float(softcap)) * float(softcap)
        s_lse = torch.logsumexp(s, dim=-1, keepdim=True)
        ids = torch.topk(s.detach(), k=min(k, s.shape[-1]), dim=-1).indices
        s_lp = torch.gather(s, dim=-1, index=ids) - s_lse
        with torch.no_grad():
            t = teacher_head(h_chunk)
            t_lp = torch.gather(t, dim=-1, index=ids) - torch.logsumexp(t, dim=-1, keepdim=True)
            del t
        q = s_lp.exp()
        q_mass = q.sum(dim=-1)
        p_mass = t_lp.exp().sum(dim=-1)
        loss = (q * (s_lp - t_lp)).sum(dim=-1)
        if tail_bucket:
            q_tail = (1.0 - q_mass).clamp_min(1e-6)
            p_tail = (1.0 - p_mass).clamp_min(1e-6)
            loss = loss + q_tail * (torch.log(q_tail) - torch.log(p_tail))
        return loss, q_mass, p_mass

    losses, smass, tmass = [], [], []
    lengths = seq_lengths.tolist()
    for j, n in enumerate(lengths):
        for start in range(0, n, chunk_rows):
            end = min(n, start + chunk_rows)
            l, sm, tm = checkpoint(
                chunk_fn, student_logits[j, start:end], teacher_hidden[j, start:end], use_reentrant=False
            )
            losses.append(l)
            smass.append(sm)
            tmass.append(tm)
    return {
        "distillation_losses": torch.cat(losses),
        "student_mass": torch.cat(smass),
        "teacher_mass": torch.cat(tmass),
    }
