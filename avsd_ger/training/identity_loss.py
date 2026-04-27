"""Bidirectional InfoNCE for C1 identity learning (spec §7 training).

Objective
---------
Given a batch of N utterances, each producing:

    a_i  : pooled *audio-side* feature   (from Whisper encoder + ID projection),
    v_i  : pooled *visual-side* feature  (from AV-HuBERT + ID projection),

we optimise both cross-modal directions of InfoNCE:

    L_{A→V} = -1/N Σ_i  log  exp(sim(a_i, v_i)/τ)  / Σ_j exp(sim(a_i, v_j)/τ)
    L_{V→A} = -1/N Σ_i  log  exp(sim(v_i, a_i)/τ)  / Σ_j exp(sim(v_i, a_j)/τ)
    L_total =  L_{A→V} + L_{V→A}

Spec note: bidirectionality is **non-optional** — the symmetric form is
what makes the fused identity vector equally discriminative whether the
query comes from voice (enrolment day) or face (meeting day). Running
only one direction biases the pool toward the dominant modality and
silently breaks cross-modal retrieval at inference.

Module contract
---------------
``BidirectionalInfoNCE`` is a plain ``nn.Module``: forward takes a pair
of L2-normalised batches ``(a, v)`` of shape ``[N, D]`` and returns a
scalar loss. It is independent of the encoder stack so you can slot it
into any Stage-1 trainer without refactoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InfoNCEReport:
    loss: torch.Tensor
    loss_av: torch.Tensor
    loss_va: torch.Tensor
    acc_av: float     # top-1 retrieval accuracy A→V
    acc_va: float     # top-1 retrieval accuracy V→A


def info_nce(
    query: torch.Tensor,
    key: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, float]:
    """One-direction InfoNCE.

    Args:
        query: [N, D] L2-normalised query embeddings.
        key:   [N, D] L2-normalised key embeddings aligned index-wise.
        temperature: softmax temperature τ.

    Returns:
        (loss_scalar, top1_acc).  Diagonal targets (i-th query → i-th key).
    """
    # Since both are L2-normalised, query @ key.T is the cosine-sim matrix.
    logits = query @ key.t() / temperature
    targets = torch.arange(query.size(0), device=query.device)
    loss = F.cross_entropy(logits, targets)
    with torch.no_grad():
        acc = float((logits.argmax(dim=-1) == targets).float().mean().item())
    return loss, acc


class BidirectionalInfoNCE(nn.Module):
    """L_total = L_{A→V} + L_{V→A}, both with shared temperature τ.

    The module does not own the projection heads — the caller is expected
    to pass already-projected, L2-normalised features. This keeps the loss
    reusable for both (pre-fuser) modality-specific projections and
    (post-fuser) fused-identity contrastive training.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__()
        cfg = cfg or {}
        self.temperature = float(cfg.get("temperature", 0.07))
        self.bidirectional = bool(cfg.get("bidirectional", True))

    def forward(self, a: torch.Tensor, v: torch.Tensor) -> InfoNCEReport:
        if a.ndim != 2 or v.ndim != 2 or a.shape != v.shape:
            raise ValueError(f"a, v must be [N,D] and equal shape, got {a.shape} vs {v.shape}")

        a = F.normalize(a, dim=-1)
        v = F.normalize(v, dim=-1)

        loss_av, acc_av = info_nce(a, v, self.temperature)
        if self.bidirectional:
            loss_va, acc_va = info_nce(v, a, self.temperature)
            total = loss_av + loss_va
        else:
            # Diagnostic-only; spec mandates bidirectional for production.
            loss_va = torch.zeros_like(loss_av)
            acc_va = 0.0
            total = loss_av

        return InfoNCEReport(
            loss=total, loss_av=loss_av, loss_va=loss_va,
            acc_av=acc_av, acc_va=acc_va,
        )
