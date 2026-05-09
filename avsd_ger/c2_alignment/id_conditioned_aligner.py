"""C2 -- ID-conditioned cross-modal alignment (spec-aligned).

Differences from the earlier FiLM version:

* **ID injection = Concat + Linear** (spec section 2 C2, section 5.9 id_injection=concat_linear).
  FiLM is relegated to an Appendix-Table-1 ablation variant.

      Q = Linear( Concat(f_a_token, e_id) )        # [N_tok, D]
      K = V = f_v_segment                          # only frames assigned to speaker i

* **Token-level granularity** (spec section 2 C2, section 5.9 align_granularity=token_level).
  Upstream code pools the Whisper encoder sequence into per-token vectors using
  word-level timestamps. This module operates on those token vectors, not on
  frame-level features.

* **Soft gating** (spec section 2 C2, section 5.9 low_conf_handling=soft_gating):

      attn_weight *= min( SNR_score_per_token , lip_conf_per_frame ).

  Implemented as an additive log-bias on raw attention logits so gradients flow.

* **Per-speaker key/value masking**: VSR frames not assigned to speaker i are
  excluded via a `key_padding_mask`. This is the structural reason the aligner
  can't hallucinate lip evidence from other speakers.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatLinearInject(nn.Module):
    """Q = Linear( Concat(f_a_token, e_id) ). Spec-mandated injection."""

    def __init__(self, d_token: int, d_id: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_token + d_id, d_out)

    def forward(self, f_tok: torch.Tensor, e_id: torch.Tensor) -> torch.Tensor:
        # f_tok: [N, D_tok]  e_id: [D_id]  ->  [N, D_out]
        if e_id.ndim == 1:
            e_id = e_id.unsqueeze(0).expand(f_tok.size(0), -1)
        return self.proj(torch.cat([f_tok, e_id], dim=-1))


class SoftGatedCrossAttention(nn.Module):
    """Single cross-attention block with additive log-bias soft gating."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_h = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,                    # [N, D]
        kv: torch.Tensor,                   # [M, D]
        key_padding_mask: torch.Tensor | None = None,   # [M] bool, True = MASK
        soft_gate: torch.Tensor | None = None,          # [N, M] in [0,1]
    ) -> torch.Tensor:
        N, D = q.shape
        M = kv.shape[0]
        qh = self.q_proj(q).view(N, self.h, self.d_h).transpose(0, 1)  # [h,N,d]
        kh = self.k_proj(kv).view(M, self.h, self.d_h).transpose(0, 1)  # [h,M,d]
        vh = self.v_proj(kv).view(M, self.h, self.d_h).transpose(0, 1)

        logits = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(self.d_h)  # [h,N,M]

        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask.view(1, 1, M), float("-inf"))

        if soft_gate is not None:
            # additive log-bias: attn_weight *= gate  =>  logits += log(gate + eps)
            logits = logits + torch.log(soft_gate.clamp_min(1e-4)).unsqueeze(0)

        attn = torch.softmax(logits, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, vh)                                      # [h,N,d]
        out = out.transpose(0, 1).contiguous().view(N, D)
        return self.out(out)


class IDConditionedAligner(nn.Module):
    """
    Inputs at inference time (per-speaker):
        asr_tok_feats : [N_tok, D_asr]   pooled Whisper encoder features, one per word/token
        vsr_feats     : [T_v,   D_vsr]   AV-HuBERT features (whole clip)
        e_id          : [D_id]           this speaker's fused identity vector
        speaker_mask_v: [T_v]  bool      True where speaker-i is active
        snr_per_tok   : [N_tok]          optional, per-token SNR in [0,1]
        lip_conf_v    : [T_v]            optional, per-frame lip-detector conf in [0,1]

    Output:
        f_align : [N_tok, d_model]
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        z_dim: int,
        d_asr: int = 1280,
        d_vsr: int = 1024,
    ):
        super().__init__()
        d = int(cfg["d_model"])
        self.d = d
        self.use_soft_gate = cfg.get("low_conf_handling", "soft_gating") == "soft_gating"
        assert cfg.get("id_injection", "concat_linear") == "concat_linear", (
            "Spec mandates concat_linear; FiLM lives in ablations only."
        )

        self.inject_q = ConcatLinearInject(d_asr, z_dim, d)
        self.proj_kv = nn.Linear(d_vsr, d)

        self.blocks = nn.ModuleList([
            SoftGatedCrossAttention(d, int(cfg["n_heads"]), float(cfg["dropout"]))
            for _ in range(int(cfg["n_layers"]))
        ])
        self.ffs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, 4 * d),
                nn.GELU(),
                nn.Dropout(float(cfg["dropout"])),
                nn.Linear(4 * d, d),
            )
            for _ in range(int(cfg["n_layers"]))
        ])
        self.q_norm = nn.LayerNorm(d)
        self.kv_norm = nn.LayerNorm(d)
        self.out_norm = nn.LayerNorm(d)

    def forward(
        self,
        asr_tok_feats: torch.Tensor,
        vsr_feats: torch.Tensor,
        e_id: torch.Tensor,
        speaker_mask_v: torch.Tensor | None = None,
        snr_per_tok: torch.Tensor | None = None,
        lip_conf_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.q_norm(self.inject_q(asr_tok_feats, e_id))         # [N,D]
        kv = self.kv_norm(self.proj_kv(vsr_feats))                  # [M,D]

        key_pad = None
        if speaker_mask_v is not None:
            key_pad = ~speaker_mask_v.to(q.device).bool()  # True = mask this frame

        soft_gate = None
        if self.use_soft_gate and (snr_per_tok is not None or lip_conf_v is not None):
            N, M = q.size(0), kv.size(0)
            s = (
                snr_per_tok.to(q.device)
                if snr_per_tok is not None
                else torch.ones(N, device=q.device)
            )
            v = (
                lip_conf_v.to(kv.device)
                if lip_conf_v is not None
                else torch.ones(M, device=kv.device)
            )
            # gate[n,m] = min(SNR(n), lip_conf(m))
            soft_gate = torch.minimum(s.view(N, 1), v.view(1, M))

        h = q
        for blk, ff in zip(self.blocks, self.ffs):
            h = h + blk(h, kv, key_padding_mask=key_pad, soft_gate=soft_gate)
            h = h + ff(h)
        return self.out_norm(h)
