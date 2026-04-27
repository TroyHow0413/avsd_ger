"""CTC loss head over aligned token-level features (spec Section 5.3 / 7).

In Stage 1 and Stage 2 we optimise a CTC head on top of `f_align` so the
aligner learns to produce features that are already transcribable. This
is the spec's "CTC on aligned features" objective -- it regularises the
cross-attention block and acts as a free sanity signal independent of
the LLM.

Design notes
------------
* Input is `f_align` of shape [N_tok, D] (or [B, N_tok, D]). Because we pool
  encoder frames into word-level tokens, the CTC sequence length is already
  compressed -- the output still needs blank tokens for CTC to work, so we
  **repeat** each token K times (``expansion``) before emitting logits. This
  is the cheapest way to get CTC to tolerate our short token sequences.
* Vocabulary is a simple character set (lower-case + space + apostrophe +
  blank). That's sufficient for AMI/LRS3 transcripts. Swap with a BPE
  tokenizer if needed; the loss code doesn't care.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CHARSET = " 'abcdefghijklmnopqrstuvwxyz"  # index 0 is blank by convention


@dataclass
class CTCReport:
    loss: torch.Tensor
    log_probs: torch.Tensor
    mean_lp: float


class CharVocab:
    """Tiny deterministic char vocab. Index 0 is reserved for the CTC blank."""

    def __init__(self, charset: str = DEFAULT_CHARSET):
        self.blank_id = 0
        self._chars = ["<blank>"] + list(charset)
        self._to_id = {c: i for i, c in enumerate(self._chars)}

    def __len__(self) -> int:
        return len(self._chars)

    def encode(self, text: str) -> list[int]:
        text = text.lower()
        return [self._to_id[c] for c in text if c in self._to_id]

    def decode(self, ids: list[int]) -> str:
        # Greedy CTC collapse: remove repeats + blanks.
        out, prev = [], None
        for i in ids:
            if i == self.blank_id:
                prev = None
                continue
            if i != prev:
                out.append(self._chars[i])
                prev = i
        return "".join(out)


class CTCHead(nn.Module):
    """f_align -> vocab logits; supports training loss + greedy decode.

    Args:
        d_align: feature dim of f_align.
        vocab_size: defaults to ``len(CharVocab())``.
        expansion: how many logit frames to emit per input token. CTC needs
            T >= 2L+1 ish for any target; 4x is a safe default for word-level
            pooled features.
    """

    def __init__(self, d_align: int, vocab_size: int | None = None, expansion: int = 4):
        super().__init__()
        self.vocab = CharVocab()
        vsz = vocab_size or len(self.vocab)
        self.expansion = int(expansion)
        self.proj = nn.Linear(d_align, vsz)

    def forward(
        self,
        f_align: torch.Tensor,                 # [B, N, D] or [N, D]
        targets: list[str] | None = None,
    ) -> CTCReport:
        if f_align.ndim == 2:
            f_align = f_align.unsqueeze(0)
        B, N, D = f_align.shape
        # Repeat each token `expansion` times along the time axis
        x = f_align.unsqueeze(2).expand(B, N, self.expansion, D).reshape(B, N * self.expansion, D)
        logits = self.proj(x)                                          # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1)

        if targets is None:
            return CTCReport(
                loss=torch.zeros((), device=f_align.device),
                log_probs=log_probs,
                mean_lp=0.0,
            )

        # Build CTC target tensors
        tgt_ids = [self.vocab.encode(t) for t in targets]
        tgt_lens = torch.tensor([len(t) for t in tgt_ids], dtype=torch.long, device=f_align.device)
        input_lens = torch.full((B,), log_probs.shape[1], dtype=torch.long, device=f_align.device)
        flat = torch.tensor(
            [c for t in tgt_ids for c in t], dtype=torch.long, device=f_align.device
        )
        # CTC expects time-first: [T, B, V]
        loss = F.ctc_loss(
            log_probs.transpose(0, 1),
            flat,
            input_lens,
            tgt_lens,
            blank=self.vocab.blank_id,
            reduction="mean",
            zero_infinity=True,
        )
        return CTCReport(
            loss=loss,
            log_probs=log_probs,
            mean_lp=float(log_probs.max(dim=-1).values.mean().item()),
        )

    @torch.no_grad()
    def greedy_decode(self, f_align: torch.Tensor) -> list[str]:
        rep = self.forward(f_align)
        ids = rep.log_probs.argmax(dim=-1).cpu().tolist()
        return [self.vocab.decode(row) for row in ids]
