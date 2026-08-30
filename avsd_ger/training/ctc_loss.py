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
  compressed. A learnable temporal expansion emits distinct character-scale
  subframes for every aligned token. The expansion factor is selected from
  the target's exact CTC feasibility requirement during training.
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
    input_lengths: torch.Tensor
    target_lengths: torch.Tensor
    minimum_steps: torch.Tensor
    expansion: int
    feasible: bool
    zero_loss_count: int
    nonfinite_count: int


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
        min_expansion: minimum number of learned subframes per aligned token.
        max_expansion: largest permitted expansion. Targets that cannot be
            represented at this limit fail closed instead of becoming a zero
            loss through ``zero_infinity=True``.
    """

    def __init__(
        self,
        d_align: int,
        vocab_size: int | None = None,
        min_expansion: int = 8,
        max_expansion: int = 16,
    ):
        super().__init__()
        self.vocab = CharVocab()
        vsz = vocab_size or len(self.vocab)
        self.min_expansion = int(min_expansion)
        self.max_expansion = int(max_expansion)
        if self.min_expansion <= 0 or self.max_expansion < self.min_expansion:
            raise ValueError(
                "CTC expansion requires 0 < min_expansion <= max_expansion"
            )
        self.d_align = int(d_align)
        self.temporal_expand = nn.Sequential(
            nn.LayerNorm(d_align),
            nn.Linear(d_align, self.max_expansion * d_align),
            nn.GELU(),
        )
        self.subframe_position = nn.Parameter(
            torch.zeros(self.max_expansion, d_align)
        )
        nn.init.normal_(self.subframe_position, std=0.02)
        self.proj = nn.Linear(d_align, vsz)

    @staticmethod
    def minimum_ctc_steps(ids: list[int]) -> int:
        """Exact minimum time steps required by CTC for one target."""
        repeats = sum(left == right for left, right in zip(ids, ids[1:]))
        return len(ids) + repeats

    def _select_expansion(self, n_tokens: int, minimum_steps: list[int]) -> int:
        if n_tokens <= 0:
            raise ValueError("CTC received an empty aligned sequence")
        required = max(minimum_steps, default=0)
        expansion = max(
            self.min_expansion,
            (required + n_tokens - 1) // n_tokens,
        )
        if expansion > self.max_expansion:
            raise ValueError(
                "CTC target is infeasible: "
                f"minimum_steps={required}, aligned_tokens={n_tokens}, "
                f"required_expansion={expansion}, max_expansion={self.max_expansion}"
            )
        return expansion

    def forward(
        self,
        f_align: torch.Tensor,                 # [B, N, D] or [N, D]
        targets: list[str] | None = None,
    ) -> CTCReport:
        if f_align.ndim == 2:
            f_align = f_align.unsqueeze(0)
        B, N, D = f_align.shape
        if D != self.d_align:
            raise ValueError(
                f"CTC expected aligned dim {self.d_align}, received {D}"
            )

        tgt_ids = [self.vocab.encode(t) for t in targets] if targets is not None else []
        if targets is not None and len(tgt_ids) != B:
            raise ValueError(
                f"CTC target batch size {len(tgt_ids)} does not match input batch {B}"
            )
        if targets is not None and any(len(ids) == 0 for ids in tgt_ids):
            raise ValueError("CTC target becomes empty after character normalization")
        minimum_steps_list = [self.minimum_ctc_steps(ids) for ids in tgt_ids]
        expansion = self._select_expansion(N, minimum_steps_list)

        # Every subframe has a distinct learned transformation and positional
        # offset. This supplies real temporal degrees of freedom; unlike
        # identical repetition, adjacent characters can receive different
        # logits and gradients.
        x = self.temporal_expand(f_align).view(
            B, N, self.max_expansion, D
        )[:, :, :expansion, :]
        x = x + self.subframe_position[:expansion].view(1, 1, expansion, D)
        x = x.reshape(B, N * expansion, D)
        logits = self.proj(x)                                          # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1)

        input_lens = torch.full(
            (B,), log_probs.shape[1], dtype=torch.long, device=f_align.device
        )

        if targets is None:
            return CTCReport(
                loss=torch.zeros((), device=f_align.device),
                log_probs=log_probs,
                mean_lp=0.0,
                input_lengths=input_lens,
                target_lengths=torch.zeros(B, dtype=torch.long, device=f_align.device),
                minimum_steps=torch.zeros(B, dtype=torch.long, device=f_align.device),
                expansion=expansion,
                feasible=True,
                zero_loss_count=0,
                nonfinite_count=0,
            )

        # Build CTC target tensors
        tgt_lens = torch.tensor([len(t) for t in tgt_ids], dtype=torch.long, device=f_align.device)
        minimum_steps = torch.tensor(
            minimum_steps_list, dtype=torch.long, device=f_align.device
        )
        feasible_mask = input_lens >= minimum_steps
        if not bool(feasible_mask.all().item()):
            raise ValueError(
                "CTC feasibility accounting failed after temporal expansion: "
                f"input_lengths={input_lens.tolist()}, "
                f"minimum_steps={minimum_steps.tolist()}"
            )
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
            zero_infinity=False,
        )
        nonfinite_count = int((~torch.isfinite(loss.detach())).sum().item())
        if nonfinite_count:
            raise FloatingPointError(
                "Non-finite CTC loss despite feasible length accounting: "
                f"input_lengths={input_lens.tolist()}, "
                f"target_lengths={tgt_lens.tolist()}, "
                f"minimum_steps={minimum_steps.tolist()}"
            )
        zero_loss_count = int((loss.detach() == 0).sum().item())
        return CTCReport(
            loss=loss,
            log_probs=log_probs,
            mean_lp=float(log_probs.max(dim=-1).values.mean().item()),
            input_lengths=input_lens,
            target_lengths=tgt_lens,
            minimum_steps=minimum_steps,
            expansion=expansion,
            feasible=True,
            zero_loss_count=zero_loss_count,
            nonfinite_count=nonfinite_count,
        )

    @torch.no_grad()
    def greedy_decode(self, f_align: torch.Tensor) -> list[str]:
        rep = self.forward(f_align)
        ids = rep.log_probs.argmax(dim=-1).cpu().tolist()
        return [self.vocab.decode(row) for row in ids]
