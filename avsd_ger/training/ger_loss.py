"""Teacher-forced GER cross-entropy (spec Section 5.3 / 7, Stage 2).

In Stage 2 we fine-tune the full stack including LLM LoRA on a
GER-style correction objective:

    prompt = [<[Speaker: ID_i]> Audio hyp: ... Visual hyp: ... <AV_CTX> ...]
    target = ground-truth transcript for this utterance/speaker

We teacher-force the target tokens against the LLM's next-token logits and
compute cross-entropy ONLY over the target span (the prompt's loss is
masked out). Token-level mean log-prob is also returned so callers can
monitor training without extra passes.

This module does NOT own the GER head's LoRA weights or the Q-Former --
it calls into ``GERHead`` using its prompt assembly so training and
inference paths share the exact same prefix construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GERLossReport:
    loss: torch.Tensor
    mean_token_lp: float
    n_target_tokens: int


class GERCrossEntropy(nn.Module):
    """Wraps an existing :class:`GERHead` with a teacher-forcing loss.

    Usage::

        loss_fn = GERCrossEntropy(ger_head)
        rep = loss_fn(z_id, f_align, nbest, lip_hyp, target="hello world")
        rep.loss.backward()
    """

    def __init__(self, ger_head: nn.Module):
        super().__init__()
        self.ger = ger_head

    def forward(
        self,
        z_id: torch.Tensor,
        f_align: torch.Tensor,
        nbest: list[str],
        lip_hyp: str,
        target: str,
        speaker_id: str | None = None,
    ) -> GERLossReport:
        if self.ger.stub:
            # Stub mode: fabricate a deterministic loss for wiring tests.
            fake = torch.tensor(0.5 + 0.01 * len(target), requires_grad=True)
            return GERLossReport(loss=fake, mean_token_lp=-0.5, n_target_tokens=max(1, len(target.split())))

        tok = self.ger._tok
        text = self.ger._render_text(speaker_id, nbest, lip_hyp)
        prompt_embeds = self.ger._inputs_embeds(z_id, f_align, text)   # [1, P, H]

        # Tokenise target and build target embeddings + labels.
        tgt_ids = tok(target, return_tensors="pt", add_special_tokens=False).input_ids.to(prompt_embeds.device)
        emb = self.ger._llm.get_input_embeddings()
        tgt_embeds = emb(tgt_ids).to(prompt_embeds.dtype)              # [1, T, H]
        full = torch.cat([prompt_embeds, tgt_embeds], dim=1)           # [1, P+T, H]

        # Labels: -100 for prompt positions, tgt_ids for target positions,
        # plus the standard "predict next token" off-by-one.
        P = prompt_embeds.shape[1]
        T = tgt_ids.shape[1]
        labels = torch.full((1, P + T), -100, dtype=torch.long, device=prompt_embeds.device)
        labels[0, P : P + T] = tgt_ids[0]

        out = self.ger._llm(inputs_embeds=full)
        logits = out.logits[:, :-1, :]            # predict positions 1..N
        labels = labels[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )
        with torch.no_grad():
            mask = (labels != -100)
            if mask.any():
                logp = F.log_softmax(logits, dim=-1)
                gathered = logp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                mean_lp = float(gathered[mask].mean().item())
            else:
                mean_lp = 0.0
        return GERLossReport(loss=loss, mean_token_lp=mean_lp, n_target_tokens=T)
