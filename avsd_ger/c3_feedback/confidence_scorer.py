"""C3 -- composite confidence (spec-aligned).

Primary signal is **acoustic rescoring** (spec section 2 C3): the corrected
text y_hat is re-fed through the ASR acoustic model and the mean token
log-prob is the confidence score s_i. Secondary signals are kept as small
tiebreakers.

    total =   w_rescore  * sigma(s_i)
            + w_av       * av_consistency
            + w_nbest    * nbest_agreement
            + w_llm      * sigma(mean LLM token logprob)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from ..utils import squash_logprob


def _edit_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return 1.0 - dp[m][n] / max(m, n, 1)


@dataclass
class ConfidenceReport:
    total: float
    s_acoustic: float          # raw Whisper mean logprob (can be negative)
    s_acoustic_conf: float     # squashed to [0,1]
    av_consistency: float
    nbest_agreement: float
    llm_entropy_conf: float
    components: dict[str, float]


class ConfidenceScorer:
    """
    Inputs expected per call:
      * asr_rescore_logprob : float  (mean token logprob of y_hat under Whisper)
      * av_consistency      : float  (cosine of z_query vs matched z_id)
      * nbest               : list[str]
      * token_logprobs      : Tensor (LLM per-token logprobs for y_hat)
    """

    def __init__(self, cfg: dict[str, Any]):
        w = cfg["weights"]
        self.w_rescore = float(w.get("asr_rescore", 0.6))
        self.w_av = float(w.get("av_consistency", 0.25))
        self.w_nb = float(w.get("nbest_variance", 0.1))
        self.w_llm = float(w.get("llm_entropy", 0.05))

    def score(
        self,
        asr_rescore_logprob: float,
        av_consistency: float,
        nbest: list[str],
        token_logprobs: torch.Tensor | None = None,
    ) -> ConfidenceReport:
        s_acoustic_conf = squash_logprob(asr_rescore_logprob)

        av_conf = max(0.0, min(1.0, (av_consistency + 1.0) / 2.0))

        if len(nbest) <= 1:
            nb_conf = 1.0
        else:
            sims = []
            for i in range(len(nbest)):
                for j in range(i + 1, len(nbest)):
                    sims.append(_edit_sim(nbest[i], nbest[j]))
            nb_conf = sum(sims) / len(sims)

        if token_logprobs is None or token_logprobs.numel() == 0:
            llm_conf = 0.5
        else:
            llm_conf = 1.0 / (1.0 + math.exp(-token_logprobs.mean().item() * 1.5 + 2.0))

        total = (
            self.w_rescore * s_acoustic_conf
            + self.w_av * av_conf
            + self.w_nb * nb_conf
            + self.w_llm * llm_conf
        )
        return ConfidenceReport(
            total=total,
            s_acoustic=asr_rescore_logprob,
            s_acoustic_conf=s_acoustic_conf,
            av_consistency=av_conf,
            nbest_agreement=nb_conf,
            llm_entropy_conf=llm_conf,
            components={"rescore": s_acoustic_conf, "av": av_conf, "nb": nb_conf, "llm": llm_conf},
        )
