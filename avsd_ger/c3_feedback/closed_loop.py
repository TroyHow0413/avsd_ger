"""C3 — closed-loop controller (spec-aligned).

Four possible actions:

    ACCEPT_AND_UPDATE   s_acoustic >= tau_update              -> output kept,
                                                                 ID Pool refreshed via EMA
    ACCEPT_NO_UPDATE    s_acoustic <  tau_update              -> output kept, pool NOT touched
                        but total >= mid confidence             (this gate is the structural
                                                                 safety mechanism that prevents
                                                                 error amplification; spec section 2 C3)
    REALIGN             mid > total >= low                    -> same speaker, re-run C2
    REIDENTIFY          total < low                           -> pop top-1 speaker, re-run C1->C2

Bounded by `max_iters`. Returning ACCEPT_AND_UPDATE means the pipeline will
call `IdentityPool.ema_update(speaker_id, new_voice, new_face, alpha)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class LoopAction(str, Enum):
    ACCEPT_AND_UPDATE = "accept_and_update"
    ACCEPT_NO_UPDATE = "accept_no_update"
    REALIGN = "realign"
    REIDENTIFY = "reidentify"


@dataclass
class LoopDecision:
    action: LoopAction
    reason: str
    iteration: int


class ClosedLoopController:
    def __init__(self, cfg: dict[str, Any]):
        self.max_iters = int(cfg["max_iters"])
        self.low = float(cfg["confidence_low"])
        self.mid = float(cfg["confidence_mid"])
        self.tau_update = float(cfg["tau_update"])

    def decide(
        self,
        total_confidence: float,
        s_acoustic_conf: float,
        iteration: int,
    ) -> LoopDecision:
        last = iteration >= self.max_iters - 1

        if total_confidence < self.low and not last:
            return LoopDecision(
                LoopAction.REIDENTIFY,
                f"total {total_confidence:.2f} < low {self.low}",
                iteration,
            )

        if total_confidence < self.mid and not last:
            return LoopDecision(
                LoopAction.REALIGN,
                f"mid band: total {total_confidence:.2f}",
                iteration,
            )

        # Accept -- now decide whether to refresh the ID pool.
        if s_acoustic_conf >= self.tau_update:
            return LoopDecision(
                LoopAction.ACCEPT_AND_UPDATE,
                f"s_acoustic_conf {s_acoustic_conf:.2f} >= tau_update {self.tau_update}",
                iteration,
            )
        return LoopDecision(
            LoopAction.ACCEPT_NO_UPDATE,
            f"s_acoustic_conf {s_acoustic_conf:.2f} < tau_update {self.tau_update} -- pool frozen",
            iteration,
        )
