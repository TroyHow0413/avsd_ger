"""Evaluation infrastructure (spec section 13).

Modules:
    session   -- multi-speaker session runner (pipeline fan-out + stitching)
    metrics   -- SA-WER, SCR, AV-SID Acc, DER, JER (spec section 13)
    power     -- pynvml + psutil idle-corrected energy measurement (spec section 5.10)
"""
from .session import SessionRunner, SessionTurn, SessionTurnResult, SessionResult
from .metrics import (
    compute_sa_wer,
    compute_scr,
    compute_av_sid_accuracy,
    compute_der,
    compute_jer,
    MetricsReport,
)
from .power import PowerMonitor, PowerSample, PowerReport

__all__ = [
    "SessionRunner",
    "SessionTurn",
    "SessionTurnResult",
    "SessionResult",
    "compute_sa_wer",
    "compute_scr",
    "compute_av_sid_accuracy",
    "compute_der",
    "compute_jer",
    "MetricsReport",
    "PowerMonitor",
    "PowerSample",
    "PowerReport",
]
