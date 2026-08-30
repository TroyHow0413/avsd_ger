"""Deterministic quality-track preparation for cached and live Stage-2 runs."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..c1_identity.gate import estimate_frame_snr


def resample_quality_track(
    values: Sequence[float] | np.ndarray | torch.Tensor | None,
    target_length: int,
) -> torch.Tensor:
    """Linearly resample a confidence track to a model feature clock."""
    if target_length < 0:
        raise ValueError("target_length must be non-negative")
    if target_length == 0:
        return torch.empty(0, dtype=torch.float32)
    if values is None:
        return torch.zeros(target_length, dtype=torch.float32)
    track = torch.as_tensor(values, dtype=torch.float32).flatten()
    if track.numel() == 0:
        return torch.zeros(target_length, dtype=torch.float32)
    if not bool(torch.isfinite(track).all().item()):
        raise ValueError("quality track contains non-finite values")
    track = track.clamp(0.0, 1.0)
    if track.numel() == target_length:
        return track.contiguous()
    return F.interpolate(
        track.view(1, 1, -1),
        size=target_length,
        mode="linear",
        align_corners=False,
    ).view(-1).clamp(0.0, 1.0)


def _word_bounds(word: Any) -> tuple[float | None, float | None]:
    if isinstance(word, dict):
        start, end = word.get("start"), word.get("end")
    else:
        start, end = getattr(word, "start", None), getattr(word, "end", None)
    try:
        return float(start), float(end)
    except (TypeError, ValueError):
        return None, None


def token_snr_scores(
    wav: torch.Tensor | np.ndarray,
    words: Sequence[Any],
    n_tokens: int,
    *,
    tau_snr_db: float,
    soft_scale_db: float = 4.0,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Pool frame SNR over word timestamps and map it smoothly to [0, 1]."""
    if n_tokens < 0:
        raise ValueError("n_tokens must be non-negative")
    if n_tokens == 0:
        return torch.empty(0, dtype=torch.float32)
    if soft_scale_db <= 0:
        raise ValueError("soft_scale_db must be positive")
    snr_db = estimate_frame_snr(wav, sr=sample_rate)
    if snr_db.size == 0 or not np.isfinite(snr_db).all():
        return torch.zeros(n_tokens, dtype=torch.float32)

    # estimate_frame_snr uses a 10 ms hop.
    frame_hz = 100.0
    global_db = float(np.mean(snr_db))
    pooled: list[float] = []
    for word in words:
        start, end = _word_bounds(word)
        if start is None or end is None or end <= start:
            pooled.append(global_db)
            continue
        left = max(0, min(len(snr_db) - 1, int(np.floor(start * frame_hz))))
        right = max(left + 1, min(len(snr_db), int(np.ceil(end * frame_hz))))
        pooled.append(float(np.mean(snr_db[left:right])))

    if not pooled:
        pooled = [global_db]
    db_track = torch.tensor(pooled, dtype=torch.float32)
    if db_track.numel() != n_tokens:
        db_track = F.interpolate(
            db_track.view(1, 1, -1),
            size=n_tokens,
            mode="linear",
            align_corners=False,
        ).view(-1)
    return torch.sigmoid((db_track - float(tau_snr_db)) / float(soft_scale_db))
