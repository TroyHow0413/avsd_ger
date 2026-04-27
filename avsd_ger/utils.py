"""Shared helpers: config loading, device resolution, token-feature pooling."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass
class BackboneOutputs:
    """Container passed between pipeline stages."""
    # ASR
    nbest: list[str] = field(default_factory=list)
    nbest_scores: list[float] = field(default_factory=list)
    asr_features: torch.Tensor | None = None
    asr_frame_rate: float = 50.0
    words: list[Any] = field(default_factory=list)   # list[WordTiming]

    # VSR
    lip_hyp: str = ""
    vsr_features: torch.Tensor | None = None
    vsr_frame_rate: float = 25.0

    # raw inputs
    audio: torch.Tensor | None = None
    face_crop: torch.Tensor | None = None


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def squash_logprob(mean_logprob: float, slope: float = 1.5) -> float:
    """Map a Whisper mean-token log-prob in (-inf, 0] into a [0,1] confidence.

    sigmoid(mean_logprob * slope + 2) -- calibrated so lp~=-0.5 -> ~0.78,
    lp~=-2 -> ~0.27.
    """
    return 1.0 / (1.0 + math.exp(-(mean_logprob * slope + 2.0)))


def pool_encoder_to_tokens(
    encoder_feats: torch.Tensor,
    words: list[Any],
    frame_rate_hz: float = 50.0,
    min_frames: int = 1,
) -> torch.Tensor:
    """
    Mean-pool encoder frames within each word's [start, end] window.

    Args:
        encoder_feats: [T_a, D] Whisper encoder output.
        words:         list of WordTiming (any object with .start and .end in seconds).
        frame_rate_hz: frames per second.
    Returns:
        [N_tok, D] -- one vector per word. If `words` is empty, returns a single
        global mean-pool token to keep downstream shapes valid.
    """
    T, D = encoder_feats.shape
    if not words:
        return encoder_feats.mean(dim=0, keepdim=True)

    rows: list[torch.Tensor] = []
    for w in words:
        a = max(0, int(math.floor(w.start * frame_rate_hz)))
        b = min(T, int(math.ceil(w.end * frame_rate_hz)))
        if b - a < min_frames:
            b = min(T, a + min_frames)
        if b <= a:
            rows.append(encoder_feats.mean(dim=0))
        else:
            rows.append(encoder_feats[a:b].mean(dim=0))
    return torch.stack(rows, dim=0)
