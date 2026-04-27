"""C1 — Dual-gate filtering (spec §8 step 2).

Purpose
-------
Before a frame is allowed to contribute to enrollment, cold-start clustering,
or EMA refresh, it must pass BOTH of:

    audio_gate : SNR(frame)   > τ_a      (tau_a_snr_db)
    visual_gate: lip_conf(frame) > τ_v   (tau_v_lip_conf)

Rationale (from the spec): only high-quality, well-aligned A/V frames should
seed identity. This kills the biggest failure mode of self-supervised speaker
pools — attractor drift from noisy, off-mic, or occluded frames.

The SNR estimator here is a lightweight frame-energy based approximation;
for production, plug in a dedicated SNR module (e.g. WADA-SNR) by swapping
`estimate_frame_snr`. The gate interface does not change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


# ------------------------------------------------------------------ SNR
def estimate_frame_snr(
    wav: torch.Tensor | np.ndarray,
    sr: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    noise_percentile: float = 10.0,
) -> np.ndarray:
    """Lightweight per-frame SNR estimate in dB.

    Uses the common 'percentile-of-energy as noise floor' heuristic:
      noise_power ≈ p10(frame_power)
      snr_db     = 10·log10(frame_power / noise_power)

    Args:
        wav: 1-D waveform (torch or numpy).
        sr: sample rate.
        frame_ms: analysis window length.
        hop_ms: hop length.
        noise_percentile: percentile of frame energy used as noise floor.

    Returns:
        [N_frames] array of per-frame SNR in dB.
    """
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().numpy()
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)

    win = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    if win <= 0 or hop <= 0 or wav.size < win:
        return np.zeros(1, dtype=np.float32)

    n_frames = 1 + (wav.size - win) // hop
    powers = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = wav[i * hop : i * hop + win]
        powers[i] = float(np.mean(seg * seg) + 1e-10)

    noise = float(np.percentile(powers, noise_percentile)) + 1e-10
    snr_db = 10.0 * np.log10(powers / noise)
    return snr_db.astype(np.float32)


# ------------------------------------------------------------------ gate
@dataclass
class DualGateResult:
    """Output of a dual-gate pass.

    mask[t] is True iff frame t satisfies both gates. `keep_ratio` is the
    fraction of frames retained — useful as a sanity signal (a recording
    that drops below ~0.2 is usually unusable).
    """
    mask: np.ndarray          # [N] bool
    snr_db: np.ndarray        # [N] float32
    lip_conf: np.ndarray      # [N] float32
    keep_ratio: float


class DualGate:
    """Implements ``passes = (SNR > τ_a) AND (lip_conf > τ_v)``.

    Both thresholds come from the `identity.dual_gate` config block. The
    gate can be applied to:
      * a raw waveform + lip-confidence track (training / enrollment),
      * or a pre-computed (snr_db, lip_conf) pair (online pipeline).

    Callers that only care about the pass/fail decision can use
    :meth:`apply`; those that want the underlying signals (for tracing or
    visualisation) can use :meth:`filter`.
    """

    def __init__(self, cfg: dict[str, Any]):
        dg = cfg.get("dual_gate", cfg)   # accept either {identity: {dual_gate: ...}} or the inner dict
        self.tau_a_snr_db = float(dg["tau_a_snr_db"])
        self.tau_v_lip_conf = float(dg["tau_v_lip_conf"])

    # ---------------- core ---------------------------------------------
    def apply(
        self,
        snr_db: np.ndarray | torch.Tensor,
        lip_conf: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Return a boolean mask of frames passing both gates.

        If the inputs have different lengths (common — audio frames @ 100 Hz,
        lip confidence @ 25 Hz), the shorter is upsampled to the longer via
        nearest-neighbour. This is intentional: we err on the side of using
        the audio clock since SNR changes faster than occlusion state.
        """
        snr_db = _to_np(snr_db)
        lip_conf = _to_np(lip_conf)
        snr_db, lip_conf = _align_lengths(snr_db, lip_conf)
        audio_ok = snr_db > self.tau_a_snr_db
        visual_ok = lip_conf > self.tau_v_lip_conf
        return np.logical_and(audio_ok, visual_ok)

    def filter(
        self,
        wav: torch.Tensor | np.ndarray,
        lip_conf: np.ndarray | torch.Tensor,
        sr: int = 16000,
    ) -> DualGateResult:
        """Run the full pipeline: estimate SNR from wav, gate, return signals."""
        snr_db = estimate_frame_snr(wav, sr=sr)
        lip_conf = _to_np(lip_conf)
        snr_db, lip_conf = _align_lengths(snr_db, lip_conf)
        mask = np.logical_and(snr_db > self.tau_a_snr_db, lip_conf > self.tau_v_lip_conf)
        keep = float(mask.mean()) if mask.size else 0.0
        return DualGateResult(mask=mask, snr_db=snr_db, lip_conf=lip_conf, keep_ratio=keep)


# ------------------------------------------------------------------ helpers
def _to_np(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32).reshape(-1)
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _align_lengths(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour upsample the shorter of (a, b) to match the longer."""
    if a.shape[0] == b.shape[0]:
        return a, b
    if a.shape[0] == 0 or b.shape[0] == 0:
        n = max(a.shape[0], b.shape[0])
        return (np.zeros(n, dtype=np.float32) if a.shape[0] == 0 else a,
                np.zeros(n, dtype=np.float32) if b.shape[0] == 0 else b)
    if a.shape[0] < b.shape[0]:
        idx = np.linspace(0, a.shape[0] - 1, b.shape[0]).round().astype(np.int64)
        return a[idx], b
    idx = np.linspace(0, b.shape[0] - 1, a.shape[0]).round().astype(np.int64)
    return a, b[idx]
