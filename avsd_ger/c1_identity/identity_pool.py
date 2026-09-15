"""C1 — Identity Pool.

Stores enrolled speakers as (voice_emb, face_emb) pairs. At query time,
retrieves the top-k speakers and produces a fused identity vector `z_id`
that is consumed by C2's FiLM layer and the GER prompt prefix.

Design notes
------------
* Fusion is a small MLP (`fuse`) over concat(voice, face) producing a
  `fused_dim`-dim vector. This projection is learnable — the only learnable
  parameters in C1 — and is trained jointly with C2 via a contrastive loss
  (NT-Xent between enrolled z_id and pooled Whisper+AV-HuBERT features of
  the same utterance). See `scripts/train_identity.py` (TODO).
* `av_consistency` is read off the top-1 fused-space match score. Below
  `min_av_consistency` we fall back to a zero z_id so downstream modules
  behave like modality-only processing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import cosine_sim


@dataclass
class EnrolledSpeaker:
    speaker_id: str
    voice_emb: torch.Tensor
    face_emb: torch.Tensor
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class IdentityQueryResult:
    top_ids: list[str]
    top_scores: list[float]
    av_consistency: float
    z_id: torch.Tensor
    is_unknown: bool
    evidence_mode: str = "audio_visual"
    logged_top_ids: list[str] = field(default_factory=list)
    logged_top_scores: list[float] = field(default_factory=list)


class IdentityFuser(nn.Module):
    """Small MLP combining voice + face -> fused identity vector."""

    def __init__(self, voice_dim: int, face_dim: int, fused_dim: int):
        super().__init__()
        self.voice_proj = nn.Linear(voice_dim, fused_dim)
        self.face_proj = nn.Linear(face_dim, fused_dim)
        self.fuse = nn.Sequential(
            nn.Linear(2 * fused_dim, fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, fused_dim),
        )

    def forward(self, voice: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        v = F.normalize(self.voice_proj(voice), dim=-1)
        f = F.normalize(self.face_proj(face), dim=-1)
        z = self.fuse(torch.cat([v, f], dim=-1))
        return F.normalize(z, dim=-1)


class IdentityPool(nn.Module):
    def __init__(self, cfg: dict[str, Any], device: str | torch.device = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.top_k = int(cfg["top_k"])
        # Keep the model decision at top_k (normally 3), while retaining a
        # wider ranking for appendix Top-1/3/5 identity metrics.
        self.log_top_k = max(self.top_k, int(cfg.get("log_top_k", 5)))
        self.min_av = float(cfg["min_av_consistency"])

        self.fuser = IdentityFuser(
            voice_dim=cfg["voice_dim"],
            face_dim=cfg["face_dim"],
            fused_dim=cfg["fused_dim"],
        ).to(self.device)

        self._speakers: dict[str, EnrolledSpeaker] = {}

    # -------------------------------------------------------------- enroll
    def enroll(
        self,
        speaker_id: str,
        voice_emb: torch.Tensor,
        face_emb: torch.Tensor | None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self._speakers[speaker_id] = EnrolledSpeaker(
            speaker_id=speaker_id,
            voice_emb=voice_emb.detach().to(self.device),
            face_emb=face_emb.detach().to(self.device),
            meta=meta or {},
        )

    def __len__(self) -> int:
        return len(self._speakers)

    @property
    def speaker_ids(self) -> tuple[str, ...]:
        """Return the enrolled gallery IDs without exposing mutable state."""
        return tuple(self._speakers)

    def clear_gallery(self) -> None:
        """Remove enrollment records while preserving the trained fuser."""
        self._speakers.clear()

    # -------------------------------------------------------------- EMA update
    def ema_update(
        self,
        speaker_id: str,
        new_voice_emb: torch.Tensor | None = None,
        new_face_emb: torch.Tensor | None = None,
        alpha: float = 0.1,
    ) -> bool:
        """Spec §2 C3: e_id_new = (1-α)·e_id_old + α·e_obs, applied per modality.

        Normally called only when the C3 acoustic-rescore gate passes. The
        explicit update-gate ablation may call it unconditionally for a known
        enrolled speaker so the safety mechanism can be measured.
        """
        if speaker_id not in self._speakers:
            return False
        spk = self._speakers[speaker_id]
        updated = False
        if new_voice_emb is not None:
            v = new_voice_emb.detach().to(spk.voice_emb.device)
            spk.voice_emb = (1.0 - alpha) * spk.voice_emb + alpha * v
            updated = True
        if new_face_emb is not None:
            f = new_face_emb.detach().to(spk.face_emb.device)
            spk.face_emb = (1.0 - alpha) * spk.face_emb + alpha * f
            updated = True
        return updated

    # -------------------------------------------------------------- query
    def _query_embedding(
        self,
        voice_emb: torch.Tensor,
        face_emb: torch.Tensor | None,
    ) -> tuple[torch.Tensor, str]:
        voice_emb = voice_emb.to(self.device)
        face_emb = face_emb.to(self.device) if face_emb is not None else None
        if face_emb is None:
            return (
                F.normalize(
                    self.fuser.voice_proj(voice_emb.unsqueeze(0)), dim=-1
                ).squeeze(0),
                "voice_only",
            )
        return (
            self.fuser(voice_emb.unsqueeze(0), face_emb.unsqueeze(0)).squeeze(0),
            "audio_visual",
        )

    def _speaker_embedding(
        self,
        speaker_id: str,
        *,
        voice_only: bool,
    ) -> torch.Tensor:
        if speaker_id not in self._speakers:
            raise KeyError(f"Speaker {speaker_id!r} is not enrolled")
        speaker = self._speakers[speaker_id]
        if voice_only:
            return F.normalize(
                self.fuser.voice_proj(speaker.voice_emb.unsqueeze(0)), dim=-1
            ).squeeze(0)
        return self.fuser(
            speaker.voice_emb.unsqueeze(0), speaker.face_emb.unsqueeze(0)
        ).squeeze(0)

    def conditioning_vector_for_speaker(
        self,
        voice_emb: torch.Tensor,
        face_emb: torch.Tensor | None,
        speaker_id: str,
    ) -> torch.Tensor:
        """Build the normal C1 conditioning vector for an explicit gallery ID.

        This is used by the deterministic shuffled-z causal intervention. It
        changes only the vector supplied to C2/GER; retrieval scores, labels,
        confidence, and speaker hints remain those of the original query.
        """
        z_query, evidence_mode = self._query_embedding(voice_emb, face_emb)
        z_speaker = self._speaker_embedding(
            speaker_id, voice_only=evidence_mode == "voice_only",
        )
        return F.normalize(0.5 * z_query + 0.5 * z_speaker, dim=-1)

    def deterministic_derangement(self) -> dict[str, str]:
        """Map every sorted gallery ID to the next ID (cyclic, no self-map)."""
        ids = sorted(self._speakers)
        if len(ids) < 2:
            raise ValueError(
                "shuffled_z_id requires at least two enrolled speakers"
            )
        return {speaker_id: ids[(index + 1) % len(ids)] for index, speaker_id in enumerate(ids)}

    def query(
        self,
        voice_emb: torch.Tensor,
        face_emb: torch.Tensor,
        skip_ids: set[str] | None = None,
    ) -> IdentityQueryResult:
        skip_ids = skip_ids or set()
        voice_emb = voice_emb.to(self.device)
        face_emb = face_emb.to(self.device) if face_emb is not None else None

        # Voice and face live in different native dims (192 vs 512); they can
        # only be compared after the learnable fuser projects both into
        # `fused_dim` space.
        z_query, evidence_mode = self._query_embedding(voice_emb, face_emb)

        scored: list[tuple[str, float]] = []
        for sid, spk in self._speakers.items():
            if sid in skip_ids:
                continue
            z_spk = self._speaker_embedding(sid, voice_only=face_emb is None)
            scored.append((sid, float(cosine_sim(z_query, z_spk).item())))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: self.top_k]
        logged_top = scored[: self.log_top_k]

        av_consistency = top[0][1] if top else 0.0
        is_unknown = av_consistency < self.min_av

        if is_unknown or not top:
            z_id = torch.zeros_like(z_query)
        else:
            sid_top, _ = top[0]
            z_id = self.conditioning_vector_for_speaker(
                voice_emb, face_emb, sid_top,
            )

        return IdentityQueryResult(
            top_ids=[sid for sid, _ in top],
            top_scores=[s for _, s in top],
            av_consistency=av_consistency,
            z_id=z_id,
            is_unknown=is_unknown,
            evidence_mode=evidence_mode,
            logged_top_ids=[sid for sid, _ in logged_top],
            logged_top_scores=[s for _, s in logged_top],
        )

    # -------------------------------------------------------------- i/o
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "fuser": self.fuser.state_dict(),
            "speakers": {
                sid: {
                    "voice_emb": spk.voice_emb.cpu().tolist(),
                    "face_emb": spk.face_emb.cpu().tolist(),
                    "meta": spk.meta,
                }
                for sid, spk in self._speakers.items()
            },
        }
        torch.save(state, path)

    def load(self, path: str | Path, *, load_gallery: bool = True) -> None:
        """Load fuser weights and, unless disabled, the enrollment gallery.

        ``load_gallery=False`` is the safe evaluation/fresh-pool operation: it
        deliberately reuses the learned identity model but never imports
        training or another meeting's speaker identities.
        """
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.fuser.load_state_dict(state["fuser"])
        self.clear_gallery()
        if not load_gallery:
            return
        for sid, rec in state.get("speakers", {}).items():
            self._speakers[sid] = EnrolledSpeaker(
                speaker_id=sid,
                voice_emb=torch.tensor(rec["voice_emb"], device=self.device),
                face_emb=torch.tensor(rec["face_emb"], device=self.device),
                meta=rec.get("meta", {}),
            )
