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
        face_emb: torch.Tensor,
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

    # -------------------------------------------------------------- EMA update
    def ema_update(
        self,
        speaker_id: str,
        new_voice_emb: torch.Tensor | None = None,
        new_face_emb: torch.Tensor | None = None,
        alpha: float = 0.1,
    ) -> None:
        """Spec §2 C3: e_id_new = (1-α)·e_id_old + α·e_obs, applied per modality.

        Only called when the C3 acoustic-rescore gate (s_i ≥ tau_update) passes —
        the pipeline never calls this unconditionally. That gate is the
        structural safety mechanism preventing error amplification.
        """
        if speaker_id not in self._speakers:
            return
        spk = self._speakers[speaker_id]
        if new_voice_emb is not None:
            v = new_voice_emb.detach().to(spk.voice_emb.device)
            spk.voice_emb = (1.0 - alpha) * spk.voice_emb + alpha * v
        if new_face_emb is not None:
            f = new_face_emb.detach().to(spk.face_emb.device)
            spk.face_emb = (1.0 - alpha) * spk.face_emb + alpha * f

    # -------------------------------------------------------------- query
    def query(
        self,
        voice_emb: torch.Tensor,
        face_emb: torch.Tensor,
        skip_ids: set[str] | None = None,
    ) -> IdentityQueryResult:
        skip_ids = skip_ids or set()
        voice_emb = voice_emb.to(self.device)
        face_emb = face_emb.to(self.device)

        # Voice and face live in different native dims (192 vs 512); they can
        # only be compared after the learnable fuser projects both into
        # `fused_dim` space.
        z_query = self.fuser(
            voice_emb.unsqueeze(0), face_emb.unsqueeze(0)
        ).squeeze(0)

        scored: list[tuple[str, float]] = []
        for sid, spk in self._speakers.items():
            if sid in skip_ids:
                continue
            z_spk = self.fuser(
                spk.voice_emb.unsqueeze(0), spk.face_emb.unsqueeze(0)
            ).squeeze(0)
            scored.append((sid, float(cosine_sim(z_query, z_spk).item())))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[: self.top_k]

        av_consistency = top[0][1] if top else 0.0
        is_unknown = av_consistency < self.min_av

        if is_unknown or not top:
            z_id = torch.zeros_like(z_query)
        else:
            sid_top, _ = top[0]
            spk = self._speakers[sid_top]
            z_spk = self.fuser(
                spk.voice_emb.unsqueeze(0), spk.face_emb.unsqueeze(0)
            ).squeeze(0)
            z_id = F.normalize(0.5 * z_query + 0.5 * z_spk, dim=-1)

        return IdentityQueryResult(
            top_ids=[sid for sid, _ in top],
            top_scores=[s for _, s in top],
            av_consistency=av_consistency,
            z_id=z_id,
            is_unknown=is_unknown,
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

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.fuser.load_state_dict(state["fuser"])
        for sid, rec in state["speakers"].items():
            self._speakers[sid] = EnrolledSpeaker(
                speaker_id=sid,
                voice_emb=torch.tensor(rec["voice_emb"], device=self.device),
                face_emb=torch.tensor(rec["face_emb"], device=self.device),
                meta=rec.get("meta", {}),
            )
