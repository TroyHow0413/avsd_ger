"""Full AVSD-GER pipeline (spec-aligned): C1 -> C2 -> C3.

Single-speaker-per-utterance execution path, but with the structural pieces
that make the multi-speaker story work:
  * identity retrieval + optional EMA refresh (C3 gated by s_acoustic >= tau_update)
  * token-level pooling of Whisper encoder features using word timestamps
  * per-speaker cross-attention with an (optional) speaker activity mask
  * acoustic rescoring as the primary C3 signal

For true multi-speaker meeting evaluation the caller runs this per-speaker in
the session and concatenates outputs (see avsd_ger/eval/session.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .backbones import AVHubertVSR, WhisperASR
from .c1_identity import FaceEncoder, IdentityPool, VoiceEncoder
from .c1_identity.identity_pool import IdentityQueryResult
from .c2_alignment import GERHead, IDConditionedAligner
from .c3_feedback import ClosedLoopController, ConfidenceScorer
from .c3_feedback.closed_loop import LoopAction, LoopDecision
from .utils import load_config, pool_encoder_to_tokens, resolve_device, seed_all, squash_logprob


class AVSDGERPipeline:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        seed_all(int(cfg.get("seed", 1337)))
        self.device = resolve_device(cfg.get("device", "cpu"))
        stub = bool(cfg.get("stub_backbones", True))
        self.stub = stub

        # Backbones
        self.asr = WhisperASR(cfg["asr"], stub=stub, device=self.device)
        self.vsr = AVHubertVSR(cfg["vsr"], stub=stub, device=self.device)

        # C1
        self.voice = VoiceEncoder(cfg["identity"]["voice_encoder"], stub=stub, device=self.device)
        self.face = FaceEncoder(cfg["identity"]["face_encoder"], stub=stub, device=self.device)
        self.pool = IdentityPool(cfg["identity"], device=self.device)
        self.ema_alpha = float(cfg["identity"]["ema_alpha"])

        # C2
        self.aligner = IDConditionedAligner(
            cfg["alignment"],
            z_dim=cfg["identity"]["fused_dim"],
            d_asr=WhisperASR.ENCODER_DIM,
            d_vsr=AVHubertVSR.FEATURE_DIM,
        ).to(self.device)
        self.ger = GERHead(
            cfg["ger"],
            z_dim=cfg["identity"]["fused_dim"],
            d_align=cfg["alignment"]["d_model"],
            stub=stub,
            device=self.device,
        )

        # C3
        self.scorer = ConfidenceScorer(cfg["feedback"])
        self.loop = ClosedLoopController(cfg["feedback"])

        # Ablation flags (spec section 10, Table 2). All False by default
        # = "Full Model". Flip one at a time from config to run rows of
        # the ablation table without code changes.
        abl = cfg.get("ablation", {}) or {}
        self.disable_c1 = bool(abl.get("disable_c1", False))                # bypass ID conditioning
        self.disable_c2 = bool(abl.get("disable_c2", False))                # skip GER; return ASR 1-best
        self.disable_c3 = bool(abl.get("disable_c3", False))                # skip closed-loop
        self.disable_conf_gate = bool(abl.get("disable_conf_gate", False))  # EMA-update unconditionally

    @classmethod
    def from_config(cls, path: str | Path) -> "AVSDGERPipeline":
        return cls(load_config(path))

    # ------------------------------------------------------------------ C1 enrol
    def enroll(self, speaker_id: str, audio_wav, face_image, meta: dict | None = None) -> None:
        voice_emb = self.voice.embed(audio_wav)
        face_emb = self.face.embed(face_image)
        self.pool.enroll(speaker_id, voice_emb, face_emb, meta=meta)

    def save_pool(self, path: str | Path) -> None:
        self.pool.save(path)

    def load_pool(self, path: str | Path) -> None:
        self.pool.load(path)

    # ------------------------------------------------------------------ main run
    def run(
        self,
        audio_wav: torch.Tensor | np.ndarray,
        video_frames: torch.Tensor,
        face_image: np.ndarray | None = None,
        speaker_mask_v: torch.Tensor | None = None,
        snr_per_tok: torch.Tensor | None = None,
        lip_conf_v: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        asr_out = self.asr.transcribe(audio_wav)
        vsr_out = self.vsr.extract(video_frames)

        if asr_out.encoder_features is None:
            T_a = 150
            asr_feats = torch.randn(T_a, WhisperASR.ENCODER_DIM, device=self.device)
        else:
            asr_feats = asr_out.encoder_features.to(self.device)
        asr_tok_feats = pool_encoder_to_tokens(
            asr_feats, asr_out.words, frame_rate_hz=asr_out.frame_rate_hz
        )

        voice_emb = self.voice.embed(audio_wav)
        if face_image is None:
            face_image = self._first_frame_as_rgb(video_frames)
        face_emb = self.face.embed(face_image)

        trace: list[dict[str, Any]] = []
        skip_ids: set[str] = set()
        id_q = self.pool.query(voice_emb, face_emb, skip_ids=skip_ids)

        # Ablation: w/o C1 -> zero out z_id so C2 degenerates to modality-only
        if self.disable_c1:
            id_q = IdentityQueryResult(
                top_ids=[], top_scores=[], av_consistency=0.0,
                z_id=torch.zeros_like(id_q.z_id), is_unknown=True,
            )

        ger_out: dict[str, Any] | None = None
        rep = None
        decision: LoopDecision | None = None

        # Ablation: w/o C3 -> cap iterations at 1 and never update pool
        max_iters = 1 if self.disable_c3 else self.loop.max_iters

        for it in range(max_iters):
            f_align = self.aligner(
                asr_tok_feats=asr_tok_feats,
                vsr_feats=vsr_out["vsr_features"].to(self.device),
                e_id=id_q.z_id,
                speaker_mask_v=speaker_mask_v,
                snr_per_tok=snr_per_tok,
                lip_conf_v=lip_conf_v,
            )
            speaker_id_hint = id_q.top_ids[0] if id_q.top_ids and not id_q.is_unknown else None

            if self.disable_c2:
                # Ablation: skip GER -> return ASR 1-best as the hypothesis.
                top = asr_out.nbest[0] if asr_out.nbest else ""
                ger_out = {"text": top, "token_logprobs": torch.zeros(0), "prompt": ""}
            else:
                ger_out = self.ger.generate(
                    z_id=id_q.z_id,
                    f_align=f_align,
                    nbest=asr_out.nbest,
                    nbest_scores=asr_out.nbest_scores,
                    lip_hyp=vsr_out.get("lip_hyp", ""),
                    speaker_id=speaker_id_hint,
                )

            s_acoustic = self.asr.rescore(audio_wav, ger_out["text"])
            s_acoustic_conf = squash_logprob(s_acoustic)
            rep = self.scorer.score(
                asr_rescore_logprob=s_acoustic,
                av_consistency=id_q.av_consistency,
                nbest=asr_out.nbest,
                token_logprobs=ger_out.get("token_logprobs"),
            )
            decision = self.loop.decide(
                total_confidence=rep.total,
                s_acoustic_conf=s_acoustic_conf,
                iteration=it,
            )
            # Ablation: C3 w/o Conf. Gate -- promote every ACCEPT to
            # ACCEPT_AND_UPDATE so the pool is refreshed unconditionally.
            # Spec section 10 note: this variant must perform *worse* than
            # disabling C3 entirely -- proves the gate is structural safety.
            if self.disable_conf_gate and decision.action == LoopAction.ACCEPT_NO_UPDATE:
                decision = LoopDecision(
                    LoopAction.ACCEPT_AND_UPDATE,
                    "conf-gate DISABLED (ablation) -- unconditional pool update",
                    it,
                )

            trace.append({
                "iter": it,
                "top_ids": list(id_q.top_ids),
                "total_conf": rep.total,
                "s_acoustic": s_acoustic,
                "s_acoustic_conf": s_acoustic_conf,
                "components": rep.components,
                "decision": decision.action.value,
                "reason": decision.reason,
                "text": ger_out["text"],
            })

            if decision.action == LoopAction.ACCEPT_AND_UPDATE:
                if speaker_id_hint is not None and not self.disable_c3:
                    self.pool.ema_update(
                        speaker_id_hint,
                        new_voice_emb=voice_emb,
                        new_face_emb=face_emb,
                        alpha=self.ema_alpha,
                    )
                break
            if decision.action == LoopAction.ACCEPT_NO_UPDATE:
                break
            if decision.action == LoopAction.REIDENTIFY and id_q.top_ids:
                skip_ids.add(id_q.top_ids[0])
                id_q = self.pool.query(voice_emb, face_emb, skip_ids=skip_ids)
                if id_q.is_unknown:
                    break
            # REALIGN falls through to re-run C2 with same id_q.

        return {
            "text": ger_out["text"] if ger_out else "",
            "speaker_id": id_q.top_ids[0] if (id_q.top_ids and not id_q.is_unknown) else None,
            "confidence": rep.total if rep else 0.0,
            "s_acoustic": trace[-1]["s_acoustic"] if trace else None,
            "iterations": len(trace),
            "pool_updated": (
                decision.action == LoopAction.ACCEPT_AND_UPDATE
                and not self.disable_c3 if decision else False
            ),
            "trace": trace,
        }

    @staticmethod
    def _first_frame_as_rgb(video: torch.Tensor) -> np.ndarray:
        frame = video[0, 0].detach().cpu().numpy()
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        return np.stack([frame, frame, frame], axis=-1)
