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
import re
from typing import Any

import numpy as np
import torch

from .backbones import AVHubertVSR, WhisperASR
from .c1_identity import FaceEncoder, IdentityPool, VoiceEncoder
from .frontend.mouth_roi import MouthROIExtractor
from .c1_identity.identity_pool import IdentityQueryResult
from .c2_alignment import GERHead, IDConditionedAligner
from .c3_feedback import ClosedLoopController, ConfidenceScorer
from .c3_feedback.closed_loop import LoopAction, LoopDecision
from .utils import load_config, pool_encoder_to_tokens, resolve_device, seed_all, squash_logprob


def _tensor_debug(x: Any) -> dict[str, Any] | None:
    if x is None:
        return None
    try:
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            t = x.detach().cpu()
        else:
            return None
        out: dict[str, Any] = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "numel": int(t.numel()),
        }
        if t.numel() > 0 and t.is_floating_point():
            tf = t.float()
            out.update({
                "min": float(tf.min().item()),
                "max": float(tf.max().item()),
                "mean": float(tf.mean().item()),
                "std": float(tf.std(unbiased=False).item()) if tf.numel() > 1 else 0.0,
                "norm": float(tf.norm().item()),
            })
        return out
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


class AVSDGERPipeline:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        seed_all(int(cfg.get("seed", 1337)))
        self.device = resolve_device(cfg.get("device", "cpu"))
        stub = bool(cfg.get("stub_backbones", True))
        self.stub = stub

        self.ger_mode = str(cfg.get("ger", {}).get("mode", "audio_only")).lower()
        if self.ger_mode not in {"audio_only", "av", "visual_only"}:
            raise ValueError(
                f"Unsupported ger.mode={self.ger_mode!r}; "
                "expected audio_only, av, or visual_only"
            )

        # Backbones
        self.asr = WhisperASR(cfg["asr"], stub=stub, device=self.device)
        self.vsr = (
            AVHubertVSR(cfg["vsr"], stub=stub, device=self.device)
            if self.ger_mode in {"av", "visual_only"}
            else None
        )

        # Mouth-ROI extractor (av-hubert preprocessing, ported from align_mouth.py)
        # backend='dlib'  → production: identical to av-hubert official pipeline.
        #   Requires: shape_predictor_68_face_landmarks.dat, mmod_human_face_detector.dat,
        #             20words_mean_face.npy  (see avsd_ger/frontend/mouth_roi.py for links)
        # backend='haar'  → fallback: no external model files, works out of the box.
        roi_cfg = cfg.get("mouth_roi", {})
        roi_backend = str(roi_cfg.get("backend", "haar"))
        self.mouth_roi_extractor = MouthROIExtractor(
            backend=roi_backend,
            crop_height=int(roi_cfg.get("crop_height", 48)),
            crop_width=int(roi_cfg.get("crop_width", 48)),
            window_margin=int(roi_cfg.get("window_margin", 12)),
            face_predictor_path=roi_cfg.get("face_predictor_path"),
            cnn_detector_path=roi_cfg.get("cnn_detector_path"),
            mean_face_path=roi_cfg.get("mean_face_path"),
        )

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
        self.enable_ger_safety_gate = bool(cfg["feedback"].get("enable_ger_safety_gate", True))
        self.enable_ger_artifact_gate = bool(cfg["feedback"].get("enable_ger_artifact_gate", True))
        self.enable_ger_length_gate = bool(cfg["feedback"].get("enable_ger_length_gate", True))
        self.enable_ger_overlap_gate = bool(cfg["feedback"].get("enable_ger_overlap_gate", True))
        self.enable_ger_acoustic_fallback = bool(cfg["feedback"].get("enable_ger_acoustic_fallback", True))
        self.ger_fallback_min_conf = float(cfg["feedback"].get("ger_fallback_min_conf", 0.20))
        self.ger_fallback_margin = float(cfg["feedback"].get("ger_fallback_margin", 0.0))
        self.ger_max_len_ratio = float(cfg["feedback"].get("ger_max_len_ratio", 1.8))
        self.ger_min_token_overlap = float(cfg["feedback"].get("ger_min_token_overlap", 0.50))
        self.ger_artifact_blacklist = [
            str(x).lower()
            for x in cfg["feedback"].get(
                "ger_artifact_blacklist",
                [
                    "Please provide",
                    "I'm happy to help",
                    "Here is the corrected transcript",
                    "Audio hypothesis:",
                    "Corrected transcript:",
                    "transcript provided",
                    "speaker label",
                ],
            )
        ]
        # Default AMI-safe behavior: C3 may accept/reject/retry, but it does
        # not self-train the identity pool unless explicitly enabled.
        self.enable_pool_update = bool(cfg["feedback"].get("enable_pool_update", False))

        # Ablation flags (spec section 10, Table 2). All False by default
        # = "Full Model". Flip one at a time from config to run rows of
        # the ablation table without code changes.
        abl = cfg.get("ablation", {}) or {}
        self.disable_c1 = bool(abl.get("disable_c1", False))                # bypass ID conditioning
        self.disable_c2 = bool(abl.get("disable_c2", False))                # skip GER; return ASR 1-best
        self.disable_c3 = bool(abl.get("disable_c3", False))                # skip closed-loop
        self.disable_conf_gate = bool(abl.get("disable_conf_gate", False))  # EMA-update unconditionally

    @staticmethod
    def _gate_tokens(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9']+", text.lower())

    def _ger_safety_reject_reason(self, ger_text: str, asr_text: str) -> str | None:
        if not self.enable_ger_safety_gate:
            return None

        ger = ger_text.strip()
        asr = asr_text.strip()
        if not ger:
            return "GER cleaned to empty text"

        ger_lower = ger.lower()
        if self.enable_ger_artifact_gate:
            for artifact in self.ger_artifact_blacklist:
                if artifact and artifact in ger_lower:
                    return f"GER artifact matched blacklist: {artifact!r}"

        asr_tokens = self._gate_tokens(asr)
        ger_tokens = self._gate_tokens(ger)
        if asr_tokens and ger_tokens:
            if (
                self.enable_ger_length_gate
                and len(ger_tokens) > max(len(asr_tokens) + 8, int(len(asr_tokens) * self.ger_max_len_ratio))
            ):
                return (
                    "GER too long "
                    f"({len(ger_tokens)} vs ASR {len(asr_tokens)} tokens)"
                )
            overlap = len(set(asr_tokens) & set(ger_tokens)) / max(1, len(set(asr_tokens)))
            if self.enable_ger_overlap_gate and overlap < self.ger_min_token_overlap:
                return (
                    "GER/ASR token overlap too low "
                    f"({overlap:.2f} < {self.ger_min_token_overlap:.2f})"
                )
        return None

    def _ger_safety_features(self, ger_text: str, asr_text: str) -> dict[str, Any]:
        asr_tokens = self._gate_tokens(asr_text)
        ger_tokens = self._gate_tokens(ger_text)
        overlap = None
        if asr_tokens and ger_tokens:
            overlap = len(set(asr_tokens) & set(ger_tokens)) / max(1, len(set(asr_tokens)))
        artifacts = [
            artifact
            for artifact in self.ger_artifact_blacklist
            if artifact and artifact in ger_text.lower()
        ]
        return {
            "asr_token_count": len(asr_tokens),
            "ger_token_count": len(ger_tokens),
            "length_ratio": (len(ger_tokens) / max(1, len(asr_tokens))) if asr_tokens else None,
            "token_overlap": overlap,
            "artifact_hits": artifacts,
            "gate_enabled": self.enable_ger_safety_gate,
            "artifact_gate_enabled": self.enable_ger_artifact_gate,
            "length_gate_enabled": self.enable_ger_length_gate,
            "overlap_gate_enabled": self.enable_ger_overlap_gate,
            "acoustic_fallback_enabled": self.enable_ger_acoustic_fallback,
            "max_len_ratio": self.ger_max_len_ratio,
            "min_token_overlap": self.ger_min_token_overlap,
        }

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
        video_frames: torch.Tensor | None = None,
        video_path: str | None = None,
        face_image: np.ndarray | None = None,
        has_visual: bool = True,
        speaker_mask_v: torch.Tensor | None = None,
        snr_per_tok: torch.Tensor | None = None,
        lip_conf_v: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        asr_out = self.asr.transcribe(audio_wav)
        wants_visual = self.ger_mode in {"av", "visual_only"}
        input_debug = {
            "audio": _tensor_debug(audio_wav),
            "video_frames_input": _tensor_debug(video_frames),
            "face_image": _tensor_debug(face_image),
            "has_visual_flag": bool(has_visual),
            "video_path": video_path,
            "speaker_mask_v": _tensor_debug(speaker_mask_v),
            "snr_per_tok": _tensor_debug(snr_per_tok),
            "lip_conf_v": _tensor_debug(lip_conf_v),
        }

        # Auto-extract mouth ROI from video_path if video_frames not already provided
        if video_frames is None and video_path is not None and wants_visual:
            try:
                video_frames = self.mouth_roi_extractor.extract_from_file(video_path)
                video_frames = video_frames.to(self.device)
            except Exception as _e:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "MouthROIExtractor failed for %r (%s); falling back to audio-only.",
                    video_path, _e,
                )
                video_frames = None

        use_visual = bool(wants_visual and has_visual and video_frames is not None)
        if use_visual and self.vsr is not None:
            vsr_out = self.vsr.extract(video_frames)
        else:
            # Explicit audio-only path: no random mouth ROI, no lip_hyp, and no
            # <AV_CTX>. A single zero frame only keeps tensor shapes available
            # for code paths that still construct an aligner output.
            vsr_out = {
                "vsr_features": torch.zeros(1, AVHubertVSR.FEATURE_DIM, device=self.device),
                "lip_hyp": "",
            }
        effective_ger_mode = self.ger_mode if use_visual else "audio_only"
        use_av_context = effective_ger_mode in {"av", "visual_only"}
        visual_debug = {
            "requested_ger_mode": self.ger_mode,
            "effective_ger_mode": effective_ger_mode,
            "wants_visual": wants_visual,
            "has_visual": use_visual,
            "use_av_context": use_av_context,
            "video_frames": _tensor_debug(video_frames),
            "vsr_features": _tensor_debug(vsr_out.get("vsr_features")),
            "lip_hyp": vsr_out.get("lip_hyp", ""),
            "lip_hyp_token_count": len(self._gate_tokens(vsr_out.get("lip_hyp", ""))),
            "vsr_emit_text": bool(getattr(self.vsr, "emit_text", False)) if self.vsr is not None else False,
            "vsr_generator_built": bool(getattr(self.vsr, "_generator", None)) if self.vsr is not None else False,
            "vsr_decode_error": getattr(self.vsr, "last_decode_error", None) if self.vsr is not None else None,
            "vsr_decode_token_count": int(getattr(self.vsr, "last_decode_token_count", 0)) if self.vsr is not None else 0,
            "vsr_target_dictionary_size": (
                len(getattr(getattr(self.vsr, "_task", None), "target_dictionary", []) or [])
                if self.vsr is not None else 0
            ),
        }

        if asr_out.encoder_features is None:
            T_a = 150
            asr_feats = torch.randn(T_a, WhisperASR.ENCODER_DIM, device=self.device)
        else:
            asr_feats = asr_out.encoder_features.to(self.device)
        asr_tok_feats = pool_encoder_to_tokens(
            asr_feats, asr_out.words, frame_rate_hz=asr_out.frame_rate_hz
        )
        asr_debug = {
            "nbest": list(asr_out.nbest),
            "nbest_scores": [float(x) for x in asr_out.nbest_scores],
            "top": asr_out.nbest[0] if asr_out.nbest else "",
            "frame_rate_hz": float(asr_out.frame_rate_hz),
            "encoder_features": _tensor_debug(asr_feats),
            "token_features": _tensor_debug(asr_tok_feats),
            "words": [
                {
                    "word": str(getattr(w, "word", "")),
                    "start": float(getattr(w, "start", 0.0)),
                    "end": float(getattr(w, "end", 0.0)),
                }
                for w in asr_out.words
            ],
        }

        voice_emb = self.voice.embed(audio_wav)
        if face_image is None:
            face_emb = torch.zeros(FaceEncoder.EMB_DIM, device=self.device)
        else:
            face_emb = self.face.embed(face_image)
        embedding_debug = {
            "voice_emb": _tensor_debug(voice_emb),
            "face_emb": _tensor_debug(face_emb),
        }

        trace: list[dict[str, Any]] = []
        skip_ids: set[str] = set()
        id_q = self.pool.query(voice_emb, face_emb, skip_ids=skip_ids)
        c1_initial_debug = {
            "disable_c1": self.disable_c1,
            "pool_size": len(self.pool),
            "pool_speaker_ids": sorted(list(self.pool._speakers.keys())),
            "top_ids": list(id_q.top_ids),
            "top_scores": [float(x) for x in id_q.top_scores],
            "av_consistency_raw": float(id_q.av_consistency),
            "is_unknown": bool(id_q.is_unknown),
            "z_id": _tensor_debug(id_q.z_id),
        }

        # Ablation: w/o C1 -> zero out z_id so C2 degenerates to modality-only
        if self.disable_c1:
            id_q = IdentityQueryResult(
                top_ids=[], top_scores=[], av_consistency=0.0,
                z_id=torch.zeros_like(id_q.z_id), is_unknown=True,
            )
        c1_effective_debug = {
            "top_ids": list(id_q.top_ids),
            "top_scores": [float(x) for x in id_q.top_scores],
            "av_consistency_raw": float(id_q.av_consistency),
            "is_unknown": bool(id_q.is_unknown),
            "z_id": _tensor_debug(id_q.z_id),
        }

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
            align_debug = {
                "asr_token_features": _tensor_debug(asr_tok_feats),
                "vsr_features": _tensor_debug(vsr_out["vsr_features"]),
                "f_align": _tensor_debug(f_align),
                "speaker_mask_v_present": speaker_mask_v is not None,
                "snr_per_tok_present": snr_per_tok is not None,
                "lip_conf_v_present": lip_conf_v is not None,
            }

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
                    mode=effective_ger_mode,
                    use_av_context=use_av_context,
                )

            fallback_applied = False
            fallback_reason = None
            asr_top = (asr_out.nbest[0].strip() if asr_out.nbest else "")
            raw_generated_before_gate = ger_out.get("raw_text")
            cleaned_generated_before_gate = ger_out.get("text", "")

            if not self.disable_c2 and asr_top:
                fallback_reason = self._ger_safety_reject_reason(ger_out["text"], asr_top)

            if fallback_reason is not None:
                raw_generation = ger_out.get("raw_text")
                raw_ger_text = ger_out.get("text", "")
                ger_out = dict(ger_out)
                ger_out["raw_ger_text"] = raw_ger_text
                ger_out["raw_generation"] = raw_generation
                ger_out["text"] = asr_top
                ger_out["token_logprobs"] = torch.zeros(0, device=self.device)
                fallback_applied = True
                fallback_reason = f"{fallback_reason}; used ASR 1-best"

            # The GER head is useful only when its corrected text is still
            # acoustically plausible. In zero-shot or untrained-LoRA smoke
            # runs, Llama can sometimes emit a prompt fragment such as
            # "The speaker label."; Whisper rescoring catches that sharply.
            # Fall back to ASR 1-best instead of accepting an LLM artifact.
            s_acoustic = self.asr.rescore(audio_wav, ger_out["text"])
            s_acoustic_conf = squash_logprob(s_acoustic)
            asr_s_acoustic = None
            asr_s_acoustic_conf = None
            raw_ger_conf = None
            if (
                not self.disable_c2
                and self.enable_ger_acoustic_fallback
                and asr_top
                and ger_out["text"].strip() != asr_top
            ):
                asr_s_acoustic = self.asr.rescore(audio_wav, asr_top)
                asr_s_acoustic_conf = squash_logprob(asr_s_acoustic)
                if (
                    s_acoustic_conf < self.ger_fallback_min_conf
                    or asr_s_acoustic_conf >= s_acoustic_conf + self.ger_fallback_margin
                ):
                    raw_ger_text = ger_out["text"]
                    raw_generation = ger_out.get("raw_text")
                    raw_ger_conf = s_acoustic_conf
                    ger_out = dict(ger_out)
                    ger_out["raw_ger_text"] = raw_ger_text
                    ger_out["raw_generation"] = raw_generation
                    ger_out["text"] = asr_top
                    ger_out["token_logprobs"] = torch.zeros(0, device=self.device)
                    s_acoustic = asr_s_acoustic
                    s_acoustic_conf = asr_s_acoustic_conf
                    fallback_applied = True
                    if raw_ger_conf < self.ger_fallback_min_conf:
                        fallback_reason = (
                            "GER acoustic confidence "
                            f"{raw_ger_conf:.3f} "
                            f"< {self.ger_fallback_min_conf:.3f}; used ASR 1-best"
                        )
                    else:
                        fallback_reason = (
                            "ASR acoustic confidence "
                            f"{asr_s_acoustic_conf:.3f} >= GER {raw_ger_conf:.3f} "
                            f"+ margin {self.ger_fallback_margin:.3f}; used ASR 1-best"
                        )
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
                "top_scores": [float(x) for x in id_q.top_scores],
                "is_unknown": bool(id_q.is_unknown),
                "av_consistency_raw": float(id_q.av_consistency),
                "speaker_id_hint": speaker_id_hint,
                "total_conf": rep.total,
                "s_acoustic": s_acoustic,
                "s_acoustic_conf": s_acoustic_conf,
                "asr_s_acoustic": asr_s_acoustic,
                "asr_s_acoustic_conf": asr_s_acoustic_conf,
                "raw_ger_acoustic_conf": raw_ger_conf,
                "components": rep.components,
                "decision": decision.action.value,
                "reason": decision.reason,
                "text": ger_out["text"],
                "asr_top": asr_top,
                "asr_nbest": list(asr_out.nbest),
                "asr_nbest_scores": [float(x) for x in asr_out.nbest_scores],
                "lip_hyp": vsr_out.get("lip_hyp", ""),
                "prompt": ger_out.get("prompt"),
                "raw_generation_before_gate": raw_generated_before_gate,
                "cleaned_ger_text_before_gate": cleaned_generated_before_gate,
                "safety_features": self._ger_safety_features(cleaned_generated_before_gate, asr_top),
                "fallback_applied": fallback_applied,
                "fallback_reason": fallback_reason,
                "raw_ger_text": ger_out.get("raw_ger_text") or ger_out.get("raw_text"),
                "raw_generation": ger_out.get("raw_generation"),
                "ger_mode": effective_ger_mode,
                "has_visual": use_visual,
                "use_av_context": use_av_context,
                "alignment": align_debug,
            })

            if decision.action == LoopAction.ACCEPT_AND_UPDATE:
                if not self.enable_pool_update and not self.disable_conf_gate:
                    decision = LoopDecision(
                        LoopAction.ACCEPT_NO_UPDATE,
                        f"{decision.reason}; pool update disabled by feedback.enable_pool_update=false",
                        decision.iteration,
                    )
                    trace[-1]["decision"] = decision.action.value
                    trace[-1]["reason"] = decision.reason
                elif speaker_id_hint is not None and not self.disable_c3:
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
                and not self.disable_c3
                and (self.enable_pool_update or self.disable_conf_gate)
                if decision else False
            ),
            "trace": trace,
            "debug": {
                "input": input_debug,
                "asr": asr_debug,
                "visual": visual_debug,
                "embeddings": embedding_debug,
                "c1_initial": c1_initial_debug,
                "c1_effective": c1_effective_debug,
                "c2": {
                    "disable_c2": self.disable_c2,
                    "speaker_special_token": self.ger.speaker_special_token,
                    "use_av_context": use_av_context,
                    "mode": effective_ger_mode,
                },
                "c3": {
                    "disable_c3": self.disable_c3,
                    "disable_conf_gate": self.disable_conf_gate,
                    "max_iters": max_iters,
                    "enable_pool_update": self.enable_pool_update,
                    "enable_ger_safety_gate": self.enable_ger_safety_gate,
                    "enable_ger_acoustic_fallback": self.enable_ger_acoustic_fallback,
                },
                "final": {
                    "text": ger_out["text"] if ger_out else "",
                    "speaker_id": id_q.top_ids[0] if (id_q.top_ids and not id_q.is_unknown) else None,
                    "confidence": rep.total if rep else 0.0,
                    "iterations": len(trace),
                    "pool_updated": (
                        decision.action == LoopAction.ACCEPT_AND_UPDATE
                        and not self.disable_c3
                        and (self.enable_pool_update or self.disable_conf_gate)
                        if decision else False
                    ),
                },
            },
        }

    @staticmethod
    def _first_frame_as_rgb(video: torch.Tensor) -> np.ndarray:
        frame = video[0, 0].detach().cpu().numpy()
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        return np.stack([frame, frame, frame], axis=-1)
