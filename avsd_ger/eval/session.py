"""Multi-speaker session runner.

`AVSDGERPipeline.run()` is a single-speaker-per-utterance path -- that's the
spec's intentional execution shape (identity is attached per turn, not per
frame). A real meeting has multiple speakers taking turns, so we need a
wrapper that:

    1. Iterates over session turns.
    2. Calls the single-speaker pipeline with the correct speaker-masked
       VSR frames, SNR-per-token, and lip-conf-per-frame inputs.
    3. Collects per-turn outputs and stitches them into a time-ordered
       transcript that SA-WER / SCR / DER can consume.

The wrapper deliberately does NOT do diarization itself. Diarization output
(who spoke when) is a *pre-condition* for a turn list. If the caller wants
to stress-test ID attribution under noisy diarization they perturb the
turn boundaries before calling us. That keeps the two concerns cleanly
separated and matches spec section 13 (SA-WER = text correctness + speaker
attribution, given segmentation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import torch

from ..pipeline import AVSDGERPipeline


@dataclass
class SessionTurn:
    """One utterance by one (assumed) speaker inside a session.

    Required:
        turn_id       -- string ID unique within the session
        start, end    -- seconds, relative to session start (for ordering/DER)
        audio_wav     -- per-turn mono waveform (torch.Tensor | np.ndarray)
        video_frames  -- per-turn lip ROI video (torch.Tensor, shape per VSR backbone)
        has_visual    -- True only when video_frames came from a real mouth ROI

    Optional (passed through to the pipeline):
        face_image    -- RGB face image for this speaker (falls back to first frame)
        speaker_mask_v-- bool [T_v]; True where the nominal speaker is active
        snr_per_tok   -- float [N_tok] in [0,1]
        lip_conf_v    -- float [T_v]   in [0,1]

    Optional (evaluation-only; attached to the result so metrics can consume
    everything in one object):
        ref_text      -- reference transcript for SA-WER/SCR
        ref_speaker   -- reference speaker label for AV-SID Acc / DER / JER
    """

    turn_id: str
    start: float
    end: float
    audio_wav: torch.Tensor | np.ndarray
    video_frames: torch.Tensor | None
    has_visual: bool = True
    face_image: np.ndarray | None = None
    speaker_mask_v: torch.Tensor | None = None
    snr_per_tok: torch.Tensor | None = None
    lip_conf_v: torch.Tensor | None = None
    ref_text: str | None = None
    ref_speaker: str | None = None


@dataclass
class SessionTurnResult:
    """Per-turn output after the pipeline has run."""

    turn_id: str
    start: float
    end: float
    hyp_text: str
    hyp_speaker: str | None
    confidence: float
    s_acoustic: float | None
    iterations: int
    pool_updated: bool
    trace: list[dict[str, Any]] = field(default_factory=list)
    # Passthrough ground truth so metrics don't need the original SessionTurn.
    ref_text: str | None = None
    ref_speaker: str | None = None


@dataclass
class SessionResult:
    """Full session output -- everything SA-WER / SCR / DER need.

    Attributes:
        turns       -- per-turn results, same order as input turns (stitched by time).
        transcript  -- human-readable joined transcript
                       ("[Speaker: ID_i] text\n[Speaker: ID_j] text\n...").
        speaker_order -- distinct speaker IDs in first-spoken order (useful for
                         session-level reporting + AV-SID mapping diagnostics).
    """

    turns: list[SessionTurnResult]
    transcript: str = ""
    speaker_order: list[str] = field(default_factory=list)


class SessionRunner:
    """Fan out the single-speaker pipeline across a session of turns.

    Usage:
        runner = SessionRunner(pipeline)
        session_result = runner.run(turns)
        # Then feed session_result.turns to avsd_ger.eval.metrics.*.
    """

    def __init__(self, pipeline: AVSDGERPipeline):
        self.pipeline = pipeline

    def run(self, turns: Iterable[SessionTurn]) -> SessionResult:
        turns = list(turns)
        # Stitch by start time so that trace ordering survives out-of-order input.
        turns.sort(key=lambda t: (t.start, t.end, t.turn_id))

        results: list[SessionTurnResult] = []
        speaker_order: list[str] = []
        seen_speakers: set[str] = set()
        transcript_lines: list[str] = []

        for turn in turns:
            out = self.pipeline.run(
                audio_wav=turn.audio_wav,
                video_frames=turn.video_frames,
                face_image=turn.face_image,
                has_visual=turn.has_visual,
                speaker_mask_v=turn.speaker_mask_v,
                snr_per_tok=turn.snr_per_tok,
                lip_conf_v=turn.lip_conf_v,
            )
            hyp_text = out.get("text", "") or ""
            hyp_speaker = out.get("speaker_id")
            tr = SessionTurnResult(
                turn_id=turn.turn_id,
                start=turn.start,
                end=turn.end,
                hyp_text=hyp_text,
                hyp_speaker=hyp_speaker,
                confidence=float(out.get("confidence", 0.0) or 0.0),
                s_acoustic=out.get("s_acoustic"),
                iterations=int(out.get("iterations", 0) or 0),
                pool_updated=bool(out.get("pool_updated", False)),
                trace=list(out.get("trace", []) or []),
                ref_text=turn.ref_text,
                ref_speaker=turn.ref_speaker,
            )
            results.append(tr)

            spk_tag = hyp_speaker if hyp_speaker is not None else "UNKNOWN"
            if hyp_speaker is not None and hyp_speaker not in seen_speakers:
                seen_speakers.add(hyp_speaker)
                speaker_order.append(hyp_speaker)

            transcript_lines.append(f"[Speaker: {spk_tag}] {hyp_text}")

        return SessionResult(
            turns=results,
            transcript="\n".join(transcript_lines),
            speaker_order=speaker_order,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def turns_from_manifest(manifest: list[dict[str, Any]]) -> list[SessionTurn]:
        """Convert a plain-dict manifest (e.g. decoded JSON) into SessionTurns.

        This is a convenience for eval scripts that read a .jsonl manifest
        describing a session. Tensor fields are expected to already be
        loaded by the caller (we just do a light structural coercion).
        """
        turns: list[SessionTurn] = []
        for i, row in enumerate(manifest):
            turns.append(SessionTurn(
                turn_id=str(row.get("turn_id", f"t{i:04d}")),
                start=float(row.get("start", i)),
                end=float(row.get("end", i + 1)),
                audio_wav=row["audio_wav"],
                video_frames=row["video_frames"],
                has_visual=bool(row.get("has_visual", True)),
                face_image=row.get("face_image"),
                speaker_mask_v=row.get("speaker_mask_v"),
                snr_per_tok=row.get("snr_per_tok"),
                lip_conf_v=row.get("lip_conf_v"),
                ref_text=row.get("ref_text"),
                ref_speaker=row.get("ref_speaker"),
            ))
        return turns
