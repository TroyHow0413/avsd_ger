"""Canonical AVSD frontend profiles used for robustness experiments.

The core AVSD-GER model operates on turn-level manifests. A raw meeting video
therefore needs an upstream frontend that proposes who-spoke-when segments,
active face tracks, and mouth ROIs. This registry makes those choices explicit
so reports can compare oracle, common, strong, and degraded frontends without
hard-coding prose into every script.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrontendProfile:
    """Metadata for one raw-video frontend condition."""

    key: str
    label: str
    tier: str
    diarization: str
    active_speaker: str
    face_tracking: str
    mouth_roi: str
    role: str
    claim: str
    manifest_contract: tuple[str, ...]


_PROFILES: dict[str, FrontendProfile] = {
    "oracle_turns": FrontendProfile(
        key="oracle_turns",
        label="Oracle turns",
        tier="upper-bound",
        diarization="Ground-truth segment boundaries and reference speaker labels",
        active_speaker="Ground-truth or manually aligned active face track",
        face_tracking="Dataset annotations or manually verified tracks",
        mouth_roi="AV-HuBERT align_mouth.py or verified precomputed mouth ROI",
        role="Upper bound for C1/C2/C3 when frontend segmentation is not the bottleneck.",
        claim=(
            "Shows the best-case recognition and speaker-attribution gain of "
            "identity-conditioned GER under clean turn boundaries."
        ),
        manifest_contract=("turns", "audio", "mouth_roi", "ref_text", "ref_speaker"),
    ),
    "common_pyannote_lightasd": FrontendProfile(
        key="common_pyannote_lightasd",
        label="Common open-source frontend",
        tier="common-backbone",
        diarization="pyannote speaker-diarization-community-1",
        active_speaker="Light-ASD or TalkNet",
        face_tracking="RetinaFace or InsightFace detector + SORT/ByteTrack tracking",
        mouth_roi="Landmark-based crop, preferably AV-HuBERT align_mouth.py",
        role="Main practical baseline: widely used, local, and reproducible.",
        claim=(
            "Demonstrates that AVSD-GER improves speaker-aware transcripts even "
            "with a normal open-source diarization/ASD frontend."
        ),
        manifest_contract=("turns", "audio", "mouth_roi", "speaker_mask_v"),
    ),
    "strong_sortformer_talknet": FrontendProfile(
        key="strong_sortformer_talknet",
        label="Strong diarization frontend",
        tier="strong/sota-reference",
        diarization="NVIDIA NeMo Sortformer v2.1 or pyannote Precision-2",
        active_speaker="TalkNet or Light-ASD with verified face tracks",
        face_tracking="RetinaFace/InsightFace + ByteTrack/DeepSORT",
        mouth_roi="AV-HuBERT align_mouth.py on active speaker tracks",
        role="High-quality frontend reference for near-SOTA segmentation conditions.",
        claim=(
            "Shows whether C1/C2/C3 continue to add value when segmentation is "
            "already strong."
        ),
        manifest_contract=("turns", "audio", "mouth_roi", "speaker_mask_v"),
    ),
    "degraded_pyannote": FrontendProfile(
        key="degraded_pyannote",
        label="Degraded common frontend",
        tier="robustness",
        diarization="pyannote community frontend with synthetic boundary jitter/drop/swap noise",
        active_speaker="Light-ASD/TalkNet outputs with optional track noise",
        face_tracking="Same as common frontend, then perturb track assignment",
        mouth_roi="Same ROI extractor as common frontend",
        role="Stress test for imperfect turn splitting and speaker assignment.",
        claim=(
            "Side proof that the framework does not require a perfect AVSD "
            "frontend to retain gains."
        ),
        manifest_contract=("turns", "audio", "mouth_roi", "frontend_noise"),
    ),
}


def list_frontend_profiles() -> list[FrontendProfile]:
    """Return profiles in the recommended reporting order."""

    return [
        _PROFILES["oracle_turns"],
        _PROFILES["common_pyannote_lightasd"],
        _PROFILES["strong_sortformer_talknet"],
        _PROFILES["degraded_pyannote"],
    ]


def get_frontend_profile(key: str) -> FrontendProfile:
    """Return one profile by key, raising a helpful error for unknown keys."""

    try:
        return _PROFILES[key]
    except KeyError as exc:
        known = ", ".join(sorted(_PROFILES))
        raise KeyError(f"Unknown frontend profile {key!r}. Known profiles: {known}") from exc


def render_frontend_profiles_markdown() -> str:
    """Render the registry as a compact Markdown table."""

    lines = [
        "| Key | Tier | Diarization | Active speaker | Claim |",
        "|---|---|---|---|---|",
    ]
    for p in list_frontend_profiles():
        lines.append(
            f"| `{p.key}` | {p.tier} | {p.diarization} | "
            f"{p.active_speaker} | {p.claim} |"
        )
    return "\n".join(lines)
