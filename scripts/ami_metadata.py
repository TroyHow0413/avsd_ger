"""AMI corpus-resource metadata helpers.

The per-meeting transcript files identify speakers with local NXT agents such
as ``A`` and ``B``.  Those labels are not corpus-global identities, and their
Closeup camera is not a fixed ``A=Closeup1`` mapping.  The authoritative
mapping lives in ``corpusResources/meetings.xml``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def load_ami_meetings(path: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Return ``meeting -> nxt_agent -> authoritative speaker metadata``."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"AMI meetings.xml not found: {source}")

    root = ET.parse(source).getroot()
    meetings: dict[str, dict[str, dict[str, Any]]] = {}
    for meeting in root:
        if _local_name(meeting.tag) != "meeting":
            continue
        meeting_id = str(meeting.attrib.get("observation", "")).strip()
        if not meeting_id:
            continue
        speakers: dict[str, dict[str, Any]] = {}
        for speaker in meeting:
            if _local_name(speaker.tag) != "speaker":
                continue
            agent = str(speaker.attrib.get("nxt_agent", "")).strip()
            if not agent:
                continue
            channel_raw = speaker.attrib.get("channel")
            try:
                channel = int(channel_raw) if channel_raw is not None else None
            except ValueError:
                channel = None
            speakers[agent] = {
                "nxt_agent": agent,
                "participant_id": str(speaker.attrib.get("global_name", "")).strip() or None,
                "camera": str(speaker.attrib.get("camera", "")).strip() or None,
                "channel": channel,
                "role": str(speaker.attrib.get("role", "")).strip() or None,
            }
        meetings[meeting_id] = speakers
    return meetings


def infer_nxt_agent(value: str | None, meeting_id: str) -> str | None:
    """Recover the local agent from legacy IDs such as ``ES2004a_A``."""
    if not value:
        return None
    text = str(value)
    prefix = f"{meeting_id}_"
    if text.startswith(prefix):
        suffix = text[len(prefix):]
        return suffix if suffix else None
    return text if len(text) == 1 and text.isalpha() else None


def apply_ami_speaker_metadata(
    manifest: dict[str, Any],
    meeting_id: str,
    meeting_speakers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Attach global identities/cameras and rewrite references consistently.

    ``speaker_id`` and ``ref_speaker`` become the AMI global participant ID
    when available.  ``session_speaker_id`` and ``nxt_agent`` retain the local
    meeting identity needed to join transcript, microphone, and video streams.
    """
    out = dict(manifest)
    old_to_agent: dict[str, str] = {}
    speakers_out: list[dict[str, Any]] = []

    for speaker in manifest.get("speakers", []):
        row = dict(speaker)
        old_id = str(row.get("speaker_id", ""))
        agent = str(row.get("nxt_agent", "")).strip() or infer_nxt_agent(old_id, meeting_id)
        if not agent or agent not in meeting_speakers:
            raise ValueError(
                f"No meetings.xml entry for speaker {old_id!r} in meeting {meeting_id}"
            )
        meta = meeting_speakers[agent]
        participant_id = meta.get("participant_id")
        session_speaker_id = f"{meeting_id}_{agent}"
        if old_id:
            old_to_agent[old_id] = agent
        old_to_agent[session_speaker_id] = agent
        row.update(
            {
                "speaker_id": participant_id or session_speaker_id,
                "participant_id": participant_id,
                "session_speaker_id": session_speaker_id,
                "nxt_agent": agent,
                "camera": meta.get("camera"),
                "channel": meta.get("channel"),
                "role": meta.get("role"),
                "identity_scope": "corpus_global" if participant_id else "meeting_local",
            }
        )
        speakers_out.append(row)

    turns_out: list[dict[str, Any]] = []
    for turn in manifest.get("turns", []):
        row = dict(turn)
        old_ref = str(row.get("ref_speaker", ""))
        agent = str(row.get("nxt_agent", "")).strip()
        if not agent:
            agent = old_to_agent.get(old_ref) or infer_nxt_agent(old_ref, meeting_id)
        if not agent or agent not in meeting_speakers:
            raise ValueError(
                f"No meetings.xml entry for turn speaker {old_ref!r} in meeting {meeting_id}"
            )
        meta = meeting_speakers[agent]
        participant_id = meta.get("participant_id")
        session_speaker_id = f"{meeting_id}_{agent}"
        row.update(
            {
                "ref_speaker": participant_id or session_speaker_id,
                "participant_id": participant_id,
                "session_speaker_id": session_speaker_id,
                "nxt_agent": agent,
                "camera": meta.get("camera"),
                "channel": meta.get("channel"),
                "identity_scope": "corpus_global" if participant_id else "meeting_local",
            }
        )
        turns_out.append(row)

    meta_out = dict(manifest.get("meta", {}))
    meta_out.update(
        {
            "meeting_id": meeting_id,
            "speaker_identity_source": "AMI corpusResources/meetings.xml",
            "speaker_identity_scope": "corpus_global",
            "camera_mapping_source": "AMI corpusResources/meetings.xml",
        }
    )
    out["meeting_id"] = meeting_id
    out["speakers"] = speakers_out
    out["turns"] = turns_out
    out["meta"] = meta_out
    return out


def meeting_closeup_map(meeting_speakers: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Return the verified local-agent to Closeup camera mapping."""
    mapping = {
        agent: str(meta["camera"])
        for agent, meta in meeting_speakers.items()
        if meta.get("camera")
    }
    if not mapping:
        raise ValueError("meetings.xml contains no camera mapping for this meeting")
    return mapping
