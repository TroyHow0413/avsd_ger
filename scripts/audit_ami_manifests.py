"""Audit repaired AMI visual manifests and fail on known pipeline regressions."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_split(path: Path) -> tuple[dict[str, Any], set[str], set[str], set[str]]:
    files = sorted(path.glob("*.json"))
    stats: Counter[str] = Counter()
    participants: set[str] = set()
    meetings: set[str] = set()
    build_ids: set[str] = set()
    durations = 0.0

    for manifest_path in files:
        manifest = _load(manifest_path)
        meta = manifest.get("meta", {})
        if meta.get("dataset_build_id"):
            build_ids.add(str(meta["dataset_build_id"]))
        meeting_id = str(meta.get("meeting_id") or manifest.get("meeting_id") or manifest_path.stem)
        meetings.add(meeting_id)
        turns = manifest.get("turns", [])
        stats["manifests"] += 1
        stats["turns"] += len(turns)
        stats["meetings_with_exactly_50_turns"] += int(len(turns) == 50)
        stats["manifests_with_turn_cap"] += int(meta.get("turn_limit") is not None)
        if meta.get("speaker_closeup_source") == "manual_unverified":
            stats["unverified_camera_mappings"] += 1
        audio_condition = meta.get("audio_condition", {})
        if audio_condition.get("description") == "individual_headset_microphone":
            stats["ihm_manifests"] += 1
        if meta.get("turn_boundary_source") == "oracle_reference_transcript":
            stats["oracle_turn_manifests"] += 1

        for speaker in manifest.get("speakers", []):
            enrollment_mode = speaker.get("meta", {}).get("enrollment_mode")
            if enrollment_mode not in {"turn_quality", "existing_quality_enrollment"}:
                stats["non_quality_enrollment_entries"] += 1
            participant = speaker.get("participant_id")
            if participant:
                participants.add(str(participant))
                stats["global_speaker_entries"] += 1
            else:
                stats["missing_global_speaker_entries"] += 1

        for turn in turns:
            durations += max(0.0, float(turn.get("end", 0.0)) - float(turn.get("start", 0.0)))
            confidence = turn.get("lip_conf_v")
            if confidence is None:
                stats["missing_lip_conf_turns"] += 1
            else:
                values = np.asarray(confidence, dtype=np.float32).reshape(-1)
                stats["lip_conf_turns"] += 1
                stats["all_one_lip_conf_turns"] += int(
                    values.size > 0 and bool(np.allclose(values, 1.0))
                )
                roi_path = turn.get("mouth_roi")
                if roi_path and Path(roi_path).exists():
                    try:
                        frames = int(np.load(roi_path, mmap_mode="r").shape[0])
                        stats["lip_conf_length_mismatch"] += int(frames != values.size)
                    except Exception:
                        stats["unreadable_roi"] += 1

    return (
        {**dict(stats), "hours": durations / 3600.0},
        meetings,
        participants,
        build_ids,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/ami_full_v2"))
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--dev", type=Path, default=None)
    parser.add_argument("--test", type=Path, default=None)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    split_paths = {
        split: getattr(args, split) or (args.run_dir / split / "visual")
        for split in ("train", "dev", "test")
    }

    results: dict[str, dict[str, Any]] = {}
    meeting_sets: dict[str, set[str]] = {}
    participant_sets: dict[str, set[str]] = {}
    build_id_sets: dict[str, set[str]] = {}
    failed = False
    for split in ("train", "dev", "test"):
        stats, meetings, participants, build_ids = audit_split(split_paths[split])
        stats["dataset_build_ids"] = sorted(build_ids)
        results[split] = stats
        meeting_sets[split] = meetings
        participant_sets[split] = participants
        build_id_sets[split] = build_ids
        for forbidden in (
            "manifests_with_turn_cap",
            "unverified_camera_mappings",
            "missing_global_speaker_entries",
            "missing_lip_conf_turns",
            "lip_conf_length_mismatch",
            "unreadable_roi",
            "non_quality_enrollment_entries",
        ):
            failed |= stats.get(forbidden, 0) > 0
        failed |= stats.get("ihm_manifests", 0) != stats.get("manifests", 0)
        failed |= stats.get("oracle_turn_manifests", 0) != stats.get("manifests", 0)
        failed |= (
            stats.get("manifests", 0) > 0
            and stats.get("meetings_with_exactly_50_turns", 0) * 2
            > stats.get("manifests", 0)
        )
        failed |= build_ids != {args.run_dir.name}
        failed |= (
            stats.get("lip_conf_turns", 0) > 0
            and stats.get("all_one_lip_conf_turns", 0) == stats.get("lip_conf_turns", 0)
        )

    overlap = {
        "train_dev_meetings": sorted(meeting_sets["train"] & meeting_sets["dev"]),
        "train_test_meetings": sorted(meeting_sets["train"] & meeting_sets["test"]),
        "dev_test_meetings": sorted(meeting_sets["dev"] & meeting_sets["test"]),
        "train_dev_participants": sorted(participant_sets["train"] & participant_sets["dev"]),
        "train_test_participants": sorted(participant_sets["train"] & participant_sets["test"]),
        "dev_test_participants": sorted(participant_sets["dev"] & participant_sets["test"]),
    }
    failed |= any(overlap[key] for key in overlap if key.endswith("_meetings"))
    print(json.dumps({"splits": results, "overlap": overlap, "passed": not failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
