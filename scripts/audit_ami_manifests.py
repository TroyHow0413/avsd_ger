"""Audit repaired AMI manifests, including visual coverage and failure evidence."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ami_visual_policy import is_official_missing_closeup  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _eligible_turns(manifest: dict[str, Any], minimum: float, maximum: float) -> int:
    return sum(
        minimum
        <= float(turn.get("end", 0.0)) - float(turn.get("start", 0.0))
        <= maximum
        for turn in manifest.get("turns", [])
    )


def _read_failure_log(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    unreadable = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except (TypeError, ValueError):
            unreadable += 1
    return records, unreadable


def audit_split(
    path: Path,
    *,
    base_path: Path | None = None,
    min_turn_seconds: float = 1.0,
    max_turn_seconds: float = 12.0,
    min_frame_coverage: float = 0.90,
    max_frame_coverage: float = 1.10,
) -> tuple[dict[str, Any], set[str], set[str], set[str]]:
    files = sorted(path.glob("*.json"))
    stats: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    participants: set[str] = set()
    meetings: set[str] = set()
    build_ids: set[str] = set()
    durations = 0.0
    frame_coverages: list[float] = []

    base_files = sorted(base_path.glob("*.json")) if base_path is not None else []
    base_by_name = {item.name: item for item in base_files}
    visual_names = {item.name for item in files}
    base_names = set(base_by_name)
    missing_visual_manifests = sorted(base_names - visual_names)
    unexpected_visual_manifests = sorted(visual_names - base_names) if base_path else []
    stats["base_manifests"] = len(base_files)
    stats["missing_visual_manifests"] = len(missing_visual_manifests)
    stats["unexpected_visual_manifests"] = len(unexpected_visual_manifests)

    for manifest_path in files:
        manifest = _load(manifest_path)
        meta = manifest.get("meta", {})
        if meta.get("dataset_build_id"):
            build_ids.add(str(meta["dataset_build_id"]))
        meeting_id = str(
            meta.get("meeting_id") or manifest.get("meeting_id") or manifest_path.stem
        )
        meetings.add(meeting_id)
        turns = manifest.get("turns", [])
        attempts = int(meta.get("attempts", len(turns)))
        official_exclusions = int(meta.get("official_visual_exclusions", 0))
        duration_exclusions = int(meta.get("source_duration_exclusions", 0))
        source_exclusions = int(
            meta.get("visual_source_exclusions", official_exclusions)
        )
        eligible_turns = int(
            meta.get("eligible_turns", attempts + source_exclusions)
        )
        failures = int(meta.get("failures", 0))
        stats["manifests"] += 1
        stats["turns"] += len(turns)
        stats["processing_attempts"] += attempts
        stats["processing_failures"] += failures
        stats["eligible_turns"] += eligible_turns
        stats["visual_source_exclusions"] += source_exclusions
        stats["official_visual_exclusions"] += official_exclusions
        stats["source_duration_exclusions"] += duration_exclusions
        stats["manifests_with_processing_failures"] += int(failures > 0)
        stats["attempt_accounting_mismatch"] += int(attempts != len(turns) + failures)
        stats["eligible_accounting_mismatch"] += int(
            eligible_turns != attempts + source_exclusions
        )
        stats["source_exclusion_subtotal_mismatch"] += int(
            source_exclusions != official_exclusions + duration_exclusions
        )
        if meta.get("successful_visual_turns") is not None:
            stats["success_accounting_mismatch"] += int(
                int(meta["successful_visual_turns"]) != len(turns)
            )
        stats["meetings_with_exactly_50_turns"] += int(len(turns) == 50)
        stats["manifests_with_turn_cap"] += int(meta.get("turn_limit") is not None)
        stats["skipped_unmapped_turns"] += int(meta.get("skipped_unmapped", 0))
        if meta.get("speaker_closeup_source") == "manual_unverified":
            stats["unverified_camera_mappings"] += 1
        audio_condition = meta.get("audio_condition", {})
        if audio_condition.get("description") == "individual_headset_microphone":
            stats["ihm_manifests"] += 1
        if meta.get("turn_boundary_source") == "oracle_reference_transcript":
            stats["oracle_turn_manifests"] += 1

        base_manifest_path = base_by_name.get(manifest_path.name)
        if base_manifest_path is not None:
            expected = _eligible_turns(
                _load(base_manifest_path), min_turn_seconds, max_turn_seconds
            )
            stats["eligible_base_turns"] += expected
            gap = eligible_turns - expected
            stats["eligible_turn_gap_absolute"] += abs(gap)
            stats["manifests_with_eligible_turn_mismatch"] += int(gap != 0)

        failure_log = _artifact_path(meta.get("failure_log"))
        if failures > 0 and failure_log is None:
            stats["missing_failure_logs"] += 1
        elif failure_log is not None:
            if not failure_log.is_file():
                stats["missing_failure_log_files"] += 1
            else:
                records, unreadable = _read_failure_log(failure_log)
                stats["failure_log_records"] += len(records)
                stats["unreadable_failure_log_records"] += unreadable
                stats["failure_log_count_mismatch"] += int(len(records) != failures)
                for record in records:
                    failure_reasons[str(record.get("reason", "missing_reason"))] += 1
                    stats["failure_records_missing_turn_id"] += int(
                        not record.get("turn_id")
                    )

        exclusion_log = _artifact_path(
            meta.get("visual_source_exclusion_log")
            or meta.get("official_visual_exclusion_log")
        )
        if source_exclusions > 0 and exclusion_log is None:
            stats["missing_source_exclusion_logs"] += 1
        elif exclusion_log is not None:
            if not exclusion_log.is_file():
                stats["missing_source_exclusion_log_files"] += 1
            else:
                records, unreadable = _read_failure_log(exclusion_log)
                stats["source_exclusion_log_records"] += len(records)
                stats["unreadable_source_exclusion_log_records"] += unreadable
                stats["source_exclusion_log_count_mismatch"] += int(
                    len(records) != source_exclusions
                )
                for record in records:
                    reason = record.get("reason")
                    official_valid = (
                        reason == "official_missing_closeup"
                        and bool(record.get("turn_id"))
                        and is_official_missing_closeup(
                            str(record.get("meeting_id", "")),
                            str(record.get("closeup", "")),
                        )
                    )
                    source_probe = record.get("source_probe") or {}
                    source_duration = source_probe.get("reported_duration_seconds")
                    end = record.get("end")
                    tolerance = float(
                        meta.get("source_duration_tolerance_seconds", 0.25)
                    )
                    duration_valid = (
                        reason == "source_duration_out_of_bounds"
                        and bool(record.get("turn_id"))
                        and source_probe.get("opened") is True
                        and source_duration is not None
                        and end is not None
                        and float(end) > float(source_duration) + tolerance
                    )
                    stats["invalid_source_exclusion_records"] += int(
                        not (official_valid or duration_valid)
                    )

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
            duration = max(
                0.0, float(turn.get("end", 0.0)) - float(turn.get("start", 0.0))
            )
            durations += duration
            confidence = turn.get("lip_conf_v")
            if confidence is None:
                stats["missing_lip_conf_turns"] += 1
            else:
                values = np.asarray(confidence, dtype=np.float32).reshape(-1)
                stats["lip_conf_turns"] += 1
                stats["all_one_lip_conf_turns"] += int(
                    values.size > 0 and bool(np.allclose(values, 1.0))
                )
                stats["all_zero_lip_conf_turns"] += int(
                    values.size > 0 and bool(np.allclose(values, 0.0))
                )
                roi_path = _artifact_path(turn.get("mouth_roi"))
                if roi_path is None:
                    stats["missing_roi_paths"] += 1
                elif not roi_path.is_file():
                    stats["missing_roi_files"] += 1
                else:
                    try:
                        frames = int(np.load(roi_path, mmap_mode="r").shape[0])
                        stats["lip_conf_length_mismatch"] += int(frames != values.size)
                        expected_frames = max(1, int(round(duration * 25.0)))
                        coverage = frames / expected_frames
                        frame_coverages.append(coverage)
                        stats["short_roi_turns"] += int(coverage < min_frame_coverage)
                        stats["long_roi_turns"] += int(coverage > max_frame_coverage)
                    except Exception:
                        stats["unreadable_roi"] += 1

    result: dict[str, Any] = {
        **dict(stats),
        "hours": durations / 3600.0,
        "failure_rate": (
            stats["processing_failures"] / stats["processing_attempts"]
            if stats["processing_attempts"]
            else 0.0
        ),
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "minimum_roi_frame_coverage": min(frame_coverages) if frame_coverages else None,
        "maximum_roi_frame_coverage": max(frame_coverages) if frame_coverages else None,
        "missing_visual_manifest_ids": [Path(name).stem for name in missing_visual_manifests],
        "unexpected_visual_manifest_ids": [
            Path(name).stem for name in unexpected_visual_manifests
        ],
    }
    return result, meetings, participants, build_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/ami_full_v2"))
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--dev", type=Path, default=None)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--min-frame-coverage", type=float, default=0.90)
    parser.add_argument("--max-frame-coverage", type=float, default=1.10)
    parser.add_argument(
        "--allow-processing-failures",
        action="store_true",
        help="Diagnostic only: report failed turns without failing solely because they exist.",
    )
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    split_paths = {
        split: (getattr(args, split) or (args.run_dir / split / "visual")).resolve()
        for split in ("train", "dev", "test")
    }

    results: dict[str, dict[str, Any]] = {}
    meeting_sets: dict[str, set[str]] = {}
    participant_sets: dict[str, set[str]] = {}
    build_id_sets: dict[str, set[str]] = {}
    failed = False
    for split in ("train", "dev", "test"):
        base_path = args.run_dir / split / "base" / "manifests"
        stats, meetings, participants, build_ids = audit_split(
            split_paths[split],
            base_path=base_path if base_path.is_dir() else None,
            min_frame_coverage=args.min_frame_coverage,
            max_frame_coverage=args.max_frame_coverage,
        )
        stats["dataset_build_ids"] = sorted(build_ids)
        results[split] = stats
        meeting_sets[split] = meetings
        participant_sets[split] = participants
        build_id_sets[split] = build_ids
        forbidden = [
            "missing_visual_manifests",
            "unexpected_visual_manifests",
            "manifests_with_turn_cap",
            "unverified_camera_mappings",
            "missing_global_speaker_entries",
            "missing_lip_conf_turns",
            "lip_conf_length_mismatch",
            "unreadable_roi",
            "missing_roi_paths",
            "missing_roi_files",
            "non_quality_enrollment_entries",
            "attempt_accounting_mismatch",
            "eligible_accounting_mismatch",
            "source_exclusion_subtotal_mismatch",
            "success_accounting_mismatch",
            "manifests_with_eligible_turn_mismatch",
            "eligible_turn_gap_absolute",
            "skipped_unmapped_turns",
            "missing_failure_logs",
            "missing_failure_log_files",
            "failure_log_count_mismatch",
            "unreadable_failure_log_records",
            "failure_records_missing_turn_id",
            "missing_source_exclusion_logs",
            "missing_source_exclusion_log_files",
            "source_exclusion_log_count_mismatch",
            "unreadable_source_exclusion_log_records",
            "invalid_source_exclusion_records",
            "short_roi_turns",
            "long_roi_turns",
        ]
        if not args.allow_processing_failures:
            forbidden.append("processing_failures")
        for name in forbidden:
            failed |= stats.get(name, 0) > 0
        failed |= stats.get("manifests", 0) == 0
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
            and stats.get("all_one_lip_conf_turns", 0)
            == stats.get("lip_conf_turns", 0)
        )

    overlap = {
        "train_dev_meetings": sorted(meeting_sets["train"] & meeting_sets["dev"]),
        "train_test_meetings": sorted(meeting_sets["train"] & meeting_sets["test"]),
        "dev_test_meetings": sorted(meeting_sets["dev"] & meeting_sets["test"]),
        "train_dev_participants": sorted(
            participant_sets["train"] & participant_sets["dev"]
        ),
        "train_test_participants": sorted(
            participant_sets["train"] & participant_sets["test"]
        ),
        "dev_test_participants": sorted(
            participant_sets["dev"] & participant_sets["test"]
        ),
    }
    failed |= any(overlap[key] for key in overlap if key.endswith("_meetings"))
    print(json.dumps({"splits": results, "overlap": overlap, "passed": not failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
