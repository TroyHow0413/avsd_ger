"""Summarize structured AMI visual preprocessing failures without changing data."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_report(run_dir: Path, *, top: int = 20) -> dict[str, Any]:
    split_stats: dict[str, Counter[str]] = {
        split: Counter() for split in ("train", "dev", "test")
    }
    reasons: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    cameras: Counter[str] = Counter()
    reason_camera: Counter[str] = Counter()
    meetings: list[dict[str, Any]] = []
    missing_ledgers: list[str] = []
    annotation_overruns: list[dict[str, Any]] = []
    unreadable_clips = Counter()

    for split in ("train", "dev", "test"):
        for manifest_path in sorted((run_dir / split / "visual").glob("*.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            meta = manifest.get("meta", {})
            attempts = int(meta.get("attempts", len(manifest.get("turns", []))))
            failures = int(meta.get("failures", 0))
            official_exclusions = int(meta.get("official_visual_exclusions", 0))
            source_exclusions = int(
                meta.get("visual_source_exclusions", official_exclusions)
            )
            duration_exclusions = int(meta.get("source_duration_exclusions", 0))
            successes = len(manifest.get("turns", []))
            split_stats[split]["manifests"] += 1
            split_stats[split]["attempts"] += attempts
            split_stats[split]["successes"] += successes
            split_stats[split]["failures"] += failures
            split_stats[split]["visual_source_exclusions"] += source_exclusions
            split_stats[split]["official_visual_exclusions"] += official_exclusions
            split_stats[split]["source_duration_exclusions"] += duration_exclusions
            meetings.append(
                {
                    "split": split,
                    "meeting_id": manifest_path.stem,
                    "attempts": attempts,
                    "successes": successes,
                    "failures": failures,
                    "failure_rate": failures / attempts if attempts else 0.0,
                }
            )
            if failures == 0:
                continue
            failure_log = _resolve(meta.get("failure_log"))
            if failure_log is None or not failure_log.is_file():
                missing_ledgers.append(f"{split}/{manifest_path.stem}")
                reasons["unclassified_missing_ledger"] += failures
                continue
            for record in _read_jsonl(failure_log):
                reason = str(record.get("reason", "missing_reason"))
                stage = str(record.get("stage", "missing_stage"))
                camera = str(record.get("closeup") or "missing_camera")
                reasons[reason] += 1
                stages[stage] += 1
                cameras[camera] += 1
                reason_camera[f"{reason}|{camera}"] += 1
                clip_probe = record.get("clip_probe") or {}
                if reason == "clip_unreadable":
                    unreadable_clips["total"] += 1
                    unreadable_clips["missing_clip"] += int(not clip_probe.get("exists"))
                    unreadable_clips["zero_byte_clip"] += int(
                        clip_probe.get("size_bytes") == 0
                    )
                    unreadable_clips["decoder_not_opened"] += int(
                        clip_probe.get("opened") is False
                    )
                source_probe = record.get("source_probe") or {}
                source_duration = source_probe.get("reported_duration_seconds")
                end = record.get("end")
                if source_duration is not None and end is not None:
                    overrun = float(end) - float(source_duration)
                    if overrun > 0.25:
                        annotation_overruns.append(
                            {
                                "split": split,
                                "meeting_id": record.get("meeting_id"),
                                "turn_id": record.get("turn_id"),
                                "camera": record.get("closeup"),
                                "turn_end": float(end),
                                "source_duration": float(source_duration),
                                "overrun_seconds": overrun,
                            }
                        )

    process_failures: list[dict[str, Any]] = []
    result_root = run_dir / "logs" / "visual"
    for result_path in result_root.glob("*/*.result.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if int(result.get("returncode", 0)) != 0:
            process_failures.append(
                {
                    "split": result.get("split"),
                    "meeting_id": result.get("meeting_id"),
                    "returncode": result.get("returncode"),
                    "reason": result.get("process_failure_reason"),
                    "log": result.get("log"),
                }
            )

    split_report: dict[str, dict[str, Any]] = {}
    for split, stats in split_stats.items():
        split_report[split] = {
            **dict(stats),
            "failure_rate": (
                stats["failures"] / stats["attempts"] if stats["attempts"] else 0.0
            ),
        }
    return {
        "run_dir": str(run_dir),
        "splits": split_report,
        "failure_reasons": dict(reasons.most_common()),
        "failure_stages": dict(stages.most_common()),
        "failure_cameras": dict(cameras.most_common()),
        "failure_reason_camera": dict(reason_camera.most_common()),
        "worst_meetings": sorted(
            meetings, key=lambda row: row["failure_rate"], reverse=True
        )[:top],
        "missing_failure_ledgers": missing_ledgers,
        "unreadable_clip_diagnostics": dict(unreadable_clips),
        "annotation_overrun_count": len(annotation_overruns),
        "annotation_overrun_examples": sorted(
            annotation_overruns,
            key=lambda row: row["overrun_seconds"],
            reverse=True,
        )[:top],
        "process_failures": process_failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/ami_full_v3"))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = build_report(args.run_dir.resolve(), top=args.top)
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(output, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
