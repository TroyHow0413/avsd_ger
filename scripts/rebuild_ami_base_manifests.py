"""Rebuild disjoint AMI base manifests with global IDs and quality enrollment."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "data" / "ami_full_v2"


def _cli_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_or_verify_plan(path: Path, payload: dict, *, dry_run: bool) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"Immutable build plan differs at {path}; choose a new --run-dir"
            )
        return
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="New immutable dataset-version directory (default: data/ami_full_v2).",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev", "test"]
    )
    parser.add_argument("--overwrite-turn-audio", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted build by creating only missing meeting manifests.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.run_dir = args.run_dir.resolve()

    legacy_dirs = {
        (args.root / "data" / f"ami_{split}").resolve()
        for split in ("train", "dev", "test")
    }
    if args.run_dir in legacy_dirs:
        parser.error("--run-dir must be a new version directory, not a legacy AMI split directory")

    ami_root = args.root / "datasets" / "ami"
    meetings_xml = ami_root / "corpusResources" / "meetings.xml"
    if not meetings_xml.is_file():
        print(f"[error] missing {meetings_xml}; run scripts/fetch_ami_metadata.py", file=sys.stderr)
        return 2

    source_ids = {
        split: {
            path.stem
            for path in (args.root / "data" / f"ami_{split}" / "manifests").glob("*.json")
        }
        for split in ("train", "dev", "test")
    }
    exclusions = {
        "train": set(),
        "dev": source_ids["train"],
        "test": source_ids["train"] | source_ids["dev"],
    }
    split_meetings = {
        split: sorted(source_ids[split] - exclusions[split])
        for split in ("train", "dev", "test")
    }
    try:
        _write_or_verify_plan(
            args.run_dir / "base_build_plan.json",
            {
                "dataset_build_id": args.run_dir.name,
                "pipeline": "repaired_ami_base_v2",
                "source_membership": "legacy manifests, deduplicated train > dev > test",
                "audio_condition": "AMI IHM individual headset microphone",
                "turn_boundary_source": "oracle reference transcript",
                "enrollment_mode": "turn_quality",
                "splits": split_meetings,
            },
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    failed = False
    for split in args.splits:
        meetings = split_meetings[split]
        if not meetings:
            print(f"[error] no disjoint {split} meetings found", file=sys.stderr)
            failed = True
            continue
        output = args.run_dir / split / "base"
        existing = {
            path.stem for path in (output / "manifests").glob("*.json")
        }
        has_any_output = output.exists() and any(path.is_file() for path in output.rglob("*"))
        if has_any_output and not args.resume and not args.dry_run:
            print(
                f"[error] {output} already contains build artifacts "
                f"({len(existing)} completed manifests). "
                "Use a new --run-dir for a new comparison version, or --resume "
                "only for an interrupted build.",
                file=sys.stderr,
            )
            failed = True
            continue
        pending = [meeting for meeting in meetings if meeting not in existing]
        if not pending:
            print(f"[complete] split={split} already has {len(existing)} manifests")
            continue
        command = [
            sys.executable,
            str(args.root / "scripts" / "prepare_ami_manifest.py"),
            "--ami",
            _cli_path(ami_root, args.root),
            "--out",
            _cli_path(output, args.root),
            "--meetings-xml",
            _cli_path(meetings_xml, args.root),
            "--enrollment-mode",
            "turn_quality",
            "--dataset-build-id",
            args.run_dir.name,
            "--meetings",
            *pending,
        ]
        if args.overwrite_turn_audio:
            command.append("--overwrite-turn-audio")
        print(
            f"[rebuild] version={args.run_dir.name} split={split} "
            f"pending={len(pending)} total={len(meetings)} output={output}"
        )
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=args.root)
        failed |= completed.returncode != 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
