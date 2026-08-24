"""Build complete AMI visual manifests for train/dev/test splits.

This is the production batch entry point. By default it applies no turn cap,
uses official meeting-specific camera mappings, and writes only inside a new
version directory. Existing completed outputs are immutable.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _command(args: argparse.Namespace, split: str, manifest: Path) -> list[str]:
    meeting_id = manifest.stem
    output_base = args.run_dir / split / "visual"
    command = [
        sys.executable,
        str(args.root / "scripts" / "prepare_ami_visual_manifest.py"),
        "--manifest",
        _cli_path(manifest, args.root),
        "--ami-video-dir",
        _cli_path(args.root / "datasets" / "ami" / "video", args.root),
        "--meetings-xml",
        _cli_path(args.meetings_xml, args.root),
        "--out-manifest",
        _cli_path(output_base / f"{meeting_id}.json", args.root),
        "--out-dir",
        _cli_path(output_base / meeting_id, args.root),
        "--min-turn-secs",
        str(args.min_turn_secs),
        "--max-turn-secs",
        str(args.max_turn_secs),
        "--roi-backend",
        args.roi_backend,
    ]
    if args.max_turns is not None:
        command.extend(["--max-turns", str(args.max_turns)])
    if args.max_turns_per_speaker is not None:
        command.extend(["--max-turns-per-speaker", str(args.max_turns_per_speaker)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Dataset-version directory created by rebuild_ami_base_manifests.py.",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev", "test"]
    )
    parser.add_argument(
        "--meetings-xml",
        type=Path,
        default=ROOT / "datasets" / "ami" / "corpusResources" / "meetings.xml",
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--min-turn-secs", type=float, default=1.0)
    parser.add_argument("--max-turn-secs", type=float, default=12.0)
    parser.add_argument("--max-turns", type=int, default=None, help="Smoke-test cap only.")
    parser.add_argument("--max-turns-per-speaker", type=int, default=None, help="Smoke-test cap only.")
    parser.add_argument("--roi-backend", choices=["dlib", "haar"], default="dlib")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted build by creating only missing visual manifests.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.run_dir = args.run_dir.resolve()
    args.meetings_xml = args.meetings_xml.resolve()

    if not args.meetings_xml.is_file():
        print(
            f"[error] missing {args.meetings_xml}; run scripts/fetch_ami_metadata.py",
            file=sys.stderr,
        )
        return 2
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    try:
        _write_or_verify_plan(
            args.run_dir / "visual_build_plan.json",
            {
                "dataset_build_id": args.run_dir.name,
                "pipeline": "repaired_ami_visual_v2",
                "roi_backend": args.roi_backend,
                "min_turn_seconds": args.min_turn_secs,
                "max_turn_seconds": args.max_turn_secs,
                "max_turns": args.max_turns,
                "max_turns_per_speaker": args.max_turns_per_speaker,
                "camera_mapping": "AMI corpusResources/meetings.xml",
                "lip_confidence": "direct_detection_1/interpolation_0.5/extrapolation_0.25",
            },
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    jobs: list[tuple[str, Path, Path]] = []
    for split in args.splits:
        source = args.run_dir / split / "base" / "manifests"
        manifests = sorted(source.glob("*.json"))
        if not manifests:
            print(f"[error] no manifests under {source}", file=sys.stderr)
            return 2
        output_base = args.run_dir / split / "visual"
        existing = {path.stem for path in output_base.glob("*.json")}
        has_any_output = output_base.exists() and any(
            path.is_file() for path in output_base.rglob("*")
        )
        if has_any_output and not args.resume and not args.dry_run:
            print(
                f"[error] {output_base} already contains build artifacts "
                f"({len(existing)} completed manifests). "
                "Use a new --run-dir for a new comparison version, or --resume "
                "only for an interrupted build.",
                file=sys.stderr,
            )
            return 2
        if not args.dry_run:
            output_base.mkdir(parents=True, exist_ok=True)
        for manifest in manifests:
            output = output_base / manifest.name
            if output.exists():
                print(f"[complete] {split}/{manifest.stem}")
                continue
            jobs.append((split, manifest, output))

    print(
        f"[build] version={args.run_dir.name} jobs={len(jobs)} workers={args.jobs} "
        f"turn_cap={args.max_turns} per_speaker_cap={args.max_turns_per_speaker}"
    )
    if args.dry_run:
        counts = {split: sum(item[0] == split for item in jobs) for split in args.splits}
        print(f"[dry-run] split_jobs={counts}")
        return 0

    def run_one(item: tuple[str, Path, Path]):
        split, manifest, _ = item
        completed = subprocess.run(
            _command(args, split, manifest),
            cwd=args.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return split, manifest.stem, completed.returncode, completed.stdout

    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run_one, item) for item in jobs]
        for future in as_completed(futures):
            split, meeting_id, returncode, output = future.result()
            tail = "\n".join(output.strip().splitlines()[-5:])
            print(f"[{('ok' if returncode == 0 else 'fail')}] {split}/{meeting_id}\n{tail}")
            failures += int(returncode != 0)

    print(f"[done] built={len(jobs) - failures} failed={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
