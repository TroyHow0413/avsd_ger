"""Build complete AMI visual manifests for train/dev/test splits.

This is the production batch entry point. By default it applies no turn cap,
uses official meeting-specific camera mappings, and writes only inside a new
version directory. Existing completed outputs are immutable.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
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
        "-X",
        "faulthandler",
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


def _manifest_summary(path: Path) -> dict[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "readable": False,
            "error": f"{type(exc).__name__}: {exc}",
            "turns": None,
            "attempts": None,
            "failures": None,
        }
    meta = manifest.get("meta", {})
    return {
        "readable": True,
        "turns": len(manifest.get("turns", [])),
        "attempts": int(meta.get("attempts", len(manifest.get("turns", [])))),
        "failures": int(meta.get("failures", 0)),
        "failure_log": meta.get("failure_log"),
        "failure_reason_counts": meta.get("failure_reason_counts", {}),
    }


def _environment_snapshot(args: argparse.Namespace) -> dict[str, object]:
    packages: dict[str, object] = {}
    for name in ("dlib", "cv2", "numpy", "skimage"):
        try:
            module = __import__(name)
            entry: dict[str, object] = {
                "version": getattr(module, "__version__", None),
                "file": getattr(module, "__file__", None),
            }
            if name == "dlib":
                entry["cuda_enabled"] = bool(getattr(module, "DLIB_USE_CUDA", False))
            packages[name] = entry
        except Exception as exc:
            packages[name] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        ffmpeg = subprocess.run(
            ["ffmpeg", "-version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
        ffmpeg_version = (ffmpeg.stdout or "").splitlines()[0]
    except Exception as exc:
        ffmpeg_version = f"{type(exc).__name__}: {exc}"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "workers": args.jobs,
        "run_dir": str(args.run_dir),
        "splits": args.splits,
        "roi_backend": args.roi_backend,
        "ffmpeg": ffmpeg_version,
        "packages": packages,
    }


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
                "pipeline": "ami_avhubert_reference_v3",
                "roi_backend": args.roi_backend,
                "min_turn_seconds": args.min_turn_secs,
                "max_turn_seconds": args.max_turn_secs,
                "max_turns": args.max_turns,
                "max_turns_per_speaker": args.max_turns_per_speaker,
                "camera_mapping": "AMI corpusResources/meetings.xml",
                "lip_confidence": (
                    "direct_detection_1/interpolation_0.5/extrapolation_0.25/"
                    "all_landmarks_missing_resize_0"
                ),
                "all_landmarks_missing": "AV-HuBERT full-frame resize",
                "official_missing_media": "fixed AMI corpus exclusion ledger",
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
                summary = _manifest_summary(output)
                print(
                    f"[complete] {split}/{manifest.stem} turns={summary.get('turns')} "
                    f"attempts={summary.get('attempts')} failures={summary.get('failures')} "
                    f"failure_log={summary.get('failure_log')}",
                    flush=True,
                )
                continue
            jobs.append((split, manifest, output))

    print(
        f"[build] version={args.run_dir.name} jobs={len(jobs)} workers={args.jobs} "
        f"turn_cap={args.max_turns} per_speaker_cap={args.max_turns_per_speaker}"
        , flush=True
    )
    if args.dry_run:
        counts = {split: sum(item[0] == split for item in jobs) for split in args.splits}
        print(f"[dry-run] split_jobs={counts}")
        return 0

    invocation_path = args.run_dir / "logs" / "visual" / "last_invocation.json"
    invocation_path.parent.mkdir(parents=True, exist_ok=True)
    invocation_path.write_text(
        json.dumps(_environment_snapshot(args), indent=2) + "\n", encoding="utf-8"
    )

    def run_one(item: tuple[str, Path, Path]):
        split, manifest, output = item
        meeting_id = manifest.stem
        log_path = args.run_dir / "logs" / "visual" / split / f"{meeting_id}.log"
        result_path = args.run_dir / "logs" / "visual" / split / f"{meeting_id}.result.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = _command(args, split, manifest)
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log_stream:
            log_stream.write(f"[command] {' '.join(command)}\n")
            log_stream.flush()
            completed = subprocess.run(
                command,
                cwd=args.root,
                text=True,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
        summary = _manifest_summary(output) if output.exists() else {
            "readable": False,
            "turns": None,
            "attempts": None,
            "failures": None,
        }
        result = {
            "split": split,
            "meeting_id": meeting_id,
            "returncode": completed.returncode,
            "process_failure_reason": (
                f"signal_{-completed.returncode}"
                if completed.returncode < 0
                else (f"exit_{completed.returncode}" if completed.returncode else None)
            ),
            "elapsed_seconds": time.time() - started,
            "command": command,
            "log": str(log_path),
            "output_manifest": str(output),
            **summary,
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result

    process_failures = 0
    partial_manifests = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run_one, item) for item in jobs]
        for future in as_completed(futures):
            result = future.result()
            returncode = int(result["returncode"])
            internal_failures = int(result.get("failures") or 0)
            status = "fail" if returncode else ("partial" if internal_failures else "ok")
            print(
                f"[{status}] {result['split']}/{result['meeting_id']} "
                f"turns={result.get('turns')} attempts={result.get('attempts')} "
                f"failures={result.get('failures')} elapsed={result['elapsed_seconds']:.1f}s "
                f"log={result['log']}",
                flush=True,
            )
            process_failures += int(returncode != 0)
            partial_manifests += int(returncode == 0 and internal_failures > 0)

    print(
        f"[done] built={len(jobs) - process_failures} "
        f"process_failed={process_failures} partial_manifests={partial_manifests}",
        flush=True,
    )
    return 1 if process_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
