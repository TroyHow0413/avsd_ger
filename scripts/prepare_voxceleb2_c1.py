"""Prepare face-tracked VoxCeleb2 videos for AVSD-GER C1 identity training.

Point --dev-root at the directory whose immediate children are VoxCeleb speaker
IDs (normally ``dev/mp4``). Official VoxCeleb2 ``dev`` is training data; this
script deterministically withholds speakers for a local validation manifest.
The official test root, when supplied, is emitted separately and never mixed
into training.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from av_preprocess_common import (  # noqa: E402
    AtomicJsonlWriter,
    VIDEO_SUFFIXES,
    atomic_json,
    ensure_build_config,
    extract_face,
    extract_wav,
    manifest_path,
    valid_image,
    valid_wav,
    wav_duration,
)


@dataclass(frozen=True)
class VoxItem:
    index: int
    split: str
    speaker_id: str
    relative: str
    video: str


_WORKER: dict[str, Any] = {}


def discover(root: Path, split: str) -> list[VoxItem]:
    items: list[VoxItem] = []
    for video in sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES):
        relative = video.relative_to(root)
        if len(relative.parts) < 2:
            continue
        items.append(
            VoxItem(
                index=len(items),
                split=split,
                speaker_id=relative.parts[0],
                relative=relative.as_posix(),
                video=str(video),
            )
        )
    return items


def validation_speakers(speakers: Iterable[str], percentage: float, seed: int) -> set[str]:
    ordered = sorted(
        set(speakers),
        key=lambda speaker: hashlib.sha256(f"{seed}:{speaker}".encode()).digest(),
    )
    if not ordered or percentage <= 0:
        return set()
    count = max(2, round(len(ordered) * percentage / 100.0))
    count = min(count, max(0, len(ordered) - 2))
    return set(ordered[:count])


def curate(
    items: list[VoxItem],
    *,
    max_per_speaker: int | None,
    max_per_video: int | None,
    min_per_speaker: int,
) -> list[VoxItem]:
    grouped: dict[str, list[VoxItem]] = {}
    for item in items:
        grouped.setdefault(item.speaker_id, []).append(item)
    selected: list[VoxItem] = []
    for speaker, speaker_items in sorted(grouped.items()):
        by_source: dict[str, list[VoxItem]] = {}
        for item in speaker_items:
            relative = Path(item.relative)
            source = relative.parts[1] if len(relative.parts) > 2 else "_root"
            by_source.setdefault(source, []).append(item)
        buckets: list[list[VoxItem]] = []
        for source in sorted(by_source):
            candidates = sorted(by_source[source], key=lambda value: value.relative)
            buckets.append(candidates[:max_per_video] if max_per_video else candidates)
        # Round-robin source videos so a per-speaker cap favours diversity
        # instead of taking every utterance from the earliest YouTube IDs.
        kept = []
        for offset in range(max((len(bucket) for bucket in buckets), default=0)):
            kept.extend(bucket[offset] for bucket in buckets if offset < len(bucket))
        if max_per_speaker:
            kept = kept[:max_per_speaker]
        if len(kept) >= min_per_speaker:
            selected.extend(kept)
    return selected


def _init_worker(config: dict[str, Any]) -> None:
    global _WORKER
    _WORKER = config
    import cv2

    cv2.setNumThreads(config["opencv_threads"])


def _output_paths(item: VoxItem) -> tuple[Path, Path]:
    output = Path(_WORKER["output_root"])
    relative = Path(item.split) / Path(item.relative)
    return (
        output / "audio" / relative.with_suffix(".wav"),
        output / "faces" / relative.with_suffix(".jpg"),
    )


def _process(item: VoxItem) -> dict[str, Any]:
    video = Path(item.video)
    wav, face = _output_paths(item)
    try:
        if not video.is_file():
            raise FileNotFoundError(f"missing video: {video}")
        if _WORKER["dry_run"]:
            return {"ok": True, "index": item.index, "split": item.split, "dry_run": True}
        if _WORKER["overwrite"] or not valid_wav(wav):
            extract_wav(
                video,
                wav,
                ffmpeg=_WORKER["ffmpeg"],
                threads=_WORKER["ffmpeg_threads"],
            )
        duration = wav_duration(wav)
        if duration < _WORKER["min_duration"] or duration > _WORKER["max_duration"]:
            raise ValueError(
                f"duration {duration:.3f}s outside "
                f"[{_WORKER['min_duration']}, {_WORKER['max_duration']}]"
            )
        if _WORKER["overwrite"] or not valid_image(face):
            extract_face(
                video,
                face,
                ffmpeg=_WORKER["ffmpeg"],
                size=_WORKER["face_size"],
                threads=_WORKER["ffmpeg_threads"],
            )

        root = Path(_WORKER["manifest_root"])
        absolute = bool(_WORKER["absolute_paths"])
        clip_key = Path(item.relative).with_suffix("").as_posix()
        record = {
            "utt_id": f"voxceleb2/{item.split}/{clip_key}",
            "dataset": "voxceleb2",
            "dataset_build_id": _WORKER["build_id"],
            "split": item.split,
            "wav_path": manifest_path(wav, root, absolute=absolute),
            "face_path": manifest_path(face, root, absolute=absolute),
            "speaker_id": f"vox2:{item.speaker_id}",
            "participant_id": f"vox2:{item.speaker_id}",
            "duration": duration,
            # VoxCeleb2 MP4s are released as face tracks. A scalar quality
            # value deliberately expands over the audio clock in DualGate;
            # it denotes source-track availability, not landmark confidence.
            "lip_conf": [1.0],
            "lip_conf_source": "voxceleb2_face_track_available",
            "source_video": item.relative,
        }
        return {"ok": True, "index": item.index, "split": item.split, "record": record}
    except Exception as exc:
        return {
            "ok": False,
            "index": item.index,
            "split": item.split,
            "speaker_id": item.speaker_id,
            "video": str(video),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _ordered_map(items: list[VoxItem], workers: int) -> Iterable[dict[str, Any]]:
    if workers == 1:
        for item in items:
            yield _process(item)
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(_WORKER,)) as pool:
        yield from pool.map(_process, items, chunksize=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-root", type=Path, required=True, help="Official dev/mp4 speaker directory (training source).")
    parser.add_argument("--test-root", type=Path, default=None, help="Optional official test/mp4 speaker directory.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, default=ROOT)
    parser.add_argument("--build-id", default="voxceleb2_c1_v1")
    parser.add_argument("--val-speaker-percent", type=float, default=2.0)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--min-per-speaker", type=int, default=2)
    parser.add_argument("--max-per-speaker", type=int, default=None)
    parser.add_argument("--max-per-video", type=int, default=None)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--face-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=None, help="Global smoke-test cap after curation.")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    parser.add_argument("--opencv-threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--absolute-paths", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if (
        args.log_every < 1
        or args.face_size < 1
        or args.ffmpeg_threads < 1
        or args.opencv_threads < 1
    ):
        raise ValueError(
            "--log-every, --face-size, --ffmpeg-threads and --opencv-threads must be >= 1"
        )
    if not 0 <= args.val_speaker_percent < 100:
        raise ValueError("--val-speaker-percent must be in [0, 100)")
    if args.min_duration <= 0 or args.max_duration < args.min_duration:
        raise ValueError("require 0 < --min-duration <= --max-duration")
    for name, value in (
        ("--min-per-speaker", args.min_per_speaker),
        ("--max-per-speaker", args.max_per_speaker),
        ("--max-per-video", args.max_per_video),
        ("--max-items", args.max_items),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be >= 1")
    args.dev_root = args.dev_root.resolve()
    args.output_root = args.output_root.resolve()
    args.manifest_root = args.manifest_root.resolve()
    if not args.dev_root.is_dir():
        raise FileNotFoundError(f"--dev-root not found: {args.dev_root}")

    dev_items = curate(
        discover(args.dev_root, "official-dev"),
        max_per_speaker=args.max_per_speaker,
        max_per_video=args.max_per_video,
        min_per_speaker=args.min_per_speaker,
    )
    if not dev_items:
        raise RuntimeError(
            f"no face-tracked videos found below {args.dev_root}; "
            "VoxCeleb2 AAC-only archives cannot produce C1 face pairs"
        )
    held_out = validation_speakers(
        (item.speaker_id for item in dev_items),
        args.val_speaker_percent,
        args.split_seed,
    )
    items: list[VoxItem] = []
    for item in dev_items:
        split = "dev" if item.speaker_id in held_out else "train"
        items.append(VoxItem(len(items), split, item.speaker_id, item.relative, item.video))

    if args.test_root:
        test_root = args.test_root.resolve()
        if not test_root.is_dir():
            raise FileNotFoundError(f"--test-root not found: {test_root}")
        test_items = curate(
            discover(test_root, "test"),
            max_per_speaker=args.max_per_speaker,
            max_per_video=args.max_per_video,
            min_per_speaker=args.min_per_speaker,
        )
        for item in test_items:
            items.append(VoxItem(len(items), "test", item.speaker_id, item.relative, item.video))
    if args.max_items is not None:
        items = items[: args.max_items]

    global _WORKER
    _WORKER = {
        "output_root": str(args.output_root),
        "manifest_root": str(args.manifest_root),
        "build_id": args.build_id,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "face_size": args.face_size,
        "ffmpeg": args.ffmpeg,
        "ffmpeg_threads": args.ffmpeg_threads,
        "opencv_threads": args.opencv_threads,
        "overwrite": args.overwrite,
        "absolute_paths": args.absolute_paths,
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        ensure_build_config(
            args.output_root / "build_config.json",
            {
                "dataset": "voxceleb2",
                "build_id": args.build_id,
                "validation_speaker_percent": args.val_speaker_percent,
                "split_seed": args.split_seed,
                "min_per_speaker": args.min_per_speaker,
                "max_per_speaker": args.max_per_speaker,
                "max_per_video": args.max_per_video,
                "min_duration": args.min_duration,
                "max_duration": args.max_duration,
                "face_size": args.face_size,
                "absolute_paths": args.absolute_paths,
            },
            overwrite=args.overwrite,
        )
    _init_worker(_WORKER)
    counts = {split: {"succeeded": 0, "failed": 0} for split in ("train", "dev", "test")}
    with ExitStack() as stack:
        writers = {
            split: stack.enter_context(
                AtomicJsonlWriter(args.output_root / "manifests" / f"{split}.jsonl")
            )
            for split in ("train", "dev", "test")
            if not args.dry_run and any(item.split == split for item in items)
        }
        failure_writer = (
            stack.enter_context(AtomicJsonlWriter(args.output_root / "failures.jsonl"))
            if not args.dry_run
            else None
        )
        failed_so_far = 0
        for position, result in enumerate(_ordered_map(items, args.workers), 1):
            split = result["split"]
            if result.get("record"):
                counts[split]["succeeded"] += 1
                writers[split].write(result["record"])
            elif result["ok"]:  # dry-run success
                counts[split]["succeeded"] += 1
            else:
                counts[split]["failed"] += 1
                failed_so_far += 1
                if failure_writer:
                    failure_writer.write(result)
            if position % args.log_every == 0 or position == len(items):
                print(f"[voxceleb2] progress={position}/{len(items)} failed={failed_so_far}", flush=True)
    summary: dict[str, Any] = {
        "dataset": "voxceleb2",
        "build_id": args.build_id,
        "validation_split": "speaker-disjoint deterministic subset of official dev",
        "validation_speakers": len(held_out),
        "splits": {},
    }
    for split in ("train", "dev", "test"):
        requested = sum(item.split == split for item in items)
        summary["splits"][split] = {
            "requested": requested,
            "succeeded": counts[split]["succeeded"],
            "failed": counts[split]["failed"],
            "speakers": len({item.speaker_id for item in items if item.split == split}),
        }
        print(
            f"[{split}] requested={requested} "
            f"succeeded={counts[split]['succeeded']} failed={counts[split]['failed']}"
        )

    if not args.dry_run:
        atomic_json(args.output_root / "audit.json", summary)
    return 1 if failed_so_far else 0


if __name__ == "__main__":
    raise SystemExit(main())
