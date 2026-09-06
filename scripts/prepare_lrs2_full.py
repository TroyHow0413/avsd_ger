"""Prepare the complete official LRS2 splits for AVSD-GER Stage-2.

The source MP4 files are already utterance-level. This script does not segment
long videos; it materialises the inputs required by the current trainers:
16 kHz mono WAV, AV-HuBERT-style mouth ROI, a representative face crop, and
one deterministic JSONL manifest per official split.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from av_preprocess_common import (  # noqa: E402
    AtomicJsonlWriter,
    atomic_json,
    ensure_build_config,
    extract_face,
    extract_wav,
    manifest_path,
    save_npy_atomic,
    valid_image,
    valid_npy,
    valid_wav,
    wav_duration,
)


DEFAULT_MODELS = {
    "face_predictor": ROOT / "checkpoints/shape_predictor_68_face_landmarks.dat",
    "cnn_detector": ROOT / "checkpoints/mmod_human_face_detector.dat",
    "mean_face": ROOT / "av_hubert/avhubert/preparation/data/20words_mean_face.npy",
}
SPLIT_SOURCE_DIR = {"pretrain": "pretrain", "train": "main", "val": "main", "test": "main"}


@dataclass(frozen=True)
class LRS2Item:
    index: int
    split: str
    clip_id: str
    tags: tuple[str, ...]
    video: str
    transcript: str


_WORKER: dict[str, Any] = {}
_EXTRACTOR: Any = None


def parse_filelist(path: Path, split: str, source_dir: Path) -> list[LRS2Item]:
    items: list[LRS2Item] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw_line.strip().split()
        if not parts:
            continue
        clip_id = parts[0].replace("\\", "/")
        video = source_dir / f"{clip_id}.mp4"
        transcript = source_dir / f"{clip_id}.txt"
        items.append(
            LRS2Item(
                index=len(items),
                split=split,
                clip_id=clip_id,
                tags=tuple(parts[1:]),
                video=str(video),
                transcript=str(transcript),
            )
        )
    return items


def read_transcript(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Text:"):
            return line.partition(":")[2].strip()
    raise ValueError(f"missing 'Text:' line in {path}")


def _init_worker(config: dict[str, Any]) -> None:
    global _WORKER, _EXTRACTOR
    _WORKER = config
    if config["dry_run"]:
        return
    from avsd_ger.frontend.mouth_roi import MouthROIExtractor

    if config["roi_backend"] == "dlib":
        _EXTRACTOR = MouthROIExtractor(
            backend="dlib",
            face_predictor_path=config["face_predictor"],
            cnn_detector_path=config["cnn_detector"],
            mean_face_path=config["mean_face"],
        )
    else:
        _EXTRACTOR = MouthROIExtractor(backend="haar")


def _output_paths(item: LRS2Item) -> tuple[Path, Path, Path, Path]:
    output = Path(_WORKER["output_root"])
    relative = Path(item.split) / Path(item.clip_id)
    return (
        output / "audio" / relative.with_suffix(".wav"),
        output / "mouth_roi" / relative.with_suffix(".npy"),
        output / "lip_conf" / relative.with_suffix(".npy"),
        output / "faces" / relative.with_suffix(".jpg"),
    )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _process(item: LRS2Item) -> dict[str, Any]:
    video = Path(item.video)
    transcript_path = Path(item.transcript)
    wav, roi, confidence_path, face = _output_paths(item)
    try:
        if not video.is_file():
            raise FileNotFoundError(f"missing video: {video}")
        if not transcript_path.is_file():
            raise FileNotFoundError(f"missing transcript: {transcript_path}")
        text = read_transcript(transcript_path)
        if not text:
            raise ValueError("empty transcript")

        if _WORKER["dry_run"]:
            return {"ok": True, "index": item.index, "split": item.split, "dry_run": True}

        if _WORKER["overwrite"] or not valid_wav(wav):
            extract_wav(
                video,
                wav,
                ffmpeg=_WORKER["ffmpeg"],
                threads=_WORKER["ffmpeg_threads"],
            )
        if (
            _WORKER["overwrite"]
            or not valid_npy(
                roi,
                ndim=4,
                dtype=_WORKER["roi_dtype"],
                shape_tail=(1, 96, 96),
            )
            or not valid_npy(confidence_path, ndim=1)
        ):
            roi_value, confidence = _EXTRACTOR.extract_with_confidence_from_file(str(video))
            roi_array = _to_numpy(roi_value)
            if roi_array.ndim != 4 or roi_array.shape[1:] != (1, 96, 96):
                raise ValueError(f"unexpected ROI shape {roi_array.shape}")
            if _WORKER["roi_dtype"] == "uint8":
                roi_array = np.clip(np.rint(roi_array * 255.0), 0, 255).astype(np.uint8)
            else:
                roi_array = roi_array.astype(_WORKER["roi_dtype"])
            confidence = np.asarray(confidence, dtype=np.float32)
            if confidence.shape != (roi_array.shape[0],):
                raise ValueError("mouth ROI/confidence length mismatch")
            save_npy_atomic(roi, roi_array)
            save_npy_atomic(confidence_path, confidence)
        if _WORKER["overwrite"] or not valid_image(face):
            extract_face(
                video,
                face,
                ffmpeg=_WORKER["ffmpeg"],
                size=_WORKER["face_size"],
                threads=_WORKER["ffmpeg_threads"],
            )

        confidence = np.load(confidence_path, allow_pickle=False).astype(np.float32)
        duration = wav_duration(wav)
        speaker_local = Path(item.clip_id).parts[0]
        root = Path(_WORKER["manifest_root"])
        absolute = bool(_WORKER["absolute_paths"])
        record = {
            "utt_id": f"lrs2/{item.split}/{item.clip_id}",
            "dataset": "lrs2",
            "dataset_build_id": _WORKER["build_id"],
            "split": item.split,
            "wav_path": manifest_path(wav, root, absolute=absolute),
            "audio": manifest_path(wav, root, absolute=absolute),
            "video_path": manifest_path(roi, root, absolute=absolute),
            "mouth_roi": manifest_path(roi, root, absolute=absolute),
            "face_path": manifest_path(face, root, absolute=absolute),
            "enrollment_face": manifest_path(face, root, absolute=absolute),
            "target": text,
            "ref_text": text,
            "speaker_id": f"lrs2:{speaker_local}",
            "start": 0.0,
            "end": duration,
            "duration": duration,
            "lip_conf": np.round(confidence, 3).tolist(),
            "lip_conf_source": f"{_WORKER['roi_backend']}_detection_interpolation",
            "source_video": f"{SPLIT_SOURCE_DIR[item.split]}/{item.clip_id}.mp4",
            "source_transcript": f"{SPLIT_SOURCE_DIR[item.split]}/{item.clip_id}.txt",
            "split_tags": list(item.tags),
        }
        return {"ok": True, "index": item.index, "split": item.split, "record": record}
    except Exception as exc:  # keep a complete failure ledger for long server jobs
        return {
            "ok": False,
            "index": item.index,
            "split": item.split,
            "clip_id": item.clip_id,
            "video": str(video),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _ordered_map(items: list[LRS2Item], workers: int) -> Iterable[dict[str, Any]]:
    if workers == 1:
        for item in items:
            yield _process(item)
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(_WORKER,)) as pool:
        yield from pool.map(_process, items, chunksize=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lrs2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, default=ROOT)
    parser.add_argument("--splits", nargs="+", choices=tuple(SPLIT_SOURCE_DIR), default=list(SPLIT_SOURCE_DIR))
    parser.add_argument("--build-id", default="lrs2_full_v1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=None, help="Per-split smoke-test cap.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--roi-backend", choices=("dlib", "haar"), default="dlib")
    parser.add_argument("--roi-dtype", choices=("uint8", "float16", "float32"), default="uint8")
    parser.add_argument("--face-size", type=int, default=224)
    parser.add_argument("--face-predictor", type=Path, default=DEFAULT_MODELS["face_predictor"])
    parser.add_argument("--cnn-detector", type=Path, default=DEFAULT_MODELS["cnn_detector"])
    parser.add_argument("--mean-face", type=Path, default=DEFAULT_MODELS["mean_face"])
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffmpeg-threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--absolute-paths", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.log_every < 1 or args.face_size < 1 or args.ffmpeg_threads < 1:
        raise ValueError("--log-every, --face-size and --ffmpeg-threads must be >= 1")
    if args.max_items is not None and args.max_items < 0:
        raise ValueError("--max-items must be >= 0")
    args.lrs2_root = args.lrs2_root.resolve()
    args.output_root = args.output_root.resolve()
    args.manifest_root = args.manifest_root.resolve()

    global _WORKER
    _WORKER = {
        "output_root": str(args.output_root),
        "manifest_root": str(args.manifest_root),
        "build_id": args.build_id,
        "roi_backend": args.roi_backend,
        "roi_dtype": args.roi_dtype,
        "face_size": args.face_size,
        "face_predictor": str(args.face_predictor.resolve()),
        "cnn_detector": str(args.cnn_detector.resolve()),
        "mean_face": str(args.mean_face.resolve()),
        "ffmpeg": args.ffmpeg,
        "ffmpeg_threads": args.ffmpeg_threads,
        "overwrite": args.overwrite,
        "absolute_paths": args.absolute_paths,
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        ensure_build_config(
            args.output_root / "build_config.json",
            {
                "dataset": "lrs2",
                "build_id": args.build_id,
                "roi_backend": args.roi_backend,
                "roi_dtype": args.roi_dtype,
                "face_size": args.face_size,
                "face_predictor": args.face_predictor.name,
                "cnn_detector": args.cnn_detector.name,
                "mean_face": args.mean_face.name,
                "absolute_paths": args.absolute_paths,
            },
            overwrite=args.overwrite,
        )
    _init_worker(_WORKER)

    total_failures = 0
    summary: dict[str, Any] = {"dataset": "lrs2", "build_id": args.build_id, "splits": {}}
    failure_writer_context = (
        AtomicJsonlWriter(args.output_root / "failures.jsonl") if not args.dry_run else None
    )
    if failure_writer_context:
        failure_writer_context.__enter__()
    try:
        for split in args.splits:
            filelist = args.lrs2_root / f"{split}.txt"
            source_dir = args.lrs2_root / SPLIT_SOURCE_DIR[split]
            if not filelist.is_file() or not source_dir.is_dir():
                raise FileNotFoundError(f"missing LRS2 split input: {filelist} or {source_dir}")
            items = parse_filelist(filelist, split, source_dir)
            if args.max_items is not None:
                items = items[: args.max_items]
            record_writer_context = (
                AtomicJsonlWriter(args.output_root / "manifests" / f"{split}.jsonl")
                if not args.dry_run
                else None
            )
            if record_writer_context:
                record_writer_context.__enter__()
            succeeded = failed = 0
            try:
                for position, result in enumerate(_ordered_map(items, args.workers), 1):
                    if result.get("record"):
                        succeeded += 1
                        if record_writer_context:
                            record_writer_context.write(result["record"])
                    elif result["ok"]:  # dry-run success
                        succeeded += 1
                    else:
                        failed += 1
                        total_failures += 1
                        if failure_writer_context:
                            failure_writer_context.write(result)
                    if position % args.log_every == 0 or position == len(items):
                        print(f"[{split}] progress={position}/{len(items)} failed={failed}", flush=True)
            except BaseException:
                if record_writer_context:
                    record_writer_context.__exit__(*sys.exc_info())
                raise
            else:
                if record_writer_context:
                    record_writer_context.__exit__(None, None, None)
            summary["splits"][split] = {
                "requested": len(items),
                "succeeded": succeeded,
                "failed": failed,
            }
            print(f"[{split}] requested={len(items)} succeeded={succeeded} failed={failed}")
    except BaseException:
        if failure_writer_context:
            failure_writer_context.__exit__(*sys.exc_info())
        raise
    else:
        if failure_writer_context:
            failure_writer_context.__exit__(None, None, None)

    if not args.dry_run:
        atomic_json(args.output_root / "audit.json", summary)
    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
