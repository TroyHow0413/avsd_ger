"""Attach mouth-ROI clips to an existing AMI session manifest.

This is a conservative AMI visual-prep smoke script. AMI Closeup videos are
meeting-length camera streams, not per-utterance face videos, and the corpus
does not make Closeup1/2/3/4 agent-specific in the filename. To avoid visual
speaker mismatch, this script requires an explicit speaker-to-closeup mapping
and only emits turns for mapped speakers.

Example:
    python scripts/prepare_ami_visual_manifest.py \
      --manifest data/ami_test/manifests/IS1009c.json \
      --ami-video-dir datasets/ami/video \
      --out-manifest data/ami_visual_smoke/IS1009c_closeup12.json \
      --out-dir data/ami_visual_smoke/IS1009c \
      --speaker-closeup A=Closeup1 B=Closeup2 \
      --max-turns 12 \
      --roi-backend dlib

Then smoke-test AV mode:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_ablations.py \
      --config one_go/runs/config_real_en.yaml \
      --safe-core-preset full \
      --manifest data/ami_visual_smoke/IS1009c_closeup12.json \
      --pool checkpoints/identity_pool.pt \
      --fresh-pool \
      --out out/safe_core_av/IS1009c_closeup12.json \
      --ger-mode av \
      --frontend-profile common_pyannote_lightasd \
      --only full_model wo_c2 wo_c3 \
      --no-power
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_FACE_PREDICTOR = "checkpoints/shape_predictor_68_face_landmarks.dat"
_DEFAULT_CNN_DETECTOR = "checkpoints/mmod_human_face_detector.dat"
_DEFAULT_MEAN_FACE = "av_hubert/avhubert/preparation/data/20words_mean_face.npy"


def _parse_speaker_closeup(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected SPEAKER=CloseupN mapping, got {item!r}")
        speaker, closeup = item.split("=", 1)
        speaker = speaker.strip()
        closeup = closeup.strip()
        if not speaker or not closeup:
            raise ValueError(f"Invalid speaker-closeup mapping: {item!r}")
        mapping[speaker] = closeup
    return mapping


def _speaker_suffix(ref_speaker: str) -> str:
    # AMI manifests use IDs like IS1009c_A.
    return ref_speaker.rsplit("_", 1)[-1]


def _extract_meeting_id(manifest: dict[str, Any], manifest_path: Path) -> str:
    for turn in manifest.get("turns", []):
        ref = str(turn.get("ref_speaker", ""))
        if "_" in ref:
            return ref.rsplit("_", 1)[0]
        tid = str(turn.get("turn_id", ""))
        if "." in tid:
            return tid.split(".", 1)[0]
    return manifest_path.stem


def _resolve_path(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return p
    if "\\" in path:
        p = Path(path.replace("\\", "/"))
        if p.exists():
            return p
    return None


def _build_extractor(args):
    sys.path.insert(0, str(ROOT))
    from avsd_ger.frontend.mouth_roi import MouthROIExtractor

    if args.roi_backend == "dlib":
        return MouthROIExtractor(
            backend="dlib",
            face_predictor_path=args.face_predictor,
            cnn_detector_path=args.cnn_detector,
            mean_face_path=args.mean_face,
        )
    return MouthROIExtractor(backend="haar")


def _ffmpeg_slice_video(src: Path, dst: Path, start: float, end: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(src),
            "-r",
            "25",
            "-an",
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ],
        check=True,
    )


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--ami-video-dir", default="datasets/ami/video", type=Path)
    p.add_argument("--out-manifest", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--speaker-closeup",
        nargs="+",
        required=True,
        help="Explicit AMI mapping, e.g. A=Closeup1 B=Closeup2.",
    )
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--min-turn-secs", type=float, default=1.0)
    p.add_argument("--max-turn-secs", type=float, default=12.0)
    p.add_argument("--keep-clips", action="store_true")
    p.add_argument("--roi-backend", default="dlib", choices=["dlib", "haar"])
    p.add_argument("--face-predictor", default=_DEFAULT_FACE_PREDICTOR)
    p.add_argument("--cnn-detector", default=_DEFAULT_CNN_DETECTOR)
    p.add_argument("--mean-face", default=_DEFAULT_MEAN_FACE)
    args = p.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    closeup_by_speaker = _parse_speaker_closeup(args.speaker_closeup)
    meeting_id = _extract_meeting_id(manifest, args.manifest)
    extractor = _build_extractor(args)

    out_roi_dir = args.out_dir / "mouth_roi"
    out_clip_dir = args.out_dir / "video_clips"
    out_roi_dir.mkdir(parents=True, exist_ok=True)
    out_clip_dir.mkdir(parents=True, exist_ok=True)

    turns_out: list[dict[str, Any]] = []
    attempts = 0
    failures = 0
    skipped_unmapped = 0
    skipped_duration = 0

    for turn in manifest.get("turns", []):
        if len(turns_out) >= args.max_turns:
            break

        ref_speaker = str(turn.get("ref_speaker", ""))
        suffix = _speaker_suffix(ref_speaker)
        closeup = closeup_by_speaker.get(suffix)
        if closeup is None:
            skipped_unmapped += 1
            continue

        start = float(turn["start"])
        end = float(turn["end"])
        dur = end - start
        if dur < args.min_turn_secs or dur > args.max_turn_secs:
            skipped_duration += 1
            continue

        src_video = args.ami_video_dir / f"{meeting_id}.{closeup}.avi"
        if not src_video.exists():
            print(f"[missing] {src_video}")
            failures += 1
            continue

        turn_id = str(turn.get("turn_id", f"turn_{len(turns_out):04d}"))
        safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in turn_id)
        clip_path = out_clip_dir / f"{safe_id}_{closeup}.mp4"
        roi_path = out_roi_dir / f"{safe_id}_{closeup}_mouth.npy"

        attempts += 1
        try:
            _ffmpeg_slice_video(src_video, clip_path, start, end)
            arr = _to_numpy(extractor.extract_from_file(str(clip_path)))
            if arr.ndim != 4 or arr.shape[1:] != (1, 96, 96):
                raise RuntimeError(f"unexpected ROI shape {arr.shape}")
            np.save(roi_path, arr)
            if not args.keep_clips:
                try:
                    clip_path.unlink()
                except OSError:
                    pass
        except Exception as exc:
            failures += 1
            print(f"[fail] {turn_id} {suffix}->{closeup}: {exc}")
            continue

        row = dict(turn)
        row["mouth_roi"] = str(roi_path)
        row["video_source"] = str(src_video)
        row["video_closeup"] = closeup
        row["video_speaker_map"] = f"{suffix}={closeup}"
        row["has_visual"] = True
        turns_out.append(row)
        print(
            f"[ok] {turn_id} {suffix}->{closeup} "
            f"{dur:.2f}s roi={tuple(arr.shape)} range=({arr.min():.3f},{arr.max():.3f})"
        )

    if not turns_out:
        print(
            "No visual turns produced. Check --speaker-closeup mapping, video paths, "
            "duration filters, and dlib model files.",
            file=sys.stderr,
        )
        return 1

    out_manifest = {
        "speakers": manifest.get("speakers", []),
        "turns": turns_out,
        "meta": {
            "source_manifest": str(args.manifest),
            "meeting_id": meeting_id,
            "speaker_closeup": closeup_by_speaker,
            "roi_backend": args.roi_backend,
            "visual_frontend": "ami_closeup_explicit_map_mouth_roi",
            "attempts": attempts,
            "failures": failures,
            "skipped_unmapped": skipped_unmapped,
            "skipped_duration": skipped_duration,
            "note": (
                "AMI closeup cameras are not filename-bound to speakers; "
                "interpret AV results only for explicitly verified mappings."
            ),
        },
    }
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_manifest, "w", encoding="utf-8") as f:
        json.dump(out_manifest, f, indent=2, ensure_ascii=False)

    print(
        f"\n[wrote] {args.out_manifest} "
        f"({len(turns_out)} visual turns, failures={failures}, attempts={attempts})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
