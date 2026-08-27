"""Attach mouth-ROI clips to an existing AMI session manifest.

AMI Closeup videos are meeting-length camera streams, not per-utterance face
videos.  Their speaker mapping changes between meetings, so production runs
must use the authoritative ``corpusResources/meetings.xml`` mapping.  Manual
``--speaker-closeup`` mappings remain available for isolated smoke tests.

Example:
    python scripts/prepare_ami_visual_manifest.py \
      --manifest data/ami_test/manifests/IS1009c.json \
      --ami-video-dir datasets/ami/video \
      --out-manifest data/ami_visual_smoke/IS1009c_closeup12.json \
      --out-dir data/ami_visual_smoke/IS1009c \
      --speaker-closeup A=Closeup1 B=Closeup2 \
      --max-turns 12 \
      --max-turns-per-speaker 6 \
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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ami_metadata import (  # noqa: E402
    apply_ami_speaker_metadata,
    infer_nxt_agent,
    load_ami_meetings,
    meeting_closeup_map,
)
from scripts.ami_visual_policy import (  # noqa: E402
    AMI_DATA_PROBLEMS_URL,
    is_official_missing_closeup,
)

_DEFAULT_FACE_PREDICTOR = "checkpoints/shape_predictor_68_face_landmarks.dat"
_DEFAULT_CNN_DETECTOR = "checkpoints/mmod_human_face_detector.dat"
_DEFAULT_MEAN_FACE = "av_hubert/avhubert/preparation/data/20words_mean_face.npy"
_FAILURE_LOG_SCHEMA = "ami_visual_failure_v1"
_EXCLUSION_LOG_SCHEMA = "ami_visual_source_exclusion_v2"


class VideoSliceError(RuntimeError):
    """Raised when ffmpeg cannot create a turn-level video clip."""

    def __init__(self, returncode: int, output: str):
        self.returncode = int(returncode)
        self.output = output.strip()
        super().__init__(
            f"ffmpeg slice failed with exit code {self.returncode}: "
            f"{self.output[-1000:]}"
        )


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


def _turn_agent(turn: dict[str, Any], meeting_id: str) -> str | None:
    return (
        str(turn.get("nxt_agent", "")).strip()
        or infer_nxt_agent(turn.get("session_speaker_id"), meeting_id)
        or infer_nxt_agent(turn.get("ref_speaker"), meeting_id)
    )


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
    completed = subprocess.run(
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
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise VideoSliceError(completed.returncode, completed.stdout or "")


def _probe_video(path: Path) -> dict[str, Any]:
    """Return lightweight decoder evidence suitable for a failure sidecar."""

    probe: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }
    if not path.exists():
        return probe
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            probe["opened"] = bool(cap.isOpened())
            probe["reported_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            probe["fps"] = float(cap.get(cv2.CAP_PROP_FPS))
            if probe["fps"] > 0 and probe["reported_frames"] >= 0:
                probe["reported_duration_seconds"] = (
                    probe["reported_frames"] / probe["fps"]
                )
            ok, frame = cap.read()
            probe["first_frame_readable"] = bool(ok and frame is not None)
            if frame is not None:
                probe["first_frame_shape"] = list(frame.shape)
        finally:
            cap.release()
    except Exception as exc:  # Diagnostic code must never hide the real failure.
        probe["probe_error"] = f"{type(exc).__name__}: {exc}"
    return probe


def _classify_failure(exc: Exception) -> tuple[str, str]:
    message = str(exc)
    lowered = message.lower()
    if isinstance(exc, VideoSliceError):
        return "ffmpeg_slice_failed", "ffmpeg_slice"
    if "could not read any frames" in lowered:
        return "clip_unreadable", "video_decode"
    if "landmark detection failed on all frames" in lowered:
        return "landmark_all_frames", "landmark_detection"
    if "unexpected roi shape" in lowered:
        return "unexpected_roi_shape", "roi_validation"
    if "lip confidence length" in lowered:
        return "confidence_length_mismatch", "roi_validation"
    if "frame coverage" in lowered:
        return "roi_frame_coverage", "roi_validation"
    return "unexpected_exception", "unknown"


def _append_failure(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float32)


def _save_enrollment_frame(video_path: Path, out_path: Path, timestamp_s: float) -> bool:
    """Save one closeup frame for optional face enrollment.

    The closeup stream is the best available face evidence in this smoke
    manifest. InsightFace will still reject it later if no face is visible.
    """

    try:
        import cv2
    except ImportError:
        return False

    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_s) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(out_path), frame))
    finally:
        cap.release()


def _with_visual_speaker_fields(row: dict[str, Any], arr: np.ndarray) -> dict[str, Any]:
    row["has_visual"] = True
    # The clip was sliced from the explicitly mapped closeup stream for this
    # reference speaker, so every ROI frame belongs to that speaker track.
    row["speaker_mask_v"] = [True] * int(arr.shape[0])
    return row


def _with_enrollment_faces(
    speakers: list[dict[str, Any]],
    face_by_suffix: dict[str, str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for speaker in speakers:
        row = dict(speaker)
        agent = (
            str(row.get("nxt_agent", "")).strip()
            or _speaker_suffix(str(row.get("session_speaker_id") or row.get("speaker_id", "")))
        )
        if not row.get("enrollment_face") and agent in face_by_suffix:
            row["enrollment_face"] = face_by_suffix[agent]
        out.append(row)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--ami-video-dir", default="datasets/ami/video", type=Path)
    p.add_argument("--out-manifest", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--speaker-closeup",
        nargs="+",
        default=None,
        help="Manual mapping for smoke tests. Production should use --meetings-xml.",
    )
    p.add_argument(
        "--meetings-xml",
        type=Path,
        default=Path("datasets/ami/corpusResources/meetings.xml"),
        help="Authoritative AMI speaker/camera metadata.",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Optional smoke-test cap. Default: process every eligible turn.",
    )
    p.add_argument(
        "--max-turns-per-speaker",
        type=int,
        default=None,
        help="Optional cap per mapped speaker suffix, useful for multi-speaker smoke manifests.",
    )
    p.add_argument("--min-turn-secs", type=float, default=1.0)
    p.add_argument("--max-turn-secs", type=float, default=12.0)
    p.add_argument(
        "--source-duration-tolerance",
        type=float,
        default=0.25,
        help="Maximum allowed turn-end overrun beyond the probed source video.",
    )
    p.add_argument("--keep-clips", action="store_true")
    p.add_argument(
        "--failure-log",
        type=Path,
        default=None,
        help="Per-turn JSONL failure ledger. Default: OUT_DIR/failures.jsonl.",
    )
    p.add_argument(
        "--exclusion-log",
        type=Path,
        default=None,
        help="Fixed AV-valid source-exclusion JSONL ledger. Default: OUT_DIR/exclusions.jsonl.",
    )
    p.add_argument("--roi-backend", default="dlib", choices=["dlib", "haar"])
    p.add_argument("--face-predictor", default=_DEFAULT_FACE_PREDICTOR)
    p.add_argument("--cnn-detector", default=_DEFAULT_CNN_DETECTOR)
    p.add_argument("--mean-face", default=_DEFAULT_MEAN_FACE)
    p.add_argument(
        "--no-enrollment-faces",
        action="store_true",
        help="Do not save closeup-frame enrollment_face images into the output manifest.",
    )
    args = p.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    meeting_id = _extract_meeting_id(manifest, args.manifest)
    mapping_source: str
    if args.meetings_xml.is_file():
        all_meetings = load_ami_meetings(args.meetings_xml)
        if meeting_id not in all_meetings:
            raise KeyError(f"Meeting {meeting_id} is missing from {args.meetings_xml}")
        meeting_speakers = all_meetings[meeting_id]
        manifest = apply_ami_speaker_metadata(manifest, meeting_id, meeting_speakers)
        official_mapping = meeting_closeup_map(meeting_speakers)
        if args.speaker_closeup:
            manual_mapping = _parse_speaker_closeup(args.speaker_closeup)
            disagreements = {
                agent: (camera, official_mapping.get(agent))
                for agent, camera in manual_mapping.items()
                if official_mapping.get(agent) != camera
            }
            if disagreements:
                raise ValueError(
                    f"Manual camera mapping disagrees with meetings.xml: {disagreements}"
                )
            closeup_by_speaker = manual_mapping
            mapping_source = "AMI meetings.xml verified manual subset"
        else:
            closeup_by_speaker = official_mapping
            mapping_source = "AMI corpusResources/meetings.xml"
    elif args.speaker_closeup:
        closeup_by_speaker = _parse_speaker_closeup(args.speaker_closeup)
        mapping_source = "manual_unverified"
        print(
            f"[warning] {args.meetings_xml} is missing; global participant IDs are unavailable",
            file=sys.stderr,
        )
    else:
        raise FileNotFoundError(
            f"{args.meetings_xml} is required for a verified AMI camera mapping. "
            "Run scripts/fetch_ami_metadata.py first."
        )
    extractor = _build_extractor(args)

    out_roi_dir = args.out_dir / "mouth_roi"
    out_clip_dir = args.out_dir / "video_clips"
    out_face_dir = args.out_dir / "enrollment_faces"
    failure_log_path = args.failure_log or (args.out_dir / "failures.jsonl")
    exclusion_log_path = args.exclusion_log or (args.out_dir / "exclusions.jsonl")
    out_roi_dir.mkdir(parents=True, exist_ok=True)
    out_clip_dir.mkdir(parents=True, exist_ok=True)
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)
    failure_log_path.write_text("", encoding="utf-8")
    exclusion_log_path.write_text("", encoding="utf-8")

    turns_out: list[dict[str, Any]] = []
    face_by_suffix: dict[str, str] = {}
    attempts = 0
    eligible_turns = 0
    failures = 0
    source_exclusions = 0
    official_missing_exclusions = 0
    source_duration_exclusions = 0
    skipped_unmapped = 0
    skipped_duration = 0
    skipped_per_speaker_cap = 0
    turns_by_suffix: dict[str, int] = {}
    confidence_values: list[float] = []
    failure_reasons: Counter[str] = Counter()
    source_exclusion_counts: Counter[str] = Counter()
    avhubert_resize_fallback_turns = 0
    source_probes: dict[Path, dict[str, Any]] = {}

    def record_failure(
        turn: dict[str, Any],
        *,
        reason: str,
        stage: str,
        message: str,
        suffix: str | None,
        closeup: str | None,
        src_video: Path | None,
        clip_path: Path | None,
        roi_path: Path | None,
        exception_type: str | None = None,
        ffmpeg_output: str | None = None,
    ) -> None:
        nonlocal failures
        failures += 1
        failure_reasons[reason] += 1
        source_probe = None
        if src_video is not None:
            if src_video not in source_probes:
                source_probes[src_video] = _probe_video(src_video)
            source_probe = source_probes[src_video]
        record = {
            "schema": _FAILURE_LOG_SCHEMA,
            "dataset_build_id": manifest.get("meta", {}).get("dataset_build_id"),
            "meeting_id": meeting_id,
            "turn_id": str(turn.get("turn_id", "<unknown>")),
            "reason": reason,
            "stage": stage,
            "exception_type": exception_type,
            "message": message,
            "start": turn.get("start"),
            "end": turn.get("end"),
            "duration_seconds": (
                float(turn["end"]) - float(turn["start"])
                if turn.get("start") is not None and turn.get("end") is not None
                else None
            ),
            "ref_speaker": turn.get("ref_speaker"),
            "session_speaker_id": turn.get("session_speaker_id"),
            "participant_id": turn.get("participant_id"),
            "nxt_agent": suffix,
            "closeup": closeup,
            "source_video": str(src_video) if src_video is not None else None,
            "clip_path": str(clip_path) if clip_path is not None else None,
            "roi_path": str(roi_path) if roi_path is not None else None,
            "source_probe": source_probe,
            "clip_probe": _probe_video(clip_path) if clip_path is not None else None,
            "ffmpeg_output": ffmpeg_output[-4000:] if ffmpeg_output else None,
        }
        _append_failure(failure_log_path, record)
        print(
            f"[fail] reason={reason} stage={stage} turn={record['turn_id']} "
            f"agent={suffix} camera={closeup}: {message}",
            flush=True,
        )

    def record_source_exclusion(
        turn: dict[str, Any],
        *,
        reason: str,
        suffix: str,
        closeup: str,
        src_video: Path,
        source_probe: dict[str, Any] | None,
    ) -> None:
        nonlocal source_exclusions
        source_exclusions += 1
        source_exclusion_counts[f"{reason}|{meeting_id}.{closeup}"] += 1
        _append_failure(
            exclusion_log_path,
            {
                "schema": _EXCLUSION_LOG_SCHEMA,
                "dataset_build_id": manifest.get("meta", {}).get("dataset_build_id"),
                "meeting_id": meeting_id,
                "turn_id": str(turn.get("turn_id", "<unknown>")),
                "reason": reason,
                "start": turn.get("start"),
                "end": turn.get("end"),
                "nxt_agent": suffix,
                "closeup": closeup,
                "source_video": str(src_video),
                "source_probe": source_probe,
                "evidence": (
                    AMI_DATA_PROBLEMS_URL
                    if reason == "official_missing_closeup"
                    else "official AMI mirror byte-size parity plus local ffprobe duration"
                ),
            },
        )
        print(
            f"[exclude] reason={reason} turn={turn.get('turn_id')} "
            f"agent={suffix} camera={closeup}",
            flush=True,
        )

    for turn in manifest.get("turns", []):
        if args.max_turns is not None and len(turns_out) >= args.max_turns:
            break

        turn_id = str(turn.get("turn_id", f"turn_{len(turns_out):04d}"))
        start = float(turn["start"])
        end = float(turn["end"])
        dur = end - start
        if dur < args.min_turn_secs or dur > args.max_turn_secs:
            skipped_duration += 1
            continue

        ref_speaker = str(turn.get("ref_speaker", ""))
        suffix = _turn_agent(turn, meeting_id)
        if not suffix:
            attempts += 1
            record_failure(
                turn,
                reason="missing_nxt_agent",
                stage="metadata",
                message="No meeting-local NXT agent could be resolved",
                suffix=None,
                closeup=None,
                src_video=None,
                clip_path=None,
                roi_path=None,
            )
            continue
        closeup = closeup_by_speaker.get(suffix)
        if closeup is None:
            skipped_unmapped += 1
            attempts += 1
            record_failure(
                turn,
                reason="missing_closeup_mapping",
                stage="metadata",
                message=f"No Closeup camera mapping for NXT agent {suffix}",
                suffix=suffix,
                closeup=None,
                src_video=None,
                clip_path=None,
                roi_path=None,
            )
            continue
        if (
            args.max_turns_per_speaker is not None
            and turns_by_suffix.get(suffix, 0) >= args.max_turns_per_speaker
        ):
            skipped_per_speaker_cap += 1
            continue

        src_video = args.ami_video_dir / f"{meeting_id}.{closeup}.avi"
        safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in turn_id)
        clip_path = out_clip_dir / f"{safe_id}_{closeup}.mp4"
        roi_path = out_roi_dir / f"{safe_id}_{closeup}_mouth.npy"
        face_path = out_face_dir / f"{meeting_id}_{suffix}_{closeup}.jpg"
        eligible_turns += 1
        if is_official_missing_closeup(meeting_id, closeup):
            official_missing_exclusions += 1
            record_source_exclusion(
                turn,
                reason="official_missing_closeup",
                suffix=suffix,
                closeup=closeup,
                src_video=src_video,
                source_probe=None,
            )
            continue
        if not src_video.exists():
            attempts += 1
            record_failure(
                turn,
                reason="missing_source_video",
                stage="source_validation",
                message=f"Source video does not exist: {src_video}",
                suffix=suffix,
                closeup=closeup,
                src_video=src_video,
                clip_path=clip_path,
                roi_path=roi_path,
            )
            continue
        if src_video not in source_probes:
            source_probes[src_video] = _probe_video(src_video)
        source_probe = source_probes[src_video]
        source_duration = source_probe.get("reported_duration_seconds")
        if (
            source_probe.get("opened") is True
            and source_duration is not None
            and end > float(source_duration) + args.source_duration_tolerance
        ):
            source_duration_exclusions += 1
            record_source_exclusion(
                turn,
                reason="source_duration_out_of_bounds",
                suffix=suffix,
                closeup=closeup,
                src_video=src_video,
                source_probe=source_probe,
            )
            continue
        attempts += 1
        try:
            _ffmpeg_slice_video(src_video, clip_path, start, end)
            extracted, lip_conf = extractor.extract_with_confidence_from_file(str(clip_path))
            arr = _to_numpy(extracted)
            lip_conf = np.asarray(lip_conf, dtype=np.float32).reshape(-1)
            if arr.ndim != 4 or arr.shape[1:] != (1, 96, 96):
                raise RuntimeError(f"unexpected ROI shape {arr.shape}")
            if lip_conf.shape[0] != arr.shape[0]:
                raise RuntimeError(
                    f"lip confidence length {lip_conf.shape[0]} != ROI frames {arr.shape[0]}"
                )
            expected_frames = max(1, int(round(dur * 25.0)))
            frame_coverage = float(arr.shape[0]) / expected_frames
            np.save(roi_path, arr)
            best_frame = int(np.argmax(lip_conf)) if lip_conf.size else 0
            face_timestamp = min(end, start + (best_frame + 0.5) / 25.0)
            if (
                not args.no_enrollment_faces
                and suffix not in face_by_suffix
                and _save_enrollment_frame(src_video, face_path, face_timestamp)
            ):
                face_by_suffix[suffix] = str(face_path)
            if not args.keep_clips:
                try:
                    clip_path.unlink()
                except OSError:
                    pass
        except Exception as exc:
            reason, stage = _classify_failure(exc)
            record_failure(
                turn,
                reason=reason,
                stage=stage,
                message=str(exc),
                suffix=suffix,
                closeup=closeup,
                src_video=src_video,
                clip_path=clip_path,
                roi_path=roi_path,
                exception_type=type(exc).__name__,
                ffmpeg_output=exc.output if isinstance(exc, VideoSliceError) else None,
            )
            continue

        row = dict(turn)
        row["mouth_roi"] = str(roi_path)
        row["video_source"] = str(src_video)
        row["video_closeup"] = closeup
        row["video_speaker_map"] = f"{suffix}={closeup}"
        row["lip_conf_v"] = lip_conf.tolist()
        used_resize_fallback = bool(
            args.roi_backend == "dlib"
            and lip_conf.size > 0
            and np.allclose(lip_conf, 0.0)
        )
        if used_resize_fallback:
            avhubert_resize_fallback_turns += 1
            row["lip_conf_source"] = "avhubert_full_frame_resize_fallback"
            row["visual_preprocessing_status"] = "all_landmarks_missing_resized"
        else:
            row["lip_conf_source"] = (
                f"{args.roi_backend}_direct_detection_and_interpolation"
            )
        row["video_expected_frames_25fps"] = expected_frames
        row["video_frame_coverage"] = frame_coverage
        row = _with_visual_speaker_fields(row, arr)
        turns_out.append(row)
        confidence_values.extend(float(value) for value in lip_conf)
        turns_by_suffix[suffix] = turns_by_suffix.get(suffix, 0) + 1
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
        "speakers": _with_enrollment_faces(manifest.get("speakers", []), face_by_suffix),
        "turns": turns_out,
        "meta": {
            **manifest.get("meta", {}),
            "source_manifest": str(args.manifest),
            "meeting_id": meeting_id,
            "speaker_closeup": closeup_by_speaker,
            "speaker_closeup_source": mapping_source,
            "roi_backend": args.roi_backend,
            "visual_frontend": "ami_closeup_explicit_map_mouth_roi",
            "enrollment_faces": face_by_suffix,
            "speaker_mask_v": "all_true_for_each_explicitly_mapped_closeup_turn",
            "attempts": attempts,
            "eligible_turns": eligible_turns,
            "failures": failures,
            "visual_source_exclusions": source_exclusions,
            "official_visual_exclusions": official_missing_exclusions,
            "source_duration_exclusions": source_duration_exclusions,
            "visual_source_exclusion_counts": dict(
                sorted(source_exclusion_counts.items())
            ),
            "visual_source_exclusion_log": str(exclusion_log_path),
            "visual_source_exclusion_log_schema": _EXCLUSION_LOG_SCHEMA,
            "official_visual_exclusion_source": AMI_DATA_PROBLEMS_URL,
            "source_duration_tolerance_seconds": args.source_duration_tolerance,
            "successful_visual_turns": len(turns_out),
            "visual_turn_coverage": len(turns_out) / attempts if attempts else None,
            "eligible_visual_turn_coverage": (
                len(turns_out) / eligible_turns if eligible_turns else None
            ),
            "failure_log": str(failure_log_path),
            "failure_log_schema": _FAILURE_LOG_SCHEMA,
            "failure_reason_counts": dict(sorted(failure_reasons.items())),
            "skipped_unmapped": skipped_unmapped,
            "skipped_duration": skipped_duration,
            "skipped_per_speaker_cap": skipped_per_speaker_cap,
            "turns_by_speaker_suffix": turns_by_suffix,
            "turn_limit": args.max_turns,
            "turn_limit_per_speaker": args.max_turns_per_speaker,
            "lip_conf_source": f"{args.roi_backend}_direct_detection_and_interpolation",
            "all_landmarks_missing_policy": "AV-HuBERT full-frame resize",
            "avhubert_resize_fallback_turns": avhubert_resize_fallback_turns,
            "lip_conf_mean": (
                float(np.mean(confidence_values)) if confidence_values else None
            ),
            "note": (
                "AMI closeup mappings and global identities come from meetings.xml; "
                "officially missing streams are excluded, and all-landmark-missing "
                "readable clips use the AV-HuBERT reference resize fallback."
            ),
        },
    }
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_manifest, "w", encoding="utf-8") as f:
        json.dump(out_manifest, f, indent=2, ensure_ascii=False)

    print(
        f"\n[wrote] {args.out_manifest} "
        f"({len(turns_out)} visual turns, failures={failures}, attempts={attempts}, "
        f"source_exclusions={source_exclusions}, "
        f"coverage={len(turns_out) / attempts if attempts else 0:.3f}, "
        f"failure_log={failure_log_path})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
