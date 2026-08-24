"""prepare_ami_from_hf.py — Download AMI audio + transcripts from HuggingFace
and convert to the project's manifest JSON format.

HuggingFace dataset  : edinburghcstr/ami  (config: ihm = individual headset mic)
Output audio         : datasets/ami/audio/utterances/*.wav and enrollment/*.wav
Output manifests     : data/ami_{split}/manifests/{MID}.json

Speaker ID mapping   : microphone_id H0N → seat suffix chr(ord('A')+N)
                       e.g.  H00 → IS1009c_A,  H01 → IS1009c_B  ...

Usage:
    python scripts/prepare_ami_from_hf.py [--splits train dev test] [--out-dir datasets/ami]

Options:
    --splits     Which splits to process (default: train dev test)
    --out-dir    Root dir for audio output (default: datasets/ami)
    --cache-dir  HuggingFace cache dir (default: ~/.cache/huggingface)
    --overwrite  Overwrite existing manifest/audio files

Notes:
    • Audio is stored as per-utterance WAVs plus quality-selected enrollment WAVs.
    • Video (Closeup*.avi) is NOT in HuggingFace — download separately:
          bash scripts/download_ami.sh --train --audio-only  (skip audio flag)
      or just run download_ami.sh for video-only by editing AUDIO_ONLY=1 manually.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ami_metadata import apply_ami_speaker_metadata, load_ami_meetings  # noqa: E402

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _mic_to_suffix(mic_id: str) -> str:
    """H00 -> A, H01 -> B, H02 -> C, H03 -> D"""
    m = re.match(r"H0?(\d)", mic_id)
    if not m:
        return mic_id
    return chr(ord("A") + int(m.group(1)))


def _speaker_id(meeting_id: str, mic_id: str) -> str:
    """IS1009c + H01 -> IS1009c_B"""
    return f"{meeting_id}_{_mic_to_suffix(mic_id)}"


def _save_wav(path: Path, array, sr: int) -> None:
    import numpy as np
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array, dtype=np.float32)
    sf.write(str(path), arr, sr)


def _safe_audio_id(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "utterance"


def _save_utterance_wav(
    row: dict[str, Any],
    meeting_id: str,
    audio_dir: Path,
    overwrite: bool,
) -> Path:
    mic_id = str(row["microphone_id"])
    audio_id = _safe_audio_id(row.get("audio_id", "utterance"))
    destination = audio_dir / "utterances" / f"{meeting_id}.{mic_id}.{audio_id}.wav"
    if overwrite or not destination.exists():
        _save_wav(
            destination,
            row["audio"]["array"],
            int(row["audio"]["sampling_rate"]),
        )
    return destination


def _build_quality_enrollment(
    rows: list[dict[str, Any]],
    destination: Path,
    overwrite: bool,
    target_seconds: float = 30.0,
) -> tuple[Path, dict[str, Any]]:
    """Concatenate high-RMS, text-bearing 3-8 s utterances for enrollment."""
    import numpy as np

    if destination.exists() and not overwrite:
        return destination, {"enrollment_mode": "existing_quality_enrollment"}
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        duration = float(row["end_time"]) - float(row["begin_time"])
        text = str(row.get("text", "")).strip()
        if not text or duration < 3.0 or duration > 8.0:
            continue
        samples = np.asarray(row["audio"]["array"], dtype=np.float32).reshape(-1)
        if samples.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
        candidates.append((rms, row))
    candidates.sort(key=lambda item: item[0], reverse=True)

    selected: list[dict[str, Any]] = []
    seconds = 0.0
    for _, row in candidates:
        selected.append(row)
        seconds += float(row["end_time"]) - float(row["begin_time"])
        if seconds >= target_seconds:
            break
    if not selected:
        raise RuntimeError(f"No quality enrollment utterances available for {destination.stem}")

    sampling_rates = {int(row["audio"]["sampling_rate"]) for row in selected}
    if len(sampling_rates) != 1:
        raise ValueError(f"Mixed enrollment sampling rates: {sampling_rates}")
    sampling_rate = sampling_rates.pop()
    merged = np.concatenate(
        [np.asarray(row["audio"]["array"], dtype=np.float32).reshape(-1) for row in selected]
    )
    merged = merged[: int(target_seconds * sampling_rate)]
    _save_wav(destination, merged, sampling_rate)
    return destination, {
        "enrollment_mode": "turn_quality",
        "enrollment_target_seconds": target_seconds,
        "enrollment_selected_seconds": float(merged.size / sampling_rate),
        "enrollment_selected_audio_ids": [str(row.get("audio_id", "")) for row in selected],
    }


# --------------------------------------------------------------------------- #
# Manifest builder
# --------------------------------------------------------------------------- #

def build_manifest(
    meeting_id: str,
    rows: list[dict],
    audio_dir: Path,
    overwrite: bool,
    meeting_speakers: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert HF rows for one meeting into the project manifest schema."""
    # Group by microphone so we can build per-speaker enrollment WAVs.
    by_mic: dict[str, list] = defaultdict(list)
    for row in rows:
        by_mic[row["microphone_id"]].append(row)

    # Build speakers block (one entry per unique participant/mic).
    seen_speakers: dict[str, dict] = {}
    for mic_id, utt_rows in by_mic.items():
        spk_id = _speaker_id(meeting_id, mic_id)
        enrollment_path, enrollment_meta = _build_quality_enrollment(
            utt_rows,
            audio_dir / "enrollment" / f"{spk_id}.wav",
            overwrite,
        )
        if spk_id not in seen_speakers:
            seen_speakers[spk_id] = {
                "speaker_id": spk_id,
                "session_speaker_id": spk_id,
                "nxt_agent": _mic_to_suffix(mic_id),
                "identity_scope": "meeting_local",
                "microphone_id": mic_id,
                "enrollment_audio": str(enrollment_path),
                "meta": enrollment_meta,
                # enrollment_face: populated later by prepare_ami_visual_manifest.py
            }

    # Build turns — one turn per utterance
    turns = []
    for row in sorted(rows, key=lambda r: (float(r["begin_time"]), r["microphone_id"])):
        mic_id = row["microphone_id"]
        spk_id = _speaker_id(meeting_id, mic_id)
        utterance_path = _save_utterance_wav(row, meeting_id, audio_dir, overwrite)
        turns.append({
            "turn_id": f"{meeting_id}.{row['audio_id']}",
            "start": float(row["begin_time"]),
            "end": float(row["end_time"]),
            "ref_speaker": spk_id,
            "session_speaker_id": spk_id,
            "nxt_agent": _mic_to_suffix(mic_id),
            "identity_scope": "meeting_local",
            "ref_text": row["text"].strip(),
            "audio": str(utterance_path),
            "audio_source_type": "individual_headset_microphone",
            "turn_boundary_source": "oracle_reference_transcript",
            # mouth_roi, video, speaker_mask_v: populated by prepare_ami_visual_manifest.py
        })

    manifest = {
        "meeting_id": meeting_id,
        "speakers": list(seen_speakers.values()),
        "turns": turns,
        "meta": {
            "source": "edinburghcstr/ami",
            "config": "ihm",
            "mic_to_suffix": "H0N -> chr(A+N)",
            "audio_condition": {
                "ami_microphone_setup": "ihm",
                "description": "individual_headset_microphone",
                "far_field": False,
            },
            "turn_boundary_source": "oracle_reference_transcript",
            "diarization_is_system_output": False,
            "speaker_identity_scope": "meeting_local",
        },
    }
    if meeting_speakers is not None:
        manifest = apply_ami_speaker_metadata(manifest, meeting_id, meeting_speakers)
    return manifest


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"],
                    choices=["train", "dev", "test"],
                    help="Which HF splits to process (default: all)")
    ap.add_argument("--out-dir", default=str(ROOT / "datasets" / "ami"),
                    help="Root dir for audio output")
    ap.add_argument("--cache-dir", default=None,
                    help="HuggingFace cache dir (default: ~/.cache/huggingface)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing manifest/audio files")
    ap.add_argument("--meeting-filter", nargs="*", default=None,
                    help="Only process these meeting IDs (default: all)")
    ap.add_argument(
        "--meetings-xml",
        type=Path,
        default=None,
        help=(
            "AMI corpusResources/meetings.xml. Defaults to "
            "<out-dir>/corpusResources/meetings.xml and is required unless "
            "--allow-meeting-local-speakers is set."
        ),
    )
    ap.add_argument(
        "--allow-meeting-local-speakers",
        action="store_true",
        help="Allow legacy meeting-local IDs when meetings.xml is unavailable.",
    )
    args = ap.parse_args()

    try:
        from datasets import load_dataset
        import soundfile  # noqa: F401
    except ImportError as e:
        print(f"[error] Missing dependency: {e}")
        print("Install with:  pip install datasets soundfile --break-system-packages")
        return 1

    audio_dir = Path(args.out_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    meetings_xml = args.meetings_xml or (Path(args.out_dir) / "corpusResources" / "meetings.xml")
    if meetings_xml.is_file():
        ami_meetings = load_ami_meetings(meetings_xml)
        print(f"[metadata] loaded {len(ami_meetings)} meetings from {meetings_xml}")
    elif args.allow_meeting_local_speakers:
        ami_meetings = {}
        print(
            f"[warning] {meetings_xml} is missing; using meeting-local speaker IDs",
            file=sys.stderr,
        )
    else:
        print(
            f"[error] AMI metadata missing: {meetings_xml}\n"
            "Run scripts/fetch_ami_metadata.py or pass --allow-meeting-local-speakers.",
            file=sys.stderr,
        )
        return 2

    # HF uses 'validation' not 'dev'
    hf_split_map = {"train": "train", "dev": "validation", "test": "test"}
    manifest_split_map = {"train": "ami_train", "dev": "ami_dev", "test": "ami_test"}

    for split in args.splits:
        hf_split = hf_split_map[split]
        manifest_dir = ROOT / "data" / manifest_split_map[split] / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f" Loading HF split: {hf_split}")
        print(f" Manifests -> {manifest_dir}")
        print(f"{'='*60}")

        ds = load_dataset(
            "edinburghcstr/ami",
            "ihm",
            split=hf_split,
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )

        # Group rows by meeting_id
        by_meeting: dict[str, list] = defaultdict(list)
        for row in ds:
            mid = row["meeting_id"]
            if args.meeting_filter and mid not in args.meeting_filter:
                continue
            by_meeting[mid].append(row)

        print(f"[info] {len(by_meeting)} meetings in split '{split}'")

        for i, (mid, rows) in enumerate(sorted(by_meeting.items()), 1):
            manifest_path = manifest_dir / f"{mid}.json"
            if manifest_path.exists() and not args.overwrite:
                print(f"[skip] {mid} ({len(rows)} utts) — manifest exists")
                continue

            print(f"[{i:3d}/{len(by_meeting)}] {mid}  ({len(rows)} utts)")
            meeting_speakers = ami_meetings.get(mid)
            if ami_meetings and meeting_speakers is None:
                raise KeyError(f"Meeting {mid} is missing from {meetings_xml}")
            manifest = build_manifest(
                mid,
                rows,
                audio_dir,
                args.overwrite,
                meeting_speakers=meeting_speakers,
            )

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"\n[done] {split}: {len(by_meeting)} meetings -> {manifest_dir}")

    print("\n" + "="*60)
    print(" All splits done.")
    print(" Next: download video then build visual manifests:")
    print("   bash scripts/download_ami.sh --train  (video only, edit AUDIO_ONLY=1)")
    print("   bash scripts/build_ami_train_manifests.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
