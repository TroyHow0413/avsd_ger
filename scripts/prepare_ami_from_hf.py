"""prepare_ami_from_hf.py — Download AMI audio + transcripts from HuggingFace
and convert to the project's manifest JSON format.

HuggingFace dataset  : edinburghcstr/ami  (config: ihm = individual headset mic)
Output audio         : datasets/ami/audio/{MID}.Headset-{N}.wav  (full meeting WAV,
                       reconstructed by concatenating utterances in time order)
Output manifests     : data/ami_{split}/manifests/{MID}.json

Speaker ID mapping   : microphone_id H0N → seat suffix chr(ord('A')+N)
                       e.g.  H00 → IS1009c_A,  H01 → IS1009c_B  ...

Usage:
    python scripts/prepare_ami_from_hf.py [--splits train dev test] [--out-dir datasets/ami]

Options:
    --splits     Which splits to process (default: train dev test)
    --out-dir    Root dir for audio output (default: datasets/ami)
    --cache-dir  HuggingFace cache dir (default: ~/.cache/huggingface)
    --jobs       Parallel workers for WAV writing (default: 4)
    --overwrite  Overwrite existing manifest/audio files

Notes:
    • Audio stored as per-utterance WAVs under datasets/ami/audio/utterances/
      AND merged into per-speaker meeting WAVs (Headset-N.wav) for eval pipeline.
    • Video (Closeup*.avi) is NOT in HuggingFace — download separately:
          bash scripts/download_ami.sh --train --audio-only  (skip audio flag)
      or just run download_ami.sh for video-only by editing AUDIO_ONLY=1 manually.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

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


def _merge_speaker_wav(
    utt_rows: list[dict],
    meeting_id: str,
    mic_id: str,
    audio_dir: Path,
    overwrite: bool,
) -> Path:
    """Concatenate all utterances for one speaker into a full-meeting WAV.

    The merged file is written to datasets/ami/audio/{MID}.Headset-{N}.wav,
    matching the layout expected by eval_ablations.py and prepare_ami_visual_manifest.py.

    Gaps between utterances are filled with silence so begin/end times in the
    manifest remain valid offsets into the merged file.
    """
    import numpy as np
    import soundfile as sf

    n = int(re.match(r"H0?(\d)", mic_id).group(1))
    dst = audio_dir / f"{meeting_id}.Headset-{n}.wav"
    if dst.exists() and not overwrite:
        return dst

    if not utt_rows:
        return dst

    sr = utt_rows[0]["audio"]["sampling_rate"]
    # Sort by begin_time so the merged file is chronological
    rows = sorted(utt_rows, key=lambda r: float(r["begin_time"]))

    max_end = max(float(r["end_time"]) for r in rows)
    total_samples = int(max_end * sr) + sr  # +1 s safety margin
    merged = np.zeros(total_samples, dtype=np.float32)

    for row in rows:
        start_s = float(row["begin_time"])
        start_i = int(start_s * sr)
        arr = np.asarray(row["audio"]["array"], dtype=np.float32)
        end_i = start_i + len(arr)
        if end_i > len(merged):
            merged = np.pad(merged, (0, end_i - len(merged)))
        merged[start_i:end_i] = arr

    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), merged, sr)
    print(f"  [wav] {dst.name}  ({len(rows)} utts, {max_end:.1f}s)")
    return dst


# --------------------------------------------------------------------------- #
# Manifest builder
# --------------------------------------------------------------------------- #

def build_manifest(
    meeting_id: str,
    rows: list[dict],
    audio_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Convert HF rows for one meeting into the project manifest schema."""
    # Group by microphone so we can build per-speaker merged WAVs
    by_mic: dict[str, list] = defaultdict(list)
    for row in rows:
        by_mic[row["microphone_id"]].append(row)

    # Build merged Headset WAVs
    headset_paths: dict[str, Path] = {}
    for mic_id, utt_rows in by_mic.items():
        p = _merge_speaker_wav(utt_rows, meeting_id, mic_id, audio_dir, overwrite)
        headset_paths[mic_id] = p

    # Build speakers block (one entry per unique participant/mic)
    # enrollment_audio = path to merged Headset WAV for that speaker
    seen_speakers: dict[str, dict] = {}
    for mic_id, utt_rows in by_mic.items():
        spk_id = _speaker_id(meeting_id, mic_id)
        if spk_id not in seen_speakers:
            seen_speakers[spk_id] = {
                "speaker_id": spk_id,
                "microphone_id": mic_id,
                "enrollment_audio": str(headset_paths[mic_id]),
                # enrollment_face: populated later by prepare_ami_visual_manifest.py
            }

    # Build turns — one turn per utterance
    turns = []
    for row in sorted(rows, key=lambda r: (float(r["begin_time"]), r["microphone_id"])):
        mic_id = row["microphone_id"]
        spk_id = _speaker_id(meeting_id, mic_id)
        turns.append({
            "turn_id": f"{meeting_id}.{row['audio_id']}",
            "start": float(row["begin_time"]),
            "end": float(row["end_time"]),
            "ref_speaker": spk_id,
            "ref_text": row["text"].strip(),
            "audio": str(headset_paths.get(mic_id, "")),
            # mouth_roi, video, speaker_mask_v: populated by prepare_ami_visual_manifest.py
        })

    return {
        "meeting_id": meeting_id,
        "speakers": list(seen_speakers.values()),
        "turns": turns,
        "meta": {
            "source": "edinburghcstr/ami",
            "config": "ihm",
            "mic_to_suffix": "H0N -> chr(A+N)",
        },
    }


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
            manifest = build_manifest(mid, rows, audio_dir, args.overwrite)

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
