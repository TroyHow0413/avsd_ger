"""Build a real-data manifest for Phase D smoke / eval from LRS2 mp4 clips.

LRS2 ships per-utterance:
    datasets/lrs2/main/<id>/<utt>.mp4    # 25 fps, 160x160 face-cropped video + audio
    datasets/lrs2/main/<id>/<utt>.txt    # 'Text: <ref>\nConf: <int>'

For each chosen utterance this script:
    1. Extracts 16 kHz mono wav  via ffmpeg.
    2. Extracts a [T, 1, 96, 96] grayscale mouth ROI npy by:
         - reading frames with opencv,
         - center-crop to 88x88 (LRS2 is already face-cropped so the mouth
           sits roughly at the lower-middle; this is good enough for Phase D
           smoke. For paper-grade results use av_hubert/preparation/align_mouth.py),
         - converting to gray + resizing to 96x96, normalising to [0, 1].
    3. Pairs it with the reference transcript from <utt>.txt.
    4. Writes everything to a manifest JSON ready for run_sample / eval_ablations.

Speakers in LRS2 are anonymous (the parent directory name is a hash). For
single-speaker enrollment we treat every clip in the same parent directory as
the same speaker; for multi-speaker testing pick utterances from different
directories.

Usage:
    # Pick 1 utterance for run_sample.py:
    python scripts/prepare_lrs2_manifest.py \
        --lrs2-root datasets/lrs2/main \
        --out-manifest data/lrs2_smoke_manifest.json \
        --out-utts data/utts/ \
        --max-utts 1

    # Pick 3 utterances from 2 distinct dirs for a session manifest:
    python scripts/prepare_lrs2_manifest.py \
        --lrs2-root datasets/lrs2/main \
        --out-manifest data/lrs2_session_manifest.json \
        --out-utts data/utts/ \
        --max-utts 3 --session
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _read_lrs2_text(path: Path) -> str:
    """Parse LRS2 transcript file — first line is `Text: <ref>`."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("Text:"):
            return line.split("Text:", 1)[1].strip()
    return ""


def _extract_wav(mp4_path: Path, out_wav: Path) -> bool:
    """ffmpeg: mp4 audio -> 16 kHz mono wav."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(mp4_path),
        "-ar", "16000", "-ac", "1", "-vn",
        str(out_wav),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return out_wav.exists()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [ffmpeg] failed on {mp4_path}: {e}")
        return False


def _extract_mouth_roi(mp4_path: Path, out_npy: Path) -> bool:
    """opencv: mp4 frames -> [T, 1, 96, 96] grayscale, normalised to [0,1].

    LRS2 videos are already face-cropped 160x160 at 25 fps. We do a
    center-crop to 88x88 (covering mouth + chin region in most frames) then
    resize to 96x96 grayscale. This is rougher than av_hubert's landmark-based
    align_mouth.py but lets Phase D smoke run today without setting up dlib.
    """
    try:
        import cv2
    except ImportError:
        print("  [cv2] not available; cannot build mouth ROI")
        return False

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"  [cv2] cannot open {mp4_path}")
        return False

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # frame: [H, W, 3] BGR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # [H, W]
        h, w = gray.shape
        # Center-crop to 88x88 around the lower-middle (mouth region).
        # LRS2 face crops have eyes ~y=60, mouth ~y=110-130 in 160x160.
        cy = min(h - 1, int(h * 0.7))   # ~70% from top => mouth-ish
        cx = w // 2
        half = 44  # 88 / 2
        y0, y1 = max(0, cy - half), min(h, cy + half)
        x0, x1 = max(0, cx - half), min(w, cx + half)
        crop = gray[y0:y1, x0:x1]
        # Pad to 88x88 if the crop fell off an edge
        if crop.shape != (88, 88):
            padded = np.zeros((88, 88), dtype=np.uint8)
            ph, pw = crop.shape
            padded[:ph, :pw] = crop
            crop = padded
        crop96 = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
        frames.append(crop96)
    cap.release()

    if not frames:
        print(f"  [cv2] zero frames decoded from {mp4_path}")
        return False

    arr = np.stack(frames, axis=0)              # [T, 96, 96] uint8
    arr = arr.astype(np.float32) / 255.0        # [0, 1]
    arr = arr[:, np.newaxis, :, :]              # [T, 1, 96, 96]
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, arr)
    return True


def _iter_lrs2_utts(root: Path):
    """Yield (mp4_path, txt_path, speaker_dir) for every LRS2 clip."""
    for spk_dir in sorted(root.iterdir()):
        if not spk_dir.is_dir():
            continue
        for mp4 in sorted(spk_dir.glob("*.mp4")):
            txt = mp4.with_suffix(".txt")
            if txt.exists():
                yield mp4, txt, spk_dir.name


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lrs2-root", default="datasets/lrs2/main", type=Path)
    p.add_argument("--out-manifest", required=True, type=Path)
    p.add_argument("--out-utts", default="data/utts", type=Path,
                   help="Directory for per-utt extracted wav + npy files.")
    p.add_argument("--max-utts", type=int, default=1,
                   help="Max number of utterances to include.")
    p.add_argument("--max-speakers", type=int, default=1,
                   help="Max distinct speaker dirs to draw from (>=2 for multi-speaker test).")
    p.add_argument("--session", action="store_true",
                   help="Emit a `turns`-style session manifest instead of `utterances` style.")
    args = p.parse_args()

    if not args.lrs2_root.exists():
        print(f"LRS2 root not found: {args.lrs2_root}", file=sys.stderr)
        return 2

    speakers: dict[str, list] = {}
    for mp4, txt, spk in _iter_lrs2_utts(args.lrs2_root):
        if len(speakers) >= args.max_speakers and spk not in speakers:
            continue
        speakers.setdefault(spk, []).append((mp4, txt))
        n_total = sum(len(v) for v in speakers.values())
        if n_total >= args.max_utts and len(speakers) >= 1:
            break

    print(f"[LRS2] picked {sum(len(v) for v in speakers.values())} utts "
          f"across {len(speakers)} speakers")

    speakers_block = []
    utterances_block = []
    turns_block = []
    t_offset = 0.0

    for i, (spk, items) in enumerate(speakers.items()):
        speakers_block.append({
            "speaker_id": f"spk_{i+1:02d}",
            # Use the first clip's wav as the enrolment audio (extracted next).
            "enrollment_audio": str(args.out_utts / spk / f"{items[0][0].stem}.wav"),
            "enrollment_face": None,
            "meta": {"lrs2_dir": spk},
        })
        for mp4, txt in items:
            utt_id = f"{spk}_{mp4.stem}"
            wav_out = args.out_utts / spk / f"{mp4.stem}.wav"
            roi_out = args.out_utts / spk / f"{mp4.stem}_mouth.npy"
            print(f"  -> {utt_id}")
            print(f"     wav: {wav_out}")
            print(f"     roi: {roi_out}")
            ok_wav = _extract_wav(mp4, wav_out)
            ok_roi = _extract_mouth_roi(mp4, roi_out)
            if not (ok_wav and ok_roi):
                print(f"     SKIP (extraction failed)")
                continue
            ref_text = _read_lrs2_text(txt)
            arr = np.load(roi_out, mmap_mode="r")
            duration = arr.shape[0] / 25.0   # 25 fps

            utterances_block.append({
                "utt_id": utt_id,
                "speaker_id": f"spk_{i+1:02d}",
                "audio": str(wav_out),
                "mouth_roi": str(roi_out),
                "transcript_gold": ref_text,
            })
            turns_block.append({
                "turn_id": utt_id,
                "start": float(t_offset),
                "end": float(t_offset + duration),
                "audio": str(wav_out),
                "mouth_roi": str(roi_out),
                "ref_text": ref_text,
                "ref_speaker": f"spk_{i+1:02d}",
            })
            t_offset += duration + 0.2

    if not utterances_block:
        print("No utterances extracted — check ffmpeg + opencv-python install.")
        return 1

    manifest = {"speakers": speakers_block}
    if args.session:
        manifest["turns"] = turns_block
    else:
        manifest["utterances"] = utterances_block

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[wrote] {args.out_manifest}  "
          f"({len(speakers_block)} speakers, "
          f"{len(turns_block) if args.session else len(utterances_block)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
