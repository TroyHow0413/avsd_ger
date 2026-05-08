"""Build a real-data manifest for Phase D smoke / eval from LRS2 mp4 clips.

LRS2 ships per-utterance:
    datasets/lrs2/main/<id>/<utt>.mp4    # 25 fps, 160x160 face-cropped video + audio
    datasets/lrs2/main/<id>/<utt>.txt    # 'Text: <ref>\nConf: <int>'

For each chosen utterance this script:
    1. Extracts 16 kHz mono wav  via ffmpeg.
    2. Extracts a [T, 1, 96, 96] grayscale mouth ROI npy via MouthROIExtractor:
         - backend='dlib'  (default): landmark detection + affine alignment,
           identical to av_hubert/preparation/align_mouth.py. Requires dlib +
           three model files (see configs/default.yaml mouth_roi section).
         - backend='haar'  (fallback): fixed center-crop, no model files needed.
           Use --roi-backend haar if dlib is not installed.
    3. Pairs it with the reference transcript from <utt>.txt.
    4. Writes everything to a manifest JSON ready for run_sample / eval_ablations.

Speakers in LRS2 are anonymous (the parent directory name is a hash). For
single-speaker enrollment we treat every clip in the same parent directory as
the same speaker; for multi-speaker testing pick utterances from different
directories.

Usage:
    # Pick 1 utterance for run_sample.py (dlib, production):
    python scripts/prepare_lrs2_manifest.py \
        --lrs2-root datasets/lrs2/main \
        --out-manifest data/lrs2_smoke_manifest.json \
        --out-utts data/utts/ \
        --max-utts 1

    # Fallback without dlib:
    python scripts/prepare_lrs2_manifest.py \
        --lrs2-root datasets/lrs2/main \
        --out-manifest data/lrs2_smoke_manifest.json \
        --out-utts data/utts/ \
        --max-utts 1 --roi-backend haar

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

# Default dlib model paths (relative to project root).
# Override via --face-predictor / --cnn-detector / --mean-face args.
_DEFAULT_FACE_PREDICTOR = "checkpoints/shape_predictor_68_face_landmarks.dat"
_DEFAULT_CNN_DETECTOR   = "checkpoints/mmod_human_face_detector.dat"
_DEFAULT_MEAN_FACE      = "av_hubert/avhubert/preparation/data/20words_mean_face.npy"


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


def _extract_mouth_roi(
    mp4_path: Path,
    out_npy: Path,
    extractor,
) -> bool:
    """Use MouthROIExtractor to produce [T, 1, 96, 96] float32 npy.

    Args:
        mp4_path:  Input LRS2 160x160 face-cropped video.
        out_npy:   Destination .npy path.
        extractor: A MouthROIExtractor instance (dlib or haar backend).

    Returns True on success.
    """
    try:
        result = extractor.extract_from_file(str(mp4_path))
        # result is torch.Tensor or numpy array [T, 1, 96, 96] float32
        if hasattr(result, "numpy"):
            arr = result.numpy()
        else:
            arr = result
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, arr)
        return True
    except Exception as e:
        print(f"  [mouth_roi] failed on {mp4_path}: {e}")
        return False


def _build_extractor(args):
    """Instantiate MouthROIExtractor based on CLI args."""
    sys.path.insert(0, str(ROOT))
    from avsd_ger.frontend.mouth_roi import MouthROIExtractor

    backend = args.roi_backend
    if backend == "dlib":
        fp = args.face_predictor or _DEFAULT_FACE_PREDICTOR
        cd = args.cnn_detector   or _DEFAULT_CNN_DETECTOR
        mf = args.mean_face      or _DEFAULT_MEAN_FACE
        print(f"[mouth_roi] backend=dlib")
        print(f"  face_predictor : {fp}")
        print(f"  cnn_detector   : {cd}")
        print(f"  mean_face      : {mf}")
        return MouthROIExtractor(
            backend="dlib",
            face_predictor_path=fp,
            cnn_detector_path=cd,
            mean_face_path=mf,
        )
    else:
        print(f"[mouth_roi] backend=haar (fallback)")
        return MouthROIExtractor(backend="haar")


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
    # Mouth ROI backend
    p.add_argument("--roi-backend", default="dlib", choices=["dlib", "haar"],
                   help="Mouth ROI extraction backend. "
                        "'dlib' (default): landmark-based, identical to av_hubert pipeline. "
                        "'haar': fixed center-crop fallback, no model files needed.")
    p.add_argument("--face-predictor", default=None,
                   help=f"Path to shape_predictor_68_face_landmarks.dat "
                        f"(default: {_DEFAULT_FACE_PREDICTOR})")
    p.add_argument("--cnn-detector", default=None,
                   help=f"Path to mmod_human_face_detector.dat "
                        f"(default: {_DEFAULT_CNN_DETECTOR})")
    p.add_argument("--mean-face", default=None,
                   help=f"Path to 20words_mean_face.npy "
                        f"(default: {_DEFAULT_MEAN_FACE})")
    args = p.parse_args()

    if not args.lrs2_root.exists():
        print(f"LRS2 root not found: {args.lrs2_root}", file=sys.stderr)
        return 2

    extractor = _build_extractor(args)

    speakers: dict[str, list] = {}
    for mp4, txt, spk in _iter_lrs2_utts(args.lrs2_root):
        if len(speakers) >= args.max_speakers and spk not in speakers:
            continue
        speakers.setdefault(spk, []).append((mp4, txt))
        n_total = sum(len(v) for v in speakers.values())
        if n_total >= args.max_utts and len(speakers) >= 1:
            break

    print(f"\n[LRS2] picked {sum(len(v) for v in speakers.values())} utts "
          f"across {len(speakers)} speakers")

    speakers_block = []
    utterances_block = []
    turns_block = []
    t_offset = 0.0

    for i, (spk, items) in enumerate(speakers.items()):
        speakers_block.append({
            "speaker_id": f"spk_{i+1:02d}",
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
            ok_roi = _extract_mouth_roi(mp4, roi_out, extractor)
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
        print("No utterances extracted — check dlib install + model files.")
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
