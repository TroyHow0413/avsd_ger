"""Create clean->degraded AMI session manifests.

The script preserves evaluation truth (`ref_text`, `ref_speaker`) and perturbs
frontend-side signals only:

* diarization/active-speaker label flip: replace a turn's mouth ROI with a
  same-meeting donor ROI from a different speaker.
* face-track dropout: remove the mouth ROI for a turn.
* visual blur: write blurred `.npy` mouth ROI copies under the output folder.
* ASD flip: flip entries in `speaker_mask_v`.

Use visual manifests (for example `data/ami_dev_visual/*.json`) if you want
dropout/blur/ASD noise to affect AV runs. Plain `data/ami_dev/manifests/*.json`
mostly have `mouth_roi: null`, so visual degradation cannot affect them.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


PRESETS = {
    "mild": {
        "label_flip_p": 0.1,
        "face_dropout_r": 0.10,
        "blur_sigma": 1.0,
        "asd_flip_q": 0.1,
    },
    "severe": {
        "label_flip_p": 0.3,
        "face_dropout_r": 0.50,
        "blur_sigma": 5.0,
        "asd_flip_q": 0.3,
    },
}


def _manifest_paths(src: Path) -> list[Path]:
    if src.is_dir():
        return sorted(src.glob("*.json"))
    return [src]


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "turns" not in data:
        raise ValueError(f"Expected a session JSON with a 'turns' list: {path}")
    return data


def _speaker(row: dict[str, Any]) -> str:
    return str(row.get("ref_speaker") or row.get("speaker_id") or "")


def _mouth_roi(row: dict[str, Any]) -> str | None:
    val = row.get("mouth_roi") or row.get("video") or row.get("video_path")
    return str(val) if val else None


def _exists(path: str | None) -> bool:
    return bool(path) and Path(path).exists()


def _load_roi(path: str) -> np.ndarray:
    return np.load(path)


def _blur_frame(frame: np.ndarray, sigma: float) -> np.ndarray:
    import cv2

    k = max(3, int(round(sigma * 6)) | 1)
    return cv2.GaussianBlur(frame, (k, k), sigmaX=sigma, sigmaY=sigma)


def _blur_roi(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr

    out = arr.copy()
    if arr.ndim == 3:
        for t in range(arr.shape[0]):
            out[t] = _blur_frame(arr[t], sigma)
    elif arr.ndim == 4 and arr.shape[1] in (1, 3):
        # [T, C, H, W]
        for t in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                out[t, c] = _blur_frame(arr[t, c], sigma)
    elif arr.ndim == 4 and arr.shape[-1] in (1, 3):
        # [T, H, W, C]
        for t in range(arr.shape[0]):
            out[t] = _blur_frame(arr[t], sigma)
    else:
        raise ValueError(f"Unsupported mouth ROI shape for blur: {arr.shape}")
    return out.astype(arr.dtype, copy=False)


def _roi_len(path: str | None) -> int:
    if not _exists(path):
        return 0
    try:
        return int(_load_roi(str(path)).shape[0])
    except Exception:
        return 0


def _mask_for_turn(row: dict[str, Any], roi_path: str | None) -> list[bool] | None:
    raw = row.get("speaker_mask_v")
    if isinstance(raw, list) and raw:
        return [bool(x) for x in raw]
    n = _roi_len(roi_path)
    if n <= 0:
        return None
    return [True] * n


def _flip_mask(mask: list[bool], q: float, rng: random.Random) -> tuple[list[bool], int]:
    out = []
    flips = 0
    for val in mask:
        if rng.random() < q:
            out.append(not val)
            flips += 1
        else:
            out.append(val)
    return out, flips


def _candidate_donors(turns: list[dict[str, Any]], row: dict[str, Any]) -> list[dict[str, Any]]:
    spk = _speaker(row)
    return [
        t for t in turns
        if _speaker(t) and _speaker(t) != spk and _exists(_mouth_roi(t))
    ]


def _write_blurred_roi(
    src_path: str,
    *,
    out_dir: Path,
    manifest_stem: str,
    turn_id: str,
    sigma: float,
) -> str:
    arr = _load_roi(src_path)
    blurred = _blur_roi(arr, sigma)
    dst_dir = out_dir / "_degraded_roi" / manifest_stem
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{turn_id}_{Path(src_path).stem}_blur{sigma:g}.npy"
    np.save(dst, blurred)
    return str(dst)


def degrade_manifest(
    manifest: dict[str, Any],
    *,
    manifest_stem: str,
    out_dir: Path,
    label_flip_p: float,
    face_dropout_r: float,
    blur_sigma: float,
    asd_flip_q: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed)
    degraded = copy.deepcopy(manifest)
    turns = degraded.get("turns", [])

    counts = {
        "turns": len(turns),
        "diarization_label_flips": 0,
        "face_track_dropouts": 0,
        "blurred_rois": 0,
        "asd_mask_bits_flipped": 0,
        "visual_missing_or_skipped": 0,
    }

    for i, row in enumerate(turns):
        turn_id = str(row.get("turn_id") or row.get("utt_id") or f"turn{i:06d}")
        original_roi = _mouth_roi(row)
        roi_for_turn = original_roi
        ops: dict[str, Any] = {
            "original_mouth_roi": original_roi,
            "label_flip": False,
            "face_track_dropout": False,
            "blur_sigma": None,
            "asd_flip_q": asd_flip_q,
        }

        if _exists(roi_for_turn) and rng.random() < label_flip_p:
            donors = _candidate_donors(turns, row)
            if donors:
                donor = rng.choice(donors)
                roi_for_turn = _mouth_roi(donor)
                row["frontend_speaker"] = _speaker(donor)
                row["frontend_speaker_source_turn"] = donor.get("turn_id") or donor.get("utt_id")
                ops["label_flip"] = True
                ops["donor_mouth_roi"] = roi_for_turn
                ops["donor_speaker"] = _speaker(donor)
                counts["diarization_label_flips"] += 1

        if _exists(roi_for_turn) and rng.random() < face_dropout_r:
            row["mouth_roi"] = None
            row["speaker_mask_v"] = None
            row["lip_conf_v"] = None
            ops["face_track_dropout"] = True
            counts["face_track_dropouts"] += 1
        elif _exists(roi_for_turn):
            if blur_sigma > 0:
                try:
                    row["mouth_roi"] = _write_blurred_roi(
                        str(roi_for_turn),
                        out_dir=out_dir,
                        manifest_stem=manifest_stem,
                        turn_id=turn_id.replace(".", "_"),
                        sigma=blur_sigma,
                    )
                    ops["blur_sigma"] = blur_sigma
                    counts["blurred_rois"] += 1
                except Exception as exc:
                    row["mouth_roi"] = roi_for_turn
                    ops["blur_error"] = f"{type(exc).__name__}: {exc}"
            else:
                row["mouth_roi"] = roi_for_turn

            mask = _mask_for_turn(row, row.get("mouth_roi"))
            if mask is not None:
                flipped, n_flips = _flip_mask(mask, asd_flip_q, rng)
                row["speaker_mask_v"] = flipped
                counts["asd_mask_bits_flipped"] += n_flips

            if "lip_conf" in row and "lip_conf_v" not in row:
                row["lip_conf_v"] = row["lip_conf"]
        else:
            counts["visual_missing_or_skipped"] += 1

        row["degradation"] = ops

    meta = {
        "degradation": {
            "label_flip_p": label_flip_p,
            "face_track_dropout_r": face_dropout_r,
            "visual_blur_sigma": blur_sigma,
            "asd_flip_q": asd_flip_q,
            "seed": seed,
            "truth_preserved": ["ref_text", "ref_speaker"],
            "counts": counts,
        }
    }
    degraded.setdefault("meta", {})
    degraded["meta"].update(meta)
    return degraded, meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path, help="Input manifest JSON or directory.")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--preset", choices=sorted(PRESETS), default=None)
    p.add_argument("--label-flip-p", type=float, default=None)
    p.add_argument("--face-dropout-r", type=float, default=None)
    p.add_argument("--blur-sigma", type=float, default=None)
    p.add_argument("--asd-flip-q", type=float, default=None)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    vals = dict(PRESETS.get(args.preset or "", {}))
    for key, attr in [
        ("label_flip_p", "label_flip_p"),
        ("face_dropout_r", "face_dropout_r"),
        ("blur_sigma", "blur_sigma"),
        ("asd_flip_q", "asd_flip_q"),
    ]:
        override = getattr(args, attr)
        if override is not None:
            vals[key] = override
    missing = [k for k in PRESETS["mild"] if k not in vals]
    if missing:
        raise ValueError(f"Missing degradation values: {missing}; use --preset or explicit args.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for idx, path in enumerate(_manifest_paths(args.src)):
        manifest = _load_manifest(path)
        degraded, meta = degrade_manifest(
            manifest,
            manifest_stem=path.stem,
            out_dir=args.out_dir,
            seed=args.seed + idx,
            **vals,
        )
        out_path = args.out_dir / path.name
        out_path.write_text(json.dumps(degraded, indent=2), encoding="utf-8")
        summaries.append({"input": str(path), "output": str(out_path), **meta["degradation"]})
        print(f"[wrote] {out_path}")

    summary_path = args.out_dir / "_degradation_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[wrote] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
