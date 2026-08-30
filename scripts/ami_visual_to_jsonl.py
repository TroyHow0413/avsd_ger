"""Convert AMI visual manifests into training JSONL records.

Input manifests are the per-meeting JSON files produced by
``prepare_ami_visual_manifest.py``.  The output is a flat JSONL file that can be
consumed by ``train_identity.py`` for Stage 1 and ``train_stage2.py`` for
Stage 2.

Example:
    python scripts/ami_visual_to_jsonl.py \
        --manifest-dir data/ami_train_visual \
        --out data/ami_stage_train.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _resolve_existing(path: str | None, *, base: Path) -> str | None:
    if not path:
        return None
    raw = Path(path)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(base / raw)
    if "\\" in path:
        normalized = Path(path.replace("\\", "/"))
        candidates.append(normalized)
        if not normalized.is_absolute():
            candidates.append(base / normalized)
    for candidate in candidates:
        if candidate.exists():
            try:
                return str(candidate.resolve().relative_to(base.resolve()))
            except ValueError:
                return str(candidate.resolve())
    return str(raw)


def _roi_frame_count(path: str | None) -> int:
    if not path:
        return 0
    try:
        return int(np.load(path, mmap_mode="r").shape[0])
    except Exception:
        return 0


def _speaker_suffix(speaker_id: str) -> str:
    return speaker_id.rsplit("_", 1)[-1]


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_records(
    manifest_paths: list[Path],
    *,
    root: Path,
    require_lip_confidence: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = _load_manifest(manifest_path)
        speakers = {
            str(spk.get("speaker_id")): spk
            for spk in manifest.get("speakers", [])
            if spk.get("speaker_id")
        }
        manifest_meta = manifest.get("meta", {})
        dataset_build_id = str(
            manifest_meta.get("dataset_build_id")
            or manifest_meta.get("version")
            or "unknown"
        )
        for i, turn in enumerate(manifest.get("turns", [])):
            speaker_id = str(turn.get("ref_speaker", ""))
            spk = speakers.get(speaker_id, {})
            wav_path = _resolve_existing(turn.get("audio"), base=root)
            video_path = _resolve_existing(turn.get("mouth_roi") or turn.get("video"), base=root)
            face_path = _resolve_existing(spk.get("enrollment_face"), base=root)
            if not wav_path or not video_path or not face_path:
                continue
            roi_for_count = Path(video_path)
            if not roi_for_count.is_absolute():
                roi_for_count = root / roi_for_count
            n_frames = _roi_frame_count(str(roi_for_count))
            lip_conf = turn.get("lip_conf_v")
            if lip_conf is None:
                if require_lip_confidence:
                    raise ValueError(
                        f"{manifest_path}:{turn.get('turn_id', i)} has no real lip_conf_v. "
                        "Regenerate the visual manifest or use "
                        "--allow-missing-lip-confidence for a non-C1 diagnostic export."
                    )
                lip_conf = None
            elif n_frames and len(lip_conf) != n_frames:
                raise ValueError(
                    f"{manifest_path}:{turn.get('turn_id', i)} lip_conf_v length "
                    f"{len(lip_conf)} != ROI frame count {n_frames}"
                )
            row = {
                "utt_id": str(turn.get("turn_id", f"{manifest_path.stem}.t{i:04d}")),
                "meeting_id": str(manifest.get("meta", {}).get("meeting_id", manifest_path.stem)),
                "dataset_build_id": dataset_build_id,
                "speaker_id": speaker_id,
                "speaker_suffix": turn.get("nxt_agent") or spk.get("nxt_agent") or _speaker_suffix(speaker_id),
                "participant_id": turn.get("participant_id") or spk.get("participant_id"),
                "session_speaker_id": turn.get("session_speaker_id") or spk.get("session_speaker_id"),
                "nxt_agent": turn.get("nxt_agent") or spk.get("nxt_agent"),
                "identity_scope": turn.get("identity_scope") or spk.get("identity_scope"),
                "wav_path": wav_path,
                "video_path": video_path,
                "face_path": face_path,
                "target": str(turn.get("ref_text", "")),
                "lip_conf": lip_conf,
                "lip_conf_source": turn.get("lip_conf_source", "missing" if lip_conf is None else "unspecified"),
                "start": float(turn.get("start", 0.0)),
                "end": float(turn.get("end", 0.0)),
                "audio_source_type": turn.get("audio_source_type"),
                "turn_boundary_source": turn.get("turn_boundary_source"),
            }
            rows.append(row)

    # Deterministic in-corpus negatives for Stage-2 InfoNCE. Keep this O(N):
    # the old all-pairs construction becomes impractical once AMI is uncapped.
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for idx, row in enumerate(rows):
        by_speaker.setdefault(row["speaker_id"], []).append(row)
    speaker_ids = sorted(by_speaker)
    speaker_position = {speaker_id: i for i, speaker_id in enumerate(speaker_ids)}
    for idx, row in enumerate(rows):
        if len(speaker_ids) < 2:
            continue
        other_id = speaker_ids[(speaker_position[row["speaker_id"]] + 1) % len(speaker_ids)]
        candidates = by_speaker[other_id]
        neg = candidates[idx % len(candidates)]
        row["neg_wav_path"] = neg["wav_path"]
        row["neg_face_path"] = neg["face_path"]
        row["neg_speaker_id"] = neg["speaker_id"]
    return rows


def _resolve_input_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for item in args.manifests or []:
        paths.append(Path(item))
    if args.manifest_dir is not None:
        paths.extend(sorted(Path(args.manifest_dir).glob("*.json")))
    paths = [p for p in paths if p.exists() and p.is_file()]
    if not paths:
        raise FileNotFoundError("No input manifest JSON files found.")
    return sorted(dict.fromkeys(paths))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-dir", type=Path, default=None)
    p.add_argument("--manifests", nargs="*", default=None)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--root", default=Path.cwd(), type=Path)
    p.add_argument(
        "--allow-missing-lip-confidence",
        action="store_true",
        help="Export missing confidence as null. Such rows are not suitable for C1 dual-gate training.",
    )
    args = p.parse_args()

    manifest_paths = _resolve_input_paths(args)
    rows = iter_records(
        manifest_paths,
        root=args.root,
        require_lip_confidence=not args.allow_missing_lip_confidence,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[wrote] {args.out} ({len(rows)} records from {len(manifest_paths)} manifests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
