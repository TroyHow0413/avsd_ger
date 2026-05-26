"""Quick manifest visual-field sanity check.

This is a read-only helper for verifying whether manifests contain usable
mouth ROI / speaker-mask / lip-confidence fields before running AV experiments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _paths(spec: Path) -> list[Path]:
    if spec.is_dir():
        return sorted(spec.glob("*.json")) + sorted(spec.glob("*.jsonl"))
    return [spec]


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return list(data.get("turns", data.get("utterances", [])))


def _roi(row: dict[str, Any]) -> str | None:
    val = row.get("mouth_roi") or row.get("video") or row.get("video_path")
    return str(val) if val else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path, help="Manifest JSON/JSONL file or directory.")
    p.add_argument("--show-missing", type=int, default=5)
    args = p.parse_args()

    total = with_roi = roi_exists = with_mask = with_lip = 0
    missing_examples: list[str] = []
    per_file = []

    for path in _paths(args.manifest):
        rows = _rows(path)
        f_total = len(rows)
        f_with_roi = f_roi_exists = f_mask = f_lip = 0
        for row in rows:
            total += 1
            roi = _roi(row)
            if roi:
                with_roi += 1
                f_with_roi += 1
                if Path(roi).exists():
                    roi_exists += 1
                    f_roi_exists += 1
                elif len(missing_examples) < args.show_missing:
                    missing_examples.append(roi)
            elif len(missing_examples) < args.show_missing:
                turn_id = row.get("turn_id") or row.get("utt_id") or "<unknown>"
                missing_examples.append(f"{path.name}:{turn_id}: mouth_roi=null")

            if row.get("speaker_mask_v") is not None:
                with_mask += 1
                f_mask += 1
            if row.get("lip_conf_v") is not None or row.get("lip_conf") is not None:
                with_lip += 1
                f_lip += 1

        per_file.append((path.name, f_total, f_with_roi, f_roi_exists, f_mask, f_lip))

    def pct(n: int) -> str:
        return f"{(100.0 * n / total):.1f}%" if total else "0.0%"

    print(f"manifest: {args.manifest}")
    print(f"files: {len(per_file)}")
    print(f"turns: {total}")
    print(f"mouth_roi non-null: {with_roi}/{total} ({pct(with_roi)})")
    print(f"mouth_roi file exists: {roi_exists}/{total} ({pct(roi_exists)})")
    print(f"speaker_mask_v present: {with_mask}/{total} ({pct(with_mask)})")
    print(f"lip_conf/lip_conf_v present: {with_lip}/{total} ({pct(with_lip)})")
    print("")
    print("per file: name,total,roi_non_null,roi_exists,speaker_mask,lip_conf")
    for item in per_file[:50]:
        print(",".join(str(x) for x in item))
    if len(per_file) > 50:
        print(f"... {len(per_file) - 50} more files")

    if missing_examples:
        print("")
        print("missing examples:")
        for x in missing_examples:
            print(f"- {x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
