"""Add public-library metrics to existing eval debug sidecars, without inference.

This is intentionally independent of model checkpoints and media.  The debug
JSON already stores the normalized-scoring essentials for every turn, so old
evaluations can be rescored after adding/upgrading JiWER, MeetEval, or
pyannote.metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avsd_ger.eval.session import SessionTurnResult  # noqa: E402
from avsd_ger.eval.standard_metrics import compute_standard_metrics  # noqa: E402


def _turns(payload: dict) -> list[SessionTurnResult]:
    turns: list[SessionTurnResult] = []
    for index, row in enumerate(payload.get("turns", [])):
        summary = row.get("summary", {}) or {}
        asr = row.get("asr", {}) or {}
        c1 = row.get("c1_effective", {}) or {}
        hyp_speaker = summary.get("hyp_speaker")
        if "hyp_speaker" not in summary:
            top_ids = c1.get("top_ids", []) or []
            if top_ids and not bool(c1.get("is_unknown", False)):
                hyp_speaker = top_ids[0]
        turns.append(SessionTurnResult(
            turn_id=str(summary.get("turn_id", f"t{index:06d}")),
            start=float(summary.get("start", 0.0)),
            end=float(summary.get("end", summary.get("start", 0.0))),
            hyp_text=str(summary.get("final_text") or ""),
            hyp_speaker=hyp_speaker,
            confidence=float(summary.get("confidence") or 0.0),
            s_acoustic=summary.get("s_acoustic"),
            iterations=int(summary.get("iterations") or 0),
            pool_updated=bool(summary.get("pool_updated", False)),
            asr_language=(
                asr.get("detected_language")
                or summary.get("asr_language")
            ),
            ref_text=summary.get("ref_text"),
            ref_speaker=summary.get("ref_speaker"),
        ))
    return turns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs", nargs="+",
        help="Debug JSON file(s), directories, or glob patterns.",
    )
    parser.add_argument(
        "--language", default="auto",
        help="Metric language passed to the project's frozen normalizer.",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Add standard_metrics to each source debug JSON atomically.",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Write rescored JSON copies here (required without --in-place).",
    )
    args = parser.parse_args()
    if not args.in_place and not args.out_dir:
        parser.error("provide --in-place or --out-dir")

    paths: set[Path] = set()
    for spec in args.inputs:
        candidate = Path(spec)
        if candidate.is_dir():
            paths.update(candidate.rglob("*.debug.json"))
        elif candidate.is_file():
            paths.add(candidate)
        else:
            paths.update(Path().glob(spec))
    if not paths:
        raise FileNotFoundError("No debug JSON files matched")

    out_dir = Path(args.out_dir) if args.out_dir else None
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        turns = _turns(payload)
        if not turns:
            print(f"[skip] {path}: no turn records")
            continue
        payload["standard_metrics"] = compute_standard_metrics(
            turns, language=args.language
        )
        if args.in_place:
            destination = path
        else:
            destination = out_dir / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(destination)
        statuses = {
            name: result.get("status")
            for name, result in payload["standard_metrics"].items()
        }
        print(f"[wrote] {destination} {statuses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
