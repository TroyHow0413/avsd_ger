"""Build the preregistered scoring gate from completed eval JSON, no inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avsd_ger.eval.statistics import (  # noqa: E402
    build_paired_comparisons,
    build_statistics_report,
)
from avsd_ger.eval.standard_metrics import (  # noqa: E402
    compute_jiwer_metrics,
    compute_meeteval_metrics,
    compute_pyannote_diarization_metrics,
)
from scripts.rescore_standard_metrics import _turns  # noqa: E402


def _load_runs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("runs"), list):
        return payload["runs"]
    if "manifest" in payload and isinstance(payload.get("results"), list):
        return [payload]
    raise ValueError(
        f"{path} is neither a multi-manifest summary nor a single eval report"
    )


def _resolve_debug_path(summary_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    candidates = [candidate]
    if not candidate.is_absolute():
        # Multi-run summaries live at <repo>/out/<run>/summary.json while
        # debug_path is normally stored relative to <repo>.
        candidates.extend([
            summary_path.parent / candidate,
            summary_path.parent.parent.parent / candidate,
        ])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not resolve debug sidecar {raw_path!r} from {summary_path}"
    )


def _rescore_runs(
    runs: list[dict[str, Any]], *, summary_path: Path, language: str,
) -> None:
    for run in runs:
        for result in run.get("results", []):
            debug_path = result.get("debug_path")
            if not debug_path:
                raise ValueError(
                    f"{summary_path}: {result.get('ablation')} has no debug_path"
                )
            resolved = _resolve_debug_path(summary_path, str(debug_path))
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            turns = _turns(payload)
            if not turns:
                raise ValueError(f"{resolved} contains no turn records")
            result["standard_metrics"] = {
                "jiwer": compute_jiwer_metrics(turns, language=language),
                "meeteval": compute_meeteval_metrics(
                    turns,
                    language=language,
                    score_names=("cpwer", "tcpwer_collar_5s"),
                ),
                "pyannote": compute_pyannote_diarization_metrics(turns),
            }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute session-cluster confidence intervals and paired topology/"
            "identity gates from an existing eval summary; does not load a model."
        )
    )
    parser.add_argument(
        "--input", required=True, type=Path, nargs="+",
        help="One or more completed summary/single-report JSON files.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--no-rescore", action="store_true",
        help="Trust public metrics embedded in summaries instead of rescoring debug sidecars.",
    )
    args = parser.parse_args()

    runs: list[dict[str, Any]] = []
    for path in args.input:
        loaded = _load_runs(path)
        if not args.no_rescore:
            _rescore_runs(loaded, summary_path=path, language=args.language)
        runs.extend(loaded)
    statistics = build_statistics_report(
        runs, samples=args.samples, seed=args.seed,
    )
    comparisons = build_paired_comparisons(
        runs, samples=args.samples, seed=args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "statistics.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (args.out_dir / "paired_comparisons.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    completed = 0
    for name, comparison in comparisons["comparisons"].items():
        gate = comparison.get("selection_gate") or comparison.get("causal_gate")
        primary = comparison.get("metrics", {}).get("tcpwer_5s", {}).get(
            "session_cluster", {}
        )
        if gate is None or primary.get("status") != "ok":
            continue
        completed += 1
        print(
            f"[gate:{name}] decision={gate['decision']} "
            f"delta={primary['delta_challenger_minus_reference']:.6f} "
            f"ci95={primary['ci95']} n_sessions={primary['n_pairs']}"
        )
    if completed == 0:
        print(
            "[incomplete] no complete C3 or identity paired tcpWER@5s gate "
            "was found in the supplied summaries",
            file=sys.stderr,
        )
        return 2
    print(f"[wrote] {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
