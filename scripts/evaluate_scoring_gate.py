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
) -> dict[tuple[str, str], dict[str, Any]]:
    debug_payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        meeting = Path(str(run.get("manifest", ""))).stem
        for result in run.get("results", []):
            debug_path = result.get("debug_path")
            if not debug_path:
                raise ValueError(
                    f"{summary_path}: {result.get('ablation')} has no debug_path"
                )
            resolved = _resolve_debug_path(summary_path, str(debug_path))
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            key = (meeting, str(result.get("ablation")))
            if key in debug_payloads:
                raise ValueError(f"Duplicate debug payload for {key[0]}/{key[1]}")
            debug_payloads[key] = payload
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
    return debug_payloads


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _control_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Fields that must be invariant before a pure z_id intervention."""
    summary = row.get("summary", {}) or {}
    turn = row.get("turn", {}) or {}
    asr = row.get("asr", {}) or {}
    visual = row.get("visual", {}) or {}
    c1 = row.get("c1_effective", {}) or {}
    return {
        "immutable_turn": {
            "turn_id": summary.get("turn_id"),
            "start": summary.get("start"), "end": summary.get("end"),
            "ref_text": summary.get("ref_text"),
            "ref_speaker": summary.get("ref_speaker"),
            "audio_path": turn.get("audio_path"),
            "mouth_roi_path": turn.get("mouth_roi_path"),
        },
        "asr": {
            "asr_top": summary.get("asr_top"),
            "nbest": asr.get("nbest"), "nbest_scores": asr.get("nbest_scores"),
            "detected_language": asr.get("detected_language"),
        },
        "visual": {
            "lip_hyp": visual.get("lip_hyp"),
            "has_visual": summary.get("has_visual"),
            "lip_conf_mean": summary.get("lip_conf_mean"),
        },
        "frontend_features": {
            "asr_encoder_features": asr.get("encoder_features"),
            "asr_token_features": asr.get("token_features"),
            "vsr_features": visual.get("vsr_features"),
            "embeddings": row.get("embeddings", {}),
        },
        "c1_pre_intervention": {
            "top_ids": c1.get("top_ids"), "top_scores": c1.get("top_scores"),
            "logged_top_ids": c1.get("logged_top_ids"),
            "logged_top_scores": c1.get("logged_top_scores"),
            "is_unknown": c1.get("is_unknown"),
            "av_consistency_raw": c1.get("av_consistency_raw"),
            "z_id": c1.get("z_id"),
        },
    }


def validate_identity_controls(
    debug_payloads: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless identity rows differ only after conditioning z_id."""
    output: dict[str, Any] = {
        "protocol": "identity_normal is the paired reference; turn/input/ASR/visual/C1 must match",
        "comparisons": {},
    }
    for challenger in ("zero_z_id", "shuffled_z_id"):
        reference_meetings = {
            meeting for meeting, ablation in debug_payloads
            if ablation == "identity_normal"
        }
        challenger_meetings = {
            meeting for meeting, ablation in debug_payloads
            if ablation == challenger
        }
        meetings = sorted(reference_meetings | challenger_meetings)
        if not reference_meetings and not challenger_meetings:
            output["comparisons"][challenger] = {
                "status": "not_run", "n_meetings": 0, "n_paired_turns": 0,
                "failure_counts": {}, "examples": [],
            }
            continue
        counts = {
            "missing_payload": 0, "turn_set": 0, "immutable_turn": 0,
            "asr": 0, "visual": 0, "frontend_features": 0,
            "c1_pre_intervention": 0,
        }
        examples: list[dict[str, Any]] = []
        n_turns = 0
        for meeting in meetings:
            reference = debug_payloads.get((meeting, "identity_normal"))
            candidate = debug_payloads.get((meeting, challenger))
            if reference is None or candidate is None:
                counts["missing_payload"] += 1
                examples.append({"meeting_id": meeting, "category": "missing_payload"})
                continue
            ref_rows = {
                str((row.get("summary", {}) or {}).get("turn_id")): row
                for row in reference.get("turns", [])
            }
            cand_rows = {
                str((row.get("summary", {}) or {}).get("turn_id")): row
                for row in candidate.get("turns", [])
            }
            if set(ref_rows) != set(cand_rows):
                counts["turn_set"] += 1
                examples.append({
                    "meeting_id": meeting, "category": "turn_set",
                    "missing_from_reference": sorted(set(cand_rows) - set(ref_rows))[:10],
                    "missing_from_challenger": sorted(set(ref_rows) - set(cand_rows))[:10],
                })
            for turn_id in sorted(set(ref_rows) & set(cand_rows)):
                n_turns += 1
                ref_fields = _control_fields(ref_rows[turn_id])
                cand_fields = _control_fields(cand_rows[turn_id])
                for category in (
                    "immutable_turn", "asr", "visual", "frontend_features",
                    "c1_pre_intervention",
                ):
                    if _canonical(ref_fields[category]) != _canonical(cand_fields[category]):
                        counts[category] += 1
                        if len(examples) < 25:
                            examples.append({
                                "meeting_id": meeting, "turn_id": turn_id,
                                "category": category,
                                "reference": ref_fields[category],
                                "challenger": cand_fields[category],
                            })
        failures = sum(counts.values())
        output["comparisons"][challenger] = {
            "status": "pass" if failures == 0 else "control_failed",
            "n_meetings": len(meetings), "n_paired_turns": n_turns,
            "failure_counts": counts, "examples": examples,
        }
    return output


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
    debug_payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for path in args.input:
        loaded = _load_runs(path)
        if not args.no_rescore:
            rescored = _rescore_runs(loaded, summary_path=path, language=args.language)
            overlap = set(debug_payloads) & set(rescored)
            if overlap:
                raise ValueError(f"Duplicate meeting/ablation inputs: {sorted(overlap)}")
            debug_payloads.update(rescored)
        runs.extend(loaded)
    statistics = build_statistics_report(
        runs, samples=args.samples, seed=args.seed,
    )
    comparisons = build_paired_comparisons(
        runs, samples=args.samples, seed=args.seed,
    )
    controls = validate_identity_controls(debug_payloads) if debug_payloads else {
        "status": "not_checked", "reason": "--no-rescore omitted debug sidecars",
    }
    comparisons["identity_control_invariants"] = controls
    for comparison_name, challenger in (
        ("identity_zero", "zero_z_id"),
        ("identity_shuffle", "shuffled_z_id"),
    ):
        control = controls.get("comparisons", {}).get(challenger, {})
        comparison = comparisons.get("comparisons", {}).get(comparison_name, {})
        gate = comparison.get("causal_gate")
        if gate is not None and control.get("status") != "pass":
            gate["statistical_decision_before_control_audit"] = gate.get("decision")
            gate["decision"] = (
                "control_failed" if control.get("status") == "control_failed"
                else "control_not_checked"
            )
            gate["control_failure_counts"] = control.get("failure_counts", {})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "statistics.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (args.out_dir / "paired_comparisons.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (args.out_dir / "identity_control_invariants.json").write_text(
        json.dumps(controls, indent=2, ensure_ascii=False), encoding="utf-8",
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
