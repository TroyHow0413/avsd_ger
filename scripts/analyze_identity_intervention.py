"""Audit identity interventions from eval_ablations debug sidecars.

This is intentionally offline: it reads the debug JSON files already emitted by
``scripts/eval_ablations.py`` and does not load ASR, VSR, or GER checkpoints.
It distinguishes changes in the raw GER generation from changes in the final
post-gate output, which is important when the C3 fallback returns ASR 1-best.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


BASELINE = "identity_normal"
INTERVENTIONS = ("zero_z_id", "shuffled_z_id")


def _series_id(meeting_id: str) -> str:
    match = re.fullmatch(r"(.+?)[a-d]", meeting_id)
    return match.group(1) if match else meeting_id


def _last_trace(turn: dict[str, Any]) -> dict[str, Any]:
    trace = turn.get("trace") or []
    return dict(trace[-1]) if trace else {}


def _turn_id(turn: dict[str, Any]) -> str:
    return str((turn.get("summary") or {}).get("turn_id") or "")


def _value(turn: dict[str, Any], name: str) -> Any:
    summary = turn.get("summary") or {}
    trace = _last_trace(turn)
    if name == "raw_ger_text":
        return trace.get("raw_ger_text") or trace.get("cleaned_ger_text_before_gate")
    if name == "raw_generation":
        return trace.get("raw_generation_before_gate") or trace.get("raw_generation")
    if name == "final_text":
        return summary.get("final_text")
    if name == "hyp_speaker":
        return summary.get("hyp_speaker")
    if name == "final_source":
        return trace.get("final_source")
    if name == "fallback_applied":
        return bool(summary.get("fallback_applied"))
    if name == "c1_top_ids":
        return tuple((turn.get("c1_effective") or {}).get("top_ids") or [])
    if name == "f_align_stats":
        align = trace.get("alignment") or {}
        stats = align.get("f_align") or {}
        return tuple(stats.get(key) for key in ("shape", "min", "max", "mean", "std", "norm"))
    raise KeyError(name)


def _load_runs(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    runs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    paths = sorted((root / "debug").glob("*/*.debug.json"))
    if not paths:
        paths = sorted(root.glob("*_debug/*.debug.json"))
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ablation = str(payload.get("ablation") or path.parent.name)
        manifest = Path(str(payload.get("manifest") or path.stem)).stem
        if manifest.endswith(".debug"):
            manifest = manifest[: -len(".debug")]
        turns = payload.get("turns") or []
        indexed = {_turn_id(turn): turn for turn in turns if _turn_id(turn)}
        if not indexed:
            raise ValueError(f"No turn IDs found in {path}")
        runs[ablation][manifest] = indexed
    missing = [name for name in (BASELINE, *INTERVENTIONS) if name not in runs]
    if missing:
        raise FileNotFoundError(
            f"Missing debug sidecars for {missing} under {root}; expected debug/<ablation>/*.debug.json"
        )
    return dict(runs)


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _integrity(turns: list[dict[str, Any]], intervention: str, tolerance: float) -> dict[str, Any]:
    mode_counts: dict[str, int] = defaultdict(int)
    eligible = 0
    zero_norm_violations = 0
    derangement_fixed_points = 0
    shuffled_source_violations = 0
    for turn in turns:
        c1 = turn.get("c1_effective") or {}
        mode = str(c1.get("identity_conditioning_mode") or "<missing>")
        mode_counts[mode] += 1
        is_eligible = bool(c1.get("identity_causal_eligible"))
        eligible += int(is_eligible)
        norm = ((c1.get("conditioning_z_id") or {}).get("norm"))
        if intervention == "zero_z_id":
            if norm is None or not math.isfinite(float(norm)) or abs(float(norm)) > tolerance:
                zero_norm_violations += 1
        else:
            mapping = c1.get("identity_derangement_map") or {}
            derangement_fixed_points += sum(key == value for key, value in mapping.items())
            if is_eligible:
                source = c1.get("identity_conditioning_source_id")
                predicted = ((c1.get("top_ids") or [None])[0])
                if mode != "shuffled" or source is None or source == predicted:
                    shuffled_source_violations += 1
    if intervention == "zero_z_id":
        passed = zero_norm_violations == 0 and set(mode_counts) == {"zero"}
    else:
        allowed_modes = {"shuffled", "shuffle_ineligible_unknown"}
        passed = (
            not (set(mode_counts) - allowed_modes)
            and derangement_fixed_points == 0
            and shuffled_source_violations == 0
        )
    return {
        "status": "pass" if passed else "fail",
        "n_turns": len(turns),
        "n_causal_eligible": eligible,
        "mode_counts": dict(sorted(mode_counts.items())),
        "zero_norm_violations": zero_norm_violations,
        "derangement_fixed_points": derangement_fixed_points,
        "shuffled_source_violations": shuffled_source_violations,
    }


def analyze(root: Path, tolerance: float = 1e-8) -> dict[str, Any]:
    runs = _load_runs(root)
    baseline = runs[BASELINE]
    comparisons: dict[str, Any] = {}
    fields = (
        "raw_ger_text",
        "raw_generation",
        "final_text",
        "hyp_speaker",
        "final_source",
        "fallback_applied",
        "c1_top_ids",
        "f_align_stats",
    )

    for intervention in INTERVENTIONS:
        challenger = runs[intervention]
        meeting_ids = sorted(set(baseline) & set(challenger))
        totals = {field: 0 for field in fields}
        common_turns = 0
        missing_baseline = 0
        missing_intervention = 0
        intervention_turns: list[dict[str, Any]] = []
        by_series_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"n_turns": 0, **{field: 0 for field in fields}}
        )
        by_meeting: dict[str, Any] = {}

        for meeting in meeting_ids:
            normal_turns = baseline[meeting]
            changed_turns = challenger[meeting]
            common_ids = sorted(set(normal_turns) & set(changed_turns))
            missing_baseline += len(set(changed_turns) - set(normal_turns))
            missing_intervention += len(set(normal_turns) - set(changed_turns))
            meeting_counts = {field: 0 for field in fields}
            series_counts = by_series_counts[_series_id(meeting)]
            for turn_id in common_ids:
                normal = normal_turns[turn_id]
                changed = changed_turns[turn_id]
                intervention_turns.append(changed)
                common_turns += 1
                series_counts["n_turns"] += 1
                for field in fields:
                    differs = _value(normal, field) != _value(changed, field)
                    totals[field] += int(differs)
                    meeting_counts[field] += int(differs)
                    series_counts[field] += int(differs)
            by_meeting[meeting] = {
                "n_turns": len(common_ids),
                "changed_counts": meeting_counts,
                "changed_rates": {
                    field: _rate(count, len(common_ids))
                    for field, count in meeting_counts.items()
                },
            }

        comparisons[intervention] = {
            "n_meetings": len(meeting_ids),
            "n_common_turns": common_turns,
            "missing_from_baseline": missing_baseline,
            "missing_from_intervention": missing_intervention,
            "integrity": _integrity(intervention_turns, intervention, tolerance),
            "changed_counts": totals,
            "changed_rates": {
                field: _rate(count, common_turns) for field, count in totals.items()
            },
            "by_series": {
                series: {
                    "n_turns": counts["n_turns"],
                    "changed_counts": {field: counts[field] for field in fields},
                    "changed_rates": {
                        field: _rate(counts[field], counts["n_turns"])
                        for field in fields
                    },
                }
                for series, counts in sorted(by_series_counts.items())
            },
            "by_meeting": by_meeting,
        }

    integrity_pass = all(
        item["integrity"]["status"] == "pass" for item in comparisons.values()
    )
    return {
        "schema_version": 1,
        "artifact_root": str(root.resolve()),
        "baseline": BASELINE,
        "interventions": list(INTERVENTIONS),
        "note": (
            "hyp_speaker is expected to remain stable because zero/shuffle is a pure "
            "conditioning intervention and intentionally preserves the C1 retrieval decision"
        ),
        "integrity_status": "pass" if integrity_pass else "fail",
        "comparisons": comparisons,
    }


def _print_summary(report: dict[str, Any]) -> None:
    print(f"[identity-audit] integrity={report['integrity_status']}")
    for name, comparison in report["comparisons"].items():
        rates = comparison["changed_rates"]
        print(
            f"[{name}] meetings={comparison['n_meetings']} turns={comparison['n_common_turns']} "
            f"integrity={comparison['integrity']['status']} "
            f"raw_ger_changed={rates['raw_ger_text']:.2%} "
            f"final_text_changed={rates['final_text']:.2%} "
            f"speaker_changed={rates['hyp_speaker']:.2%} "
            f"fallback_changed={rates['fallback_applied']:.2%}"
        )
        for series, values in comparison["by_series"].items():
            series_rates = values["changed_rates"]
            print(
                f"  {series}: turns={values['n_turns']} "
                f"raw={series_rates['raw_ger_text']:.2%} "
                f"final={series_rates['final_text']:.2%}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 when intervention integrity checks fail.",
    )
    args = parser.parse_args()
    report = analyze(args.artifact_root, tolerance=args.tolerance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_summary(report)
    print(f"[wrote] {args.out}")
    return 2 if args.strict and report["integrity_status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
