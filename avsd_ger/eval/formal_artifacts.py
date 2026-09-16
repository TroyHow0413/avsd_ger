"""Write self-contained, reproducible evaluation artifacts.

The legacy evaluator JSON remains the compatibility surface.  This module
adds an analysis-first directory whose turn records are sufficient to rerun
all text, speaker, calibration, grouping, and meeting scorers offline.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
from typing import Any, Iterable

import numpy as np

from .metrics import MetricsReport, evaluate_session
from .session import SessionTurnResult
from .standard_metrics import compute_sklearn_metrics, compute_standard_metrics
from .statistics import build_paired_comparisons, build_statistics_report
from ..text_normalization import NORMALIZER_VERSION, normalize_text


SCHEMA_VERSION = "avsd-ger-eval-record-v1"
VISUAL_CATEGORIES = [
    "real_visual", "low_quality_visual", "missing_mouth_roi",
    "source_video_excluded", "audio_only_by_ablation",
]
CANONICAL_ABLATIONS = {
    "full_model": "full_model",
    "wo_c1": "wo_c1",
    "wo_c2": "wo_c2",
    "wo_c3": "wo_c3",
    "c3_wo_conf_gates": "c3_wo_confidence_gates",
    "c3_wo_confidence_gates": "c3_wo_confidence_gates",
}

GROUP_BINS = {
    "by_snr": [
        ("lt_0_db", None, 0.0), ("0_to_5_db", 0.0, 5.0),
        ("5_to_10_db", 5.0, 10.0), ("10_to_20_db", 10.0, 20.0),
        ("ge_20_db", 20.0, None),
    ],
    "by_lip_conf": [
        ("0_to_0p25", 0.0, 0.25), ("0p25_to_0p50", 0.25, 0.50),
        ("0p50_to_0p75", 0.50, 0.75), ("0p75_to_1p00", 0.75, 1.000001),
    ],
    "by_turn_length": [
        ("1_to_5_words", 1.0, 6.0), ("6_to_10_words", 6.0, 11.0),
        ("11_to_20_words", 11.0, 21.0), ("21_to_40_words", 21.0, 41.0),
        ("gt_40_words", 41.0, None),
    ],
    "by_duration": [
        ("lt_2s", None, 2.0), ("2_to_5s", 2.0, 5.0),
        ("5_to_10s", 5.0, 10.0), ("10_to_20s", 10.0, 20.0),
        ("ge_20s", 20.0, None),
    ],
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    source = Path(path)
    if not source.exists():
        return None
    digest = hashlib.sha256()
    if source.is_file():
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    else:
        for child in sorted(p for p in source.rglob("*") if p.is_file()):
            digest.update(child.relative_to(source).as_posix().encode("utf-8"))
            with child.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _git_snapshot(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None
    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for package in ("jiwer", "meeteval", "scikit-learn", "pyannote.metrics", "torch"):
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = None
    return out


def records_schema() -> dict[str, Any]:
    required = [
        "schema_version", "utt_id", "meeting_id", "ablation", "speaker_ref",
        "speaker_hyp", "speaker_hyp_top5", "c1_similarity_top5", "ref_text",
        "asr_hyp", "ger_hyp_raw", "ger_hyp_clean", "ger_hyp_final",
        "fallback", "fallback_reason", "ger_confidence", "c1_similarity",
        "lip_conf_mean", "snr_estimate_db", "snr_score_mean",
        "turn_length_words", "start_time", "end_time", "duration_s",
        "wall_time_ms", "gpu_memory_allocated_mb", "gpu_peak_mb",
        "visual_availability", "identity_conditioning_mode",
        "identity_conditioning_source_id", "identity_derangement_map",
        "identity_causal_eligible",
    ]
    properties = {key: {} for key in required}
    properties.update({
        "schema_version": {"const": SCHEMA_VERSION},
        "utt_id": {"type": "string"},
        "meeting_id": {"type": "string"},
        "ablation": {"type": "string"},
        "speaker_hyp_top5": {"type": "array", "maxItems": 5},
        "c1_similarity_top5": {"type": "array", "maxItems": 5},
        "fallback": {"type": "boolean"},
        "start_time": {"type": "number"},
        "end_time": {"type": "number"},
        "duration_s": {"type": "number", "minimum": 0},
        "gpu_peak_mb": {
            "type": ["number", "null"],
            "description": (
                "Meeting-ablation run-level CUDA peak allocated memory, "
                "repeated on every turn for self-contained records."
            ),
        },
        "gpu_memory_allocated_mb": {
            "type": ["number", "null"],
            "description": "Instantaneous CUDA allocated memory sampled after this turn.",
        },
        "identity_conditioning_mode": {
            "enum": ["normal", "zero", "shuffled", "shuffle_ineligible_unknown"],
        },
        "identity_conditioning_source_id": {"type": ["string", "null"]},
        "identity_derangement_map": {
            "type": "object", "additionalProperties": {"type": "string"},
        },
        "identity_causal_eligible": {"type": "boolean"},
    })
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_VERSION,
        "title": "AVSD-GER per-turn evaluation record",
        "type": "object",
        "required": required,
        "properties": properties,
        "additionalProperties": True,
    }


def _last_trace(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("trace") or []
    return trace[-1] if trace else {}


def _visual_availability(row: dict[str, Any], ablation: str) -> str:
    summary = row.get("summary", {}) or {}
    turn = row.get("turn", {}) or {}
    manifest = turn.get("manifest_row", {}) or {}
    if manifest.get("official_visual_exclusion") or manifest.get("source_duration_exclusion"):
        return "source_video_excluded"
    input_debug = row.get("input", {}) or {}
    if "has_visual_flag" in input_debug:
        visual_available = bool(input_debug.get("has_visual_flag"))
    else:
        # Backward-compatible fallback for debug files written before the
        # explicit input availability flag was retained.
        visual_available = bool(manifest.get("mouth_roi") or manifest.get("video"))
    pipeline_used_visual = bool(summary.get("has_visual"))
    effective_mode = _last_trace(row).get("ger_mode")
    if visual_available and not pipeline_used_visual and effective_mode == "audio_only":
        return "audio_only_by_ablation"
    if not visual_available or not pipeline_used_visual:
        return "missing_mouth_roi"
    lip = summary.get("lip_conf_mean")
    if lip is not None and float(lip) < 0.5:
        return "low_quality_visual"
    return "real_visual"


def _record(row: dict[str, Any], meeting_id: str, ablation: str) -> dict[str, Any]:
    summary = row.get("summary", {}) or {}
    last = _last_trace(row)
    asr = summary.get("asr_top") or last.get("asr_top") or ""
    ger_raw = last.get("raw_generation_before_gate")
    ger_clean = last.get("cleaned_ger_text_before_gate")
    ger_final = summary.get("final_text") or last.get("text") or ""
    top_ids = last.get("logged_top_ids") or summary.get("speaker_hyp_top5") or summary.get("top_ids") or []
    top_scores = last.get("logged_top_scores") or summary.get("c1_similarity_top5") or summary.get("top_scores") or []
    components = last.get("components") or {}
    ref_text = summary.get("ref_text") or ""
    start = float(summary.get("start", 0.0))
    end = float(summary.get("end", start))
    return {
        "schema_version": SCHEMA_VERSION,
        "utt_id": str(summary.get("turn_id", "")),
        "meeting_id": meeting_id,
        "ablation": ablation,
        "speaker_ref": summary.get("ref_speaker"),
        "speaker_is_unknown_ref": (row.get("turn", {}) or {}).get("manifest_row", {}).get("speaker_is_unknown_ref"),
        "speaker_hyp": summary.get("hyp_speaker"),
        "speaker_hyp_top5": list(top_ids)[:5],
        "c1_similarity_top5": [float(v) for v in list(top_scores)[:5]],
        "ref_text": ref_text,
        "asr_hyp": asr,
        "ger_hyp_raw": ger_raw,
        "ger_hyp_clean": ger_clean,
        "ger_hyp_final": ger_final,
        "fallback": bool(summary.get("fallback_applied", False)),
        "fallback_reason": summary.get("fallback_reason"),
        "final_source": last.get("final_source"),
        "ger_confidence": summary.get("confidence"),
        "acoustic_confidence": last.get("s_acoustic_conf"),
        "c1_similarity": last.get("av_consistency_raw", summary.get("av_consistency_raw")),
        "c1_confidence": components.get("id") if isinstance(components, dict) else None,
        "lip_conf_mean": summary.get("lip_conf_mean"),
        "snr_estimate_db": summary.get("snr_estimate_db_mean"),
        "snr_score_mean": summary.get("snr_score_mean"),
        "turn_length_words": len(str(ref_text).split()),
        "start_time": start,
        "end_time": end,
        "duration_s": max(0.0, end - start),
        "wall_time_ms": summary.get("wall_time_ms"),
        "gpu_memory_allocated_mb": summary.get("gpu_memory_allocated_mb"),
        "gpu_peak_mb": None,
        "has_visual": bool(summary.get("has_visual")),
        "visual_availability": _visual_availability(row, ablation),
        "pool_updated": bool(summary.get("pool_updated", False)),
        "iterations": summary.get("iterations"),
        "identity_conditioning_mode": last.get(
            "identity_conditioning_mode", "normal",
        ),
        "identity_conditioning_source_id": last.get(
            "identity_conditioning_source_id",
        ),
        "identity_derangement_map": last.get("identity_derangement_map") or {},
        "identity_causal_eligible": bool(
            last.get("identity_causal_eligible", False)
        ),
    }


def _as_turns(records: list[dict[str, Any]], hypothesis_key: str = "ger_hyp_final") -> list[SessionTurnResult]:
    offsets: dict[str, float] = {}
    next_offset = 0.0
    for meeting in sorted({r["meeting_id"] for r in records}):
        offsets[meeting] = next_offset
        duration = max((float(r["end_time"]) for r in records if r["meeting_id"] == meeting), default=0.0)
        next_offset += duration + 10.0
    turns: list[SessionTurnResult] = []
    for row in records:
        meeting = row["meeting_id"]
        offset = offsets[meeting]
        ref_speaker = row.get("speaker_ref")
        hyp_speaker = row.get("speaker_hyp")
        turns.append(SessionTurnResult(
            turn_id=f"{meeting}:{row['utt_id']}",
            start=offset + float(row["start_time"]),
            end=offset + float(row["end_time"]),
            hyp_text=str(row.get(hypothesis_key) or ""),
            hyp_speaker=f"{meeting}:{hyp_speaker}" if hyp_speaker is not None else None,
            confidence=float(row.get("ger_confidence") or 0.0),
            s_acoustic=row.get("acoustic_confidence"),
            iterations=int(row.get("iterations") or 0),
            pool_updated=bool(row.get("pool_updated")),
            ref_text=str(row.get("ref_text") or ""),
            ref_speaker=f"{meeting}:{ref_speaker}" if ref_speaker is not None else None,
        ))
    return turns


def _sum_fields(payloads: list[dict[str, Any]], fields: Iterable[str]) -> dict[str, float | int]:
    totals: dict[str, float | int] = {}
    for field in fields:
        values = [payload.get(field) for payload in payloads]
        numeric = [value for value in values if isinstance(value, (int, float))]
        if numeric:
            total = sum(numeric)
            totals[field] = int(total) if all(isinstance(value, int) for value in numeric) else float(total)
    return totals


def _aggregate_task_reports(
    reports: list[tuple[str, MetricsReport]],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Micro-aggregate meeting-local task metrics from additive components."""
    sa_rows = [report.details.get("sa_wer", {}) for _, report in reports]
    scr_rows = [report.details.get("scr", {}) for _, report in reports]
    sid_rows = [report.details.get("av_sid", {}) for _, report in reports]
    der_rows = [report.details.get("der", {}) for _, report in reports]

    sa = _sum_fields(
        sa_rows,
        ("n_ref_words", "n_sub", "n_del", "n_ins", "n_spk_err",
         "legacy_raw_n_ref_words"),
    )
    ref_words = int(sa.get("n_ref_words", 0))
    text_errors = sum(int(sa.get(key, 0)) for key in ("n_sub", "n_del", "n_ins"))
    speaker_errors = int(sa.get("n_spk_err", 0))
    sa["wer"] = text_errors / ref_words if ref_words else 0.0
    sa_wer = (text_errors + speaker_errors) / ref_words if ref_words else 0.0
    raw_ref_words = int(sa.get("legacy_raw_n_ref_words", 0))
    raw_weighted_errors = sum(
        float(row.get("legacy_raw_wer", 0.0))
        * int(row.get("legacy_raw_n_ref_words", 0))
        for row in sa_rows
    )
    sa["legacy_raw_wer"] = raw_weighted_errors / raw_ref_words if raw_ref_words else 0.0
    sa["mapping_by_meeting"] = {
        meeting: report.details.get("sa_wer", {}).get("mapping", {})
        for meeting, report in reports
    }

    scr = _sum_fields(scr_rows, ("n_matched", "n_spk_err"))
    n_matched = int(scr.get("n_matched", 0))
    scr_value = int(scr.get("n_spk_err", 0)) / n_matched if n_matched else 0.0
    scr["mapping_by_meeting"] = {
        meeting: report.details.get("scr", {}).get("mapping", {})
        for meeting, report in reports
    }

    sid = _sum_fields(sid_rows, ("n", "n_correct"))
    sid_n = int(sid.get("n", 0))
    sid_value = int(sid.get("n_correct", 0)) / sid_n if sid_n else 0.0
    sid["mapping_by_meeting"] = {
        meeting: report.details.get("av_sid", {}).get("mapping", {})
        for meeting, report in reports
    }

    der = _sum_fields(
        der_rows,
        ("total_ref", "total_hyp", "miss", "false_alarm", "confusion", "correct"),
    )
    total_ref = float(der.get("total_ref", 0.0))
    der_value = sum(
        float(der.get(key, 0.0)) for key in ("miss", "false_alarm", "confusion")
    ) / total_ref if total_ref else 0.0
    der["mapping_by_meeting"] = {
        meeting: report.details.get("der", {}).get("mapping", {})
        for meeting, report in reports
    }

    jer_per_speaker: dict[str, float] = {}
    jer_mapping: dict[str, Any] = {}
    for meeting, report in reports:
        detail = report.details.get("jer", {})
        jer_mapping[meeting] = detail.get("mapping", {})
        for speaker, value in detail.get("per_speaker", {}).items():
            jer_per_speaker[f"{meeting}:{speaker}"] = float(value)
    jer_value = (
        sum(jer_per_speaker.values()) / len(jer_per_speaker)
        if jer_per_speaker else 0.0
    )

    return ({
        "sa_wer": sa_wer,
        "wer": float(sa["wer"]),
        "scr": scr_value,
        "av_sid_acc": sid_value,
        "der": der_value,
        "jer": jer_value,
    }, {
        "sa_wer": sa,
        "scr": scr,
        "av_sid": sid,
        "der": der,
        "jer": {"per_speaker": jer_per_speaker, "mapping_by_meeting": jer_mapping},
    })


def _aggregate_jiwer(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    available = [payload for payload in payloads if payload.get("status") == "ok"]
    if not available:
        return payloads[0] if payloads else {"status": "unavailable", "library": "jiwer"}
    totals = _sum_fields(available, (
        "hits", "substitutions", "deletions", "insertions",
        "reference_words", "hypothesis_words", "char_hits",
        "char_substitutions", "char_deletions", "char_insertions",
        "reference_characters",
    ))
    hits = int(totals.get("hits", 0))
    sub = int(totals.get("substitutions", 0))
    delete = int(totals.get("deletions", 0))
    insert = int(totals.get("insertions", 0))
    ref = int(totals.get("reference_words", hits + sub + delete))
    hyp = int(totals.get("hypothesis_words", hits + sub + insert))
    errors = sub + delete + insert
    char_errors = sum(int(totals.get(key, 0)) for key in (
        "char_substitutions", "char_deletions", "char_insertions",
    ))
    ref_chars = int(totals.get("reference_characters", 0))
    wip = (hits / ref) * (hits / hyp) if ref and hyp else 0.0
    return {
        "status": "ok" if len(available) == len(payloads) else "partial",
        "n_meetings_expected": len(payloads),
        "n_meetings_scored": len(available),
        "library": available[0].get("library", "jiwer"),
        "library_version": available[0].get("library_version"),
        "normalizer_version": available[0].get("normalizer_version", NORMALIZER_VERSION),
        "wer": errors / ref if ref else 0.0,
        "mer": errors / (hits + errors) if hits + errors else 0.0,
        "wil": 1.0 - wip,
        "wip": wip,
        "cer": char_errors / ref_chars if ref_chars else 0.0,
        **totals,
        "aggregation": "micro from meeting-local additive counts",
    }


def _aggregate_meeteval(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    available = [
        payload for payload in payloads
        if payload.get("status") in {"ok", "partial"}
    ]
    if not available:
        return payloads[0] if payloads else {"status": "unavailable", "library": "meeteval"}
    score_names = sorted({
        name for payload in available for name in payload.get("scores", {})
    })
    scores: dict[str, Any] = {}
    for name in score_names:
        rows = [
            payload["scores"][name] for payload in available
            if name in payload.get("scores", {})
        ]
        numeric_keys = sorted({
            key for row in rows for key, value in row.items()
            if key != "error_rate" and isinstance(value, (int, float))
        })
        aggregate = _sum_fields(rows, numeric_keys)
        errors = aggregate.get("errors")
        length = aggregate.get("length")
        if isinstance(errors, (int, float)) and isinstance(length, (int, float)):
            aggregate["error_rate"] = float(errors) / float(length) if length else 0.0
        else:
            weights = [float(row.get("length", 1.0)) for row in rows]
            denominator = sum(weights)
            aggregate["error_rate"] = (
                sum(float(row.get("error_rate", 0.0)) * weight
                    for row, weight in zip(rows, weights)) / denominator
                if denominator else 0.0
            )
        scores[name] = aggregate
    failures = {
        f"meeting_{index}": payload.get("failures", {})
        for index, payload in enumerate(available)
        if payload.get("failures")
    }
    diagnostics = [payload.get("diagnostics", {}) for payload in available]
    self_overlap_seconds = sum(
        float(item.get("hypothesis_self_overlap_seconds", 0.0) or 0.0)
        for item in diagnostics
    )
    overlap_meetings = sum(
        int(float(item.get("hypothesis_self_overlap_seconds", 0.0) or 0.0) > 0)
        for item in diagnostics
    )
    complete = len(available) == len(payloads) and not failures
    return {
        "status": "ok" if complete else "partial",
        "n_meetings_expected": len(payloads),
        "n_meetings_scored": len(available),
        "library": available[0].get("library", "meeteval"),
        "library_version": available[0].get("library_version"),
        "normalizer_version": available[0].get("normalizer_version", NORMALIZER_VERSION),
        "protocol": available[0].get("protocol", {}),
        "diagnostics": {
            "hypothesis_self_overlap_seconds": self_overlap_seconds,
            "meetings_with_hypothesis_self_overlap": overlap_meetings,
            "n_meetings_checked": len(diagnostics),
            "status": "warning" if self_overlap_seconds > 0 else "ok",
        },
        "scores": scores,
        "failures": failures,
        "aggregation": "micro from meeting-local error counts",
    }


def _aggregate_pyannote(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    available = [payload for payload in payloads if payload.get("status") == "ok"]
    if not available:
        return payloads[0] if payloads else {
            "status": "unavailable", "library": "pyannote.metrics",
        }
    score_names = sorted({
        name for payload in available for name in payload.get("scores", {})
    })
    scores: dict[str, Any] = {}
    for name in score_names:
        rows = [payload["scores"][name] for payload in available]
        components = [row.get("components", {}) for row in rows]
        keys = sorted({
            key for component in components for key, value in component.items()
            if isinstance(value, (int, float))
        })
        total = _sum_fields(components, keys)
        if name.startswith("der_") and float(total.get("total", 0.0)):
            value = sum(float(total.get(key, 0.0)) for key in (
                "false alarm", "missed detection", "confusion",
            )) / float(total["total"])
        elif name.startswith("jer_") and float(total.get("speaker count", 0.0)):
            value = float(total.get("speaker error", 0.0)) / float(total["speaker count"])
        else:
            value = sum(float(row.get("value", 0.0)) for row in rows) / len(rows)
        scores[name] = {"value": value, "components": total}
    return {
        "status": "ok" if len(available) == len(payloads) else "partial",
        "n_meetings_expected": len(payloads),
        "n_meetings_scored": len(available),
        "library": available[0].get("library", "pyannote.metrics"),
        "library_version": available[0].get("library_version"),
        "protocol": available[0].get("protocol", {}),
        "scores": scores,
        "aggregation": "meeting-local additive components",
    }


def _score_records(records: list[dict[str, Any]], language: str) -> dict[str, Any]:
    records_by_meeting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        records_by_meeting[str(row["meeting_id"])].append(row)

    meeting_reports: list[tuple[str, MetricsReport]] = []
    meeting_standard: list[dict[str, Any]] = []
    per_meeting: list[dict[str, Any]] = []
    all_turns: list[SessionTurnResult] = []
    for meeting in sorted(records_by_meeting):
        turns = _as_turns(records_by_meeting[meeting])
        all_turns.extend(turns)
        report = evaluate_session(turns, language=language)
        standard_payload = compute_standard_metrics(turns, language=language)
        meeting_reports.append((meeting, report))
        meeting_standard.append(standard_payload)
        per_meeting.append({
            "meeting_id": meeting,
            "counts": {
                "n_turns": len(records_by_meeting[meeting]),
                "n_ref_words": report.n_ref_words,
            },
            "task_metrics": {
                "sa_wer": report.sa_wer, "wer": report.wer,
                "scr": report.scr, "av_sid_acc": report.av_sid_acc,
                "der": report.der, "jer": report.jer,
            },
            "task_metric_details": report.details,
            "standard_metrics": standard_payload,
        })

    task_metrics, task_details = _aggregate_task_reports(meeting_reports)
    standard = {
        "jiwer": _aggregate_jiwer([
            payload.get("jiwer", {}) for payload in meeting_standard
        ]),
        "meeteval": _aggregate_meeteval([
            payload.get("meeteval", {}) for payload in meeting_standard
        ]),
        # Confidence calibration and meeting-prefixed speaker classification
        # are inexpensive and are intentionally evaluated on all turns.
        "sklearn": compute_sklearn_metrics(all_turns, language=language),
        "pyannote": _aggregate_pyannote([
            payload.get("pyannote", {}) for payload in meeting_standard
        ]),
    }
    return {
        "counts": {
            "n_turns": len(records),
            "n_ref_words": sum(report.n_ref_words for _, report in meeting_reports),
            "n_meetings": len(records_by_meeting),
        },
        "task_metrics": task_metrics,
        "task_metric_details": task_details,
        "standard_metrics": standard,
        "per_meeting": per_meeting,
    }


def _jiwer_error(text_ref: str, text_hyp: str, language: str) -> int:
    ref = normalize_text(text_ref, language=language)
    hyp = normalize_text(text_hyp, language=language)
    try:
        import jiwer
        out = jiwer.process_words(ref, hyp)
        return int(out.substitutions + out.deletions + out.insertions)
    except Exception:
        return int(ref != hyp)


def _correction_metrics(records: list[dict[str, Any]], language: str) -> dict[str, Any]:
    improved = harmed = unchanged = eligible = 0
    for row in records:
        before = _jiwer_error(row["ref_text"], row["asr_hyp"], language)
        after = _jiwer_error(row["ref_text"], row["ger_hyp_final"], language)
        if before > 0:
            eligible += 1
        if after < before:
            improved += 1
        elif after > before:
            harmed += 1
        else:
            unchanged += 1
    precision = improved / max(1, improved + harmed)
    recall = improved / max(1, eligible)
    return {
        "definition": "turn-level edit-count improvement; project-specific",
        "improved_turns": improved, "harmed_turns": harmed,
        "unchanged_turns": unchanged, "eligible_asr_error_turns": eligible,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "overcorrection_rate": harmed / max(1, len(records)),
    }


def _topk_sid(records: list[dict[str, Any]], mappings: dict[str, dict[str, str]]) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    labeled = [r for r in records if r.get("speaker_ref") is not None]
    for k in (1, 3, 5):
        hits = 0
        available = 0
        for row in labeled:
            candidates = row.get("speaker_hyp_top5") or []
            if candidates:
                available += 1
            mapping = mappings.get(row["meeting_id"], {})
            mapped = [mapping.get(value, value) for value in candidates[:k]]
            hits += int(row["speaker_ref"] in mapped)
        scores[f"top_{k}_accuracy"] = hits / max(1, len(labeled))
        scores[f"top_{k}_coverage"] = available / max(1, len(labeled))
    scores["n_labeled_turns"] = len(labeled)
    scores["note"] = "Top-k uses the meeting-level Hungarian mapping; log_top_k=5 does not change top_k=3 decisions."
    open_set_rows = [r for r in records if r.get("speaker_is_unknown_ref") is not None]
    labels = [int(bool(r["speaker_is_unknown_ref"])) for r in open_set_rows]
    unknown_scores = [1.0 - float(r.get("c1_similarity") or 0.0) for r in open_set_rows]
    if open_set_rows and len(set(labels)) == 2:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
            fpr, tpr, _ = roc_curve(labels, unknown_scores)
            fnr = 1.0 - tpr
            index = int(np.nanargmin(np.abs(fpr - fnr)))
            scores["unknown_detection"] = {
                "status": "ok", "score": "1-c1_similarity",
                "auroc": float(roc_auc_score(labels, unknown_scores)),
                "average_precision": float(average_precision_score(labels, unknown_scores)),
                "eer": float((fpr[index] + fnr[index]) / 2.0),
                "n_turns": len(open_set_rows),
            }
        except Exception as exc:
            scores["unknown_detection"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    else:
        scores["unknown_detection"] = {
            "status": "unavailable",
            "reason": "requires speaker_is_unknown_ref labels containing both known and unknown turns",
            "n_labeled_turns": len(open_set_rows),
        }
    return scores


def _binary_confidence_diagnostics(
    labels: list[int],
    probabilities: list[float],
    *,
    target: str,
    probability_definition: str,
) -> dict[str, Any]:
    """Calibration/selective-risk diagnostics with fixed, auditable rules."""
    if not labels:
        return {
            "status": "unavailable", "target": target,
            "probability_definition": probability_definition,
            "reason": "no labeled records",
        }
    probabilities = [min(1.0, max(0.0, float(value))) for value in probabilities]
    n_bins = 15
    ece = 0.0
    bins: list[dict[str, Any]] = []
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        members = [
            i for i, value in enumerate(probabilities)
            if lower <= value < upper or (index == n_bins - 1 and value == 1.0)
        ]
        if not members:
            bins.append({
                "lower_inclusive": lower, "upper_exclusive": upper,
                "n": 0, "accuracy": None, "mean_confidence": None,
            })
            continue
        accuracy = sum(labels[i] for i in members) / len(members)
        mean_confidence = sum(probabilities[i] for i in members) / len(members)
        ece += len(members) / len(labels) * abs(accuracy - mean_confidence)
        bins.append({
            "lower_inclusive": lower, "upper_exclusive": upper,
            "n": len(members), "accuracy": accuracy,
            "mean_confidence": mean_confidence,
        })

    order = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)
    cumulative_errors = 0
    risks: list[float] = []
    for rank, row_index in enumerate(order, start=1):
        cumulative_errors += 1 - labels[row_index]
        risks.append(cumulative_errors / rank)

    payload: dict[str, Any] = {
        "status": "ok", "target": target,
        "probability_definition": probability_definition,
        "n": len(labels), "n_positive": sum(labels),
        "expected_calibration_error_15_bins": ece,
        "area_under_risk_coverage_curve": sum(risks) / len(risks),
        "bins": bins,
    }
    try:
        from sklearn.metrics import (
            average_precision_score, brier_score_loss, log_loss, roc_auc_score,
        )
        payload.update({
            "library": "scikit-learn",
            "brier_score": float(brier_score_loss(labels, probabilities)),
            "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
            "auroc": (
                float(roc_auc_score(labels, probabilities))
                if len(set(labels)) == 2 else None
            ),
            "average_precision": (
                float(average_precision_score(labels, probabilities))
                if len(set(labels)) == 2 else None
            ),
        })
    except Exception as exc:
        payload["library_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def _c1_calibration(
    records: list[dict[str, Any]],
    mappings: dict[str, dict[str, str]],
) -> dict[str, Any]:
    labels: list[int] = []
    probabilities: list[float] = []
    for row in records:
        reference = row.get("speaker_ref")
        hypothesis = row.get("speaker_hyp")
        similarity = row.get("c1_similarity")
        if reference is None or similarity is None:
            continue
        mapped = mappings.get(row["meeting_id"], {}).get(hypothesis, hypothesis)
        labels.append(int(hypothesis is not None and mapped == reference))
        # av_consistency_raw is cosine similarity rather than a learned
        # probability. Clipping follows the model's existing [0,1]-threshold
        # interpretation and is recorded explicitly in scoring_protocol.json.
        probabilities.append(min(1.0, max(0.0, float(similarity))))
    return _binary_confidence_diagnostics(
        labels,
        probabilities,
        target="meeting-level Hungarian-mapped top-1 speaker correctness",
        probability_definition="clip(av_consistency_raw cosine similarity, 0, 1)",
    )


def _bucket(value: float | None, bins: list[tuple[str, float | None, float | None]]) -> str:
    if value is None:
        return "unavailable"
    for label, low, high in bins:
        if (low is None or value >= low) and (high is None or value < high):
            return label
    return "unavailable"


def _group_metrics(records_by_ablation: dict[str, list[dict[str, Any]]], language: str) -> dict[str, Any]:
    value_fields = {
        "by_snr": "snr_estimate_db", "by_lip_conf": "lip_conf_mean",
        "by_turn_length": "turn_length_words", "by_duration": "duration_s",
    }
    outputs: dict[str, Any] = {}
    for group_name, bins in GROUP_BINS.items():
        group_payload: dict[str, Any] = {"bins": [b[0] for b in bins], "ablations": {}}
        field = value_fields[group_name]
        for ablation, records in records_by_ablation.items():
            subsets: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in records:
                value = row.get(field)
                subsets[_bucket(float(value) if value is not None else None, bins)].append(row)
            labels = [item[0] for item in bins] + ["unavailable"]
            group_payload["ablations"][ablation] = {}
            for label in labels:
                rows = subsets.get(label, [])
                group_payload["ablations"][ablation][label] = (
                    _score_records(rows, language) if rows else {
                        "status": "empty",
                        "counts": {"n_turns": 0, "n_ref_words": 0, "n_meetings": 0},
                    }
                )
        outputs[group_name] = group_payload

    visual: dict[str, Any] = {"ablations": {}}
    for ablation, records in records_by_ablation.items():
        subsets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            subsets[row["visual_availability"]].append(row)
        visual["ablations"][ablation] = {}
        for label in VISUAL_CATEGORIES:
            rows = subsets.get(label, [])
            visual["ablations"][ablation][label] = (
                _score_records(rows, language) if rows else {
                    "status": "empty",
                    "counts": {"n_turns": 0, "n_ref_words": 0, "n_meetings": 0},
                }
            )
    outputs["by_visual_availability"] = visual
    return outputs


def _scoring_text(
    row: dict[str, Any], *, hypothesis: bool, language: str, normalized: bool,
) -> str:
    text = str(row.get("ger_hyp_final" if hypothesis else "ref_text") or "")
    if normalized:
        return normalize_text(text, language=language)
    return " ".join(text.split())


def _stm(
    records: list[dict[str, Any]], hypothesis: bool, *, language: str = "en",
    normalized: bool = True,
) -> str:
    lines: list[str] = []
    ordered = sorted(records, key=lambda row: (
        str(row["meeting_id"]), float(row["start_time"]),
        float(row["end_time"]), str(row["utt_id"]),
    ))
    for row in ordered:
        text = _scoring_text(
            row, hypothesis=hypothesis, language=language, normalized=normalized,
        )
        if not text:
            continue
        speaker = row.get("speaker_hyp" if hypothesis else "speaker_ref") or "__NONE__"
        safe = hashlib.sha1(str(speaker).encode()).hexdigest()[:12]
        lines.append(
            f"{row['meeting_id']} 1 spk_{safe} {row['start_time']:.6f} "
            f"{row['end_time']:.6f} {text}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _rttm(records: list[dict[str, Any]], hypothesis: bool) -> str:
    lines: list[str] = []
    ordered = sorted(records, key=lambda row: (
        str(row["meeting_id"]), float(row["start_time"]),
        float(row["end_time"]), str(row["utt_id"]),
    ))
    for row in ordered:
        speaker = row.get("speaker_hyp" if hypothesis else "speaker_ref")
        # Match compute_pyannote_diarization_metrics: a missing label means
        # missing speech attribution, not a real speaker named UNKNOWN.
        if speaker is None:
            continue
        lines.append(
            f"SPEAKER {row['meeting_id']} 1 {row['start_time']:.6f} "
            f"{row['duration_s']:.6f} <NA> <NA> {speaker} <NA> <NA>"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _seglst(
    records: list[dict[str, Any]], hypothesis: bool, *, language: str = "en",
    normalized: bool = True,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in sorted(records, key=lambda item: (
        str(item["meeting_id"]), float(item["start_time"]),
        float(item["end_time"]), str(item["utt_id"]),
    )):
        words = _scoring_text(
            row, hypothesis=hypothesis, language=language, normalized=normalized,
        )
        if not words:
            continue
        output.append({
            "session_id": row["meeting_id"],
            "speaker": row.get(
                "speaker_hyp" if hypothesis else "speaker_ref"
            ) or "__NONE__",
            "start_time": row["start_time"], "end_time": row["end_time"],
            "words": words,
        })
    return output


def _scoring_protocol(language: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "language": language,
        "text_normalizer": NORMALIZER_VERSION,
        "libraries": _package_versions(),
        "confirmatory_analysis": {
            "primary_metric": "tcpwer_5s",
            "primary_comparison": "wo_c3 - full_model",
            "test": "two-sided paired session-cluster bootstrap",
            "bootstrap_samples": 10000,
            "independent_unit": "session/manifest; never turn",
            "secondary_metrics": ["cpwer", "wer"],
            "exploratory_metrics": [
                "oracle_turn_der", "oracle_turn_jer", "sa_wer_custom",
                "scr", "av_sid_acc",
            ],
            "multiplicity": (
                "No multiplicity-adjusted claims for secondary/exploratory metrics; "
                "effect estimates and confidence intervals are reported."
            ),
        },
        "primary_public_metrics": {
            "text": "JiWER; MeetEval for meeting/permutation-aware WER",
            "diarization": "pyannote.metrics, oracle-turn segmentation, collars 0.0 and 0.25 s, overlap included",
            "classification_calibration": "scikit-learn",
        },
        "tcpwer_5s": {
            "scorer": "MeetEval",
            "collar_seconds": 5.0,
            "collar_application": (
                "MeetEval expands hypothesis pseudo-word temporal annotations; "
                "applying the tolerance to reference is equivalent"
            ),
            "reference_pseudo_word_timing": "character_based",
            "hypothesis_pseudo_word_timing": "character_based_points",
            "overlap_handling": "all input STM segments retained; no overlap exclusion",
            "empty_hypothesis": "all reference words score as deletions",
            "aggregation": "MeetEval combine_error_rates: sum(errors) / sum(length)",
            "cross_meeting_alignment": "forbidden; every meeting is scored independently",
        },
        "scoring_inputs": {
            "canonical_stm_and_seglst": (
                "reference.stm, hypothesis.stm and *.seglst.json contain "
                f"{NORMALIZER_VERSION}-normalized text"
            ),
            "raw_transcript_copies": "*.raw.stm and *.raw.seglst.json",
            "empty_text_segments": "omitted from STM and SegLST",
            "missing_hypothesis_speaker_rttm": (
                "omitted and scored as missed attribution; never converted to "
                "a literal UNKNOWN speaker"
            ),
        },
        "diarization_protocol": {
            "scorer": "pyannote.metrics",
            "segmentation": "oracle_turns",
            "collars_seconds": [0.0, 0.25],
            "skip_overlap": False,
            "unknown_hypothesis": "None is omitted and contributes missed speech; explicit UNKNOWN is scored as a speaker label",
            "interpretation": "speaker-attribution diagnostic, not end-to-end diarization segmentation performance",
        },
        "project_specific_metrics": {
            "sa_wer_scr": "legacy AVSD-GER speaker-attributed word alignment",
            "werr": "relative WER reduction against ASR 1-best",
            "ocr": "fraction of turns harmed relative to ASR 1-best",
            "correction_precision_recall_f1": "turn-level edit-count improvement/harm",
        },
        "group_bins": {
            key: [{"label": label, "lower_inclusive": low, "upper_exclusive": high}
                  for label, low, high in bins]
            for key, bins in GROUP_BINS.items()
        },
        "visual_availability_categories": VISUAL_CATEGORIES,
        "visual_availability_note": (
            "source_video_excluded can only be populated when excluded turns are present in the evaluation input; "
            "AMI v4 manifests that omit excluded turns retain an explicit empty category"
        ),
        "aggregation": (
            "score each meeting independently, then micro-aggregate additive "
            "error counts; speaker mappings and word alignments never cross "
            "meeting boundaries; per-meeting rows are retained"
        ),
        "efficiency": "CUDA synchronized per turn; peak memory reset once per ablation; RTF=wall_time/audio_duration",
        "memory_semantics": {
            "gpu_peak_mb": (
                "meeting-ablation run-level CUDA peak allocated memory; repeated "
                "on each turn to keep records self-contained"
            ),
            "gpu_memory_allocated_mb": (
                "instantaneous CUDA allocated memory sampled after that turn"
            ),
        },
        "c1_calibration": {
            "target": "meeting-level Hungarian-mapped top-1 speaker correctness",
            "probability": "clip(av_consistency_raw cosine similarity, 0, 1)",
            "warning": "diagnostic confidence proxy, not a learned calibrated probability",
        },
    }


def write_formal_artifacts(
    output_root: str | Path,
    raw_runs: list[dict[str, Any]],
    *,
    repo_root: str | Path,
    config_path: str,
    config: dict[str, Any],
    pool_path: str | None,
    aligner_ckpt: str | None,
    ger_ckpt: str | None,
    seed: int,
    started_at: str,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 1337,
) -> Path:
    """Materialize the formal artifact tree from in-memory evaluation rows."""
    root = Path(output_root)
    language = str(config.get("asr", {}).get("language") or "en")
    records_by_ablation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_meeting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mappings: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    profiles: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    ablation_meta: dict[str, dict[str, Any]] = {}

    for run in raw_runs:
        manifest_path = Path(run["manifest"])
        meeting = manifest_path.stem
        for result in run["results"]:
            legacy = result["ablation"]
            canonical = CANONICAL_ABLATIONS.get(legacy, legacy)
            ablation_meta[canonical] = {
                "id": canonical, "legacy_id": legacy,
                "display_name": canonical.replace("_", " "),
                "flags": result.get("flags", {}),
            }
            rows = [_record(row, meeting, canonical) for row in result.get("turn_debug", [])]
            for row in rows:
                row["gpu_peak_mb"] = result.get("profile", {}).get("gpu_peak_allocated_mb")
            records_by_ablation[canonical].extend(rows)
            per_meeting[canonical].append({
                "meeting_id": meeting,
                "manifest": str(manifest_path),
                "metrics": result.get("metrics", {}),
                "standard_metrics": result.get("standard_metrics", {}),
                "trace_summary": result.get("trace_summary", {}),
                "profile": result.get("profile", {}),
                "power": result.get("power"),
            })
            mappings[canonical][meeting] = (
                result.get("metric_details", {}).get("av_sid", {}).get("mapping", {})
            )
            profiles.append({"ablation": canonical, "meeting_id": meeting, **result.get("profile", {})})
            if result.get("power"):
                power = {"ablation": canonical, "meeting_id": meeting, **result["power"]}
                n_turns = max(1, int(result.get("metrics", {}).get("n_turns") or len(rows)))
                n_words = max(1, int(result.get("metrics", {}).get("n_ref_words") or 0))
                power["energy_j_per_turn"] = float(power.get("energy_j") or 0.0) / n_turns
                power["energy_j_per_ref_word"] = float(power.get("energy_j") or 0.0) / n_words
                power_rows.append(power)
            for row in rows:
                latency_rows.append({
                    "ablation": canonical, "meeting_id": meeting,
                    "utt_id": row["utt_id"], "duration_s": row["duration_s"],
                    "wall_time_ms": row["wall_time_ms"],
                    "gpu_memory_allocated_mb": row["gpu_memory_allocated_mb"],
                })
            _write_json(
                root / "debug" / canonical / f"{meeting}.debug.json",
                {"manifest": str(manifest_path), "ablation": canonical, "turns": result.get("turn_debug", [])},
            )

    _write_json(root / "records.schema.json", records_schema())
    for ablation, rows in records_by_ablation.items():
        _write_jsonl(root / "records" / f"{ablation}.jsonl", rows)

    all_reference = records_by_ablation.get("full_model") or next(iter(records_by_ablation.values()), [])
    reference_dir = root / "scoring_inputs" / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    (reference_dir / "reference.stm").write_text(
        _stm(all_reference, False, language=language), encoding="utf-8",
    )
    (reference_dir / "reference.raw.stm").write_text(
        _stm(all_reference, False, language=language, normalized=False),
        encoding="utf-8",
    )
    (reference_dir / "reference.rttm").write_text(_rttm(all_reference, False), encoding="utf-8")
    _write_json(
        reference_dir / "reference.seglst.json",
        _seglst(all_reference, False, language=language),
    )
    _write_json(
        reference_dir / "reference.raw.seglst.json",
        _seglst(all_reference, False, language=language, normalized=False),
    )

    per_ablation_metrics: dict[str, Any] = {}
    main_rows: list[dict[str, Any]] = []
    correction: dict[str, Any] = {}
    sid: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    public_scoring_complete = True
    for ablation, rows in records_by_ablation.items():
        scored = _score_records(rows, language)
        source_meeting_rows = {
            row["meeting_id"]: row for row in per_meeting[ablation]
        }
        rescored_meetings: list[dict[str, Any]] = []
        for meeting_score in scored["per_meeting"]:
            meeting_id = meeting_score["meeting_id"]
            source = source_meeting_rows.get(meeting_id, {})
            rescored_meetings.append({
                **source,
                "meeting_id": meeting_id,
                "metrics": meeting_score["task_metrics"],
                "metric_details": meeting_score["task_metric_details"],
                "standard_metrics": meeting_score["standard_metrics"],
                "counts": meeting_score["counts"],
            })
        _write_jsonl(
            root / "metrics" / "per_meeting" / f"{ablation}.jsonl",
            rescored_meetings,
        )
        correction[ablation] = _correction_metrics(rows, language)
        sid[ablation] = _topk_sid(rows, mappings[ablation])
        calibration[ablation] = {
            "ger": scored["standard_metrics"].get("sklearn", {}).get("confidence_quality", {}),
            "c1": _c1_calibration(rows, mappings[ablation]),
        }
        per_ablation_metrics[ablation] = {
            **scored,
            "correction": correction[ablation],
            "speaker_identification": sid[ablation],
        }
        _write_json(root / "metrics" / "per_ablation" / f"{ablation}.json", per_ablation_metrics[ablation])
        profile_rows = [p for p in profiles if p["ablation"] == ablation]
        total_wall = sum(float(p.get("wall_time_s") or 0) for p in profile_rows)
        total_audio = sum(float(p.get("audio_duration_s") or 0) for p in profile_rows)
        task = scored["task_metrics"]
        standard = scored["standard_metrics"]
        jiwer_wer = standard.get("jiwer", {}).get("wer")
        cpwer = standard.get("meeteval", {}).get("scores", {}).get("cpwer", {}).get("error_rate")
        tcpwer = (
            standard.get("meeteval", {}).get("scores", {})
            .get("tcpwer_collar_5s", {}).get("error_rate")
        )
        pyannote_scores = standard.get("pyannote", {}).get("scores", {})
        pyannote_der = pyannote_scores.get("der_collar_0s", {}).get("value")
        pyannote_jer = pyannote_scores.get("jer_collar_0s", {}).get("value")
        public_complete = all(
            standard.get(name, {}).get("status") == "ok"
            for name in ("jiwer", "meeteval", "pyannote")
        ) and all(value is not None for value in (jiwer_wer, cpwer, tcpwer, pyannote_der, pyannote_jer))
        public_scoring_complete = public_scoring_complete and public_complete
        asr_scored = _score_records([
            {**row, "ger_hyp_final": row["asr_hyp"]} for row in rows
        ], language)
        asr_wer = (
            asr_scored["standard_metrics"].get("jiwer", {}).get("wer")
            if asr_scored["standard_metrics"].get("jiwer", {}).get("wer") is not None
            else asr_scored["task_metrics"]["wer"]
        )
        primary_wer = jiwer_wer if jiwer_wer is not None else task["wer"]
        main_rows.append({
            "ablation": ablation, **scored["counts"],
            "wer": primary_wer,
            "cpwer": cpwer,
            "tcpwer_5s": tcpwer,
            "sa_wer": task["sa_wer"], "scr": task["scr"],
            "av_sid_acc": task["av_sid_acc"],
            # Standard columns never silently fall back to project metrics.
            "der": pyannote_der,
            "jer": pyannote_jer,
            "der_custom": task["der"],
            "jer_custom": task["jer"],
            "public_scoring_complete": public_complete,
            "asr_baseline_wer": asr_wer,
            "werr": ((asr_wer - primary_wer) / asr_wer) if asr_wer else None,
            "ocr": correction[ablation]["overcorrection_rate"],
            "rtf": total_wall / total_audio if total_audio else None,
            "metric_sources": {
                "wer": "jiwer" if jiwer_wer is not None else "project_fallback",
                "cpwer": "meeteval" if cpwer is not None else None,
                "tcpwer_5s": "meeteval" if tcpwer is not None else None,
                "sa_wer": "project_specific",
                "scr": "project_specific",
                "av_sid_acc": "project_specific_meeting_local_hungarian",
                "der": "pyannote.metrics" if pyannote_der is not None else None,
                "jer": "pyannote.metrics" if pyannote_jer is not None else None,
                "der_custom": "project_specific",
                "jer_custom": "project_specific",
            },
        })
        hyp_dir = root / "scoring_inputs" / "hypothesis" / ablation
        hyp_dir.mkdir(parents=True, exist_ok=True)
        (hyp_dir / "hypothesis.stm").write_text(
            _stm(rows, True, language=language), encoding="utf-8",
        )
        (hyp_dir / "hypothesis.raw.stm").write_text(
            _stm(rows, True, language=language, normalized=False),
            encoding="utf-8",
        )
        (hyp_dir / "hypothesis.rttm").write_text(_rttm(rows, True), encoding="utf-8")
        _write_json(
            hyp_dir / "hypothesis.seglst.json",
            _seglst(rows, True, language=language),
        )
        _write_json(
            hyp_dir / "hypothesis.raw.seglst.json",
            _seglst(rows, True, language=language, normalized=False),
        )

    _write_json(root / "metrics" / "main_table.json", {"rows": main_rows})
    _write_json(root / "metrics" / "appendix_sdi.json", {
        ablation: payload.get("standard_metrics", {}).get("jiwer", {})
        for ablation, payload in per_ablation_metrics.items()
    })
    _write_json(root / "metrics" / "appendix_correction.json", correction)
    _write_json(root / "metrics" / "appendix_sid.json", sid)
    _write_json(root / "metrics" / "appendix_calibration.json", calibration)
    scoring_protocol = _scoring_protocol(language)
    scoring_protocol["confirmatory_analysis"]["bootstrap_samples"] = int(bootstrap_samples)
    scoring_protocol["confirmatory_analysis"]["bootstrap_seed"] = int(bootstrap_seed)
    _write_json(root / "metrics" / "scoring_protocol.json", scoring_protocol)
    _write_json(
        root / "metrics" / "statistics.json",
        build_statistics_report(
            raw_runs, samples=bootstrap_samples, seed=bootstrap_seed,
        ),
    )
    _write_json(
        root / "metrics" / "paired_comparisons.json",
        build_paired_comparisons(
            raw_runs, samples=bootstrap_samples, seed=bootstrap_seed,
        ),
    )
    for name, payload in _group_metrics(records_by_ablation, language).items():
        _write_json(root / "metrics" / "groups" / f"{name}.json", payload)

    _write_json(root / "profiles" / "efficiency.json", {"per_meeting_ablation": profiles})
    _write_json(root / "profiles" / "power.json", {"per_meeting_ablation": power_rows})
    _write_jsonl(root / "profiles" / "latency_per_turn.jsonl", latency_rows)

    config_file = Path(config_path)
    git_snapshot = _git_snapshot(Path(repo_root))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if public_scoring_complete else "metrics_incomplete",
        "public_scoring_complete": public_scoring_complete,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "ablations": [ablation_meta[key] for key in records_by_ablation],
        "inputs": {
            "config": {"path": config_path, "sha256": _sha256(config_file)},
            "manifests": [{"path": run["manifest"], "sha256": _sha256(run["manifest"])} for run in raw_runs],
            "checkpoints": {
                "identity_pool": {"path": pool_path, "sha256": _sha256(pool_path)},
                "aligner": {"path": aligner_ckpt, "sha256": _sha256(aligner_ckpt)},
                "ger": {"path": ger_ckpt, "sha256": _sha256(ger_ckpt)},
            },
        },
        "config_path": config_path,
        "git_commit": git_snapshot.get("commit"),
        "git": git_snapshot,
        "seed": seed,
        "bootstrap": {"samples": int(bootstrap_samples), "seed": int(bootstrap_seed)},
        "identity_causal_protocol": {
            "normal": "original C1 z_id; retrieval metadata unchanged",
            "zero": "only C2/GER conditioning z_id is zeroed",
            "shuffled": (
                "only C2/GER conditioning z_id is replaced using the next "
                "sorted within-meeting speaker ID in a cyclic no-self mapping"
            ),
            "per_meeting_derangements": {
                ablation: {
                    meeting: next(
                        (
                            row["identity_derangement_map"]
                            for row in records_by_ablation[ablation]
                            if row["meeting_id"] == meeting
                            and row.get("identity_derangement_map")
                        ),
                        {},
                    )
                    for meeting in sorted({
                        row["meeting_id"] for row in records_by_ablation[ablation]
                    })
                }
                for ablation in records_by_ablation
                if ablation in {"identity_normal", "zero_z_id", "shuffled_z_id"}
            },
        },
        "effective_config_sha256": hashlib.sha256(
            json.dumps(config, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "effective_settings": {
            "vsr_emit_text": bool(config.get("vsr", {}).get("emit_text", False)),
            "enable_pool_update": bool(
                config.get("feedback", {}).get("enable_pool_update", False)
            ),
            "tau_update": config.get("feedback", {}).get("tau_update"),
            "ger_mode": config.get("ger", {}).get("mode"),
            "soft_token_count": config.get("ger", {}).get("bridge", {}).get("n_queries"),
        },
        "libraries": _package_versions(),
        "record_schema": "records.schema.json",
    }
    _write_json(root / "run_manifest.json", manifest)
    return root
