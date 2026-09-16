"""Metrics delegated to public, citable evaluation libraries.

The project-specific metrics in :mod:`avsd_ger.eval.metrics` are retained for
backwards compatibility.  This module is the formal/appendix scoring path:

* JiWER supplies the single-stream word and character error measures.
* MeetEval supplies permutation- and time-constrained meeting WER measures.
* pyannote.metrics supplies oracle-turn diarization scores when it is installed.

Every function returns JSON-safe data and isolates optional-library failures so
an expensive model inference run is never lost because one scorer is missing.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
from decimal import Decimal
import hashlib
import math
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Sequence

from .session import SessionTurnResult
from ..text_normalization import NORMALIZER_VERSION, normalize_text


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalized_turns(
    turns: Sequence[SessionTurnResult], language: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for turn in turns:
        detected = getattr(turn, "asr_language", None)
        rows.append({
            "turn_id": turn.turn_id,
            "start_time": float(turn.start),
            "end_time": float(turn.end),
            "ref_speaker": turn.ref_speaker or "__NONE__",
            "hyp_speaker": turn.hyp_speaker or "__NONE__",
            "reference": normalize_text(
                turn.ref_text, language=language, detected_language=detected
            ),
            "hypothesis": normalize_text(
                turn.hyp_text, language=language, detected_language=detected
            ),
        })
    # AMI manifests may be grouped by participant rather than timestamp. All
    # meeting scorers require chronological segment order, especially after
    # multiple predicted speakers collapse to the same UNKNOWN label.
    rows.sort(key=lambda row: (
        row["start_time"], row["end_time"], str(row["turn_id"]),
    ))
    return rows


def compute_jiwer_metrics(
    turns: Sequence[SessionTurnResult], *, language: str = "en"
) -> dict[str, Any]:
    """Return JiWER's full public word/character metric payload."""
    try:
        import jiwer
    except Exception as exc:
        return {
            "status": "unavailable",
            "library": "jiwer",
            "library_version": _package_version("jiwer"),
            "error": f"{type(exc).__name__}: {exc}",
        }

    rows = _normalized_turns(turns, language)
    reference = " ".join(r["reference"] for r in rows).strip()
    hypothesis = " ".join(r["hypothesis"] for r in rows).strip()
    words = jiwer.process_words(reference, hypothesis)
    chars = jiwer.process_characters(reference, hypothesis)
    return {
        "status": "ok",
        "library": "jiwer",
        "library_version": _package_version("jiwer"),
        "normalizer_version": NORMALIZER_VERSION,
        "wer": float(words.wer),
        "mer": float(words.mer),
        "wil": float(words.wil),
        "wip": float(words.wip),
        "cer": float(chars.cer),
        "hits": int(words.hits),
        "substitutions": int(words.substitutions),
        "deletions": int(words.deletions),
        "insertions": int(words.insertions),
        "reference_words": int(words.hits + words.substitutions + words.deletions),
        "hypothesis_words": int(words.hits + words.substitutions + words.insertions),
        "char_hits": int(chars.hits),
        "char_substitutions": int(chars.substitutions),
        "char_deletions": int(chars.deletions),
        "char_insertions": int(chars.insertions),
        "reference_characters": int(chars.hits + chars.substitutions + chars.deletions),
    }


def _speaker_streams(rows: list[dict[str, Any]], key: str, text: str) -> list[str]:
    streams: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row[text]:
            streams[row[key]].append(row[text])
    return [" ".join(streams[speaker]) for speaker in sorted(streams)]


def _stm(rows: list[dict[str, Any]], *, reference: bool) -> str:
    speaker_key = "ref_speaker" if reference else "hyp_speaker"
    text_key = "reference" if reference else "hypothesis"
    lines: list[str] = []
    for index, row in enumerate(rows):
        words = row[text_key]
        if not words:
            continue
        # STM fields cannot contain whitespace. Stable local IDs are sufficient
        # because meeting scorers optimize speaker permutations themselves.
        digest = hashlib.sha1(
            str(row[speaker_key]).encode("utf-8")
        ).hexdigest()[:12]
        speaker = f"spk_{digest}"
        lines.append(
            f"session 1 {speaker} {row['start_time']:.6f} "
            f"{row['end_time']:.6f} {words}"
        )
    return "\n".join(lines)


def _error_rate_payload(score: Any) -> dict[str, Any]:
    payload = _json_safe(score)
    if not isinstance(payload, dict):
        payload = {"value": payload}
    if "error_rate" in payload:
        payload["error_rate"] = float(payload["error_rate"])
    return payload


def _speaker_self_overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Diagnose overlapping output segments assigned to the same speaker.

    This is not a replacement metric. MeetEval can warn about this malformed
    hypothesis geometry, so record its exact extent alongside the public
    scores instead of leaving the warning only in stderr.
    """
    per_speaker: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if not row["hypothesis"] or row["end_time"] <= row["start_time"]:
            continue
        per_speaker[str(row["hyp_speaker"])].append(
            (float(row["start_time"]), float(row["end_time"]))
        )

    overlap_by_speaker: dict[str, float] = {}
    for speaker, intervals in per_speaker.items():
        events: list[tuple[float, int]] = []
        for start, end in intervals:
            events.extend(((start, 1), (end, -1)))
        # End events precede start events at the same timestamp, so touching
        # segments are not counted as overlap.
        events.sort(key=lambda item: (item[0], item[1]))
        active = 0
        previous: float | None = None
        overlap = 0.0
        for timestamp, delta in events:
            if previous is not None and active > 1:
                overlap += timestamp - previous
            active += delta
            previous = timestamp
        if overlap > 0:
            overlap_by_speaker[speaker] = overlap

    total = sum(overlap_by_speaker.values())
    return {
        "hypothesis_self_overlap_seconds": total,
        "speakers_with_self_overlap": len(overlap_by_speaker),
        "per_speaker_seconds": overlap_by_speaker,
        "status": "warning" if total > 0 else "ok",
    }


def compute_meeteval_metrics(
    turns: Sequence[SessionTurnResult],
    *,
    language: str = "en",
    tcpwer_collar: float = 5.0,
    score_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute the applicable public MeetEval meeting-ASR metrics.

    Exact and greedy variants are both retained for appendix/error analysis.
    A per-metric failure is recorded instead of aborting the inference run.
    """
    try:
        import meeteval
    except Exception as exc:
        return {
            "status": "unavailable",
            "library": "meeteval",
            "library_version": _package_version("meeteval"),
            "error": f"{type(exc).__name__}: {exc}",
        }

    rows = _normalized_turns(turns, language)
    ref_streams = _speaker_streams(rows, "ref_speaker", "reference")
    hyp_streams = _speaker_streams(rows, "hyp_speaker", "hypothesis")
    ref_utterances = [r["reference"] for r in rows if r["reference"]]
    reference = " ".join(r["reference"] for r in rows).strip()
    hypothesis = " ".join(r["hypothesis"] for r in rows).strip()
    ref_stm = meeteval.io.STM.parse(_stm(rows, reference=True))
    hyp_stm = meeteval.io.STM.parse(_stm(rows, reference=False))

    # STM timestamps are Decimal in MeetEval 0.4.x. Keep the collar in the
    # same numeric domain; a float collar raises Decimal-minus-float inside
    # the official time-constrained scorer.
    collar = Decimal(str(tcpwer_collar))
    scorers: dict[str, Callable[[], Any]] = {
        "siso_wer": lambda: meeteval.wer.wer.siso.siso_word_error_rate(
            reference, hypothesis
        ),
        "cpwer": lambda: meeteval.wer.wer.cp.cp_word_error_rate(
            ref_streams, hyp_streams
        ),
        "orcwer": lambda: meeteval.wer.wer.orc.orc_word_error_rate(
            ref_utterances, hyp_streams
        ),
        "tcpwer_collar_5s": lambda: meeteval.wer.combine_error_rates(
            meeteval.wer.tcpwer(ref_stm, hyp_stm, collar=collar)
        ),
        "tcorcwer_collar_5s": lambda: meeteval.wer.combine_error_rates(
            meeteval.wer.tcorcwer(ref_stm, hyp_stm, collar=collar)
        ),
        "greedy_orcwer": lambda: meeteval.wer.combine_error_rates(
            meeteval.wer.greedy_orcwer(ref_stm, hyp_stm)
        ),
        "greedy_tcorcwer_collar_5s": lambda: meeteval.wer.combine_error_rates(
            meeteval.wer.greedy_tcorcwer(
                ref_stm, hyp_stm, collar=collar
            )
        ),
        "greedy_dicpwer": lambda: meeteval.wer.combine_error_rates(
            meeteval.wer.greedy_dicpwer(ref_stm, hyp_stm)
        ),
        "mimower": lambda: meeteval.wer.combine_error_rates(
            meeteval.wer.mimower(ref_stm, hyp_stm)
        ),
        "tcmimower_collar_5s": lambda: meeteval.wer.combine_error_rates(
            meeteval.wer.tcmimower(ref_stm, hyp_stm, collar=collar)
        ),
    }
    if score_names is not None:
        unknown = sorted(set(score_names) - set(scorers))
        if unknown:
            raise ValueError(f"Unknown MeetEval score(s): {unknown}")
        scorers = {name: scorers[name] for name in score_names}
    scores: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for name, scorer in scorers.items():
        try:
            scores[name] = _error_rate_payload(scorer())
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"

    return {
        "status": "ok" if not failures else "partial",
        "library": "meeteval",
        "library_version": _package_version("meeteval"),
        "normalizer_version": NORMALIZER_VERSION,
        "protocol": {
            "segmentation": "oracle_turns",
            "tcpwer_collar_seconds": float(tcpwer_collar),
            "pseudo_word_timing": "character_based (MeetEval default)",
        },
        "diagnostics": _speaker_self_overlap(rows),
        "scores": scores,
        "failures": failures,
    }


def compute_pyannote_diarization_metrics(
    turns: Sequence[SessionTurnResult],
) -> dict[str, Any]:
    """Compute public DER/JER under the explicitly named oracle-turn protocol."""
    try:
        from pyannote.core import Annotation, Segment
        from pyannote.metrics.diarization import (
            DiarizationErrorRate,
            JaccardErrorRate,
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "library": "pyannote.metrics",
            "library_version": _package_version("pyannote.metrics"),
            "error": f"{type(exc).__name__}: {exc}",
        }

    reference = Annotation(uri="session")
    hypothesis = Annotation(uri="session")
    for index, turn in enumerate(turns):
        segment = Segment(float(turn.start), float(turn.end))
        if turn.ref_speaker is not None:
            reference[segment, f"ref_{index}"] = str(turn.ref_speaker)
        if turn.hyp_speaker is not None:
            hypothesis[segment, f"hyp_{index}"] = str(turn.hyp_speaker)

    scores: dict[str, Any] = {}
    for collar in (0.0, 0.25):
        suffix = "0s" if collar == 0 else "0p25s"
        der_metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
        jer_metric = JaccardErrorRate(collar=collar, skip_overlap=False)
        der_details = der_metric(reference, hypothesis, detailed=True)
        jer_details = jer_metric(reference, hypothesis, detailed=True)
        scores[f"der_collar_{suffix}"] = {
            "value": float(abs(der_metric)),
            "components": _json_safe(der_details),
        }
        scores[f"jer_collar_{suffix}"] = {
            "value": float(abs(jer_metric)),
            "components": _json_safe(jer_details),
        }
    return {
        "status": "ok",
        "library": "pyannote.metrics",
        "library_version": _package_version("pyannote.metrics"),
        "protocol": {
            "segmentation": "oracle_turns",
            "skip_overlap": False,
            "collars_seconds": [0.0, 0.25],
        },
        "scores": scores,
    }


def compute_sklearn_metrics(
    turns: Sequence[SessionTurnResult], *, language: str = "en"
) -> dict[str, Any]:
    """Public turn-classification and confidence-quality diagnostics.

    Speaker labels are first mapped with the same meeting-level Hungarian
    mapping used by the backwards-compatible AV-SID score.  Confidence is
    evaluated against exact normalized turn transcription correctness; this
    target definition is recorded because it is task-specific.
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            balanced_accuracy_score,
            brier_score_loss,
            cohen_kappa_score,
            f1_score,
            log_loss,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from .metrics import compute_av_sid_accuracy
    except Exception as exc:
        return {
            "status": "unavailable",
            "library": "scikit-learn",
            "library_version": _package_version("scikit-learn"),
            "error": f"{type(exc).__name__}: {exc}",
        }

    _, sid_details = compute_av_sid_accuracy(turns)
    mapping = sid_details.get("mapping", {})
    labeled = [t for t in turns if t.ref_speaker is not None]
    y_true = [str(t.ref_speaker) for t in labeled]
    y_pred = [
        str(mapping.get(t.hyp_speaker, t.hyp_speaker or "__NONE__"))
        for t in labeled
    ]
    multiple_labels = len(set(y_true) | set(y_pred)) > 1
    # Speaker IDs are categorical by construction. Small subgroup slices can
    # legitimately contain almost as many IDs as turns, and an unknown/predicted
    # ID may be absent from y_true. Scikit-learn warns about both situations as
    # heuristics; neither indicates a malformed target in this scoring protocol.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50%.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
        )
        classification = {
            "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else None,
            "balanced_accuracy": (
                float(balanced_accuracy_score(y_true, y_pred))
                if y_true and multiple_labels else None
            ),
            "macro_precision": (
                float(precision_score(y_true, y_pred, average="macro", zero_division=0))
                if y_true else None
            ),
            "macro_recall": (
                float(recall_score(y_true, y_pred, average="macro", zero_division=0))
                if y_true else None
            ),
            "macro_f1": (
                float(f1_score(y_true, y_pred, average="macro", zero_division=0))
                if y_true else None
            ),
            "weighted_f1": (
                float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
                if y_true else None
            ),
            "matthews_correlation_coefficient": (
                float(matthews_corrcoef(y_true, y_pred))
                if y_true and multiple_labels else None
            ),
            "cohen_kappa": (
                float(cohen_kappa_score(y_true, y_pred))
                if y_true and multiple_labels else None
            ),
            "n_turns": len(y_true),
        }

    rows = _normalized_turns(turns, language)
    correctness = [int(r["reference"] == r["hypothesis"]) for r in rows]
    confidence = [min(1.0, max(0.0, float(t.confidence))) for t in turns]
    confidence_metrics: dict[str, Any] = {
        "target": "exact_normalized_turn_transcript_match",
        "exact_match_accuracy": float(accuracy_score(correctness, [1] * len(correctness)))
        if correctness else None,
        "brier_score": float(brier_score_loss(correctness, confidence))
        if correctness else None,
        "log_loss": float(log_loss(correctness, confidence, labels=[0, 1]))
        if correctness else None,
        "roc_auc": None,
        "average_precision": None,
        "n_turns": len(correctness),
        "n_correct": sum(correctness),
    }
    if correctness:
        n_bins = 15
        ece = 0.0
        for lower in [i / n_bins for i in range(n_bins)]:
            upper = lower + 1.0 / n_bins
            members = [
                i for i, value in enumerate(confidence)
                if lower <= value < upper or (upper >= 1.0 and value == 1.0)
            ]
            if members:
                bin_acc = sum(correctness[i] for i in members) / len(members)
                bin_conf = sum(confidence[i] for i in members) / len(members)
                ece += len(members) / len(correctness) * abs(bin_acc - bin_conf)
        order = sorted(range(len(confidence)), key=lambda i: confidence[i], reverse=True)
        cumulative_errors = 0
        risks: list[float] = []
        for rank, index in enumerate(order, start=1):
            cumulative_errors += 1 - correctness[index]
            risks.append(cumulative_errors / rank)
        confidence_metrics.update({
            "expected_calibration_error_15_bins": float(ece),
            "area_under_risk_coverage_curve": float(sum(risks) / len(risks)),
            "calibration_note": (
                "ECE and AURC are necessary task diagnostics computed from "
                "scikit-learn-compatible probabilities; definitions are fixed in scoring_protocol.json"
            ),
        })
    if len(set(correctness)) == 2:
        confidence_metrics["roc_auc"] = float(
            roc_auc_score(correctness, confidence)
        )
        confidence_metrics["average_precision"] = float(
            average_precision_score(correctness, confidence)
        )

    return _json_safe({
        "status": "ok",
        "library": "scikit-learn",
        "library_version": _package_version("scikit-learn"),
        "protocol": {
            "speaker_labels": "meeting-level Hungarian mapped",
            "confidence_target": "exact normalized turn transcript match",
        },
        "speaker_classification": classification,
        "confidence_quality": confidence_metrics,
    })


def _never_raise(name: str, scorer: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return scorer()
    except Exception as exc:
        return {
            "status": "failed",
            "library": name,
            "library_version": _package_version(name),
            "error": f"{type(exc).__name__}: {exc}",
        }


def compute_standard_metrics(
    turns: Sequence[SessionTurnResult], *, language: str = "en",
    meeteval_score_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute public-library metrics while the turn outputs are resident.

    ``meeteval_score_names`` lets aggregate/formal reporting request only the
    preregistered metrics it publishes. Some appendix scorers (notably MIMO
    variants) have very high memory complexity and must not run implicitly for
    every meeting and subgroup during an offline rebuild.
    """
    return {
        "jiwer": _never_raise(
            "jiwer", lambda: compute_jiwer_metrics(turns, language=language)
        ),
        "meeteval": _never_raise(
            "meeteval", lambda: compute_meeteval_metrics(
                turns, language=language, score_names=meeteval_score_names,
            )
        ),
        "sklearn": _never_raise(
            "scikit-learn",
            lambda: compute_sklearn_metrics(turns, language=language),
        ),
        "pyannote": _never_raise(
            "pyannote.metrics", lambda: compute_pyannote_diarization_metrics(turns)
        ),
    }
