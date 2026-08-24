"""Recompute canonical Phase-A diagnostics from eval summary/debug JSON files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avsd_ger.text_normalization import (  # noqa: E402
    NORMALIZER_VERSION,
    normalize_text,
    resolve_language,
)


SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _edit_counts(ref: str, hyp: str) -> dict[str, int | float]:
    ref_words, hyp_words = ref.split(), hyp.split()
    n, m = len(ref_words), len(hyp_words)
    dp: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)
    ]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            diag = dp[i - 1][j - 1]
            delete = dp[i - 1][j]
            insert = dp[i][j - 1]
            candidates = (
                (diag[0] + 1, diag[1] + 1, diag[2], diag[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            )
            dp[i][j] = min(candidates, key=lambda value: value[0])
    edits, substitutions, deletions, insertions = dp[n][m]
    result: dict[str, int | float] = {
        "ref_words": n,
        "edits": edits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "wer": edits / n if n else (float(m > 0)),
    }
    try:
        import jiwer
    except ImportError as exc:
        raise RuntimeError("Phase-A analysis requires the pinned jiwer==4.0.0") from exc
    reference = jiwer.process_words(ref, hyp)
    independent_edits = (
        reference.substitutions + reference.deletions + reference.insertions
    )
    if independent_edits != edits:
        raise AssertionError(
            f"WER implementation disagrees with jiwer: {edits} != {independent_edits}"
        )
    return result


def _iter_results(payload: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(payload.get("runs"), list):
        for run in payload["runs"]:
            manifest = str(run.get("manifest", ""))
            for result in run.get("results", []):
                yield manifest, result
    elif isinstance(payload.get("results"), list):
        manifest = str(payload.get("manifest", ""))
        for result in payload["results"]:
            yield manifest, result
    elif isinstance(payload.get("turns"), list):
        yield str(payload.get("manifest", "")), payload
    else:
        raise ValueError("unsupported input schema: expected runs, results, or turns")


def _debug_path(summary_path: Path, manifest: str, result: dict[str, Any]) -> Path:
    if isinstance(result.get("turns"), list):
        return summary_path
    raw = result.get("debug_path")
    if raw:
        candidate = Path(str(raw))
        name = candidate.name
    else:
        name = f"{Path(manifest).stem}.{result.get('ablation', 'unknown')}.debug.json"
    local = summary_path.parent / f"{Path(manifest).stem}_debug" / name
    if local.is_file():
        return local
    if raw and candidate.is_file():
        return candidate
    raise FileNotFoundError(f"debug sidecar not found for {manifest}: {raw or name}")


def _turn_parts(turn: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    summary = turn.get("summary") or {}
    pipeline = turn.get("pipeline") if isinstance(turn.get("pipeline"), dict) else turn
    trace = turn.get("trace") or pipeline.get("trace") or []
    return summary, pipeline, list(trace)


def _text_score(
    ref: str,
    hyp: str,
    *,
    language: str,
    detected_language: str | None,
) -> tuple[str, dict[str, int | float]]:
    normalized = normalize_text(
        hyp, language=language, detected_language=detected_language
    )
    normalized_ref = normalize_text(
        ref, language=language, detected_language=detected_language
    )
    return normalized, _edit_counts(normalized_ref, normalized)


def analyze(inputs: list[Path], *, language: str) -> dict[str, Any]:
    turn_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    gate_counts: dict[str, Counter[str]] = defaultdict(Counter)
    fallback_reasons: Counter[str] = Counter()

    for input_path in inputs:
        summary_path = input_path / "summary.json" if input_path.is_dir() else input_path
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        lowered = str(summary_path).lower()
        model_label = "qwen" if "qwen" in lowered else "llama" if "llama" in lowered else summary_path.parent.name
        input_records.append({"path": str(summary_path), "sha256": _sha256(summary_path)})
        for manifest, result in _iter_results(payload):
            debug_path = _debug_path(summary_path, manifest, result)
            debug = result if isinstance(result.get("turns"), list) else json.loads(
                debug_path.read_text(encoding="utf-8")
            )
            ablation = str(result.get("ablation") or debug.get("ablation") or "unknown")
            semantics = (
                "legacy_update_gate_only"
                if ablation == "c3_wo_conf_gate"
                else "c3_v2" if ablation == "c3_wo_conf_gates" else "not_applicable"
            )
            flags = result.get("flags") or debug.get("flags") or {}
            eligible = not bool(flags.get("disable_c2")) and ablation != "wo_c2"
            session_start = len(turn_rows)
            for index, turn in enumerate(debug.get("turns", [])):
                summary, pipeline, trace = _turn_parts(turn)
                turn_meta = turn.get("turn") or {}
                asr_debug = pipeline.get("asr") or {}
                detected = asr_debug.get("detected_language")
                resolved = resolve_language(language, detected)
                ref = str(summary.get("ref_text") or turn_meta.get("ref_text") or "")
                asr = str(summary.get("asr_top") or asr_debug.get("top") or (trace[0].get("asr_top") if trace else "") or "")
                final = str(summary.get("final_text") or (pipeline.get("final") or {}).get("text") or "")
                lip = str(summary.get("lip_hyp") or (trace[0].get("lip_hyp") if trace else "") or "")
                last = trace[-1] if trace else {}
                raw_ger = (
                    str(last.get("cleaned_ger_text_before_gate") or "")
                    if eligible else ""
                )
                asr_norm, asr_score = _text_score(ref, asr, language=resolved, detected_language=detected)
                final_norm, final_score = _text_score(ref, final, language=resolved, detected_language=detected)
                if eligible:
                    raw_norm, raw_score = _text_score(
                        ref, raw_ger, language=resolved, detected_language=detected
                    )
                else:
                    raw_norm, raw_score = "", None
                lip_norm, lip_score = _text_score(ref, lip, language=resolved, detected_language=detected)
                final_source = str(last.get("final_source") or ("ASR" if last.get("fallback_applied") else "GER"))
                accepted = eligible and final_source == "GER" and final_norm != asr_norm
                delta = (
                    float(final_score["wer"]) - float(asr_score["wer"])
                    if eligible else None
                )
                outcome = (
                    "not_applicable" if not eligible
                    else "improved" if delta < 0
                    else "harmed" if delta > 0 else "unchanged"
                )
                c1 = pipeline.get("c1_initial") or {}
                top_ids = c1.get("top_ids") or summary.get("top_ids") or []
                ref_speaker = summary.get("ref_speaker") or turn_meta.get("ref_speaker")
                raw_top1 = top_ids[0] if top_ids else None
                is_unknown = bool(c1.get("is_unknown", summary.get("is_unknown", True)))
                iteration_fallbacks = sum(bool(item.get("fallback_applied")) for item in trace)
                for item in trace:
                    if item.get("fallback_applied"):
                        fallback_reasons[str(item.get("fallback_reason") or "<unknown>")] += 1
                    for gate in item.get("safety_gates") or []:
                        gate_counts[str(gate.get("gate", "unknown"))][
                            "passed" if gate.get("passed") else "failed"
                        ] += 1
                row = {
                    "model": model_label,
                    "input": str(summary_path),
                    "debug_path": str(debug_path),
                    "manifest": manifest or str(debug.get("manifest", "")),
                    "session": Path(manifest or str(debug.get("manifest", "unknown"))).stem,
                    "ablation": ablation,
                    "ablation_semantics": semantics,
                    "turn_index": index,
                    "turn_id": summary.get("turn_id") or turn_meta.get("turn_id"),
                    "duration_s": float(summary.get("duration") or turn_meta.get("duration") or 0.0),
                    "language": resolved,
                    "ref_text": ref,
                    "asr_text": asr,
                    "raw_ger_text": raw_ger,
                    "final_text": final,
                    "lip_text": lip,
                    "asr_normalized": asr_norm,
                    "raw_ger_normalized": raw_norm,
                    "final_normalized": final_norm,
                    "lip_normalized": lip_norm,
                    "asr_wer": asr_score["wer"],
                    "raw_ger_wer": raw_score["wer"] if raw_score is not None else None,
                    "final_wer": final_score["wer"],
                    "lip_wer": lip_score["wer"],
                    "asr_edits": asr_score["edits"],
                    "raw_ger_edits": raw_score["edits"] if raw_score is not None else None,
                    "final_edits": final_score["edits"],
                    "lip_edits": lip_score["edits"],
                    "ref_words": final_score["ref_words"],
                    "edit_delta": delta,
                    "outcome": outcome,
                    "ger_eligible": eligible,
                    "ger_candidate_available": bool(raw_norm),
                    "ger_accepted": accepted,
                    "iteration_fallbacks": iteration_fallbacks,
                    "final_fallback": (
                        bool(last.get("fallback_applied")) if eligible else None
                    ),
                    "final_source": final_source,
                    "iterations": len(trace),
                    "fallback_reason": last.get("fallback_reason"),
                    "ref_speaker": ref_speaker,
                    "c1_raw_top1": raw_top1,
                    "c1_raw_top1_correct": raw_top1 is not None and raw_top1 == ref_speaker,
                    "c1_is_unknown": is_unknown,
                }
                turn_rows.append(row)

            rows = turn_rows[session_start:]
            session_row = _aggregate_rows(rows, manifest=manifest, ablation=ablation)
            session_row["model"] = model_label
            power = result.get("power") or {}
            session_row["wall_duration_s"] = power.get("duration_s")
            session_rows.append(session_row)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in turn_rows:
        grouped[(row["model"], row["ablation"])].append(row)
    aggregates = {
        f"{model}/{ablation}": {
            **_aggregate_rows(rows, manifest="ALL", ablation=ablation),
            "model": model,
        }
        for (model, ablation), rows in sorted(grouped.items())
    }
    runtime: dict[str, Any] = {}
    for (model, ablation), rows in grouped.items():
        key = f"{model}/{ablation}"
        duration = sum(float(row["duration_s"]) for row in rows)
        matching_sessions = [
            row for row in session_rows
            if row["model"] == model and row["ablation"] == ablation
        ]
        wall_values = [
            float(row["wall_duration_s"])
            for row in matching_sessions if row.get("wall_duration_s") is not None
        ]
        wall = sum(wall_values) if wall_values else None
        runtime[key] = {
            "audio_duration_s": duration,
            "wall_duration_s": wall,
            "turns": len(rows),
            "turns_per_audio_hour": len(rows) / duration * 3600.0 if duration else None,
            "turns_per_wall_second": len(rows) / wall if wall else None,
            "real_time_factor": wall / duration if wall and duration else None,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "normalizer_version": NORMALIZER_VERSION,
        "language_request": language,
        "comparison_interpretation": "per-model configuration; not a strictly controlled backbone comparison",
        "inputs": input_records,
        "aggregates": aggregates,
        "runtime": runtime,
        "gate_counts": {name: dict(counts) for name, counts in sorted(gate_counts.items())},
        "fallback_reasons": dict(fallback_reasons.most_common()),
        "sessions": session_rows,
        "turns": turn_rows,
    }


def _aggregate_rows(rows: list[dict[str, Any]], *, manifest: str, ablation: str) -> dict[str, Any]:
    ref_words = sum(int(row["ref_words"]) for row in rows)
    eligible = sum(bool(row["ger_eligible"]) for row in rows)
    labeled_c1 = [row for row in rows if row.get("ref_speaker") is not None]
    def micro(key: str, subset: list[dict[str, Any]] | None = None) -> float | None:
        selected = rows if subset is None else subset
        selected_ref_words = sum(int(row["ref_words"]) for row in selected)
        if not selected:
            return None
        return sum(int(row[key]) for row in selected) / selected_ref_words if selected_ref_words else 0.0
    outcomes = Counter(str(row["outcome"]) for row in rows)
    ger_rows = [row for row in rows if bool(row["ger_eligible"])]
    raw_rows = [row for row in ger_rows if row["raw_ger_edits"] is not None]
    return {
        "manifest": manifest,
        "ablation": ablation,
        "n_turns": len(rows),
        "n_ref_words": ref_words,
        "asr_wer_micro": micro("asr_edits"),
        "raw_ger_wer_micro": micro("raw_ger_edits", raw_rows),
        "final_wer_micro": micro("final_edits"),
        "lip_wer_micro": micro("lip_edits"),
        "asr_wer_macro_turn": mean([float(row["asr_wer"]) for row in rows]) if rows else 0.0,
        "raw_ger_wer_macro_turn": (
            mean([float(row["raw_ger_wer"]) for row in raw_rows]) if raw_rows else None
        ),
        "final_wer_macro_turn": mean([float(row["final_wer"]) for row in rows]) if rows else 0.0,
        "outcomes": dict(outcomes),
        "ger_eligible_turns": eligible,
        "ger_candidate_coverage": sum(bool(row["ger_candidate_available"]) for row in rows) / eligible if eligible else 0.0,
        "ger_acceptance_coverage": sum(bool(row["ger_accepted"]) for row in rows) / eligible if eligible else 0.0,
        "iteration_fallbacks": sum(int(row["iteration_fallbacks"]) for row in rows),
        "final_fallback_turns": sum(bool(row["final_fallback"]) for row in ger_rows),
        "final_fallback_rate": (
            sum(bool(row["final_fallback"]) for row in ger_rows) / len(ger_rows)
            if ger_rows else None
        ),
        "c1_labeled_turns": len(labeled_c1),
        "c1_raw_top1_accuracy": sum(bool(row["c1_raw_top1_correct"]) for row in labeled_c1) / len(labeled_c1) if labeled_c1 else 0.0,
        "c1_known_coverage": sum(not bool(row["c1_is_unknown"]) for row in labeled_c1) / len(labeled_c1) if labeled_c1 else 0.0,
        "c1_unknown_coverage": sum(bool(row["c1_is_unknown"]) for row in labeled_c1) / len(labeled_c1) if labeled_c1 else 0.0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase A canonical debug analysis",
        "",
        f"Normalizer: `{report['normalizer_version']}`; language: `{report['language_request']}`.",
        "",
        "| Ablation | ASR WER | Raw GER WER | Final WER | Lip WER | GER accepted | Final fallback | C1 raw top-1 | Known coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def percent(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2%}"

    def metric(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    for name, row in report["aggregates"].items():
        lines.append(
            f"| {name} | {metric(row['asr_wer_micro'])} | {metric(row['raw_ger_wer_micro'])} | "
            f"{metric(row['final_wer_micro'])} | {metric(row['lip_wer_micro'])} | "
            f"{percent(row['ger_acceptance_coverage'])} | {percent(row['final_fallback_rate'])} | "
            f"{percent(row['c1_raw_top1_accuracy'])} | {percent(row['c1_known_coverage'])} |"
        )
    lines.extend(["", "Historical `c3_wo_conf_gate` rows are legacy update-gate-only semantics.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--language", default="auto", help="auto or ISO language code")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.inputs, language=args.language)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    _write_csv(args.out_dir / "sessions.csv", report["sessions"])
    _write_csv(args.out_dir / "turns.csv", report["turns"])
    print(_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
