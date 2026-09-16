"""Rebuild a complete formal scoring tree from saved per-turn records.

This script never loads model checkpoints or media.  By default it copies the
existing artifact tree to a new directory and replaces only reproducible
scoring products.  Use ``--in-place`` explicitly to update the source tree.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avsd_ger.eval.formal_artifacts import (  # noqa: E402
    _c1_calibration,
    _correction_metrics,
    _git_snapshot,
    _group_metrics,
    _package_versions,
    _rttm,
    _score_records,
    _scoring_protocol,
    _seglst,
    _stm,
    _topk_sid,
    _write_json,
    _write_jsonl,
)
from avsd_ger.eval.statistics import (  # noqa: E402
    build_paired_comparisons,
    build_statistics_report,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _existing_main_rows(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "metrics" / "main_table.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row.get("ablation")): row
        for row in payload.get("rows", []) if row.get("ablation")
    }


def _assert_public_complete(ablation: str, scored: dict[str, Any]) -> None:
    standard = scored["standard_metrics"]
    failures = {
        name: payload
        for name, payload in standard.items()
        if name in {"jiwer", "meeteval", "pyannote"}
        and payload.get("status") != "ok"
    }
    required_values = {
        "wer": standard.get("jiwer", {}).get("wer"),
        "cpwer": standard.get("meeteval", {}).get("scores", {}).get("cpwer", {}).get("error_rate"),
        "tcpwer_5s": standard.get("meeteval", {}).get("scores", {}).get("tcpwer_collar_5s", {}).get("error_rate"),
        "der": standard.get("pyannote", {}).get("scores", {}).get("der_collar_0s", {}).get("value"),
        "jer": standard.get("pyannote", {}).get("scores", {}).get("jer_collar_0s", {}).get("value"),
    }
    missing = sorted(name for name, value in required_values.items() if value is None)
    if failures or missing:
        statuses = {name: payload.get("status") for name, payload in failures.items()}
        raise RuntimeError(
            f"{ablation}: public scoring is incomplete; statuses={statuses}, "
            f"missing={missing}. Run this script in the dedicated eval environment."
        )


def _raw_runs_from_scores(
    scores: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    meetings = sorted({
        row["meeting_id"]
        for scored in scores.values() for row in scored["per_meeting"]
    })
    by_ablation = {
        ablation: {row["meeting_id"]: row for row in scored["per_meeting"]}
        for ablation, scored in scores.items()
    }
    runs: list[dict[str, Any]] = []
    for meeting in meetings:
        results = []
        for ablation in sorted(scores):
            row = by_ablation[ablation].get(meeting)
            if row is None:
                continue
            results.append({
                "ablation": ablation,
                "metrics": row["task_metrics"],
                "metric_details": row["task_metric_details"],
                "standard_metrics": row["standard_metrics"],
            })
        runs.append({"manifest": f"{meeting}.json", "results": results})
    return runs


def _validate_reference_identity(records: dict[str, list[dict[str, Any]]]) -> None:
    """All ablations must contain exactly the same immutable reference turns."""
    baseline_name = sorted(records)[0]
    fields = ("meeting_id", "utt_id", "start_time", "end_time", "ref_text", "speaker_ref")
    baseline = {
        (str(row["meeting_id"]), str(row["utt_id"])):
        tuple(row.get(field) for field in fields)
        for row in records[baseline_name]
    }
    for ablation, rows in records.items():
        candidate = {
            (str(row["meeting_id"]), str(row["utt_id"])):
            tuple(row.get(field) for field in fields)
            for row in rows
        }
        if candidate != baseline:
            missing = sorted(set(baseline) - set(candidate))[:10]
            extra = sorted(set(candidate) - set(baseline))[:10]
            changed = sorted(
                key for key in set(baseline) & set(candidate)
                if baseline[key] != candidate[key]
            )[:10]
            raise ValueError(
                f"Reference-turn invariant failed for {ablation} against "
                f"{baseline_name}; missing={missing}, extra={extra}, changed={changed}"
            )


def rebuild(
    source: Path, destination: Path, *, language: str,
    bootstrap_samples: int, bootstrap_seed: int, allow_incomplete: bool,
) -> None:
    record_paths = sorted((source / "records").glob("*.jsonl"))
    if not record_paths:
        raise FileNotFoundError(f"No per-turn records found under {source / 'records'}")
    records = {path.stem: _read_jsonl(path) for path in record_paths}
    if any(not rows for rows in records.values()):
        empty = sorted(name for name, rows in records.items() if not rows)
        raise ValueError(f"Empty ablation record files: {empty}")
    _validate_reference_identity(records)

    scored = {name: _score_records(rows, language) for name, rows in records.items()}
    if not allow_incomplete:
        for name, payload in scored.items():
            _assert_public_complete(name, payload)

    if destination != source:
        if destination.exists():
            raise FileExistsError(
                f"Destination already exists: {destination}; choose a new --out-dir"
            )
        shutil.copytree(source, destination)

    old_main = _existing_main_rows(source)
    per_ablation: dict[str, Any] = {}
    correction: dict[str, Any] = {}
    sid: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    main_rows: list[dict[str, Any]] = []

    for ablation, rows in records.items():
        payload = scored[ablation]
        mappings = {
            meeting["meeting_id"]: (
                meeting["task_metric_details"].get("av_sid", {}).get("mapping", {})
            )
            for meeting in payload["per_meeting"]
        }
        correction[ablation] = _correction_metrics(rows, language)
        sid[ablation] = _topk_sid(rows, mappings)
        calibration[ablation] = {
            "ger": payload["standard_metrics"].get("sklearn", {}).get("confidence_quality", {}),
            "c1": _c1_calibration(rows, mappings),
        }
        per_ablation[ablation] = {
            **payload,
            "correction": correction[ablation],
            "speaker_identification": sid[ablation],
        }
        _write_json(
            destination / "metrics" / "per_ablation" / f"{ablation}.json",
            per_ablation[ablation],
        )
        _write_jsonl(
            destination / "metrics" / "per_meeting" / f"{ablation}.jsonl",
            payload["per_meeting"],
        )

        standard = payload["standard_metrics"]
        task = payload["task_metrics"]
        wer = standard["jiwer"].get("wer")
        cpwer = standard["meeteval"].get("scores", {}).get("cpwer", {}).get("error_rate")
        tcpwer = standard["meeteval"].get("scores", {}).get("tcpwer_collar_5s", {}).get("error_rate")
        der = standard["pyannote"].get("scores", {}).get("der_collar_0s", {}).get("value")
        jer = standard["pyannote"].get("scores", {}).get("jer_collar_0s", {}).get("value")
        asr = _score_records([
            {**row, "ger_hyp_final": row.get("asr_hyp", "")} for row in rows
        ], language)
        asr_wer = asr["standard_metrics"]["jiwer"].get("wer")
        previous = old_main.get(ablation, {})
        main_rows.append({
            "ablation": ablation, **payload["counts"],
            "wer": wer, "cpwer": cpwer, "tcpwer_5s": tcpwer,
            "sa_wer": task["sa_wer"], "scr": task["scr"],
            "av_sid_acc": task["av_sid_acc"], "der": der, "jer": jer,
            "der_custom": task["der"], "jer_custom": task["jer"],
            "asr_baseline_wer": asr_wer,
            "werr": ((asr_wer - wer) / asr_wer) if asr_wer and wer is not None else None,
            "ocr": correction[ablation]["overcorrection_rate"],
            "rtf": previous.get("rtf"),
            "public_scoring_complete": all(
                standard.get(name, {}).get("status") == "ok"
                for name in ("jiwer", "meeteval", "pyannote")
            ),
            "metric_sources": {
                "wer": "jiwer", "cpwer": "meeteval", "tcpwer_5s": "meeteval",
                "der": "pyannote.metrics", "jer": "pyannote.metrics",
                "sa_wer": "project_specific", "scr": "project_specific",
                "av_sid_acc": "project_specific_meeting_local_hungarian",
                "der_custom": "project_specific", "jer_custom": "project_specific",
            },
        })

        reference = destination / "scoring_inputs" / "reference"
        reference.mkdir(parents=True, exist_ok=True)
        hypothesis = destination / "scoring_inputs" / "hypothesis" / ablation
        hypothesis.mkdir(parents=True, exist_ok=True)
        # The reference is identical across ablations and can be overwritten
        # deterministically after every validation pass.
        (reference / "reference.stm").write_text(
            _stm(rows, False, language=language), encoding="utf-8",
        )
        (reference / "reference.raw.stm").write_text(
            _stm(rows, False, language=language, normalized=False), encoding="utf-8",
        )
        (reference / "reference.rttm").write_text(_rttm(rows, False), encoding="utf-8")
        _write_json(reference / "reference.seglst.json", _seglst(rows, False, language=language))
        _write_json(reference / "reference.raw.seglst.json", _seglst(rows, False, language=language, normalized=False))
        (hypothesis / "hypothesis.stm").write_text(
            _stm(rows, True, language=language), encoding="utf-8",
        )
        (hypothesis / "hypothesis.raw.stm").write_text(
            _stm(rows, True, language=language, normalized=False), encoding="utf-8",
        )
        (hypothesis / "hypothesis.rttm").write_text(_rttm(rows, True), encoding="utf-8")
        _write_json(hypothesis / "hypothesis.seglst.json", _seglst(rows, True, language=language))
        _write_json(hypothesis / "hypothesis.raw.seglst.json", _seglst(rows, True, language=language, normalized=False))

    _write_json(destination / "metrics" / "main_table.json", {"rows": main_rows})
    _write_json(destination / "metrics" / "appendix_sdi.json", {
        name: payload["standard_metrics"]["jiwer"] for name, payload in scored.items()
    })
    _write_json(destination / "metrics" / "appendix_correction.json", correction)
    _write_json(destination / "metrics" / "appendix_sid.json", sid)
    _write_json(destination / "metrics" / "appendix_calibration.json", calibration)
    protocol = _scoring_protocol(language)
    protocol["confirmatory_analysis"]["bootstrap_samples"] = bootstrap_samples
    protocol["confirmatory_analysis"]["bootstrap_seed"] = bootstrap_seed
    _write_json(destination / "metrics" / "scoring_protocol.json", protocol)

    raw_runs = _raw_runs_from_scores(scored)
    _write_json(
        destination / "metrics" / "statistics.json",
        build_statistics_report(raw_runs, samples=bootstrap_samples, seed=bootstrap_seed),
    )
    _write_json(
        destination / "metrics" / "paired_comparisons.json",
        build_paired_comparisons(raw_runs, samples=bootstrap_samples, seed=bootstrap_seed),
    )
    for name, payload in _group_metrics(records, language).items():
        _write_json(destination / "metrics" / "groups" / f"{name}.json", payload)

    manifest_path = destination / "run_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists() else {}
    )
    actual_public_complete = all(
        all(
            payload["standard_metrics"].get(name, {}).get("status") == "ok"
            for name in ("jiwer", "meeteval", "pyannote")
        )
        for payload in scored.values()
    )
    manifest.update({
        "status": "complete" if actual_public_complete else "metrics_incomplete",
        "public_scoring_complete": actual_public_complete,
        "metrics_rebuilt_at": datetime.now(timezone.utc).isoformat(),
        "metrics_rebuild_git": _git_snapshot(ROOT),
        "metrics_libraries": _package_versions(),
    })
    _write_json(manifest_path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--out-dir", type=Path)
    destination.add_argument("--in-place", action="store_true")
    parser.add_argument("--language", default="en")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    source = args.artifact_root.resolve()
    target = source if args.in_place else args.out_dir.resolve()
    rebuild(
        source, target, language=args.language,
        bootstrap_samples=args.samples, bootstrap_seed=args.seed,
        allow_incomplete=args.allow_incomplete,
    )
    print(f"[rebuilt] {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
