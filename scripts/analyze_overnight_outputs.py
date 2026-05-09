"""Aggregate overnight AVSD-GER eval outputs into comparison tables.

This reads local eval_ablations JSON outputs and their debug sidecars, so it
does not depend on W&B web exports. It is meant for quick overnight analysis:

    python scripts/analyze_overnight_outputs.py \
      --safe-core-dir out/overnight_safe_core \
      --av-dir out/overnight_av \
      --out-dir out/overnight_analysis

Outputs:
    metrics_long.csv          one row per experiment/session/ablation/metric
    metrics_wide.csv          one row per experiment/session/ablation
    safe_core_comparison.csv  full vs no_ger_gate deltas
    av_comparison.csv         AV vs audio_only deltas for visual smoke
    debug_summary.csv         artifact/fallback/lip_hyp summaries from debug JSON
    report.md                 compact human-readable report
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METRIC_KEYS = [
    "wer",
    "sa_wer",
    "scr",
    "av_sid_acc",
    "der",
    "jer",
    "n_ref_words",
    "n_turns",
    "fallback_rate",
    "fallback_turns",
    "pool_updates",
    "mean_iterations",
]


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    preferred = [
        "experiment",
        "condition",
        "session",
        "ablation",
        "metric",
        "value",
        "delta",
    ]
    for key in preferred:
        seen.add(key)
        keys.append(key)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)


def _condition_from_path(path: Path) -> str:
    parts = path.parts
    for name in ("full", "no_ger_gate", "artifact_only", "text_gates", "pool_update_on"):
        if name in parts:
            return name
    stem = path.stem
    if stem.endswith("_audio_only"):
        return "audio_only"
    if stem.endswith("_av"):
        return "av"
    if "audio_only" in stem:
        return "audio_only"
    if stem.endswith("av") or "_av" in stem:
        return "av"
    return path.parent.name


def _session_from_payload(path: Path, payload: dict[str, Any]) -> str:
    manifest = payload.get("manifest")
    if manifest:
        return Path(str(manifest)).stem
    return path.stem


def _iter_result_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    paths: list[Path] = []
    for path in sorted(root.rglob("*.json")):
        if path.name.endswith(".debug.json"):
            continue
        if "_debug" in {p.name for p in path.parents}:
            continue
        paths.append(path)
    return paths


def _rows_from_result(path: Path, experiment: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = _read_json(path)
    condition = _condition_from_path(path)
    session = _session_from_payload(path, payload)
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []

    for result in payload.get("results", []):
        ablation = result.get("ablation")
        metrics = dict(result.get("metrics", {}) or {})
        trace = dict(result.get("trace_summary", {}) or {})
        combined = {**metrics}
        for key in ("fallback_rate", "fallback_turns", "pool_updates", "mean_iterations"):
            if key in trace:
                combined[key] = trace[key]

        base = {
            "experiment": experiment,
            "condition": condition,
            "session": session,
            "ablation": ablation,
            "source_json": str(path),
            "debug_path": result.get("debug_path"),
        }
        wide = {**base}
        for key in METRIC_KEYS:
            if key in combined:
                value = combined[key]
                wide[key] = value
                long_rows.append({**base, "metric": key, "value": value})
        wide_rows.append(wide)
    return long_rows, wide_rows


def _index(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row.get(k) for k in keys): row for row in rows}


def _delta_row(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_name: str,
    right_name: str,
    metrics: list[str],
) -> dict[str, Any]:
    row = {
        "session": left.get("session") or right.get("session"),
        "ablation": left.get("ablation") or right.get("ablation"),
        "left": left_name,
        "right": right_name,
    }
    for metric in metrics:
        a = left.get(metric)
        b = right.get(metric)
        row[f"{left_name}_{metric}"] = a
        row[f"{right_name}_{metric}"] = b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            row[f"delta_{metric}"] = a - b
    return row


def _safe_core_comparison(wide_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full = [
        r for r in wide_rows
        if r.get("experiment") == "safe_core" and r.get("condition") == "full"
    ]
    no_gate = [
        r for r in wide_rows
        if r.get("experiment") == "safe_core" and r.get("condition") == "no_ger_gate"
    ]
    no_gate_idx = _index(no_gate, "session", "ablation")
    rows: list[dict[str, Any]] = []
    for row in full:
        peer = no_gate_idx.get((row.get("session"), row.get("ablation")))
        if not peer:
            continue
        out = _delta_row(row, peer, left_name="full", right_name="no_ger_gate", metrics=METRIC_KEYS)
        out["condition"] = "full_minus_no_ger_gate"
        rows.append(out)
    return sorted(rows, key=lambda r: (str(r.get("session")), str(r.get("ablation"))))


def _av_comparison(wide_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    av = [
        r for r in wide_rows
        if r.get("experiment") == "av" and r.get("condition") == "av"
    ]
    audio = [
        r for r in wide_rows
        if r.get("experiment") == "av" and r.get("condition") == "audio_only"
    ]
    audio_idx = _index(audio, "session", "ablation")
    rows: list[dict[str, Any]] = []
    for row in av:
        peer = audio_idx.get((row.get("session"), row.get("ablation")))
        if not peer:
            continue
        out = _delta_row(row, peer, left_name="av", right_name="audio_only", metrics=METRIC_KEYS)
        out["condition"] = "av_minus_audio_only"
        rows.append(out)
    return sorted(rows, key=lambda r: (str(r.get("session")), str(r.get("ablation"))))


def _short_text(s: Any, n: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(s or "")).strip()
    return text[:n]


def _debug_rows(wide_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in wide_rows:
        path = row.get("debug_path")
        if not path:
            continue
        debug_path = Path(str(path))
        if not debug_path.exists():
            continue
        try:
            data = _read_json(debug_path)
        except Exception:
            continue

        fallback_reasons: Counter[str] = Counter()
        artifact_hits: Counter[str] = Counter()
        lip_nonempty = 0
        visual_true = 0
        turns = data.get("turns", [])
        examples: list[str] = []
        for turn in turns:
            summary = turn.get("summary", {}) or {}
            if summary.get("has_visual"):
                visual_true += 1
            if summary.get("lip_hyp"):
                lip_nonempty += 1
            if summary.get("fallback_reason"):
                fallback_reasons[str(summary.get("fallback_reason"))] += 1
            for tr in turn.get("trace", []) or []:
                for hit in ((tr.get("safety_features") or {}).get("artifact_hits") or []):
                    artifact_hits[str(hit)] += 1
            if len(examples) < 5 and (summary.get("fallback_applied") or summary.get("lip_hyp")):
                examples.append(
                    " | ".join([
                        str(summary.get("turn_id")),
                        f"ref={summary.get('ref_speaker')}",
                        f"hyp={summary.get('hyp_speaker')}",
                        f"asr={_short_text(summary.get('asr_top'), 80)}",
                        f"lip={_short_text(summary.get('lip_hyp'), 80)}",
                        f"fb={_short_text(summary.get('fallback_reason'), 80)}",
                    ])
                )

        rows.append({
            "experiment": row.get("experiment"),
            "condition": row.get("condition"),
            "session": row.get("session"),
            "ablation": row.get("ablation"),
            "debug_path": str(debug_path),
            "n_turns_debug": len(turns),
            "has_visual_turns": visual_true,
            "lip_hyp_nonempty_turns": lip_nonempty,
            "top_fallback_reasons": json.dumps(fallback_reasons.most_common(8), ensure_ascii=False),
            "top_artifact_hits": json.dumps(artifact_hits.most_common(8), ensure_ascii=False),
            "examples": "\n".join(examples),
        })
    return rows


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return "" if x is None else str(x)


def _write_report(
    path: Path,
    wide_rows: list[dict[str, Any]],
    safe_cmp: list[dict[str, Any]],
    av_cmp: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Overnight Analysis")
    lines.append("")

    full_rows = [
        r for r in wide_rows
        if r.get("experiment") == "safe_core"
        and r.get("condition") == "full"
        and r.get("ablation") == "full_model"
    ]
    if full_rows:
        lines.append("## Safe Core Full")
        for metric in ("wer", "sa_wer", "der", "jer", "fallback_rate", "pool_updates"):
            vals = [r.get(metric) for r in full_rows if isinstance(r.get(metric), (int, float))]
            if vals:
                lines.append(f"- avg {metric}: {_fmt(mean(vals))}")
        lines.append("")

    cmp_full_model = [r for r in safe_cmp if r.get("ablation") == "full_model"]
    if cmp_full_model:
        lines.append("## Full Preset vs No GER Gate")
        improved = [
            r for r in cmp_full_model
            if isinstance(r.get("delta_wer"), (int, float)) and r["delta_wer"] < 0
        ]
        worsened = [
            r for r in cmp_full_model
            if isinstance(r.get("delta_wer"), (int, float)) and r["delta_wer"] > 0
        ]
        lines.append(f"- WER improved sessions: {len(improved)}/{len(cmp_full_model)}")
        lines.append(f"- WER worsened sessions: {len(worsened)}/{len(cmp_full_model)}")
        best = sorted(
            [r for r in cmp_full_model if isinstance(r.get("delta_wer"), (int, float))],
            key=lambda r: r["delta_wer"],
        )[:5]
        if best:
            lines.append("- largest WER improvements:")
            for r in best:
                lines.append(f"  - {r['session']}: delta_wer={_fmt(r['delta_wer'])}")
        lines.append("")

    if av_cmp:
        lines.append("## AV vs Audio-Only Smoke")
        for r in av_cmp:
            if r.get("ablation") != "full_model":
                continue
            lines.append(
                "- "
                f"{r.get('session')}: "
                f"delta_wer={_fmt(r.get('delta_wer'))}, "
                f"delta_sa_wer={_fmt(r.get('delta_sa_wer'))}, "
                f"delta_der={_fmt(r.get('delta_der'))}, "
                f"delta_fallback_rate={_fmt(r.get('delta_fallback_rate'))}"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--safe-core-dir", default="out/overnight_safe_core", type=Path)
    p.add_argument("--av-dir", default="out/overnight_av", type=Path)
    p.add_argument("--out-dir", default="out/overnight_analysis", type=Path)
    args = p.parse_args()

    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []

    for path in _iter_result_files(args.safe_core_dir):
        lr, wr = _rows_from_result(path, "safe_core")
        long_rows.extend(lr)
        wide_rows.extend(wr)
    for path in _iter_result_files(args.av_dir):
        lr, wr = _rows_from_result(path, "av")
        long_rows.extend(lr)
        wide_rows.extend(wr)

    long_rows.sort(key=lambda r: (str(r.get("experiment")), str(r.get("condition")), str(r.get("session")), str(r.get("ablation")), str(r.get("metric"))))
    wide_rows.sort(key=lambda r: (str(r.get("experiment")), str(r.get("condition")), str(r.get("session")), str(r.get("ablation"))))

    safe_cmp = _safe_core_comparison(wide_rows)
    av_cmp = _av_comparison(wide_rows)
    debug = _debug_rows(wide_rows)

    _write_csv(args.out_dir / "metrics_long.csv", long_rows)
    _write_csv(args.out_dir / "metrics_wide.csv", wide_rows)
    _write_csv(args.out_dir / "safe_core_comparison.csv", safe_cmp)
    _write_csv(args.out_dir / "av_comparison.csv", av_cmp)
    _write_csv(args.out_dir / "debug_summary.csv", debug)
    _write_json(args.out_dir / "metrics_wide.json", wide_rows)
    _write_report(args.out_dir / "report.md", wide_rows, safe_cmp, av_cmp)

    print(f"[wrote] {args.out_dir / 'metrics_wide.csv'}")
    print(f"[wrote] {args.out_dir / 'safe_core_comparison.csv'}")
    print(f"[wrote] {args.out_dir / 'av_comparison.csv'}")
    print(f"[wrote] {args.out_dir / 'debug_summary.csv'}")
    print(f"[wrote] {args.out_dir / 'report.md'}")
    print(f"[summary] result_rows={len(wide_rows)} metric_rows={len(long_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
