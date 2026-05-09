"""Reshape downloaded W&B exports into analysis-friendly metric tables.

Input is the directory produced by scripts/download_wandb_runs.py. The script
understands eval_ablations summary keys such as:

    summary/IS1009c/full_model/wer
    manifest/IS1009c/ablation/full_model/fallback_rate

It writes long and wide CSVs where rows are run/session/ablation and columns are
metrics, which is much easier to sort, pivot, and paste into paper tables.

Examples:
    python scripts/summarize_wandb_export.py \
      --export-dir out/wandb_export/overnight \
      --out-dir out/wandb_export/overnight_tables
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SUMMARY_RE = re.compile(r"^summary/(?P<session>[^/]+)/(?P<ablation>[^/]+)/(?P<metric>[^/]+)$")
MANIFEST_RE = re.compile(
    r"^manifest/(?P<session>[^/]+)/ablation/(?P<ablation>[^/]+)/(?P<metric>[^/]+)$"
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _as_number(value: Any) -> Any:
    if value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            if value.lower() in {"nan", "inf", "-inf"}:
                return value
            x = float(value)
            return int(x) if x.is_integer() else x
        except Exception:
            return value
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    preferred = ["run_name", "run_id", "session", "ablation", "metric", "value"]
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


def _extract_metrics(run_row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = {
        "run_name": run_row.get("run_name"),
        "run_id": run_row.get("run_id"),
        "run_path": run_row.get("run_path"),
        "run_url": run_row.get("run_url"),
        "state": run_row.get("state"),
    }
    long_rows: list[dict[str, Any]] = []
    wide: dict[tuple[str, str], dict[str, Any]] = {}

    for key, value in run_row.items():
        raw_key = key
        if raw_key.startswith("summary/"):
            raw_key = raw_key[len("summary/"):]

        m = SUMMARY_RE.match(raw_key) or MANIFEST_RE.match(raw_key)
        if not m:
            continue

        session = m.group("session")
        ablation = m.group("ablation")
        metric = m.group("metric")
        value = _as_number(value)

        long_rows.append({
            **base,
            "session": session,
            "ablation": ablation,
            "metric": metric,
            "value": value,
        })

        wide_key = (session, ablation)
        if wide_key not in wide:
            wide[wide_key] = {
                **base,
                "session": session,
                "ablation": ablation,
            }
        # Prefer summary values, but allow manifest-only extras such as fallback_rate.
        wide[wide_key][metric] = value

    return long_rows, list(wide.values())


def _metric_first_rows(wide_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_metric: dict[tuple[str, str, str], dict[str, Any]] = {}
    id_cols = {"run_name", "run_id", "run_path", "run_url", "state", "session", "ablation"}
    for row in wide_rows:
        for metric, value in row.items():
            if metric in id_cols:
                continue
            key = (str(row.get("run_name")), str(row.get("ablation")), metric)
            if key not in by_metric:
                by_metric[key] = {
                    "run_name": row.get("run_name"),
                    "run_id": row.get("run_id"),
                    "ablation": row.get("ablation"),
                    "metric": metric,
                }
            by_metric[key][str(row.get("session"))] = value
    return list(by_metric.values())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--export-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    runs_jsonl = args.export_dir / "runs.jsonl"
    if not runs_jsonl.exists():
        raise FileNotFoundError(f"Missing {runs_jsonl}; run download_wandb_runs.py first.")

    run_rows = _load_jsonl(runs_jsonl)
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    for row in run_rows:
        lr, wr = _extract_metrics(row)
        long_rows.extend(lr)
        wide_rows.extend(wr)

    # Deterministic ordering for easy diffing and spreadsheet viewing.
    long_rows.sort(key=lambda r: (str(r.get("run_name")), str(r.get("session")), str(r.get("ablation")), str(r.get("metric"))))
    wide_rows.sort(key=lambda r: (str(r.get("run_name")), str(r.get("session")), str(r.get("ablation"))))
    metric_rows = _metric_first_rows(wide_rows)
    metric_rows.sort(key=lambda r: (str(r.get("run_name")), str(r.get("ablation")), str(r.get("metric"))))

    _write_csv(args.out_dir / "metrics_long.csv", long_rows)
    _write_csv(args.out_dir / "metrics_wide.csv", wide_rows)
    _write_csv(args.out_dir / "metrics_by_metric.csv", metric_rows)

    grouped: dict[str, Any] = defaultdict(lambda: defaultdict(dict))
    for row in wide_rows:
        grouped[str(row["session"])][str(row["ablation"])] = {
            k: v for k, v in row.items()
            if k not in {"run_name", "run_id", "run_path", "run_url", "state", "session", "ablation"}
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "metrics_grouped.json", "w", encoding="utf-8") as f:
        json.dump(grouped, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"[wrote] {args.out_dir / 'metrics_long.csv'}")
    print(f"[wrote] {args.out_dir / 'metrics_wide.csv'}")
    print(f"[wrote] {args.out_dir / 'metrics_by_metric.csv'}")
    print(f"[wrote] {args.out_dir / 'metrics_grouped.json'}")
    print(f"[summary] runs={len(run_rows)} long_rows={len(long_rows)} wide_rows={len(wide_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
