"""Download W&B run data without relying on the web UI CSV limit.

Examples:
    # Download every run in a project.
    python scripts/download_wandb_runs.py \
      --wandb-project avsd-ger-4090-overnight \
      --out-dir out/wandb_export/overnight

    # Download only one display-name run.
    python scripts/download_wandb_runs.py \
      --wandb-project avsd-ger-4090-overnight \
      --wandb-run-name IS1009c-closeup50-map21-av-debug \
      --out-dir out/wandb_export/IS1009c_av

    # If your default W&B entity is not set, pass it explicitly.
    python scripts/download_wandb_runs.py \
      --wandb-entity khaichean_how-xiamen-university-malaysia \
      --wandb-project avsd-ger-4090-overnight \
      --out-dir out/wandb_export/overnight
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable


def _safe_name(text: str) -> str:
    text = text.strip() or "unnamed"
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", text)[:180]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


def _flatten(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obj.items():
        k = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, k))
        else:
            out[k] = _jsonable(value)
    return out


def _cell(value: Any) -> Any:
    value = _jsonable(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, ensure_ascii=False, sort_keys=True)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in keys})


def _project_path(entity: str | None, project: str, api: Any) -> str:
    if "/" in project:
        return project
    entity = entity or os.environ.get("WANDB_ENTITY")
    if entity:
        return f"{entity}/{project}"
    default_entity = getattr(api, "default_entity", None)
    if default_entity:
        return f"{default_entity}/{project}"
    return project


def _run_base_row(run: Any) -> dict[str, Any]:
    return {
        "run_id": run.id,
        "run_name": run.name,
        "run_path": "/".join(run.path),
        "run_url": run.url,
        "state": run.state,
        "created_at": str(getattr(run, "created_at", "")),
        "updated_at": str(getattr(run, "updated_at", "")),
        "group": getattr(run, "group", None),
        "job_type": getattr(run, "job_type", None),
        "tags": list(getattr(run, "tags", []) or []),
    }


def _download_history(run: Any, out_dir: Path, *, page_size: int) -> tuple[Path, Path, int]:
    stem = _safe_name(f"{run.name}__{run.id}")
    jsonl_path = out_dir / "history" / f"{stem}.history.jsonl"
    csv_path = out_dir / "history" / f"{stem}.history.csv"

    rows: list[dict[str, Any]] = []
    for row in run.scan_history(page_size=page_size):
        rows.append(_flatten(dict(row)))
    _write_jsonl(jsonl_path, rows)
    _write_csv(csv_path, rows)
    return jsonl_path, csv_path, len(rows)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Download W&B summary/config/history for one run or a whole project."
    )
    p.add_argument("--wandb-project", required=True, help="Project name, or entity/project.")
    p.add_argument("--wandb-entity", default=None, help="W&B entity/team. Optional if default entity works.")
    p.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional display name or run id. If omitted, downloads every run in the project.",
    )
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    p.add_argument("--page-size", type=int, default=1000, help="W&B scan_history page size.")
    p.add_argument(
        "--include-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download full run history CSV/JSONL. Enabled by default.",
    )
    args = p.parse_args()

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb is not installed. Install with `pip install wandb`.", file=sys.stderr)
        return 2

    api = wandb.Api()
    path = _project_path(args.wandb_entity, args.wandb_project, api)
    print(f"[wandb] project path: {path}")

    runs_iter = api.runs(path)
    selected = []
    for run in runs_iter:
        if args.wandb_run_name is None:
            selected.append(run)
        elif run.name == args.wandb_run_name or run.id == args.wandb_run_name:
            selected.append(run)

    if not selected:
        msg = f"No runs found in {path}"
        if args.wandb_run_name:
            msg += f" matching --wandb-run-name {args.wandb_run_name!r}"
        print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for i, run in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {run.name} ({run.id})")
        stem = _safe_name(f"{run.name}__{run.id}")
        base = _run_base_row(run)
        config = _jsonable(dict(run.config or {}))
        summary = _jsonable(dict(run.summary or {}))

        _write_json(args.out_dir / "configs" / f"{stem}.config.json", config)
        _write_json(args.out_dir / "summaries" / f"{stem}.summary.json", summary)

        row = {
            **base,
            **{f"config/{k}": v for k, v in _flatten(config).items()},
            **{f"summary/{k}": v for k, v in _flatten(summary).items()},
        }

        history_jsonl = None
        history_csv = None
        history_rows = 0
        if args.include_history:
            history_jsonl, history_csv, history_rows = _download_history(
                run, args.out_dir, page_size=args.page_size
            )
            row["history_rows"] = history_rows
            row["history_jsonl"] = str(history_jsonl)
            row["history_csv"] = str(history_csv)

        summary_rows.append(row)
        manifest_rows.append({
            **base,
            "config_json": str(args.out_dir / "configs" / f"{stem}.config.json"),
            "summary_json": str(args.out_dir / "summaries" / f"{stem}.summary.json"),
            "history_jsonl": str(history_jsonl) if history_jsonl else None,
            "history_csv": str(history_csv) if history_csv else None,
            "history_rows": history_rows,
        })

    _write_jsonl(args.out_dir / "runs.jsonl", summary_rows)
    _write_csv(args.out_dir / "runs.csv", summary_rows)
    _write_json(args.out_dir / "manifest.json", {
        "project_path": path,
        "run_name_filter": args.wandb_run_name,
        "n_runs": len(selected),
        "runs": manifest_rows,
    })
    _write_csv(args.out_dir / "manifest.csv", manifest_rows)

    print(f"[wrote] {args.out_dir / 'runs.csv'}")
    print(f"[wrote] {args.out_dir / 'runs.jsonl'}")
    print(f"[wrote] {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
