from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.finalize_eval_batch import load_completed_runs


ABLATIONS = ("identity_normal", "zero_z_id", "shuffled_z_id")


def _write_report(root: Path, meeting: str, *, repo_relative_debug: bool = False) -> None:
    results = []
    for ablation in ABLATIONS:
        debug = root / f"{meeting}_debug" / f"{meeting}.{ablation}.debug.json"
        debug.parent.mkdir(parents=True, exist_ok=True)
        debug.write_text(
            json.dumps({"turns": [{"summary": {"turn_id": f"{meeting}.1"}}]}),
            encoding="utf-8",
        )
        debug_path = (
            Path("out") / root.name / f"{meeting}_debug" / debug.name
            if repo_relative_debug else debug
        )
        results.append({
            "ablation": ablation,
            "debug_path": str(debug_path),
            "metrics": {},
        })
    (root / f"{meeting}.json").write_text(
        json.dumps({"manifest": f"data/{meeting}.json", "results": results}),
        encoding="utf-8",
    )


def test_load_completed_runs_ignores_partial_summary(tmp_path: Path) -> None:
    _write_report(tmp_path, "ES2004a")
    _write_report(tmp_path, "TS3003d")
    (tmp_path / "summary.json").write_text(json.dumps({"runs": []}), encoding="utf-8")

    runs = load_completed_runs(tmp_path)

    assert [Path(run["manifest"]).stem for run in runs] == ["ES2004a", "TS3003d"]
    assert all(result["turn_debug"] for run in runs for result in run["results"])


def test_load_completed_runs_rejects_missing_ablation(tmp_path: Path) -> None:
    _write_report(tmp_path, "ES2004a")
    path = tmp_path / "ES2004a.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"].pop()
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="expected ablations"):
        load_completed_runs(tmp_path)


def test_load_completed_runs_rebases_repo_relative_debug_path(tmp_path: Path) -> None:
    _write_report(tmp_path, "ES2004a", repo_relative_debug=True)

    runs = load_completed_runs(tmp_path, repo_root=tmp_path / "different_checkout")

    assert len(runs) == 1
    assert all(result["turn_debug"] for result in runs[0]["results"])
