"""Finalize an interrupted/completed eval batch from meeting reports and debug files.

Use this when ``eval_ablations.py --no-formal-artifacts`` produced one JSON
report per meeting.  The script discovers all completed meeting reports,
reattaches their debug turn rows, verifies batch completeness, and writes the
same formal artifact tree that an uninterrupted evaluation would have written.
It never loads model checkpoints or media.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avsd_ger.eval.formal_artifacts import write_formal_artifacts  # noqa: E402
from avsd_ger.utils import load_config  # noqa: E402


IDENTITY_ABLATIONS = {"identity_normal", "zero_z_id", "shuffled_z_id"}


def _resolve_debug_path(repo_root: Path, artifact_root: Path, raw: str) -> Path:
    candidate = Path(raw)
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.extend((repo_root / candidate, artifact_root / candidate))
        # Reports often retain a repo-relative path such as
        # out/<artifact-name>/<meeting>_debug/....  Allow the artifact tree to
        # be moved to another checkout/drive by rebasing the suffix after its
        # own directory name.
        parts = candidate.parts
        if artifact_root.name in parts:
            index = parts.index(artifact_root.name)
            candidates.append(artifact_root.joinpath(*parts[index + 1 :]))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Could not resolve debug sidecar {raw!r}; tried "
        + ", ".join(str(path) for path in candidates)
    )


def load_completed_runs(
    artifact_root: Path,
    *,
    repo_root: Path = ROOT,
    expected_ablations: set[str] = IDENTITY_ABLATIONS,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    seen_meetings: set[str] = set()
    for path in sorted(artifact_root.glob("*.json")):
        if path.name in {"summary.json", "run_manifest.json", "records.schema.json"}:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("results"), list) or not payload.get("manifest"):
            continue
        meeting = Path(str(payload["manifest"])).stem
        if meeting in seen_meetings:
            raise ValueError(f"Duplicate meeting report for {meeting}: {path}")
        seen_meetings.add(meeting)
        names = {str(result.get("ablation")) for result in payload["results"]}
        if names != expected_ablations:
            raise ValueError(
                f"{path}: expected ablations {sorted(expected_ablations)}, got {sorted(names)}"
            )
        results: list[dict[str, Any]] = []
        for result in payload["results"]:
            result = dict(result)
            debug_path = result.get("debug_path")
            if not debug_path:
                raise ValueError(f"{path}: {result.get('ablation')} has no debug_path")
            resolved = _resolve_debug_path(repo_root, artifact_root, str(debug_path))
            debug = json.loads(resolved.read_text(encoding="utf-8"))
            turns = debug.get("turns") or []
            if not turns:
                raise ValueError(f"{resolved}: no turn debug rows")
            result["turn_debug"] = turns
            results.append(result)
        runs.append({"manifest": str(payload["manifest"]), "results": results})
    if not runs:
        raise FileNotFoundError(f"No completed meeting reports found under {artifact_root}")
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pool", default=None)
    parser.add_argument("--aligner-ckpt", default=None)
    parser.add_argument("--ger-ckpt", default=None)
    parser.add_argument("--expected-meetings", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--samples", type=int, default=10_000)
    args = parser.parse_args()

    source = args.artifact_root.resolve()
    destination = args.out_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    runs = load_completed_runs(source)
    meetings = sorted(Path(str(run["manifest"])).stem for run in runs)
    if len(meetings) != args.expected_meetings:
        raise RuntimeError(
            f"Incomplete batch: expected {args.expected_meetings} meetings, "
            f"found {len(meetings)}: {meetings}"
        )

    cfg = load_config(args.config)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "summary.json").write_text(
        json.dumps({"n_manifests": len(runs), "runs": runs}, indent=2),
        encoding="utf-8",
    )
    written = write_formal_artifacts(
        destination,
        runs,
        repo_root=ROOT,
        config_path=args.config,
        config=cfg,
        pool_path=args.pool,
        aligner_ckpt=args.aligner_ckpt,
        ger_ckpt=args.ger_ckpt,
        seed=args.seed,
        started_at=datetime.now(timezone.utc).isoformat(),
        bootstrap_samples=args.samples,
        bootstrap_seed=args.seed,
    )
    print(f"[finalized] meetings={len(meetings)} root={written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
