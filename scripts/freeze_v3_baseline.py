"""Freeze the two dense-backbone v3 evaluations without writing to the source.

The source tree is opened read-only.  Tensor payloads are never materialised
when checkpoint metadata is inspected: a restricted unpickler replaces tensor
rebuild operations with ``None`` and rejects every unrelated global.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import shutil
import zipfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASELINE_ID = "v3_before_phase_a_20260821"
EXPERIMENT_COMMIT = "60f86fc442e234fd4366c1a35ac77a8fd44151bd"
MODELS = {
    "qwen": {
        "eval": "out/eval_server_qwen3b_joint_e3_v3_av_all_ablations",
        "checkpoint": "checkpoints/server_qwen3b_av_joint_e3_v3",
        "wandb": (
            "run-20260821_024647-npes8vhk",
            "run-20260821_070504-4vpt97bj",
        ),
    },
    "llama": {
        "eval": "out/eval_server_llama32_3b_joint_e3_v3_av_all_ablations",
        "checkpoint": "checkpoints/server_llama32_3b_av_joint_e3_v3",
        "wandb": (
            "run-20260821_085930-e9o5qudm",
            "run-20260821_131346-mt94hrxl",
        ),
    },
}
WANDB_FILES = ("config.yaml", "wandb-summary.json", "wandb-metadata.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path, *, exclude: set[str] | None = None) -> dict[str, Any]:
    excluded = exclude or set()
    files: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel in excluded:
            continue
        files.append({"path": rel, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    canonical = "\n".join(
        f"{item['path']}\t{item['bytes']}\t{item['sha256']}" for item in files
    ).encode("utf-8")
    return {
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def _inventory_summary(files: list[dict[str, Any]]) -> dict[str, Any]:
    canonical = "\n".join(
        f"{item['path']}\t{item['bytes']}\t{item['sha256']}" for item in files
    ).encode("utf-8")
    return {
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
    }


class _MetadataOnlyUnpickler(pickle.Unpickler):
    """Unpickle plain metadata while refusing executable checkpoint globals."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name.startswith("_rebuild"):
            return lambda *args, **kwargs: None
        if module == "torch" and name.endswith("Storage"):
            return type(name, (), {})
        raise pickle.UnpicklingError(f"blocked checkpoint global: {module}.{name}")

    def persistent_load(self, pid: Any) -> None:
        return None


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            member = next(name for name in archive.namelist() if name.endswith("data.pkl"))
            payload = _MetadataOnlyUnpickler(io.BytesIO(archive.read(member))).load()
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        if not isinstance(metadata, dict):
            return {"status": "missing"}
        return {"status": "extracted", "value": metadata}
    except (OSError, KeyError, StopIteration, zipfile.BadZipFile, pickle.UnpicklingError) as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def freeze(source_root: Path, destination: Path, compact_output: Path) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    destination = destination.resolve()
    compact_output = compact_output.resolve()
    project_root = Path(__file__).resolve().parents[1]
    allowed_root = (project_root / "out" / "baselines").resolve()
    if destination == source_root or source_root in destination.parents:
        raise ValueError("destination must not be inside the read-only source tree")
    if compact_output == source_root or source_root in compact_output.parents:
        raise ValueError("provenance output must not be inside the read-only source tree")
    if allowed_root not in destination.parents:
        raise ValueError(f"destination must be below {allowed_root}")
    if project_root not in compact_output.parents:
        raise ValueError(f"provenance output must be below {project_root}")

    destination.mkdir(parents=True, exist_ok=True)
    manifest_paths: set[Path] = set()
    for model, entry in MODELS.items():
        eval_source = source_root / entry["eval"]
        if not eval_source.is_dir():
            raise FileNotFoundError(eval_source)
        shutil.copytree(eval_source, destination / "eval" / model, dirs_exist_ok=True)
        summary = json.loads((eval_source / "summary.json").read_text(encoding="utf-8"))
        for run in summary.get("runs", []):
            manifest_paths.add(source_root / str(run["manifest"]))
        for run_id in entry["wandb"]:
            for filename in WANDB_FILES:
                source = source_root / "wandb" / run_id / "files" / filename
                if not source.is_file():
                    raise FileNotFoundError(source)
                _copy_file(source, destination / "wandb" / run_id / filename)

    for source in sorted(manifest_paths):
        if not source.is_file():
            raise FileNotFoundError(source)
        _copy_file(source, destination / "manifests" / source.name)

    checkpoints: dict[str, Any] = {}
    for model, entry in MODELS.items():
        checkpoint_root = source_root / entry["checkpoint"]
        inventory = _inventory(checkpoint_root)
        projector = checkpoint_root / "ger" / "ger_projectors.pt"
        checkpoints[model] = {
            "source": str(checkpoint_root),
            **inventory,
            "metadata": _checkpoint_metadata(projector),
        }

    full_name = "provenance.full.json"
    snapshot = _inventory(destination, exclude={full_name})
    full = {
        "schema_version": 1,
        "baseline_id": BASELINE_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_git_commit": EXPERIMENT_COMMIT,
        "source_root": str(source_root),
        "source_access": "read_only",
        "snapshot_root": str(destination),
        "snapshot": snapshot,
        "checkpoints": checkpoints,
    }
    (destination / full_name).write_text(
        json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    compact = {
        "schema_version": 1,
        "baseline_id": BASELINE_ID,
        "experiment_git_commit": EXPERIMENT_COMMIT,
        "source_access": "read_only",
        "snapshot": {
            key: snapshot[key] for key in ("file_count", "total_bytes", "inventory_sha256")
        },
        "snapshot_manifest": f"out/baselines/{BASELINE_ID}/{full_name}",
        "manifests": [
            item for item in snapshot["files"] if item["path"].startswith("manifests/")
        ],
        "wandb_metadata": [
            item for item in snapshot["files"] if item["path"].startswith("wandb/")
        ],
        "eval_artifacts": {
            model: _inventory_summary([
                item for item in snapshot["files"]
                if item["path"].startswith(f"eval/{model}/")
            ])
            for model in MODELS
        },
        "models": {
            model: {
                "checkpoint_source": value["source"],
                "checkpoint_file_count": value["file_count"],
                "checkpoint_total_bytes": value["total_bytes"],
                "checkpoint_files": value["files"],
                "checkpoint_metadata": value["metadata"],
            }
            for model, value in checkpoints.items()
        },
        "notes": [
            "Raw eval/debug artifacts are intentionally ignored by Git.",
            "Historical c3_wo_conf_gate means update-gate-only, not all C3 confidence gates.",
            "Qwen and Llama used per-model Stage-3 settings; this is not a strictly controlled backbone comparison.",
        ],
    }
    compact_output.parent.mkdir(parents=True, exist_ok=True)
    compact_output.write_text(
        json.dumps(compact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return compact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--provenance-out", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.source_root, args.destination, args.provenance_out)
    print(json.dumps(result["snapshot"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
