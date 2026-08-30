"""Versioned, provenance-checked training state for safe best/last resume."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


RUN_STATE_VERSION = 1


class RunStateCompatibilityError(ValueError):
    pass


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_file():
        return None
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_provenance(
    *,
    stage: str,
    cfg: dict[str, Any],
    train_manifest: str | Path,
    dev_manifest: str | Path | None,
    cache_signature: str | None = None,
) -> dict[str, Any]:
    return {
        "stage": str(stage),
        "config_sha256": canonical_digest(cfg),
        "train_manifest": str(Path(train_manifest).resolve()),
        "train_manifest_sha256": sha256_file(train_manifest),
        "dev_manifest": str(Path(dev_manifest).resolve()) if dev_manifest else None,
        "dev_manifest_sha256": sha256_file(dev_manifest),
        "cache_signature": cache_signature,
        "model_family": cfg.get("ger", {}).get("model_family"),
        "dataset_build_id": cfg.get("dataset_build_id", "ami_full_v4"),
    }


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_run_state(
    path: str | Path,
    *,
    provenance: dict[str, Any],
    epoch: int,
    global_step: int,
    modules: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    best_metric: float,
    best_epoch: int,
    extra: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": RUN_STATE_VERSION,
        "provenance": provenance,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "modules": {name: module.state_dict() for name, module in modules.items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "rng": capture_rng_state(),
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "extra": extra or {},
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_run_state(
    path: str | Path,
    *,
    expected_provenance: dict[str, Any],
    modules: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    source = Path(path)
    state = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(state, dict) or state.get("version") != RUN_STATE_VERSION:
        raise RunStateCompatibilityError(
            f"Unsupported training state version in {source}"
        )
    actual = state.get("provenance")
    if actual != expected_provenance:
        keys = sorted(set(actual or {}) | set(expected_provenance))
        differences = [
            f"{key}: checkpoint={(actual or {}).get(key)!r}, "
            f"current={expected_provenance.get(key)!r}"
            for key in keys
            if (actual or {}).get(key) != expected_provenance.get(key)
        ]
        raise RunStateCompatibilityError(
            "Training state provenance mismatch:\n- " + "\n- ".join(differences)
        )
    saved_modules = state.get("modules", {})
    if set(saved_modules) != set(modules):
        raise RunStateCompatibilityError(
            f"Training state modules {sorted(saved_modules)} do not match "
            f"current modules {sorted(modules)}"
        )
    for name, module in modules.items():
        module.load_state_dict(saved_modules[name], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng_state(state["rng"])
    return state
