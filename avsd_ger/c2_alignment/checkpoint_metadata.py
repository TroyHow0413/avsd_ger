"""Versioned GER projector checkpoint metadata and strict compatibility checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any
import warnings

import torch


FORMAT_VERSION = 2


class CheckpointCompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class GERCheckpointMetadata:
    format_version: int
    model_identifier: str
    model_family: str
    model_type: str
    hidden_size: int
    model_config_fingerprint: str
    tokenizer_class: str
    tokenizer_vocab_size: int
    tokenizer_fingerprint: str
    chat_template_fingerprint: str
    speaker_special_token: str
    speaker_token_id: int
    prompt_template_fingerprint: str
    z_dim: int
    d_align: int
    n_queries: int
    n_heads: int
    lora_target_modules: tuple[str, ...]
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    lora_bias: str
    torch_version: str
    transformers_version: str
    peft_version: str
    git_commit: str | None
    training_hyperparameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["lora_target_modules"] = list(self.lora_target_modules)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GERCheckpointMetadata":
        if not isinstance(value, dict):
            raise CheckpointCompatibilityError("GER checkpoint metadata is not a mapping")
        expected = {item.name for item in fields(cls)}
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        if missing or unknown:
            raise CheckpointCompatibilityError(
                f"Malformed GER checkpoint metadata; missing={missing}, unknown={unknown}"
            )
        copied = dict(value)
        targets = copied.get("lora_target_modules")
        if not isinstance(targets, (list, tuple)):
            raise CheckpointCompatibilityError("lora_target_modules must be a list")
        copied["lora_target_modules"] = tuple(str(item) for item in targets)
        try:
            return cls(**copied)
        except (TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError(
                f"Malformed GER checkpoint metadata: {exc}"
            ) from exc


def validate_metadata(
    saved: GERCheckpointMetadata, current: GERCheckpointMetadata
) -> None:
    # Runtime/library versions and git commit are provenance, not shape or
    # semantic compatibility constraints. Training hyperparameters remain
    # auditable but do not prevent evaluation.
    informational = {
        "torch_version", "transformers_version", "peft_version", "git_commit",
        "training_hyperparameters",
    }
    comparable = [item.name for item in fields(GERCheckpointMetadata)
                  if item.name not in informational]
    differences = [
        f"{field}: checkpoint={getattr(saved, field)!r}, current={getattr(current, field)!r}"
        for field in comparable
        if getattr(saved, field) != getattr(current, field)
    ]
    if differences:
        raise CheckpointCompatibilityError(
            "Incompatible GER checkpoint metadata:\n- " + "\n- ".join(differences)
        )


def save_projector_checkpoint(path: str | Path, head: Any) -> None:
    torch.save(
        {
            "metadata": head.checkpoint_metadata().to_dict(),
            "qformer": head.qformer.state_dict(),
            "id_proj": head.id_proj.state_dict(),
        },
        Path(path),
    )


def load_projector_checkpoint(
    path: str | Path, head: Any, *, map_location: Any = None,
    allow_legacy: bool = False,
) -> None:
    try:
        state = torch.load(Path(path), map_location=map_location, weights_only=True)
    except Exception as exc:
        raise CheckpointCompatibilityError(f"Cannot read GER checkpoint: {exc}") from exc
    if not isinstance(state, dict):
        raise CheckpointCompatibilityError("GER checkpoint root is not a mapping")
    raw_metadata = state.get("metadata")
    if raw_metadata is None:
        if not allow_legacy:
            raise CheckpointCompatibilityError("GER checkpoint has no metadata")
        warnings.warn(
            "UNSAFE LEGACY OPT-IN: loading GER projector checkpoint without "
            "compatibility metadata",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        validate_metadata(
            GERCheckpointMetadata.from_dict(raw_metadata),
            head.checkpoint_metadata(),
        )
    for key in ("qformer", "id_proj"):
        if key not in state or not isinstance(state[key], dict):
            raise CheckpointCompatibilityError(f"GER checkpoint is missing {key!r} state")
    try:
        head.qformer.load_state_dict(state["qformer"], strict=True)
        head.id_proj.load_state_dict(state["id_proj"], strict=True)
    except (RuntimeError, KeyError, TypeError) as exc:
        raise CheckpointCompatibilityError(
            f"GER checkpoint parameter state is incompatible: {exc}"
        ) from exc
