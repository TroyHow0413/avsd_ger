"""Versioned GER projector checkpoint metadata and compatibility checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import warnings

import torch


FORMAT_VERSION = 1


class CheckpointCompatibilityError(ValueError):
    pass


@dataclass(frozen=True)
class GERCheckpointMetadata:
    format_version: int
    model_family: str
    hidden_size: int
    tokenizer_class: str
    tokenizer_vocab_size: int
    speaker_special_token: str
    speaker_token_id: int
    lora_target_modules: tuple[str, ...]
    z_dim: int
    d_align: int
    n_queries: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["lora_target_modules"] = list(self.lora_target_modules)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GERCheckpointMetadata":
        copied = dict(value)
        copied["lora_target_modules"] = tuple(copied["lora_target_modules"])
        return cls(**copied)


def validate_metadata(
    saved: GERCheckpointMetadata, current: GERCheckpointMetadata
) -> None:
    fields = (
        "format_version", "model_family", "hidden_size", "tokenizer_class",
        "tokenizer_vocab_size", "speaker_special_token", "speaker_token_id",
        "lora_target_modules", "z_dim", "d_align", "n_queries",
    )
    differences = [
        f"{field}: checkpoint={getattr(saved, field)!r}, current={getattr(current, field)!r}"
        for field in fields
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
    allow_legacy: bool = True,
) -> None:
    state = torch.load(Path(path), map_location=map_location, weights_only=True)
    raw_metadata = state.get("metadata")
    if raw_metadata is None:
        if not allow_legacy:
            raise CheckpointCompatibilityError("GER checkpoint has no metadata")
        warnings.warn(
            "Loading legacy GER projector checkpoint without compatibility metadata",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        validate_metadata(
            GERCheckpointMetadata.from_dict(raw_metadata),
            head.checkpoint_metadata(),
        )
    head.qformer.load_state_dict(state["qformer"])
    head.id_proj.load_state_dict(state["id_proj"])
