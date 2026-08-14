"""Deterministic safety checks for generated error-correction text."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DEFAULT_ARTIFACT_BLACKLIST = (
    "Please provide",
    "I'm happy to help",
    "Here is the corrected transcript",
    "Audio hypothesis:",
    "Corrected transcript:",
    "transcript provided",
    "speaker label",
)


@dataclass(frozen=True)
class GERSafetyGate:
    """Reject obviously unsafe GER output before acoustic rescoring.

    Keeping this policy independent of the model pipeline makes the thresholds
    auditable and allows the text-only behavior to be tested without loading
    any neural-network backbones.
    """

    enabled: bool = True
    artifact_gate_enabled: bool = True
    length_gate_enabled: bool = True
    overlap_gate_enabled: bool = True
    acoustic_fallback_enabled: bool = True
    max_length_ratio: float = 1.8
    min_token_overlap: float = 0.50
    artifact_blacklist: tuple[str, ...] = tuple(
        item.lower() for item in DEFAULT_ARTIFACT_BLACKLIST
    )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "GERSafetyGate":
        artifacts: Sequence[Any] = cfg.get(
            "ger_artifact_blacklist", DEFAULT_ARTIFACT_BLACKLIST
        )
        return cls(
            enabled=bool(cfg.get("enable_ger_safety_gate", True)),
            artifact_gate_enabled=bool(cfg.get("enable_ger_artifact_gate", True)),
            length_gate_enabled=bool(cfg.get("enable_ger_length_gate", True)),
            overlap_gate_enabled=bool(cfg.get("enable_ger_overlap_gate", True)),
            acoustic_fallback_enabled=bool(
                cfg.get("enable_ger_acoustic_fallback", True)
            ),
            max_length_ratio=float(cfg.get("ger_max_len_ratio", 1.8)),
            min_token_overlap=float(cfg.get("ger_min_token_overlap", 0.50)),
            artifact_blacklist=tuple(str(item).lower() for item in artifacts),
        )

    @staticmethod
    def tokens(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9']+", text.lower())

    def reject_reason(self, ger_text: str, asr_text: str) -> str | None:
        if not self.enabled:
            return None

        ger = ger_text.strip()
        if not ger:
            return "GER cleaned to empty text"

        if self.artifact_gate_enabled:
            ger_lower = ger.lower()
            for artifact in self.artifact_blacklist:
                if artifact and artifact in ger_lower:
                    return f"GER artifact matched blacklist: {artifact!r}"

        asr_tokens = self.tokens(asr_text)
        ger_tokens = self.tokens(ger)
        if not asr_tokens or not ger_tokens:
            return None

        if self.length_gate_enabled and len(ger_tokens) > max(
            len(asr_tokens) + 8,
            int(len(asr_tokens) * self.max_length_ratio),
        ):
            return (
                "GER too long "
                f"({len(ger_tokens)} vs ASR {len(asr_tokens)} tokens)"
            )

        overlap = len(set(asr_tokens) & set(ger_tokens)) / len(set(asr_tokens))
        if self.overlap_gate_enabled and overlap < self.min_token_overlap:
            return (
                "GER/ASR token overlap too low "
                f"({overlap:.2f} < {self.min_token_overlap:.2f})"
            )
        return None

    def features(self, ger_text: str, asr_text: str) -> dict[str, Any]:
        asr_tokens = self.tokens(asr_text)
        ger_tokens = self.tokens(ger_text)
        overlap = None
        if asr_tokens and ger_tokens:
            overlap = len(set(asr_tokens) & set(ger_tokens)) / len(set(asr_tokens))
        ger_lower = ger_text.lower()
        return {
            "asr_token_count": len(asr_tokens),
            "ger_token_count": len(ger_tokens),
            "length_ratio": (
                len(ger_tokens) / len(asr_tokens) if asr_tokens else None
            ),
            "token_overlap": overlap,
            "artifact_hits": [
                artifact
                for artifact in self.artifact_blacklist
                if artifact and artifact in ger_lower
            ],
            "gate_enabled": self.enabled,
            "artifact_gate_enabled": self.artifact_gate_enabled,
            "length_gate_enabled": self.length_gate_enabled,
            "overlap_gate_enabled": self.overlap_gate_enabled,
            "acoustic_fallback_enabled": self.acoustic_fallback_enabled,
            "max_len_ratio": self.max_length_ratio,
            "min_token_overlap": self.min_token_overlap,
        }
