"""Deterministic, explainable safety checks for generated correction text."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DEFAULT_ARTIFACT_BLACKLIST = (
    "Please provide", "I'm happy to help", "Here is the corrected transcript",
    "Audio hypothesis:", "Corrected transcript:", "transcript provided",
    "speaker label", "<|assistant|>", "<|user|>", "[INST]", "</s>",
)
DEFAULT_FILLERS = ("um", "uh", "erm", "hmm")


@dataclass(frozen=True)
class GateEvaluation:
    accepted: bool
    reason: str | None
    checks: tuple[dict[str, Any], ...]
    features: dict[str, Any]


@dataclass(frozen=True)
class GERSafetyGate:
    enabled: bool = True
    artifact_gate_enabled: bool = True
    length_gate_enabled: bool = True
    overlap_gate_enabled: bool = True
    acoustic_fallback_enabled: bool = True
    max_length_ratio: float = 1.8
    min_token_overlap: float = 0.50
    max_repeated_ngram_ratio: float = 0.35
    max_added_fillers: int = 0
    high_conf_asr_threshold: float = 0.85
    language_min_ascii_ratio: float = 0.8
    artifact_blacklist: tuple[str, ...] = tuple(
        item.casefold() for item in DEFAULT_ARTIFACT_BLACKLIST
    )
    fillers: tuple[str, ...] = DEFAULT_FILLERS

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
            acoustic_fallback_enabled=bool(cfg.get("enable_ger_acoustic_fallback", True)),
            max_length_ratio=float(cfg.get("ger_max_len_ratio", 1.8)),
            min_token_overlap=float(cfg.get("ger_min_token_overlap", 0.50)),
            max_repeated_ngram_ratio=float(cfg.get("ger_max_repeated_ngram_ratio", 0.35)),
            max_added_fillers=int(cfg.get("ger_filler_max_added", 0)),
            high_conf_asr_threshold=float(cfg.get("ger_high_conf_asr_threshold", 0.85)),
            language_min_ascii_ratio=float(cfg.get("ger_language_min_ascii_ratio", 0.8)),
            artifact_blacklist=tuple(str(item).casefold() for item in artifacts),
            fillers=tuple(str(item).casefold() for item in cfg.get("ger_fillers", DEFAULT_FILLERS)),
        )

    @staticmethod
    def normalize(text: str) -> str:
        text = unicodedata.normalize("NFKC", str(text)).casefold()
        text = re.sub(r"<\|[^|>]+\|>|\[(?:/?inst|speaker:[^]]+)\]|</?s>", " ", text)
        return " ".join(re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", text, re.UNICODE))

    @classmethod
    def tokens(cls, text: str) -> list[str]:
        normalized = cls.normalize(text)
        return normalized.split() if normalized else []

    @staticmethod
    def _repetition_ratio(tokens: list[str], n: int = 2) -> float:
        if len(tokens) < n:
            return 0.0
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        return 1.0 - len(set(grams)) / len(grams)

    def features(
        self, ger_text: str, asr_text: str, vsr_text: str = "",
        asr_confidence: float | None = None,
    ) -> dict[str, Any]:
        asr_tokens = self.tokens(asr_text)
        ger_tokens = self.tokens(ger_text)
        vsr_tokens = self.tokens(vsr_text)
        overlap = None
        if asr_tokens and ger_tokens:
            overlap = len(set(asr_tokens) & set(ger_tokens)) / len(set(asr_tokens))
        ger_lower = unicodedata.normalize("NFKC", str(ger_text)).casefold()
        asr_fillers = sum(asr_tokens.count(item) for item in self.fillers)
        ger_fillers = sum(ger_tokens.count(item) for item in self.fillers)
        letters = [char for char in ger_text if char.isalpha()]
        ascii_ratio = (
            sum(char.isascii() for char in letters) / len(letters) if letters else None
        )
        proper_names = re.findall(r"(?<!^)(?<![.!?]\s)(\b[A-Z][\w'-]+\b)", asr_text)
        return {
            "normalized_asr": self.normalize(asr_text),
            "normalized_vsr": self.normalize(vsr_text),
            "normalized_ger": self.normalize(ger_text),
            "asr_token_count": len(asr_tokens),
            "ger_token_count": len(ger_tokens),
            "length_ratio": len(ger_tokens) / len(asr_tokens) if asr_tokens else None,
            "token_overlap": overlap,
            "repeated_bigram_ratio": self._repetition_ratio(ger_tokens),
            "added_fillers": max(0, ger_fillers - asr_fillers),
            "artifact_hits": [item for item in self.artifact_blacklist if item and item in ger_lower],
            "missing_proper_names": [name for name in proper_names if name.casefold() not in ger_tokens],
            "ascii_letter_ratio": ascii_ratio,
            "asr_confidence": asr_confidence,
            "vsr_repeat_count": (
                " ".join(ger_tokens).count(" ".join(vsr_tokens))
                if len(vsr_tokens) >= 2 else 0
            ),
            "thresholds": {
                "max_len_ratio": self.max_length_ratio,
                "min_token_overlap": self.min_token_overlap,
                "max_repeated_ngram_ratio": self.max_repeated_ngram_ratio,
                "max_added_fillers": self.max_added_fillers,
                "high_conf_asr": self.high_conf_asr_threshold,
                "min_ascii_ratio": self.language_min_ascii_ratio,
            },
            # Flat aliases retain the earlier debug contract.
            "max_len_ratio": self.max_length_ratio,
            "min_token_overlap": self.min_token_overlap,
            "gate_enabled": self.enabled,
            "artifact_gate_enabled": self.artifact_gate_enabled,
            "length_gate_enabled": self.length_gate_enabled,
            "overlap_gate_enabled": self.overlap_gate_enabled,
            "acoustic_fallback_enabled": self.acoustic_fallback_enabled,
        }

    def evaluate(
        self, ger_text: str, asr_text: str, vsr_text: str = "",
        asr_confidence: float | None = None,
    ) -> GateEvaluation:
        features = self.features(ger_text, asr_text, vsr_text, asr_confidence)
        if not self.enabled:
            return GateEvaluation(True, None, ({"gate": "safety", "enabled": False, "passed": True},), features)
        checks: list[dict[str, Any]] = []

        def check(name: str, passed: bool, value: Any, threshold: Any, reason: str) -> str | None:
            checks.append({"gate": name, "input": value, "threshold": threshold, "passed": passed})
            return None if passed else reason

        ger_tokens = self.tokens(ger_text)
        asr_tokens = self.tokens(asr_text)
        candidates = [
            check("non_empty", bool(ger_tokens), len(ger_tokens), "> 0", "GER cleaned to empty text"),
            check("special_or_prompt_artifact", not self.artifact_gate_enabled or not features["artifact_hits"], features["artifact_hits"], "blacklist empty", "GER artifact matched blacklist or leaked a special token"),
        ]
        if asr_tokens and ger_tokens:
            candidates.extend([
                check("length", not self.length_gate_enabled or len(ger_tokens) <= max(len(asr_tokens) + 8, int(len(asr_tokens) * self.max_length_ratio)), features["length_ratio"], self.max_length_ratio, "GER too long"),
                check("overlap", not self.overlap_gate_enabled or float(features["token_overlap"]) >= self.min_token_overlap, features["token_overlap"], self.min_token_overlap, "GER/ASR token overlap too low"),
                check("repetition", features["repeated_bigram_ratio"] <= self.max_repeated_ngram_ratio, features["repeated_bigram_ratio"], self.max_repeated_ngram_ratio, "GER contains excessive repeated n-grams"),
                check("filler", features["added_fillers"] <= self.max_added_fillers, features["added_fillers"], self.max_added_fillers, "GER added unsupported filler"),
                check("proper_name", not features["missing_proper_names"], features["missing_proper_names"], "preserve ASR names", "GER removed or changed a probable person name"),
            ])
        if features["ascii_letter_ratio"] is not None and self.tokens(asr_text):
            candidates.append(check("language", features["ascii_letter_ratio"] >= self.language_min_ascii_ratio, features["ascii_letter_ratio"], self.language_min_ascii_ratio, "GER language/script switched unexpectedly"))
        if vsr_text:
            candidates.append(check("vsr_repetition", features["vsr_repeat_count"] < 2, features["vsr_repeat_count"], "< 2", "GER duplicated the VSR transcript"))
        changed = features["normalized_ger"] != features["normalized_asr"]
        added_content = bool(set(ger_tokens) - set(asr_tokens))
        if asr_confidence is not None and asr_confidence >= self.high_conf_asr_threshold:
            candidates.append(check("high_conf_asr", not (changed and added_content), {"changed": changed, "added_tokens": sorted(set(ger_tokens) - set(asr_tokens)), "confidence": asr_confidence}, self.high_conf_asr_threshold, "High-confidence ASR has no evidenced GER benefit"))

        reason = next((item for item in candidates if item is not None), None)
        return GateEvaluation(reason is None, reason, tuple(checks), features)

    def reject_reason(
        self, ger_text: str, asr_text: str, vsr_text: str = "",
        asr_confidence: float | None = None,
    ) -> str | None:
        return self.evaluate(ger_text, asr_text, vsr_text, asr_confidence).reason
