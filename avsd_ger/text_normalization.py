"""Language-aware canonical text normalization for evaluation and safety audits."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable


NORMALIZER_VERSION = "whisper-canonical-v1"


class LanguageResolutionError(ValueError):
    """Raised when auto language was requested but no detector result exists."""


def resolve_language(requested: str | None, detected: str | None = None) -> str:
    value = (requested or "auto").strip().lower().replace("_", "-")
    if value != "auto":
        return value
    detected_value = (detected or "").strip().lower().replace("_", "-")
    if detected_value:
        return detected_value
    raise LanguageResolutionError(
        "language=auto requires ASR detected_language metadata; this legacy result "
        "does not contain it. Re-run with --language en/zh/<ISO language>."
    )


@lru_cache(maxsize=32)
def _whisper_normalizer(language: str) -> Callable[[str], str]:
    try:
        from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
    except ImportError as exc:
        raise RuntimeError(
            "Canonical evaluation requires openai-whisper (whisper.normalizers)."
        ) from exc
    primary = language.split("-", 1)[0]
    if primary == "en":
        return EnglishTextNormalizer()
    return BasicTextNormalizer(remove_diacritics=False, split_letters=False)


def normalize_text(
    text: str | None,
    *,
    language: str | None,
    detected_language: str | None = None,
) -> str:
    resolved = resolve_language(language, detected_language)
    return " ".join(_whisper_normalizer(resolved)(text or "").strip().split())

