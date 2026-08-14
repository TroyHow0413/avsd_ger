"""Deterministic GER generation configuration and score extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class GenerationPolicy:
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "GenerationPolicy":
        generation = cfg.get("generation", {})
        return cls(
            max_new_tokens=int(
                generation.get("max_new_tokens", cfg.get("max_new_tokens", 64))
            ),
            do_sample=bool(generation.get("do_sample", False)),
            temperature=generation.get("temperature"),
        )

    def kwargs(self, pad_token_id: int) -> dict[str, Any]:
        result: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": pad_token_id,
        }
        if self.do_sample and self.temperature is not None:
            result["temperature"] = float(self.temperature)
        return result

    @staticmethod
    def generated_ids(output: Any) -> torch.Tensor:
        scores = getattr(output, "scores", None) or ()
        count = len(scores)
        sequence = output.sequences[0]
        return sequence[-count:] if count else sequence.new_empty((0,))

    @staticmethod
    def token_logprobs(output: Any, generated_ids: torch.Tensor) -> torch.Tensor:
        scores = getattr(output, "scores", None) or ()
        if not scores:
            return torch.zeros(0, device=generated_ids.device)
        logits = torch.stack(tuple(scores), dim=0).squeeze(1)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, generated_ids.unsqueeze(-1)).squeeze(-1)
