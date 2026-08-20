"""Deterministic GER generation configuration and score extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import copy

import torch


@dataclass(frozen=True)
class GenerationPolicy:
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "GenerationPolicy":
        generation = cfg.get("generation", {})
        return cls(
            max_new_tokens=int(
                generation.get("max_new_tokens", cfg.get("max_new_tokens", 64))
            ),
            do_sample=bool(generation.get("do_sample", False)),
            temperature=generation.get("temperature"),
            top_p=generation.get("top_p"),
            top_k=generation.get("top_k"),
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
        if self.do_sample and self.top_p is not None:
            result["top_p"] = float(self.top_p)
        if self.do_sample and self.top_k is not None:
            result["top_k"] = int(self.top_k)
        return result

    def generation_config(self, model: Any) -> Any | None:
        """Return a warning-free copy of the model generation defaults."""
        source = getattr(model, "generation_config", None)
        if source is None:
            return None
        cfg = copy.deepcopy(source)
        cfg.do_sample = self.do_sample
        if self.do_sample:
            if self.temperature is not None:
                cfg.temperature = float(self.temperature)
            if self.top_p is not None:
                cfg.top_p = float(self.top_p)
            if self.top_k is not None:
                cfg.top_k = int(self.top_k)
        else:
            cfg.temperature = None
            cfg.top_p = None
            cfg.top_k = None
        return cfg

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
