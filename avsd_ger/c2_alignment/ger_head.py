"""Identity-aware GER head composed from independent backend policies."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .checkpoint_metadata import (
    FORMAT_VERSION,
    GERCheckpointMetadata,
    load_projector_checkpoint,
    save_projector_checkpoint,
)
from .generation_policy import GenerationPolicy
from .model_backend import create_model_backend
from .prompt_builder import GERPromptBuilder
from .soft_token_bridge import QFormerProjector, SoftTokenQFormerBridge


class GERHead(nn.Module):
    """Orchestrate prompting, soft-token injection and causal-LM generation.

    The public ``generate`` contract and the legacy ``_tok``, ``_llm``,
    ``qformer`` and ``id_proj`` accessors are retained for existing training
    and evaluation scripts.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        z_dim: int,
        d_align: int,
        stub: bool = False,
        device: str | torch.device = "cpu",
        backend: Any | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        selected_backend = backend or create_model_backend(
            cfg, self.device, force_fake=stub
        )
        self.backend = selected_backend
        self.stub = bool(stub or selected_backend.kind == "fake")

        # Compatibility handles used by Stage-2 scripts.
        self._tok = selected_backend.tokenizer
        self._llm = selected_backend.model
        self._llm_embed_dim = int(selected_backend.hidden_size)
        self._spk_token_id = int(selected_backend.speaker_token_id)

        self.speaker_special_token = str(
            cfg.get("speaker_special_token", "[Speaker: ID_i]")
        )
        self.prompt_builder = GERPromptBuilder(cfg, self._tok)
        self.generation_policy = GenerationPolicy.from_config(cfg)
        self.max_new_tokens = self.generation_policy.max_new_tokens
        self.template = self.prompt_builder.template

        bridge_cfg = cfg.get("bridge", {})
        self._z_dim = int(z_dim)
        self._d_align = int(d_align)
        self.bridge = SoftTokenQFormerBridge(
            z_dim=z_dim,
            d_align=d_align,
            d_llm=self._llm_embed_dim,
            n_queries=int(bridge_cfg.get("n_queries", 16)),
            n_heads=int(bridge_cfg.get("n_heads", 8)),
            device=self.device,
        )

    @property
    def qformer(self) -> QFormerProjector:
        return self.bridge.qformer

    @property
    def id_proj(self) -> nn.Linear:
        return self.bridge.id_proj

    def _render_text(
        self,
        speaker_id: str | None,
        nbest: list[str],
        lip_hyp: str,
        mode: str = "av",
        use_av_context: bool = True,
    ) -> str:
        return self.prompt_builder.render(
            speaker_id,
            nbest,
            lip_hyp,
            mode=mode,
            use_av_context=use_av_context,
        )

    @staticmethod
    def _clean_generated_text(text: str) -> str:
        return GERPromptBuilder.clean_generated_text(text)

    def _inputs_embeds(
        self,
        z_id: torch.Tensor,
        f_align: torch.Tensor,
        text: str,
        use_av_context: bool = True,
    ) -> torch.Tensor:
        return self.bridge.build_inputs_embeds(
            text=text,
            tokenizer=self._tok,
            embedding_layer=self._llm.get_input_embeddings(),
            speaker_token_id=self._spk_token_id,
            z_id=z_id,
            f_align=f_align,
            device=self.device,
            use_av_context=use_av_context,
        )

    def checkpoint_metadata(self) -> GERCheckpointMetadata:
        return GERCheckpointMetadata(
            format_version=FORMAT_VERSION,
            model_family=self.backend.profile.family,
            hidden_size=self._llm_embed_dim,
            tokenizer_class=type(self._tok).__name__,
            tokenizer_vocab_size=len(self._tok),
            speaker_special_token=self.speaker_special_token,
            speaker_token_id=self._spk_token_id,
            lora_target_modules=tuple(self.backend.lora_target_modules),
            z_dim=self._z_dim,
            d_align=self._d_align,
            n_queries=self.bridge.n_queries,
        )

    def save_projector_checkpoint(self, path: str | Path) -> None:
        save_projector_checkpoint(path, self)

    def load_projector_checkpoint(
        self, path: str | Path, *, map_location: Any = None,
        allow_legacy: bool = True,
    ) -> None:
        load_projector_checkpoint(
            path,
            self,
            map_location=map_location,
            allow_legacy=allow_legacy,
        )

    @torch.no_grad()
    def generate(
        self,
        z_id: torch.Tensor,
        f_align: torch.Tensor,
        nbest: list[str],
        nbest_scores: list[float] | None = None,
        lip_hyp: str = "",
        speaker_id: str | None = None,
        mode: str = "av",
        use_av_context: bool = True,
    ) -> dict[str, Any]:
        del nbest_scores
        if self.stub:
            top = nbest[0] if nbest else ""
            return {
                "text": top,
                "raw_text": top,
                "token_logprobs": torch.zeros(0, device=self.device),
                "prompt": "",
            }

        prompt = self._render_text(
            speaker_id,
            nbest,
            lip_hyp,
            mode=mode,
            use_av_context=use_av_context,
        )
        inputs_embeds = self._inputs_embeds(
            z_id, f_align, prompt, use_av_context=use_av_context
        )
        output = self._llm.generate(
            inputs_embeds=inputs_embeds,
            **self.generation_policy.kwargs(self._tok.pad_token_id),
        )
        generated_ids = self.generation_policy.generated_ids(output)
        raw_text = self._tok.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        return {
            "text": self._clean_generated_text(raw_text),
            "raw_text": raw_text,
            "token_logprobs": self.generation_policy.token_logprobs(
                output, generated_ids
            ),
            "prompt": prompt,
        }


__all__ = ["GERHead", "QFormerProjector"]
