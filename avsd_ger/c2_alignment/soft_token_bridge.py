"""Identity and aligned-feature bridge into causal-LM embedding space."""
from __future__ import annotations

import torch
import torch.nn as nn


def _tokenizer_mask(encoded, input_ids: torch.Tensor) -> torch.Tensor:
    mask = getattr(encoded, "attention_mask", None)
    if mask is None:
        return torch.ones_like(input_ids, dtype=torch.long)
    return mask.to(device=input_ids.device, dtype=torch.long)


class QFormerProjector(nn.Module):
    def __init__(self, d_in: int, d_llm: int, n_queries: int = 16, n_heads: int = 8):
        super().__init__()
        if d_in % n_heads:
            raise ValueError(f"d_align={d_in} must be divisible by qformer n_heads={n_heads}")
        self.queries = nn.Parameter(torch.randn(n_queries, d_in) * 0.02)
        self.attn = nn.MultiheadAttention(d_in, n_heads, batch_first=True)
        self.out = nn.Linear(d_in, d_llm)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            features = features.unsqueeze(0)
        queries = self.queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        values, _ = self.attn(queries, features, features, need_weights=False)
        return self.out(values)


class SoftTokenQFormerBridge(nn.Module):
    def __init__(
        self,
        *,
        z_dim: int,
        d_align: int,
        d_llm: int,
        n_queries: int = 16,
        n_heads: int = 8,
        device: torch.device,
    ):
        super().__init__()
        self.d_llm = d_llm
        self.n_queries = n_queries
        self.n_heads = n_heads
        self.id_proj = nn.Linear(z_dim, d_llm)
        self.qformer = QFormerProjector(d_align, d_llm, n_queries, n_heads)
        self.to(device)

    def build_inputs_embeds(
        self,
        *,
        text: str,
        tokenizer,
        embedding_layer: nn.Module,
        speaker_token_id: int | None,
        z_id: torch.Tensor,
        f_align: torch.Tensor,
        device: torch.device,
        use_av_context: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_av_context:
            if "<AV_CTX>" not in text:
                raise ValueError("Rendered AV prompt must contain <AV_CTX>")
            pre, post = text.split("<AV_CTX>", 1)
        else:
            pre, post = text, ""
        pre_encoded = tokenizer(
            pre,
            return_tensors="pt",
            # Instruct chat templates already emit their model-specific BOS.
            add_special_tokens=not bool(getattr(tokenizer, "chat_template", None)),
        )
        pre_ids = pre_encoded.input_ids.to(device).long()
        pre_mask = _tokenizer_mask(pre_encoded, pre_ids).to(device)
        if post:
            post_encoded = tokenizer(post, return_tensors="pt", add_special_tokens=False)
            post_ids = post_encoded.input_ids.to(device).long()
            post_mask = _tokenizer_mask(post_encoded, post_ids).to(device)
        else:
            batch_size = pre_ids.shape[0]
            post_ids = torch.empty(batch_size, 0, dtype=torch.long, device=device)
            post_mask = torch.empty(batch_size, 0, dtype=torch.long, device=device)
        pre_emb = embedding_layer(pre_ids)
        post_emb = embedding_layer(post_ids)
        target_dtype = pre_emb.dtype
        self.to(device=device, dtype=target_dtype)

        if speaker_token_id is not None:
            identity = z_id.to(device=device, dtype=target_dtype)
            if identity.ndim == 1:
                identity = identity.unsqueeze(0)
            if identity.shape[0] == 1 and pre_ids.shape[0] > 1:
                identity = identity.expand(pre_ids.shape[0], -1)
            if identity.shape[0] != pre_ids.shape[0]:
                raise ValueError("z_id batch size must match tokenized prompt batch size")
            bias = self.id_proj(identity)
            pre_emb = pre_emb.clone()
            for row in range(pre_ids.shape[0]):
                positions = (pre_ids[row] == speaker_token_id).nonzero(as_tuple=True)[0]
                if positions.numel():
                    pre_emb[row, positions[0]] = pre_emb[row, positions[0]] + bias[row]

        pieces = [pre_emb]
        masks = [pre_mask]
        if use_av_context:
            aligned = f_align.to(device=device, dtype=target_dtype)
            projected = self.qformer(aligned)
            pieces.append(projected)
            masks.append(torch.ones(
                projected.shape[0], projected.shape[1], dtype=torch.long, device=device
            ))
        pieces.append(post_emb)
        masks.append(post_mask)
        inputs_embeds = torch.cat(pieces, dim=1)
        attention_mask = torch.cat(masks, dim=1)
        if inputs_embeds.shape[:2] != attention_mask.shape:
            raise RuntimeError("bridge attention mask is not aligned with inputs_embeds")
        return inputs_embeds, attention_mask
