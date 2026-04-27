"""ECAPA-TDNN voice embedding from SpeechBrain.

Fixed 192-dim embedding per utterance. Frozen at inference; if you want to
fine-tune for in-domain speakers, do so in a separate training script and
load a new checkpoint path.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


class VoiceEncoder(nn.Module):
    EMB_DIM = 192

    def __init__(self, model_id: str, stub: bool = False, device: str | torch.device = "cpu"):
        super().__init__()
        self.model_id = model_id
        self.stub = stub
        self.device = torch.device(device)
        self._model = None
        if not stub:
            self._load()

    def _load(self) -> None:
        from speechbrain.inference.speaker import EncoderClassifier
        self._model = EncoderClassifier.from_hparams(
            source=self.model_id,
            savedir=f".cache/speechbrain/{self.model_id.replace('/', '_')}",
            run_opts={"device": str(self.device)},
        )

    @torch.no_grad()
    def embed(self, wav: torch.Tensor | np.ndarray, sr: int = 16000) -> torch.Tensor:
        """Returns a single [192] vector (L2-normalised)."""
        if self.stub:
            v = torch.randn(self.EMB_DIM, device=self.device)
            return v / v.norm()

        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)  # [1, samples]
        wav = wav.to(self.device)
        emb = self._model.encode_batch(wav).squeeze(0).squeeze(0)  # [192]
        return emb / (emb.norm() + 1e-8)
