"""InsightFace ArcFace embedding.

Uses the 'buffalo_l' pack (ResNet-100 backbone), 512-dim output. Expects an
RGB image (H, W, 3) or a pre-cropped aligned face; we rely on InsightFace's
internal detector when given a raw frame.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


class FaceEncoder(nn.Module):
    EMB_DIM = 512

    def __init__(self, pack_name: str = "buffalo_l", stub: bool = False, device: str | torch.device = "cpu"):
        super().__init__()
        self.pack_name = pack_name
        self.stub = stub
        self.device = torch.device(device)
        self._app = None
        if not stub:
            self._load()

    def _load(self) -> None:
        from insightface.app import FaceAnalysis
        ctx_id = 0 if self.device.type == "cuda" else -1
        self._app = FaceAnalysis(name=self.pack_name)
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    @torch.no_grad()
    def embed(self, image: np.ndarray | torch.Tensor | None) -> torch.Tensor:
        """
        Args:
            image: H x W x 3 uint8 RGB (or a CHW tensor — we convert).
        Returns:
            [512] L2-normalised face embedding. If no face found, returns a
            zero vector so downstream cosine sim yields 0.
        """
        if image is None:
            return torch.zeros(self.EMB_DIM, device=self.device)

        if self.stub:
            v = torch.randn(self.EMB_DIM, device=self.device)
            return v / v.norm()

        if isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            image = image.detach().cpu().numpy()
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)

        faces = self._app.get(image[..., ::-1])  # BGR for InsightFace
        if not faces:
            return torch.zeros(self.EMB_DIM, device=self.device)
        # Largest detection wins.
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        emb = torch.from_numpy(faces[0].normed_embedding).to(self.device).float()
        return emb
