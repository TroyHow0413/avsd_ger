"""C1 — Agglomerative cold-start clustering (spec §8 step 3).

Before any speaker is explicitly enrolled, we still need to discover the
speaker set from an unlabelled pool of utterances. The spec prescribes:

    * Agglomerative (average-linkage) clustering of fused identity
      vectors in cosine space.
    * A **data-driven K** via ``distance_threshold=δ`` rather than fixed
      ``n_clusters`` — this lets the recording dictate how many speakers
      it contains.
    * Samples whose nearest-cluster distance exceeds ``delta_unknown``
      are assigned label = ``None`` (the "unknown" bucket), feeding the
      open-set branch of the pool.

This module is intentionally input-agnostic: it receives pre-computed
fused embeddings (from :class:`IdentityFuser`) and returns cluster
assignments + centroids. Wiring it to a session loader, passing frames
through the dual gate, and only then clustering is the caller's job
(see ``scripts/train_identity.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class ColdStartResult:
    """Output of a cold-start clustering pass.

    Attributes:
        labels: [N] int, one per input embedding. ``-1`` means 'unknown'
                (nearest cluster distance > delta_unknown).
        centroids: [K, D] float32 — L2-normalised cluster means.
        cluster_sizes: [K] int.
        n_unknown: int — count of samples labelled ``-1``.
    """
    labels: np.ndarray
    centroids: np.ndarray
    cluster_sizes: np.ndarray
    n_unknown: int


class AgglomerativeColdStart:
    """Cold-start clustering with native unknown detection.

    Uses sklearn's ``AgglomerativeClustering`` with ``distance_threshold``
    to auto-choose K. After clustering, each point's distance to its
    assigned centroid is checked against ``delta_unknown``; anything
    farther is relabelled as ``-1``.

    This two-stage design (cluster first, then unknown-detect) matches the
    spec's separation of concerns: clustering captures structure in the
    "known" region; the unknown threshold is a separate knob for open-set.
    """

    def __init__(self, cfg: dict[str, Any]):
        cs = cfg.get("cold_start", cfg)
        self.linkage = str(cs.get("linkage", "average"))
        self.distance_threshold = float(cs["distance_threshold"])
        self.delta_unknown = float(cs.get("delta_unknown", self.distance_threshold + 0.1))

    # ---------------------------------------------------------------- fit
    def fit(self, embeddings: torch.Tensor | np.ndarray) -> ColdStartResult:
        """Cluster N fused identity embeddings.

        Args:
            embeddings: [N, D] — ideally L2-normalised so the cosine
                distance approximation (1 - cos) behaves as expected.

        Returns:
            ColdStartResult with labels (``-1`` for unknown), centroids,
            cluster sizes, and unknown count.
        """
        X = _to_np_2d(embeddings)
        X = _l2_normalize(X)
        n = X.shape[0]
        if n == 0:
            return ColdStartResult(
                labels=np.zeros(0, dtype=np.int64),
                centroids=np.zeros((0, X.shape[1] if X.ndim == 2 else 0), dtype=np.float32),
                cluster_sizes=np.zeros(0, dtype=np.int64),
                n_unknown=0,
            )
        if n == 1:
            return ColdStartResult(
                labels=np.zeros(1, dtype=np.int64),
                centroids=X.copy(),
                cluster_sizes=np.ones(1, dtype=np.int64),
                n_unknown=0,
            )

        labels = self._cluster(X)
        centroids, sizes = self._centroids(X, labels)

        # Unknown gate: distance to assigned centroid
        dists = np.empty(n, dtype=np.float32)
        for i in range(n):
            c = centroids[labels[i]]
            dists[i] = 1.0 - float(np.dot(X[i], c) / (np.linalg.norm(c) + 1e-8))
        unknown = dists > self.delta_unknown
        labels[unknown] = -1
        n_unknown = int(unknown.sum())

        # Re-derive centroids excluding unknowns so they stay clean.
        known = labels != -1
        if known.any():
            centroids, sizes = self._centroids(X[known], labels[known])
        return ColdStartResult(
            labels=labels,
            centroids=centroids.astype(np.float32),
            cluster_sizes=sizes.astype(np.int64),
            n_unknown=n_unknown,
        )

    # ---------------------------------------------------------------- internals
    def _cluster(self, X: np.ndarray) -> np.ndarray:
        """Run agglomerative clustering with distance_threshold → auto-K."""
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            # Graceful fallback for environments without sklearn: everyone
            # in the same cluster. Keeps tests runnable.
            return np.zeros(X.shape[0], dtype=np.int64)

        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            linkage=self.linkage,
            metric="cosine" if self.linkage != "ward" else "euclidean",
        )
        return model.fit_predict(X).astype(np.int64)

    @staticmethod
    def _centroids(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute L2-normalised per-cluster centroids and sizes.

        Labels are assumed to be contiguous non-negative ints starting at 0;
        this method is only called after the unknown gate (or before, when
        all labels are known).
        """
        uniq = np.unique(labels)
        # Re-number labels to be contiguous 0..K-1 for downstream convenience.
        remap = {int(u): i for i, u in enumerate(uniq)}
        K = len(uniq)
        D = X.shape[1]
        centroids = np.zeros((K, D), dtype=np.float32)
        sizes = np.zeros(K, dtype=np.int64)
        for i, row in enumerate(X):
            k = remap[int(labels[i])]
            centroids[k] += row
            sizes[k] += 1
        for k in range(K):
            if sizes[k] > 0:
                centroids[k] /= sizes[k]
        centroids = _l2_normalize(centroids)
        return centroids, sizes


# ------------------------------------------------------------------ helpers
def _to_np_2d(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)
