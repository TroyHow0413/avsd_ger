import unittest
from types import SimpleNamespace

import numpy as np
import torch

from avsd_ger.c2_alignment.id_conditioned_aligner import SoftGatedCrossAttention
from avsd_ger.training.quality import resample_quality_track, token_snr_scores


class TrainingQualityTest(unittest.TestCase):
    def test_confidence_resampling_preserves_bounds_and_endpoints(self):
        out = resample_quality_track([0.0, 1.0], 5)

        self.assertEqual(tuple(out.shape), (5,))
        self.assertTrue(bool(((out >= 0) & (out <= 1)).all()))
        self.assertAlmostEqual(float(out[0]), 0.0)
        self.assertAlmostEqual(float(out[-1]), 1.0)

    def test_missing_confidence_becomes_explicit_zero(self):
        self.assertTrue(torch.equal(
            resample_quality_track(None, 3), torch.zeros(3)
        ))

    def test_snr_scores_follow_word_clock(self):
        wav = np.concatenate([
            np.zeros(8000, dtype=np.float32),
            np.ones(8000, dtype=np.float32) * 0.5,
        ])
        words = [
            SimpleNamespace(start=0.0, end=0.5),
            SimpleNamespace(start=0.5, end=1.0),
        ]

        scores = token_snr_scores(
            wav, words, 2, tau_snr_db=8.0, soft_scale_db=4.0
        )

        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(bool(torch.isfinite(scores).all()))
        self.assertGreater(float(scores[1]), float(scores[0]))

    def test_all_zero_gate_produces_exact_zero_visual_residual(self):
        torch.manual_seed(3)
        block = SoftGatedCrossAttention(d_model=8, n_heads=2, dropout=0.0)
        block.eval()
        out = block(
            torch.randn(3, 8),
            torch.randn(4, 8),
            soft_gate=torch.zeros(3, 4),
        )

        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_all_masked_keys_produce_zero_not_nan(self):
        block = SoftGatedCrossAttention(d_model=8, n_heads=2, dropout=0.0)
        out = block(
            torch.randn(2, 8),
            torch.randn(3, 8),
            key_padding_mask=torch.ones(3, dtype=torch.bool),
            soft_gate=torch.ones(2, 3),
        )

        self.assertTrue(torch.equal(out, torch.zeros_like(out)))
        self.assertTrue(bool(torch.isfinite(out).all()))


if __name__ == "__main__":
    unittest.main()
