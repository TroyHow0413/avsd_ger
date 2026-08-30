import unittest

import torch

from avsd_ger.training.ctc_loss import CTCHead


class CTCTrainingTest(unittest.TestCase):
    def test_default_expansion_accepts_more_than_sixteen_steps_per_token(self):
        head = CTCHead(d_align=8)
        features = torch.randn(1, 8, requires_grad=True)

        report = head(features, targets=["abcdefghijklmnopqrst"])

        self.assertEqual(report.expansion, 20)
        self.assertTrue(torch.isfinite(report.loss))
        report.loss.backward()
        self.assertIsNotNone(features.grad)

    def test_minimum_steps_counts_adjacent_repeats(self):
        head = CTCHead(d_align=8)
        ids = head.vocab.encode("book")

        self.assertEqual(head.minimum_ctc_steps(ids), len(ids) + 1)

    def test_learned_subframes_are_not_identical_and_receive_gradients(self):
        torch.manual_seed(7)
        head = CTCHead(d_align=8, min_expansion=2, max_expansion=8)
        aligned = torch.randn(3, 8, requires_grad=True)

        report = head(aligned, targets=["hello"])
        report.loss.backward()

        self.assertTrue(report.feasible)
        self.assertGreaterEqual(report.input_lengths.item(), report.minimum_steps.item())
        self.assertTrue(torch.isfinite(report.loss))
        self.assertNotEqual(float(report.loss.detach()), 0.0)
        self.assertIsNotNone(aligned.grad)
        self.assertGreater(float(aligned.grad.norm()), 0.0)
        self.assertGreater(float(head.temporal_expand[1].weight.grad.norm()), 0.0)
        first = report.log_probs[0, 0]
        second = report.log_probs[0, 1]
        self.assertFalse(torch.allclose(first, second))

    def test_dynamic_expansion_meets_target_requirement(self):
        head = CTCHead(d_align=4, min_expansion=1, max_expansion=8)
        aligned = torch.randn(2, 4)

        report = head(aligned, targets=["letter"])

        self.assertGreater(report.expansion, 1)
        self.assertGreaterEqual(report.input_lengths.item(), report.minimum_steps.item())

    def test_infeasible_target_fails_closed(self):
        head = CTCHead(d_align=4, min_expansion=1, max_expansion=2)
        aligned = torch.randn(1, 4)

        with self.assertRaisesRegex(ValueError, "infeasible"):
            head(aligned, targets=["impossible"])

    def test_empty_alignment_and_empty_normalized_target_are_rejected(self):
        head = CTCHead(d_align=4)
        with self.assertRaisesRegex(ValueError, "empty aligned"):
            head(torch.empty(0, 4), targets=["hello"])
        with self.assertRaisesRegex(ValueError, "becomes empty"):
            head(torch.randn(2, 4), targets=["!!!"])


if __name__ == "__main__":
    unittest.main()
