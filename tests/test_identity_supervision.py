import json
import tempfile
import unittest
from pathlib import Path

import torch

from avsd_ger.training.identity_loss import BidirectionalInfoNCE
from avsd_ger.utils import load_config
from avsd_ger.wandb_logger import WandbLogger
from scripts.train_identity import _speaker_balanced_batches, train


class IdentitySupervisionTest(unittest.TestCase):
    def test_multi_positive_loss_does_not_treat_same_speaker_as_wrong(self):
        loss = BidirectionalInfoNCE({"temperature": 0.07, "bidirectional": True})
        audio = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        visual = audio.clone()
        labels = torch.tensor([0, 0, 1, 1])

        report = loss(audio, visual, speaker_labels=labels)

        self.assertAlmostEqual(report.acc_av, 1.0)
        self.assertAlmostEqual(report.acc_va, 1.0)
        self.assertEqual(report.mean_positives, 2.0)
        self.assertLess(float(report.loss), 0.01)

    def test_diagonal_behavior_remains_available(self):
        loss = BidirectionalInfoNCE({"temperature": 0.1, "bidirectional": True})
        values = torch.eye(3)

        report = loss(values, values)

        self.assertEqual(report.mean_positives, 1.0)
        self.assertEqual(report.acc_av, 1.0)

    def test_balanced_batches_have_fixed_speaker_and_turn_counts(self):
        labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
        torch.manual_seed(11)

        batches = _speaker_balanced_batches(
            labels, speakers_per_batch=3, turns_per_speaker=2
        )

        self.assertTrue(batches)
        for batch in batches:
            selected = labels[batch]
            self.assertEqual(batch.numel(), 6)
            self.assertEqual(torch.unique(selected).numel(), 3)
            for speaker in torch.unique(selected):
                self.assertEqual(int((selected == speaker).sum()), 2)

    def test_balanced_sampler_requires_two_speakers(self):
        with self.assertRaisesRegex(RuntimeError, "at least two"):
            _speaker_balanced_batches(
                torch.zeros(4, dtype=torch.long),
                speakers_per_batch=2,
                turns_per_speaker=2,
            )

    def test_stub_training_writes_best_last_and_pool(self):
        cfg = load_config("configs/default.yaml")
        cfg["stub_backbones"] = True
        cfg["device"] = "cpu"
        cfg["training"]["stage1"].update({
            "epochs": 1,
            "identity_supervision": "participant",
            "speakers_per_batch": 2,
            "turns_per_speaker": 1,
        })
        rows = [
            {"utt_id": f"{speaker}-{index}", "participant_id": speaker}
            for speaker in ("A", "B") for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_manifest = root / "train.jsonl"
            dev_manifest = root / "dev.jsonl"
            payload = "".join(json.dumps(row) + "\n" for row in rows)
            train_manifest.write_text(payload, encoding="utf-8")
            dev_manifest.write_text(payload, encoding="utf-8")
            output = root / "output"
            train(
                cfg,
                train_manifest,
                output,
                dev_manifest=dev_manifest,
                wb=WandbLogger(None),
            )
            self.assertTrue((output / "best.pt").is_file())
            self.assertTrue((output / "last.pt").is_file())
            self.assertTrue((output / "identity_pool_stage1.pt").is_file())


if __name__ == "__main__":
    unittest.main()
