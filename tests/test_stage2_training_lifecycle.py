import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from avsd_ger.utils import load_config
from avsd_ger.wandb_logger import WandbLogger
from scripts.train_stage2_pro6000 import build_feature_cache, train_cached


class Stage2TrainingLifecycleTest(unittest.TestCase):
    def test_production_config_enables_bounded_early_stopping(self):
        cfg = load_config("one_go/runs/config_real_en_llama3_8b.yaml")
        stage2 = cfg["training"]["stage2"]
        self.assertEqual(stage2["epochs"], 10)
        self.assertEqual(stage2["early_stopping_patience"], 2)
        self.assertEqual(stage2["early_stopping_min_epochs"], 3)
        self.assertEqual(stage2["early_stopping_min_delta"], 0.0001)

    def test_stub_align_ctc_writes_best_and_last(self):
        cfg = load_config("configs/default.yaml")
        cfg["stub_backbones"] = True
        cfg["device"] = "cpu"
        cfg["training"]["stage2"]["epochs"] = 1

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_manifest = root / "train-missing.jsonl"
            dev_manifest = root / "dev-missing.jsonl"
            train_index = build_feature_cache(
                cfg, train_manifest, root / "train-cache", shard_size=4, rebuild=False
            )
            dev_index = build_feature_cache(
                cfg, dev_manifest, root / "dev-cache", shard_size=4, rebuild=False
            )
            output = root / "output"
            train_cached(
                cfg,
                train_index,
                dev_index,
                output,
                wb=WandbLogger(None),
                warmup="align_ctc",
                aligner_checkpoint=None,
                ctc_checkpoint=None,
                ger_projectors_checkpoint=None,
                ger_adapter_checkpoint=None,
                debug_loss_every=0,
                fail_on_nonfinite=True,
                grad_clip_norm=1.0,
                resume=None,
            )

            self.assertTrue((output / "best.pt").is_file())
            self.assertTrue((output / "last.pt").is_file())
            self.assertTrue((output / "aligner_stage2.pt").is_file())
            self.assertTrue((output / "ctc_head_stage2.pt").is_file())

    def test_early_stopping_restores_best_without_running_max_epochs(self):
        cfg = load_config("configs/default.yaml")
        cfg["stub_backbones"] = True
        cfg["device"] = "cpu"
        cfg["training"]["stage2"].update({
            "epochs": 5,
            "early_stopping_patience": 1,
            "early_stopping_min_epochs": 2,
            "early_stopping_min_delta": 0.0,
        })
        constant_dev = {
            "ctc_loss": 1.0,
            "ger_loss": 0.0,
            "ctc_records": 1.0,
            "ger_records": 0.0,
            "skipped_empty_targets": 0.0,
            "skipped_infeasible_ctc": 0.0,
            "selection_name": "dev_ctc_loss",
            "selection_value": 1.0,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_manifest = root / "train-missing.jsonl"
            dev_manifest = root / "dev-missing.jsonl"
            train_index = build_feature_cache(
                cfg, train_manifest, root / "train-cache", shard_size=4, rebuild=False
            )
            dev_index = build_feature_cache(
                cfg, dev_manifest, root / "dev-cache", shard_size=4, rebuild=False
            )
            output = root / "output"
            with patch(
                "scripts.train_stage2_pro6000._evaluate_cached",
                return_value=constant_dev,
            ) as evaluate:
                train_cached(
                    cfg,
                    train_index,
                    dev_index,
                    output,
                    wb=WandbLogger(None),
                    warmup="align_ctc",
                    aligner_checkpoint=None,
                    ctc_checkpoint=None,
                    ger_projectors_checkpoint=None,
                    ger_adapter_checkpoint=None,
                    debug_loss_every=0,
                    fail_on_nonfinite=True,
                    grad_clip_norm=1.0,
                    resume=None,
                )

            state = torch.load(output / "last.pt", weights_only=False)
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(state["epoch"], 1)
            self.assertEqual(state["best_epoch"], 0)
            self.assertTrue((output / "best.pt").is_file())


if __name__ == "__main__":
    unittest.main()
