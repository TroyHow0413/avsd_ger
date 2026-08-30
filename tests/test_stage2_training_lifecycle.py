import tempfile
import unittest
from pathlib import Path

from avsd_ger.utils import load_config
from avsd_ger.wandb_logger import WandbLogger
from scripts.train_stage2_pro6000 import build_feature_cache, train_cached


class Stage2TrainingLifecycleTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
