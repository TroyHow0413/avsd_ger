import tempfile
import unittest
from pathlib import Path

import torch

from avsd_ger.training.run_state import (
    RunStateCompatibilityError,
    build_provenance,
    load_run_state,
    save_run_state,
)


class RunStateTest(unittest.TestCase):
    def test_round_trip_restores_module_and_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "train.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            cfg = {"ger": {"model_family": "llama-3-8b-instruct"}}
            provenance = build_provenance(
                stage="stage2", cfg=cfg, train_manifest=manifest,
                dev_manifest=None, cache_signature="cache",
            )
            module = torch.nn.Linear(2, 2)
            optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
            original = module.weight.detach().clone()
            path = root / "last.pt"
            save_run_state(
                path,
                provenance=provenance,
                epoch=2,
                global_step=7,
                modules={"model": module},
                optimizer=optimizer,
                best_metric=0.3,
                best_epoch=1,
            )
            with torch.no_grad():
                module.weight.zero_()

            state = load_run_state(
                path,
                expected_provenance=provenance,
                modules={"model": module},
                optimizer=optimizer,
            )

            self.assertTrue(torch.equal(module.weight, original))
            self.assertEqual(state["epoch"], 2)
            self.assertEqual(state["global_step"], 7)

    def test_provenance_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "train.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            module = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
            first = build_provenance(
                stage="stage1", cfg={"seed": 1}, train_manifest=manifest,
                dev_manifest=None,
            )
            path = root / "last.pt"
            save_run_state(
                path, provenance=first, epoch=0, global_step=1,
                modules={"model": module}, optimizer=optimizer,
                best_metric=0.0, best_epoch=0,
            )
            second = build_provenance(
                stage="stage1", cfg={"seed": 2}, train_manifest=manifest,
                dev_manifest=None,
            )

            with self.assertRaisesRegex(RunStateCompatibilityError, "mismatch"):
                load_run_state(
                    path,
                    expected_provenance=second,
                    modules={"model": module},
                    optimizer=optimizer,
                )


if __name__ == "__main__":
    unittest.main()
