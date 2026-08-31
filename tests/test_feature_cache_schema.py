import tempfile
import unittest
from pathlib import Path

import torch

import scripts.train_stage2_pro6000 as cache_trainer
from scripts.train_stage2_pro6000 import (
    _cache_signature,
    _ctc_target_eligibility,
    _validate_cached_record,
)
from avsd_ger.training.ctc_loss import CTCHead


def valid_record():
    return {
        "utt_id": "turn-1",
        "start": 0.0,
        "end": 1.0,
        "asr_tok": torch.zeros(2, 3),
        "asr_nbest": ["hello"],
        "vsr_features": torch.zeros(4, 5),
        "voice_emb": torch.zeros(2),
        "face_emb": torch.zeros(3),
        "target": "hello",
        "lip_conf_v": torch.ones(4),
        "lip_conf_source": "detected",
        "snr_per_tok": torch.ones(2),
        "speaker_mask_v": torch.ones(4, dtype=torch.bool),
        "dataset_build_id": "ami_full_v4",
    }


class FeatureCacheSchemaTest(unittest.TestCase):
    def test_feature_fingerprint_is_independent_of_python_entry_mode(self):
        functions = (
            cache_trainer.base._load_record,
            cache_trainer.pool_encoder_to_tokens,
            cache_trainer.resample_quality_track,
            cache_trainer.token_snr_scores,
            cache_trainer._extract_cached_record,
            cache_trainer._validate_cached_record,
        )
        imported = cache_trainer._feature_source_fingerprint()
        original_modules = [function.__module__ for function in functions]
        try:
            for function in functions:
                function.__module__ = "__main__"
            executed = cache_trainer._feature_source_fingerprint()
        finally:
            for function, module_name in zip(functions, original_modules):
                function.__module__ = module_name

        self.assertEqual(imported, executed)

    def test_ctc_eligibility_accounts_for_empty_and_infeasible_targets(self):
        ctc = CTCHead(d_align=8, max_expansion=32)
        self.assertEqual(
            _ctc_target_eligibility(ctc, ".", 4)[1],
            "empty_after_normalization",
        )
        eligible, reason, minimum, required = _ctc_target_eligibility(
            ctc, "Yeah , yeah , maybe . Yeah , maybe .", 1
        )
        self.assertFalse(eligible)
        self.assertEqual(reason, "infeasible_expansion")
        self.assertEqual(minimum, 35)
        self.assertEqual(required, 35)

    def test_record_schema_accepts_aligned_quality_tracks(self):
        _validate_cached_record(valid_record())

    def test_record_schema_rejects_quality_length_mismatch(self):
        record = valid_record()
        record["lip_conf_v"] = torch.ones(3)

        with self.assertRaisesRegex(ValueError, "lip_conf_v length"):
            _validate_cached_record(record)

    def test_ger_model_does_not_change_frozen_feature_signature(self):
        cfg = {
            "stub_backbones": True,
            "asr": {"model_name": "stub"},
            "vsr": {"checkpoint": "missing.pt"},
            "identity": {
                "voice_encoder": "voice",
                "face_encoder": "face",
                "dual_gate": {"tau_a_snr_db": 8.0},
            },
            "alignment": {"snr_soft_scale_db": 4.0},
            "ger": {"model_family": "llama-3-8b-instruct"},
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "train.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            first = _cache_signature(cfg, manifest)
            cfg["ger"]["model_family"] = "qwen2.5-7b-instruct"
            second = _cache_signature(cfg, manifest)

        self.assertEqual(first, second)

    def test_feature_configuration_changes_signature(self):
        cfg = {
            "stub_backbones": True,
            "asr": {"model_name": "one"},
            "vsr": {},
            "identity": {"dual_gate": {"tau_a_snr_db": 8.0}},
            "alignment": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "train.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            first = _cache_signature(cfg, manifest)
            cfg["asr"]["model_name"] = "two"
            second = _cache_signature(cfg, manifest)

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
