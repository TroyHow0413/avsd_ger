import json
import tempfile
import unittest
from pathlib import Path

import torch

from avsd_ger.c1_identity.identity_pool import IdentityPool
from avsd_ger.c3_feedback.closed_loop import ClosedLoopController, LoopAction
from avsd_ger.eval.metrics import compute_sa_wer
from avsd_ger.eval.standard_metrics import compute_jiwer_metrics
from avsd_ger.eval.formal_artifacts import write_formal_artifacts
from avsd_ger.text_normalization import (
    LanguageResolutionError,
    normalize_text,
    resolve_language,
)
from avsd_ger.eval.session import SessionTurnResult
from scripts.analyze_debug_outputs import _edit_counts, analyze
from avsd_ger.c3_statistics import c3_cluster_bootstrap_spec_check


def _turn(ref: str, hyp: str, language: str | None = "en") -> SessionTurnResult:
    return SessionTurnResult(
        turn_id="t1", start=0.0, end=1.0, hyp_text=hyp,
        hyp_speaker="speaker", confidence=1.0, s_acoustic=1.0,
        iterations=1, pool_updated=False, asr_language=language,
        ref_text=ref, ref_speaker="speaker",
    )


class CanonicalNormalizationTest(unittest.TestCase):
    def test_english_whisper_normalization(self):
        self.assertEqual(
            normalize_text("Hello, WORLD!", language="en"), "hello world"
        )
        self.assertEqual(
            normalize_text("I have two cats.", language="en"), "i have 2 cats"
        )

    def test_multilingual_basic_normalizer_preserves_script(self):
        normalized = normalize_text("你好，世界！", language="zh")
        self.assertIn("你好", normalized)
        self.assertIn("世界", normalized)

    def test_auto_requires_detector_metadata(self):
        with self.assertRaises(LanguageResolutionError):
            resolve_language("auto", None)
        self.assertEqual(resolve_language("auto", "EN"), "en")


class CanonicalWERTest(unittest.TestCase):
    def test_normalized_primary_and_legacy_raw_are_both_reported(self):
        score, details = compute_sa_wer(
            [_turn("Hello, WORLD!", "hello world")], language="auto"
        )
        self.assertEqual(score, 0.0)
        self.assertGreater(details["legacy_raw_wer"], 0.0)

    def test_independent_jiwer_cross_check_covers_edits(self):
        result = _edit_counts("one two three", "one four three five")
        self.assertEqual(result["edits"], 2)
        self.assertEqual(result["ref_words"], 3)

    def test_public_jiwer_payload_retains_full_error_family(self):
        result = compute_jiwer_metrics(
            [_turn("hello world", "hello duck")], language="en"
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["wer"], 0.5)
        self.assertEqual(result["substitutions"], 1)
        for key in ("mer", "wil", "wip", "cer", "hits"):
            self.assertIn(key, result)


class C3SemanticsTest(unittest.TestCase):
    def setUp(self):
        self.controller = ClosedLoopController({
            "max_iters": 3, "confidence_low": 0.3,
            "confidence_mid": 0.7, "tau_update": 0.8,
        })

    def test_decision_and_update_gates_are_orthogonal(self):
        normal = self.controller.decide(0.1, 0.1, 0)
        self.assertEqual(normal.action, LoopAction.REIDENTIFY)
        decision_off = self.controller.decide(
            0.1, 0.1, 0, disable_decision_gate=True
        )
        self.assertEqual(decision_off.action, LoopAction.ACCEPT_NO_UPDATE)
        both_off = self.controller.decide(
            0.1, 0.1, 0,
            disable_decision_gate=True, disable_update_gate=True,
        )
        self.assertEqual(both_off.action, LoopAction.ACCEPT_AND_UPDATE)

    def test_pool_update_reports_only_actual_mutation(self):
        cfg = {"top_k": 1, "min_av_consistency": 0.0,
               "voice_dim": 2, "face_dim": 2, "fused_dim": 2}
        pool = IdentityPool(cfg)
        self.assertFalse(pool.ema_update("missing", torch.ones(2)))
        pool.enroll("known", torch.zeros(2), torch.zeros(2))
        self.assertFalse(pool.ema_update("known"))
        self.assertTrue(pool.ema_update("known", torch.ones(2), alpha=0.5))

    def test_cluster_bootstrap_never_treats_equality_as_pass(self):
        runs = []
        for index in range(3):
            runs.append({
                "manifest": f"m{index}",
                "results": [
                    {"ablation": "wo_c3", "metrics": {"sa_wer": 0.5}},
                    {"ablation": "c3_wo_conf_gates", "metrics": {"sa_wer": 0.5}},
                ],
            })
        report = c3_cluster_bootstrap_spec_check(runs, samples=100, seed=1)
        self.assertEqual(report["status"], "inconclusive")
        self.assertIsNone(report["pass"])
        self.assertEqual(
            c3_cluster_bootstrap_spec_check(runs[:1], samples=10)["status"],
            "insufficient",
        )

    def test_cluster_bootstrap_classifies_direction_and_crossing_ci(self):
        def runs_for(deltas):
            return [
                {
                    "manifest": f"m{index}",
                    "results": [
                        {"ablation": "wo_c3", "metrics": {"sa_wer": 0.5}},
                        {"ablation": "c3_wo_conf_gates", "metrics": {"sa_wer": 0.5 + delta}},
                    ],
                }
                for index, delta in enumerate(deltas)
            ]
        self.assertEqual(
            c3_cluster_bootstrap_spec_check(
                runs_for([0.1, 0.2, 0.1]), samples=500, seed=2
            )["status"],
            "degraded",
        )
        self.assertEqual(
            c3_cluster_bootstrap_spec_check(
                runs_for([-0.1, -0.2, -0.1]), samples=500, seed=2
            )["status"],
            "improved",
        )
        self.assertEqual(
            c3_cluster_bootstrap_spec_check(
                runs_for([-0.2, 0.0, 0.2]), samples=1000, seed=2
            )["status"],
            "inconclusive",
        )


class OfflineAnalyzerTest(unittest.TestCase):
    def test_direct_debug_schema_and_coverage(self):
        turn = {
            "summary": {
                "turn_id": "t1", "duration": 1.0,
                "ref_text": "Hello world", "ref_speaker": "spk",
                "asr_top": "hello word", "final_text": "hello world",
                "lip_hyp": "hello",
            },
            "asr": {"detected_language": "en"},
            "c1_initial": {"top_ids": ["spk"], "is_unknown": False},
            "trace": [{
                "cleaned_ger_text_before_gate": "hello world",
                "final_source": "GER", "fallback_applied": False,
                "safety_gates": [{"gate": "overlap", "passed": True}],
            }],
        }
        payload = {
            "manifest": "meeting.json", "ablation": "full_model",
            "flags": {}, "turns": [turn],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "debug.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            report = analyze([path], language="auto")
        aggregate = next(iter(report["aggregates"].values()))
        self.assertEqual(aggregate["final_wer_micro"], 0.0)
        self.assertEqual(aggregate["ger_acceptance_coverage"], 1.0)
        self.assertEqual(aggregate["outcomes"], {"improved": 1})
        self.assertEqual(aggregate["c1_raw_top1_accuracy"], 1.0)


class FormalArtifactTest(unittest.TestCase):
    def test_formal_tree_and_canonical_ablation_name(self):
        turn = {
            "summary": {
                "turn_id": "t1", "start": 0.0, "end": 1.0,
                "ref_text": "hello world", "ref_speaker": "alice",
                "hyp_speaker": "pool_a", "asr_top": "hello word",
                "final_text": "hello world", "confidence": 0.9,
                "speaker_hyp_top5": ["pool_a", "pool_b"],
                "c1_similarity_top5": [0.9, 0.2],
                "av_consistency_raw": 0.9, "has_visual": True,
                "lip_conf_mean": 0.8, "snr_estimate_db_mean": 12.0,
                "wall_time_ms": 100.0, "gpu_memory_allocated_mb": 200.0,
            },
            "trace": [{
                "text": "hello world", "asr_top": "hello word",
                "cleaned_ger_text_before_gate": "hello world",
                "fallback_applied": False, "final_source": "GER",
            }],
            "turn": {"manifest_row": {}},
        }
        result = {
            "ablation": "c3_wo_conf_gates", "flags": {},
            "metrics": {"wer": 0.0}, "standard_metrics": {},
            "metric_details": {"av_sid": {"mapping": {"pool_a": "alice"}}},
            "trace_summary": {}, "profile": {
                "wall_time_s": 0.1, "audio_duration_s": 1.0,
            }, "power": None, "turn_debug": [turn],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "meeting.json"
            manifest_path.write_text("{}", encoding="utf-8")
            output = root / "eval"
            write_formal_artifacts(
                output,
                [{"manifest": str(manifest_path), "results": [result]}],
                repo_root=Path(__file__).resolve().parents[1],
                config_path=str(manifest_path), config={"asr": {"language": "en"}},
                pool_path=None, aligner_ckpt=None, ger_ckpt=None,
                seed=42, started_at="2026-01-01T00:00:00+00:00",
            )
            expected = [
                "run_manifest.json", "records.schema.json",
                "records/c3_wo_confidence_gates.jsonl",
                "metrics/main_table.json",
                "metrics/per_ablation/c3_wo_confidence_gates.json",
                "metrics/per_meeting/c3_wo_confidence_gates.jsonl",
                "metrics/groups/by_visual_availability.json",
                "scoring_inputs/reference/reference.stm",
                "scoring_inputs/hypothesis/c3_wo_confidence_gates/hypothesis.stm",
                "profiles/latency_per_turn.jsonl",
                "debug/c3_wo_confidence_gates/meeting.debug.json",
            ]
            for relative in expected:
                self.assertTrue((output / relative).exists(), relative)
            run_manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(
                run_manifest["ablations"][0]["id"],
                "c3_wo_confidence_gates",
            )


if __name__ == "__main__":
    unittest.main()
