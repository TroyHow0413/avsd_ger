import json
import tempfile
import unittest
from pathlib import Path

import torch

from avsd_ger.c1_identity.identity_pool import IdentityPool
from avsd_ger.c3_feedback.closed_loop import ClosedLoopController, LoopAction
from avsd_ger.eval.metrics import compute_sa_wer
from avsd_ger.eval.standard_metrics import compute_jiwer_metrics, _normalized_turns
from avsd_ger.eval.formal_artifacts import _visual_availability, write_formal_artifacts
from avsd_ger.eval.statistics import (
    build_paired_comparisons,
    build_statistics_report,
    meeting_series_id,
)
from avsd_ger.pipeline import AVSDGERPipeline
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

    def test_identity_derangement_is_sorted_cyclic_and_changes_only_vector(self):
        cfg = {"top_k": 2, "min_av_consistency": -1.0,
               "voice_dim": 2, "face_dim": 2, "fused_dim": 2}
        pool = IdentityPool(cfg)
        pool.enroll("speaker_b", torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))
        pool.enroll("speaker_a", torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
        self.assertEqual(
            pool.deterministic_derangement(),
            {"speaker_a": "speaker_b", "speaker_b": "speaker_a"},
        )
        query_voice = torch.tensor([1.0, 0.0])
        query_face = torch.tensor([1.0, 0.0])
        result = pool.query(query_voice, query_face)
        original_metadata = (
            list(result.top_ids), list(result.top_scores), result.av_consistency,
            result.is_unknown,
        )
        shuffled = pool.conditioning_vector_for_speaker(
            query_voice, query_face,
            pool.deterministic_derangement()[result.top_ids[0]],
        )
        self.assertEqual(
            original_metadata,
            (list(result.top_ids), list(result.top_scores), result.av_consistency,
             result.is_unknown),
        )
        self.assertEqual(shuffled.shape, result.z_id.shape)

    def test_pure_identity_interventions_preserve_query_metadata(self):
        cfg = {"top_k": 2, "min_av_consistency": -1.0,
               "voice_dim": 2, "face_dim": 2, "fused_dim": 2}
        pool = IdentityPool(cfg)
        pool.enroll("a", torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
        pool.enroll("b", torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))
        voice = torch.tensor([1.0, 0.0])
        face = torch.tensor([1.0, 0.0])
        query = pool.query(voice, face)
        metadata = (list(query.top_ids), list(query.top_scores), query.is_unknown)

        pipe = object.__new__(AVSDGERPipeline)
        pipe.pool = pool
        pipe.zero_z_id, pipe.shuffle_z_id = True, False
        zero, mode, source, mapping, eligible = pipe._identity_conditioning(
            query, voice, face,
        )
        self.assertTrue(torch.count_nonzero(zero).item() == 0)
        self.assertEqual((mode, source, mapping, eligible), ("zero", None, {}, True))

        pipe.zero_z_id, pipe.shuffle_z_id = False, True
        shuffled, mode, source, mapping, eligible = pipe._identity_conditioning(
            query, voice, face,
        )
        self.assertEqual(mode, "shuffled")
        self.assertNotEqual(source, query.top_ids[0])
        self.assertEqual(mapping[query.top_ids[0]], source)
        self.assertTrue(eligible)
        self.assertEqual(metadata, (list(query.top_ids), list(query.top_scores), query.is_unknown))
        self.assertEqual(shuffled.shape, query.z_id.shape)

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
    def test_visual_availability_uses_input_availability_not_effective_mode(self):
        audio_only = {
            "summary": {"has_visual": False},
            "input": {"has_visual_flag": True},
            "turn": {"manifest_row": {"mouth_roi": "mouth.npy"}},
            "trace": [{"ger_mode": "audio_only"}],
        }
        self.assertEqual(
            _visual_availability(audio_only, "full_model"),
            "audio_only_by_ablation",
        )
        wo_c2_still_uses_visual = {
            "summary": {"has_visual": True, "lip_conf_mean": 0.9},
            "input": {"has_visual_flag": True},
            "turn": {"manifest_row": {"mouth_roi": "mouth.npy"}},
            "trace": [{"ger_mode": "av"}],
        }
        self.assertEqual(
            _visual_availability(wo_c2_still_uses_visual, "wo_c2"),
            "real_visual",
        )

    def test_formal_tree_and_canonical_ablation_name(self):
        turn = {
            "summary": {
                "turn_id": "t1", "start": 0.0, "end": 1.0,
                "ref_text": "hello world", "ref_speaker": "alice",
                "hyp_speaker": "pool_a", "asr_top": "hello word",
                "final_text": "hello world", "confidence": 0.9,
                "speaker_hyp_top5": ["pool_a", "pool_b"],
                "c1_similarity_top5": [0.9, 0.2],
                "av_consistency_raw": 0.1, "has_visual": True,
                "lip_conf_mean": 0.8, "snr_estimate_db_mean": 12.0,
                "wall_time_ms": 100.0, "gpu_memory_allocated_mb": 200.0,
            },
            "trace": [{
                "text": "hello world", "asr_top": "hello word",
                "cleaned_ger_text_before_gate": "hello world",
                "fallback_applied": False, "final_source": "GER",
                "av_consistency_raw": 0.9,
            }],
            "turn": {"manifest_row": {}},
            "input": {"has_visual_flag": True},
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
                "metrics/statistics.json",
                "metrics/paired_comparisons.json",
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
            calibration = json.loads(
                (output / "metrics/appendix_calibration.json").read_text()
            )
            c1 = calibration["c3_wo_confidence_gates"]["c1"]
            self.assertEqual(c1["status"], "ok")
            self.assertEqual(c1["n_positive"], 1)
            self.assertAlmostEqual(c1["brier_score"], 0.01)
            protocol = json.loads(
                (output / "metrics/scoring_protocol.json").read_text()
            )
            self.assertIn("memory_semantics", protocol)


class ClusterStatisticsTest(unittest.TestCase):
    @staticmethod
    def _result(ablation, errors, length=100):
        value = errors / length
        return {
            "ablation": ablation,
            "metrics": {
                "wer": value, "sa_wer": value, "scr": 0.0,
                "av_sid_acc": 1.0, "der": 0.0, "jer": 0.0,
            },
            "standard_metrics": {
                "jiwer": {
                    "wer": value, "substitutions": errors,
                    "deletions": 0, "insertions": 0,
                    "reference_words": length,
                },
                "meeteval": {"scores": {
                    "cpwer": {"error_rate": value, "errors": errors, "length": length},
                    "tcpwer_collar_5s": {"error_rate": value, "errors": errors, "length": length},
                }},
            },
            "metric_details": {
                "sa_wer": {
                    "n_sub": errors, "n_del": 0, "n_ins": 0,
                    "n_spk_err": 0, "n_ref_words": length,
                },
                "scr": {"n_spk_err": 0, "n_matched": length - errors},
                "av_sid": {"n_correct": 1, "n": 1},
                "der": {"miss": 0, "false_alarm": 0, "confusion": 0, "total_ref": 1},
                "jer": {"per_speaker": {"speaker": 0.0}},
            },
        }

    def test_session_bootstrap_and_paired_delta(self):
        runs = []
        for meeting, full_errors, wo_errors in [
            ("ES2011a", 10, 20), ("ES2011b", 20, 30),
            ("IS1008a", 15, 25), ("TS3004a", 25, 35),
        ]:
            runs.append({
                "manifest": f"{meeting}.json",
                "results": [
                    self._result("full_model", full_errors),
                    self._result("wo_c3", wo_errors),
                ],
            })
        statistics = build_statistics_report(runs, samples=200, seed=7)
        tcp = statistics["ablations"]["full_model"]["tcpwer_5s"]
        self.assertEqual(tcp["n_sessions"], 4)
        self.assertEqual(tcp["n_meeting_series"], 3)
        self.assertEqual(tcp["session_cluster"]["n_clusters"], 4)
        paired = build_paired_comparisons(runs, samples=200, seed=7)
        comparison = paired["comparisons"]["c3_topology"]
        c3 = comparison["metrics"]["tcpwer_5s"]["session_cluster"]
        self.assertAlmostEqual(c3["delta_challenger_minus_reference"], 0.1)
        self.assertEqual(c3["direction"], "challenger_worse")
        self.assertEqual(comparison["selection_gate"]["decision"], "full_model")
        sensitivity = comparison["metrics"]["tcpwer_5s"]["meeting_series_sensitivity"]
        self.assertEqual(sensitivity["n_pairs"], 3)
        self.assertIn("warning", sensitivity)

    def test_ami_series_id(self):
        self.assertEqual(meeting_series_id("ES2011d"), "ES2011")

    def test_identity_causal_gate(self):
        runs = []
        for meeting in ("ES2011a", "IS1008a", "TS3004a"):
            runs.append({
                "manifest": f"{meeting}.json",
                "results": [
                    self._result("identity_normal", 10),
                    self._result("zero_z_id", 20),
                    self._result("shuffled_z_id", 25),
                ],
            })
        paired = build_paired_comparisons(runs, samples=100, seed=3)
        self.assertEqual(
            paired["comparisons"]["identity_zero"]["causal_gate"]["decision"],
            "supports_identity_conditioning",
        )

    def test_public_scorer_rows_are_chronological(self):
        from avsd_ger.eval.session import SessionTurnResult
        turns = [
            SessionTurnResult("late", 4.0, 5.0, "b", "s", 1.0, None, 1, False,
                              ref_text="b", ref_speaker="s"),
            SessionTurnResult("early", 1.0, 2.0, "a", "s", 1.0, None, 1, False,
                              ref_text="a", ref_speaker="s"),
        ]
        rows = _normalized_turns(turns, "en")
        self.assertEqual([row["turn_id"] for row in rows], ["early", "late"])


if __name__ == "__main__":
    unittest.main()
