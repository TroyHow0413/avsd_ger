import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.analyze_debug_outputs import analyze


def _score(ref, hyp):
    words = len(ref.split())
    edits = int(ref != hyp)
    return {
        "ref_words": words, "edits": edits, "substitutions": edits,
        "deletions": 0, "insertions": 0,
        "wer": edits / words if words else 0.0,
    }


class OfflineAnalyzerSchemaTest(unittest.TestCase):
    @patch("scripts.analyze_debug_outputs._edit_counts", side_effect=_score)
    @patch(
        "scripts.analyze_debug_outputs.normalize_text",
        side_effect=lambda text, **kwargs: " ".join(str(text or "").lower().split()),
    )
    def test_nested_summary_prefers_frozen_local_debug(self, _normalize, _edits):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external_dir = root / "external"
            external_dir.mkdir()
            external = external_dir / "meeting.full_model.debug.json"
            external.write_text(json.dumps({"turns": []}), encoding="utf-8")
            debug_dir = root / "meeting_debug"
            debug_dir.mkdir()
            local = debug_dir / "meeting.full_model.debug.json"
            local.write_text(json.dumps({
                "manifest": "meeting.json", "ablation": "full_model",
                "flags": {},
                "turns": [{
                    "summary": {
                        "turn_id": "t1", "duration": 1.0,
                        "ref_text": "hello world", "ref_speaker": "spk",
                        "asr_top": "hello word", "final_text": "hello world",
                        "lip_hyp": "hello",
                    },
                    "asr": {"detected_language": "en"},
                    "c1_initial": {"top_ids": ["spk"], "is_unknown": False},
                    "trace": [{
                        "cleaned_ger_text_before_gate": "hello world",
                        "final_source": "GER", "fallback_applied": False,
                    }],
                }],
            }), encoding="utf-8")
            summary = root / "summary.json"
            summary.write_text(json.dumps({
                "runs": [{
                    "manifest": "meeting.json",
                    "results": [{
                        "ablation": "full_model", "flags": {},
                        "debug_path": str(external),
                    }],
                }],
            }), encoding="utf-8")
            report = analyze([summary], language="auto")
        aggregate = next(iter(report["aggregates"].values()))
        self.assertEqual(aggregate["n_turns"], 1)
        self.assertEqual(aggregate["ger_acceptance_coverage"], 1.0)
        self.assertEqual(aggregate["c1_raw_top1_accuracy"], 1.0)
        self.assertTrue(report["turns"][0]["debug_path"].endswith("meeting.full_model.debug.json"))


if __name__ == "__main__":
    unittest.main()
