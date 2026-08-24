import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from avsd_ger.frontend.mouth_roi import _detection_confidence
from scripts.ami_metadata import (
    apply_ami_speaker_metadata,
    load_ami_meetings,
    meeting_closeup_map,
)
from scripts.ami_visual_to_jsonl import iter_records
from scripts.build_ami_visual_manifests import _cli_path as visual_cli_path
from scripts.build_ami_visual_manifests import _write_or_verify_plan as visual_build_plan
from scripts.rebuild_ami_base_manifests import _cli_path as base_cli_path
from scripts.rebuild_ami_base_manifests import _write_or_verify_plan as base_build_plan


MEETINGS_XML = """<?xml version="1.0" encoding="ISO-8859-1"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
  <meeting observation="ES2002a">
    <speaker nxt_agent="A" channel="0" camera="Closeup1" global_name="MEE006" role="ID"/>
    <speaker nxt_agent="B" channel="1" camera="Closeup4" global_name="FEE005" role="PM"/>
  </meeting>
</nite:root>
"""


class AmiMetadataTest(unittest.TestCase):
    def test_official_metadata_controls_camera_and_global_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "meetings.xml"
            path.write_text(MEETINGS_XML, encoding="utf-8")
            speakers = load_ami_meetings(path)["ES2002a"]

        self.assertEqual(meeting_closeup_map(speakers), {"A": "Closeup1", "B": "Closeup4"})
        manifest = {
            "meta": {"dataset_build_id": "ami_full_v2"},
            "speakers": [
                {"speaker_id": "ES2002a_A", "enrollment_audio": "a.wav"},
                {"speaker_id": "ES2002a_B", "enrollment_audio": "b.wav"},
            ],
            "turns": [
                {"turn_id": "a1", "ref_speaker": "ES2002a_A"},
                {"turn_id": "b1", "ref_speaker": "ES2002a_B"},
            ],
        }
        out = apply_ami_speaker_metadata(manifest, "ES2002a", speakers)

        self.assertEqual(out["speakers"][0]["speaker_id"], "MEE006")
        self.assertEqual(out["speakers"][1]["speaker_id"], "FEE005")
        self.assertEqual(out["turns"][1]["ref_speaker"], "FEE005")
        self.assertEqual(out["turns"][1]["camera"], "Closeup4")
        self.assertEqual(out["turns"][1]["session_speaker_id"], "ES2002a_B")
        self.assertEqual(out["meta"]["speaker_identity_scope"], "corpus_global")
        self.assertEqual(out["meta"]["dataset_build_id"], "ami_full_v2")

    def test_versioned_build_paths_are_repository_relative(self):
        root = Path("repo").resolve()
        target = root / "data" / "ami_full_v2" / "train" / "base"
        expected = str(Path("data") / "ami_full_v2" / "train" / "base")
        self.assertEqual(base_cli_path(target, root), expected)
        self.assertEqual(visual_cli_path(target, root), expected)

    def test_build_plans_are_immutable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for writer, name in (
                (base_build_plan, "base.json"),
                (visual_build_plan, "visual.json"),
            ):
                path = root / name
                writer(path, {"dataset_build_id": "ami_full_v2"}, dry_run=False)
                writer(path, {"dataset_build_id": "ami_full_v2"}, dry_run=False)
                with self.assertRaisesRegex(ValueError, "choose a new --run-dir"):
                    writer(path, {"dataset_build_id": "ami_full_v3"}, dry_run=False)


class AmiVisualConfidenceTest(unittest.TestCase):
    def test_confidence_distinguishes_detection_interpolation_and_extrapolation(self):
        confidence = _detection_confidence([False, True, False, True, False])
        np.testing.assert_allclose(confidence, [0.25, 1.0, 0.5, 1.0, 0.25])

    def test_all_missing_detection_is_zero_not_fake_one(self):
        np.testing.assert_array_equal(_detection_confidence([False] * 4), np.zeros(4))


class AmiJsonlConversionTest(unittest.TestCase):
    def _fixture(self, root: Path, *, include_confidence: bool) -> Path:
        roi_a = root / "a.npy"
        roi_b = root / "b.npy"
        np.save(roi_a, np.zeros((3, 1, 96, 96), dtype=np.float32))
        np.save(roi_b, np.zeros((3, 1, 96, 96), dtype=np.float32))
        for name in ("a.wav", "b.wav", "a.jpg", "b.jpg"):
            (root / name).touch()
        turns = [
            {
                "turn_id": "a1",
                "ref_speaker": "P_A",
                "audio": str(root / "a.wav"),
                "mouth_roi": str(roi_a),
                "ref_text": "hello",
            },
            {
                "turn_id": "b1",
                "ref_speaker": "P_B",
                "audio": str(root / "b.wav"),
                "mouth_roi": str(roi_b),
                "ref_text": "world",
            },
        ]
        if include_confidence:
            for turn in turns:
                turn["lip_conf_v"] = [1.0, 0.5, 0.25]
                turn["lip_conf_source"] = "test_detector"
        manifest = {
            "speakers": [
                {"speaker_id": "P_A", "enrollment_face": str(root / "a.jpg")},
                {"speaker_id": "P_B", "enrollment_face": str(root / "b.jpg")},
            ],
            "turns": turns,
            "meta": {"meeting_id": "M1"},
        }
        path = root / "M1.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_missing_confidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._fixture(root, include_confidence=False)
            with self.assertRaisesRegex(ValueError, "no real lip_conf_v"):
                iter_records([manifest], root=root)

            rows = iter_records(
                [manifest], root=root, require_lip_confidence=False
            )
            self.assertIsNone(rows[0]["lip_conf"])
            self.assertEqual(rows[0]["lip_conf_source"], "missing")

    def test_negative_sampling_never_uses_same_global_speaker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._fixture(root, include_confidence=True)
            rows = iter_records([manifest], root=root)

        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertNotEqual(row["speaker_id"], row["neg_speaker_id"])
            self.assertEqual(row["lip_conf_source"], "test_detector")


if __name__ == "__main__":
    unittest.main()
