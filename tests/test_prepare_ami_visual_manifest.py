import unittest

import numpy as np

from scripts.prepare_ami_visual_manifest import (
    VideoSliceError,
    _classify_failure,
    _with_enrollment_faces,
    _with_visual_speaker_fields,
)
from scripts.ami_visual_policy import is_official_missing_closeup


class PrepareAmiVisualManifestTest(unittest.TestCase):
    def test_official_ami_missing_closeups_are_fixed_and_narrow(self):
        self.assertTrue(is_official_missing_closeup("TS3003d", "Closeup1"))
        self.assertTrue(is_official_missing_closeup("TS3003d", "Closeup3"))
        self.assertFalse(is_official_missing_closeup("TS3003d", "Closeup4"))
        self.assertFalse(is_official_missing_closeup("TS3003c", "Closeup1"))

    def test_failures_are_classified_for_structured_logging(self):
        self.assertEqual(
            _classify_failure(RuntimeError("dlib: landmark detection failed on all frames")),
            ("landmark_all_frames", "landmark_detection"),
        )
        self.assertEqual(
            _classify_failure(RuntimeError("Could not read any frames from clip.mp4")),
            ("clip_unreadable", "video_decode"),
        )
        self.assertEqual(
            _classify_failure(VideoSliceError(1, "decoder error")),
            ("ffmpeg_slice_failed", "ffmpeg_slice"),
        )

    def test_visual_turn_gets_full_speaker_mask(self):
        arr = np.zeros((7, 1, 96, 96), dtype=np.float32)
        row = _with_visual_speaker_fields({"turn_id": "IS1009c.sync.1"}, arr)

        self.assertTrue(row["has_visual"])
        self.assertEqual(row["speaker_mask_v"], [True] * 7)

    def test_enrollment_faces_are_added_without_overwriting_existing_faces(self):
        speakers = [
            {"speaker_id": "IS1009c_A", "enrollment_audio": "a.wav"},
            {
                "speaker_id": "IS1009c_B",
                "enrollment_audio": "b.wav",
                "enrollment_face": "existing.jpg",
            },
            {"speaker_id": "IS1009c_C", "enrollment_audio": "c.wav"},
        ]

        out = _with_enrollment_faces(
            speakers,
            {"A": "new_a.jpg", "B": "new_b.jpg"},
        )

        self.assertEqual(out[0]["enrollment_face"], "new_a.jpg")
        self.assertEqual(out[1]["enrollment_face"], "existing.jpg")
        self.assertNotIn("enrollment_face", out[2])


if __name__ == "__main__":
    unittest.main()
