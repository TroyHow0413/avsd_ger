import unittest

from scripts.prepare_ami_manifest import _enrollment_windows


class PrepareAmiManifestEnrollmentTest(unittest.TestCase):
    def test_long_turn_is_split_instead_of_discarded(self):
        windows = _enrollment_windows(
            {"id": "ES2005a.B.seg1", "start": 10.0, "end": 36.5},
            min_secs=3.0,
            max_secs=8.0,
        )

        self.assertEqual(
            [(row["start"], row["end"]) for row in windows],
            [(10.0, 18.0), (18.0, 26.0), (26.0, 34.0)],
        )
        self.assertEqual(windows[0]["id"], "ES2005a.B.seg1.window1")

    def test_in_range_turn_is_kept_unchanged(self):
        segment = {"id": "seg", "start": 1.0, "end": 6.0}
        self.assertEqual(_enrollment_windows(segment, 3.0, 8.0), [segment])

    def test_short_turn_is_not_an_enrollment_candidate(self):
        self.assertEqual(
            _enrollment_windows({"id": "seg", "start": 1.0, "end": 2.0}, 3.0, 8.0),
            [],
        )


if __name__ == "__main__":
    unittest.main()
