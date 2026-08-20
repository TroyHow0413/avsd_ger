import unittest

from avsd_ger.c3_feedback.ger_safety import GERSafetyGate


class GERSafetyGateTest(unittest.TestCase):
    def test_accepts_plausible_correction(self):
        gate = GERSafetyGate()

        self.assertIsNone(
            gate.reject_reason("please close the door", "please close door")
        )

    def test_rejects_prompt_artifact(self):
        gate = GERSafetyGate()

        reason = gate.reject_reason(
            "Here is the corrected transcript: hello", "hello"
        )

        self.assertIn("blacklist", reason)

    def test_disabled_gate_never_rejects(self):
        gate = GERSafetyGate(enabled=False)

        self.assertIsNone(gate.reject_reason("", "hello"))

    def test_features_expose_policy_and_overlap(self):
        gate = GERSafetyGate(min_token_overlap=0.25)

        features = gate.features("alpha delta", "alpha beta")

        self.assertEqual(features["token_overlap"], 0.5)
        self.assertEqual(features["min_token_overlap"], 0.25)


if __name__ == "__main__":
    unittest.main()
