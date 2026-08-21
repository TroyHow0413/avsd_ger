import unittest

from avsd_ger.c3_statistics import c3_cluster_bootstrap_spec_check


def _runs(deltas):
    return [
        {
            "manifest": f"meeting-{index}",
            "results": [
                {"ablation": "wo_c3", "metrics": {"sa_wer": 0.5}},
                {
                    "ablation": "c3_wo_conf_gates",
                    "metrics": {"sa_wer": 0.5 + delta},
                },
            ],
        }
        for index, delta in enumerate(deltas)
    ]


class C3StatisticsTest(unittest.TestCase):
    def test_direction_equality_crossing_and_insufficient(self):
        self.assertEqual(
            c3_cluster_bootstrap_spec_check(
                _runs([0.1, 0.2, 0.1]), samples=500, seed=7
            )["status"],
            "degraded",
        )
        self.assertEqual(
            c3_cluster_bootstrap_spec_check(
                _runs([-0.1, -0.2, -0.1]), samples=500, seed=7
            )["status"],
            "improved",
        )
        for deltas in ([0.0, 0.0, 0.0], [-0.2, 0.0, 0.2]):
            result = c3_cluster_bootstrap_spec_check(
                _runs(deltas), samples=1000, seed=7
            )
            self.assertEqual(result["status"], "inconclusive")
            self.assertIsNone(result["pass"])
        self.assertEqual(
            c3_cluster_bootstrap_spec_check(_runs([0.1]))["status"],
            "insufficient",
        )


if __name__ == "__main__":
    unittest.main()
