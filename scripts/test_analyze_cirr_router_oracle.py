import unittest

import numpy as np

from analyze_cirr_router_oracle import action_scores, metrics


class CIRRRouterOracleTest(unittest.TestCase):
    def test_action_scores_keep_exact_mcot_fallback(self):
        paths = {
            "q1_g1": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "q1_g0": np.asarray([[0.0, 1.0]], dtype=np.float32),
            "cross_mean": np.asarray([[0.5, 0.5]], dtype=np.float32),
        }
        actions = action_scores(paths, (0.25, 1.0))
        np.testing.assert_array_equal(actions["mcot"], paths["q1_g1"])
        np.testing.assert_allclose(actions["q1_g0_a0.25"], [[0.75, 0.25]])
        np.testing.assert_array_equal(actions["cross_mean_a1"], paths["cross_mean"])

    def test_metrics_uses_official_cirr_average(self):
        result = metrics(
            np.asarray([1, 5, 6, 50]),
            np.asarray([1, 2, 1, 3]),
        )
        self.assertEqual(result["R@5"], 50.0)
        self.assertEqual(result["subset_R@1"], 50.0)
        self.assertEqual(result["Avg"], 50.0)


if __name__ == "__main__":
    unittest.main()
