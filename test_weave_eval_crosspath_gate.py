import unittest

import numpy as np

from weave_eval_crosspath_gate import paired_bootstrap, realized_utilities


class CrossPathEvaluationTest(unittest.TestCase):
    def test_realized_utilities_average_three_cutoffs(self):
        ranks = np.asarray([[11, 1], [1, 12]])
        utilities = realized_utilities(ranks, regression_cost=2.0)

        np.testing.assert_allclose(utilities[0], [0.0, 1.0])
        np.testing.assert_allclose(utilities[1], [0.0, -2.0])

    def test_paired_bootstrap_reports_percentage_point_delta(self):
        ranks = np.asarray([[2, 1], [1, 2], [2, 1], [1, 1]])
        actions = np.asarray([1, 0, 1, 0])
        report = paired_bootstrap(ranks, actions, samples=1000, seed=7)

        self.assertEqual(report["R@1"]["delta"], 50.0)
        self.assertGreaterEqual(report["R@1"]["ci95"][0], 0.0)
        self.assertEqual(report["R@5"]["delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
