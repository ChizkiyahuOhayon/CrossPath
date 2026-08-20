import unittest

import numpy as np

from eval_cross_compatibility import recall_metrics, target_ranks


class CrossCompatibilityTest(unittest.TestCase):
    def test_target_ranks_and_recall(self):
        scores = np.asarray([[0.9, 0.8, 0.1], [0.2, 0.1, 0.3]])
        ranks = target_ranks(scores, np.asarray([1, 1]))
        np.testing.assert_array_equal(ranks, np.asarray([2, 3]))
        self.assertEqual(recall_metrics(ranks)["R@1"], 0.0)
        self.assertEqual(recall_metrics(ranks)["R@10"], 100.0)

    def test_ties_follow_ascending_gallery_index(self):
        scores = np.asarray([[0.5, 0.5, 0.5]])
        ranks = target_ranks(scores, np.asarray([1]))
        np.testing.assert_array_equal(ranks, np.asarray([2]))


if __name__ == "__main__":
    unittest.main()
