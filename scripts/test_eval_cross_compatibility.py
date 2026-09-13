import unittest

import numpy as np

from eval_cross_compatibility import recall_metrics, target_ranks, validate_endpoints


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

    def test_endpoint_validation_rejects_dimension_mismatch(self):
        endpoints = {
            "first": (np.zeros((2, 3)), np.zeros((4, 3))),
            "second": (np.zeros((2, 5)), np.zeros((4, 5))),
        }
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            validate_endpoints(endpoints, query_count=2, gallery_count=4)


if __name__ == "__main__":
    unittest.main()
