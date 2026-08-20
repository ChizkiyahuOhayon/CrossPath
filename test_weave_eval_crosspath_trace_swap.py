import unittest

import numpy as np

from weave_eval_crosspath_trace_swap import (
    adjacent_margin_partners,
    paired_difference,
    transfer_trace,
)


class CrossPathTraceSwapTest(unittest.TestCase):
    def test_adjacent_margin_partners_is_deterministic_and_leaves_odd_record(self):
        partners = adjacent_margin_partners(
            np.asarray([4, 1, 3, 2, 5]),
            np.asarray([0.4, 0.1, 0.2, 0.2, 0.9]),
        )

        self.assertEqual(partners, {4: 3, 1: 2, 3: 4, 2: 1, 5: 5})

    def test_transfer_trace_matches_gallery_identity_and_zeros_absent(self):
        transferred, overlap = transfer_trace(
            np.asarray([2, 5, 9]),
            np.asarray([1, 5, 9]),
            np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.uint8),
        )

        np.testing.assert_array_equal(
            transferred,
            np.asarray([[0, 0], [0, 1], [1, 1]], dtype=np.uint8),
        )
        self.assertEqual(overlap, 2)

    def test_paired_difference_reports_a_minus_b(self):
        report = paired_difference(
            np.asarray([2.0, 1.0, 3.0]),
            np.asarray([1.0, 1.0, 1.0]),
            samples=1000,
            seed=7,
        )

        self.assertAlmostEqual(report["delta"], 1.0)
        self.assertGreaterEqual(report["ci95"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
