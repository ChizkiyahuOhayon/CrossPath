import unittest

import numpy as np

from weave_crosspath import (
    boundary_trace,
    build_rank_path,
    calibrate_threshold,
    estimate_utilities,
    select_action,
)


class CrossPathTest(unittest.TestCase):
    def test_path_has_deterministic_exact_endpoints(self):
        candidate_ids = np.asarray(["c", "a", "b"])
        base_scores = np.asarray([0.2, 0.5, 0.5])
        correction_scores = np.asarray([0.9, 0.1, 0.8])

        path = build_rank_path(
            candidate_ids,
            base_scores,
            correction_scores,
            alphas=np.asarray([0.0, 0.5, 1.0]),
        )

        self.assertEqual(path.orders[0].tolist(), [1, 2, 0])
        self.assertEqual(path.orders[-1].tolist(), [0, 2, 1])
        self.assertEqual(candidate_ids[path.orders[0]].tolist(), ["a", "b", "c"])

    def test_rank_path_is_invariant_to_endpoint_score_scales(self):
        candidate_ids = np.asarray(["a", "b", "c", "d"])
        base_scores = np.asarray([4.0, 3.0, 2.0, 1.0])
        correction_scores = np.asarray([1.0, 4.0, 3.0, 2.0])
        alphas = np.linspace(0.0, 1.0, 9)

        first = build_rank_path(
            candidate_ids, base_scores, correction_scores, alphas=alphas
        )
        second = build_rank_path(
            candidate_ids,
            base_scores * 100.0 + 17.0,
            correction_scores * 0.001 - 8.0,
            alphas=alphas,
        )

        np.testing.assert_array_equal(first.orders, second.orders)

    def test_boundary_trace_is_lossless_for_cutoff_delta(self):
        candidate_ids = np.asarray(["a", "b", "c", "d"])
        path = build_rank_path(
            candidate_ids,
            np.asarray([4.0, 3.0, 2.0, 1.0]),
            np.asarray([1.0, 4.0, 3.0, 2.0]),
            alphas=np.linspace(0.0, 1.0, 9),
        )
        trace = boundary_trace(path, k=1)

        self.assertEqual(candidate_ids[trace.candidate_indices].tolist(), ["a", "b"])
        for target_index in range(candidate_ids.size):
            full_hits = np.asarray(
                [target_index in order[:1] for order in path.orders], dtype=np.int8
            )
            if target_index in trace.candidate_indices:
                boundary_index = int(
                    np.flatnonzero(trace.candidate_indices == target_index)[0]
                )
                reconstructed = trace.membership[boundary_index]
                np.testing.assert_array_equal(
                    reconstructed - reconstructed[0], full_hits - full_hits[0]
                )
            else:
                np.testing.assert_array_equal(full_hits - full_hits[0], 0)

    def test_utility_and_fallback_contract(self):
        membership = np.asarray([[0, 1, 1], [1, 1, 0]], dtype=np.int8)
        responsibility = np.asarray([0.7, 0.2, 0.1])
        utilities = estimate_utilities(
            membership, responsibility, regression_cost=2.0
        )
        np.testing.assert_allclose(utilities, [0.0, 0.7, 0.3])

        base_order = np.asarray([4, 1, 3, 0, 2])
        path_orders = np.stack([base_order, base_order[::-1], np.roll(base_order, 1)])
        selected, order = select_action(
            utilities=np.asarray([0.0, 0.2, 0.1]),
            threshold=0.25,
            path_orders=path_orders,
        )
        self.assertEqual(selected, 0)
        np.testing.assert_array_equal(order, base_order)
        self.assertTrue(np.shares_memory(order, path_orders))

    def test_threshold_calibration_breaks_ties_conservatively(self):
        predicted = np.asarray(
            [[0.0, 0.9], [0.0, 0.8], [0.0, 0.1]], dtype=np.float64
        )
        realized = np.zeros_like(predicted)
        threshold, mean_utility = calibrate_threshold(predicted, realized)

        self.assertEqual(threshold, 0.9)
        self.assertEqual(mean_utility, 0.0)


if __name__ == "__main__":
    unittest.main()
