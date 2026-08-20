import unittest

import numpy as np

from weave_build_crosspath_cache import (
    compose_correction_scores,
    compute_query_cache,
    exclude_source_candidates,
)
from weave_crosspath import build_rank_path


class CrossPathCacheTest(unittest.TestCase):
    def test_cross_mean_uses_off_diagonal_compatibilities(self):
        score01 = np.asarray([[1.0, 3.0]])
        score10 = np.asarray([[5.0, 7.0]])
        result = compose_correction_scores(
            "cross-mean", np.zeros_like(score01), score01, score10, None
        )
        np.testing.assert_allclose(result, np.asarray([[3.0, 5.0]]))

    def test_joint_path_keeps_matched_and_cross_actions(self):
        candidate_ids = np.asarray(["a", "b", "c", "d"])
        cache = compute_query_cache(
            candidate_ids,
            np.asarray([0.9, 0.8, 0.3, 0.1]),
            np.asarray([0.1, 0.9, 0.8, 0.3]),
            target_index=2,
            ks=(1, 2),
            alphas=np.linspace(0.0, 1.0, 3),
            extra_scores=np.asarray([0.8, 0.1, 0.9, 0.2]),
        )
        self.assertEqual(cache["target_ranks"].shape, (5,))
        self.assertEqual(cache["boundaries"][1]["membership"].shape[1], 5)
        self.assertEqual(
            cache["boundaries"][1]["endpoint_percentiles"].shape[1], 3
        )

    def test_source_exclusion_masks_both_endpoints(self):
        base = np.asarray([[0.8, 0.7, 0.6]], dtype=np.float32)
        correction = np.asarray([[0.5, 0.9, 0.4]], dtype=np.float32)
        extra = np.asarray([[0.4, 1.0, 0.3]], dtype=np.float32)
        rows = [{"source_id": "b"}]

        exclude_source_candidates(
            base, correction, rows, {"a": 0, "b": 1, "c": 2}, extra
        )

        self.assertLess(base[0, 1], base[0, 2])
        self.assertLess(correction[0, 1], correction[0, 2])
        self.assertLess(extra[0, 1], extra[0, 2])

    def test_query_cache_matches_core_rank_path(self):
        candidate_ids = np.asarray(["a", "b", "c", "d"])
        base_scores = np.asarray([0.9, 0.8, 0.3, 0.1])
        correction_scores = np.asarray([0.1, 0.9, 0.8, 0.3])
        alphas = np.linspace(0.0, 1.0, 9)

        cache = compute_query_cache(
            candidate_ids,
            base_scores,
            correction_scores,
            target_index=2,
            ks=(1, 2),
            alphas=alphas,
        )
        path = build_rank_path(
            candidate_ids, base_scores, correction_scores, alphas=alphas
        )
        expected_ranks = []
        for order in path.orders:
            expected_ranks.append(int(np.flatnonzero(order == 2)[0]) + 1)

        self.assertEqual(cache["target_ranks"].tolist(), expected_ranks)
        self.assertEqual(cache["boundaries"][1]["candidate_indices"].tolist(), [0, 1])
        self.assertEqual(cache["boundaries"][1]["target_position"], 2)
        self.assertEqual(cache["boundaries"][2]["target_position"], 1)


if __name__ == "__main__":
    unittest.main()
