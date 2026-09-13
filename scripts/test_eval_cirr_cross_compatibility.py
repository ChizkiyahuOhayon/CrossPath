import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from eval_cirr_cross_compatibility import (
    recall,
    score_paths,
    submission_for_path,
    subset_target_ranks,
    target_ranks,
)


class CIRRCrossCompatibilityTest(unittest.TestCase):
    def test_global_rank_uses_stable_gallery_order_for_ties(self):
        scores = np.asarray([[0.8, 0.8, 0.1], [0.2, 0.1, 0.3]])
        ranks = target_ranks(scores, np.asarray([1, 1]))
        np.testing.assert_array_equal(ranks, np.asarray([2, 3]))
        self.assertEqual(recall(ranks, (1, 2)), {"R@1": 0.0, "R@2": 50.0})

    def test_subset_rank_excludes_reference(self):
        scores = np.asarray([[0.9, 0.8, 0.7, 0.6]])
        ranks = subset_target_ranks(
            scores,
            [np.asarray([0, 1, 2])],
            np.asarray([1]),
            np.asarray([0]),
        )
        np.testing.assert_array_equal(ranks, np.asarray([1]))

    def test_all_mean_is_average_of_four_compatibilities(self):
        query0 = np.asarray([[1.0, 0.0]], dtype=np.float32)
        query1 = np.asarray([[0.0, 1.0]], dtype=np.float32)
        gallery0 = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        gallery1 = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        paths = score_paths(query0, query1, gallery0, gallery1)
        expected = 0.25 * (
            paths["q0_g0"] + paths["q0_g1"] + paths["q1_g0"] + paths["q1_g1"]
        )
        np.testing.assert_allclose(paths["all_mean"], expected)

    def test_borda_ranks_are_built_after_source_exclusion(self):
        query = np.asarray([[1.0, 0.0]], dtype=np.float32)
        gallery = np.asarray(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32
        )
        paths = score_paths(query, query, gallery, gallery, np.asarray([0]))
        self.assertGreater(paths["all_borda"][0, 1], paths["all_borda"][0, 2])
        self.assertGreater(paths["all_borda"][0, 2], paths["all_borda"][0, 0])

    def test_submission_excludes_reference_and_ranks_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gallery = np.asarray(
                [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]],
                dtype=np.float32,
            )
            query = np.asarray([[1.0, 0.0]], dtype=np.float32)
            for name, array in (
                ("base_gallery", gallery),
                ("correction_gallery", gallery),
                ("base_queries", query),
                ("correction_queries", query),
            ):
                np.save(root / f"{name}.npy", array)
            (root / "gallery_ids.json").write_text(json.dumps(["g0", "g1", "g2", "g3"]))
            (root / "queries.jsonl").write_text(
                json.dumps(
                    {
                        "source_id": "g0",
                        "group_members": ["g0", "g1", "g2"],
                        "pair_id": 7,
                    }
                )
                + "\n"
            )
            general, subset = submission_for_path(root, "all_mean", batch_size=1)
            self.assertEqual(general["7"][0], "g1")
            self.assertNotIn("g0", general["7"])
            self.assertEqual(subset["7"][:2], ["g1", "g2"])


if __name__ == "__main__":
    unittest.main()
