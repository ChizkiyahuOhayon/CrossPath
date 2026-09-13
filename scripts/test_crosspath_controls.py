import unittest

import numpy as np

from analyze_crosspath_controls import rescue_harm, signed_permutation


class CrossPathControlsTest(unittest.TestCase):
    def test_signed_permutation_preserves_pairwise_scores(self):
        rng = np.random.default_rng(3)
        queries = rng.normal(size=(4, 8)).astype(np.float32)
        gallery = rng.normal(size=(6, 8)).astype(np.float32)
        permutation = rng.permutation(8)
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=8)
        transformed_queries = signed_permutation(queries, permutation, signs)
        transformed_gallery = signed_permutation(gallery, permutation, signs)
        np.testing.assert_allclose(
            queries @ gallery.T,
            transformed_queries @ transformed_gallery.T,
            atol=1e-5,
        )

    def test_rescue_harm_partition(self):
        baseline = np.asarray([1, 2, 4, 5])
        method = np.asarray([2, 4, 1, 5])
        result = rescue_harm(baseline, method, (2,))["R@2"]
        self.assertEqual(result["rescued"], 1)
        self.assertEqual(result["harmed"], 1)
        self.assertEqual(result["retained"], 1)
        self.assertEqual(result["missed_by_both"], 1)
        self.assertEqual(result["net"], 0)


if __name__ == "__main__":
    unittest.main()
