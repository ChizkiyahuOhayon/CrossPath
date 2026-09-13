import unittest

import numpy as np

from export_cirr_attention_maps import normalize_map


class CIRRAttentionMapsTest(unittest.TestCase):
    def test_normalization_spans_unit_interval(self):
        normalized = normalize_map(np.asarray([[2.0, 4.0], [3.0, 2.0]]))
        self.assertEqual(float(normalized.min()), 0.0)
        self.assertEqual(float(normalized.max()), 1.0)

    def test_constant_map_is_zero(self):
        np.testing.assert_array_equal(
            normalize_map(np.ones((2, 2))), np.zeros((2, 2))
        )


if __name__ == "__main__":
    unittest.main()
