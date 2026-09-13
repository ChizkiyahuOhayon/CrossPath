import unittest

import numpy as np

from export_retrieval_cases import parse_category_paths, select_cases


class RetrievalCaseSelectionTest(unittest.TestCase):
    def test_selects_largest_unused_rescue_deterministically(self):
        base = np.array([2, 20, 12, 60, 30])
        method = np.array([1, 4, 5, 8, 9])

        self.assertEqual(select_cases(base, method, (1, 5, 10)), [(1, 0), (5, 1), (10, 3)])

    def test_method_rank_limit(self):
        base = np.array([30, 20])
        method = np.array([8, 4])

        self.assertEqual(select_cases(base, method, (10,), method_rank_limit=5), [(10, 1)])

    def test_category_paths_require_all_categories(self):
        with self.assertRaisesRegex(ValueError, "dress, shirt, and toptee"):
            parse_category_paths(("dress=/tmp/dress",), "--embedding-dirs")


if __name__ == "__main__":
    unittest.main()
