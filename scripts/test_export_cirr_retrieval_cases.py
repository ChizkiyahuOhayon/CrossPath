import unittest

import numpy as np

from export_cirr_retrieval_cases import select_rescues


class CIRRRetrievalCasesTest(unittest.TestCase):
    def test_selects_largest_positive_rescues_deterministically(self):
        base = np.asarray([20, 12, 8, 4])
        method = np.asarray([2, 1, 4, 1])
        self.assertEqual(select_rescues(base, method, count=2), [0, 1])

    def test_rejects_insufficient_positive_cases(self):
        with self.assertRaisesRegex(ValueError, "only 1"):
            select_rescues(
                np.asarray([20, 4]), np.asarray([2, 1]), count=2
            )


if __name__ == "__main__":
    unittest.main()
