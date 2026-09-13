import unittest

import numpy as np
import torch

from weave_train_candidate_compatibility import (
    CandidateCompatibilityResidual,
    hard_candidates,
)


class CandidateCompatibilityTest(unittest.TestCase):
    def test_zero_initialization_is_exact_mcot_score(self):
        model = CandidateCompatibilityResidual(hidden_width=8)
        scores = torch.randn(3, 5, 4)
        torch.testing.assert_close(model(scores), scores[..., 3])

    def test_hard_candidates_exclude_target_and_source(self):
        hard = np.asarray([[2, 1, 3, 4], [0, 2, 3, 4]])
        result = hard_candidates(
            np.asarray([0, 1]),
            np.asarray([1, 2]),
            np.asarray([2, 0]),
            hard,
            2,
        )
        np.testing.assert_array_equal(result, [[1, 3, 4], [2, 3, 4]])


if __name__ == "__main__":
    unittest.main()
