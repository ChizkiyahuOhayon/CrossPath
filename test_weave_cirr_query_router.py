import unittest

import numpy as np
import torch

from weave_cirr_query_router import (
    action_scores,
    choose_actions,
    soft_oracle_targets,
    subset_target_ranks,
)


class CIRRQueryRouterTest(unittest.TestCase):
    def test_action_zero_is_exact_mcot_and_source_is_excluded(self):
        paths = {
            "q1_g1": torch.tensor([[0.9, 0.8, 0.1]]),
            "q1_g0": torch.tensor([[0.7, 0.6, 0.5]]),
            "cross_mean": torch.tensor([[0.8, 0.7, 0.3]]),
        }
        scores = action_scores(paths, (0.25,), torch.tensor([0]))
        self.assertTrue(torch.isneginf(scores[0, 0, 0]))
        torch.testing.assert_close(scores[0, 0, 1:], paths["q1_g1"][0, 1:])

    def test_subset_rank_excludes_source(self):
        ranks = subset_target_ranks(
            torch.tensor([[0.9, 0.8, 0.7]]),
            torch.tensor([[0, 1, 2]]),
            torch.tensor([1]),
            torch.tensor([0]),
        )
        torch.testing.assert_close(ranks, torch.tensor([1]))

    def test_soft_target_prefers_base_on_tied_utility(self):
        global_ranks = np.asarray([[1, 1], [6, 5]])
        subset_ranks = np.asarray([[1, 1], [2, 1]])
        targets = soft_oracle_targets(global_ranks, subset_ranks)
        np.testing.assert_array_equal(targets[0], [1.0, 0.0])
        np.testing.assert_array_equal(targets[1], [0.0, 1.0])

    def test_threshold_falls_back_to_base(self):
        logits = np.asarray([[0.0, 0.2], [0.0, 1.0]])
        np.testing.assert_array_equal(choose_actions(logits, 0.5), [0, 1])


if __name__ == "__main__":
    unittest.main()
