import unittest

import torch

from fit_cross_matrix import evaluate, exclude_sources, fit_weights, target_ranks


class CrossMatrixTest(unittest.TestCase):
    def test_source_exclusion_and_tie_break(self):
        scores = torch.tensor([[[0.9, 0.8, 0.8]]]).repeat(4, 1, 1)
        exclude_sources(scores, torch.tensor([0]))
        combined = scores.mean(dim=0)
        ranks = target_ranks(combined, torch.tensor([2]))
        self.assertEqual(ranks.tolist(), [2])

    def test_fit_prefers_informative_path(self):
        scores = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 2.0]],
                [[0.0, 2.0], [2.0, 0.0]],
                [[0.0, 2.0], [2.0, 0.0]],
                [[0.0, 2.0], [2.0, 0.0]],
            ]
        )
        targets = torch.tensor([0, 1])
        weights, losses = fit_weights(scores, targets, 30, 0.2)
        self.assertGreater(float(weights[0]), 0.9)
        self.assertLess(losses[-1], losses[0])
        self.assertEqual(evaluate(scores, targets, weights, (1,))["R@1"], 100.0)


if __name__ == "__main__":
    unittest.main()
