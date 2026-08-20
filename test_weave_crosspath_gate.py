import unittest

import numpy as np
import torch

from weave_crosspath_gate import (
    CrossPathGate,
    aggregate_cutoff_utilities,
    batch_listwise_loss,
    build_candidate_features,
    listwise_loss,
    listwise_target,
    realized_cutoff_utilities,
)


class CrossPathGateTest(unittest.TestCase):
    def test_features_match_predeclared_contract(self):
        query = np.asarray([0.2, -0.4], dtype=np.float32)
        documents = np.asarray([[0.1, 0.3], [-0.2, 0.5]], dtype=np.float32)
        percentiles = np.asarray([[1.0, 0.5], [0.7, 0.9]], dtype=np.float32)
        membership = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.float32)

        features = build_candidate_features(
            query, documents, percentiles, membership, k=5
        )

        self.assertEqual(features.shape, (2, 14))
        np.testing.assert_allclose(features[:, -1], 0.5)
        np.testing.assert_allclose(features[0, -4:-1], membership[0])

    def test_none_target_is_last_listwise_class(self):
        boundary = np.asarray([8, 3, 5])
        self.assertEqual(listwise_target(boundary, target_index=3), 1)
        self.assertEqual(listwise_target(boundary, target_index=7), 3)

    def test_gate_outputs_normalized_candidate_and_none_probability(self):
        torch.manual_seed(7)
        gate = CrossPathGate(embedding_dim=2, num_actions=3, hidden_width=4)
        features = torch.randn(5, 14)

        probabilities = gate.probabilities(features)

        self.assertEqual(tuple(probabilities.shape), (6,))
        self.assertAlmostEqual(float(probabilities.sum().detach()), 1.0, places=6)
        self.assertTrue(torch.isfinite(probabilities).all())

    def test_gate_accepts_three_endpoint_percentiles(self):
        gate = CrossPathGate(
            embedding_dim=2, num_actions=5, hidden_width=4, endpoint_count=3
        )
        probabilities = gate.probabilities(torch.randn(2, 17))
        self.assertEqual(tuple(probabilities.shape), (3,))

    def test_one_action_aggregates_all_official_cutoffs(self):
        cutoff_utilities = np.asarray(
            [[0.0, 1.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.5, 0.5]]
        )
        aggregate = aggregate_cutoff_utilities(cutoff_utilities)
        np.testing.assert_allclose(aggregate, [0.0, 0.5, 1.0 / 6.0])
        self.assertEqual(int(np.argmax(aggregate)), 1)

        realized = realized_cutoff_utilities(
            target_ranks=np.asarray([11, 1, 6]),
            cutoffs=(1, 5, 10),
            regression_cost=2.0,
        )
        np.testing.assert_array_equal(
            realized,
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        )

    def test_batched_listwise_loss_matches_record_mean(self):
        torch.manual_seed(11)
        gate = CrossPathGate(embedding_dim=2, num_actions=3, hidden_width=4)
        records = [torch.randn(2, 14), torch.empty(0, 14), torch.randn(3, 14)]
        targets = [1, 0, 3]
        expected = torch.stack(
            [listwise_loss(gate, features, target) for features, target in zip(records, targets)]
        ).mean()
        actual = batch_listwise_loss(gate, records, targets)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
