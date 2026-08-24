import numpy as np
import torch

from weave_train_crosspath_adapter import (
    CrossPathAdapter,
    path_scores,
    query_indices,
    recall_metrics,
    target_and_source_indices,
)


def test_adapter_is_identity_at_initialization():
    model = CrossPathAdapter(dim=4, rank=2, scale=0.1)
    values = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    assert torch.allclose(model(values, 0), values)
    assert torch.allclose(model(values, 1), values)


def test_all_mean_matches_mean_embedding_dot_product():
    tensors = [torch.nn.functional.normalize(torch.randn(3, 4), dim=-1) for _ in range(4)]
    q0, q1, g0, g1 = tensors
    scores = path_scores(q0, q1, g0, g1)
    expected = ((q0 + q1) / 2) @ ((g0 + g1) / 2).T
    assert torch.allclose(scores["all_mean"], expected)


def test_recall_excludes_source_and_uses_gallery_order_for_ties():
    scores = torch.tensor([[0.9, 0.8, 0.8], [0.7, 0.8, 0.9]])
    targets = torch.tensor([2, 2])
    sources = torch.tensor([0, 0])
    metrics = recall_metrics(scores, targets, sources, (1, 2))
    assert metrics == {"R@1": 50.0, "R@2": 100.0}


def test_metadata_indices_and_split_are_deterministic():
    gallery = ["a", "b", "c"]
    queries = [
        {"source_id": "a", "target_id": "b"},
        {"source_id": "b", "target_id": "c"},
    ] * 20
    targets, sources = target_and_source_indices(gallery, queries)
    assert np.array_equal(targets[:2], [1, 2])
    assert np.array_equal(sources[:2], [0, 1])
    first = query_indices(queries, 20)
    second = query_indices(queries, 20)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
