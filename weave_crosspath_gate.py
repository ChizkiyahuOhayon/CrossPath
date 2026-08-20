#!/usr/bin/env python3
"""The minimal listwise responsibility gate used by CrossPath."""

import numpy as np
import torch
from torch import nn


def build_candidate_features(
    query_embedding,
    candidate_embeddings,
    endpoint_percentiles,
    membership,
    k,
):
    """Build [q, d, q*d, |q-d|, p0, p1, trace, k/10] features."""
    query = np.asarray(query_embedding, dtype=np.float32)
    documents = np.asarray(candidate_embeddings, dtype=np.float32)
    percentiles = np.asarray(endpoint_percentiles, dtype=np.float32)
    trace = np.asarray(membership, dtype=np.float32)
    if query.ndim != 1:
        raise ValueError("query_embedding must be one-dimensional")
    if documents.ndim != 2 or documents.shape[1] != query.size:
        raise ValueError("candidate_embeddings must have shape [candidates, dim]")
    if percentiles.ndim != 2 or percentiles.shape[0] != documents.shape[0]:
        raise ValueError("endpoint_percentiles must have shape [candidates, endpoints]")
    if trace.ndim != 2 or trace.shape[0] != documents.shape[0]:
        raise ValueError("membership must have shape [candidates, actions]")
    if not np.isin(trace, (0.0, 1.0)).all():
        raise ValueError("membership must be binary")
    if k not in (1, 5, 10):
        raise ValueError("k must be one of 1, 5, or 10")

    tiled_query = np.broadcast_to(query, documents.shape)
    cutoff = np.full((documents.shape[0], 1), k / 10.0, dtype=np.float32)
    return np.concatenate(
        [
            tiled_query,
            documents,
            tiled_query * documents,
            np.abs(tiled_query - documents),
            percentiles,
            trace,
            cutoff,
        ],
        axis=1,
    )


def listwise_target(boundary_candidate_indices, target_index):
    """Return the boundary position, or the final none-class position."""
    boundary = np.asarray(boundary_candidate_indices, dtype=np.int64)
    if boundary.ndim != 1 or np.unique(boundary).size != boundary.size:
        raise ValueError("boundary_candidate_indices must be unique and one-dimensional")
    positions = np.flatnonzero(boundary == int(target_index))
    return int(positions[0]) if positions.size else int(boundary.size)


def aggregate_cutoff_utilities(cutoff_utilities):
    """Average cutoff utilities so one query emits one benchmark ranking."""
    values = np.asarray(cutoff_utilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("cutoff_utilities must have shape [cutoffs, actions]")
    if not np.isfinite(values).all():
        raise ValueError("cutoff utilities must be finite")
    return values.mean(axis=0)


def realized_cutoff_utilities(target_ranks, cutoffs=(1, 5, 10), regression_cost=2.0):
    """Return fully observed recovery-minus-regression utility per cutoff/action."""
    ranks = np.asarray(target_ranks, dtype=np.int64)
    ks = np.asarray(cutoffs, dtype=np.int64)
    if ranks.ndim != 1 or ranks.size < 2 or np.any(ranks <= 0):
        raise ValueError("target_ranks must be positive with one value per action")
    if ks.ndim != 1 or ks.size == 0 or np.any(ks <= 0):
        raise ValueError("cutoffs must be positive")
    if not np.isfinite(regression_cost) or regression_cost <= 0.0:
        raise ValueError("regression_cost must be positive and finite")
    hits = ranks[None, :] <= ks[:, None]
    base = hits[:, :1]
    return ((~base) & hits).astype(np.float64) - regression_cost * (
        base & (~hits)
    ).astype(np.float64)


class CrossPathGate(nn.Module):
    """Two-layer candidate scorer with one learned global none logit."""

    def __init__(
        self, embedding_dim, num_actions=9, hidden_width=128, endpoint_count=2
    ):
        super().__init__()
        if (
            embedding_dim <= 0
            or num_actions < 2
            or hidden_width <= 0
            or endpoint_count < 2
        ):
            raise ValueError("gate dimensions must be positive")
        input_dim = 4 * embedding_dim + endpoint_count + num_actions + 1
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )
        self.none_logit = nn.Parameter(torch.zeros(()))

    def forward(self, candidate_features):
        if candidate_features.ndim != 2:
            raise ValueError("candidate_features must be two-dimensional")
        candidate_logits = self.scorer(candidate_features).squeeze(-1)
        return torch.cat([candidate_logits, self.none_logit.reshape(1)])

    def probabilities(self, candidate_features):
        return torch.softmax(self(candidate_features), dim=0)


def listwise_loss(gate, candidate_features, target_position):
    logits = gate(candidate_features)
    if not 0 <= target_position < logits.numel():
        raise ValueError("target_position is outside the candidate-plus-none list")
    target = torch.tensor([target_position], dtype=torch.long, device=logits.device)
    return nn.functional.cross_entropy(logits.unsqueeze(0), target)


def batch_listwise_loss(gate, feature_records, target_positions):
    """Score concatenated candidates once and average variable-list losses."""
    if not feature_records or len(feature_records) != len(target_positions):
        raise ValueError("feature records and targets must be non-empty and aligned")
    feature_dim = feature_records[0].shape[1]
    for features in feature_records:
        if features.ndim != 2 or features.shape[1] != feature_dim:
            raise ValueError("feature records must share a two-dimensional feature size")

    lengths = [features.shape[0] for features in feature_records]
    nonempty = [features for features in feature_records if features.shape[0]]
    if nonempty:
        candidate_logits = gate.scorer(torch.cat(nonempty, dim=0)).squeeze(-1)
    else:
        candidate_logits = gate.none_logit.new_empty(0)

    losses = []
    offset = 0
    for length, target in zip(lengths, target_positions):
        logits = torch.cat(
            [candidate_logits[offset : offset + length], gate.none_logit.reshape(1)]
        )
        if not 0 <= target < logits.numel():
            raise ValueError("target position is outside a candidate-plus-none list")
        losses.append(-torch.log_softmax(logits, dim=0)[target])
        offset += length
    return torch.stack(losses).mean()
