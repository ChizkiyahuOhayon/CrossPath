#!/usr/bin/env python3
"""Core, model-agnostic contracts for CrossPath.

The two frozen checkpoints need not share a calibrated score scale.  CrossPath
therefore interpolates deterministic endpoint rank percentiles, not raw model
scores.  This module deliberately contains no encoder or gate training code;
it defines the path, cutoff trace, utility, and exact-fallback behavior that
the later GPU feature extractor must satisfy.
"""

from dataclasses import dataclass

import numpy as np


DEFAULT_ALPHAS = np.linspace(0.0, 1.0, 9)


@dataclass(frozen=True)
class RankPath:
    candidate_ids: np.ndarray
    alphas: np.ndarray
    endpoint_percentiles: np.ndarray
    orders: np.ndarray


@dataclass(frozen=True)
class BoundaryTrace:
    candidate_indices: np.ndarray
    membership: np.ndarray


def _as_finite_scores(scores, expected_size, name):
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},)")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def _deterministic_order(scores, candidate_ids):
    return np.lexsort((candidate_ids, -scores)).astype(np.int64, copy=False)


def _rank_percentiles(scores, candidate_ids):
    order = _deterministic_order(scores, candidate_ids)
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(order.size, dtype=np.int64)
    return 1.0 - ranks / (order.size - 1), order


def build_rank_path(
    candidate_ids,
    base_scores,
    correction_scores,
    alphas=DEFAULT_ALPHAS,
):
    """Build a scale-invariant path with deterministic, exact endpoints."""
    ids = np.asarray(candidate_ids, dtype=str)
    if ids.ndim != 1 or ids.size < 2:
        raise ValueError("candidate_ids must be one-dimensional with at least 2 items")
    if np.unique(ids).size != ids.size:
        raise ValueError("candidate_ids must be unique")

    base = _as_finite_scores(base_scores, ids.size, "base_scores")
    correction = _as_finite_scores(
        correction_scores, ids.size, "correction_scores"
    )
    grid = np.asarray(alphas, dtype=np.float64)
    if (
        grid.ndim != 1
        or grid.size < 2
        or not np.isfinite(grid).all()
        or grid[0] != 0.0
        or grid[-1] != 1.0
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("alphas must increase strictly from 0 to 1")

    base_percentiles, base_order = _rank_percentiles(base, ids)
    correction_percentiles, correction_order = _rank_percentiles(correction, ids)
    endpoint_percentiles = np.stack(
        [base_percentiles, correction_percentiles], axis=0
    )

    orders = []
    for index, alpha in enumerate(grid):
        if index == 0:
            order = base_order
        elif index == grid.size - 1:
            order = correction_order
        else:
            path_scores = (
                (1.0 - alpha) * base_percentiles
                + alpha * correction_percentiles
            )
            order = _deterministic_order(path_scores, ids)
        orders.append(order)

    return RankPath(
        candidate_ids=ids,
        alphas=grid,
        endpoint_percentiles=endpoint_percentiles,
        orders=np.stack(orders, axis=0),
    )


def boundary_trace(path, k):
    """Return candidates whose top-k membership changes along ``path``."""
    if not 1 <= k <= path.candidate_ids.size:
        raise ValueError("k must be between 1 and the gallery size")

    membership = np.zeros(
        (path.candidate_ids.size, path.alphas.size), dtype=np.int8
    )
    for action, order in enumerate(path.orders):
        membership[order[:k], action] = 1
    crossing = np.flatnonzero(np.any(membership != membership[:, :1], axis=1))
    if crossing.size:
        crossing = crossing[np.argsort(path.candidate_ids[crossing], kind="stable")]
    return BoundaryTrace(
        candidate_indices=crossing,
        membership=membership[crossing],
    )


def estimate_utilities(membership, responsibility, regression_cost=2.0):
    """Estimate recovery-minus-regression utility for every path action.

    ``responsibility`` may contain one probability per crossing candidate or
    one additional final probability for the none class.  The none class has
    zero utility by definition.
    """
    trace = np.asarray(membership, dtype=np.int8)
    if trace.ndim != 2 or trace.shape[1] < 2:
        raise ValueError("membership must have shape [boundary_candidates, actions]")
    if not np.isin(trace, (0, 1)).all():
        raise ValueError("membership values must be binary")
    if not np.isfinite(regression_cost) or regression_cost <= 0.0:
        raise ValueError("regression_cost must be positive and finite")

    probability = np.asarray(responsibility, dtype=np.float64)
    if probability.shape not in ((trace.shape[0],), (trace.shape[0] + 1,)):
        raise ValueError("responsibility has the wrong length")
    if (
        not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or not np.isclose(probability.sum(), 1.0)
    ):
        raise ValueError("responsibility must be a probability vector")

    candidate_probability = probability[: trace.shape[0]]
    base = trace[:, :1]
    change_utility = (
        (1 - base) * trace - regression_cost * base * (1 - trace)
    )
    return (candidate_probability[:, None] * change_utility).sum(axis=0)


def select_action(utilities, threshold, path_orders):
    """Select the smallest best action above threshold, else return A1."""
    values = np.asarray(utilities, dtype=np.float64)
    orders = np.asarray(path_orders)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("utilities must be a finite one-dimensional array")
    if orders.ndim != 2 or orders.shape[0] != values.size:
        raise ValueError("path_orders must have one row per action")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    best_action = int(np.argmax(values))
    if best_action == 0 or values[best_action] <= threshold:
        best_action = 0
    return best_action, orders[best_action]


def calibrate_threshold(predicted_utilities, realized_utilities):
    """Choose the development threshold, breaking utility ties conservatively."""
    predicted = np.asarray(predicted_utilities, dtype=np.float64)
    realized = np.asarray(realized_utilities, dtype=np.float64)
    if (
        predicted.ndim != 2
        or predicted.shape != realized.shape
        or predicted.shape[0] == 0
        or predicted.shape[1] < 2
        or not np.isfinite(predicted).all()
        or not np.isfinite(realized).all()
    ):
        raise ValueError("utility tables must be aligned, finite [queries, actions]")

    best_actions = np.argmax(predicted, axis=1)
    best_values = predicted[np.arange(predicted.shape[0]), best_actions]
    if np.any(best_values < 0.0):
        raise ValueError("action zero must keep maximum predicted utility non-negative")
    thresholds = np.unique(np.concatenate(([0.0], best_values)))

    best_threshold = 0.0
    best_mean = -np.inf
    query_indices = np.arange(predicted.shape[0])
    for threshold in thresholds:
        selected = np.where(best_values > threshold, best_actions, 0)
        mean_utility = float(realized[query_indices, selected].mean())
        if mean_utility > best_mean or (
            np.isclose(mean_utility, best_mean) and threshold > best_threshold
        ):
            best_threshold = float(threshold)
            best_mean = mean_utility
    return best_threshold, best_mean
