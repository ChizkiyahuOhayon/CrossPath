#!/usr/bin/env python3
"""Core components for conservative per-query routing on CIRR."""

import hashlib

import numpy as np
import torch
from torch import nn


GLOBAL_K = 5
SUBSET_K = 1


def action_names(alphas):
    return ["mcot"] + [
        f"{branch}_a{alpha:g}"
        for branch in ("q1_g0", "cross_mean")
        for alpha in alphas
    ]


def pair_partition(row, seed):
    left, right = sorted((str(row["source_id"]), str(row["target_id"])))
    value = int.from_bytes(
        hashlib.sha256(f"{seed}\0{left}\0{right}".encode()).digest()[:8], "big"
    ) / 2**64
    if value < 0.70:
        return 0
    if value < 0.85:
        return 1
    return 2


def score_paths(query0, query1, gallery0, gallery1):
    score10 = query1 @ gallery0.T
    score11 = query1 @ gallery1.T
    score01 = query0 @ gallery1.T
    return {
        "q1_g1": score11,
        "q1_g0": score10,
        "cross_mean": 0.5 * (score01 + score10),
    }


def action_scores(paths, alphas, source_indices):
    base = paths["q1_g1"]
    scores = [base]
    for branch in ("q1_g0", "cross_mean"):
        scores.extend((1.0 - alpha) * base + alpha * paths[branch] for alpha in alphas)
    stacked = torch.stack(scores, dim=1)
    rows = torch.arange(len(stacked), device=stacked.device)
    stacked[rows, :, source_indices] = -torch.inf
    return stacked


def target_ranks(scores, target_indices):
    rows = torch.arange(len(scores), device=scores.device)
    targets = target_indices.to(scores.device)
    target_scores = scores[rows, targets]
    indices = torch.arange(scores.shape[1], device=scores.device).unsqueeze(0)
    ties_before = (scores == target_scores.unsqueeze(1)) & (indices < targets.unsqueeze(1))
    return 1 + ((scores > target_scores.unsqueeze(1)) | ties_before).sum(dim=1)


def subset_target_ranks(scores, groups, target_indices, source_indices):
    safe_groups = groups.clamp_min(0)
    member_scores = scores.gather(1, safe_groups)
    target_scores = scores[
        torch.arange(len(scores), device=scores.device), target_indices
    ]
    valid = (groups >= 0) & (groups != source_indices.unsqueeze(1))
    if not torch.all(torch.any((groups == target_indices.unsqueeze(1)) & valid, dim=1)):
        raise ValueError("target is absent from a CIRR subset")
    ties_before = (member_scores == target_scores.unsqueeze(1)) & (
        groups < target_indices.unsqueeze(1)
    )
    return 1 + (
        valid & ((member_scores > target_scores.unsqueeze(1)) | ties_before)
    ).sum(dim=1)


def query_features(query0, query1, paths, source_indices, topk=10):
    if topk <= 1:
        raise ValueError("topk must be greater than one")
    rows = torch.arange(len(query0), device=query0.device)
    top_values, top_indices = [], []
    for name in ("q1_g1", "q1_g0", "cross_mean"):
        scores = paths[name].clone()
        scores[rows, source_indices] = -torch.inf
        values, indices = torch.topk(scores, topk, dim=1, sorted=True)
        top_values.append(values)
        top_indices.append(indices)

    statistics = []
    positions = (0, 1, min(4, topk - 1), topk - 1)
    for values in top_values:
        selected = values[:, positions]
        statistics.append(
            torch.cat(
                [
                    selected,
                    values[:, :1] - values[:, 1:2],
                    values[:, :1] - values[:, positions[2] : positions[2] + 1],
                    values[:, :1] - values[:, -1:],
                    values.mean(dim=1, keepdim=True),
                    values.std(dim=1, keepdim=True),
                ],
                dim=1,
            )
        )
    agreements = []
    for indices in top_indices[1:]:
        agreements.append((top_indices[0][:, :1] == indices[:, :1]).float())
        overlap = (
            top_indices[0].unsqueeze(2) == indices.unsqueeze(1)
        ).any(dim=2).float().mean(dim=1, keepdim=True)
        agreements.append(overlap)
    cosine = (query0 * query1).sum(dim=1, keepdim=True)
    return torch.cat(
        [
            query0,
            query1,
            torch.abs(query0 - query1),
            query0 * query1,
            cosine,
            *statistics,
            *agreements,
        ],
        dim=1,
    )


def soft_oracle_targets(global_ranks, subset_ranks):
    utility = (global_ranks <= GLOBAL_K).astype(np.int8) + (
        subset_ranks <= SUBSET_K
    ).astype(np.int8)
    targets = np.zeros_like(utility, dtype=np.float32)
    maxima = utility.max(axis=1)
    base_is_best = utility[:, 0] == maxima
    targets[base_is_best, 0] = 1.0
    for row in np.flatnonzero(~base_is_best):
        winners = np.flatnonzero(utility[row] == maxima[row])
        targets[row, winners] = 1.0 / len(winners)
    return targets


def choose_actions(logits, threshold):
    logits = np.asarray(logits)
    best = np.argmax(logits, axis=1)
    margins = logits[np.arange(len(logits)), best] - logits[:, 0]
    return np.where((best != 0) & (margins >= threshold), best, 0)


def cirr_avg(global_ranks, subset_ranks, actions):
    rows = np.arange(len(actions))
    global_recall = np.mean(global_ranks[rows, actions] <= GLOBAL_K) * 100.0
    subset_recall = np.mean(subset_ranks[rows, actions] <= SUBSET_K) * 100.0
    return 0.5 * (global_recall + subset_recall)


class CIRRQueryRouter(nn.Module):
    def __init__(self, mean, std, action_count, hidden_width=256):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)
        self.network = nn.Sequential(
            nn.Linear(len(mean), hidden_width),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_width, action_count),
        )

    def forward(self, features):
        return self.network((features - self.feature_mean) / self.feature_std)
