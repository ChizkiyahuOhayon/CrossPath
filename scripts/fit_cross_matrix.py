#!/usr/bin/env python3
"""Fit four global compatibility weights on train embeddings and evaluate validation."""

import argparse
import json
from pathlib import Path

import numpy as np


CATEGORIES = ("dress", "shirt", "toptee")
PATHS = ("q0_g0", "q0_g1", "q1_g0", "q1_g1")


def load_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_problem(embedding_dir, device, exclude_source):
    import torch
    import torch.nn.functional as functional

    gallery_ids = json.loads((embedding_dir / "gallery_ids.json").read_text())
    gallery_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
    rows = load_rows(embedding_dir / "queries.jsonl")
    target_indices = torch.tensor(
        [gallery_index[row["target_id"]] for row in rows],
        dtype=torch.long,
        device=device,
    )
    source_indices = None
    if exclude_source:
        source_indices = torch.tensor(
            [gallery_index[row["source_id"]] for row in rows],
            dtype=torch.long,
            device=device,
        )
    arrays = {
        name: np.load(embedding_dir / f"{name}.npy")
        for name in (
            "base_gallery",
            "correction_gallery",
            "base_queries",
            "correction_queries",
        )
    }
    tensors = {
        name: functional.normalize(
            torch.as_tensor(array, dtype=torch.float32, device=device), dim=-1
        )
        for name, array in arrays.items()
    }
    return tensors, target_indices, source_indices


def compatibility_scores(tensors):
    import torch

    query0 = tensors["base_queries"]
    query1 = tensors["correction_queries"]
    gallery0 = tensors["base_gallery"]
    gallery1 = tensors["correction_gallery"]
    with torch.no_grad():
        return torch.stack(
            (
                query0 @ gallery0.T,
                query0 @ gallery1.T,
                query1 @ gallery0.T,
                query1 @ gallery1.T,
            )
        )


def exclude_sources(scores, source_indices):
    if source_indices is None:
        return
    import torch

    rows = torch.arange(scores.shape[1], device=scores.device)
    scores[:, rows, source_indices] = -1e4


def fit_weights(scores, target_indices, epochs, learning_rate):
    import torch
    import torch.nn.functional as functional

    logits = torch.zeros(scores.shape[0], device=scores.device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.softmax(logits, dim=0)
        combined = torch.einsum("p,pqg->qg", weights, scores)
        loss = functional.cross_entropy(combined, target_indices)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return torch.softmax(logits.detach(), dim=0), losses


def target_ranks(scores, target_indices):
    import torch

    rows = torch.arange(scores.shape[0], device=scores.device)
    target_scores = scores[rows, target_indices]
    candidate_indices = torch.arange(scores.shape[1], device=scores.device)[None, :]
    ties_before_target = (scores == target_scores[:, None]) & (
        candidate_indices < target_indices[:, None]
    )
    return 1 + torch.sum(
        (scores > target_scores[:, None]) | ties_before_target, dim=1
    )


def evaluate(scores, target_indices, weights, cutoffs):
    import torch

    with torch.no_grad():
        combined = torch.einsum("p,pqg->qg", weights, scores)
        ranks = target_ranks(combined, target_indices)
    return {
        f"R@{cutoff}": round(
            float((ranks <= cutoff).float().mean().cpu() * 100.0), 6
        )
        for cutoff in cutoffs
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-run-root", required=True, type=Path)
    parser.add_argument("--official-run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 10, 50])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs and learning rate must be positive")
    if any(cutoff <= 0 for cutoff in args.cutoffs):
        raise ValueError("cutoffs must be positive")

    categories = {}
    for category in CATEGORIES:
        internal_dir = args.internal_run_root / category / "internal" / "embeddings"
        official_dir = args.official_run_root / category / "official" / "embeddings"
        tensors, targets, sources = load_problem(internal_dir, args.device, True)
        train_scores = compatibility_scores(tensors)
        exclude_sources(train_scores, sources)
        weights, losses = fit_weights(
            train_scores, targets, args.epochs, args.learning_rate
        )
        del tensors, train_scores, targets, sources

        tensors, targets, sources = load_problem(official_dir, args.device, True)
        official_scores = compatibility_scores(tensors)
        exclude_sources(official_scores, sources)
        metrics = evaluate(official_scores, targets, weights, tuple(args.cutoffs))
        categories[category] = {
            "weights": {
                path: round(float(weight), 8)
                for path, weight in zip(PATHS, weights.cpu())
            },
            "initial_train_loss": losses[0],
            "final_train_loss": losses[-1],
            "metrics": metrics,
        }
        print(category, categories[category], flush=True)

    average = {
        metric: round(
            sum(categories[category]["metrics"][metric] for category in CATEGORIES)
            / len(CATEGORIES),
            6,
        )
        for metric in categories[CATEGORIES[0]]["metrics"]
    }
    report = {
        "paths": list(PATHS),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "categories": categories,
        "average": average,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(average, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
