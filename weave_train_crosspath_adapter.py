#!/usr/bin/env python3
"""Train lightweight residual adapters with a four-path contrastive loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualAdapter(nn.Module):
    def __init__(self, dim: int, rank: int, scale: float) -> None:
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.scale = scale
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(values)))
        return F.normalize(values + self.scale * residual, dim=-1)


class CrossPathAdapter(nn.Module):
    """One shared query/gallery adapter per frozen endpoint."""

    def __init__(self, dim: int, rank: int, scale: float) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(
            [ResidualAdapter(dim, rank, scale), ResidualAdapter(dim, rank, scale)]
        )

    def forward(self, values: torch.Tensor, endpoint: int) -> torch.Tensor:
        return self.adapters[endpoint](values)


def load_metadata(embedding_dir: Path) -> tuple[list[str], list[dict]]:
    gallery_ids = json.loads((embedding_dir / "gallery_ids.json").read_text())
    queries = [
        json.loads(line)
        for line in (embedding_dir / "queries.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return gallery_ids, queries


def query_indices(queries: list[dict], val_percent: int) -> tuple[np.ndarray, np.ndarray]:
    train, val = [], []
    for index, row in enumerate(queries):
        key = f"{row['source_id']}\0{row['target_id']}\0{index}".encode()
        bucket = int(hashlib.sha256(key).hexdigest()[:8], 16) % 100
        (val if bucket < val_percent else train).append(index)
    if not train or not val:
        raise ValueError("the deterministic train/validation split is empty")
    return np.asarray(train), np.asarray(val)


def target_and_source_indices(
    gallery_ids: list[str], queries: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    index = {str(image_id): i for i, image_id in enumerate(gallery_ids)}
    try:
        targets = np.asarray([index[str(row["target_id"])] for row in queries])
        sources = np.asarray([index[str(row["source_id"])] for row in queries])
    except KeyError as error:
        raise ValueError(f"query image is absent from gallery: {error}") from error
    return targets, sources


def path_scores(
    q0: torch.Tensor,
    q1: torch.Tensor,
    g0: torch.Tensor,
    g1: torch.Tensor,
) -> dict[str, torch.Tensor]:
    scores = {
        "q0_g0": q0 @ g0.T,
        "q0_g1": q0 @ g1.T,
        "q1_g0": q1 @ g0.T,
        "q1_g1": q1 @ g1.T,
    }
    scores["diagonal_mean"] = (scores["q0_g0"] + scores["q1_g1"]) / 2
    scores["cross_mean"] = (scores["q0_g1"] + scores["q1_g0"]) / 2
    scores["all_mean"] = (
        scores["q0_g0"] + scores["q0_g1"] + scores["q1_g0"] + scores["q1_g1"]
    ) / 4
    return scores


def recall_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    sources: torch.Tensor,
    cutoffs: tuple[int, ...],
) -> dict[str, float]:
    scores = scores.clone()
    scores[torch.arange(scores.shape[0], device=scores.device), sources] = -torch.inf
    target_scores = scores.gather(1, targets[:, None])
    gallery_order = torch.arange(scores.shape[1], device=scores.device)[None, :]
    ranks = 1 + (scores > target_scores).sum(1)
    ranks += ((scores == target_scores) & (gallery_order < targets[:, None])).sum(1)
    return {f"R@{cutoff}": 100.0 * (ranks <= cutoff).float().mean().item() for cutoff in cutoffs}


def evaluate(
    model: CrossPathAdapter,
    arrays: dict[str, np.ndarray],
    queries: list[dict],
    gallery_ids: list[str],
    indices: np.ndarray,
    device: torch.device,
    cutoffs: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    targets, sources = target_and_source_indices(gallery_ids, queries)
    selected = torch.as_tensor(indices, dtype=torch.long)
    with torch.no_grad():
        g0 = model(torch.from_numpy(arrays["base_gallery"]).to(device), 0)
        g1 = model(torch.from_numpy(arrays["correction_gallery"]).to(device), 1)
        q0 = model(torch.from_numpy(arrays["base_queries"])[selected].to(device), 0)
        q1 = model(torch.from_numpy(arrays["correction_queries"])[selected].to(device), 1)
        scores = path_scores(q0, q1, g0, g1)
        target_tensor = torch.from_numpy(targets[indices]).to(device)
        source_tensor = torch.from_numpy(sources[indices]).to(device)
        return {
            name: recall_metrics(value, target_tensor, source_tensor, cutoffs)
            for name, value in scores.items()
        }


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    return {
        name: np.load(path / f"{name}.npy").astype(np.float32, copy=False)
        for name in ("base_gallery", "base_queries", "correction_gallery", "correction_queries")
    }


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    arrays = load_arrays(args.train_embeddings)
    gallery_ids, queries = load_metadata(args.train_embeddings)
    targets, sources = target_and_source_indices(gallery_ids, queries)
    train_indices, val_indices = query_indices(queries, args.val_percent)
    dim = arrays["base_queries"].shape[1]
    model = CrossPathAdapter(dim, args.rank, args.residual_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    path_names = {
        "full": ("q0_g0", "q0_g1", "q1_g0", "q1_g1"),
        "diagonal": ("q0_g0", "q1_g1"),
        "cross": ("q0_g1", "q1_g0"),
    }[args.objective]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "method": "crosspath_residual_adapter",
        "train_embeddings": str(args.train_embeddings.resolve()),
        "official_embeddings": str(args.official_embeddings.resolve()),
        "objective": args.objective,
        "rank": args.rank,
        "residual_scale": args.residual_scale,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "negative_gallery_size": args.negative_gallery_size,
        "val_percent": args.val_percent,
        "selection_reducer": args.selection_reducer,
        "selection_metric": "R@10+R@50",
        "seed": args.seed,
        "train_queries": int(len(train_indices)),
        "validation_queries": int(len(val_indices)),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    tensor_arrays = {name: torch.from_numpy(value) for name, value in arrays.items()}
    best_score = -float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = rng.permutation(train_indices)
        losses = []
        for start in range(0, len(shuffled), args.batch_size):
            batch = shuffled[start : start + args.batch_size]
            positives = np.unique(targets[batch])
            available = np.setdiff1d(np.arange(len(gallery_ids)), positives, assume_unique=False)
            negative_count = min(args.negative_gallery_size, len(available))
            candidates = np.concatenate([positives, rng.choice(available, negative_count, replace=False)])
            rng.shuffle(candidates)
            candidate_lookup = {gallery_index: i for i, gallery_index in enumerate(candidates.tolist())}
            labels = torch.tensor([candidate_lookup[value] for value in targets[batch]], device=device)

            q0 = model(tensor_arrays["base_queries"][batch].to(device), 0)
            q1 = model(tensor_arrays["correction_queries"][batch].to(device), 1)
            g0 = model(tensor_arrays["base_gallery"][candidates].to(device), 0)
            g1 = model(tensor_arrays["correction_gallery"][candidates].to(device), 1)
            scores = path_scores(q0, q1, g0, g1)
            for row, source in enumerate(sources[batch]):
                if int(source) in candidate_lookup:
                    for value in scores.values():
                        value[row, candidate_lookup[int(source)]] = -1e4
            loss = sum(F.cross_entropy(scores[name] / args.temperature, labels) for name in path_names)
            loss = loss / len(path_names)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        metrics = evaluate(
            model, arrays, queries, gallery_ids, val_indices, device, (1, 10, 50)
        )
        selected = metrics[args.selection_reducer]
        selection_score = selected["R@10"] + selected["R@50"]
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "selection_score": selection_score,
            "validation": metrics,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if selection_score > best_score:
            best_score = selection_score
            torch.save(model.state_dict(), args.output_dir / "best_adapter.pt")

    model.load_state_dict(torch.load(args.output_dir / "best_adapter.pt", map_location=device, weights_only=True))
    model.eval()
    official_arrays = load_arrays(args.official_embeddings)
    official_gallery_ids, official_queries = load_metadata(args.official_embeddings)
    official_indices = np.arange(len(official_queries))
    official = evaluate(
        model,
        official_arrays,
        official_queries,
        official_gallery_ids,
        official_indices,
        device,
        (1, 10, 50),
    )
    result = {"best_validation_score": best_score, "official": official}
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-embeddings", type=Path, required=True)
    parser.add_argument("--official-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objective", choices=("full", "diagonal", "cross"), default="full")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--negative-gallery-size", type=int, default=2048)
    parser.add_argument("--val-percent", type=int, default=15)
    parser.add_argument("--selection-reducer", choices=("all_mean", "cross_mean", "diagonal_mean"), default="all_mean")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.rank <= 0 or args.batch_size <= 0 or args.negative_gallery_size <= 0:
        parser.error("rank and batch sizes must be positive")
    if not 0 < args.val_percent < 100:
        parser.error("val-percent must be between 1 and 99")
    return args


if __name__ == "__main__":
    train(parse_args())
