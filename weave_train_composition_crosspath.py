#!/usr/bin/env python3
"""Train a full-gallery composition-path head over frozen DQU branches."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from weave_train_crosspath_adapter import (
    load_metadata,
    query_indices,
    recall_metrics,
    target_and_source_indices,
)


class CompositionCrossPath(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(4 * dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        text: torch.Tensor,
        visual: torch.Tensor,
        original_lambda: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat(
            (text, visual, text - visual, text * visual, original_lambda), dim=-1
        )
        correction = 0.5 * torch.tanh(self.head(features))
        mixing = torch.clamp(original_lambda + correction, 0.0, 1.0)
        query = F.normalize(mixing * text + (1.0 - mixing) * visual, dim=-1)
        return query, mixing.squeeze(-1)


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    return {
        name: np.load(path / f"{name}.npy").astype(np.float32, copy=False)
        for name in ("gallery", "text", "visual", "original_lambda")
    }


def original_query(arrays: dict[str, np.ndarray]) -> torch.Tensor:
    text = torch.from_numpy(arrays["text"])
    visual = torch.from_numpy(arrays["visual"])
    mixing = torch.from_numpy(arrays["original_lambda"])
    return F.normalize(mixing * text + (1.0 - mixing) * visual, dim=-1)


def evaluate(
    model: CompositionCrossPath,
    arrays: dict[str, np.ndarray],
    queries: list[dict],
    gallery_ids: list[str],
    indices: np.ndarray,
    device: torch.device,
) -> dict:
    targets, sources = target_and_source_indices(gallery_ids, queries)
    selected = torch.from_numpy(indices)
    gallery = torch.from_numpy(arrays["gallery"]).to(device)
    text = torch.from_numpy(arrays["text"])[selected].to(device)
    visual = torch.from_numpy(arrays["visual"])[selected].to(device)
    original_lambda = torch.from_numpy(arrays["original_lambda"])[selected].to(device)
    with torch.no_grad():
        corrected, mixing = model(text, visual, original_lambda)
        original = F.normalize(original_lambda * text + (1.0 - original_lambda) * visual, dim=-1)
        target_tensor = torch.from_numpy(targets[indices]).to(device)
        source_tensor = torch.from_numpy(sources[indices]).to(device)
        return {
            "original": recall_metrics(original @ gallery.T, target_tensor, source_tensor, (1, 10, 50)),
            "crosspath": recall_metrics(corrected @ gallery.T, target_tensor, source_tensor, (1, 10, 50)),
            "lambda": {
                "original_mean": float(original_lambda.mean()),
                "crosspath_mean": float(mixing.mean()),
                "crosspath_std": float(mixing.std()),
                "crosspath_min": float(mixing.min()),
                "crosspath_max": float(mixing.max()),
            },
        }


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    arrays = load_arrays(args.train_features)
    gallery_ids, queries = load_metadata(args.train_features)
    targets, sources = target_and_source_indices(gallery_ids, queries)
    if args.val_percent == 0:
        train_indices = np.arange(len(queries))
        val_indices = np.asarray([], dtype=np.int64)
    else:
        train_indices, val_indices = query_indices(queries, args.val_percent)
    model = CompositionCrossPath(arrays["text"].shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "method": "full_gallery_composition_crosspath",
        "train_features": str(args.train_features.resolve()),
        "official_features": str(args.official_features.resolve()),
        "hidden_dim": args.hidden_dim,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "negative_gallery_size": args.negative_gallery_size,
        "val_percent": args.val_percent,
        "selection_metric": (
            "fixed epoch on 100% train" if args.val_percent == 0 else "CrossPath R@10+R@50"
        ),
        "seed": args.seed,
        "train_queries": int(len(train_indices)),
        "validation_queries": int(len(val_indices)),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    tensors = {name: torch.from_numpy(value) for name, value in arrays.items()}
    best_score, history = -float("inf"), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled, losses = rng.permutation(train_indices), []
        for start in range(0, len(shuffled), args.batch_size):
            batch = shuffled[start : start + args.batch_size]
            positives = np.unique(targets[batch])
            available = np.setdiff1d(np.arange(len(gallery_ids)), positives, assume_unique=False)
            negative_count = min(args.negative_gallery_size, len(available))
            candidates = np.concatenate([positives, rng.choice(available, negative_count, replace=False)])
            rng.shuffle(candidates)
            lookup = {gallery_index: i for i, gallery_index in enumerate(candidates.tolist())}
            labels = torch.tensor([lookup[value] for value in targets[batch]], device=device)

            query, _ = model(
                tensors["text"][batch].to(device),
                tensors["visual"][batch].to(device),
                tensors["original_lambda"][batch].to(device),
            )
            scores = query @ tensors["gallery"][candidates].to(device).T / args.temperature
            for row, source in enumerate(sources[batch]):
                if int(source) in lookup:
                    scores[row, lookup[int(source)]] = -1e4
            loss = F.cross_entropy(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        if len(val_indices):
            validation = evaluate(model, arrays, queries, gallery_ids, val_indices, device)
            score = validation["crosspath"]["R@10"] + validation["crosspath"]["R@50"]
        else:
            validation = None
            score = float(epoch)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "selection_score": score, "validation": validation}
        history.append(row)
        print(json.dumps(row), flush=True)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), args.output_dir / "best_composition_head.pt")

    model.load_state_dict(torch.load(args.output_dir / "best_composition_head.pt", map_location=device, weights_only=True))
    official_arrays = load_arrays(args.official_features)
    official_gallery_ids, official_queries = load_metadata(args.official_features)
    official = evaluate(
        model,
        official_arrays,
        official_queries,
        official_gallery_ids,
        np.arange(len(official_queries)),
        device,
    )
    result = {"best_validation_score": best_score, "official": official}
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--official-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--negative-gallery-size", type=int, default=2048)
    parser.add_argument("--val-percent", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.hidden_dim <= 0 or args.batch_size <= 0 or args.negative_gallery_size <= 0:
        parser.error("hidden dimension and batch sizes must be positive")
    if not 0 <= args.val_percent < 100:
        parser.error("val-percent must be between 0 and 99")
    return args


if __name__ == "__main__":
    train(parse_args())
