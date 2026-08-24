#!/usr/bin/env python3
"""Train a single-endpoint relational CrossPath composer on frozen CIR embeddings."""

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


class RelationalCrossPath(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, max_step: float) -> None:
        super().__init__()
        self.max_step = max_step
        self.step_head = nn.Sequential(
            nn.Linear(4 * dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.step_head[-1].weight)
        nn.init.zeros_(self.step_head[-1].bias)

    def forward(
        self, query: torch.Tensor, source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        direction = F.normalize(query - source, dim=-1)
        features = torch.cat((query, source, query - source, query * source), dim=-1)
        step = self.max_step * torch.tanh(self.step_head(features))
        relation = F.normalize(query + step * direction, dim=-1)
        fused = F.normalize(query + relation, dim=-1)
        return fused, relation, step.squeeze(-1)


def load_endpoint_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    gallery = np.load(path / "base_gallery.npy").astype(np.float32, copy=False)
    queries = np.load(path / "base_queries.npy").astype(np.float32, copy=False)
    return gallery, queries


def evaluate(
    model: RelationalCrossPath,
    gallery: np.ndarray,
    query_embeddings: np.ndarray,
    queries: list[dict],
    gallery_ids: list[str],
    indices: np.ndarray,
    device: torch.device,
) -> dict:
    targets, sources = target_and_source_indices(gallery_ids, queries)
    selected_queries = torch.from_numpy(query_embeddings[indices]).to(device)
    gallery_tensor = torch.from_numpy(gallery).to(device)
    source_tensor = gallery_tensor[torch.from_numpy(sources[indices]).to(device)]
    with torch.no_grad():
        fused, relation, steps = model(selected_queries, source_tensor)
        target_tensor = torch.from_numpy(targets[indices]).to(device)
        source_indices = torch.from_numpy(sources[indices]).to(device)
        outputs = {}
        for name, embeddings in (
            ("base", selected_queries),
            ("relation", relation),
            ("fused", fused),
        ):
            outputs[name] = recall_metrics(
                embeddings @ gallery_tensor.T,
                target_tensor,
                source_indices,
                (1, 10, 50),
            )
        outputs["step"] = {
            "mean": float(steps.mean()),
            "std": float(steps.std()),
            "min": float(steps.min()),
            "max": float(steps.max()),
        }
        return outputs


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    gallery, query_embeddings = load_endpoint_arrays(args.train_embeddings)
    gallery_ids, queries = load_metadata(args.train_embeddings)
    targets, sources = target_and_source_indices(gallery_ids, queries)
    train_indices, val_indices = query_indices(queries, args.val_percent)
    model = RelationalCrossPath(query_embeddings.shape[1], args.hidden_dim, args.max_step).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "method": "single_endpoint_relational_crosspath",
        "train_embeddings": str(args.train_embeddings.resolve()),
        "official_embeddings": str(args.official_embeddings.resolve()),
        "hidden_dim": args.hidden_dim,
        "max_step": args.max_step,
        "temperature": args.temperature,
        "relation_loss_weight": args.relation_loss_weight,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "negative_gallery_size": args.negative_gallery_size,
        "val_percent": args.val_percent,
        "selection_metric": "fused R@10+R@50",
        "seed": args.seed,
        "train_queries": int(len(train_indices)),
        "validation_queries": int(len(val_indices)),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    gallery_tensor = torch.from_numpy(gallery)
    query_tensor = torch.from_numpy(query_embeddings)
    best_score, history = -float("inf"), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for start in range(0, len(train_indices), args.batch_size):
            if start == 0:
                shuffled = rng.permutation(train_indices)
            batch = shuffled[start : start + args.batch_size]
            positives = np.unique(targets[batch])
            available = np.setdiff1d(np.arange(len(gallery_ids)), positives, assume_unique=False)
            negative_count = min(args.negative_gallery_size, len(available))
            candidates = np.concatenate([positives, rng.choice(available, negative_count, replace=False)])
            rng.shuffle(candidates)
            lookup = {gallery_index: i for i, gallery_index in enumerate(candidates.tolist())}
            labels = torch.tensor([lookup[value] for value in targets[batch]], device=device)

            batch_queries = query_tensor[batch].to(device)
            batch_sources = gallery_tensor[sources[batch]].to(device)
            candidate_gallery = gallery_tensor[candidates].to(device)
            fused, relation, _ = model(batch_queries, batch_sources)
            fused_scores = fused @ candidate_gallery.T / args.temperature
            relation_scores = relation @ candidate_gallery.T / args.temperature
            for row, source in enumerate(sources[batch]):
                if int(source) in lookup:
                    fused_scores[row, lookup[int(source)]] = -1e4
                    relation_scores[row, lookup[int(source)]] = -1e4
            loss = F.cross_entropy(fused_scores, labels)
            loss += args.relation_loss_weight * F.cross_entropy(relation_scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        validation = evaluate(
            model, gallery, query_embeddings, queries, gallery_ids, val_indices, device
        )
        selection_score = validation["fused"]["R@10"] + validation["fused"]["R@50"]
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "selection_score": selection_score,
            "validation": validation,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if selection_score > best_score:
            best_score = selection_score
            torch.save(model.state_dict(), args.output_dir / "best_composer.pt")

    model.load_state_dict(torch.load(args.output_dir / "best_composer.pt", map_location=device, weights_only=True))
    model.eval()
    official_gallery, official_query_embeddings = load_endpoint_arrays(args.official_embeddings)
    official_gallery_ids, official_queries = load_metadata(args.official_embeddings)
    official = evaluate(
        model,
        official_gallery,
        official_query_embeddings,
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
    parser.add_argument("--train-embeddings", type=Path, required=True)
    parser.add_argument("--official-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-step", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--relation-loss-weight", type=float, default=0.5)
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
    if not 0 < args.val_percent < 100:
        parser.error("val-percent must be between 1 and 99")
    return args


if __name__ == "__main__":
    train(parse_args())
