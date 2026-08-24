#!/usr/bin/env python3
"""Evaluate a learned composition query inside the frozen 2x2 path matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from weave_train_composition_crosspath import CompositionCrossPath, load_arrays
from weave_train_crosspath_adapter import (
    load_metadata,
    path_scores,
    recall_metrics,
    target_and_source_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-features", type=Path, required=True)
    parser.add_argument("--paired-embeddings", type=Path, required=True)
    parser.add_argument("--composition-checkpoint", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)

    arrays = load_arrays(args.composition_features)
    gallery_ids, queries = load_metadata(args.composition_features)
    paired_ids, paired_queries = load_metadata(args.paired_embeddings)
    identity_keys = ("query_index", "source_id", "target_id")
    composition_identity = [tuple(row[key] for key in identity_keys) for row in queries]
    paired_identity = [tuple(row[key] for key in identity_keys) for row in paired_queries]
    if gallery_ids != paired_ids or composition_identity != paired_identity:
        raise ValueError("composition and paired embedding metadata differ")
    base_gallery = np.load(args.paired_embeddings / "base_gallery.npy")
    if not np.allclose(arrays["gallery"], base_gallery, atol=1e-6):
        raise ValueError("base gallery embeddings differ")

    model = CompositionCrossPath(arrays["text"].shape[1], args.hidden_dim).to(device)
    model.load_state_dict(
        torch.load(args.composition_checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        q0, mixing = model(
            torch.from_numpy(arrays["text"]).to(device),
            torch.from_numpy(arrays["visual"]).to(device),
            torch.from_numpy(arrays["original_lambda"]).to(device),
        )
        g0 = torch.from_numpy(base_gallery).to(device)
        q1 = torch.from_numpy(np.load(args.paired_embeddings / "correction_queries.npy")).to(device)
        g1 = torch.from_numpy(np.load(args.paired_embeddings / "correction_gallery.npy")).to(device)
        scores = path_scores(q0, q1, g0, g1)
        targets, sources = target_and_source_indices(gallery_ids, queries)
        target_tensor = torch.from_numpy(targets).to(device)
        source_tensor = torch.from_numpy(sources).to(device)
        metrics = {
            name: recall_metrics(value, target_tensor, source_tensor, (1, 10, 50))
            for name, value in scores.items()
        }
    result = {
        "metrics": metrics,
        "composition_lambda": {
            "mean": float(mixing.mean()),
            "std": float(mixing.std()),
            "min": float(mixing.min()),
            "max": float(mixing.max()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
