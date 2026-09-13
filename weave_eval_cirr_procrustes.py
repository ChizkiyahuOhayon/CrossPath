#!/usr/bin/env python3
"""Fit a label-free orthogonal endpoint map and evaluate it on CIRR."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from scripts.eval_cirr_cross_compatibility import (
    load_inputs,
    normalize,
    recall,
    subset_target_ranks,
    target_ranks,
)


def orthogonal_map(source, target):
    cross_covariance = source.T @ target
    left, _, right_t = torch.linalg.svd(cross_covariance, full_matrices=False)
    return left @ right_t


def metrics(scores, targets, groups, sources):
    global_ranks = target_ranks(scores, targets)
    subset_ranks = subset_target_ranks(scores, groups, targets, sources)
    global_metrics = recall(global_ranks, (1, 5, 10, 50))
    subset_metrics = {
        f"subset_{name}": value
        for name, value in recall(subset_ranks, (1, 2, 3)).items()
    }
    return {
        **global_metrics,
        **subset_metrics,
        "Avg": round(0.5 * (global_metrics["R@5"] + subset_metrics["subset_R@1"]), 6),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-embedding-dir", required=True, type=Path)
    parser.add_argument("--val-embedding-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train0 = functional.normalize(
        torch.tensor(np.load(args.train_embedding_dir / "base_gallery.npy"), device=device), dim=1
    )
    train1 = functional.normalize(
        torch.tensor(np.load(args.train_embedding_dir / "correction_gallery.npy"), device=device), dim=1
    )
    mapping = orthogonal_map(train0, train1)
    before = torch.sum(train0 * train1, dim=1).mean()
    after = torch.sum((train0 @ mapping) * train1, dim=1).mean()
    identity = torch.eye(mapping.shape[0], device=device)
    orthogonality_error = torch.linalg.matrix_norm(mapping.T @ mapping - identity) / mapping.shape[0]

    rows, gallery_ids, sources, groups, arrays = load_inputs(args.val_embedding_dir)
    gallery_index = {name: index for index, name in enumerate(gallery_ids)}
    targets = np.asarray([gallery_index[row["target_id"]] for row in rows])
    query1 = torch.tensor(normalize(arrays["correction_queries"]), device=device)
    gallery0 = torch.tensor(normalize(arrays["base_gallery"]), device=device)
    gallery1 = torch.tensor(normalize(arrays["correction_gallery"]), device=device)
    score11 = (query1 @ gallery1.T).cpu().numpy()
    score10 = (query1 @ gallery0.T).cpu().numpy()
    mapped10 = (query1 @ (gallery0 @ mapping).T).cpu().numpy()
    for scores in (score11, score10, mapped10):
        scores[np.arange(len(scores)), sources] = -np.inf
    evaluated = {
        "mcot": score11,
        f"unaligned_blend_a{args.alpha:g}": (1.0 - args.alpha) * score11 + args.alpha * score10,
        "mapped_q1_g0": mapped10,
        f"mapped_blend_a{args.alpha:g}": (1.0 - args.alpha) * score11 + args.alpha * mapped10,
    }
    report = {
        "fit": {
            "pairs": len(train0),
            "mean_pair_cosine_before": float(before),
            "mean_pair_cosine_after": float(after),
            "orthogonality_error": float(orthogonality_error),
        },
        "alpha": args.alpha,
        "metrics": {
            name: metrics(scores, targets, groups, sources)
            for name, scores in evaluated.items()
        },
    }
    torch.save(mapping.cpu(), args.output_dir / "orthogonal_map.pt")
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
