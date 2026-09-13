#!/usr/bin/env python3
"""Evaluate a 3x3 DQU-base/DQU-GradCache/MCoT compatibility matrix."""

import argparse
import json
from pathlib import Path

import numpy as np

from eval_cross_compatibility import (
    CATEGORIES,
    exclude_sources_,
    load_alignment,
    normalized_tensor,
    recall_metrics,
    target_ranks,
    validate_endpoints,
)


ENDPOINTS = ("dqu_base", "dqu_gradcache", "mcot")
PAIRS = (("dqu_base", "dqu_gradcache"), ("dqu_base", "mcot"), ("dqu_gradcache", "mcot"))


def endpoint_paths(category, dqu_root, mcot_root):
    dqu = dqu_root / category / "official" / "embeddings"
    mcot = mcot_root / category
    return {
        "dqu_base": (dqu / "base_queries.npy", dqu / "base_gallery.npy"),
        "dqu_gradcache": (dqu / "correction_queries.npy", dqu / "correction_gallery.npy"),
        "mcot": (mcot / "queries.npy", mcot / "gallery.npy"),
    }, dqu


def evaluate_category(category, dqu_root, mcot_root, device, batch_size, cutoffs):
    import torch

    paths, alignment_dir = endpoint_paths(category, dqu_root, mcot_root)
    rows, gallery_ids, targets, sources = load_alignment(alignment_dir)
    arrays = {
        endpoint: (
            np.load(query_path, mmap_mode="r"),
            np.load(gallery_path, mmap_mode="r"),
        )
        for endpoint, (query_path, gallery_path) in paths.items()
    }
    validate_endpoints(arrays, len(rows), len(gallery_ids))
    galleries = {
        endpoint: normalized_tensor(array[1], device)
        for endpoint, array in arrays.items()
    }
    rank_parts = {}
    for query_name in ENDPOINTS:
        for gallery_name in ENDPOINTS:
            rank_parts[f"{query_name}__{gallery_name}"] = []
    for left, right in PAIRS:
        for fusion in ("diagonal_mean", "cross_mean", "all_mean"):
            rank_parts[f"{left}+{right}__{fusion}"] = []

    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        queries = {
            endpoint: normalized_tensor(array[0][start:stop], device)
            for endpoint, array in arrays.items()
        }
        with torch.no_grad():
            scores = {
                (query_name, gallery_name): queries[query_name] @ galleries[gallery_name].T
                for query_name in ENDPOINTS
                for gallery_name in ENDPOINTS
            }
            exclude_sources_(scores.values(), sources[start:stop])
            tables = {
                f"{query_name}__{gallery_name}": score
                for (query_name, gallery_name), score in scores.items()
            }
            for left, right in PAIRS:
                diagonal = scores[left, left] + scores[right, right]
                cross = scores[left, right] + scores[right, left]
                tables[f"{left}+{right}__diagonal_mean"] = 0.5 * diagonal
                tables[f"{left}+{right}__cross_mean"] = 0.5 * cross
                tables[f"{left}+{right}__all_mean"] = 0.25 * (diagonal + cross)
            for name, score in tables.items():
                rank_parts[name].append(target_ranks(score.cpu().numpy(), targets[start:stop]))

    return {
        name: recall_metrics(np.concatenate(parts), cutoffs)
        for name, parts in rank_parts.items()
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dqu-root", required=True, type=Path)
    parser.add_argument("--mcot-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=(1, 10, 50))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or any(cutoff <= 0 for cutoff in args.cutoffs):
        raise ValueError("batch size and cutoffs must be positive")
    categories = {
        category: evaluate_category(
            category,
            args.dqu_root,
            args.mcot_root,
            args.device,
            args.batch_size,
            tuple(args.cutoffs),
        )
        for category in CATEGORIES
    }
    average = {
        name: {
            metric: round(
                float(
                    np.mean(
                        [categories[category][name][metric] for category in CATEGORIES]
                    )
                ),
                6,
            )
            for metric in categories[CATEGORIES[0]][name]
        }
        for name in categories[CATEGORIES[0]]
    }
    report = {
        "exclude_source": True,
        "endpoints": list(ENDPOINTS),
        "inputs": {"dqu_root": str(args.dqu_root), "mcot_root": str(args.mcot_root)},
        "categories": categories,
        "average": average,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(average, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
