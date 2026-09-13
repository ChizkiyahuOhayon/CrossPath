#!/usr/bin/env python3
"""Evaluate the four query-gallery compatibility paths of two aligned endpoints."""

import argparse
import json
from pathlib import Path

import numpy as np


CATEGORIES = ("dress", "shirt", "toptee")
PATHS = (
    "q0_g0",
    "q0_g1",
    "q1_g0",
    "q1_g1",
    "diagonal_mean",
    "cross_mean",
    "all_mean",
    "diagonal_max",
    "cross_max",
    "all_max",
    "diagonal_borda",
    "cross_borda",
    "all_borda",
)
CUTOFFS = (1, 10, 50)


def load_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_alignment(embedding_dir):
    gallery_ids = json.loads((embedding_dir / "gallery_ids.json").read_text())
    gallery_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
    if len(gallery_index) != len(gallery_ids):
        raise ValueError("gallery_ids.json contains duplicate IDs")
    rows = load_jsonl(embedding_dir / "queries.jsonl")
    targets = np.asarray([gallery_index[row["target_id"]] for row in rows])
    sources = np.asarray([gallery_index[row["source_id"]] for row in rows])
    return rows, gallery_ids, targets, sources


def validate_endpoints(endpoints, query_count, gallery_count):
    dimensions = set()
    for name, (queries, gallery) in endpoints.items():
        if queries.ndim != 2 or gallery.ndim != 2:
            raise ValueError(f"{name} embeddings must be rank-2 arrays")
        if len(queries) != query_count or len(gallery) != gallery_count:
            raise ValueError(
                f"{name} alignment mismatch: queries={len(queries)}/{query_count}, "
                f"gallery={len(gallery)}/{gallery_count}"
            )
        dimensions.update((queries.shape[1], gallery.shape[1]))
    if len(dimensions) != 1:
        raise ValueError(f"embedding dimensions differ: {sorted(dimensions)}")


def normalized_tensor(array, device):
    import torch
    import torch.nn.functional as functional

    return functional.normalize(
        torch.tensor(array, dtype=torch.float32, device=device), dim=-1
    )


def exclude_sources_(scores, source_indices):
    import torch

    scores = tuple(scores)
    first = scores[0]
    rows = torch.arange(len(source_indices), device=first.device)
    sources = torch.tensor(source_indices, dtype=torch.long, device=first.device)
    for score in scores:
        score[rows, sources] = -torch.inf


def target_ranks(scores, target_indices):
    rows = np.arange(scores.shape[0])
    target_scores = scores[rows, target_indices]
    candidate_indices = np.arange(scores.shape[1])[None, :]
    ties_before_target = (scores == target_scores[:, None]) & (
        candidate_indices < target_indices[:, None]
    )
    return 1 + np.sum(
        (scores > target_scores[:, None]) | ties_before_target, axis=1
    )


def recall_metrics(ranks, cutoffs=CUTOFFS):
    return {
        f"R@{cutoff}": round(float(np.mean(ranks <= cutoff) * 100.0), 6)
        for cutoff in cutoffs
    }


def evaluate_category(
    embedding_dir, device, batch_size, cutoffs=CUTOFFS, exclude_source=True
):
    import torch

    rows, gallery_ids, target_indices, source_indices = load_alignment(embedding_dir)

    arrays = {
        name: np.load(embedding_dir / f"{name}.npy", mmap_mode="r")
        for name in ("base_gallery", "correction_gallery", "base_queries", "correction_queries")
    }
    validate_endpoints(
        {
            "base": (arrays["base_queries"], arrays["base_gallery"]),
            "correction": (
                arrays["correction_queries"],
                arrays["correction_gallery"],
            ),
        },
        len(rows),
        len(gallery_ids),
    )
    gallery0 = normalized_tensor(arrays["base_gallery"], device)
    gallery1 = normalized_tensor(arrays["correction_gallery"], device)
    ranks = {path: [] for path in PATHS}

    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        query0 = normalized_tensor(arrays["base_queries"][start:stop], device)
        query1 = normalized_tensor(arrays["correction_queries"][start:stop], device)
        with torch.no_grad():
            score00 = query0 @ gallery0.T
            score01 = query0 @ gallery1.T
            score10 = query1 @ gallery0.T
            score11 = query1 @ gallery1.T
            if exclude_source:
                exclude_sources_(
                    (score00, score01, score10, score11), source_indices[start:stop]
                )
            rank_tables = []
            for scores in (score00, score01, score10, score11):
                order = torch.argsort(scores, dim=1, descending=True, stable=True)
                ranks_for_path = torch.empty_like(order)
                ranks_for_path.scatter_(
                    1,
                    order,
                    torch.arange(scores.shape[1], device=device)[None, :].expand_as(order),
                )
                rank_tables.append(-ranks_for_path.float())
            rank00, rank01, rank10, rank11 = rank_tables
            score_tables = {
                "q0_g0": score00,
                "q0_g1": score01,
                "q1_g0": score10,
                "q1_g1": score11,
                "diagonal_mean": 0.5 * (score00 + score11),
                "cross_mean": 0.5 * (score01 + score10),
                "all_mean": 0.25 * (score00 + score01 + score10 + score11),
                "diagonal_max": torch.maximum(score00, score11),
                "cross_max": torch.maximum(score01, score10),
                "all_max": torch.maximum(
                    torch.maximum(score00, score01),
                    torch.maximum(score10, score11),
                ),
                "diagonal_borda": 0.5 * (rank00 + rank11),
                "cross_borda": 0.5 * (rank01 + rank10),
                "all_borda": 0.25 * (rank00 + rank01 + rank10 + rank11),
            }
            for path, scores in score_tables.items():
                ranks[path].append(
                    target_ranks(
                        scores.float().cpu().numpy(), target_indices[start:stop]
                    )
                )

    return {
        path: recall_metrics(np.concatenate(path_ranks), cutoffs)
        for path, path_ranks in ranks.items()
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-root", type=Path)
    source.add_argument("--embedding-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=list(CUTOFFS))
    source_policy = parser.add_mutually_exclusive_group()
    source_policy.add_argument(
        "--exclude-source", dest="exclude_source", action="store_true"
    )
    source_policy.add_argument(
        "--include-source", dest="exclude_source", action="store_false"
    )
    parser.set_defaults(exclude_source=None)
    parser.add_argument("--stage", choices=("internal", "official"), default="official")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if any(cutoff <= 0 for cutoff in args.cutoffs):
        raise ValueError("--cutoffs must be positive")
    if args.embedding_dir is not None:
        exclude_source = bool(args.exclude_source)
        metrics = evaluate_category(
            args.embedding_dir,
            args.device,
            args.batch_size,
            tuple(args.cutoffs),
            exclude_source,
        )
        report = {
            "embedding_dir": str(args.embedding_dir),
            "exclude_source": exclude_source,
            "metrics": metrics,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return
    exclude_source = True if args.exclude_source is None else args.exclude_source
    categories = {
        category: evaluate_category(
            args.run_root / category / args.stage / "embeddings",
            args.device,
            args.batch_size,
            tuple(args.cutoffs),
            exclude_source,
        )
        for category in CATEGORIES
    }
    average = {
        path: {
            metric: round(
                sum(categories[category][path][metric] for category in CATEGORIES)
                / len(CATEGORIES),
                6,
            )
            for metric in categories[CATEGORIES[0]][path]
        }
        for path in PATHS
    }
    report = {
        "exclude_source": exclude_source,
        "categories": categories,
        "average": average,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(average, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
