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
    import torch.nn.functional as functional

    gallery_ids = json.loads((embedding_dir / "gallery_ids.json").read_text())
    gallery_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
    rows = load_jsonl(embedding_dir / "queries.jsonl")
    target_indices = np.asarray([gallery_index[row["target_id"]] for row in rows])
    source_indices = np.asarray([gallery_index[row["source_id"]] for row in rows])

    arrays = {
        name: np.load(embedding_dir / f"{name}.npy", mmap_mode="r")
        for name in ("base_gallery", "correction_gallery", "base_queries", "correction_queries")
    }
    gallery0 = functional.normalize(
        torch.as_tensor(arrays["base_gallery"], dtype=torch.float32, device=device), dim=-1
    )
    gallery1 = functional.normalize(
        torch.as_tensor(arrays["correction_gallery"], dtype=torch.float32, device=device), dim=-1
    )
    ranks = {path: [] for path in PATHS}

    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        query0 = functional.normalize(
            torch.as_tensor(arrays["base_queries"][start:stop], dtype=torch.float32, device=device), dim=-1
        )
        query1 = functional.normalize(
            torch.as_tensor(arrays["correction_queries"][start:stop], dtype=torch.float32, device=device), dim=-1
        )
        with torch.no_grad():
            score00 = query0 @ gallery0.T
            score01 = query0 @ gallery1.T
            score10 = query1 @ gallery0.T
            score11 = query1 @ gallery1.T
            if exclude_source:
                local_sources = torch.as_tensor(
                    source_indices[start:stop], dtype=torch.long, device=device
                )
                local_rows = torch.arange(stop - start, device=device)
                for scores in (score00, score01, score10, score11):
                    scores[local_rows, local_sources] = -torch.inf
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
    parser.add_argument("--exclude-source", action="store_true")
    parser.add_argument("--stage", choices=("internal", "official"), default="official")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if any(cutoff <= 0 for cutoff in args.cutoffs):
        raise ValueError("--cutoffs must be positive")
    if args.embedding_dir is not None:
        metrics = evaluate_category(
            args.embedding_dir,
            args.device,
            args.batch_size,
            tuple(args.cutoffs),
            args.exclude_source,
        )
        report = {
            "embedding_dir": str(args.embedding_dir),
            "exclude_source": args.exclude_source,
            "metrics": metrics,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return
    categories = {
        category: evaluate_category(
            args.run_root / category / args.stage / "embeddings",
            args.device,
            args.batch_size,
            tuple(args.cutoffs),
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
    report = {"categories": categories, "average": average}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(average, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
