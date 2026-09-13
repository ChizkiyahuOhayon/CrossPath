#!/usr/bin/env python3
"""Evaluate aligned two-endpoint CrossPath embeddings on CIRR."""

import argparse
import json
from pathlib import Path

import numpy as np


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
SINGLE_PATHS = PATHS[:4]
FUSION_PATHS = PATHS[4:]
GLOBAL_CUTOFFS = (1, 5, 10, 50)
SUBSET_CUTOFFS = (1, 2, 3)


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_inputs(embedding_dir):
    gallery_ids = json.loads(
        (embedding_dir / "gallery_ids.json").read_text(encoding="utf-8")
    )
    gallery_index = {image_id: index for index, image_id in enumerate(gallery_ids)}
    if len(gallery_index) != len(gallery_ids):
        raise ValueError("gallery_ids.json contains duplicate IDs")
    rows = load_jsonl(embedding_dir / "queries.jsonl")
    sources = np.asarray([gallery_index[row["source_id"]] for row in rows])
    groups = [np.asarray([gallery_index[name] for name in row["group_members"]]) for row in rows]
    arrays = {
        name: np.load(embedding_dir / f"{name}.npy", mmap_mode="r")
        for name in (
            "base_gallery",
            "base_queries",
            "correction_gallery",
            "correction_queries",
        )
    }
    validate_inputs(arrays, len(rows), len(gallery_ids))
    return rows, gallery_ids, sources, groups, arrays


def validate_inputs(arrays, query_count, gallery_count):
    shapes = {name: array.shape for name, array in arrays.items()}
    if any(array.ndim != 2 for array in arrays.values()):
        raise ValueError(f"embeddings must be rank-2 arrays: {shapes}")
    if len(arrays["base_queries"]) != query_count or len(arrays["correction_queries"]) != query_count:
        raise ValueError(f"query alignment mismatch: {shapes}")
    if len(arrays["base_gallery"]) != gallery_count or len(arrays["correction_gallery"]) != gallery_count:
        raise ValueError(f"gallery alignment mismatch: {shapes}")
    if len({array.shape[1] for array in arrays.values()}) != 1:
        raise ValueError(f"embedding dimensions differ: {shapes}")


def normalize(array):
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-norm embedding found")
    return np.asarray(array, dtype=np.float32) / norms


def target_ranks(scores, target_indices):
    rows = np.arange(len(scores))
    targets = np.asarray(target_indices)
    target_scores = scores[rows, targets]
    indices = np.arange(scores.shape[1])[None, :]
    ties_before = (scores == target_scores[:, None]) & (indices < targets[:, None])
    return 1 + np.sum((scores > target_scores[:, None]) | ties_before, axis=1)


def subset_target_ranks(scores, group_indices, target_indices, source_indices):
    ranks = []
    for row, (members, target, source) in enumerate(
        zip(group_indices, target_indices, source_indices)
    ):
        members = members[members != source]
        if target not in members:
            raise ValueError(f"target is absent from CIRR subset at query {row}")
        target_score = scores[row, target]
        before = (scores[row, members] == target_score) & (members < target)
        ranks.append(1 + np.sum((scores[row, members] > target_score) | before))
    return np.asarray(ranks)


def recall(ranks, cutoffs):
    return {
        f"R@{cutoff}": round(float(np.mean(ranks <= cutoff) * 100.0), 6)
        for cutoff in cutoffs
    }


def rank_scores(score):
    order = np.argsort(-score, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(len(order))[:, None], order] = np.arange(score.shape[1])[None, :]
    return -ranks.astype(np.float32)


def score_paths(query0, query1, gallery0, gallery1, source_indices=None):
    score00 = query0 @ gallery0.T
    score01 = query0 @ gallery1.T
    score10 = query1 @ gallery0.T
    score11 = query1 @ gallery1.T
    if source_indices is not None:
        rows = np.arange(len(score00))
        for scores in (score00, score01, score10, score11):
            scores[rows, source_indices] = -np.inf
    rank00, rank01, rank10, rank11 = map(
        rank_scores, (score00, score01, score10, score11)
    )
    return {
        "q0_g0": score00,
        "q0_g1": score01,
        "q1_g0": score10,
        "q1_g1": score11,
        "diagonal_mean": 0.5 * (score00 + score11),
        "cross_mean": 0.5 * (score01 + score10),
        "all_mean": 0.25 * (score00 + score01 + score10 + score11),
        "diagonal_max": np.maximum(score00, score11),
        "cross_max": np.maximum(score01, score10),
        "all_max": np.maximum(np.maximum(score00, score01), np.maximum(score10, score11)),
        "diagonal_borda": 0.5 * (rank00 + rank11),
        "cross_borda": 0.5 * (rank01 + rank10),
        "all_borda": 0.25 * (rank00 + rank01 + rank10 + rank11),
    }


def evaluate_validation(embedding_dir, batch_size):
    rows, _, sources, groups, arrays = load_inputs(embedding_dir)
    if not rows or "target_id" not in rows[0]:
        raise ValueError("validation metadata with target_id is required")
    gallery_ids = json.loads((embedding_dir / "gallery_ids.json").read_text())
    gallery_index = {name: index for index, name in enumerate(gallery_ids)}
    targets = np.asarray([gallery_index[row["target_id"]] for row in rows])
    gallery0 = normalize(arrays["base_gallery"])
    gallery1 = normalize(arrays["correction_gallery"])
    global_ranks = {path: [] for path in PATHS}
    subset_ranks = {path: [] for path in PATHS}

    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        paths = score_paths(
            normalize(arrays["base_queries"][start:stop]),
            normalize(arrays["correction_queries"][start:stop]),
            gallery0,
            gallery1,
            sources[start:stop],
        )
        batch_sources = sources[start:stop]
        for path, scores in paths.items():
            global_ranks[path].append(target_ranks(scores, targets[start:stop]))
            subset_ranks[path].append(
                subset_target_ranks(
                    scores,
                    groups[start:stop],
                    targets[start:stop],
                    batch_sources,
                )
            )

    metrics = {}
    for path in PATHS:
        global_metrics = recall(np.concatenate(global_ranks[path]), GLOBAL_CUTOFFS)
        subset_metrics = {
            f"subset_{name}": value
            for name, value in recall(
                np.concatenate(subset_ranks[path]), SUBSET_CUTOFFS
            ).items()
        }
        metrics[path] = {
            **global_metrics,
            **subset_metrics,
            "Avg": round(0.5 * (global_metrics["R@5"] + subset_metrics["subset_R@1"]), 6),
        }
    return metrics


def submission_for_path(embedding_dir, path, batch_size):
    rows, gallery_ids, sources, groups, arrays = load_inputs(embedding_dir)
    if not rows or "pair_id" not in rows[0] or "target_id" in rows[0]:
        raise ValueError("test1 metadata with pair_id and no target_id is required")
    gallery0 = normalize(arrays["base_gallery"])
    gallery1 = normalize(arrays["correction_gallery"])
    general = {"version": "rc2", "metric": "recall"}
    subset = {"version": "rc2", "metric": "recall_subset"}

    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        scores = score_paths(
            normalize(arrays["base_queries"][start:stop]),
            normalize(arrays["correction_queries"][start:stop]),
            gallery0,
            gallery1,
            sources[start:stop],
        )[path]
        order = np.argsort(-scores, axis=1, kind="stable")
        for local_index, row in enumerate(rows[start:stop]):
            source_id = row["source_id"]
            ranked_names = [
                gallery_ids[index]
                for index in order[local_index]
                if gallery_ids[index] != source_id
            ]
            names = ranked_names[:50]
            member_set = set(row["group_members"])
            subset_names = [name for name in ranked_names if name in member_set][:3]
            pair_id = str(int(row["pair_id"]))
            general[pair_id] = names
            subset[pair_id] = subset_names
    return general, subset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--submission-path", choices=PATHS)
    parser.add_argument("--submission-dir", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if bool(args.submission_path) != bool(args.submission_dir):
        raise ValueError("--submission-path and --submission-dir must be used together")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.submission_path:
        general, subset = submission_for_path(
            args.embedding_dir, args.submission_path, args.batch_size
        )
        args.submission_dir.mkdir(parents=True, exist_ok=True)
        general_path = args.submission_dir / f"recall_{args.submission_path}.json"
        subset_path = args.submission_dir / f"recall_subset_{args.submission_path}.json"
        general_path.write_text(json.dumps(general, sort_keys=True) + "\n")
        subset_path.write_text(json.dumps(subset, sort_keys=True) + "\n")
        report = {
            "split": "test1",
            "path": args.submission_path,
            "queries": len(general) - 2,
            "general_submission": str(general_path),
            "subset_submission": str(subset_path),
        }
    else:
        metrics = evaluate_validation(args.embedding_dir, args.batch_size)
        best = max(FUSION_PATHS, key=lambda path: metrics[path]["Avg"])
        best_single = max(SINGLE_PATHS, key=lambda path: metrics[path]["Avg"])
        report = {
            "split": "val",
            "selection_metric": "Avg",
            "best_path": best,
            "best_single_path": best_single,
            "metrics": metrics,
        }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
