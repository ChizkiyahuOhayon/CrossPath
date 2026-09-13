#!/usr/bin/env python3
"""Measure the attainable CIRR gain from per-query CrossPath routing."""

import argparse
import json
from pathlib import Path

import numpy as np

from eval_cirr_cross_compatibility import (
    load_inputs,
    normalize,
    recall,
    score_paths,
    subset_target_ranks,
    target_ranks,
)


def action_scores(paths, alphas):
    """Return MCoT plus blends toward the two complementary paths."""
    base = paths["q1_g1"]
    actions = {"mcot": base}
    for branch in ("q1_g0", "cross_mean"):
        for alpha in alphas:
            scores = base.copy()
            finite = np.isfinite(base) & np.isfinite(paths[branch])
            scores[finite] = (
                (1.0 - alpha) * base[finite] + alpha * paths[branch][finite]
            )
            actions[f"{branch}_a{alpha:g}"] = scores
    return actions


def metrics(global_ranks, subset_ranks):
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


def analyze(embedding_dir, alphas, batch_size):
    rows, gallery_ids, sources, groups, arrays = load_inputs(embedding_dir)
    if not rows or "target_id" not in rows[0]:
        raise ValueError("CIRR validation metadata with target_id is required")
    gallery_index = {name: index for index, name in enumerate(gallery_ids)}
    targets = np.asarray([gallery_index[row["target_id"]] for row in rows])
    gallery0 = normalize(arrays["base_gallery"])
    gallery1 = normalize(arrays["correction_gallery"])
    action_names = ["mcot"] + [
        f"{branch}_a{alpha:g}"
        for branch in ("q1_g0", "cross_mean")
        for alpha in alphas
    ]
    global_ranks = {name: [] for name in action_names}
    subset_ranks = {name: [] for name in action_names}

    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        paths = score_paths(
            normalize(arrays["base_queries"][start:stop]),
            normalize(arrays["correction_queries"][start:stop]),
            gallery0,
            gallery1,
            sources[start:stop],
        )
        for name, scores in action_scores(paths, alphas).items():
            global_ranks[name].append(target_ranks(scores, targets[start:stop]))
            subset_ranks[name].append(
                subset_target_ranks(
                    scores,
                    groups[start:stop],
                    targets[start:stop],
                    sources[start:stop],
                )
            )

    global_ranks = {name: np.concatenate(parts) for name, parts in global_ranks.items()}
    subset_ranks = {name: np.concatenate(parts) for name, parts in subset_ranks.items()}
    action_metrics = {
        name: metrics(global_ranks[name], subset_ranks[name]) for name in action_names
    }
    utility = np.stack(
        [
            (global_ranks[name] <= 5).astype(np.int8)
            + (subset_ranks[name] <= 1).astype(np.int8)
            for name in action_names
        ],
        axis=1,
    )
    best_utility = utility.max(axis=1)
    base_utility = utility[:, 0]
    return {
        "queries": len(rows),
        "alphas": list(alphas),
        "objective": "0.5 * (R@5 + subset_R@1)",
        "base": action_metrics["mcot"],
        "best_fixed_action": max(action_names, key=lambda name: action_metrics[name]["Avg"]),
        "best_fixed_metrics": max(action_metrics.values(), key=lambda item: item["Avg"]),
        "oracle": {
            "Avg": round(float(50.0 * best_utility.mean()), 6),
            "gain_over_mcot": round(
                float(50.0 * (best_utility.mean() - base_utility.mean())), 6
            ),
            "queries_improved": int(np.sum(best_utility > base_utility)),
            "queries_unchanged": int(np.sum(best_utility == base_utility)),
        },
        "actions": action_metrics,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.125, 0.25, 0.375, 0.5, 0.75, 1.0])
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if any(alpha <= 0.0 or alpha > 1.0 for alpha in args.alphas):
        raise ValueError("--alphas values must be in (0, 1]")
    if len(set(args.alphas)) != len(args.alphas):
        raise ValueError("--alphas values must be unique")
    report = analyze(args.embedding_dir, tuple(args.alphas), args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
