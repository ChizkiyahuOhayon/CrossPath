#!/usr/bin/env python3
"""Run coordinate-scrambling and rescue/harm controls on aligned endpoints."""

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


PATHS = ("q0_g0", "q0_g1", "q1_g0", "q1_g1", "diagonal_mean", "cross_mean", "all_mean")


def signed_permutation(array, permutation, signs):
    return array[:, permutation] * signs


def rescue_harm(baseline_ranks, method_ranks, cutoffs):
    result = {}
    total = len(baseline_ranks)
    for cutoff in cutoffs:
        baseline_hit = baseline_ranks <= cutoff
        method_hit = method_ranks <= cutoff
        rescued = int(np.sum(~baseline_hit & method_hit))
        harmed = int(np.sum(baseline_hit & ~method_hit))
        result[f"R@{cutoff}"] = {
            "rescued": rescued,
            "harmed": harmed,
            "retained": int(np.sum(baseline_hit & method_hit)),
            "missed_by_both": int(np.sum(~baseline_hit & ~method_hit)),
            "net": rescued - harmed,
            "delta_recall": round(100.0 * (rescued - harmed) / total, 6),
        }
    return result


def evaluate_directory(embedding_dir, device, batch_size, cutoffs, exclude_source, seed):
    import torch

    rows, gallery_ids, target_indices, source_indices = load_alignment(embedding_dir)
    arrays = {
        name: np.load(embedding_dir / f"{name}.npy", mmap_mode="r")
        for name in ("base_gallery", "correction_gallery", "base_queries", "correction_queries")
    }
    validate_endpoints(
        {
            "base": (arrays["base_queries"], arrays["base_gallery"]),
            "correction": (arrays["correction_queries"], arrays["correction_gallery"]),
        },
        len(rows),
        len(gallery_ids),
    )
    dimension = arrays["correction_queries"].shape[1]
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(dimension)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=dimension)

    permutation = torch.tensor(permutation, dtype=torch.long, device=device)
    signs = torch.tensor(signs, dtype=torch.float32, device=device)
    gallery0 = normalized_tensor(arrays["base_gallery"], device)
    gallery1 = normalized_tensor(arrays["correction_gallery"], device)
    gallery1_scrambled = signed_permutation(gallery1, permutation, signs)
    ranks = {condition: {path: [] for path in PATHS} for condition in ("real", "scrambled")}
    max_endpoint_score_error = 0.0

    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        query0 = normalized_tensor(arrays["base_queries"][start:stop], device)
        query1 = normalized_tensor(arrays["correction_queries"][start:stop], device)
        query1_scrambled = signed_permutation(query1, permutation, signs)
        with torch.no_grad():
            score00 = query0 @ gallery0.T
            score01 = query0 @ gallery1.T
            score10 = query1 @ gallery0.T
            score11 = query1 @ gallery1.T
            scrambled01 = query0 @ gallery1_scrambled.T
            scrambled10 = query1_scrambled @ gallery0.T
            scrambled11 = query1_scrambled @ gallery1_scrambled.T
            max_endpoint_score_error = max(
                max_endpoint_score_error,
                float(torch.max(torch.abs(score11 - scrambled11)).item()),
            )
            if exclude_source:
                exclude_sources_(
                    (score00, score01, score10, score11, scrambled01, scrambled10, scrambled11),
                    source_indices[start:stop],
                )
            tables = {
                "real": {
                    "q0_g0": score00,
                    "q0_g1": score01,
                    "q1_g0": score10,
                    "q1_g1": score11,
                    "diagonal_mean": 0.5 * (score00 + score11),
                    "cross_mean": 0.5 * (score01 + score10),
                    "all_mean": 0.25 * (score00 + score01 + score10 + score11),
                },
                "scrambled": {
                    "q0_g0": score00,
                    "q0_g1": scrambled01,
                    "q1_g0": scrambled10,
                    "q1_g1": scrambled11,
                    "diagonal_mean": 0.5 * (score00 + scrambled11),
                    "cross_mean": 0.5 * (scrambled01 + scrambled10),
                    "all_mean": 0.25 * (score00 + scrambled01 + scrambled10 + scrambled11),
                },
            }
            for condition, score_tables in tables.items():
                for path, scores in score_tables.items():
                    ranks[condition][path].append(
                        target_ranks(scores.cpu().numpy(), target_indices[start:stop])
                    )

    concatenated = {
        condition: {path: np.concatenate(parts) for path, parts in paths.items()}
        for condition, paths in ranks.items()
    }
    preserved_paths = ("q0_g0", "q1_g1", "diagonal_mean")
    if any(
        not np.array_equal(concatenated["real"][path], concatenated["scrambled"][path])
        for path in preserved_paths
    ):
        raise RuntimeError("coordinate scrambling changed a preserved endpoint ranking")
    metrics = {
        condition: {
            path: recall_metrics(path_ranks, cutoffs)
            for path, path_ranks in paths.items()
        }
        for condition, paths in concatenated.items()
    }
    return metrics, concatenated, max_endpoint_score_error


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-root", type=Path)
    source.add_argument("--embedding-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-path", choices=("q0_g0", "q1_g1"), required=True)
    parser.add_argument("--method-path", choices=PATHS, default="cross_mean")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cutoffs", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--exclude-source", action="store_true")
    parser.add_argument("--stage", default="official")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or any(cutoff <= 0 for cutoff in args.cutoffs):
        raise ValueError("batch size and cutoffs must be positive")
    directories = (
        {category: args.run_root / category / args.stage / "embeddings" for category in CATEGORIES}
        if args.run_root is not None
        else {"all": args.embedding_dir}
    )
    category_reports = {}
    all_baseline_ranks = []
    all_method_ranks = []
    for category, directory in directories.items():
        metrics, ranks, score_error = evaluate_directory(
            directory,
            args.device,
            args.batch_size,
            tuple(args.cutoffs),
            args.exclude_source,
            args.seed,
        )
        category_reports[category] = {
            "metrics": metrics,
            "rescue_harm": rescue_harm(
                ranks["real"][args.baseline_path], ranks["real"][args.method_path], args.cutoffs
            ),
            "max_endpoint_score_error": score_error,
            "preserved_path_rankings_identical": True,
        }
        all_baseline_ranks.append(ranks["real"][args.baseline_path])
        all_method_ranks.append(ranks["real"][args.method_path])

    macro_average = {
        condition: {
            path: {
                metric: round(float(np.mean([
                    report["metrics"][condition][path][metric]
                    for report in category_reports.values()
                ])), 6)
                for metric in next(iter(category_reports.values()))["metrics"][condition][path]
            }
            for path in PATHS
        }
        for condition in ("real", "scrambled")
    }
    report = {
        "control": "shared signed-permutation of endpoint 1 query and gallery coordinates",
        "inputs": {category: str(directory) for category, directory in directories.items()},
        "seed": args.seed,
        "exclude_source": args.exclude_source,
        "baseline_path": args.baseline_path,
        "method_path": args.method_path,
        "categories": category_reports,
        "macro_average": macro_average,
        "pooled_rescue_harm": rescue_harm(
            np.concatenate(all_baseline_ranks), np.concatenate(all_method_ranks), args.cutoffs
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    console_report = {
        "macro_average": macro_average,
        "pooled_rescue_harm": report["pooled_rescue_harm"],
    }
    print(json.dumps(console_report, indent=2))


if __name__ == "__main__":
    main()
