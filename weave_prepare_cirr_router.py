#!/usr/bin/env python3
"""Build target-labelled query-router features from aligned CIRR endpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from weave_cirr_query_router import (
    action_names,
    action_scores,
    pair_partition,
    query_features,
    score_paths,
    subset_target_ranks,
    target_ranks,
)


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.125, 0.25, 0.375, 0.5, 0.75, 1.0])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    if args.batch_size <= 0 or any(alpha <= 0 or alpha > 1 for alpha in args.alphas):
        raise ValueError("invalid batch size or alpha")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.embedding_dir / "queries.jsonl")
    gallery_ids = json.loads((args.embedding_dir / "gallery_ids.json").read_text())
    gallery_index = {name: index for index, name in enumerate(gallery_ids)}
    sources = np.asarray([gallery_index[row["source_id"]] for row in rows], dtype=np.int64)
    targets = np.asarray([gallery_index[row["target_id"]] for row in rows], dtype=np.int64)
    max_group = max(len(row["group_members"]) for row in rows)
    groups = np.full((len(rows), max_group), -1, dtype=np.int64)
    for index, row in enumerate(rows):
        members = [gallery_index[name] for name in row["group_members"]]
        groups[index, : len(members)] = members

    arrays = {
        name: np.load(args.embedding_dir / f"{name}.npy", mmap_mode="r")
        for name in ("base_gallery", "base_queries", "correction_gallery", "correction_queries")
    }
    if len(arrays["base_queries"]) != len(rows) or len(arrays["correction_queries"]) != len(rows):
        raise ValueError("query arrays are not aligned with metadata")
    if len(arrays["base_gallery"]) != len(gallery_ids) or len(arrays["correction_gallery"]) != len(gallery_ids):
        raise ValueError("gallery arrays are not aligned with metadata")

    device = torch.device(args.device)
    gallery0 = functional.normalize(torch.tensor(arrays["base_gallery"], device=device), dim=1)
    gallery1 = functional.normalize(torch.tensor(arrays["correction_gallery"], device=device), dim=1)
    names = action_names(args.alphas)
    global_ranks = np.empty((len(rows), len(names)), dtype=np.uint16)
    subset_ranks = np.empty((len(rows), len(names)), dtype=np.uint16)
    feature_file = None

    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            stop = min(start + args.batch_size, len(rows))
            query0 = functional.normalize(
                torch.tensor(arrays["base_queries"][start:stop], device=device), dim=1
            )
            query1 = functional.normalize(
                torch.tensor(arrays["correction_queries"][start:stop], device=device), dim=1
            )
            batch_sources = torch.as_tensor(sources[start:stop], device=device)
            batch_targets = torch.as_tensor(targets[start:stop], device=device)
            batch_groups = torch.as_tensor(groups[start:stop], device=device)
            paths = score_paths(query0, query1, gallery0, gallery1)
            features = query_features(query0, query1, paths, batch_sources)
            scores = action_scores(paths, args.alphas, batch_sources)
            batch_global, batch_subset = [], []
            for action in range(len(names)):
                batch_global.append(target_ranks(scores[:, action], batch_targets))
                batch_subset.append(
                    subset_target_ranks(
                        scores[:, action], batch_groups, batch_targets, batch_sources
                    )
                )
            if feature_file is None:
                feature_file = np.lib.format.open_memmap(
                    args.output_dir / "features.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(rows), features.shape[1]),
                )
            feature_file[start:stop] = features.float().cpu().numpy()
            global_ranks[start:stop] = torch.stack(batch_global, dim=1).cpu().numpy()
            subset_ranks[start:stop] = torch.stack(batch_subset, dim=1).cpu().numpy()
            print(f"prepared {stop}/{len(rows)}", flush=True)

    np.save(args.output_dir / "global_ranks.npy", global_ranks)
    np.save(args.output_dir / "subset_ranks.npy", subset_ranks)
    np.save(
        args.output_dir / "partitions.npy",
        np.asarray([pair_partition(row, args.seed) for row in rows], dtype=np.uint8),
    )
    manifest = {
        "queries": len(rows),
        "gallery": len(gallery_ids),
        "feature_dim": int(feature_file.shape[1]),
        "actions": names,
        "alphas": args.alphas,
        "partition_seed": args.seed,
        "partitions": {name: int(np.sum(np.load(args.output_dir / "partitions.npy") == value)) for name, value in (("train", 0), ("development", 1), ("test", 2))},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
