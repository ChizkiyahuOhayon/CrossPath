#!/usr/bin/env python3
"""Measure CrossPath storage, boundary size, gate parameters, and gate latency."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from weave_crosspath import estimate_utilities
from weave_crosspath_gate import CrossPathGate, aggregate_cutoff_utilities
from weave_train_crosspath_gate import RecordSource


def file_bytes(paths):
    return sum(Path(path).stat().st_size for path in paths)


def boundary_statistics(cache_dir, cutoffs):
    manifest = json.loads((Path(cache_dir) / "manifest.json").read_text())
    totals = {k: 0 for k in cutoffs}
    queries = 0
    for row in manifest["shards"]:
        with np.load(Path(cache_dir) / row["path"]) as shard:
            queries += len(shard["query_indices"])
            for k in cutoffs:
                totals[k] += int(shard[f"k{k}_offsets"][-1])
    return {
        f"R@{k}": totals[k] / queries
        for k in cutoffs
    }


def benchmark_gate(source, gate, regression_cost, max_queries):
    processed = 0
    source.torch.cuda.synchronize()
    started = time.perf_counter()
    with source.torch.no_grad():
        for shard_path in source.shard_paths:
            with np.load(shard_path) as shard:
                for local_index in range(len(shard["query_indices"])):
                    cutoff_utilities = []
                    for k in source.cutoffs:
                        _, features, _, membership = source.features(
                            shard, local_index, k
                        )
                        probabilities = gate.probabilities(features).cpu().numpy()
                        cutoff_utilities.append(
                            estimate_utilities(
                                membership.cpu().numpy(),
                                probabilities,
                                regression_cost=regression_cost,
                            )
                        )
                    aggregate_cutoff_utilities(cutoff_utilities)
                    processed += 1
                    if processed == max_queries:
                        source.torch.cuda.synchronize()
                        return 1000.0 * (time.perf_counter() - started) / processed
    source.torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - started) / processed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--gate-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    gate_manifest = json.loads((args.gate_dir / "manifest.json").read_text())
    report = gate_manifest["report"]
    training = gate_manifest.get("training", {})
    source = RecordSource(
        args.embedding_dir,
        args.cache_dir,
        args.device,
        int(report["seed"]),
        zero_trace=bool(training.get("zero_trace", False)),
        feature_mode=training.get("feature_mode", "full"),
    )
    width = int(report["selected_width"])
    gate = CrossPathGate(
        embedding_dim=source.embedding_dim,
        num_actions=len(source.alphas),
        hidden_width=width,
    ).to(source.device)
    checkpoint = args.gate_dir / f"gate_width{width}.pt"
    state = source.torch.load(checkpoint, map_location=source.device, weights_only=True)
    gate.load_state_dict(state)
    gate.eval()

    embedding_manifest = json.loads(
        (args.embedding_dir / "manifest.json").read_text()
    )
    result = {
        "dataset": embedding_manifest["dataset"],
        "queries": embedding_manifest["queries"],
        "gallery": embedding_manifest["gallery"],
        "embedding_dim": source.embedding_dim,
        "path_actions": len(source.alphas),
        "cutoffs": list(source.cutoffs),
        "gate_width": width,
        "gate_parameters": sum(parameter.numel() for parameter in gate.parameters()),
        "gate_checkpoint_bytes": checkpoint.stat().st_size,
        "two_gallery_embedding_bytes": file_bytes(
            [
                args.embedding_dir / "base_gallery.npy",
                args.embedding_dir / "correction_gallery.npy",
            ]
        ),
        "average_boundary_candidates": boundary_statistics(
            args.cache_dir, source.cutoffs
        ),
        "post_embedding_gate_ms_per_query": benchmark_gate(
            source,
            gate,
            float(report["regression_cost"]),
            min(args.max_queries, embedding_manifest["queries"]),
        ),
        "encoder_breakdown": embedding_manifest.get("encoder_breakdown"),
        "hardware": source.torch.cuda.get_device_name(source.device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
