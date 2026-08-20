#!/usr/bin/env python3
"""Evaluate one frozen CrossPath gate without tuning on the evaluation set."""

import argparse
import json
from pathlib import Path

import numpy as np

from weave_crosspath import estimate_utilities
from weave_crosspath_gate import CrossPathGate, aggregate_cutoff_utilities
from weave_train_crosspath_gate import (
    KS,
    RecordSource,
    policy_report,
    select_actions,
    sha256_file,
)


def infer_all(gate, source, regression_cost):
    gate.eval()
    query_indices = []
    target_ranks = []
    predicted = []
    with source.torch.no_grad():
        for shard_path in source.shard_paths:
            with np.load(shard_path) as shard:
                for local_index, query_index_value in enumerate(shard["query_indices"]):
                    query_index = int(query_index_value)
                    cutoff_predictions = []
                    for k in source.cutoffs:
                        _, features, _, membership = source.features(
                            shard, local_index, k
                        )
                        responsibility = gate.probabilities(features).cpu().numpy()
                        cutoff_predictions.append(
                            estimate_utilities(
                                membership.cpu().numpy(),
                                responsibility,
                                regression_cost=regression_cost,
                            )
                        )
                    query_indices.append(query_index)
                    target_ranks.append(shard["target_ranks"][local_index])
                    predicted.append(aggregate_cutoff_utilities(cutoff_predictions))
    return {
        "query_indices": np.asarray(query_indices, dtype=np.int32),
        "target_ranks": np.stack(target_ranks).astype(np.int32, copy=False),
        "predicted_utilities": np.stack(predicted),
    }


def realized_utilities(target_ranks, regression_cost, cutoffs=KS):
    ranks = np.asarray(target_ranks, dtype=np.int64)
    cutoff_array = np.asarray(cutoffs, dtype=np.int64)
    hits = ranks[:, None, :] <= cutoff_array[None, :, None]
    base = hits[:, :, :1]
    cutoff = ((~base) & hits).astype(np.float64) - regression_cost * (
        base & (~hits)
    ).astype(np.float64)
    return cutoff.mean(axis=1)


def paired_bootstrap(target_ranks, actions, samples, seed, cutoffs=KS):
    ranks = np.asarray(target_ranks, dtype=np.int64)
    selected = np.asarray(actions, dtype=np.int64)
    if samples <= 0 or ranks.ndim != 2 or selected.shape != (len(ranks),):
        raise ValueError("invalid paired bootstrap inputs")
    chosen = ranks[np.arange(len(ranks)), selected]
    rng = np.random.default_rng(seed)
    report = {}
    for k in cutoffs:
        delta = (chosen <= k).astype(np.float64) - (ranks[:, 0] <= k).astype(
            np.float64
        )
        draws = np.empty(samples, dtype=np.float64)
        for start in range(0, samples, 256):
            stop = min(start + 256, samples)
            indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
            draws[start:stop] = delta[indices].mean(axis=1) * 100.0
        report[f"R@{k}"] = {
            "delta": float(delta.mean() * 100.0),
            "ci95": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
        }
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--gate-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gate_manifest_path = args.gate_dir / "manifest.json"
    gate_manifest = json.loads(gate_manifest_path.read_text())
    training_report = gate_manifest["report"]
    width = int(training_report["selected_width"])
    threshold = float(training_report["threshold"])
    regression_cost = float(training_report["regression_cost"])
    seed = int(training_report["seed"])

    training = gate_manifest.get("training", {})
    zero_trace = bool(training.get("zero_trace", False))
    feature_mode = training.get("feature_mode", "full")
    source = RecordSource(
        args.embedding_dir,
        args.cache_dir,
        args.device,
        seed,
        zero_trace=zero_trace,
        feature_mode=feature_mode,
    )
    gate = CrossPathGate(
        embedding_dim=source.embedding_dim,
        num_actions=len(source.alphas),
        hidden_width=width,
        endpoint_count=source.endpoint_count,
    ).to(source.device)
    checkpoint = args.gate_dir / f"gate_width{width}.pt"
    state = source.torch.load(
        checkpoint, map_location=source.device, weights_only=True
    )
    gate.load_state_dict(state)

    bundle = infer_all(gate, source, regression_cost)
    bundle["realized_utilities"] = realized_utilities(
        bundle["target_ranks"], regression_cost, source.cutoffs
    )
    actions = select_actions(bundle["predicted_utilities"], threshold)
    report = policy_report(bundle, actions, regression_cost, source.cutoffs)
    report.update(
        {
            "selected_width": width,
            "threshold": threshold,
            "training_seed": seed,
            "zero_trace": zero_trace,
            "single_ranking_protocol": True,
            "paired_bootstrap_samples": args.bootstrap_samples,
            "paired_bootstrap": paired_bootstrap(
                bundle["target_ranks"],
                actions,
                args.bootstrap_samples,
                args.bootstrap_seed,
                source.cutoffs,
            ),
        }
    )

    np.savez_compressed(
        args.output_dir / "evaluation.npz",
        query_indices=bundle["query_indices"],
        target_ranks=bundle["target_ranks"],
        predicted_utilities=bundle["predicted_utilities"],
        selected_actions=actions,
    )
    manifest = {
        "embedding_dir": str(args.embedding_dir.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "gate_dir": str(args.gate_dir.resolve()),
        "gate_manifest_sha256": sha256_file(gate_manifest_path),
        "gate_checkpoint_sha256": sha256_file(checkpoint),
        "cache_manifest_sha256": sha256_file(args.cache_dir / "manifest.json"),
        "report": report,
        "bootstrap_seed": args.bootstrap_seed,
        "script_sha256": sha256_file(__file__),
    }
    with (args.output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
