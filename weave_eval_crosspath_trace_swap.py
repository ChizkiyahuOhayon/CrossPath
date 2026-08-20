#!/usr/bin/env python3
"""Evaluate the predeclared difficulty-matched CrossPath trace-swap control."""

import argparse
import json
from pathlib import Path

import numpy as np

from weave_crosspath import estimate_utilities
from weave_crosspath_gate import CrossPathGate, aggregate_cutoff_utilities
from weave_eval_crosspath_gate import infer_all, realized_utilities
from weave_train_crosspath_gate import (
    KS,
    RecordSource,
    policy_report,
    select_actions,
    sha256_file,
)


def adjacent_margin_partners(query_indices, margins):
    """Pair adjacent records after deterministic (margin, query-index) sorting."""
    indices = np.asarray(query_indices, dtype=np.int64)
    values = np.asarray(margins, dtype=np.float64)
    if indices.ndim != 1 or values.shape != indices.shape or indices.size == 0:
        raise ValueError("query indices and margins must be aligned non-empty vectors")
    if np.unique(indices).size != indices.size or not np.isfinite(values).all():
        raise ValueError("query indices must be unique and margins finite")

    order = np.lexsort((indices, values))
    partners = {int(index): int(index) for index in indices}
    for position in range(0, len(order) - 1, 2):
        left = int(indices[order[position]])
        right = int(indices[order[position + 1]])
        partners[left] = right
        partners[right] = left
    return partners


def transfer_trace(current_candidates, donor_candidates, donor_membership):
    """Transfer donor traces by gallery identity, using zero for absent IDs."""
    current = np.asarray(current_candidates, dtype=np.int64)
    donor = np.asarray(donor_candidates, dtype=np.int64)
    membership = np.asarray(donor_membership, dtype=np.uint8)
    if current.ndim != 1 or donor.ndim != 1:
        raise ValueError("candidate indices must be one-dimensional")
    if membership.ndim != 2 or membership.shape[0] != donor.size:
        raise ValueError("donor membership must align with donor candidates")
    if np.unique(current).size != current.size or np.unique(donor).size != donor.size:
        raise ValueError("candidate indices must be unique")
    if not np.isin(membership, (0, 1)).all():
        raise ValueError("donor membership must be binary")

    transferred = np.zeros((current.size, membership.shape[1]), dtype=np.uint8)
    donor_positions = {int(candidate): position for position, candidate in enumerate(donor)}
    overlap = 0
    for position, candidate in enumerate(current):
        donor_position = donor_positions.get(int(candidate))
        if donor_position is not None:
            transferred[position] = membership[donor_position]
            overlap += 1
    return transferred, overlap


def load_boundary_records(source):
    records = {}
    for shard_path in source.shard_paths:
        with np.load(shard_path) as shard:
            for local_index, query_index_value in enumerate(shard["query_indices"]):
                query_index = int(query_index_value)
                row = {
                    "target_ranks": shard["target_ranks"][local_index].astype(
                        np.int32, copy=True
                    ),
                    "base_margins": shard["base_margins"][local_index].astype(
                        np.float64, copy=True
                    ),
                    "cutoffs": {},
                }
                for k in KS:
                    prefix = f"k{k}_"
                    start = int(shard[prefix + "offsets"][local_index])
                    stop = int(shard[prefix + "offsets"][local_index + 1])
                    row["cutoffs"][k] = {
                        "candidate_indices": shard[prefix + "candidate_indices"][
                            start:stop
                        ].astype(np.int32, copy=True),
                        "membership": shard[prefix + "membership"][start:stop].astype(
                            np.uint8, copy=True
                        ),
                    }
                records[query_index] = row
    if len(records) != len(source.rows):
        raise ValueError("cache records do not cover embedding metadata exactly")
    return records


def infer_trace_swap(gate, source, regression_cost):
    gate.eval()
    records = load_boundary_records(source)
    query_indices = np.asarray(sorted(records), dtype=np.int32)
    partners = {}
    match_diagnostics = {}
    for cutoff_position, k in enumerate(KS):
        margins = np.asarray(
            [records[int(index)]["base_margins"][cutoff_position] for index in query_indices]
        )
        partners[k] = adjacent_margin_partners(query_indices, margins)
        paired = np.asarray(
            [partners[k][int(index)] != int(index) for index in query_indices]
        )
        gaps = np.asarray(
            [
                abs(
                    records[int(index)]["base_margins"][cutoff_position]
                    - records[partners[k][int(index)]]["base_margins"][cutoff_position]
                )
                for index in query_indices[paired]
            ]
        )
        match_diagnostics[f"R@{k}"] = {
            "paired_queries": int(paired.sum()),
            "unpaired_queries": int((~paired).sum()),
            "mean_absolute_margin_gap": float(gaps.mean()) if gaps.size else 0.0,
            "median_absolute_margin_gap": float(np.median(gaps)) if gaps.size else 0.0,
            "p95_absolute_margin_gap": float(np.quantile(gaps, 0.95)) if gaps.size else 0.0,
        }

    predicted = []
    target_ranks = []
    overlap_count = 0
    candidate_count = 0
    changed_bits = 0
    total_bits = 0
    trace_start = 4 * source.embedding_dim + 2
    trace_stop = trace_start + len(source.alphas)
    with source.torch.no_grad():
        for shard_path in source.shard_paths:
            with np.load(shard_path) as shard:
                for local_index, query_index_value in enumerate(shard["query_indices"]):
                    query_index = int(query_index_value)
                    cutoff_predictions = []
                    for k in KS:
                        _, features, _, membership = source.features(
                            shard, local_index, k
                        )
                        current = records[query_index]["cutoffs"][k]
                        donor = records[partners[k][query_index]]["cutoffs"][k]
                        swapped, overlap = transfer_trace(
                            current["candidate_indices"],
                            donor["candidate_indices"],
                            donor["membership"],
                        )
                        swapped_tensor = source.torch.tensor(
                            swapped, dtype=features.dtype, device=source.device
                        )
                        swapped_features = features.clone()
                        swapped_features[:, trace_start:trace_stop] = swapped_tensor
                        responsibility = gate.probabilities(
                            swapped_features
                        ).cpu().numpy()
                        cutoff_predictions.append(
                            estimate_utilities(
                                membership.cpu().numpy(),
                                responsibility,
                                regression_cost=regression_cost,
                            )
                        )
                        overlap_count += overlap
                        candidate_count += len(current["candidate_indices"])
                        changed_bits += int(
                            np.count_nonzero(swapped != current["membership"])
                        )
                        total_bits += int(swapped.size)
                    predicted.append(aggregate_cutoff_utilities(cutoff_predictions))
                    target_ranks.append(
                        records[query_index]["target_ranks"]
                    )

    return {
        "query_indices": query_indices,
        "target_ranks": np.stack(target_ranks),
        "predicted_utilities": np.stack(predicted),
        "match_diagnostics": match_diagnostics,
        "candidate_identity_overlap_rate": (
            float(overlap_count / candidate_count) if candidate_count else 0.0
        ),
        "changed_trace_bit_rate": (
            float(changed_bits / total_bits) if total_bits else 0.0
        ),
    }


def paired_difference(values_a, values_b, samples, seed):
    """Bootstrap the paired mean of a-b and return a two-sided 95% interval."""
    left = np.asarray(values_a, dtype=np.float64)
    right = np.asarray(values_b, dtype=np.float64)
    if left.ndim != 1 or right.shape != left.shape or left.size == 0 or samples <= 0:
        raise ValueError("paired bootstrap inputs must be aligned and non-empty")
    delta = left - right
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        stop = min(start + 256, samples)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        draws[start:stop] = delta[indices].mean(axis=1)
    return {
        "delta": float(delta.mean()),
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }


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

    source = RecordSource(args.embedding_dir, args.cache_dir, args.device, seed)
    gate = CrossPathGate(
        embedding_dim=source.embedding_dim,
        num_actions=len(source.alphas),
        hidden_width=width,
    ).to(source.device)
    checkpoint = args.gate_dir / f"gate_width{width}.pt"
    state = source.torch.load(checkpoint, map_location=source.device, weights_only=True)
    gate.load_state_dict(state)

    original = infer_all(gate, source, regression_cost)
    swapped = infer_trace_swap(gate, source, regression_cost)
    if not np.array_equal(original["query_indices"], swapped["query_indices"]):
        raise ValueError("original and trace-swap query order differs")
    if not np.array_equal(original["target_ranks"], swapped["target_ranks"]):
        raise ValueError("trace swap changed realized action rankings")

    utilities = realized_utilities(original["target_ranks"], regression_cost)
    original["realized_utilities"] = utilities
    swapped["realized_utilities"] = utilities
    original_actions = select_actions(original["predicted_utilities"], threshold)
    swapped_actions = select_actions(swapped["predicted_utilities"], threshold)
    original_report = policy_report(original, original_actions, regression_cost)
    swapped_report = policy_report(swapped, swapped_actions, regression_cost)

    rows = np.arange(len(original_actions))
    original_realized = utilities[rows, original_actions]
    swapped_realized = utilities[rows, swapped_actions]
    comparison = {
        "asymmetric_utility_original_minus_swap": paired_difference(
            original_realized,
            swapped_realized,
            args.bootstrap_samples,
            args.bootstrap_seed,
        ),
        "recall_original_minus_swap": {},
    }
    original_ranks = original["target_ranks"][rows, original_actions]
    swapped_ranks = original["target_ranks"][rows, swapped_actions]
    for offset, k in enumerate(KS, start=1):
        comparison["recall_original_minus_swap"][f"R@{k}"] = paired_difference(
            (original_ranks <= k).astype(np.float64) * 100.0,
            (swapped_ranks <= k).astype(np.float64) * 100.0,
            args.bootstrap_samples,
            args.bootstrap_seed + offset,
        )

    original_regressions = sum(
        row["regressions"] for row in original_report["transitions"].values()
    )
    swapped_regressions = sum(
        row["regressions"] for row in swapped_report["transitions"].values()
    )
    comparison["regressions_all_cutoffs"] = {
        "original": original_regressions,
        "trace_swap": swapped_regressions,
        "trace_swap_minus_original": swapped_regressions - original_regressions,
    }
    comparison["predeclared_trace_specificity_pass"] = bool(
        comparison["asymmetric_utility_original_minus_swap"]["ci95"][0] > 0.0
        and swapped_regressions > original_regressions
    )

    report = {
        "queries": len(original_actions),
        "training_seed": seed,
        "selected_width": width,
        "threshold": threshold,
        "regression_cost": regression_cost,
        "bootstrap_samples": args.bootstrap_samples,
        "original": original_report,
        "trace_swap": swapped_report,
        "comparison": comparison,
        "match_diagnostics": swapped["match_diagnostics"],
        "candidate_identity_overlap_rate": swapped["candidate_identity_overlap_rate"],
        "changed_trace_bit_rate": swapped["changed_trace_bit_rate"],
    }
    np.savez_compressed(
        args.output_dir / "trace_swap_evaluation.npz",
        query_indices=original["query_indices"],
        target_ranks=original["target_ranks"],
        original_actions=original_actions,
        trace_swap_actions=swapped_actions,
        original_predicted_utilities=original["predicted_utilities"],
        trace_swap_predicted_utilities=swapped["predicted_utilities"],
    )
    manifest = {
        "embedding_dir": str(args.embedding_dir.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "gate_dir": str(args.gate_dir.resolve()),
        "gate_manifest_sha256": sha256_file(gate_manifest_path),
        "gate_checkpoint_sha256": sha256_file(checkpoint),
        "cache_manifest_sha256": sha256_file(args.cache_dir / "manifest.json"),
        "bootstrap_seed": args.bootstrap_seed,
        "control_contract": (
            "within-cutoff adjacent (base_margin,query_index) pairing; transfer only "
            "same-gallery-ID trace input; zero trace for absent IDs; retain current "
            "query, candidates, embeddings, endpoint percentiles, rankings and threshold"
        ),
        "report": report,
        "script_sha256": sha256_file(__file__),
    }
    with (args.output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
