#!/usr/bin/env python3
"""Train and internally falsify the CrossPath responsibility gate."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from weave_crosspath import calibrate_threshold, estimate_utilities
from weave_crosspath_gate import (
    CrossPathGate,
    aggregate_cutoff_utilities,
    batch_listwise_loss,
    realized_cutoff_utilities,
)


KS = (1, 5, 10)
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)


def pair_partition(row, seed):
    left, right = sorted((str(row["source_id"]), str(row["target_id"])))
    payload = "\0".join((str(seed), str(row["dataset"]), left, right)).encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
    if value < SPLIT_FRACTIONS[0]:
        return "train"
    if value < SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]:
        return "calibration"
    return "test"


def policy_metrics(target_ranks, selected_actions, cutoffs=KS):
    ranks = np.asarray(target_ranks, dtype=np.int64)
    actions = np.asarray(selected_actions, dtype=np.int64)
    if ranks.ndim != 2 or actions.shape != (ranks.shape[0],):
        raise ValueError("target ranks and selected actions are not aligned")
    if np.any(actions < 0) or np.any(actions >= ranks.shape[1]):
        raise ValueError("selected action is outside the path")
    chosen = ranks[np.arange(ranks.shape[0]), actions]
    return {
        f"R@{k}": round(float((chosen <= k).mean() * 100.0), 6)
        for k in cutoffs
    }


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RecordSource:
    def __init__(
        self,
        embedding_dir,
        cache_dir,
        device,
        seed,
        zero_trace=False,
        feature_mode="full",
    ):
        import torch
        import torch.nn.functional as functional

        self.torch = torch
        self.functional = functional
        self.device = torch.device(device)
        if zero_trace:
            if feature_mode != "full":
                raise ValueError("zero_trace and feature_mode cannot both be set")
            feature_mode = "no_trace"
        if feature_mode not in (
            "full",
            "query_only",
            "margin_only",
            "percentile_only",
            "no_trace",
        ):
            raise ValueError(f"unknown feature mode: {feature_mode}")
        self.feature_mode = feature_mode
        self.zero_trace = feature_mode == "no_trace"
        self.rows = load_jsonl(Path(embedding_dir) / "queries.jsonl")
        self.partitions = [pair_partition(row, seed) for row in self.rows]
        self.query_array = np.load(
            Path(embedding_dir) / "base_queries.npy", mmap_mode="r"
        )
        gallery_array = np.load(
            Path(embedding_dir) / "base_gallery.npy", mmap_mode="r"
        )
        self.gallery = functional.normalize(
            torch.tensor(gallery_array, dtype=torch.float32, device=self.device),
            dim=-1,
        )
        self.embedding_dim = int(self.gallery.shape[1])

        manifest = json.loads((Path(cache_dir) / "manifest.json").read_text())
        self.alphas = np.asarray(manifest["alphas"], dtype=np.float64)
        self.endpoint_count = int(manifest.get("endpoint_count", 2))
        self.cutoffs = tuple(int(k) for k in manifest.get("cutoffs", KS))
        if not self.cutoffs or any(k <= 0 for k in self.cutoffs):
            raise ValueError("cache manifest has invalid cutoffs")
        self.shard_paths = [Path(cache_dir) / row["path"] for row in manifest["shards"]]
        if len(self.rows) < manifest["queries"]:
            raise ValueError("cache has more queries than embedding metadata")

    def features(self, shard, local_index, k):
        torch = self.torch
        query_index = int(shard["query_indices"][local_index])
        prefix = f"k{k}_"
        start = int(shard[prefix + "offsets"][local_index])
        stop = int(shard[prefix + "offsets"][local_index + 1])
        candidate_indices = torch.tensor(
            shard[prefix + "candidate_indices"][start:stop],
            dtype=torch.long,
            device=self.device,
        )
        documents = self.gallery[candidate_indices]
        query = self.functional.normalize(
            torch.tensor(
                self.query_array[query_index],
                dtype=torch.float32,
                device=self.device,
            ),
            dim=0,
        )
        tiled_query = query.unsqueeze(0).expand(documents.shape[0], -1)
        percentiles = torch.tensor(
            shard[prefix + "endpoint_percentiles"][start:stop],
            dtype=torch.float32,
            device=self.device,
        )
        membership = torch.tensor(
            shard[prefix + "membership"][start:stop],
            dtype=torch.float32,
            device=self.device,
        )
        if self.feature_mode == "query_only":
            documents = torch.zeros_like(documents)
            products = torch.zeros_like(tiled_query)
            differences = torch.zeros_like(tiled_query)
            percentiles = torch.zeros_like(percentiles)
            feature_membership = torch.zeros_like(membership)
        elif self.feature_mode in ("margin_only", "percentile_only"):
            tiled_query = torch.zeros_like(tiled_query)
            documents = torch.zeros_like(documents)
            products = torch.zeros_like(documents)
            differences = torch.zeros_like(documents)
            feature_membership = torch.zeros_like(membership)
        else:
            products = tiled_query * documents
            differences = torch.abs(tiled_query - documents)
            feature_membership = (
                torch.zeros_like(membership) if self.zero_trace else membership
            )
        cutoff = torch.full(
            (documents.shape[0], 1),
            k / max(self.cutoffs),
            dtype=torch.float32,
            device=self.device,
        )
        features = torch.cat(
            [
                tiled_query,
                documents,
                products,
                differences,
                percentiles,
                feature_membership,
                cutoff,
            ],
            dim=1,
        )
        target = int(shard[prefix + "target_positions"][local_index])
        return query_index, features, target, membership

    def iter_records(self, partition, epoch_seed=None):
        rng = np.random.default_rng(epoch_seed)
        shard_order = np.arange(len(self.shard_paths))
        if epoch_seed is not None:
            rng.shuffle(shard_order)
        for shard_number in shard_order:
            with np.load(self.shard_paths[int(shard_number)]) as shard:
                local_order = np.arange(len(shard["query_indices"]))
                if epoch_seed is not None:
                    rng.shuffle(local_order)
                for local_index in local_order:
                    query_index = int(shard["query_indices"][local_index])
                    if self.partitions[query_index] != partition:
                        continue
                    cutoff_order = list(self.cutoffs)
                    if epoch_seed is not None:
                        rng.shuffle(cutoff_order)
                    for k in cutoff_order:
                        yield self.features(shard, int(local_index), k)[:3]

    def iter_queries(self, partition):
        for shard_path in self.shard_paths:
            with np.load(shard_path) as shard:
                for local_index, query_index_value in enumerate(shard["query_indices"]):
                    query_index = int(query_index_value)
                    if self.partitions[query_index] == partition:
                        yield shard, local_index, query_index


def run_record_batches(records, batch_records):
    features = []
    targets = []
    for _, record_features, target in records:
        features.append(record_features)
        targets.append(target)
        if len(features) == batch_records:
            yield features, targets
            features, targets = [], []
    if features:
        yield features, targets


def evaluate_nll(gate, source, partition, batch_records):
    gate.eval()
    total = 0.0
    count = 0
    with source.torch.no_grad():
        for features, targets in run_record_batches(
            source.iter_records(partition), batch_records
        ):
            loss = batch_listwise_loss(gate, features, targets)
            total += float(loss) * len(features)
            count += len(features)
    if not count:
        raise ValueError(f"empty split: {partition}")
    return total / count


def train_width(source, width, args):
    torch = source.torch
    torch.manual_seed(args.seed)
    gate = CrossPathGate(
        embedding_dim=source.embedding_dim,
        num_actions=len(source.alphas),
        hidden_width=width,
        endpoint_count=source.endpoint_count,
    ).to(source.device)
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history = []
    for epoch in range(args.epochs):
        gate.train()
        losses = []
        records = source.iter_records("train", args.seed + epoch)
        for features, targets in run_record_batches(records, args.batch_records):
            optimizer.zero_grad(set_to_none=True)
            loss = batch_listwise_loss(gate, features, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
        calibration_nll = evaluate_nll(
            gate, source, "calibration", args.batch_records
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_batch_loss": float(np.mean(losses)),
                "calibration_nll": calibration_nll,
            }
        )
        print(json.dumps({"width": width, **history[-1]}), flush=True)
    return gate, history


def infer_split(gate, source, partition, regression_cost):
    gate.eval()
    query_indices = []
    target_ranks = []
    predicted = []
    realized = []
    with source.torch.no_grad():
        for shard, local_index, query_index in source.iter_queries(partition):
            cutoff_predictions = []
            for k in source.cutoffs:
                _, features, _, membership = source.features(shard, local_index, k)
                responsibility = gate.probabilities(features).cpu().numpy()
                cutoff_predictions.append(
                    estimate_utilities(
                        membership.cpu().numpy(),
                        responsibility,
                        regression_cost=regression_cost,
                    )
                )
            ranks = shard["target_ranks"][local_index].astype(np.int32)
            query_indices.append(query_index)
            target_ranks.append(ranks)
            predicted.append(aggregate_cutoff_utilities(cutoff_predictions))
            realized.append(
                aggregate_cutoff_utilities(
                    realized_cutoff_utilities(
                        ranks, source.cutoffs, regression_cost=regression_cost
                    )
                )
            )
    return {
        "query_indices": np.asarray(query_indices, dtype=np.int32),
        "target_ranks": np.stack(target_ranks),
        "predicted_utilities": np.stack(predicted),
        "realized_utilities": np.stack(realized),
    }


def select_actions(predicted, threshold):
    best = np.argmax(predicted, axis=1)
    values = predicted[np.arange(predicted.shape[0]), best]
    return np.where(values > threshold, best, 0).astype(np.int32)


def policy_report(bundle, actions, regression_cost, cutoffs=KS):
    ranks = bundle["target_ranks"]
    base_actions = np.zeros(len(actions), dtype=np.int32)
    full_actions = np.full(len(actions), ranks.shape[1] - 1, dtype=np.int32)
    oracle_actions = np.argmax(bundle["realized_utilities"], axis=1).astype(np.int32)
    fixed_action_metrics = {
        str(action): policy_metrics(
            ranks,
            np.full(len(actions), action, dtype=np.int32),
            cutoffs,
        )
        for action in range(ranks.shape[1])
    }
    report = {
        "queries": len(actions),
        "metrics": policy_metrics(ranks, actions, cutoffs),
        "base_metrics": policy_metrics(ranks, base_actions, cutoffs),
        "full_correction_metrics": policy_metrics(ranks, full_actions, cutoffs),
        "objective_oracle_metrics": policy_metrics(ranks, oracle_actions, cutoffs),
        "fixed_action_metrics": fixed_action_metrics,
        "per_cutoff_path_ceiling": {
            f"R@{k}": round(float(np.any(ranks <= k, axis=1).mean() * 100.0), 6)
            for k in cutoffs
        },
        "intervention_rate": float((actions != 0).mean()),
        "mean_realized_utility": float(
            bundle["realized_utilities"][np.arange(len(actions)), actions].mean()
        ),
        "objective_oracle_mean_realized_utility": float(
            bundle["realized_utilities"][
                np.arange(len(oracle_actions)), oracle_actions
            ].mean()
        ),
        "regression_cost": regression_cost,
        "transitions": {},
    }
    selected_ranks = ranks[np.arange(len(actions)), actions]
    base_ranks = ranks[:, 0]
    for k in cutoffs:
        base_hits = base_ranks <= k
        selected_hits = selected_ranks <= k
        report["transitions"][f"R@{k}"] = {
            "recoveries": int(((~base_hits) & selected_hits).sum()),
            "regressions": int((base_hits & (~selected_hits)).sum()),
        }
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--widths", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-records", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--regression-cost", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--zero-trace",
        action="store_true",
        help="Set all membership-trace input features to zero while retaining true action utility.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("full", "query_only", "margin_only", "percentile_only", "no_trace"),
        default="full",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if (
        args.epochs <= 0
        or args.batch_records <= 0
        or any(width <= 0 for width in args.widths)
        or args.learning_rate <= 0
        or args.regression_cost <= 0
    ):
        raise ValueError("invalid training configuration")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    if args.zero_trace and args.feature_mode != "full":
        raise ValueError("use either --zero-trace or --feature-mode, not both")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = RecordSource(
        args.embedding_dir,
        args.cache_dir,
        args.device,
        args.seed,
        zero_trace=args.zero_trace,
        feature_mode=args.feature_mode,
    )
    split_counts = {
        split: source.partitions.count(split)
        for split in ("train", "calibration", "test")
    }
    if not all(split_counts.values()):
        raise ValueError(f"empty partition: {split_counts}")

    candidates = []
    for width in args.widths:
        gate, history = train_width(source, width, args)
        path = args.output_dir / f"gate_width{width}.pt"
        source.torch.save(gate.state_dict(), path)
        candidates.append(
            {
                "width": width,
                "gate": gate,
                "history": history,
                "calibration_nll": history[-1]["calibration_nll"],
                "checkpoint": path.name,
            }
        )
    selected = min(candidates, key=lambda row: (row["calibration_nll"], row["width"]))
    gate = selected["gate"]

    calibration = infer_split(
        gate, source, "calibration", args.regression_cost
    )
    threshold, calibration_utility = calibrate_threshold(
        calibration["predicted_utilities"], calibration["realized_utilities"]
    )
    test = infer_split(gate, source, "test", args.regression_cost)
    test_actions = select_actions(test["predicted_utilities"], threshold)
    report = policy_report(
        test, test_actions, args.regression_cost, cutoffs=source.cutoffs
    )
    report.update(
        {
            "selected_width": selected["width"],
            "selected_calibration_nll": selected["calibration_nll"],
            "threshold": threshold,
            "calibration_mean_utility": calibration_utility,
            "split_counts": split_counts,
            "split_fractions": list(SPLIT_FRACTIONS),
            "seed": args.seed,
            "single_ranking_protocol": True,
            "cutoff_aggregation": "equal_mean_"
            + "_".join(f"R{k}" for k in source.cutoffs)
            + "_utility",
        }
    )
    metrics = report["metrics"]
    base = report["base_metrics"]
    report["internal_go"] = bool(
        report["intervention_rate"] > 0.0
        and report["mean_realized_utility"] > 0.0
        and all(metrics[name] >= base[name] for name in base)
        and any(metrics[name] > base[name] for name in base)
    )

    np.savez_compressed(
        args.output_dir / "internal_test_policy.npz",
        query_indices=test["query_indices"],
        target_ranks=test["target_ranks"],
        predicted_utilities=test["predicted_utilities"],
        realized_utilities=test["realized_utilities"],
        selected_actions=test_actions,
    )
    serializable_candidates = [
        {key: value for key, value in row.items() if key != "gate"}
        for row in candidates
    ]
    manifest = {
        "embedding_dir": str(args.embedding_dir.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "cache_manifest_sha256": sha256_file(args.cache_dir / "manifest.json"),
        "training": vars(args) | {"output_dir": str(args.output_dir.resolve())},
        "candidates": serializable_candidates,
        "report": report,
        "script_sha256": sha256_file(__file__),
    }
    for key, value in list(manifest["training"].items()):
        if isinstance(value, Path):
            manifest["training"][key] = str(value.resolve())
    with (args.output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
