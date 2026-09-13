#!/usr/bin/env python3
"""Train a conservative CIRR query router and evaluate a frozen cache."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from weave_cirr_query_router import (
    CIRRQueryRouter,
    choose_actions,
    cirr_avg,
    soft_oracle_targets,
)


def load_cache(path):
    path = Path(path)
    return {
        "manifest": json.loads((path / "manifest.json").read_text()),
        "features": np.load(path / "features.npy", mmap_mode="r"),
        "global_ranks": np.load(path / "global_ranks.npy", mmap_mode="r"),
        "subset_ranks": np.load(path / "subset_ranks.npy", mmap_mode="r"),
        "partitions": np.load(path / "partitions.npy", mmap_mode="r"),
    }


def ranking_metrics(global_ranks, subset_ranks, actions):
    rows = np.arange(len(actions))
    global_selected = global_ranks[rows, actions]
    subset_selected = subset_ranks[rows, actions]
    report = {
        f"R@{cutoff}": round(float(np.mean(global_selected <= cutoff) * 100.0), 6)
        for cutoff in (1, 5, 10, 50)
    }
    report.update(
        {
            f"subset_R@{cutoff}": round(
                float(np.mean(subset_selected <= cutoff) * 100.0), 6
            )
            for cutoff in (1, 2, 3)
        }
    )
    report["Avg"] = round(
        0.5 * (report["R@5"] + report["subset_R@1"]), 6
    )
    return report


def fixed_action(cache, indices):
    scores = [
        cirr_avg(
            cache["global_ranks"][indices],
            cache["subset_ranks"][indices],
            np.full(len(indices), action, dtype=np.int64),
        )
        for action in range(cache["global_ranks"].shape[1])
    ]
    return int(np.argmax(scores))


def calibrate(logits, global_ranks, subset_ranks):
    best = np.argmax(logits, axis=1)
    margins = logits[np.arange(len(logits)), best] - logits[:, 0]
    nonbase = margins[best != 0]
    candidates = [-np.inf, np.inf]
    if len(nonbase):
        candidates.extend(np.quantile(nonbase, np.linspace(0.0, 1.0, 101)).tolist())
    records = []
    for threshold in sorted(set(candidates)):
        actions = choose_actions(logits, threshold)
        records.append(
            (
                cirr_avg(global_ranks, subset_ranks, actions),
                -float(np.mean(actions != 0)),
                float(threshold),
                actions,
            )
        )
    return max(records, key=lambda record: (record[0], record[1], record[2]))


def feature_batch(features, indices, feature_mode):
    batch = np.asarray(features[indices])
    return batch if feature_mode == "full" else batch[:, -32:]


def infer(model, features, indices, batch_size, device, feature_mode):
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = feature_batch(
                features, indices[start : start + batch_size], feature_mode
            )
            outputs.append(
                model(torch.as_tensor(batch, device=device)).float().cpu().numpy()
            )
    return np.concatenate(outputs)


def split_report(cache, indices, actions, fixed):
    global_ranks = np.asarray(cache["global_ranks"][indices])
    subset_ranks = np.asarray(cache["subset_ranks"][indices])
    base_actions = np.zeros(len(indices), dtype=np.int64)
    fixed_actions = np.full(len(indices), fixed, dtype=np.int64)
    targets = soft_oracle_targets(global_ranks, subset_ranks)
    oracle_actions = np.argmax(targets, axis=1)
    return {
        "queries": len(indices),
        "base": ranking_metrics(global_ranks, subset_ranks, base_actions),
        "fixed": ranking_metrics(global_ranks, subset_ranks, fixed_actions),
        "router": ranking_metrics(global_ranks, subset_ranks, actions),
        "oracle": ranking_metrics(global_ranks, subset_ranks, oracle_actions),
        "intervention_rate": round(float(np.mean(actions != 0)), 6),
        "action_counts": {
            str(action): int(np.sum(actions == action))
            for action in np.unique(actions)
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--eval-cache", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hidden-width", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--nonbase-weight", type=float, default=2.0)
    parser.add_argument(
        "--objective",
        choices=("oracle_classification", "delta_regression"),
        default="oracle_classification",
    )
    parser.add_argument(
        "--feature-mode", choices=("full", "retrieval"), default="full"
    )
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_width <= 0:
        raise ValueError("training dimensions must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache = load_cache(args.train_cache)
    train_indices = np.flatnonzero(cache["partitions"] == 0)
    development_indices = np.flatnonzero(cache["partitions"] == 1)
    test_indices = np.flatnonzero(cache["partitions"] == 2)
    train_features = feature_batch(cache["features"], train_indices, args.feature_mode)
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.maximum(std, 1e-6)
    train_targets = soft_oracle_targets(
        np.asarray(cache["global_ranks"][train_indices]),
        np.asarray(cache["subset_ranks"][train_indices]),
    )
    train_utility = (
        (np.asarray(cache["global_ranks"][train_indices]) <= 5).astype(np.float32)
        + (np.asarray(cache["subset_ranks"][train_indices]) <= 1).astype(np.float32)
    )
    train_deltas = train_utility - train_utility[:, :1]
    device = torch.device(args.device)
    model = CIRRQueryRouter(
        mean,
        std,
        action_count=len(cache["manifest"]["actions"]),
        hidden_width=args.hidden_width,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    history, best_state, best_threshold, best_development = [], None, None, -np.inf

    for epoch in range(args.epochs):
        order = rng.permutation(len(train_indices))
        losses = []
        model.train()
        for start in range(0, len(order), args.batch_size):
            positions = order[start : start + args.batch_size]
            indices = train_indices[positions]
            features = torch.as_tensor(
                feature_batch(cache["features"], indices, args.feature_mode), device=device
            )
            targets = torch.as_tensor(train_targets[positions], device=device)
            weights = torch.where(
                targets[:, 0] < 1.0,
                torch.full_like(targets[:, 0], args.nonbase_weight),
                torch.ones_like(targets[:, 0]),
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            if args.objective == "oracle_classification":
                loss = (
                    -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1) * weights
                ).mean()
            else:
                deltas = torch.as_tensor(train_deltas[positions], device=device)
                delta_weights = 1.0 + args.nonbase_weight * torch.abs(deltas)
                loss = (
                    torch.nn.functional.smooth_l1_loss(
                        logits, deltas, reduction="none"
                    )
                    * delta_weights
                ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        development_logits = infer(
            model,
            cache["features"],
            development_indices,
            args.batch_size,
            device,
            args.feature_mode,
        )
        if args.objective == "delta_regression":
            development_logits[:, 0] = 0.0
        calibrated = calibrate(
            development_logits,
            np.asarray(cache["global_ranks"][development_indices]),
            np.asarray(cache["subset_ranks"][development_indices]),
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "development_Avg": calibrated[0],
            "development_intervention_rate": -calibrated[1],
            "threshold": calibrated[2],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if calibrated[0] > best_development:
            best_development = calibrated[0]
            best_threshold = calibrated[2]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    fixed = fixed_action(cache, development_indices)
    test_logits = infer(
        model,
        cache["features"],
        test_indices,
        args.batch_size,
        device,
        args.feature_mode,
    )
    if args.objective == "delta_regression":
        test_logits[:, 0] = 0.0
    test_actions = choose_actions(test_logits, best_threshold)
    report = {
        "actions": cache["manifest"]["actions"],
        "fixed_action": fixed,
        "fixed_action_name": cache["manifest"]["actions"][fixed],
        "objective": args.objective,
        "feature_mode": args.feature_mode,
        "threshold": best_threshold,
        "history": history,
        "internal_test": split_report(cache, test_indices, test_actions, fixed),
    }

    if args.eval_cache:
        evaluation = load_cache(args.eval_cache)
        if evaluation["manifest"]["actions"] != cache["manifest"]["actions"]:
            raise ValueError("training and evaluation caches use different actions")
        indices = np.arange(len(evaluation["features"]))
        logits = infer(
            model,
            evaluation["features"],
            indices,
            args.batch_size,
            device,
            args.feature_mode,
        )
        if args.objective == "delta_regression":
            logits[:, 0] = 0.0
        actions = choose_actions(logits, best_threshold)
        report["evaluation"] = split_report(evaluation, indices, actions, fixed)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "actions": cache["manifest"]["actions"],
            "hidden_width": args.hidden_width,
            "threshold": best_threshold,
            "fixed_action": fixed,
            "objective": args.objective,
            "feature_mode": args.feature_mode,
        },
        args.output_dir / "router.pt",
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
