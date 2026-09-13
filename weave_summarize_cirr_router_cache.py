#!/usr/bin/env python3
"""Summarize fixed actions and recovery/regression counts in CIRR router caches."""

import argparse
import json
from pathlib import Path

import numpy as np

from weave_train_cirr_router import load_cache, ranking_metrics


def summarize(cache, indices):
    global_ranks = np.asarray(cache["global_ranks"][indices])
    subset_ranks = np.asarray(cache["subset_ranks"][indices])
    base_global = global_ranks[:, 0] <= 5
    base_subset = subset_ranks[:, 0] <= 1
    actions = {}
    for action, name in enumerate(cache["manifest"]["actions"]):
        selected_global = global_ranks[:, action] <= 5
        selected_subset = subset_ranks[:, action] <= 1
        actions[name] = {
            "metrics": ranking_metrics(
                global_ranks,
                subset_ranks,
                np.full(len(indices), action, dtype=np.int64),
            ),
            "global_R@5": {
                "recoveries": int(np.sum(~base_global & selected_global)),
                "regressions": int(np.sum(base_global & ~selected_global)),
            },
            "subset_R@1": {
                "recoveries": int(np.sum(~base_subset & selected_subset)),
                "regressions": int(np.sum(base_subset & ~selected_subset)),
            },
        }
    return {"queries": len(indices), "actions": actions}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--eval-cache", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train = load_cache(args.train_cache)
    report = {
        name: summarize(train, np.flatnonzero(train["partitions"] == value))
        for name, value in (("train", 0), ("development", 1), ("internal_test", 2))
    }
    if args.eval_cache:
        evaluation = load_cache(args.eval_cache)
        report["evaluation"] = summarize(
            evaluation, np.arange(len(evaluation["features"]))
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
