#!/usr/bin/env python3
"""Evaluate a frozen CIRR query router without retraining it."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from weave_cirr_query_router import CIRRQueryRouter, choose_actions
from weave_train_cirr_router import infer, load_cache, split_report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cache = load_cache(args.cache)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["actions"] != cache["manifest"]["actions"]:
        raise ValueError("checkpoint and cache use different actions")
    state = checkpoint["state_dict"]
    mean = state["feature_mean"]
    std = state["feature_std"]
    device = torch.device(args.device)
    model = CIRRQueryRouter(
        mean,
        std,
        action_count=len(checkpoint["actions"]),
        hidden_width=checkpoint["hidden_width"],
    ).to(device)
    model.load_state_dict(state)
    indices = np.arange(len(cache["features"]))
    logits = infer(
        model,
        cache["features"],
        indices,
        args.batch_size,
        device,
        checkpoint["feature_mode"],
    )
    if checkpoint["objective"] == "delta_regression":
        logits[:, 0] = 0.0
    actions = choose_actions(logits, checkpoint["threshold"])
    report = {
        "checkpoint": str(args.checkpoint),
        "threshold": checkpoint["threshold"],
        "objective": checkpoint["objective"],
        "feature_mode": checkpoint["feature_mode"],
        "metrics": split_report(
            cache, indices, actions, checkpoint["fixed_action"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.save(args.output.with_suffix(".actions.npy"), actions.astype(np.int16))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
