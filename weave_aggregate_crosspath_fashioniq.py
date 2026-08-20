#!/usr/bin/env python3
"""Aggregate the three category-specific FashionIQ CrossPath evaluations."""

import argparse
import json
from pathlib import Path


CATEGORIES = ("dress", "shirt", "toptee")
GROUPS = (
    "metrics",
    "base_metrics",
    "full_correction_metrics",
    "objective_oracle_metrics",
)


def mean_metrics(reports, group):
    keys = reports[0][group]
    return {
        key: round(sum(report[group][key] for report in reports) / len(reports), 6)
        for key in keys
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    category_reports = {}
    gallery_protocols = set()
    for category in CATEGORIES:
        path = args.run_root / category / "official" / "eval" / "manifest.json"
        category_reports[category] = json.loads(path.read_text())["report"]
        embedding_manifest = args.run_root / category / "official" / "embeddings" / "manifest.json"
        if embedding_manifest.exists():
            gallery_protocols.add(
                json.loads(embedding_manifest.read_text()).get(
                    "gallery_protocol", "val-split"
                )
            )
    if len(gallery_protocols) > 1:
        raise ValueError("FashionIQ categories use different gallery protocols")
    gallery_protocol = next(iter(gallery_protocols), "val-split")
    reports = [category_reports[category] for category in CATEGORIES]

    summary = {
        "protocol": f"FashionIQ {gallery_protocol}; source excluded",
        "categories": category_reports,
        "average": {group: mean_metrics(reports, group) for group in GROUPS},
    }
    actions = sorted(reports[0]["fixed_action_metrics"], key=int)
    summary["average"]["fixed_action_metrics"] = {
        action: {
            key: round(
                sum(report["fixed_action_metrics"][action][key] for report in reports)
                / len(reports),
                6,
            )
            for key in reports[0]["fixed_action_metrics"][action]
        }
        for action in actions
    }
    metric_keys = list(summary["average"]["metrics"])
    best_action = max(
        actions,
        key=lambda action: sum(
            summary["average"]["fixed_action_metrics"][action][key]
            for key in metric_keys
        ),
    )
    summary["average"]["best_fixed_action"] = {
        "action": int(best_action),
        "metrics": summary["average"]["fixed_action_metrics"][best_action],
    }
    summary["average"]["delta_vs_base"] = {
        key: round(
            summary["average"]["metrics"][key]
            - summary["average"]["base_metrics"][key],
            6,
        )
        for key in metric_keys
    }
    output = args.run_root / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["average"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
