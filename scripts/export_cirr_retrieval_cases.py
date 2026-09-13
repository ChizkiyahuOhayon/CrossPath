#!/usr/bin/env python3
"""Export deterministic positive CIRR retrieval cases from real rankings."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from eval_cirr_cross_compatibility import (
    load_inputs,
    normalize,
    score_paths,
    target_ranks,
)


def select_rescues(base_ranks, method_ranks, count=6, hit_cutoff=5):
    candidates = np.flatnonzero(
        (base_ranks > hit_cutoff) & (method_ranks <= hit_cutoff)
    ).tolist()
    candidates.sort(
        key=lambda index: (
            -(int(base_ranks[index]) - int(method_ranks[index])),
            int(method_ranks[index]),
            index,
        )
    )
    if len(candidates) < count:
        raise ValueError(
            f"only {len(candidates)} base-miss/method-hit cases at R@{hit_cutoff}"
        )
    return candidates[:count]


def evaluate_rankings(embedding_dir, base_path, method_path, batch_size, top_k):
    rows, gallery_ids, sources, _, arrays = load_inputs(embedding_dir)
    gallery_index = {name: index for index, name in enumerate(gallery_ids)}
    targets = np.asarray([gallery_index[row["target_id"]] for row in rows])
    gallery0 = normalize(arrays["base_gallery"])
    gallery1 = normalize(arrays["correction_gallery"])
    base_ranks, method_ranks, base_top, method_top = [], [], [], []
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        paths = score_paths(
            normalize(arrays["base_queries"][start:stop]),
            normalize(arrays["correction_queries"][start:stop]),
            gallery0,
            gallery1,
            sources[start:stop],
        )
        base_scores = paths[base_path]
        method_scores = paths[method_path]
        base_ranks.append(target_ranks(base_scores, targets[start:stop]))
        method_ranks.append(target_ranks(method_scores, targets[start:stop]))
        base_top.append(np.argsort(-base_scores, axis=1, kind="stable")[:, :top_k])
        method_top.append(np.argsort(-method_scores, axis=1, kind="stable")[:, :top_k])
    return {
        "rows": rows,
        "gallery_ids": gallery_ids,
        "base_ranks": np.concatenate(base_ranks),
        "method_ranks": np.concatenate(method_ranks),
        "base_top": np.concatenate(base_top),
        "method_top": np.concatenate(method_top),
    }


def copy_originals(cirr_root, image_ids, output_dir):
    split = json.loads(
        (cirr_root / "cirr" / "image_splits" / "split.rc2.val.json").read_text()
    )
    copied = {}
    for image_id in sorted(set(image_ids)):
        source = cirr_root / split[image_id]
        destination = output_dir / image_id / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied[image_id] = str(destination.relative_to(output_dir.parent))
    return copied


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True, type=Path)
    parser.add_argument("--val-summary", required=True, type=Path)
    parser.add_argument("--cirr-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--hit-cutoff", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.count, args.hit_cutoff, args.top_k, args.batch_size) <= 0:
        raise ValueError("counts, cutoffs, and batch size must be positive")
    summary = json.loads(args.val_summary.read_text(encoding="utf-8"))
    base_path = summary["best_single_path"]
    method_path = summary["best_path"]
    result = evaluate_rankings(
        args.embedding_dir, base_path, method_path, args.batch_size, args.top_k
    )
    selected = select_rescues(
        result["base_ranks"], result["method_ranks"], args.count, args.hit_cutoff
    )
    gallery_ids = result["gallery_ids"]
    cases = []
    image_ids = []
    for index in selected:
        row = result["rows"][index]
        base_ids = [gallery_ids[value] for value in result["base_top"][index]]
        method_ids = [gallery_ids[value] for value in result["method_top"][index]]
        case = {
            "query_index": index,
            "caption": row["caption"],
            "source_id": row["source_id"],
            "target_id": row["target_id"],
            "base_rank": int(result["base_ranks"][index]),
            "method_rank": int(result["method_ranks"][index]),
            "base_top_ids": base_ids,
            "method_top_ids": method_ids,
        }
        cases.append(case)
        image_ids.extend((row["source_id"], row["target_id"], *base_ids, *method_ids))
    originals = copy_originals(
        args.cirr_root, image_ids, args.output_dir / "source_images"
    )
    report = {
        "dataset": "CIRR val",
        "selection": f"largest deterministic {base_path}-miss/{method_path}-hit rescues at R@{args.hit_cutoff}",
        "base_path": base_path,
        "method_path": method_path,
        "top_k": args.top_k,
        "cases": cases,
        "source_images": originals,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "cirr_retrieval_cases.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
