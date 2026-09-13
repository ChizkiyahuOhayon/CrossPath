#!/usr/bin/env python3
"""Select deterministic CrossPath retrieval cases and copy their source images."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from eval_cross_compatibility import (
    exclude_sources_,
    load_alignment,
    normalized_tensor,
    target_ranks,
    validate_endpoints,
)


CATEGORIES = ("dress", "shirt", "toptee")


def load_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_category_paths(values, option):
    paths = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} entries must use category=path")
        category, raw_path = value.split("=", 1)
        if category not in CATEGORIES or category in paths:
            raise ValueError(f"invalid or duplicate category in {option}: {category}")
        paths[category] = Path(raw_path)
    if set(paths) != set(CATEGORIES):
        raise ValueError(f"{option} must provide dress, shirt, and toptee")
    return paths


def evaluate_rankings(embedding_dir, device, batch_size, exclude_source, top_k):
    import torch

    rows, gallery_ids, targets, sources = load_alignment(embedding_dir)
    arrays = {
        name: np.load(embedding_dir / f"{name}.npy", mmap_mode="r")
        for name in (
            "base_gallery",
            "correction_gallery",
            "base_queries",
            "correction_queries",
        )
    }
    validate_endpoints(
        {
            "base": (arrays["base_queries"], arrays["base_gallery"]),
            "correction": (
                arrays["correction_queries"],
                arrays["correction_gallery"],
            ),
        },
        len(rows),
        len(gallery_ids),
    )
    gallery0 = normalized_tensor(arrays["base_gallery"], device)
    gallery1 = normalized_tensor(arrays["correction_gallery"], device)
    base_ranks, method_ranks = [], []
    base_top, method_top = [], []

    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        query0 = normalized_tensor(arrays["base_queries"][start:stop], device)
        query1 = normalized_tensor(arrays["correction_queries"][start:stop], device)
        with torch.no_grad():
            score00 = query0 @ gallery0.T
            score01 = query0 @ gallery1.T
            score10 = query1 @ gallery0.T
            score11 = query1 @ gallery1.T
            if exclude_source:
                exclude_sources_(
                    (score00, score01, score10, score11), sources[start:stop]
                )
            baseline = score11 if exclude_source else score00
            method = 0.5 * (score01 + score10)
            base_ranks.append(
                target_ranks(baseline.cpu().numpy(), targets[start:stop])
            )
            method_ranks.append(target_ranks(method.cpu().numpy(), targets[start:stop]))
            base_top.append(
                torch.argsort(baseline, dim=1, descending=True, stable=True)[:, :top_k]
                .cpu()
                .numpy()
            )
            method_top.append(
                torch.argsort(method, dim=1, descending=True, stable=True)[:, :top_k]
                .cpu()
                .numpy()
            )

    return {
        "rows": rows,
        "gallery_ids": gallery_ids,
        "base_ranks": np.concatenate(base_ranks),
        "method_ranks": np.concatenate(method_ranks),
        "base_top": np.concatenate(base_top),
        "method_top": np.concatenate(method_top),
    }


def select_cases(base_ranks, method_ranks, cutoffs, method_rank_limit=None):
    """Select the largest deterministic rescue at each cutoff."""
    selected = []
    used = set()
    for cutoff in cutoffs:
        limit = cutoff if method_rank_limit is None else min(cutoff, method_rank_limit)
        candidates = np.flatnonzero(
            (base_ranks > cutoff) & (method_ranks <= limit)
        ).tolist()
        candidates.sort(
            key=lambda index: (
                -(int(base_ranks[index]) - int(method_ranks[index])),
                int(method_ranks[index]),
                index,
            )
        )
        choice = next((index for index in candidates if index not in used), None)
        if choice is None:
            raise ValueError(f"no unused rescue found for cutoff {cutoff}")
        selected.append((cutoff, choice))
        used.add(choice)
    return selected


def fashiongen_annotations(path):
    return [
        row
        for row in load_jsonl(path)
        if row.get("dataset") == "fashiongen_val"
    ]


def fashioniq_annotations(path):
    return json.loads(Path(path).read_text())


def verify_annotations(rows, annotations, dataset):
    if len(rows) != len(annotations):
        raise ValueError(
            f"{dataset} annotation count differs: {len(rows)} != {len(annotations)}"
        )
    for index, (row, annotation) in enumerate(zip(rows, annotations)):
        source_key = "source_id" if dataset == "fashiongen" else "candidate"
        target_key = "target_id" if dataset == "fashiongen" else "target"
        if (
            row["source_id"] != annotation[source_key]
            or row["target_id"] != annotation[target_key]
        ):
            raise ValueError(f"{dataset} annotation mismatch at query {index}")


def copy_images(dataset, image_root, image_ids, output_root):
    copied = {}
    for image_id in sorted(set(image_ids)):
        destination = output_root / dataset / image_id
        destination.mkdir(parents=True, exist_ok=True)
        if dataset == "fashiongen":
            sources = sorted((image_root / image_id).glob("*.jpg"))
        else:
            sources = [image_root / f"{image_id}.jpg"]
        if not sources or any(not source.is_file() for source in sources):
            raise FileNotFoundError(f"missing source image for {dataset}/{image_id}")
        relative_paths = []
        for source in sources:
            target = destination / source.name
            shutil.copy2(source, target)
            relative_paths.append(str(target.relative_to(output_root.parent)))
        copied[image_id] = relative_paths
    return copied


def build_case(result, annotation, query_index, cutoff, category, top_k):
    row = result["rows"][query_index]
    gallery_ids = result["gallery_ids"]
    if category is None:
        modification = annotation["modification_text_short"]
    else:
        modification = " / ".join(annotation["captions"])
    return {
        "category": category,
        "query_index": query_index,
        "selection_cutoff": cutoff,
        "source_id": row["source_id"],
        "target_id": row["target_id"],
        "modification": modification,
        "base_rank": int(result["base_ranks"][query_index]),
        "method_rank": int(result["method_ranks"][query_index]),
        "base_top_ids": [
            gallery_ids[index] for index in result["base_top"][query_index, :top_k]
        ],
        "method_top_ids": [
            gallery_ids[index]
            for index in result["method_top"][query_index, :top_k]
        ],
    }


def collect_image_ids(cases):
    image_ids = []
    for case in cases:
        image_ids.extend((case["source_id"], case["target_id"]))
        image_ids.extend(case["base_top_ids"])
        image_ids.extend(case["method_top_ids"])
    return image_ids


def export_fashiongen(args):
    result = evaluate_rankings(
        args.embedding_dir, args.device, args.batch_size, False, args.top_k
    )
    annotations = fashiongen_annotations(args.annotations)
    verify_annotations(result["rows"], annotations, "fashiongen")
    selected = select_cases(
        result["base_ranks"], result["method_ranks"], (1, 5, 10)
    )
    cases = [
        build_case(result, annotations[index], index, cutoff, None, args.top_k)
        for cutoff, index in selected
    ]
    return cases, "A1 endpoint", "CrossPath cross mean"


def export_fashioniq(args):
    embedding_dirs = parse_category_paths(args.embedding_dirs, "--embedding-dirs")
    annotation_paths = parse_category_paths(args.annotation_files, "--annotation-files")
    cases = []
    for category in CATEGORIES:
        result = evaluate_rankings(
            embedding_dirs[category], args.device, args.batch_size, True, args.top_k
        )
        annotations = fashioniq_annotations(annotation_paths[category])
        verify_annotations(result["rows"], annotations, "fashioniq")
        selected = select_cases(
            result["base_ranks"],
            result["method_ranks"],
            (10,),
            method_rank_limit=args.top_k,
        )
        cutoff, index = selected[0]
        cases.append(
            build_case(
                result,
                annotations[index],
                index,
                cutoff,
                category,
                args.top_k,
            )
        )
    return cases, "MCoT-MVS", "CrossPath (DQU-GC x MCoT)"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("fashiongen", "fashioniq"))
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-dir", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--embedding-dirs", nargs="*", default=())
    parser.add_argument("--annotation-files", nargs="*", default=())
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.top_k <= 0:
        raise ValueError("batch size and top-k must be positive")
    if args.dataset == "fashiongen":
        if args.embedding_dir is None or args.annotations is None:
            raise ValueError("FashionGen requires --embedding-dir and --annotations")
        cases, base_label, method_label = export_fashiongen(args)
    else:
        cases, base_label, method_label = export_fashioniq(args)

    image_ids = collect_image_ids(cases)
    image_paths = copy_images(
        args.dataset,
        args.image_root,
        image_ids,
        args.output_dir / "source_images",
    )
    report = {
        "dataset": args.dataset,
        "selection": (
            "largest deterministic base-miss/method-hit rescue; ties use method "
            "rank then query index"
        ),
        "base_label": base_label,
        "method_label": method_label,
        "top_k": args.top_k,
        "cases": cases,
        "source_images": image_paths,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.dataset}_retrieval_cases.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    for case in cases:
        print(
            case["category"] or "fashiongen",
            case["query_index"],
            f"rank {case['base_rank']} -> {case['method_rank']}",
        )


if __name__ == "__main__":
    main()
