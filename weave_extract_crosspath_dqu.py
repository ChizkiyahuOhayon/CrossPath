#!/usr/bin/env python3
"""Export aligned DQU-CIR endpoint embeddings for CrossPath on FashionIQ."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


_DATASET_CACHE = {}
_ENCODING_TIMES = {}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_names(fashioniq_path, category, split):
    path = Path(fashioniq_path) / "image_splits" / f"split.{category}.{split}.json"
    return json.loads(path.read_text())


def select_gallery_ids(specs, fashioniq_path, category, split, gallery_protocol):
    if split == "val" and gallery_protocol == "original-split":
        return sorted(load_split_names(fashioniq_path, category, "val"))
    return sorted(
        {spec["source_id"] for spec in specs}
        | {spec["target_id"] for spec in specs}
    )


def build_specs(
    dataset,
    fashioniq_path,
    category,
    split,
    max_queries=None,
    gallery_protocol="val-split",
):
    if split == "train":
        rows = dataset.train_data[:max_queries]
        specs = [
            {
                "source_id": row["candidate"].split("_", 1)[1],
                "target_id": row["target"].split("_", 1)[1],
                "text": dataset.train_captions[row["candidate"].split("_", 1)[1]]
                + ", but "
                + row["captions"],
                "candidate": row["candidate"],
                "target": row["target"],
            }
            for row in rows
        ]
    else:
        names = load_split_names(fashioniq_path, category, "val")
        queries = dataset.test_queries[:max_queries]
        specs = [
            {
                "source_id": names[int(row["source_img_id"])],
                "target_id": names[int(row["target_img_id"])],
                "text": row["textual_query"],
                "visual": row["visual_query"],
            }
            for row in queries
        ]
    if not specs:
        raise ValueError("no FashionIQ queries were selected")
    gallery_ids = select_gallery_ids(
        specs, fashioniq_path, category, split, gallery_protocol
    )
    return specs, gallery_ids


def encode_endpoint(args, checkpoint):
    import torch
    import torch.nn.functional as functional
    import open_clip

    repo_src = str(Path(args.repo_src).resolve())
    if repo_src in sys.path:
        sys.path.remove(repo_src)
    sys.path.insert(0, repo_src)
    import datasets as dqu_datasets
    import model as dqu_model

    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2B-s32B-b79K"
    )
    cache_key = (args.category, args.split, args.max_queries, args.gallery_protocol)
    if cache_key not in _DATASET_CACHE:
        original_get_test_data = dqu_datasets.FashionIQ.get_test_data
        if args.split == "train":
            dqu_datasets.FashionIQ.get_test_data = lambda self: ([], [])
        try:
            dataset = dqu_datasets.FashionIQ(
                path=str(Path(args.fashioniq_path).resolve()) + "/",
                category=args.category,
                transform=[preprocess_train, preprocess_val],
                # The exporter constructs the requested gallery explicitly below.
                # Keeping the lightweight split avoids eager preprocessing of the
                # complete original gallery inside DQU's dataset constructor.
                split="val-split",
            )
        finally:
            dqu_datasets.FashionIQ.get_test_data = original_get_test_data
        specs, gallery_ids = build_specs(
            dataset,
            args.fashioniq_path,
            args.category,
            args.split,
            args.max_queries,
            args.gallery_protocol,
        )
        _DATASET_CACHE[cache_key] = dataset, specs, gallery_ids
    dataset, specs, gallery_ids = _DATASET_CACHE[cache_key]

    endpoint = dqu_model.DQU_CIR(1024, 0.5).cuda()
    state = torch.load(checkpoint, map_location="cuda", weights_only=True)
    endpoint.load_state_dict(state.get("state_dict", state))
    del state
    torch.cuda.empty_cache()
    endpoint.eval()

    gallery_embeddings = []
    gallery_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(gallery_ids), args.batch_size):
            ids = gallery_ids[start : start + args.batch_size]
            images = [
                dataset.get_img(f"{args.category}_{image_id}", stage=1)[0]
                for image_id in ids
            ]
            batch = torch.stack(images).float().cuda()
            gallery_embeddings.append(endpoint.extract_target(batch).float().cpu())
    gallery = torch.cat(gallery_embeddings).numpy()
    gallery_seconds = time.perf_counter() - gallery_started

    query_embeddings = []
    query_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(specs), args.batch_size):
            batch_specs = specs[start : start + args.batch_size]
            visuals = []
            texts = []
            for spec in batch_specs:
                if args.split == "train":
                    visual = dataset.get_written_img(
                        spec["candidate"], spec["target"], stage=1
                    )[0]
                else:
                    visual = spec["visual"]
                visuals.append(visual)
                texts.append(spec["text"])
            batch = torch.stack(visuals).float().cuda()
            query_embeddings.append(
                endpoint.extract_query(texts, batch).float().cpu()
            )
    queries = torch.cat(query_embeddings).numpy()
    queries = functional.normalize(torch.from_numpy(queries), dim=-1).numpy()
    query_seconds = time.perf_counter() - query_started
    _ENCODING_TIMES[str(Path(checkpoint).resolve())] = {
        "gallery_seconds": gallery_seconds,
        "query_seconds": query_seconds,
        "gallery_items": len(gallery_ids),
        "query_items": len(specs),
        "query_ms_per_item": 1000.0 * query_seconds / len(specs),
    }

    del endpoint
    torch.cuda.empty_cache()
    metadata = [
        {
            "dataset": f"fashioniq_{args.split}_{args.category}",
            "source_id": spec["source_id"],
            "target_id": spec["target_id"],
        }
        for spec in specs
    ]
    return gallery_ids, gallery, metadata, queries


def validate_alignment(base, correction):
    base_ids, base_gallery, base_metadata, base_queries = base
    correction_ids, correction_gallery, correction_metadata, correction_queries = correction
    if base_ids != correction_ids:
        raise ValueError("endpoint gallery IDs differ")
    if base_metadata != correction_metadata:
        raise ValueError("endpoint query metadata differs")
    if base_gallery.shape != correction_gallery.shape:
        raise ValueError("endpoint gallery shapes differ")
    if base_queries.shape != correction_queries.shape:
        raise ValueError("endpoint query shapes differ")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-src", required=True)
    parser.add_argument("--fashioniq-path", required=True)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--correction-checkpoint", required=True, type=Path)
    parser.add_argument("--category", required=True, choices=("dress", "shirt", "toptee"))
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument(
        "--gallery-protocol",
        choices=("val-split", "original-split"),
        default="val-split",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-queries", type=int)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.batch_size <= 0 or (args.max_queries is not None and args.max_queries <= 0):
        raise ValueError("invalid extraction size")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    base = encode_endpoint(args, args.base_checkpoint)
    base_seconds = time.perf_counter() - started
    started = time.perf_counter()
    correction = encode_endpoint(args, args.correction_checkpoint)
    correction_seconds = time.perf_counter() - started
    validate_alignment(base, correction)
    gallery_ids, base_gallery, metadata, base_queries = base
    _, correction_gallery, _, correction_queries = correction

    for name, array in (
        ("base_gallery", base_gallery),
        ("base_queries", base_queries),
        ("correction_gallery", correction_gallery),
        ("correction_queries", correction_queries),
    ):
        np.save(args.output_dir / f"{name}.npy", array.astype(np.float32))
    (args.output_dir / "gallery_ids.json").write_text(
        json.dumps(gallery_ids) + "\n"
    )
    with (args.output_dir / "queries.jsonl").open("w") as handle:
        for index, row in enumerate(metadata):
            handle.write(json.dumps({"query_index": index, **row}, sort_keys=True) + "\n")

    manifest = {
        "dataset": f"fashioniq_{args.split}_{args.category}",
        "category": args.category,
        "split": args.split,
        "gallery_protocol": args.gallery_protocol,
        "queries": len(metadata),
        "gallery": len(gallery_ids),
        "embedding_dim": int(base_gallery.shape[1]),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "correction_checkpoint": str(args.correction_checkpoint.resolve()),
        "checkpoint_sha256": {
            "base": sha256_file(args.base_checkpoint),
            "correction": sha256_file(args.correction_checkpoint),
        },
        "batch_size": args.batch_size,
        "max_queries": args.max_queries,
        "deterministic_eval_transform": True,
        "endpoint_encoding_seconds": {
            "base": base_seconds,
            "correction": correction_seconds,
        },
        "encoder_breakdown": {
            "base": _ENCODING_TIMES[str(args.base_checkpoint.resolve())],
            "correction": _ENCODING_TIMES[str(args.correction_checkpoint.resolve())],
        },
        "script_sha256": sha256_file(__file__),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
