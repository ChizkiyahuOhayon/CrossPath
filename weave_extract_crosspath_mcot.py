#!/usr/bin/env python3
"""Export MCoT-MVS FashionIQ embeddings aligned to a CrossPath endpoint."""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_query_alignment(expected, references, targets, offset=0):
    observed = list(zip(references, targets))
    wanted = [
        (row["source_id"], row["target_id"])
        for row in expected[offset : offset + len(observed)]
    ]
    if observed != wanted:
        for index, (actual, target) in enumerate(zip(observed, wanted), start=offset):
            if actual != target:
                raise ValueError(
                    f"query alignment mismatch at {index}: observed={actual}, expected={target}"
                )
        raise ValueError("query alignment length mismatch")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-src", required=True, type=Path)
    parser.add_argument("--fashioniq-path", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--alignment-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--category", required=True, choices=("dress", "shirt", "toptee")
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("invalid DataLoader settings")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gallery_ids = json.loads(
        (args.alignment_dir / "gallery_ids.json").read_text(encoding="utf-8")
    )
    expected_queries = load_jsonl(args.alignment_dir / "queries.jsonl")

    repo_src = str(args.repo_src.resolve())
    if repo_src in sys.path:
        sys.path.remove(repo_src)
    sys.path.insert(0, repo_src)

    import torch
    from torch.utils.data import DataLoader

    from data_utils import FashionIQDataset, collate_fn_val_fiq
    from model import CIRModel

    dataset = FashionIQDataset(
        str(args.fashioniq_path.resolve()), "val", [args.category], "relative"
    )
    if len(dataset) != len(expected_queries):
        raise ValueError(
            f"query count mismatch: MCoT={len(dataset)}, alignment={len(expected_queries)}"
        )

    model = CIRModel()
    state = torch.load(args.checkpoint, map_location="cpu")["CIRModel"]
    model.load_state_dict(state, strict=True)
    del state
    model.cuda().float().eval()

    query_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn_val_fiq,
        shuffle=False,
    )
    query_batches = []
    query_count = 0
    query_started = time.perf_counter()
    with torch.no_grad():
        for batch in query_loader:
            references = batch["reference_name"]
            targets = batch["target_hard_name"]
            validate_query_alignment(
                expected_queries, references, targets, offset=query_count
            )
            features = model.extract_query(
                batch["rel_caption"],
                batch["reference_image"],
                [feature.cuda() for feature in batch["reference_seg_feature_list"]],
                batch["llm_info"],
            )
            query_batches.append(features.float().cpu().numpy())
            query_count += len(references)
    query_seconds = time.perf_counter() - query_started
    queries = np.concatenate(query_batches).astype(np.float32)

    gallery_batches = []
    gallery_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(gallery_ids), args.batch_size):
            images = [
                dataset.get_image_by_name(image_id)
                for image_id in gallery_ids[start : start + args.batch_size]
            ]
            gallery_batches.append(model.extract_target(images).float().cpu().numpy())
    gallery_seconds = time.perf_counter() - gallery_started
    gallery = np.concatenate(gallery_batches).astype(np.float32)

    np.save(args.output_dir / "gallery.npy", gallery)
    np.save(args.output_dir / "queries.npy", queries)
    shutil.copyfile(
        args.alignment_dir / "gallery_ids.json", args.output_dir / "gallery_ids.json"
    )
    shutil.copyfile(
        args.alignment_dir / "queries.jsonl", args.output_dir / "queries.jsonl"
    )
    manifest = {
        "dataset": f"fashioniq_val_{args.category}",
        "endpoint": "MCoT-MVS",
        "queries": int(queries.shape[0]),
        "gallery": int(gallery.shape[0]),
        "embedding_dim": int(gallery.shape[1]),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "alignment_dir": str(args.alignment_dir.resolve()),
        "strict_state_dict": True,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "query_seconds": query_seconds,
        "gallery_seconds": gallery_seconds,
        "script_sha256": sha256_file(__file__),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
