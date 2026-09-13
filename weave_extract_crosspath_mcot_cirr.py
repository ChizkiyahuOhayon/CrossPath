#!/usr/bin/env python3
"""Export MCoT-MVS CIRR embeddings aligned to a DQU-CIR endpoint."""

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-src", required=True, type=Path)
    parser.add_argument("--cirr-path", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--alignment-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test1"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
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
    metadata = load_jsonl(args.alignment_dir / "queries.jsonl")
    repo_src = str(args.repo_src.resolve())
    if repo_src in sys.path:
        sys.path.remove(repo_src)
    sys.path.insert(0, repo_src)

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    from data_utils import CIRRDataset, collate_fn_test, collate_fn_val
    from model import CIRModel

    relative = CIRRDataset(str(args.cirr_path.resolve()), args.split, "relative")
    classic = CIRRDataset(str(args.cirr_path.resolve()), args.split, "classic")
    if list(classic.name_to_relpath.keys()) != gallery_ids:
        raise ValueError("MCoT gallery order differs from DQU alignment")
    if len(relative) != len(metadata):
        raise ValueError("MCoT query count differs from DQU alignment")

    model = CIRModel()
    state = torch.load(args.checkpoint, map_location="cpu")["CIRModel"]
    model.load_state_dict(state, strict=True)
    del state
    model.cuda().float().eval()

    collate = collate_fn_val if args.split == "val" else collate_fn_test
    loader = DataLoader(
        relative,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate,
        shuffle=False,
    )
    query_batches = []
    query_count = 0
    query_started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            size = len(batch["reference_name"])
            expected = metadata[query_count : query_count + size]
            if batch["reference_name"] != [row["source_id"] for row in expected]:
                raise ValueError(f"query source alignment mismatch at {query_count}")
            if args.split == "val":
                if batch["target_hard_name"] != [row["target_id"] for row in expected]:
                    raise ValueError(f"query target alignment mismatch at {query_count}")
            elif [int(value) for value in batch["pair_id"]] != [int(row["pair_id"]) for row in expected]:
                raise ValueError(f"query pair alignment mismatch at {query_count}")
            features = model.extract_query(
                batch["rel_caption"],
                batch["reference_image"],
                [feature.cuda() for feature in batch["reference_seg_feature_list"]],
                batch["llm_info"],
            )
            query_batches.append(features.float().cpu().numpy())
            query_count += size
    query_seconds = time.perf_counter() - query_started

    gallery_batches = []
    gallery_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(gallery_ids), args.batch_size):
            images = []
            for image_id in gallery_ids[start : start + args.batch_size]:
                image_path = args.cirr_path / classic.name_to_relpath[image_id]
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB").copy())
            gallery_batches.append(model.extract_target(images).float().cpu().numpy())
    gallery_seconds = time.perf_counter() - gallery_started

    queries = np.concatenate(query_batches).astype(np.float32)
    gallery = np.concatenate(gallery_batches).astype(np.float32)
    np.save(args.output_dir / "queries.npy", queries)
    np.save(args.output_dir / "gallery.npy", gallery)
    shutil.copyfile(args.alignment_dir / "gallery_ids.json", args.output_dir / "gallery_ids.json")
    shutil.copyfile(args.alignment_dir / "queries.jsonl", args.output_dir / "queries.jsonl")
    manifest = {
        "dataset": f"cirr_{args.split}",
        "endpoint": "MCoT-MVS",
        "queries": len(metadata),
        "gallery": len(gallery_ids),
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
