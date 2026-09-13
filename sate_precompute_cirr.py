#!/usr/bin/env python3
"""Precompute frozen MCoT-MVS features for SATE head training on CIRR train.

Everything except the SATE head is frozen, so the query embeddings and the
target CLS embeddings never change during head training. Dumping them once
turns each head-training epoch from a full ViT-H forward pass into a few
minutes of MLP work, which is what makes iterating on the head design cheap.

Outputs (under --output-dir):
  train_queries.npy      (Nq, 1024) frozen query embeddings, triplet order
  train_target_cls.npy   (Ng, 1024) frozen CLIP CLS for every train gallery image
  train_gallery_ids.json gallery image names, index-aligned to train_target_cls
  train_targets.json     per-triplet target index into the gallery
  manifest.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-src", required=True, type=Path)
    parser.add_argument("--cirr-path", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0,
                        help="debug: only process this many triplets")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.repo_src.resolve()))
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    from data_utils import CIRRDataset, collate_fn_train
    from model import CIRModel

    model = CIRModel()
    state = torch.load(args.checkpoint, map_location="cpu")["CIRModel"]
    model.load_state_dict(state, strict=True)
    del state
    model.cuda().float().eval()

    train_set = CIRRDataset(str(args.cirr_path.resolve()), "train", "relative")
    triplets = train_set.triplets
    if args.limit:
        triplets = triplets[: args.limit]

    # gallery: every image in the official train split, stable order
    split_path = args.cirr_path / "cirr" / "image_splits" / "split.rc2.train.json"
    gallery_ids = list(json.loads(split_path.read_text(encoding="utf-8")).keys())
    gallery_index = {name: i for i, name in enumerate(gallery_ids)}
    targets = [gallery_index[t["target_hard"]] for t in triplets]

    loader = DataLoader(
        train_set, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, collate_fn=collate_fn_train, shuffle=False,
    )

    query_batches, seen = [], 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            size = len(batch["rel_caption"])
            if args.limit and seen >= args.limit:
                break
            features = model.extract_query(
                batch["rel_caption"],
                batch["reference_image"],
                [f.cuda() for f in batch["reference_seg_feature_list"]],
                batch["llm_info"],
            )
            query_batches.append(features.float().cpu().numpy())
            seen += size
            if seen % (args.batch_size * 100) == 0:
                rate = seen / (time.perf_counter() - started)
                print(f"queries {seen}/{len(triplets)}  {rate:.1f}/s", flush=True)
    query_seconds = time.perf_counter() - started

    cls_batches = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(gallery_ids), args.batch_size):
            images = []
            for name in gallery_ids[start : start + args.batch_size]:
                path = args.cirr_path / train_set.name_to_relpath[name]
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
            cls_batches.append(model.encode_image(images).float().cpu().numpy())
            if start % (args.batch_size * 100) == 0:
                print(f"gallery {start}/{len(gallery_ids)}", flush=True)
    gallery_seconds = time.perf_counter() - started

    queries = np.concatenate(query_batches).astype(np.float32)[: len(triplets)]
    gallery_cls = np.concatenate(cls_batches).astype(np.float32)
    np.save(args.output_dir / "train_queries.npy", queries)
    np.save(args.output_dir / "train_target_cls.npy", gallery_cls)
    (args.output_dir / "train_gallery_ids.json").write_text(
        json.dumps(gallery_ids), encoding="utf-8")
    (args.output_dir / "train_targets.json").write_text(
        json.dumps(targets[: len(queries)]), encoding="utf-8")
    manifest = {
        "triplets": int(len(queries)),
        "gallery": int(len(gallery_ids)),
        "embedding_dim": int(gallery_cls.shape[1]),
        "checkpoint": str(args.checkpoint.resolve()),
        "query_seconds": query_seconds,
        "gallery_seconds": gallery_seconds,
        "note": "queries are L2-normalised by extract_query; CLS is raw (head normalises)",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
