#!/usr/bin/env python3
"""Export DQU-CIR endpoint embeddings and aligned CIRR metadata."""

import argparse
import hashlib
import json
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


def official_metadata(cirr_path, split):
    caption_path = cirr_path / "captions" / "captions" / f"cap.rc2.{split}.json"
    split_path = cirr_path / "captions" / "image_splits" / f"split.rc2.{split}.json"
    captions = json.loads(caption_path.read_text(encoding="utf-8"))
    gallery_ids = list(json.loads(split_path.read_text(encoding="utf-8")).keys())
    rows = []
    for index, caption in enumerate(captions):
        row = {
            "query_index": index,
            "dataset": f"cirr_{split}",
            "source_id": caption["reference"],
            "group_members": caption["img_set"]["members"],
            "caption": caption["caption"],
        }
        if split in ("train", "val"):
            row["target_id"] = caption["target_hard"]
        else:
            row["pair_id"] = caption["pairid"]
        rows.append(row)
    return gallery_ids, rows


def train_dataset(dataset_type, cirr_path, transforms):
    """Initialize only DQU's train fields, avoiding eager val/test image loading."""
    dataset = dataset_type.__new__(dataset_type)
    dataset.path = str(cirr_path.resolve()) + "/"
    dataset.caption_dir = dataset.path + "captions/captions/"
    dataset.split_dir = dataset.path + "captions/image_splits/"
    dataset.transform = transforms
    dataset.cirr_data = json.loads(
        (cirr_path / "captions" / "captions" / "cap.rc2.train.json").read_text()
    )
    dataset.train_image_path = json.loads(
        (cirr_path / "captions" / "image_splits" / "split.rc2.train.json").read_text()
    )
    dataset.train_image_name = list(dataset.train_image_path)
    dataset.train_captions = json.loads(
        (cirr_path / "image_captions_cirr_train.json").read_text()
    )
    dataset.key_words_train = json.loads(
        (cirr_path / "keywords_in_mods_cirr_train.json").read_text()
    )
    return dataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-src", required=True, type=Path)
    parser.add_argument("--cirr-path", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val", "test1"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    repo_src = str(args.repo_src.resolve())
    if repo_src in sys.path:
        sys.path.remove(repo_src)
    sys.path.insert(0, repo_src)

    import open_clip
    import torch
    import datasets as dqu_datasets

    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2B-s32B-b79K"
    )
    transforms = [preprocess_train, preprocess_val]
    if args.split == "train":
        dataset = train_dataset(dqu_datasets.CIRR, args.cirr_path, transforms)
    else:
        dataset = dqu_datasets.CIRR(
            path=str(args.cirr_path.resolve()) + "/",
            transform=transforms,
        )
    gallery_ids, metadata = official_metadata(args.cirr_path.resolve(), args.split)
    if args.split == "train":
        query_items = dataset.cirr_data
        gallery_items = gallery_ids
    elif args.split == "val":
        query_items = dataset.val_queries
        gallery_items = dataset.val_targets
    else:
        query_items = dataset.test_queries
        gallery_items = dataset.test_img_data
        if list(dataset.test_name_list) != gallery_ids:
            raise ValueError("DQU test gallery order differs from official split")
    if len(query_items) != len(metadata) or len(gallery_items) != len(gallery_ids):
        raise ValueError("DQU dataset length differs from official CIRR metadata")

    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.cuda().eval()

    gallery_batches = []
    gallery_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(gallery_items), args.batch_size):
            items = gallery_items[start : start + args.batch_size]
            if args.split == "train":
                images = [
                    dataset.get_img(dataset.train_image_path[name], 1) for name in items
                ]
            elif args.split == "val":
                images = [item["target_img_data"] for item in items]
            else:
                images = items
            batch = torch.stack(images).float().cuda()
            gallery_batches.append(model.extract_target(batch).float().cpu().numpy())
    gallery_seconds = time.perf_counter() - gallery_started

    query_batches = []
    query_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(query_items), args.batch_size):
            items = query_items[start : start + args.batch_size]
            if args.split == "train":
                images, texts = [], []
                for item in items:
                    reference = item["reference"]
                    target = item["target_hard"]
                    keyword = dataset.key_words_train[reference + "+" + target][-1]
                    image_path = args.cirr_path / "images" / dataset.train_image_path[reference]
                    images.append(dataset.get_written_img(str(image_path), keyword, 1))
                    texts.append(dataset.train_captions[reference] + ", but" + item["caption"])
                images = torch.stack(images).float().cuda()
            else:
                images = torch.stack([item["visual_query"] for item in items]).float().cuda()
                texts = [item["textual_query"] for item in items]
            query_batches.append(model.extract_query(texts, images).float().cpu().numpy())
    query_seconds = time.perf_counter() - query_started

    gallery = np.concatenate(gallery_batches).astype(np.float32)
    queries = np.concatenate(query_batches).astype(np.float32)
    np.save(args.output_dir / "gallery.npy", gallery)
    np.save(args.output_dir / "queries.npy", queries)
    (args.output_dir / "gallery_ids.json").write_text(
        json.dumps(gallery_ids) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "dataset": f"cirr_{args.split}",
        "endpoint": "DQU-CIR",
        "queries": len(metadata),
        "gallery": len(gallery_ids),
        "embedding_dim": int(gallery.shape[1]),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "batch_size": args.batch_size,
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
