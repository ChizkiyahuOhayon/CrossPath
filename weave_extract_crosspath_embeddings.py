#!/usr/bin/env python3
"""Export aligned frozen endpoint embeddings for a CrossPath cache.

The script intentionally loads one checkpoint at a time.  It reproduces the
official ProCIR gallery rule, including source products absent from the
ProductValDataset, then sorts product IDs so both endpoints share one stable
gallery order.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _image_paths(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )[:5]


class FashionGenTrainQueryDataset:
    """Short-modification train queries in the original triplet order."""

    def __init__(self, data_dir, image_root, max_queries=None):
        self.samples = []
        image_root = Path(image_root) / "fashiongen_train"
        with (Path(data_dir) / "train_triplets.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["dataset"] != "fashiongen_train":
                    continue
                source_id = str(row["source_id"])
                target_id = str(row["target_id"])
                source_paths = _image_paths(image_root / source_id)
                target_paths = _image_paths(image_root / target_id)
                if not source_paths or not target_paths:
                    continue
                self.samples.append(
                    {
                        "dataset": "fashiongen_train",
                        "source_id": source_id,
                        "target_id": target_id,
                        "source_image_paths": source_paths,
                        "target_image_paths": target_paths,
                        "modification_text_short": row["modification_text_short"],
                    }
                )
                if max_queries is not None and len(self.samples) == max_queries:
                    break
        if not self.samples:
            raise ValueError("no FashionGen train queries survived image filtering")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        return {
            "dataset": row["dataset"],
            "source_id": row["source_id"],
            "target_id": row["target_id"],
            "source_images": [Image.open(path).convert("RGB") for path in row["source_image_paths"]],
            "target_images": [Image.open(path).convert("RGB") for path in row["target_image_paths"]],
            "modification_text_short": row["modification_text_short"],
        }


class FashionGenTrainProductDataset:
    def __init__(self, query_samples):
        products = {}
        for row in query_samples:
            products.setdefault(str(row["source_id"]), row["source_image_paths"])
            products.setdefault(str(row["target_id"]), row["target_image_paths"])
        self.samples = [
            {"product_id": product_id, "image_paths": products[product_id]}
            for product_id in sorted(products)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        return {
            "product_id": row["product_id"],
            "dataset": "fashiongen_train",
            "images": [Image.open(path).convert("RGB") for path in row["image_paths"]],
        }


def build_gallery(product_ids, product_embeddings, metadata, source_embeddings):
    if len(product_ids) != len(product_embeddings):
        raise ValueError("product IDs and embeddings are not aligned")
    if len(metadata) != len(source_embeddings):
        raise ValueError("query metadata and source embeddings are not aligned")

    gallery = {}
    for product_id, embedding in zip(product_ids, product_embeddings):
        gallery.setdefault(str(product_id), np.asarray(embedding, dtype=np.float32))
    for meta, embedding in zip(metadata, source_embeddings):
        gallery.setdefault(
            str(meta["source_id"]), np.asarray(embedding, dtype=np.float32)
        )
    ids = sorted(gallery)
    embeddings = np.stack([gallery[product_id] for product_id in ids], axis=0)
    return ids, embeddings


def _metadata_key(row):
    return str(row["dataset"]), str(row["source_id"]), str(row["target_id"])


def align_endpoint_metadata(base_metadata, correction_metadata):
    base_keys = [_metadata_key(row) for row in base_metadata]
    correction_keys = [_metadata_key(row) for row in correction_metadata]
    if base_keys != correction_keys:
        for index, (base, correction) in enumerate(
            zip(base_keys, correction_keys)
        ):
            if base != correction:
                raise ValueError(
                    f"endpoint query metadata differs at index {index}: "
                    f"{base} != {correction}"
                )
        raise ValueError(
            f"endpoint query counts differ: {len(base_keys)} != {len(correction_keys)}"
        )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_model_file(checkpoint):
    root = Path(checkpoint)
    for name in ("model.safetensors", "pytorch_model.bin"):
        path = root / name
        if path.is_file():
            return path
    candidates = sorted(root.glob("*.safetensors"))
    return candidates[0] if len(candidates) == 1 else None


def encode_endpoint(args, model_path):
    import torch
    from torch.utils.data import DataLoader, SequentialSampler

    repo = str(Path(args.fashionmv_repo).resolve())
    if repo in sys.path:
        sys.path.remove(repo)
    sys.path.insert(0, repo)

    import evaluate as official_evaluate
    from procir.collators import CIRQueryCollator, DocCollator
    from procir.datasets import CIRValDataset, ProductValDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    endpoint = str(Path(model_path).parent)
    model, processor, emb_token_id = official_evaluate.setup_model(model_path, device)

    selected = {args.dataset}
    if args.dataset == "fashiongen_train":
        query_dataset = FashionGenTrainQueryDataset(
            args.data_dir, args.image_root, max_queries=args.max_queries
        )
        product_dataset = FashionGenTrainProductDataset(query_dataset.samples)
    else:
        product_dataset = ProductValDataset(
            args.data_dir, args.image_root, datasets=selected
        )
        query_dataset = CIRValDataset(
            args.data_dir, args.image_root, datasets=selected
        )
        if args.max_queries is not None:
            query_dataset.samples = query_dataset.samples[: args.max_queries]
    product_loader = DataLoader(
        product_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(product_dataset),
        collate_fn=DocCollator(processor, emb_token_id),
        num_workers=args.num_workers,
        pin_memory=False,
    )

    product_ids = []
    product_embeddings = []
    with torch.no_grad():
        for batch_index, batch in enumerate(product_loader, start=1):
            embeddings = model.forward_visual_batch(
                batch["doc_visual_inputs"], device
            )
            for meta, embedding in zip(batch["batch_meta"], embeddings):
                product_ids.append(str(meta["product_id"]))
                product_embeddings.append(embedding.float().cpu().numpy())
            if batch_index % 100 == 0 or batch_index == len(product_loader):
                print(
                    f"[{endpoint}] gallery batches {batch_index}/{len(product_loader)}",
                    flush=True,
                )

    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(query_dataset),
        collate_fn=CIRQueryCollator(processor, emb_token_id),
        num_workers=args.num_workers,
        pin_memory=False,
    )

    metadata = []
    query_embeddings = []
    source_embeddings = []
    with torch.no_grad():
        for batch_index, batch in enumerate(query_loader, start=1):
            sources, queries = model.forward_visual_batch_multiturn(
                batch["query_visual_inputs"], device
            )
            for meta, source, query in zip(batch["batch_meta"], sources, queries):
                metadata.append(dict(meta))
                source_embeddings.append(source.float().cpu().numpy())
                query_embeddings.append(query.float().cpu().numpy())
            if batch_index % 100 == 0 or batch_index == len(query_loader):
                print(
                    f"[{endpoint}] query batches {batch_index}/{len(query_loader)}",
                    flush=True,
                )

    gallery_ids, gallery_embeddings = build_gallery(
        product_ids, product_embeddings, metadata, source_embeddings
    )
    query_array = np.stack(query_embeddings, axis=0)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return gallery_ids, gallery_embeddings, metadata, query_array


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fashionmv-repo", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--correction-model-path", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dataset",
        default="fashiongen_train",
        choices=("fashiongen_train", "fashiongen_val"),
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-queries", type=int)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid loader configuration")
    if args.max_queries is not None and args.max_queries <= 0:
        raise ValueError("--max-queries must be positive")

    base_ids, base_gallery, base_metadata, base_queries = encode_endpoint(
        args, args.base_model_path
    )
    correction_ids, correction_gallery, correction_metadata, correction_queries = (
        encode_endpoint(args, args.correction_model_path)
    )
    align_endpoint_metadata(base_metadata, correction_metadata)
    if base_ids != correction_ids:
        raise ValueError("endpoint gallery product IDs differ")
    if base_gallery.shape != correction_gallery.shape:
        raise ValueError("endpoint gallery embedding shapes differ")
    if base_queries.shape != correction_queries.shape:
        raise ValueError("endpoint query embedding shapes differ")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "base_gallery.npy", base_gallery)
    np.save(output / "base_queries.npy", base_queries)
    np.save(output / "correction_gallery.npy", correction_gallery)
    np.save(output / "correction_queries.npy", correction_queries)
    with (output / "gallery_ids.json").open("w", encoding="utf-8") as handle:
        json.dump(base_ids, handle, ensure_ascii=False)
        handle.write("\n")
    with (output / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for index, meta in enumerate(base_metadata):
            handle.write(
                json.dumps(
                    {
                        "query_index": index,
                        "dataset": str(meta["dataset"]),
                        "source_id": str(meta["source_id"]),
                        "target_id": str(meta["target_id"]),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    model_files = {
        "base": checkpoint_model_file(args.base_model_path),
        "correction": checkpoint_model_file(args.correction_model_path),
    }
    manifest = {
        "dataset": args.dataset,
        "queries": len(base_metadata),
        "gallery": len(base_ids),
        "embedding_dim": int(base_gallery.shape[1]),
        "base_model_path": str(Path(args.base_model_path).resolve()),
        "correction_model_path": str(Path(args.correction_model_path).resolve()),
        "model_file_sha256": {
            name: sha256_file(path) if path is not None else None
            for name, path in model_files.items()
        },
        "script_sha256": sha256_file(__file__),
        "max_queries": args.max_queries,
        "batch_size": args.batch_size,
    }
    with (output / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
