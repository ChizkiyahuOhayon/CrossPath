#!/usr/bin/env python3
"""Export DQU-CIR text, visual-query, and gallery branch embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from weave_extract_crosspath_dqu import build_specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-src", type=Path, required=True)
    parser.add_argument("--fashioniq-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--category", choices=("dress", "shirt", "toptee"), required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    import torch.nn.functional as F
    import open_clip

    sys.path.insert(0, str(args.repo_src.resolve()))
    import datasets as dqu_datasets
    import model as dqu_model

    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2B-s32B-b79K"
    )
    original_get_test_data = dqu_datasets.FashionIQ.get_test_data
    if args.split == "train":
        dqu_datasets.FashionIQ.get_test_data = lambda self: ([], [])
    try:
        dataset = dqu_datasets.FashionIQ(
            path=str(args.fashioniq_path.resolve()) + "/",
            category=args.category,
            transform=[preprocess_train, preprocess_val],
            split="val-split",
        )
    finally:
        dqu_datasets.FashionIQ.get_test_data = original_get_test_data
    specs, gallery_ids = build_specs(
        dataset, args.fashioniq_path, args.category, args.split
    )

    endpoint = dqu_model.DQU_CIR(1024, 0.5).cuda()
    state = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    endpoint.load_state_dict(state.get("state_dict", state))
    endpoint.eval()

    gallery = []
    with torch.no_grad():
        for start in range(0, len(gallery_ids), args.batch_size):
            ids = gallery_ids[start : start + args.batch_size]
            images = [dataset.get_img(f"{args.category}_{image_id}", stage=1)[0] for image_id in ids]
            gallery.append(endpoint.extract_target(torch.stack(images).float().cuda()).float().cpu())

    text_features, visual_features, original_lambdas = [], [], []
    with torch.no_grad():
        for start in range(0, len(specs), args.batch_size):
            rows = specs[start : start + args.batch_size]
            texts, visuals = [], []
            for row in rows:
                texts.append(row["text"])
                if args.split == "train":
                    visual = dataset.get_written_img(row["candidate"], row["target"], stage=1)[0]
                else:
                    visual = row["visual"]
                visuals.append(visual)
            text = F.normalize(endpoint.extract_text_fea(texts), dim=-1)
            visual = F.normalize(endpoint.extract_img_fea(torch.stack(visuals).float().cuda()), dim=-1)
            combined = endpoint.combiner_fc(torch.cat((text, visual), dim=-1))
            mixing = endpoint.scaler_fc(endpoint.dropout(combined))
            text_features.append(text.float().cpu())
            visual_features.append(visual.float().cpu())
            original_lambdas.append(mixing.float().cpu())

    arrays = {
        "gallery": torch.cat(gallery).numpy(),
        "text": torch.cat(text_features).numpy(),
        "visual": torch.cat(visual_features).numpy(),
        "original_lambda": torch.cat(original_lambdas).numpy(),
    }
    for name, values in arrays.items():
        np.save(args.output_dir / f"{name}.npy", values.astype(np.float32))
    (args.output_dir / "gallery_ids.json").write_text(json.dumps(gallery_ids) + "\n")
    with (args.output_dir / "queries.jsonl").open("w") as handle:
        for index, row in enumerate(specs):
            handle.write(
                json.dumps(
                    {
                        "query_index": index,
                        "source_id": row["source_id"],
                        "target_id": row["target_id"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    manifest = {
        "dataset": f"fashioniq_{args.split}_{args.category}",
        "category": args.category,
        "split": args.split,
        "queries": len(specs),
        "gallery": len(gallery_ids),
        "embedding_dim": int(arrays["gallery"].shape[1]),
        "checkpoint": str(args.checkpoint.resolve()),
        "batch_size": args.batch_size,
        "deterministic_eval_transform": True,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
