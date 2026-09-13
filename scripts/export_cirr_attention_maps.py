#!/usr/bin/env python3
"""Export real MCoT retained/deleted CIRR attention maps for selected cases."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_map(array):
    array = np.asarray(array, dtype=np.float32)
    low, high = float(array.min()), float(array.max())
    if high <= low:
        return np.zeros_like(array)
    return (array - low) / (high - low)


def save_overlay(image, attention, path, alpha=0.45):
    import matplotlib

    heat = Image.fromarray(np.uint8(normalize_map(attention) * 255), mode="L")
    heat = heat.resize(image.size, Image.Resampling.BILINEAR)
    colors = matplotlib.colormaps["turbo"](np.asarray(heat, dtype=np.float32) / 255.0)
    colored = Image.fromarray(np.uint8(colors[:, :, :3] * 255), mode="RGB")
    Image.blend(image.convert("RGB"), colored, alpha).save(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-src", required=True, type=Path)
    parser.add_argument("--cirr-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = json.loads(args.cases.read_text(encoding="utf-8"))
    split = json.loads(
        (args.cirr_root / "cirr" / "image_splits" / "split.rc2.val.json").read_text()
    )
    repo_src = str(args.repo_src.resolve())
    if repo_src in sys.path:
        sys.path.remove(repo_src)
    sys.path.insert(0, repo_src)

    import torch
    from data_utils import CIRRDataset
    from model import CIRModel

    dataset = CIRRDataset(str(args.cirr_root.resolve()), "val", "relative")
    model = CIRModel()
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu")["CIRModel"], strict=True
    )
    model.cuda().float().eval()
    outputs = []
    for case in report["cases"]:
        index = int(case["query_index"])
        item = dataset[index]
        if item["reference_name"] != case["source_id"]:
            raise ValueError(f"case/dataset mismatch at query {index}")
        with torch.no_grad():
            image_tensor = model.preprocess_image(item["reference_image"])
            _, patch_features = model.get_visual(image_tensor)
            retained = model.get_attention_map(
                patch_features, [item["llm_info"]["retained"]]
            )[0].float().cpu().numpy()
            deleted = model.get_attention_map(
                patch_features, [item["llm_info"]["deleted"]]
            )[0].float().cpu().numpy()
        side = math.isqrt(len(retained))
        if side * side != len(retained) or retained.shape != deleted.shape:
            raise ValueError(f"unexpected patch attention shape: {retained.shape}")
        retained = retained.reshape(side, side)
        deleted = deleted.reshape(side, side)
        selected = 1.0 - (retained - deleted)
        original_path = args.cirr_root / split[case["source_id"]]
        with Image.open(original_path) as source:
            original = source.convert("RGB")
        case_dir = args.output_dir / f"query_{index:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        save_overlay(original, retained, case_dir / "retained.png")
        save_overlay(original, deleted, case_dir / "deleted.png")
        save_overlay(original, selected, case_dir / "selected_reference.png")
        np.savez_compressed(
            case_dir / "attention_raw.npz",
            retained=retained,
            deleted=deleted,
            selected_reference=selected,
        )
        metadata = {
            "query_index": index,
            "source_id": case["source_id"],
            "caption": case["caption"],
            "retained_text": item["llm_info"]["retained"],
            "deleted_text": item["llm_info"]["deleted"],
            "target_text": item["llm_info"]["target"],
            "patch_grid": [side, side],
            "original_size": list(original.size),
        }
        (case_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        outputs.append(str(case_dir.relative_to(args.output_dir)))
    manifest = {"checkpoint": str(args.checkpoint.resolve()), "cases": outputs}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
