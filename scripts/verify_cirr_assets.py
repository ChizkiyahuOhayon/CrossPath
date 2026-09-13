#!/usr/bin/env python3
"""Verify CIRR images, annotations, DQU view, and MCoT segment features."""

import argparse
import json
from pathlib import Path


SPLITS = ("train", "val", "test1")


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def split_assets(cirr_root, split):
    captions = load_json(cirr_root / "cirr" / "captions" / f"cap.rc2.{split}.json")
    images = load_json(cirr_root / "cirr" / "image_splits" / f"split.rc2.{split}.json")
    return captions, images


def verify_split(cirr_root, segment_root, split):
    captions, images = split_assets(cirr_root, split)
    image_ids = set(images)
    missing_images = [name for name, relpath in images.items() if not (cirr_root / relpath).is_file()]
    missing_segments = [name for name in images if not (segment_root / name / "seg_feature.pt").is_file()]
    invalid_queries = []
    for index, row in enumerate(captions):
        required = {row["reference"], *row["img_set"]["members"]}
        if split != "test1":
            required.add(row["target_hard"])
        if not required <= image_ids:
            invalid_queries.append(index)
    if missing_images:
        raise ValueError(f"{split}: {len(missing_images)} image files are missing")
    if missing_segments:
        raise ValueError(f"{split}: {len(missing_segments)} segment features are missing")
    if invalid_queries:
        raise ValueError(f"{split}: {len(invalid_queries)} queries reference unknown images")
    return {
        "images": len(images),
        "queries": len(captions),
        "all_images_present": True,
        "all_segments_present": True,
        "all_query_ids_valid": True,
    }, image_ids


def verify_dqu_view(dqu_root):
    required_files = [
        dqu_root / f"image_captions_cirr_{split}.json" for split in SPLITS
    ] + [dqu_root / f"keywords_in_mods_cirr_{split}.json" for split in SPLITS]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise ValueError(f"DQU metadata files are missing: {missing}")
    for split in SPLITS:
        caption = dqu_root / "captions" / "captions" / f"cap.rc2.{split}.json"
        image_split = dqu_root / "captions" / "image_splits" / f"split.rc2.{split}.json"
        if not caption.is_file() or not image_split.is_file():
            raise ValueError(f"DQU annotation view is incomplete for {split}")
    return {"metadata_files": len(required_files), "annotation_links_valid": True}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cirr-root", required=True, type=Path)
    parser.add_argument("--dqu-root", required=True, type=Path)
    parser.add_argument("--segment-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    split_reports = {}
    union = set()
    for split in SPLITS:
        report, image_ids = verify_split(args.cirr_root, args.segment_root, split)
        split_reports[split] = report
        union.update(image_ids)
    segment_ids = {
        path.parent.name for path in args.segment_root.glob("*/seg_feature.pt")
    }
    if segment_ids != union:
        raise ValueError(
            f"segment/image ID sets differ: segments={len(segment_ids)}, images={len(union)}"
        )
    report = {
        "status": "pass",
        "splits": split_reports,
        "unique_images": len(union),
        "segment_features": len(segment_ids),
        "dqu": verify_dqu_view(args.dqu_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
