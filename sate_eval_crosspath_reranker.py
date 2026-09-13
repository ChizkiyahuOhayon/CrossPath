#!/usr/bin/env python3
"""Evaluate the frozen E29 re-ranker on weak CrossPath CIRR scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from sate_reranker import SegmentReranker
from sate_train_reranker import load_jsonl, load_seg_features, rerank_scores


@torch.no_grad()
def evaluate(model, val_dir, dqu_gallery_path, seg_dir, alpha, device, topk, chunk, max_segments):
    val_dir = Path(val_dir)
    queries = functional.normalize(torch.from_numpy(np.load(val_dir / "queries.npy")).float(), dim=-1)
    mcot_gallery = functional.normalize(torch.from_numpy(np.load(val_dir / "gallery.npy")).float(), dim=-1)
    dqu_gallery = functional.normalize(torch.from_numpy(np.load(dqu_gallery_path)).float(), dim=-1)
    gallery_ids = json.loads((val_dir / "gallery_ids.json").read_text())
    rows = load_jsonl(val_dir / "queries.jsonl")
    index = {name: position for position, name in enumerate(gallery_ids)}
    sources = np.asarray([index[row["source_id"]] for row in rows])
    targets = np.asarray([index[row["target_id"]] for row in rows])
    groups = [np.asarray([index[name] for name in row["group_members"]]) for row in rows]
    seg_features = load_seg_features(seg_dir, gallery_ids)

    scores = ((1.0 - alpha) * (queries @ mcot_gallery.T) + alpha * (queries @ dqu_gallery.T)).numpy()
    scores[np.arange(len(scores)), sources] = -np.inf
    order = np.argsort(-scores, axis=1, kind="stable")
    top_indices = order[:, :topk]
    top_base = np.take_along_axis(scores, top_indices, axis=1)
    new_scores = rerank_scores(
        model, queries, top_indices, seg_features, top_base, device, chunk, max_segments
    ).numpy()
    reordered = np.take_along_axis(
        top_indices, np.argsort(-new_scores, axis=1, kind="stable"), axis=1
    )

    ranks = np.empty(len(rows), dtype=np.int64)
    for row, target in enumerate(targets):
        hit = np.flatnonzero(reordered[row] == target)
        ranks[row] = hit[0] + 1 if len(hit) else 1 + int(np.sum(scores[row] > scores[row, target]))

    width = max(len(group[group != source]) for group, source in zip(groups, sources))
    members = np.zeros((len(rows), width), dtype=np.int64)
    for row, (group, source) in enumerate(zip(groups, sources)):
        valid = group[group != source]
        members[row] = np.pad(valid, (0, width - len(valid)), mode="edge")
    member_base = np.take_along_axis(scores, members, axis=1)
    member_scores = rerank_scores(
        model, queries, members, seg_features, member_base, device, chunk, max_segments
    ).numpy()
    subset_ranks = np.empty(len(rows), dtype=np.int64)
    for row, target in enumerate(targets):
        target_position = np.flatnonzero(members[row] == target)[0]
        target_score = member_scores[row, target_position]
        unique_scores = {member: member_scores[row, column] for column, member in enumerate(members[row])}
        subset_ranks[row] = 1 + sum(
            member != target and score > target_score
            for member, score in unique_scores.items()
        )

    metrics = {f"R@{cutoff}": float(np.mean(ranks <= cutoff) * 100.0) for cutoff in (1, 5, 10, 50)}
    metrics.update(
        {f"subset_R@{cutoff}": float(np.mean(subset_ranks <= cutoff) * 100.0) for cutoff in (1, 2, 3)}
    )
    metrics["Avg"] = 0.5 * (metrics["R@5"] + metrics["subset_R@1"])
    return metrics


@torch.no_grad()
def make_submission(model, test_dir, dqu_gallery_path, seg_dir, alpha, device, topk, chunk, max_segments):
    test_dir = Path(test_dir)
    queries = functional.normalize(torch.from_numpy(np.load(test_dir / "queries.npy")).float(), dim=-1)
    mcot_gallery = functional.normalize(torch.from_numpy(np.load(test_dir / "gallery.npy")).float(), dim=-1)
    dqu_gallery = functional.normalize(torch.from_numpy(np.load(dqu_gallery_path)).float(), dim=-1)
    gallery_ids = json.loads((test_dir / "gallery_ids.json").read_text())
    rows = load_jsonl(test_dir / "queries.jsonl")
    if not rows or "pair_id" not in rows[0] or "target_id" in rows[0]:
        raise ValueError("test1 metadata with pair_id and no target_id is required")
    index = {name: position for position, name in enumerate(gallery_ids)}
    sources = np.asarray([index[row["source_id"]] for row in rows])
    groups = [np.asarray([index[name] for name in row["group_members"]]) for row in rows]
    seg_features = load_seg_features(seg_dir, gallery_ids)

    scores = ((1.0 - alpha) * (queries @ mcot_gallery.T) + alpha * (queries @ dqu_gallery.T)).numpy()
    scores[np.arange(len(scores)), sources] = -np.inf
    order = np.argsort(-scores, axis=1, kind="stable")
    top_indices = order[:, :topk]
    top_base = np.take_along_axis(scores, top_indices, axis=1)
    top_scores = rerank_scores(
        model, queries, top_indices, seg_features, top_base, device, chunk, max_segments
    ).numpy()
    top_order = np.argsort(-top_scores, axis=1, kind="stable")
    reranked = np.take_along_axis(top_indices, top_order, axis=1)

    widths = [len(group[group != source]) for group, source in zip(groups, sources)]
    width = max(widths)
    members = np.zeros((len(rows), width), dtype=np.int64)
    for row_index, (group, source) in enumerate(zip(groups, sources)):
        valid = group[group != source]
        members[row_index] = np.pad(valid, (0, width - len(valid)), mode="edge")
    member_base = np.take_along_axis(scores, members, axis=1)
    member_scores = rerank_scores(
        model, queries, members, seg_features, member_base, device, chunk, max_segments
    ).numpy()

    general = {"version": "rc2", "metric": "recall"}
    subset = {"version": "rc2", "metric": "recall_subset"}
    for row_index, row in enumerate(rows):
        pair_id = str(int(row["pair_id"]))
        general[pair_id] = [gallery_ids[item] for item in reranked[row_index, :50]]
        valid_width = widths[row_index]
        ranked_members = np.argsort(-member_scores[row_index, :valid_width], kind="stable")
        subset[pair_id] = [gallery_ids[members[row_index, item]] for item in ranked_members[:3]]
    return general, subset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-dir", required=True, type=Path)
    parser.add_argument("--dqu-gallery", required=True, type=Path)
    parser.add_argument("--seg-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--submission-dir", type=Path)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.125, 0.25])
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    model = SegmentReranker().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    if args.submission_dir:
        if len(args.alphas) != 1:
            raise ValueError("submission export requires exactly one alpha")
        alpha = args.alphas[0]
        general, subset = make_submission(
            model,
            args.val_dir,
            args.dqu_gallery,
            args.seg_dir,
            alpha,
            device,
            args.topk,
            args.chunk,
            args.max_segments,
        )
        args.submission_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"alpha{alpha:g}"
        general_path = args.submission_dir / f"recall_e29_e30_{suffix}.json"
        subset_path = args.submission_dir / f"recall_subset_e29_e30_{suffix}.json"
        general_path.write_text(json.dumps(general, sort_keys=True) + "\n")
        subset_path.write_text(json.dumps(subset, sort_keys=True) + "\n")
        report = {
            "split": "test1",
            "alpha": alpha,
            "queries": len(general) - 2,
            "checkpoint": str(args.checkpoint),
            "general_submission": str(general_path),
            "subset_submission": str(subset_path),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    results = {}
    for alpha in args.alphas:
        results[f"alpha_{alpha:g}"] = evaluate(
            model,
            args.val_dir,
            args.dqu_gallery,
            args.seg_dir,
            alpha,
            device,
            args.topk,
            args.chunk,
            args.max_segments,
        )
        print(json.dumps({"alpha": alpha, **results[f"alpha_{alpha:g}"]}), flush=True)
    report = {"checkpoint": str(args.checkpoint), "topk": args.topk, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
