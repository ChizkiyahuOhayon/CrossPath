#!/usr/bin/env python3
"""Train the query-conditioned segment re-ranker on cached CIRR features.

Training and evaluation both operate on the frozen model's own top-K list, so
the objective is exactly the deployed one. Re-ranking the top 50 can move
R@1/R@5/R@10 and, scored separately over the img_set group, subset R@1 -- both
components of the official Avg.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sate_reranker import SegmentReranker, pad_segments

GLOBAL_CUTOFFS = (1, 5, 10, 50)
SUBSET_CUTOFFS = (1, 2, 3)


def load_seg_features(seg_dir, names):
    feats = []
    for name in names:
        path = Path(seg_dir) / name / "seg_feature.pt"
        feats.append(torch.load(path, map_location="cpu").float()
                     if path.exists() else torch.zeros(1, 1024))
    return feats


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rerank_scores(model, queries, cand_idx, seg_feats, base, device, chunk, max_segments):
    """Score candidates in chunks; returns (N, K) re-ranked scores."""
    out = []
    for start in range(0, len(queries), chunk):
        stop = min(start + chunk, len(queries))
        q = queries[start:stop].to(device)
        idx = cand_idx[start:stop]
        seg_list = [[seg_feats[j] for j in row] for row in idx]
        padded, mask = pad_segments(seg_list, device, max_segments)
        b = torch.as_tensor(base[start:stop], device=device, dtype=torch.float32)
        out.append(model(q, padded, mask, b).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def validate(model, val_dir, seg_dir, device, topk, chunk, max_segments):
    queries = torch.from_numpy(np.load(Path(val_dir) / "queries.npy")).float()
    gallery = torch.from_numpy(np.load(Path(val_dir) / "gallery.npy")).float()
    gallery_ids = json.loads((Path(val_dir) / "gallery_ids.json").read_text())
    rows = load_jsonl(Path(val_dir) / "queries.jsonl")
    index = {n: i for i, n in enumerate(gallery_ids)}
    sources = np.asarray([index[r["source_id"]] for r in rows])
    targets = np.asarray([index[r["target_id"]] for r in rows])
    groups = [np.asarray([index[m] for m in r["group_members"]]) for r in rows]
    seg_feats = load_seg_features(seg_dir, gallery_ids)

    queries = F.normalize(queries, dim=-1)
    gallery = F.normalize(gallery, dim=-1)
    model.eval()

    scores = (queries @ gallery.T).numpy()
    scores[np.arange(len(scores)), sources] = -np.inf
    order = np.argsort(-scores, axis=1, kind="stable")
    top_idx = order[:, :topk]
    top_base = np.take_along_axis(scores, top_idx, axis=1)

    new_scores = rerank_scores(model, queries, top_idx, seg_feats, top_base,
                               device, chunk, max_segments).numpy()
    reordered = np.take_along_axis(top_idx, np.argsort(-new_scores, axis=1, kind="stable"), axis=1)

    ranks = np.zeros(len(rows), dtype=np.int64)
    for i, target in enumerate(targets):
        hit = np.where(reordered[i] == target)[0]
        if len(hit):
            ranks[i] = hit[0] + 1
        else:
            ranks[i] = 1 + int(np.sum(scores[i] > scores[i, target]))

    # subset ranking: re-rank the group members themselves
    width = max(len(g[g != s]) for g, s in zip(groups, sources))
    members = np.zeros((len(rows), width), dtype=np.int64)
    for i, (g, s) in enumerate(zip(groups, sources)):
        m = g[g != s]
        members[i] = np.pad(m, (0, width - len(m)), mode="edge")
    member_base = np.take_along_axis(scores, members, axis=1)
    member_scores = rerank_scores(model, queries, members, seg_feats, member_base,
                                  device, chunk, max_segments).numpy()
    subset_ranks = np.zeros(len(rows), dtype=np.int64)
    for i, target in enumerate(targets):
        pos = np.where(members[i] == target)[0]
        ts = member_scores[i, pos[0]]
        uniq = {}
        for j, m in enumerate(members[i]):
            uniq[m] = member_scores[i, j]
        subset_ranks[i] = 1 + sum(1 for m, sc in uniq.items() if m != target and sc > ts)

    metrics = {f"R@{k}": float(np.mean(ranks <= k) * 100) for k in GLOBAL_CUTOFFS}
    metrics.update({f"subset_R@{k}": float(np.mean(subset_ranks <= k) * 100)
                    for k in SUBSET_CUTOFFS})
    metrics["Avg"] = 0.5 * (metrics["R@5"] + metrics["subset_R@1"])
    return metrics


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.precomputed)
    queries = F.normalize(torch.from_numpy(np.load(data / "train_queries.npy")).float(), dim=-1)
    cls = F.normalize(torch.from_numpy(np.load(data / "train_target_cls.npy")).float(), dim=-1)
    targets = np.asarray(json.loads((data / "train_targets.json").read_text()))
    hard = np.load(data / "train_hard_negatives.npy")
    gallery_ids = json.loads((data / "train_gallery_ids.json").read_text())
    print(f"loading {len(gallery_ids)} train segment features", flush=True)
    seg_feats = load_seg_features(args.seg_dir, gallery_ids)

    model = SegmentReranker(head_dim=args.head_dim, hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = validate(model, args.val_dir, args.val_seg_dir, device,
                        args.topk, args.chunk, args.max_segments)
    print("epoch 0 (zero-init == frozen MCoT ranking): "
          + json.dumps({k: round(v, 4) for k, v in baseline.items()}), flush=True)

    best = baseline["Avg"]
    history = [{"epoch": 0, **baseline}]
    order = np.arange(len(queries))
    n_neg = args.topk - 1

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        rng.shuffle(order)
        total, steps = 0.0, 0
        started = time.perf_counter()
        for start in range(0, len(order) - args.batch_size + 1, args.batch_size):
            idx = order[start : start + args.batch_size]
            q = queries[idx].to(device)
            negatives = hard[idx][:, :n_neg]
            cand = np.concatenate([targets[idx][:, None], negatives], axis=1)
            base = (queries[idx] @ cls.T).gather(
                1, torch.from_numpy(cand)).to(device)
            seg_list = [[seg_feats[j] for j in row] for row in cand]
            padded, mask = pad_segments(seg_list, device, args.max_segments)
            logits = model(q, padded, mask, base) * args.logit_scale
            loss = F.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long, device=device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            steps += 1
        metrics = validate(model, args.val_dir, args.val_seg_dir, device,
                           args.topk, args.chunk, args.max_segments)
        history.append({"epoch": epoch, "loss": total / max(steps, 1), **metrics})
        print(f"epoch {epoch} loss {total / max(steps, 1):.4f} "
              f"({time.perf_counter() - started:.0f}s) "
              + json.dumps({k: round(v, 4) for k, v in metrics.items()}), flush=True)
        if metrics["Avg"] > best:
            best = metrics["Avg"]
            torch.save(model.state_dict(), out_dir / "reranker_best.pt")
            print(f"  saved new best Avg={best:.4f}", flush=True)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    print(json.dumps({"baseline_Avg": baseline["Avg"], "best_Avg": best}, indent=2))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precomputed", required=True)
    p.add_argument("--seg-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--val-seg-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--logit-scale", type=float, default=10.0)
    p.add_argument("--max-segments", type=int, default=8)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
