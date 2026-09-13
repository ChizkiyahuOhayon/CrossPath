#!/usr/bin/env python3
"""Train the SATE target head on precomputed frozen MCoT-MVS features.

The head operates on the L2-normalised CLS embedding, which is exactly what
`CIRModel.extract_target` returns. That makes both training and validation run
entirely off cached embeddings: the already-exported CIRR val embeddings are
valid head inputs, so no ViT-H forward pass is needed to score an epoch.

Validation reports the seven official CIRR metrics plus Avg = (R@5 + sR@1)/2,
with the reference image excluded from the gallery, matching the protocol used
for every other number in this project.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sate_model import TargetSegmentHead

GLOBAL_CUTOFFS = (1, 5, 10, 50)
SUBSET_CUTOFFS = (1, 2, 3)


def load_seg_features(seg_dir, names, device):
    feats = []
    for name in names:
        path = Path(seg_dir) / name / "seg_feature.pt"
        if path.exists():
            feats.append(torch.load(path, map_location="cpu").float())
        else:
            feats.append(torch.zeros(1, 1024))
    return feats


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@torch.no_grad()
def encode_gallery(head, gallery_norm_cls, seg_feats, batch_size, device):
    outputs = []
    for start in range(0, len(gallery_norm_cls), batch_size):
        stop = min(start + batch_size, len(gallery_norm_cls))
        cls = gallery_norm_cls[start:stop].to(device)
        out = head(cls, seg_feats[start:stop])
        outputs.append(F.normalize(out, dim=-1).cpu())
    return torch.cat(outputs)


@torch.no_grad()
def validate(head, val_dir, seg_dir, device, batch_size=256):
    queries = torch.from_numpy(np.load(Path(val_dir) / "queries.npy")).float()
    gallery = torch.from_numpy(np.load(Path(val_dir) / "gallery.npy")).float()
    gallery_ids = json.loads((Path(val_dir) / "gallery_ids.json").read_text())
    rows = load_jsonl(Path(val_dir) / "queries.jsonl")
    index = {name: i for i, name in enumerate(gallery_ids)}
    sources = np.asarray([index[r["source_id"]] for r in rows])
    targets = np.asarray([index[r["target_id"]] for r in rows])
    groups = [np.asarray([index[m] for m in r["group_members"]]) for r in rows]

    seg_feats = load_seg_features(seg_dir, gallery_ids, device)
    head.eval()
    gallery_emb = encode_gallery(head, gallery, seg_feats, batch_size, device)
    queries = F.normalize(queries, dim=-1)

    scores = (queries @ gallery_emb.T).numpy()
    scores[np.arange(len(scores)), sources] = -np.inf

    target_scores = scores[np.arange(len(scores)), targets]
    ranks = 1 + np.sum(scores > target_scores[:, None], axis=1)

    subset_ranks = []
    for i, (members, target, source) in enumerate(zip(groups, targets, sources)):
        members = members[members != source]
        ts = scores[i, target]
        subset_ranks.append(1 + int(np.sum(scores[i, members] > ts)))
    subset_ranks = np.asarray(subset_ranks)

    metrics = {f"R@{k}": float(np.mean(ranks <= k) * 100) for k in GLOBAL_CUTOFFS}
    metrics.update({f"subset_R@{k}": float(np.mean(subset_ranks <= k) * 100)
                    for k in SUBSET_CUTOFFS})
    metrics["Avg"] = 0.5 * (metrics["R@5"] + metrics["subset_R@1"])
    return metrics


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(args.precomputed)
    queries = torch.from_numpy(np.load(data / "train_queries.npy")).float()
    cls = torch.from_numpy(np.load(data / "train_target_cls.npy")).float()
    cls = F.normalize(cls, dim=-1)
    gallery_ids = json.loads((data / "train_gallery_ids.json").read_text())
    targets = json.loads((data / "train_targets.json").read_text())
    queries = F.normalize(queries, dim=-1)

    print(f"loading {len(gallery_ids)} train segment features", flush=True)
    seg_feats = load_seg_features(args.seg_dir, gallery_ids, device)

    head = TargetSegmentHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-2)
    scale = args.logit_scale

    baseline = validate(head, args.val_dir, args.val_seg_dir, device)
    print("epoch 0 (identity == frozen MCoT baseline): "
          + json.dumps({k: round(v, 4) for k, v in baseline.items()}), flush=True)

    best = baseline["Avg"]
    history = [{"epoch": 0, **baseline}]
    order = np.arange(len(queries))

    hard = None
    if args.hard_negatives:
        hard_path = data / "train_hard_negatives.npy"
        hard = np.load(hard_path)
        print(f"hard negatives: {hard.shape} from {hard_path}", flush=True)

    groups_arr = None
    if args.group_negatives:
        groups_arr = np.load(data / "train_group_members.npy")
        print(f"group negatives: {groups_arr.shape}", flush=True)

    for epoch in range(1, args.epochs + 1):
        head.train()
        rng = np.random.default_rng(args.seed + epoch)
        rng.shuffle(order)
        total, steps = 0.0, 0
        started = time.perf_counter()
        for start in range(0, len(order) - args.batch_size + 1, args.batch_size):
            idx = order[start : start + args.batch_size]
            q = queries[idx].to(device)
            tgt_idx = [targets[i] for i in idx]

            if hard is None:
                pool = list(dict.fromkeys(tgt_idx))
            else:
                # Evaluation ranks the target against the full gallery, so train
                # against the items the frozen model actually confuses it with
                # rather than random in-batch negatives.
                picks = rng.integers(0, hard.shape[1], size=(len(idx), args.hard_per_query))
                negatives = hard[np.asarray(idx)[:, None], picks].reshape(-1).tolist()
                if groups_arr is not None:
                    # subset R@1 is scored inside the img_set group, so the group
                    # members are the negatives that metric is defined over.
                    negatives = negatives + groups_arr[np.asarray(idx)].reshape(-1).tolist()
                pool = list(dict.fromkeys(tgt_idx + negatives))

            position = {g: i for i, g in enumerate(pool)}
            labels = torch.tensor([position[g] for g in tgt_idx], device=device)
            p_cls = cls[pool].to(device)
            p_seg = [seg_feats[j] for j in pool]
            t = F.normalize(head(p_cls, p_seg), dim=-1)
            logits = scale * q @ t.T
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            steps += 1
        metrics = validate(head, args.val_dir, args.val_seg_dir, device)
        history.append({"epoch": epoch, "loss": total / max(steps, 1), **metrics})
        print(f"epoch {epoch} loss {total / max(steps, 1):.4f} "
              f"({time.perf_counter() - started:.0f}s) "
              + json.dumps({k: round(v, 4) for k, v in metrics.items()}), flush=True)
        if metrics["Avg"] > best:
            best = metrics["Avg"]
            torch.save(head.state_dict(), Path(args.output_dir) / "sate_head_best.pt")
            print(f"  saved new best Avg={best:.4f}", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps({"baseline_Avg": baseline["Avg"], "best_Avg": best}, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precomputed", required=True)
    parser.add_argument("--seg-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--val-seg-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--logit-scale", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-negatives", action="store_true",
                        help="train against mined top-k confusions instead of in-batch")
    parser.add_argument("--hard-per-query", type=int, default=16)
    parser.add_argument("--group-negatives", action="store_true",
                        help="add img_set group members to the negative pool")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
