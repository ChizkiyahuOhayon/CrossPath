#!/usr/bin/env python3
"""Mine hard negatives for SATE head training from the frozen baseline.

In-batch negatives are far too easy here: the queries come from a model already
trained to match these targets, so a 256-way in-batch softmax sits near zero
loss and carries almost no gradient signal (observed: loss 0.12, no val gain).

Evaluation ranks the target against the whole gallery, so training should do the
same. We mine, for every training query, the gallery items the frozen model
ranks above or near the true target -- exactly the confusions the segment-aware
head is supposed to resolve.

Writes train_hard_negatives.npy of shape (Nq, topk), int32 gallery indices with
the true target removed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precomputed", required=True, type=Path)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=2048)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    queries = torch.from_numpy(np.load(args.precomputed / "train_queries.npy")).float()
    cls = torch.from_numpy(np.load(args.precomputed / "train_target_cls.npy")).float()
    targets = np.asarray(json.loads((args.precomputed / "train_targets.json").read_text()))

    queries = F.normalize(queries, dim=-1).to(device)
    gallery = F.normalize(cls, dim=-1).to(device)

    mined = np.zeros((len(queries), args.topk), dtype=np.int32)
    ranks = np.zeros(len(queries), dtype=np.int32)

    with torch.no_grad():
        for start in range(0, len(queries), args.chunk):
            stop = min(start + args.chunk, len(queries))
            scores = queries[start:stop] @ gallery.T
            batch_targets = torch.from_numpy(targets[start:stop]).to(device)

            target_scores = scores.gather(1, batch_targets.unsqueeze(1))
            ranks[start:stop] = (
                (scores > target_scores).sum(dim=1).add(1).cpu().numpy()
            )

            # drop the true target, then take the top-k most confusable items
            scores.scatter_(1, batch_targets.unsqueeze(1), float("-inf"))
            top = scores.topk(args.topk, dim=1).indices
            mined[start:stop] = top.cpu().numpy().astype(np.int32)

    np.save(args.precomputed / "train_hard_negatives.npy", mined)
    stats = {
        "queries": int(len(queries)),
        "topk": args.topk,
        "baseline_train_R@1": float(np.mean(ranks == 1) * 100),
        "baseline_train_R@10": float(np.mean(ranks <= 10) * 100),
        "baseline_train_R@50": float(np.mean(ranks <= 50) * 100),
        "median_target_rank": float(np.median(ranks)),
    }
    (args.precomputed / "hard_negative_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
