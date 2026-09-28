"""训练并标定第四章的边界责任门控，供系统在线选择排序动作。

流程与论文一致：只在内部划分上训练门控，再把阈值冻结下来；评测/线上都不再碰
它。损失、特征构造、效用定义全部来自仓库根目录的 ``weave_crosspath_gate.py``。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

SYSTEM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SYSTEM))
sys.path.insert(0, str(SYSTEM.parent))

from crosspath.backbone import EMBED_DIM, Endpoint  # noqa: E402
from crosspath.pipeline import (  # noqa: E402
    CUTOFFS,
    NUM_ACTIONS,
    REGRESSION_COST,
    PathScores,
    build_actions,
    membership_table,
)
from weave_crosspath_gate import (  # noqa: E402
    CrossPathGate,
    batch_listwise_loss,
    listwise_target,
    realized_cutoff_utilities,
    aggregate_cutoff_utilities,
)
from weave_crosspath import calibrate_threshold, estimate_utilities  # noqa: E402

DATA = SYSTEM / "data"
SHORTLIST = 200


def load_state():
    order = json.loads((DATA / "gallery_order.json").read_text())
    position = {item: i for i, item in enumerate(order)}
    images = np.load(DATA / "clip_image.npy")
    texts = np.load(DATA / "clip_text.npy")
    triplets = json.loads((DATA / "triplets.json").read_text())
    endpoints = []
    for i in range(2):
        endpoint = Endpoint(EMBED_DIM)
        endpoint.load_state_dict(torch.load(DATA / f"endpoint_{i}.pt")["state_dict"])
        endpoints.append(endpoint.eval().requires_grad_(False))
    galleries = [np.load(DATA / f"gallery_e{i}.npy") for i in range(2)]
    return order, position, images, texts, triplets, endpoints, galleries


def build_records(indices, position, images, texts, triplets, endpoints, galleries,
                  ids_all):
    """为每条查询算出边界候选特征、listwise 目标和已实现效用。"""
    records = []
    if len(indices) == 0:
        return records
    src = np.array([position[triplets[i]["source_id"]] for i in indices], dtype=np.int64)
    image_batch = torch.from_numpy(images[src])
    text_batch = torch.from_numpy(texts[list(indices)])
    with torch.no_grad():
        q0 = endpoints[0].encode_query(image_batch, text_batch).numpy()
        q1 = endpoints[1].encode_query(image_batch, text_batch).numpy()

    for row, qi in enumerate(indices):
        target = position[triplets[qi]["target_id"]]
        paths = PathScores(q0[row] @ galleries[0].T, q0[row] @ galleries[1].T,
                           q1[row] @ galleries[0].T, q1[row] @ galleries[1].T)
        base_order = np.argsort(-paths.s00, kind="stable")
        head = base_order[:SHORTLIST]
        if target not in head:
            continue
        sub = PathScores(paths.s00[head], paths.s01[head],
                         paths.s10[head], paths.s11[head])
        orders, percentiles, _, _ = build_actions(ids_all[head], sub)
        local_target = int(np.flatnonzero(head == target)[0])

        # 每个动作下目标的名次，用来算完全可观测的已实现效用。
        ranks = np.empty(orders.shape[0], dtype=np.int64)
        for action, order in enumerate(orders):
            ranks[action] = int(np.flatnonzero(order == local_target)[0]) + 1
        realized = aggregate_cutoff_utilities(
            realized_cutoff_utilities(ranks, CUTOFFS, REGRESSION_COST))

        per_cutoff = []
        for k in CUTOFFS:
            crossing, table = membership_table(orders, k)
            if crossing.size == 0:
                continue
            features = build_candidate_features_np(
                q0[row], galleries[0][head][crossing], percentiles[crossing], table, k)
            per_cutoff.append({
                "features": torch.from_numpy(features),
                "target": listwise_target(crossing, local_target),
                "table": table,
            })
        if per_cutoff:
            records.append({"cutoffs": per_cutoff, "realized": realized,
                            "ranks": ranks})
    return records


def build_candidate_features_np(query, documents, percentiles, table, k):
    from weave_crosspath_gate import build_candidate_features
    return build_candidate_features(query, documents, percentiles, table, k)


def main() -> None:
    order, position, images, texts, triplets, endpoints, galleries = load_state()
    ids_all = np.asarray(order, dtype=str)

    rng = np.random.default_rng(20260919)
    perm = rng.permutation(len(triplets))
    if len(triplets) >= 1700:
        internal = perm[:1200]        # 内部划分：训练门控
        holdout = perm[1200:1700]     # 标定阈值
    else:
        # 三元组数量不够 1700（比如占位演示图库）时按比例切分，保底留出至少
        # 一条做标定，不改变真实规模数据下 1200/500 的切分方式。
        split = min(len(triplets) - 1, max(1, round(len(triplets) * 0.7))) \
            if len(triplets) >= 2 else len(triplets)
        internal = perm[:split]
        holdout = perm[split:]

    print("building internal records ...")
    train_records = build_records(internal, position, images, texts, triplets,
                                  endpoints, galleries, ids_all)
    print(f"  {len(train_records)} usable queries")

    gate = CrossPathGate(EMBED_DIM, num_actions=NUM_ACTIONS, hidden_width=128)
    optimiser = torch.optim.AdamW(gate.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(3):
        total, steps = 0.0, 0
        rng.shuffle(train_records)
        for start in range(0, len(train_records), 16):
            chunk = train_records[start:start + 16]
            features, targets = [], []
            for record in chunk:
                for cut in record["cutoffs"]:
                    features.append(cut["features"])
                    targets.append(cut["target"])
            if not features:
                continue
            loss = batch_listwise_loss(gate, features, targets)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            optimiser.step()
            total += float(loss.detach())
            steps += 1
        print(f"  epoch {epoch + 1} loss {total / max(steps, 1):.4f}")

    print("calibrating threshold ...")
    cal_records = build_records(holdout, position, images, texts, triplets,
                                endpoints, galleries, ids_all)
    predicted, realized = [], []
    gate.eval()
    for record in cal_records:
        per_cutoff = []
        for cut in record["cutoffs"]:
            with torch.no_grad():
                probability = gate.probabilities(cut["features"]).numpy()
            per_cutoff.append(estimate_utilities(cut["table"], probability,
                                                 REGRESSION_COST))
        if not per_cutoff:
            continue
        predicted.append(aggregate_cutoff_utilities(np.stack(per_cutoff, axis=0)))
        realized.append(record["realized"])

    threshold, mean_utility = calibrate_threshold(np.stack(predicted),
                                                  np.stack(realized))
    print(f"  threshold {threshold:.6f}  mean realized utility {mean_utility:.4f}")

    torch.save({"state_dict": gate.state_dict(), "threshold": threshold,
                "embedding_dim": EMBED_DIM, "num_actions": NUM_ACTIONS,
                "cutoffs": list(CUTOFFS), "regression_cost": REGRESSION_COST,
                "calibration_queries": len(predicted)},
               DATA / "gate.pt")
    print(f"saved -> {DATA / 'gate.pt'}")


if __name__ == "__main__":
    main()
