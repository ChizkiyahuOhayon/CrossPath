"""离线准备：抽 CLIP 特征、挖组合式检索三元组、训练两个基础模型的轻量头。

跑一次就够，产物全部落在 ``system/data/`` 下：

    clip_image.npy      721 件商品的冻结 CLIP 图像特征
    triplets.json       从真实属性标注挖出的 (参考图, 修改文本, 目标图) 三元组
    endpoint_0.pt       基础模型 E0 的头（种子 20260718）
    endpoint_1.pt       基础模型 E1 的头（种子 20260722）
    gallery_e0.npy      E0 的图库表示 g_0
    gallery_e1.npy      E1 的图库表示 g_1

修改文本不是编出来的：它直接念出目标商品在 gold_manifest 里标注的属性值，
模板只提供「把……换成……」这层句法。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

SYSTEM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SYSTEM))
sys.path.insert(0, str(SYSTEM / "db"))

from crosspath.backbone import EMBED_DIM, SEEDS, Backbone, Endpoint  # noqa: E402
from database import Database  # noqa: E402

DATA = SYSTEM / "data"
ATTR_COLUMNS = ("color", "neckline", "sleeve_length", "material", "pattern")
PHRASE = {
    "color": "change the colour to {}",
    "neckline": "switch to a {} neckline",
    "sleeve_length": "make the sleeves {}",
    "material": "use {} instead",
    "pattern": "make the pattern {}",
}


def load_catalog() -> tuple[list[str], dict[str, dict]]:
    db = Database()
    items = db.query("SELECT item_id, title, category, image_path FROM t_clothing_item"
                     " ORDER BY item_id")
    attrs = {r["item_id"]: r for r in db.query(
        "SELECT item_id, color, neckline, sleeve_length, material, pattern FROM t_attr_label")}
    catalog = {}
    for row in items:
        catalog[row["item_id"]] = {**row, "attrs": attrs.get(row["item_id"], {})}
    return [r["item_id"] for r in items], catalog


def extract_image_features(order: list[str], catalog: dict) -> np.ndarray:
    cache = DATA / "clip_image.npy"
    if cache.exists():
        features = np.load(cache)
        if features.shape[0] == len(order):
            return features
    backbone = Backbone()
    images = [Image.open(DATA / catalog[i]["image_path"]).convert("RGB") for i in order]
    features = backbone.encode_images(images)
    np.save(cache, features)
    return features


def mine_triplets(order: list[str], catalog: dict, per_item: int = 6) -> list[dict]:
    """同类目内找属性有差异的商品对，把目标的差异属性念成修改文本。"""
    by_category: dict[str, list[str]] = {}
    for item_id in order:
        by_category.setdefault(catalog[item_id]["category"], []).append(item_id)

    rng = random.Random(20260919)
    triplets = []
    for category, members in by_category.items():
        if len(members) < 2:
            continue
        for source in members:
            pool = [m for m in members if m != source]
            rng.shuffle(pool)
            taken = 0
            for target in pool:
                changed = []
                for column in ATTR_COLUMNS:
                    sv = (catalog[source]["attrs"].get(column) or "").strip()
                    tv = (catalog[target]["attrs"].get(column) or "").strip()
                    if tv and tv != sv:
                        changed.append(PHRASE[column].format(tv))
                if not 1 <= len(changed) <= 3:
                    continue
                triplets.append({
                    "source_id": source,
                    "target_id": target,
                    "category": category,
                    "modification": ", ".join(changed[:3]) + ".",
                })
                taken += 1
                if taken >= per_item:
                    break
    rng.shuffle(triplets)
    (DATA / "triplets.json").write_text(
        json.dumps(triplets, ensure_ascii=False, indent=1), encoding="utf-8")
    return triplets


def extract_text_features(triplets: list[dict]) -> np.ndarray:
    cache = DATA / "clip_text.npy"
    if cache.exists():
        features = np.load(cache)
        if features.shape[0] == len(triplets):
            return features
    backbone = Backbone()
    features = backbone.encode_texts([t["modification"] for t in triplets])
    np.save(cache, features)
    return features


def train_endpoint(seed: int, image_features, text_features, src_idx, tgt_idx,
                   epochs: int = 12, batch: int = 128, lr: float = 3e-4) -> Endpoint:
    torch.manual_seed(seed)
    endpoint = Endpoint(EMBED_DIM, seed=seed)
    optimiser = torch.optim.AdamW(endpoint.parameters(), lr=lr, weight_decay=1e-4)
    images = torch.from_numpy(image_features)
    texts = torch.from_numpy(text_features)
    n = len(src_idx)
    order = np.arange(n)
    generator = np.random.default_rng(seed)
    scale = 20.0
    for epoch in range(epochs):
        generator.shuffle(order)
        total = 0.0
        steps = 0
        for start in range(0, n, batch):
            rows = order[start:start + batch]
            if rows.size < 8:
                continue
            q = endpoint.encode_query(images[src_idx[rows]], texts[rows])
            # 批内负样本：本批目标商品的图库表示即候选集合。
            g = endpoint.encode_gallery(images[tgt_idx[rows]])
            logits = scale * q @ g.T
            labels = torch.arange(rows.size)
            loss = nn.functional.cross_entropy(logits, labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss)
            steps += 1
        print(f"  seed {seed} epoch {epoch + 1:2d}  loss {total / max(steps, 1):.4f}")
    return endpoint.eval().requires_grad_(False)


def main() -> None:
    order, catalog = load_catalog()
    print(f"catalog: {len(order)} items")
    image_features = extract_image_features(order, catalog)
    print(f"clip image features: {image_features.shape}")

    triplets = mine_triplets(order, catalog)
    print(f"mined triplets: {len(triplets)}")
    text_features = extract_text_features(triplets)

    position = {item_id: i for i, item_id in enumerate(order)}
    src_idx = np.array([position[t["source_id"]] for t in triplets])
    tgt_idx = np.array([position[t["target_id"]] for t in triplets])

    images = torch.from_numpy(image_features)
    for index, seed in enumerate(SEEDS):
        endpoint = train_endpoint(seed, image_features, text_features, src_idx, tgt_idx)
        torch.save({"state_dict": endpoint.state_dict(), "seed": seed, "dim": EMBED_DIM},
                   DATA / f"endpoint_{index}.pt")
        with torch.no_grad():
            gallery = endpoint.encode_gallery(images).numpy().astype(np.float32)
        np.save(DATA / f"gallery_e{index}.npy", gallery)
        print(f"endpoint {index} (seed {seed}) -> gallery {gallery.shape}")

    (DATA / "gallery_order.json").write_text(json.dumps(order), encoding="utf-8")


if __name__ == "__main__":
    main()
