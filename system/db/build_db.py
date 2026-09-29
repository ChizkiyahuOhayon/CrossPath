"""把本地的 FashionGen 素材装进第五章表 5.3 定义的六张表。

商品、类别、文字描述和属性标签全部来自已有的真实标注文件，没有生成任何商品：

* ``WEAVE_HANDOFF_.../runs/*/gold_manifest.csv``
  人工审核样本的金标清单，给出 product_id、类别、视图图片、逐条描述文本
  （clause_text）以及 family/value 形式的属性标注。
* ``paper_assets/data/fashiongen_retrieval_cases.json``
  论文定性图用到的 20 个组合式检索样例，带真实的修改文本和多视图原图。

FashionGen 原始数据没有发布价格和上架季节，这两列保持 NULL，界面上显示为
「—」，不做任何填充。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from database import Database

# Windows consoles default to a legacy codepage (cp1252/cp437) that can't encode
# the Chinese status message below; force UTF-8 stdio so this never crashes the
# build partway through (the DB write itself is already done by that point, but
# a crash here reads as "build_db.py is broken" to anyone testing on Windows).
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

SYSTEM = Path(__file__).resolve().parent.parent
NLP = SYSTEM.parent.parent
RUNS = NLP / "WEAVE_HANDOFF_2026-07-31" / "server_code_and_results" / "runs"
CASES = SYSTEM.parent / "paper_assets" / "data" / "fashiongen_retrieval_cases.json"
CASE_IMAGES = SYSTEM.parent / "paper_assets"
PLACEHOLDER = SYSTEM / "data" / "demo_placeholder" / "gold_manifest.csv"
GALLERY = SYSTEM / "data" / "gallery"

# 表 5.3 的 t_attr_label 只保留五个字段；金标清单里的 family 名映射如下，
# 其余 family（closure/fit/pocket/lining/length/back_design）存进描述表。
FAMILY_TO_COLUMN = {
    "color": "color",
    "neckline": "neckline",
    "sleeve": "sleeve_length",
    "material": "material",
    "pattern": "pattern",
}


def _load_manifest(manifest: Path, products: dict[str, dict]) -> None:
    base = manifest.parent
    with manifest.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            image = base / row["image_file"]
            if not image.exists():
                continue
            pid = row["product_id"]
            item = products.setdefault(
                pid,
                {
                    "item_id": pid,
                    "category": row["category"],
                    "views": {},
                    "clauses": {},
                    "attrs": defaultdict(set),
                },
            )
            item["views"].setdefault(int(row["view_index"]), image)
            text = (row.get("clause_text") or "").strip()
            if text:
                item["clauses"][text] = None
            value = (row.get("value") or "").strip()
            if value and value != "none":
                item["attrs"][row["family"]].add(value)


def collect_audit_products() -> dict[str, dict]:
    products: dict[str, dict] = {}
    for manifest in sorted(RUNS.glob("*/gold_manifest.csv")):
        _load_manifest(manifest, products)
    return products


def collect_placeholder_products() -> dict[str, dict]:
    """Fallback for a fresh clone with none of the licensed FashionGen sources above:
    a handful of drawn (not photographed) garment silhouettes under
    data/demo_placeholder/, same manifest schema, so the demo still has real,
    searchable items instead of an empty gallery. Never mixed with real data —
    only used when collect_audit_products()/collect_case_products() found nothing."""
    products: dict[str, dict] = {}
    if PLACEHOLDER.exists():
        _load_manifest(PLACEHOLDER, products)
    return products


def collect_case_products(products: dict[str, dict]) -> list[dict]:
    if not CASES.exists():
        return []
    payload = json.loads(CASES.read_text(encoding="utf-8"))
    for pid, rel_paths in payload["source_images"].items():
        item = products.setdefault(
            pid,
            {"item_id": pid, "category": "FASHIONGEN", "views": {},
             "clauses": {}, "attrs": defaultdict(set)},
        )
        for k, rel in enumerate(rel_paths):
            path = CASE_IMAGES / rel
            if path.exists():
                item["views"].setdefault(k, path)
    return payload["cases"]


def title_for(item: dict) -> str:
    """挑一条真正像商品名的描述做标题。

    FashionGen 的描述被切成了若干子句，顺序不固定，既有商品名
    （「Short sleeve cotton jersey dress in white.」）也有零散属性
    （「Tonal stitching.」）。取第一条或最短的一条都会挑中后者，所以这里先按
    类目的中心词过滤，再取最长的一条。
    """
    heads = {w.rstrip("s") for w in re.split(r"[^a-z]+", item["category"].lower()) if len(w) > 2}
    clauses = [c for c in item["clauses"] if len(c) > 8]
    named = [c for c in clauses if any(h in c.lower() for h in heads)]
    pool = named or clauses
    if pool:
        return max(pool, key=len)[:120]
    return f"{item['category'].title()} {item['item_id']}"


def build(reset: bool = True) -> dict:
    db = Database()
    if reset:
        db.drop_all()
    db.create_schema()

    products = collect_audit_products()
    cases = collect_case_products(products)
    # collect_case_products() pre-registers an empty entry per case pid even when the
    # (gitignored, licensed) source image isn't on disk, so check for real images found
    # rather than just a non-empty dict.
    used_placeholder = False
    if not any(item["views"] for item in products.values()):
        products = collect_placeholder_products()
        cases = []
        used_placeholder = bool(products)

    GALLERY.mkdir(parents=True, exist_ok=True)
    if reset:
        for old in GALLERY.glob("*"):
            old.unlink()

    items, texts, attrs = [], [], []
    for pid, item in sorted(products.items()):
        if not item["views"]:
            continue
        # 主图固定取 view_index 最小的一张，拷贝时不做任何缩放或重压缩。
        first = min(item["views"])
        src = item["views"][first]
        dst = GALLERY / f"{pid}{src.suffix}"
        if not dst.exists():
            shutil.copy2(src, dst)
        items.append((pid, title_for(item), item["category"], None, None, None,
                      f"gallery/{dst.name}"))
        for text in item["clauses"]:
            texts.append((pid, text, " ".join(text.split()), "fashiongen"))
        row = {c: None for c in FAMILY_TO_COLUMN.values()}
        extra = []
        for family, values in item["attrs"].items():
            joined = ", ".join(sorted(values))
            column = FAMILY_TO_COLUMN.get(family)
            if column:
                row[column] = joined[:64]
            else:
                extra.append(f"{family}: {joined}")
        attrs.append((pid, row["color"], row["neckline"], row["sleeve_length"],
                      row["material"], row["pattern"]))
        if extra:
            blob = "; ".join(extra)
            texts.append((pid, blob, blob, "attribute"))

    db.executemany(
        "INSERT INTO t_clothing_item (item_id,title,category,season,gender,price,image_path)"
        " VALUES (?,?,?,?,?,?,?)", items)
    db.executemany(
        "INSERT INTO t_text_desc (item_id,raw_text,clean_text,source) VALUES (?,?,?,?)", texts)
    db.executemany(
        "INSERT INTO t_attr_label (item_id,color,neckline,sleeve_length,material,pattern)"
        " VALUES (?,?,?,?,?,?)", attrs)

    # 真实修改文本单独存一份，供 /relative 接口按参考图片回查。
    case_path = SYSTEM / "data" / "demo_cases.json"
    kept = [c for c in cases
            if c["source_id"] in products and c["target_id"] in products]
    case_path.write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")

    return {"items": len(items), "texts": len(texts), "attrs": len(attrs),
            "cases": len(kept), "backend": db.backend, "used_placeholder": used_placeholder}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="不清空已有表")
    stats = build(reset=not ap.parse_args().keep)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["used_placeholder"]:
        print("\n[提示] 没找到 WEAVE_HANDOFF/paper_assets 的授权素材，"
              "已用 data/demo_placeholder/ 里的合成占位图库（15 件手绘商品）建库，"
              "检索能跑通但不是真实 FashionGen 数据。", file=sys.stderr)
    if stats["items"] == 0:
        # 三条数据源全都没找到东西——这不该发生：仓库自带的占位图库
        # (data/demo_placeholder/) 是自包含的，正常情况下总能兜底出 15 件。
        # 大概率是下载不完整（比如只下了 system/ 的部分文件，缺了
        # data/demo_placeholder/images/ 下的图）。打印清楚缺了什么，
        # 免得看着一堆 0 猜半天，同时非零退出，后面的训练脚本不会在空
        # 图库上继续跑。
        placeholder_images = PLACEHOLDER.parent / "images"
        images_exist = placeholder_images.exists()
        image_count = len(list(placeholder_images.glob("*.jpg"))) if images_exist else 0
        lines = [
            "",
            "[错误] 图库是空的（0 件商品），三条数据源都没读到东西：",
            f"  1) WEAVE_HANDOFF: {RUNS}  存在={RUNS.exists()}",
            f"  2) paper_assets:  {CASES}  存在={CASES.exists()}",
            f"  3) 占位图库:      {PLACEHOLDER}  存在={PLACEHOLDER.exists()}",
            f"     占位图库图片目录: {placeholder_images}  存在={images_exist}  文件数={image_count}",
            "如果第 3 条的图片数不是 15，说明下载不完整——请整个仓库重新 clone/下载一遍"
            "（不要只下 system/ 子目录），确认 system/data/demo_placeholder/images/ 下有"
            " 15 张 .jpg。",
        ]
        print("\n".join(lines), file=sys.stderr)
        raise SystemExit(1)
