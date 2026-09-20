"""Replace the chapter-2 Flickr30K dataset section with CIRR, and drop PFAN.

Table 3.4 was the thesis's only Flickr30K experiment, so once it is gone the
dataset section describes something no chapter uses.  This swaps that section
for CIRR and removes the PFAN reference, whose only citation lived in the
deleted table.

Every CIRR figure below was counted from the official release files
(``image_splits/split.rc2.*.json`` and ``captions/cap.rc2.*.json``) by
``cirr_stats.py``, not taken from a paper:

    train  images= 16939  triplets= 28225  subset size 6
    val    images=  2297  triplets=  4181  subset size 6
    test1  images=  2315  triplets=  4148  subset size 6
    TOTAL  images= 21551  triplets= 36554

The published paper states 21,552 images; the de-duplicated union of the three
official split files is 21,551, and the counted value is what is reported.

Run with ``--check`` to print the plan without writing anything.
"""

from __future__ import annotations

import argparse
import json

from ch3_thesis_docx import Document

SECTION_HEADING = ("2.5.1  Flickr30K 数据集", "2.5.1  CIRR 数据集")

DESCRIPTION = (
    "CIRR 数据集由 Liu 等人提出，是组合式图像检索任务中常用的真实场景图文数据集[68]。"
    "与仅以文本作为查询的通用图文检索不同，该数据集的每个查询由一张参考图像和一条英文修改描述共同构成，"
    "模型需要在理解参考图像视觉内容的基础上，按照修改描述的语义要求检索目标图像。"
    "数据集图像取自真实生活场景，按官方划分文件去重统计共包含 21,551 张图像和 36,554 组查询三元组，"
    "其中训练、验证和测试集分别包含 28,225、4,181 和 4,148 组查询。"
    "为提高细粒度判别难度，该数据集将视觉上高度相似的图像组织为规模为 6 的子集，每个查询均关联一个这样的子集，"
    "要求模型区分外观相近但与修改描述匹配程度不同的候选图像。"
)

TABLE_CAPTION_ZH = ("表2.1 Flickr30K 数据集信息", "表2.1 CIRR 数据集信息")
TABLE_CAPTION_EN = ("Table 2.1 Flickr30K Dataset Statistics", "Table 2.1 CIRR Dataset Statistics")

TABLE_ROWS = [
    ["属性", "内容"],
    ["图像总数", "21,551 张（按官方划分文件去重统计）"],
    ["训练/验证/测试图像数", "16,939 / 2,297 / 2,315"],
    ["查询三元组总数", "36,554 组（训练 28,225 / 验证 4,181 / 测试 4,148）"],
    ["每组查询构成", "1 张参考图像 + 1 条英文修改描述 + 1 张目标图像"],
    ["相似图像子集", "每组查询关联 1 个由 6 张视觉相似图像构成的子集"],
    ["图像内容/领域", "开放域真实生活场景"],
    ["语言", "英文"],
    ["主要用途", "组合式图像检索，评估参考图像与修改文本联合表达检索意图的建模能力"],
]

CHAPTER_SUMMARY = (
    "最后，本章对本文实验所使用的数据集进行了介绍，包括通用图文检索数据集 Flickr30K 和服饰领域图文检索数据集 FashionGen。"
    "Flickr30K 可用于验证方法在通用图文检索场景中的适用性",
    "最后，本章对本文实验所使用的数据集进行了介绍，包括组合式图像检索数据集 CIRR 和服饰领域图文检索数据集 FashionGen。"
    "CIRR 可用于验证方法在组合式图像检索场景中的适用性",
)

CIRR_REFERENCE = (
    "Liu Z, Rodriguez-Opazo C, Teney D, et al. Image retrieval on real-life images with "
    "pre-trained vision-and-language models[C]//Proceedings of the IEEE/CVF International "
    "Conference on Computer Vision. 2021: 2125-2134."
)

FLICKR_REFERENCE_ANCHOR = "Young P, Lai A, Hodosh M"
PFAN_REFERENCE_ANCHOR = "Position focused attention network"


def apply(doc: Document) -> dict:
    report: dict = {}

    heading = doc.find(SECTION_HEADING[0])
    report["heading"] = doc.set_paragraph(heading, SECTION_HEADING[1])
    report["description"] = doc.set_paragraph(heading + 1, DESCRIPTION)[:60] + "…"

    caption = doc.find(TABLE_CAPTION_ZH[0])
    report["caption_zh"] = doc.set_paragraph(caption, TABLE_CAPTION_ZH[1])
    report["caption_en"] = doc.set_paragraph(caption + 1, TABLE_CAPTION_EN[1])
    doc.set_table(caption + 2, TABLE_ROWS)
    report["table"] = doc.text(caption + 2)[:70] + "…"

    hits = doc.replace(*CHAPTER_SUMMARY)
    if hits != 1:
        raise AssertionError(f"chapter-2 summary: expected one match, got {hits}")
    report["chapter_summary"] = CHAPTER_SUMMARY[1][:50] + "…"

    ref = doc.find(FLICKR_REFERENCE_ANCHOR)
    report["reference_68_was"] = doc.set_paragraph(ref, CIRR_REFERENCE)[:70] + "…"
    report["reference_68_now"] = CIRR_REFERENCE[:70] + "…"

    pfan = doc.find(PFAN_REFERENCE_ANCHOR)
    report["reference_69_deleted"] = doc.delete(pfan, pfan)[0][:80] + "…"

    # the table of contents carries its own copy of the heading
    report["toc"] = doc.replace("2.5.1  Flickr30K 数据集", "2.5.1  CIRR 数据集")
    return report


def verify(doc: Document) -> dict:
    texts = [doc.text(i) for i in range(len(doc.blocks))]
    return {
        "flickr30k_left": [f"[{i}] {t[:70]}" for i, t in enumerate(texts) if "Flickr30K" in t],
        "pfan_left": [f"[{i}] {t[:70]}" for i, t in enumerate(texts)
                      if "PFAN" in t or "Position focused" in t],
        "cirr_now": [f"[{i}] {t[:70]}" for i, t in enumerate(texts) if "CIRR" in t],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    doc = Document(args.src)
    report = apply(doc)
    report["verification"] = verify(doc)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not args.check:
        if not args.dst:
            raise SystemExit("--dst is required unless --check is given")
        doc.save(args.dst)
        print(f"\nwrote {args.dst}")


if __name__ == "__main__":
    main()
