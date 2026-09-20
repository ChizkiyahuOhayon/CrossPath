"""Remove table 3.4 and figure 3.5 from the chapter-3 thesis, and repair the text.

Both deletions pull more than a caption with them.  Table 3.4 is the only
Flickr30K result in the thesis, so every sentence that promises or interprets
that experiment goes too, from the abstracts down to the chapter summary.
Figure 3.5 is the whole of section 3.3.6.2, so the parameter-section preamble
that announces it has to be rewritten and 3.3.6.3 renumbered.

What this script deliberately does NOT touch, because it is a separate call:

* section 2.5.1 and table 2.1, which describe the Flickr30K dataset itself;
* the reference list - reference numbers are Word ``REF`` fields that Word
  renumbers on its own, and PFAN[69] loses its only citation here.

Run with ``--check`` to print the plan without writing anything.
"""

from __future__ import annotations

import argparse
import json
from typing import List, Tuple

from ch3_thesis_docx import Document, apply_renumber

# (anchor of the first block, anchor of the last block, what it is)
BLOCK_DELETIONS: List[Tuple[str, str, str]] = [
    ("鉴于服饰图文数据具有较强的领域特殊性",
     "尽管如此，本文模型在 Flickr30K",
     "table 3.4: the Flickr30K comparison, its table, caption, note and analysis"),
    ("3.3.6.2 注意力头数实验",
     "Fig. 3.5  R@1 Trends under Different Numbers of Cross-Attention Heads",
     "figure 3.5: the whole attention-head subsection, its figure and captions"),
]

# Sentences that survive only because the deleted experiments existed.
TEXT_EDITS: List[Tuple[str, str, str]] = [
    ("均取得稳定提升，并通过 Flickr30K 数据集上的补充实验验证了其在通用图文检索场景中的适用性。",
     "均取得稳定提升。",
     "Chinese abstract"),
    ("under the Sample and Full settings of the FashionGen dataset, and supplementary "
     "experiments on the Flickr30K dataset verify its applicability to general "
     "image-text retrieval scenarios.",
     "under the Sample and Full settings of the FashionGen dataset.",
     "English abstract"),
    ("本章在 FashionGen 数据集上进行了对比实验与消融实验，同时选取Flickr30K 作为通用图文检索的补充验证集，"
     "检验模型在非服饰领域图文匹配任务中的泛化性能。两类数据集的详细统计参数可参见第二章 2.6 节。",
     "本章在 FashionGen 数据集上进行了对比实验与消融实验。该数据集的详细统计参数可参见第二章 2.6 节。",
     "section 3.3.2, dataset paragraph"),
    ("表 3.6 展示了不同视觉层级组合对模型性能的影响，图 3.5 展示了不同交叉注意力头数下 R@1 指标的变化趋势，"
     "图 3.6 则展示了区域增强对齐损失权重调整对模型检索性能的影响。",
     "表 3.6 展示了不同视觉层级组合对模型性能的影响，图 3.6 则展示了区域增强对齐损失权重调整对模型检索性能的影响。",
     "section 3.3.6 preamble, figure list"),
    ("本节进一步探讨了本文模型中三个关键参数对检索性能的影响，包括视觉特征层级组合、"
     "分层跨模态交叉注意力模块中的注意力头数 h，以及区域增强对齐损失权重 λreg。",
     "本节进一步探讨了本文模型中两个关键参数对检索性能的影响，包括视觉特征层级组合和区域增强对齐损失权重 λreg。",
     "section 3.3.6 preamble, parameter list"),
    ("本文默认采用第 4 层、第 8 层和第 12 层视觉 patch 特征作为多层视觉输入，交叉注意力头数设置为 8，"
     "区域增强对齐损失权重设置为 0.5。",
     "本文默认采用第 4 层、第 8 层和第 12 层视觉 patch 特征作为多层视觉输入，区域增强对齐损失权重设置为 0.5。",
     "section 3.3.6 preamble, default settings"),
    ("本章分别在 FashionGen 数据集的 Sample 和 Full 设定下进行了对比实验，并在 Flickr30K 数据集上进行了补充实验。",
     "本章分别在 FashionGen 数据集的 Sample 和 Full 设定下进行了对比实验。",
     "section 3.4 chapter summary"),
    ("该方法在 FashionGen 数据集的 Sample 和 Full 设定下均取得了较好的检索效果，"
     "并通过 Flickr30K 数据集上的补充实验验证了方法在通用图文检索场景中的适用性。",
     "该方法在 FashionGen 数据集的 Sample 和 Full 设定下均取得了较好的检索效果。",
     "conclusions, contribution (1)"),
    ("3.3.6.3 区域增强对齐损失权重实验", "3.3.6.2 区域增强对齐损失权重实验",
     "section 3.3.6.3 renumbered after 3.3.6.2 was removed"),
]

# A trailing span that carries Word REF citation fields, so it is cut, not rewritten.
TAIL_DELETION = ("搭配属性提示模板让模型显式学习面料、版型等服饰细节特征。",
                 "section 3.3.3, the Flickr30K comparison-method paragraph")

# Deleting item 4 of each series closes the gap up to the highest item in use.
RENUMBERING = [("表", 4, 6), ("图", 5, 8)]


def apply(doc: Document) -> dict:
    report: dict = {"deleted_blocks": [], "text_edits": [], "tail_deleted": "",
                    "renumbered": []}

    for first_anchor, last_anchor, what in BLOCK_DELETIONS:
        first = doc.find(first_anchor)
        last = doc.find(last_anchor, start=first)
        report["deleted_blocks"].append(
            {"what": what, "blocks": last - first + 1,
             "removed": [t[:70] for t in doc.delete(first, last)]})

    index = doc.find(TAIL_DELETION[0])
    report["tail_deleted"] = {"what": TAIL_DELETION[1],
                              "removed": doc.truncate_after(index, TAIL_DELETION[0])}

    for old, new, what in TEXT_EDITS:
        hits = doc.replace(old, new)
        if hits != 1:
            raise AssertionError(f"{what}: expected exactly one match, got {hits}")
        report["text_edits"].append({"what": what, "new": new[:70]})

    for kind, removed, highest in RENUMBERING:
        report["renumbered"] += [{"pattern": old, "paragraphs": n}
                                 for old, n in apply_renumber(doc, kind, removed, highest)
                                 if n]
    return report


#: Nothing in the edited document may mention these any more.
MUST_BE_GONE = ("表 3.4 本文模型与其他方法", "由表 3.4 可知", "图3.5 不同交叉注意力",
                "3.3.6.2 注意力头数实验", "由图 3.5 可知", "注意力头数",
                "表 3.6", "表3.6", "图 3.8", "图3.8", "SGRAF 相比")

#: Mentions this script leaves for a separate decision, reported per location.
LEFT_BEHIND = ("Flickr30K", "SCAN", "PFAN", "IMRAM", "SGRAF")


def verify(doc: Document) -> dict:
    """What the edit was supposed to remove, and what it knowingly left."""
    texts = [doc.text(i) for i in range(len(doc.blocks))]
    body = "\n".join(texts)
    return {
        "residue": [n for n in MUST_BE_GONE if n in body],
        "left_behind": {
            n: [f"[{i}] {t[:60]}" for i, t in enumerate(texts) if n in t]
            for n in LEFT_BEHIND if any(n in t for t in texts)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
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
