"""Tests for the thesis .docx editing helpers."""

from __future__ import annotations

import zipfile
from xml.etree import ElementTree as ET

import pytest

from ch3_thesis_docx import (
    Document, apply_renumber, block_text, paragraph_text, renumber_plan,
)

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _para(*runs: str) -> str:
    body = "".join(f"<w:r><w:t>{r}</w:t></w:r>" for r in runs)
    return f"<w:p>{body}</w:p>"


def _table(*rows) -> str:
    cells = "".join(
        "<w:tr>" + "".join(f"<w:tc>{_para(c)}</w:tc>" for c in row) + "</w:tr>" for row in rows
    )
    return f"<w:tbl>{cells}</w:tbl>"


def _docx(tmp_path, *blocks: str):
    xml = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           f'<w:document xmlns:w="{W}"><w:body>{"".join(blocks)}</w:body></w:document>')
    path = tmp_path / "t.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", xml)
    return str(path)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def test_paragraph_text_joins_runs_and_keeps_tabs():
    p = ET.fromstring(f'<w:p xmlns:w="{W}"><w:r><w:t>表 </w:t></w:r>'
                      f'<w:r><w:tab/><w:t>3.4</w:t></w:r></w:p>')
    assert paragraph_text(p) == "表 \t3.4"


def test_block_text_flattens_a_table(tmp_path):
    doc = Document(_docx(tmp_path, _table(["a", "b"], ["c", "d"])))
    assert block_text(doc.blocks[0]) == "a | b | c | d"


def test_find_locates_paragraphs_and_tables(tmp_path):
    doc = Document(_docx(tmp_path, _para("intro"), _table(["71.70"]), _para("outro")))
    assert doc.find("intro") == 0
    assert doc.find("71.70") == 1
    assert doc.find("outro", start=1) == 2
    with pytest.raises(LookupError):
        doc.find("intro", start=1)


# --------------------------------------------------------------------------
# deleting
# --------------------------------------------------------------------------


def test_delete_removes_an_inclusive_range(tmp_path):
    path = _docx(tmp_path, _para("keep"), _para("caption"), _table(["cell"]),
                 _para("note"), _para("also keep"))
    doc = Document(path)
    removed = doc.delete(1, 3)
    assert removed == ["caption", "cell", "note"]
    assert [doc.text(i) for i in range(len(doc.blocks))] == ["keep", "also keep"]


def test_delete_rejects_an_out_of_range_span(tmp_path):
    doc = Document(_docx(tmp_path, _para("only")))
    with pytest.raises(IndexError):
        doc.delete(0, 5)
    with pytest.raises(IndexError):
        doc.delete(1, 1)


def test_deleting_one_block_leaves_the_rest_intact_after_a_round_trip(tmp_path):
    path = _docx(tmp_path, _para("a"), _para("b"), _para("c"))
    doc = Document(path)
    doc.delete(1, 1)
    out = str(tmp_path / "out.docx")
    doc.save(out)
    assert [block_text(b) for b in Document(out).blocks] == ["a", "c"]


def test_save_preserves_every_other_part(tmp_path):
    path = _docx(tmp_path, _para("a"))
    doc = Document(path)
    out = str(tmp_path / "out.docx")
    doc.save(out)
    with zipfile.ZipFile(out) as archive:
        assert set(archive.namelist()) == {"[Content_Types].xml", "word/document.xml"}
        assert archive.read("[Content_Types].xml") == b"<Types/>"


# --------------------------------------------------------------------------
# replacing
# --------------------------------------------------------------------------


def test_replace_within_a_single_run(tmp_path):
    doc = Document(_docx(tmp_path, _para("由表 3.5 可知")))
    assert doc.replace("表 3.5", "表 3.4") == 1
    assert doc.text(0) == "由表 3.4 可知"


def test_replace_across_a_run_boundary(tmp_path):
    doc = Document(_docx(tmp_path, _para("由表 3", ".5 可知")))
    assert doc.replace("表 3.5", "表 3.4") == 1
    assert doc.text(0) == "由表 3.4 可知"


def test_replace_reaches_inside_tables(tmp_path):
    doc = Document(_docx(tmp_path, _table(["见表 3.6"])))
    assert doc.replace("表 3.6", "表 3.5") == 1
    assert doc.text(0) == "见表 3.5"


def test_replace_reports_no_hit_when_absent(tmp_path):
    doc = Document(_docx(tmp_path, _para("nothing here")))
    assert doc.replace("表 3.5", "表 3.4") == 0


# --------------------------------------------------------------------------
# renumbering
# --------------------------------------------------------------------------


def test_renumber_plan_closes_the_gap_low_to_high():
    plan = renumber_plan("表", removed=4, highest=6)
    assert plan == [("表3.5", "表3.4"), ("表 3.5", "表 3.4"),
                    ("表3.6", "表3.5"), ("表 3.6", "表 3.5")]


def test_renumber_plan_is_empty_when_the_deleted_item_is_last():
    assert renumber_plan("图", removed=8, highest=8) == []


def test_renumber_does_not_double_shift(tmp_path):
    """3.5->3.4 must not then be caught by the 3.6->3.5 rewrite."""
    doc = Document(_docx(tmp_path, _para("表 3.5 和表 3.6")))
    apply_renumber(doc, "表", removed=4, highest=6)
    assert doc.text(0) == "表 3.4 和表 3.5"


def test_renumber_reports_hits_per_pattern(tmp_path):
    doc = Document(_docx(tmp_path, _para("表3.5"), _para("表 3.5"), _para("表 3.6")))
    hits = dict(apply_renumber(doc, "表", removed=4, highest=6))
    assert hits["表3.5"] == 1 and hits["表 3.5"] == 1 and hits["表 3.6"] == 1
    assert [doc.text(i) for i in range(3)] == ["表3.4", "表 3.4", "表 3.5"]


# --------------------------------------------------------------------------
# truncating a paragraph tail
# --------------------------------------------------------------------------


def _field_para(prefix: str, field_text: str, suffix: str) -> str:
    """A paragraph whose middle is a Word REF field, as citations are stored."""
    return (f"<w:p><w:r><w:t>{prefix}</w:t></w:r>"
            f'<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            f'<w:r><w:instrText> REF _Ref1 \\r \\h </w:instrText></w:r>'
            f'<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            f"<w:r><w:t>{field_text}</w:t></w:r>"
            f'<w:r><w:fldChar w:fldCharType="end"/></w:r>'
            f"<w:r><w:t>{suffix}</w:t></w:r></w:p>")


def test_truncate_after_cuts_at_the_marker(tmp_path):
    doc = Document(_docx(tmp_path, _para("保留这句。", "删掉这句。")))
    assert doc.truncate_after(0, "保留这句。") == "删掉这句。"
    assert doc.text(0) == "保留这句。"


def test_truncate_after_splits_a_straddling_run(tmp_path):
    doc = Document(_docx(tmp_path, _para("保留这句。删掉这句。")))
    assert doc.truncate_after(0, "保留这句。") == "删掉这句。"
    assert doc.text(0) == "保留这句。"


def test_truncate_after_removes_whole_fields_in_the_tail(tmp_path):
    doc = Document(_docx(tmp_path, _field_para("保留。此外见", "[69]", "，结束。")))
    assert doc.truncate_after(0, "保留。") == "此外见[69]，结束。"
    assert doc.text(0) == "保留。"
    paragraph = doc.blocks[0]
    assert not list(paragraph.iter(f"{{{W}}}instrText"))
    assert not list(paragraph.iter(f"{{{W}}}fldChar"))


def test_truncate_after_keeps_a_field_that_precedes_the_marker(tmp_path):
    doc = Document(_docx(tmp_path, _field_para("见", "[42]", "的方法。删掉。")))
    assert doc.truncate_after(0, "的方法。") == "删掉。"
    assert doc.text(0) == "见[42]的方法。"
    assert len(list(doc.blocks[0].iter(f"{{{W}}}fldChar"))) == 3


def test_truncate_after_rejects_a_missing_marker_or_a_table(tmp_path):
    doc = Document(_docx(tmp_path, _para("abc"), _table(["x"])))
    with pytest.raises(LookupError):
        doc.truncate_after(0, "zzz")
    with pytest.raises(TypeError):
        doc.truncate_after(1, "x")


# --------------------------------------------------------------------------
# rewriting a table and a paragraph wholesale
# --------------------------------------------------------------------------


def test_set_table_rewrites_every_cell(tmp_path):
    doc = Document(_docx(tmp_path, _table(["属性", "内容"], ["图像总数", "31,783张"])))
    doc.set_table(0, [["属性", "内容"], ["图像总数", "21,551 张"]])
    assert doc.text(0) == "属性 | 内容 | 图像总数 | 21,551 张"


def test_set_table_rejects_a_shape_mismatch(tmp_path):
    doc = Document(_docx(tmp_path, _table(["a", "b"]), _para("p")))
    with pytest.raises(ValueError):
        doc.set_table(0, [["a", "b"], ["c", "d"]])
    with pytest.raises(ValueError):
        doc.set_table(0, [["a", "b", "c"]])
    with pytest.raises(TypeError):
        doc.set_table(1, [["a"]])


def test_set_table_collapses_multi_run_cells(tmp_path):
    xml = (f"<w:tbl><w:tr><w:tc><w:p>"
           f"<w:r><w:t>31,</w:t></w:r><w:r><w:t>783张</w:t></w:r>"
           f"</w:p></w:tc></w:tr></w:tbl>")
    doc = Document(_docx(tmp_path, xml))
    doc.set_table(0, [["21,551 张"]])
    assert doc.text(0) == "21,551 张"


def test_set_paragraph_replaces_the_whole_text(tmp_path):
    doc = Document(_docx(tmp_path, _para("2.5.1  Flickr30K ", "数据集")))
    assert doc.set_paragraph(0, "2.5.1  CIRR 数据集") == "2.5.1  Flickr30K 数据集"
    assert doc.text(0) == "2.5.1  CIRR 数据集"
    with pytest.raises(TypeError):
        Document(_docx(tmp_path, _table(["x"]))).set_paragraph(0, "y")


def test_set_paragraph_does_not_trap_the_text_inside_a_leading_hyperlink(tmp_path):
    xml = (f'<w:p><w:hyperlink r:id="rId9" xmlns:r="http://x"><w:r><w:t>Flickr30K</w:t></w:r>'
           f"</w:hyperlink><w:r><w:t> 数据集由 Young 等人提出。</w:t></w:r></w:p>")
    doc = Document(_docx(tmp_path, xml))
    doc.set_paragraph(0, "CIRR 数据集由 Liu 等人提出。")
    paragraph = doc.blocks[0]
    assert doc.text(0) == "CIRR 数据集由 Liu 等人提出。"
    assert not list(paragraph.iter(f"{{{W}}}hyperlink"))


def test_set_paragraph_removes_citation_field_scaffolding(tmp_path):
    doc = Document(_docx(tmp_path, _field_para("数据集", "[68]", "。包含 31,783 张图像。")))
    doc.set_paragraph(0, "CIRR 数据集[68]。包含 21,551 张图像。")
    paragraph = doc.blocks[0]
    assert doc.text(0) == "CIRR 数据集[68]。包含 21,551 张图像。"
    assert not list(paragraph.iter(f"{{{W}}}fldChar"))
    assert not list(paragraph.iter(f"{{{W}}}instrText"))


def test_set_paragraph_keeps_the_paragraph_style_and_run_formatting(tmp_path):
    xml = (f'<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr>'
           f'<w:r><w:rPr><w:b/></w:rPr><w:t>旧标题</w:t></w:r></w:p>')
    doc = Document(_docx(tmp_path, xml))
    doc.set_paragraph(0, "新标题")
    paragraph = doc.blocks[0]
    assert paragraph.find(f"{{{W}}}pPr") is not None
    assert paragraph.find(f"{{{W}}}pPr/{{{W}}}pStyle").get(f"{{{W}}}val") == "Heading2"
    assert paragraph.find(f"{{{W}}}r/{{{W}}}rPr/{{{W}}}b") is not None
    assert doc.text(0) == "新标题"
