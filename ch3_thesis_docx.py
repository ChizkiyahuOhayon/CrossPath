"""Minimal, auditable editing of the thesis .docx.

Word stores a paragraph as a list of runs and splits a sentence across them at
arbitrary points, so "table 3.4" is frequently three runs.  Every helper here
therefore works on the *paragraph* text and writes the result back, which keeps
edits verifiable: :func:`paragraph_text` is the single definition of what a
paragraph says, and every operation is expressed in terms of it.

The module only knows how to delete body blocks and rewrite text.  Which blocks
to delete is the caller's decision and is recorded in the edit script.
"""

from __future__ import annotations

import copy
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Sequence, Tuple
from xml.etree import ElementTree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
ET.register_namespace("w", W)

DOCUMENT = "word/document.xml"


def _tag(element: ET.Element) -> str:
    return element.tag.split("}")[-1]


def paragraph_text(paragraph: ET.Element) -> str:
    """The visible text of a ``w:p``, with tabs preserved and runs joined."""
    out = []
    for node in paragraph.iter():
        name = _tag(node)
        if name == "t":
            out.append(node.text or "")
        elif name == "tab":
            out.append("\t")
        elif name in ("br", "cr"):
            out.append("\n")
    return "".join(out)


def block_text(block: ET.Element) -> str:
    """The visible text of one body child, be it a paragraph or a table."""
    if _tag(block) == "p":
        return paragraph_text(block)
    return " | ".join(paragraph_text(p) for p in block.iter(f"{{{W}}}p"))


class Document:
    """A .docx opened for editing, backed by its ``word/document.xml`` tree."""

    def __init__(self, path: str):
        self.path = path
        with zipfile.ZipFile(path) as archive:
            self._names = archive.namelist()
            self._parts = {name: archive.read(name) for name in self._names}
        self.tree = ET.fromstring(self._parts[DOCUMENT])
        self.body = self.tree.find("w:body", NS)

    # -- reading ---------------------------------------------------------

    @property
    def blocks(self) -> List[ET.Element]:
        """Body children in document order, excluding the trailing section mark."""
        return [b for b in self.body if _tag(b) in ("p", "tbl")]

    def find(self, needle: str, start: int = 0) -> int:
        """Index of the first block at or after ``start`` containing ``needle``."""
        blocks = self.blocks
        for i in range(start, len(blocks)):
            if needle in block_text(blocks[i]):
                return i
        raise LookupError(f"no block contains {needle!r} at or after index {start}")

    def text(self, index: int) -> str:
        return block_text(self.blocks[index])

    # -- writing ---------------------------------------------------------

    def delete(self, first: int, last: int) -> List[str]:
        """Remove blocks ``first..last`` inclusive; returns what was removed."""
        blocks = self.blocks
        if not 0 <= first <= last < len(blocks):
            raise IndexError(f"cannot delete {first}..{last} of {len(blocks)} blocks")
        removed = [block_text(b) for b in blocks[first:last + 1]]
        for block in blocks[first:last + 1]:
            self.body.remove(block)
        return removed

    def replace(self, old: str, new: str) -> int:
        """Rewrite ``old`` to ``new`` everywhere; returns the number of paragraphs hit."""
        hits = 0
        for paragraph in self.tree.iter(f"{{{W}}}p"):
            if old not in paragraph_text(paragraph):
                continue
            hits += 1
            runs = [t for t in paragraph.iter(f"{{{W}}}t")]
            for node in runs:                      # the common case: one run holds it
                if node.text and old in node.text:
                    node.text = node.text.replace(old, new)
            if old in paragraph_text(paragraph):   # it straddled a run boundary
                merged = paragraph_text(paragraph).replace(old, new)
                runs[0].text = merged
                runs[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                for node in runs[1:]:
                    node.text = ""
        return hits

    def truncate_after(self, index: int, marker: str) -> str:
        """Keep a paragraph's text up to and including ``marker``, drop the rest.

        Citation numbers in this thesis are Word ``REF`` fields, which span
        several elements and cannot survive having their text rewritten.  Text
        that ends a paragraph is therefore cut by dropping whole elements rather
        than by editing run text, so any field in the removed tail goes with it.
        """
        paragraph = self.blocks[index]
        if _tag(paragraph) != "p":
            raise TypeError(f"block {index} is a table, not a paragraph")
        text = paragraph_text(paragraph)
        start = text.find(marker)
        if start < 0:
            raise LookupError(f"paragraph {index} does not contain {marker!r}")
        cut = start + len(marker)

        seen, dropping = 0, False
        for child in list(paragraph):
            if _tag(child) == "pPr":
                continue
            if dropping:
                paragraph.remove(child)
                continue
            length = len("".join(t.text or "" for t in child.iter(f"{{{W}}}t")))
            if seen + length <= cut:
                seen += length
                continue
            keep = cut - seen                      # this element straddles the cut
            if _tag(child) == "r":
                for node in child.iter(f"{{{W}}}t"):
                    node.text = (node.text or "")[:keep]
                    keep = max(keep - len(node.text), 0)
            else:
                paragraph.remove(child)
            dropping = True
        return text[cut:]

    def set_table(self, index: int, rows: Sequence[Sequence[str]]) -> None:
        """Rewrite every cell of a table, keeping its formatting.

        The shape must match the table already there: this replaces contents,
        it does not restructure, so a mismatch is a mistake in the caller and
        is raised rather than silently patched.
        """
        table = self.blocks[index]
        if _tag(table) != "tbl":
            raise TypeError(f"block {index} is not a table")
        existing = table.findall(f"{{{W}}}tr")
        if len(existing) != len(rows):
            raise ValueError(f"table has {len(existing)} rows, got {len(rows)}")
        for tr, values in zip(existing, rows):
            cells = tr.findall(f"{{{W}}}tc")
            if len(cells) != len(values):
                raise ValueError(f"row has {len(cells)} cells, got {len(values)}")
            for tc, value in zip(cells, values):
                self._set_cell(tc, value)

    @staticmethod
    def _set_cell(cell: ET.Element, value: str) -> None:
        nodes = list(cell.iter(f"{{{W}}}t"))
        if not nodes:
            raise ValueError("cell has no text run to write into")
        nodes[0].text = value
        nodes[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        for node in nodes[1:]:
            node.text = ""

    def set_paragraph(self, index: int, text: str) -> str:
        """Replace a paragraph's whole content with one run of ``text``.

        Every run, hyperlink and field is dropped and a single run takes their
        place, carrying the character formatting of the paragraph's first run.
        Writing into the existing first ``w:t`` would not do: when a paragraph
        opens with a hyperlink, that would pull the entire sentence inside the
        link, and when it contains citation fields it would leave their
        ``fldChar`` scaffolding behind pointing at nothing.
        """
        paragraph = self.blocks[index]
        if _tag(paragraph) != "p":
            raise TypeError(f"block {index} is not a paragraph")
        previous = paragraph_text(paragraph)

        template = next((r for r in paragraph.iter(f"{{{W}}}r")
                         if r.find(f"{{{W}}}rPr") is not None), None)
        run_props = copy.deepcopy(template.find(f"{{{W}}}rPr")) if template is not None else None
        for child in list(paragraph):
            if _tag(child) != "pPr":
                paragraph.remove(child)

        run = ET.SubElement(paragraph, f"{{{W}}}r")
        if run_props is not None:
            run.append(run_props)
        node = ET.SubElement(run, f"{{{W}}}t")
        node.text = text
        node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        return previous

    def save(self, path: str) -> None:
        """Write a new .docx, copying every part except the edited document."""
        payload = ET.tostring(self.tree, encoding="UTF-8", xml_declaration=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in self._names:
                archive.writestr(name, payload if name == DOCUMENT else self._parts[name])


# --------------------------------------------------------------------------
# renumbering
# --------------------------------------------------------------------------


def renumber_plan(kind: str, removed: int, highest: int) -> List[Tuple[str, str]]:
    """Ordered rewrites that close the gap left by deleting item ``removed``.

    ``kind`` is the Chinese label, e.g. ``"表"`` or ``"图"``.  Items are rewritten
    low-to-high so that 3.5->3.4 happens before 3.6->3.5 and no rewrite is
    applied twice.  Both the spaced and unspaced spellings that the thesis uses
    are covered.
    """
    plan: List[Tuple[str, str]] = []
    for n in range(removed + 1, highest + 1):
        for space in ("", " "):
            plan.append((f"{kind}{space}3.{n}", f"{kind}{space}3.{n - 1}"))
    return plan


def apply_renumber(doc: Document, kind: str, removed: int, highest: int) -> List[Tuple[str, int]]:
    return [(old, doc.replace(old, new)) for old, new in renumber_plan(kind, removed, highest)]
