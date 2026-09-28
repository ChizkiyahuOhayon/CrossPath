#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw a tiny, fully-synthetic placeholder catalog (flat-shaded garment silhouettes,
nothing photographed or scraped) and write a gold_manifest.csv in the exact schema
db/build_db.py already knows how to read. Only used when no licensed FashionGen data
(WEAVE_HANDOFF_2026-07-31, paper_assets/source_images) is available, so a fresh public
clone still has a handful of real, searchable items instead of an empty gallery.

Re-run this to regenerate the images/manifest from scratch; nothing here is licensed.
"""
import csv
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"
IMAGES.mkdir(exist_ok=True)

BG = (238, 242, 247)
SIZE = 256

# (product_id, category, color, color_rgb, sleeve_length, material, pattern)
ITEMS = [
    ("demo0001", "DRESSES", "red", (192, 57, 43), "long", "cotton", "solid"),
    ("demo0002", "DRESSES", "blue", (41, 128, 185), "short", "cotton", "solid"),
    ("demo0003", "DRESSES", "green", (39, 174, 96), "long", "cotton", "striped"),
    ("demo0004", "SHIRTS", "red", (192, 57, 43), "short", "cotton", "solid"),
    ("demo0005", "SHIRTS", "blue", (41, 128, 185), "long", "cotton", "solid"),
    ("demo0006", "SHIRTS", "white", (250, 250, 250), "long", "cotton", "striped"),
    ("demo0007", "PANTS", "blue", (41, 128, 185), "-", "denim", "solid"),
    ("demo0008", "PANTS", "black", (44, 44, 44), "-", "denim", "solid"),
    ("demo0009", "PANTS", "khaki", (194, 178, 128), "-", "cotton", "solid"),
    ("demo0010", "JACKETS", "black", (44, 44, 44), "long", "leather", "solid"),
    ("demo0011", "JACKETS", "red", (192, 57, 43), "long", "wool", "solid"),
    ("demo0012", "JACKETS", "blue", (41, 128, 185), "short", "denim", "solid"),
    ("demo0013", "SHOES", "white", (250, 250, 250), "-", "leather", "solid"),
    ("demo0014", "SHOES", "black", (44, 44, 44), "-", "leather", "solid"),
    ("demo0015", "SHOES", "brown", (120, 72, 40), "-", "leather", "solid"),
]


def stripe(draw, box, rgb):
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=rgb)
    light = tuple(min(255, c + 60) for c in rgb)
    step = 14
    y = y0
    on = True
    while y < y1:
        if on:
            draw.rectangle((x0, y, x1, min(y + step, y1)), fill=light)
        y += step
        on = not on


def draw_dress(draw, rgb, pattern):
    box = (88, 60, 168, 200)
    fn = stripe if pattern == "striped" else draw.rectangle
    if pattern == "striped":
        stripe(draw, box, rgb)
    else:
        draw.polygon([(108, 60), (148, 60), (168, 200), (88, 200)], fill=rgb)
    draw.ellipse((108, 45, 148, 70), outline=(90, 90, 90), width=3)


def draw_shirt(draw, rgb, pattern):
    body = (96, 90, 160, 190)
    if pattern == "striped":
        stripe(draw, body, rgb)
    else:
        draw.rectangle(body, fill=rgb)
    draw.polygon([(96, 95), (60, 130), (78, 150), (96, 120)], fill=rgb)
    draw.polygon([(160, 95), (196, 130), (178, 150), (160, 120)], fill=rgb)
    draw.ellipse((112, 78, 144, 100), outline=(90, 90, 90), width=3)


def draw_pants(draw, rgb):
    draw.rectangle((98, 70, 158, 110), fill=rgb)
    draw.rectangle((98, 110, 124, 210), fill=rgb)
    draw.rectangle((132, 110, 158, 210), fill=rgb)


def draw_jacket(draw, rgb):
    draw.rectangle((92, 88, 164, 195), fill=rgb)
    draw.polygon([(92, 93), (56, 128), (74, 150), (92, 118)], fill=rgb)
    draw.polygon([(164, 93), (200, 128), (182, 150), (164, 118)], fill=rgb)
    draw.line((128, 88, 128, 195), fill=(255, 255, 255), width=3)
    draw.polygon([(112, 78), (128, 95), (144, 78)], outline=(90, 90, 90), width=3)


def draw_shoes(draw, rgb):
    draw.ellipse((70, 150, 190, 190), fill=rgb)
    draw.rectangle((70, 130, 110, 172), fill=rgb)


def render(item):
    pid, category, _color, rgb, _sleeve, _mat, pattern = item
    img = Image.new("RGB", (SIZE, SIZE), BG)
    d = ImageDraw.Draw(img)
    outline = tuple(max(0, c - 40) for c in rgb) if rgb != (250, 250, 250) else (170, 170, 170)
    if category == "DRESSES":
        draw_dress(d, rgb, pattern)
    elif category == "SHIRTS":
        draw_shirt(d, rgb, pattern)
    elif category == "PANTS":
        draw_pants(d, rgb)
    elif category == "JACKETS":
        draw_jacket(d, rgb)
    elif category == "SHOES":
        draw_shoes(d, rgb)
    d.rectangle((4, 4, SIZE - 4, SIZE - 4), outline=(210, 214, 220), width=2)
    out = IMAGES / f"{pid}.jpg"
    img.save(out, quality=92)
    return out.relative_to(HERE)


def clause_for(category, color, sleeve, material, pattern):
    noun = {"DRESSES": "dress", "SHIRTS": "shirt", "PANTS": "pants",
            "JACKETS": "jacket", "SHOES": "shoes"}[category]
    bits = []
    if sleeve != "-":
        bits.append(f"{sleeve} sleeve")
    bits.append(material)
    bits.append(noun)
    lead = " ".join(bits).capitalize()
    tail = f" in {color}." if pattern == "solid" else f", striped in {color}."
    return lead + tail


def main():
    rows = []
    for item in ITEMS:
        pid, category, color, rgb, sleeve, material, pattern = item
        image_rel = render(item)
        clause = clause_for(category, color, sleeve, material, pattern)
        base = dict(row_id=f"{pid}-C01", sample_id=pid, product_id=pid, view_index=0,
                    stratum="demo", image_file=str(image_rel), category=category,
                    clause_id=f"{pid}-c1", family="", value="",
                    clause_text=clause, model_label="synthetic", model_confidence=1.0)
        rows.append(base)
        attrs = [("color", color)]
        if sleeve != "-":
            attrs.append(("sleeve", sleeve))
        attrs.append(("material", material))
        if pattern != "solid":
            attrs.append(("pattern", pattern))
        for k, (family, value) in enumerate(attrs, start=2):
            rows.append(dict(row_id=f"{pid}-C{k:02d}", sample_id=pid, product_id=pid,
                              view_index=0, stratum="demo", image_file=str(image_rel),
                              category=category, clause_id=f"{pid}-c{k}", family=family,
                              value=value, clause_text=clause, model_label="synthetic",
                              model_confidence=1.0))

    out_csv = HERE / "gold_manifest.csv"
    fieldnames = ["row_id", "sample_id", "product_id", "view_index", "stratum",
                  "image_file", "category", "clause_id", "family", "value",
                  "clause_text", "model_label", "model_confidence"]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(ITEMS)} synthetic items, {len(rows)} manifest rows -> {out_csv}")


if __name__ == "__main__":
    main()
