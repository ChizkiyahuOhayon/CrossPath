#!/usr/bin/env python3
"""Render original-image CrossPath retrieval cases for the paper."""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "paper_assets"
FIGURES = ASSETS / "figures"
TARGET_COLOR = "#009E73"
BASE_COLOR = "#6C7A89"
OURS_COLOR = "#D55E00"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_report(dataset):
    if dataset == "cirr":
        root = ROOT / "results" / "e25_cirr" / "qualitative_export"
        path = root / "cirr_retrieval_cases.json"
    else:
        root = ASSETS
        path = ASSETS / "data" / f"{dataset}_retrieval_cases.json"
    report = json.loads(path.read_text())
    report["_bundle_root"] = str(root)
    return report


def primary_image(report, image_id):
    paths = report["source_images"].get(image_id)
    if not paths:
        raise KeyError(f"image {image_id} is absent from the case bundle")
    relative_path = paths[0] if isinstance(paths, list) else paths
    return Path(report["_bundle_root"]) / relative_path


def draw_image(ax, path, title, border="#D0D0D0", linewidth=0.8):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))
    ax.imshow(pixels)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=6.7, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border)
        spine.set_linewidth(linewidth)


def draw_ranked_row(spec, report, image_ids, target_id, label, color):
    grid = GridSpecFromSubplotSpec(
        1,
        len(image_ids) + 1,
        subplot_spec=spec,
        width_ratios=(0.58,) + (1,) * len(image_ids),
        wspace=0.08,
    )
    label_ax = plt.subplot(grid[0, 0])
    label_ax.axis("off")
    label_ax.text(
        0.96,
        0.5,
        label,
        color=color,
        fontweight="bold",
        fontsize=6.8,
        ha="right",
        va="center",
        transform=label_ax.transAxes,
    )
    for rank, image_id in enumerate(image_ids, start=1):
        ax = plt.subplot(grid[0, rank])
        is_target = image_id == target_id
        draw_image(
            ax,
            primary_image(report, image_id),
            f"#{rank}",
            border=TARGET_COLOR if is_target else "#D0D0D0",
            linewidth=2.2 if is_target else 0.8,
        )
def render_dataset(dataset):
    report = load_report(dataset)
    cases = report["cases"][:3] if dataset == "cirr" else report["cases"]
    figure_height = 4.1 if dataset == "cirr" else 7.2
    fig = plt.figure(figsize=(6.9, figure_height), facecolor="white")
    outer = GridSpec(
        len(cases),
        1,
        figure=fig,
        hspace=0.34,
        top=0.93,
        bottom=0.035,
        left=0.05,
        right=0.985,
    )
    for case_number, (case, slot) in enumerate(zip(cases, outer), start=1):
        case_grid = GridSpecFromSubplotSpec(
            3,
            3,
            subplot_spec=slot,
            height_ratios=(0.62, 1, 1),
            width_ratios=(0.82, 0.82, 4.7),
            hspace=0.28,
            wspace=0.13,
        )
        title_ax = fig.add_subplot(case_grid[0, :])
        title_ax.axis("off")
        category_value = case.get("category")
        category = "" if category_value is None else f"{category_value.title()} · "
        prefix = (
            f"Case {case_number} · {category}target rank "
            f"{case['base_rank']} → {case['method_rank']}"
        )
        modification = case.get("modification", case.get("caption"))
        wrapped = textwrap.fill(modification, width=96)
        source_ax = fig.add_subplot(case_grid[1:, 0])
        draw_image(
            source_ax,
            primary_image(report, case["source_id"]),
            "Reference",
            border="#555555",
            linewidth=1.1,
        )
        target_ax = fig.add_subplot(case_grid[1:, 1])
        draw_image(
            target_ax,
            primary_image(report, case["target_id"]),
            "Target",
            border=TARGET_COLOR,
            linewidth=2.2,
        )
        draw_ranked_row(
            case_grid[1, 2],
            report,
            case["base_top_ids"],
            case["target_id"],
            "A1" if dataset == "fashiongen" else "MCoT",
            BASE_COLOR,
        )
        draw_ranked_row(
            case_grid[2, 2],
            report,
            case["method_top_ids"],
            case["target_id"],
            "CrossPath",
            OURS_COLOR,
        )
        title_box = title_ax.get_position(fig)
        overlay = fig.add_axes(title_box.bounds, frameon=False, zorder=100)
        overlay.patch.set_alpha(0)
        overlay.axis("off")
        overlay.text(
            0.0,
            1.0,
            prefix,
            fontsize=8.2,
            fontweight="bold",
            va="top",
            clip_on=False,
        )
        overlay.text(
            0.0,
            0.61,
            f'Text: “{wrapped}”',
            fontsize=7.2,
            va="top",
            clip_on=False,
        )

    display_name = {
        "fashiongen": "FashionGen",
        "fashioniq": "FashionIQ",
        "cirr": "CIRR",
    }[dataset]
    fig.suptitle(
        f"CrossPath retrieval examples on {display_name}",
        y=0.982,
        fontsize=11,
        fontweight="bold",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / f"fig_{dataset}_retrieval_cases"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def render_combined():
    datasets = ("fashiongen", "fashioniq", "cirr")
    reports = {dataset: load_report(dataset) for dataset in datasets}
    entries = [
        (dataset, reports[dataset], case)
        for dataset in datasets
        for case in reports[dataset]["cases"][:1]
    ]
    fig = plt.figure(figsize=(6.9, 4.15), facecolor="white")
    outer = GridSpec(
        len(entries),
        1,
        figure=fig,
        hspace=0.34,
        top=0.94,
        bottom=0.03,
        left=0.05,
        right=0.985,
    )
    for case_number, ((dataset, report, case), slot) in enumerate(
        zip(entries, outer), start=1
    ):
        case_grid = GridSpecFromSubplotSpec(
            3,
            3,
            subplot_spec=slot,
            height_ratios=(0.62, 1, 1),
            width_ratios=(0.82, 0.82, 4.7),
            hspace=0.28,
            wspace=0.13,
        )
        title_ax = fig.add_subplot(case_grid[0, :])
        title_ax.axis("off")
        category_value = case.get("category")
        category = "" if category_value is None else f"{category_value.title()} · "
        display_name = {
            "fashiongen": "FashionGen",
            "fashioniq": "FashionIQ",
            "cirr": "CIRR",
        }[dataset]
        prefix = (
            f"{display_name} · Case {case_number} · {category}target rank "
            f"{case['base_rank']} → {case['method_rank']}"
        )
        modification = case.get("modification", case.get("caption"))
        wrapped = textwrap.fill(modification, width=160)
        source_ax = fig.add_subplot(case_grid[1:, 0])
        draw_image(
            source_ax,
            primary_image(report, case["source_id"]),
            "Reference",
            border="#555555",
            linewidth=1.1,
        )
        target_ax = fig.add_subplot(case_grid[1:, 1])
        draw_image(
            target_ax,
            primary_image(report, case["target_id"]),
            "Target",
            border=TARGET_COLOR,
            linewidth=2.2,
        )
        draw_ranked_row(
            case_grid[1, 2],
            report,
            case["base_top_ids"],
            case["target_id"],
            "A1" if dataset == "fashiongen" else "MCoT",
            BASE_COLOR,
        )
        draw_ranked_row(
            case_grid[2, 2],
            report,
            case["method_top_ids"],
            case["target_id"],
            "CrossPath",
            OURS_COLOR,
        )
        title_box = title_ax.get_position(fig)
        overlay = fig.add_axes(title_box.bounds, frameon=False, zorder=100)
        overlay.patch.set_alpha(0)
        overlay.axis("off")
        overlay.text(
            0.0,
            1.0,
            prefix,
            fontsize=8.0,
            fontweight="bold",
            va="top",
            clip_on=False,
        )
        overlay.text(
            0.0,
            0.61,
            f'Text: “{wrapped}”',
            fontsize=7.0,
            va="top",
            clip_on=False,
        )
    fig.suptitle(
        "CrossPath retrieval examples across three benchmarks",
        y=0.982,
        fontsize=11,
        fontweight="bold",
    )
    stem = FIGURES / "fig_crosspath_retrieval_cases"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def main():
    render_dataset("fashiongen")
    render_dataset("fashioniq")
    render_dataset("cirr")
    render_combined()


if __name__ == "__main__":
    main()
