#!/usr/bin/env python3
"""Generate the paper-facing CrossPath compatibility and ablation figures."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper_assets"
FIGURES = OUTPUT / "figures"
DATA = OUTPUT / "data"
ENDPOINTS = ("dqu_base", "dqu_gradcache", "mcot")
ENDPOINT_LABELS = ("DQU Base", "DQU GradCache", "MCoT")
BASELINE_COLOR = "#B0BEC5"
MATCHED_COLOR = "#56B4E9"
OURS_COLOR = "#D55E00"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_json(path):
    return json.loads(Path(path).read_text())


def save_figure(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def compatibility_matrices(report, metric):
    values = report["average"]
    return np.asarray(
        [
            [values[f"{query}__{gallery}"][metric] for gallery in ENDPOINTS]
            for query in ENDPOINTS
        ]
    )


def draw_compatibility_heatmap():
    report = load_json(
        ROOT / "results/e27_fashioniq_matrix3/fashioniq_matrix3.json"
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.75), constrained_layout=True)
    matrices = {
        "R@10": compatibility_matrices(report, "R@10"),
        "R@50": compatibility_matrices(report, "R@50"),
    }
    for ax, (metric, matrix) in zip(axes, matrices.items()):
        minimum = matrix.min() - 0.2
        maximum = matrix.max()
        image = ax.imshow(matrix, cmap="YlOrRd", vmin=minimum, vmax=maximum)
        for row in range(3):
            for column in range(3):
                is_crosspath = row == 2 and column in (0, 1)
                normalized = (matrix[row, column] - minimum) / (maximum - minimum)
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold" if is_crosspath else "normal",
                    color="white" if normalized > 0.68 else "#111111",
                )
                if is_crosspath:
                    ax.add_patch(
                        plt.Rectangle(
                            (column - 0.48, row - 0.48),
                            0.96,
                            0.96,
                            fill=False,
                            edgecolor=OURS_COLOR,
                            linewidth=2.0,
                        )
                    )
        ax.set_xticks(range(3), ENDPOINT_LABELS, rotation=18, ha="right")
        ax.set_yticks(range(3), ENDPOINT_LABELS)
        ax.set_xlabel("Gallery encoder")
        ax.set_ylabel("Query encoder")
        ax.set_title(metric)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.025)
        colorbar.set_label("Recall (%)")
    fig.suptitle(
        "Directional query-gallery compatibility on FashionIQ",
        fontsize=11,
        fontweight="bold",
    )
    save_figure(fig, "fig_fashioniq_compatibility_matrix")

    with (DATA / "fashioniq_compatibility_matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("query_encoder", "gallery_encoder", "R@10", "R@50"))
        for row, query in enumerate(ENDPOINTS):
            for column, gallery in enumerate(ENDPOINTS):
                writer.writerow(
                    (
                        query,
                        gallery,
                        f"{matrices['R@10'][row, column]:.6f}",
                        f"{matrices['R@50'][row, column]:.6f}",
                    )
                )


def metric_sum(metrics, names):
    return sum(metrics[name] for name in names)


def positive_ablation_data():
    fashiongen = load_json(ROOT / "results/fashiongen_cross_matrix_summary.json")
    routed = load_json(ROOT / "results/fashiongen_joint_matrix_official_manifest.json")
    fashioniq = load_json(
        ROOT / "results/e27_fashioniq_matrix3/fashioniq_matrix3.json"
    )
    fg_metrics = fashiongen["metrics"]
    fg_names = ("R@1", "R@5", "R@10")
    fiq_metrics = fashioniq["average"]
    fiq_names = ("R@10", "R@50")
    return {
        "FashionGen": [
            ("Single A1", metric_sum(fg_metrics["q0_g0"], fg_names)),
            ("Diagonal\nensemble", metric_sum(fg_metrics["diagonal_mean"], fg_names)),
            ("Off-diagonal\nCrossPath", metric_sum(fg_metrics["cross_mean"], fg_names)),
            (
                "CrossPath\n+ routing",
                metric_sum(routed["report"]["metrics"], fg_names),
            ),
        ],
        "FashionIQ": [
            ("Single MCoT", metric_sum(fiq_metrics["mcot__mcot"], fiq_names)),
            (
                "Diagonal\nensemble",
                metric_sum(
                    fiq_metrics["dqu_gradcache+mcot__diagonal_mean"], fiq_names
                ),
            ),
            (
                "Off-diagonal\nCrossPath",
                metric_sum(
                    fiq_metrics["dqu_gradcache+mcot__cross_mean"], fiq_names
                ),
            ),
        ],
    }


def draw_positive_ablation():
    panels = positive_ablation_data()
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.65), constrained_layout=True)
    for ax, (dataset, rows) in zip(axes, panels.items()):
        labels = [row[0] for row in rows]
        scores = [row[1] for row in rows]
        colors = [BASELINE_COLOR, MATCHED_COLOR] + [OURS_COLOR] * (len(rows) - 2)
        bars = ax.bar(range(len(rows)), scores, color=colors, width=0.68)
        lower = min(scores) - (1.0 if dataset == "FashionGen" else 0.6)
        upper = max(scores) + (0.75 if dataset == "FashionGen" else 0.45)
        ax.set_ylim(lower, upper)
        ax.set_xticks(range(len(rows)), labels)
        ax.set_ylabel("Sum of official recalls")
        ax.set_title(dataset)
        ax.grid(axis="y", alpha=0.18, linewidth=0.6)
        for index, (bar, score) in enumerate(zip(bars, scores)):
            delta = "" if index == 0 else f"\n(+{score - scores[index - 1]:.2f})"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                score + (upper - lower) * 0.025,
                f"{score:.2f}{delta}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold" if index == len(rows) - 1 else "normal",
            )
    fig.suptitle(
        "Cross-compatible paths improve cost-matched endpoint fusion",
        fontsize=11,
        fontweight="bold",
    )
    save_figure(fig, "fig_positive_module_ablation")

    with (DATA / "positive_module_ablation.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "configuration", "sum_recall", "step_delta"))
        for dataset, rows in panels.items():
            previous = None
            for label, score in rows:
                delta = "" if previous is None else f"{score - previous:.6f}"
                writer.writerow((dataset, label.replace("\n", " "), f"{score:.6f}", delta))
                previous = score


def write_main_tables():
    fashiongen = load_json(ROOT / "results/fashiongen_cross_matrix_summary.json")
    routed = load_json(ROOT / "results/fashiongen_joint_matrix_official_manifest.json")
    fashioniq = load_json(
        ROOT / "results/e27_fashioniq_matrix3/fashioniq_matrix3.json"
    )["average"]

    fg_rows = [
        ("ProCIR (paper)", "", "75.000000", "85.300000", "FashionMV Table 2"),
        (
            "A1 reproduced base",
            fashiongen["metrics"]["q0_g0"]["R@1"],
            fashiongen["metrics"]["q0_g0"]["R@5"],
            fashiongen["metrics"]["q0_g0"]["R@10"],
            "local artifact",
        ),
        (
            "Cost-matched diagonal ensemble",
            fashiongen["metrics"]["diagonal_mean"]["R@1"],
            fashiongen["metrics"]["diagonal_mean"]["R@5"],
            fashiongen["metrics"]["diagonal_mean"]["R@10"],
            "local artifact",
        ),
        (
            "CrossPath cross mean",
            fashiongen["metrics"]["cross_mean"]["R@1"],
            fashiongen["metrics"]["cross_mean"]["R@5"],
            fashiongen["metrics"]["cross_mean"]["R@10"],
            "local artifact",
        ),
        (
            "CrossPath joint routing",
            routed["report"]["metrics"]["R@1"],
            routed["report"]["metrics"]["R@5"],
            routed["report"]["metrics"]["R@10"],
            "local artifact",
        ),
    ]
    with (DATA / "main_results_fashiongen.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "R@1", "R@5", "R@10", "source"))
        writer.writerows(fg_rows)

    fiq_rows = [
        ("DQU-CIR (paper)", "62.000000", "81.580000", "DQU-CIR paper"),
        ("MCoT-MVS (paper)", "63.240000", "82.010000", "MCoT-MVS paper"),
        (
            "MCoT-MVS same pipeline",
            fashioniq["mcot__mcot"]["R@10"],
            fashioniq["mcot__mcot"]["R@50"],
            "local artifact",
        ),
        (
            "Cost-matched diagonal ensemble",
            fashioniq["dqu_gradcache+mcot__diagonal_mean"]["R@10"],
            fashioniq["dqu_gradcache+mcot__diagonal_mean"]["R@50"],
            "local artifact",
        ),
        (
            "CrossPath DQU Base x MCoT",
            fashioniq["dqu_base+mcot__cross_mean"]["R@10"],
            fashioniq["dqu_base+mcot__cross_mean"]["R@50"],
            "local artifact",
        ),
        (
            "CrossPath DQU GradCache x MCoT",
            fashioniq["dqu_gradcache+mcot__cross_mean"]["R@10"],
            fashioniq["dqu_gradcache+mcot__cross_mean"]["R@50"],
            "local artifact",
        ),
    ]
    with (DATA / "main_results_fashioniq.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "R@10", "R@50", "source"))
        writer.writerows(fiq_rows)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    write_main_tables()
    draw_compatibility_heatmap()
    draw_positive_ablation()


if __name__ == "__main__":
    main()
