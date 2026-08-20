#!/usr/bin/env python3
"""Plot representative FashionGen target-rank paths from saved evaluation data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "results" / "CrossPath_A1seedpair_20260816_official_evaluation.npz"
OUT = Path(__file__).resolve().parent

EXAMPLES = (
    (11, 1, "R@1 recovery"),
    (48, 5, "R@5 recovery"),
    (917, 10, "R@10 recovery"),
)

BASE_COLOR = "#7B8794"
PATH_COLOR = "#0072B2"
OUR_COLOR = "#D55E00"
REGION_COLOR = "#009E73"


def main():
    with np.load(EVALUATION) as data:
        query_indices = data["query_indices"]
        target_ranks = data["target_ranks"]
        selected_actions = data["selected_actions"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linestyle": "-",
        }
    )

    actions = np.linspace(0.0, 1.0, target_ranks.shape[1])
    fig, axes = plt.subplots(1, len(EXAMPLES), figsize=(6.75, 2.25))

    for ax, (query_id, cutoff, title) in zip(axes, EXAMPLES):
        matches = np.flatnonzero(query_indices == query_id)
        if len(matches) != 1:
            raise ValueError(f"expected one row for query {query_id}, found {len(matches)}")
        row = int(matches[0])
        ranks = target_ranks[row]
        selected = int(selected_actions[row])

        ax.axhspan(0.5, cutoff + 0.5, color=REGION_COLOR, alpha=0.09, zorder=0)
        ax.axhline(cutoff, color=REGION_COLOR, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.plot(actions, ranks, color=PATH_COLOR, marker="o", markersize=3.8, linewidth=1.7)
        ax.scatter(actions[0], ranks[0], color=BASE_COLOR, s=30, zorder=4, label="Base")
        ax.scatter(
            actions[selected],
            ranks[selected],
            color=OUR_COLOR,
            marker="D",
            s=34,
            zorder=5,
            label="Selected",
        )
        ax.annotate(
            f"$m^*={selected}$",
            (actions[selected], ranks[selected]),
            xytext=(0, -14 if ranks[selected] <= cutoff else 10),
            textcoords="offset points",
            ha="center",
            color=OUR_COLOR,
            fontsize=8,
        )
        ax.set_title(title)
        ax.set_xlabel("Path action $\\alpha$")
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylim(max(int(ranks.max()) + 1, cutoff + 2), 0.5)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.text(
            0.98,
            0.05,
            f"rank {int(ranks[0])} $\\rightarrow$ {int(ranks[selected])}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#444444",
        )

    axes[0].set_ylabel("Target rank (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)
    fig.subplots_adjust(wspace=0.34)
    fig.savefig(OUT / "fig_rank_paths.pdf")
    fig.savefig(OUT / "fig_rank_paths.png", dpi=300)


if __name__ == "__main__":
    main()
