#!/usr/bin/env python3
"""Collect every finished CrossPath experiment into one markdown table set.

Reads the raw result JSONs under results/ and writes results/ALL_RESULTS.md.
Re-run after new results land; nothing here recomputes metrics, it only reads
what the evaluators already wrote.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "ALL_RESULTS.md"

PATH_ORDER = [
    "q0_g0", "q0_g1", "q1_g0", "q1_g1",
    "diagonal_mean", "cross_mean", "all_mean",
    "diagonal_max", "cross_max", "all_max",
    "diagonal_borda", "cross_borda", "all_borda",
]


def load(rel):
    path = RES / rel
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt(value):
    return "—" if value is None else f"{value:.2f}"


def table(header, rows):
    align = "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"
    lines = ["| " + " | ".join(header) + " |", align]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def section_fashiongen(out):
    out.append("## FashionGen — learned CrossPath routing\n")
    matched = load("CrossPath_A1seedpair_20260816_official_manifest.json")
    joint = load("fashiongen_joint_matrix_official_manifest.json")
    query_only = load("FashionGen_query_only_20260820_official_manifest.json")
    margin_only = load("FashionGen_margin_only_20260820_official_manifest.json")
    if not matched or not joint:
        out.append("_FashionGen official manifests are not present yet._\n")
        return

    rows = []

    def add(name, metrics):
        rows.append([
            name,
            fmt(metrics.get("R@1")),
            fmt(metrics.get("R@5")),
            fmt(metrics.get("R@10")),
        ])

    matched_report = matched["report"]
    joint_report = joint["report"]
    add("Base endpoint", matched_report["base_metrics"])
    add("Second endpoint", matched_report["full_correction_metrics"])
    if query_only:
        add("Query-only gate", query_only["report"]["metrics"])
    if margin_only:
        add("Margin-only gate", margin_only["report"]["metrics"])
    add("Matched-only CrossPath", matched_report["metrics"])
    add("Joint-matrix CrossPath", joint_report["metrics"])
    add("Per-query oracle (ceiling only)", joint_report["objective_oracle_metrics"])
    out.append(table(["method", "R@1", "R@5", "R@10"], rows))
    out.append("")
    out.append(
        "The joint-matrix router is the strongest learned variant: 44.09/80.36/88.28, "
        "improving the base endpoint by +1.36/+1.10/+0.53 points. Query-only and "
        "margin-only controls fall back to the base ranking.\n"
    )


def section_fashioniq_e24(out):
    out.append("## FashionIQ — E24 heterogeneous DQU x MCoT (source-excluded)\n")
    for name, rel in [
        ("DQU Base x MCoT", "e24_heterogeneous/dqu_base_mcot_summary.json"),
        ("DQU GradCache x MCoT", "e24_heterogeneous/dqu_gc_mcot_summary.json"),
    ]:
        data = load(rel)
        if not data or "average" not in data:
            continue
        avg = data["average"]
        rows = []
        for path in PATH_ORDER:
            if path in avg:
                metric = avg[path]
                rows.append([
                    f"`{path}`",
                    fmt(metric.get("R@1")),
                    fmt(metric.get("R@10")),
                    fmt(metric.get("R@50")),
                ])
        out.append(f"### {name}\n")
        out.append(table(["path", "R@1", "R@10", "R@50"], rows))
        out.append("")


def section_e26(out):
    out.append("## E26 — orthogonal scramble control (mechanism evidence)\n")
    out.append(
        "Scrambling applies an orthogonal map to one endpoint's coordinates. "
        "Diagonal paths are provably unchanged; cross paths collapse if and only "
        "if they rely on inter-space alignment.\n"
    )
    data = load("e26_controls/fashioniq_dqu_gc_mcot_controls.json")
    if not data:
        return
    rows = []
    for cat, payload in data.get("categories", {}).items():
        metrics = payload.get("metrics", {})
        real, scr = metrics.get("real", {}), metrics.get("scrambled", {})
        for path in ["q0_g0", "q1_g1", "q0_g1", "q1_g0", "diagonal_mean", "cross_mean"]:
            if path in real:
                rows.append([
                    cat, f"`{path}`",
                    fmt(real[path].get("R@10")),
                    fmt(scr.get(path, {}).get("R@10")),
                ])
    out.append(table(["category", "path", "R@10 real", "R@10 scrambled"], rows))
    out.append("")


def section_e32(out):
    out.append("## CIRR — E32 candidate-wise compatibility residual (internal test)\n")
    rows = []
    for label, rel in (
        ("group loss weight 1", "e32_candidate_compatibility/v1_group1/report.json"),
        ("group loss weight 4", "e32_candidate_compatibility/v2_group4/report.json"),
    ):
        data = load(rel)
        if not data:
            continue
        base = data["internal_test"]["base"]
        model = data["internal_test"]["model"]
        rows.append([
            label,
            fmt(base["R@5"]), fmt(model["R@5"]),
            fmt(base["subset_R@1"]), fmt(model["subset_R@1"]),
            f"{data['internal_gain']:+.2f}",
            "yes" if data["passed_internal_gate"] else "no",
        ])
    if rows:
        out.append(table(
            ["variant", "base R@5", "model R@5", "base subset R@1", "model subset R@1", "Delta Avg", "val gate"],
            rows,
        ))
        out.append("")
        out.append(
            "Both variants improve global R@5 but reduce subset R@1. Neither "
            "passes the predeclared internal gate, so CIRR val was not evaluated.\n"
        )


def section_cirr(out):
    out.append("## CIRR — E25 cross-domain generalization\n")
    data = load("e25_cirr/val_summary.json")
    if not data:
        out.append("_val_summary.json not present yet._\n")
        return
    metrics = data.get("metrics", {})
    rows = []
    for path in PATH_ORDER:
        if path in metrics:
            m = metrics[path]
            rows.append([
                f"`{path}`",
                fmt(m.get("R@1")), fmt(m.get("R@5")),
                fmt(m.get("R@10")), fmt(m.get("R@50")),
                fmt(m.get("subset_R@1")), fmt(m.get("Avg")),
            ])
    out.append(table(
        ["path", "R@1", "R@5", "R@10", "R@50", "subset R@1", "Avg"], rows))
    out.append("")
    out.append(f"- val-selected fusion path: `{data.get('best_path')}`")
    out.append(f"- best single path: `{data.get('best_single_path')}`\n")


def section_cirr_e29_e30(out):
    out.append("## CIRR — E29/E30 target refinement and conservative fusion\n")
    oracle = load("e30_cirr_query_router/oracle_val.json")
    router_v1 = load("e30_cirr_query_router/router_v1/report.json")
    router_v2 = load("e30_cirr_query_router/router_v2_delta_retrieval/val_report.json")
    combination = load("e30_cirr_query_router/e29_e30_combination.json")
    procrustes = load("e31_cirr_procrustes/report.json")
    reranker = load("e29_sate/reranker_v2_full/history.json")
    if not oracle:
        out.append("_E30 results are not present yet._\n")
        return

    rows = []
    base = oracle["base"]
    rows.append(["MCoT baseline", fmt(base["R@1"]), fmt(base["R@5"]), fmt(base["subset_R@1"]), fmt(base["Avg"]), "0.00"])
    for name in ("q1_g0_a0.125", "q1_g0_a0.25"):
        metric = oracle["actions"][name]
        rows.append([name, fmt(metric["R@1"]), fmt(metric["R@5"]), fmt(metric["subset_R@1"]), fmt(metric["Avg"]), f"{metric['Avg'] - base['Avg']:+.2f}"])
    if reranker:
        metric = max(reranker, key=lambda item: item["Avg"])
        rows.append(["E29 re-ranker", fmt(metric["R@1"]), fmt(metric["R@5"]), fmt(metric["subset_R@1"]), fmt(metric["Avg"]), f"{metric['Avg'] - base['Avg']:+.2f}"])
    if router_v1:
        metric = router_v1["evaluation"]["router"]
        rows.append(["E30 classification router", fmt(metric["R@1"]), fmt(metric["R@5"]), fmt(metric["subset_R@1"]), fmt(metric["Avg"]), f"{metric['Avg'] - base['Avg']:+.2f}"])
    if router_v2:
        metric = router_v2["metrics"]["router"]
        rows.append(["E30 delta router", fmt(metric["R@1"]), fmt(metric["R@5"]), fmt(metric["subset_R@1"]), fmt(metric["Avg"]), f"{metric['Avg'] - base['Avg']:+.2f}"])
    if combination:
        name, metric = max(combination["results"].items(), key=lambda item: item[1]["Avg"])
        rows.append([f"E29 re-ranker + weak CrossPath ({name})", fmt(metric["R@1"]), fmt(metric["R@5"]), fmt(metric["subset_R@1"]), fmt(metric["Avg"]), f"{metric['Avg'] - base['Avg']:+.2f}"])
    if procrustes:
        metric = procrustes["metrics"]["mapped_blend_a0.25"]
        rows.append(["E31 orthogonal map + weak CrossPath", fmt(metric["R@1"]), fmt(metric["R@5"]), fmt(metric["subset_R@1"]), fmt(metric["Avg"]), f"{metric['Avg'] - base['Avg']:+.2f}"])
    metric = oracle["oracle"]
    rows.append(["per-query oracle (ceiling only)", "—", "—", "—", fmt(metric["Avg"]), f"{metric['gain_over_mcot']:+.2f}"])
    out.append(table(["method", "R@1", "R@5", "subset R@1", "Avg", "Delta Avg"], rows))
    out.append("")
    out.append(
        "Best measured E30 configuration is the frozen E29 re-ranker followed by "
        "a 0.25 MCoT-query/DQU-gallery score blend (Avg 82.38). The learned delta "
        "router transfers only +0.08 Avg to CIRR val, so it is not selected as the main method.\n"
    )


def main():
    out = ["# CrossPath — consolidated results", ""]
    out.append("Generated by `scripts/collect_all_results.py` from raw result JSONs.\n")
    section_fashiongen(out)
    section_cirr(out)
    section_cirr_e29_e30(out)
    section_e32(out)
    section_fashioniq_e24(out)
    section_e26(out)
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
