#!/usr/bin/env python3
"""Verify the public CrossPath experiment snapshot."""

import json
import re
import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REQUIRED = (
    RESULTS / "fashiongen_joint_matrix_official_manifest.json",
    RESULTS / "fashiongen_joint_matrix_gate_manifest.json",
    RESULTS / "fashiongen_joint_matrix_evaluation.npz",
    RESULTS / "fashioniq_valsplit_cross_matrix_summary.json",
    RESULTS / "fashioniq_original" / "cross_matrix_summary.json",
    RESULTS / "e24_heterogeneous" / "dqu_base_mcot_summary.json",
    RESULTS / "e24_heterogeneous" / "dqu_gc_mcot_summary.json",
    RESULTS / "e24_heterogeneous" / "dqu_base_mcot_include_source_summary.json",
    RESULTS / "e24_heterogeneous" / "dqu_gc_mcot_include_source_summary.json",
    RESULTS / "e26_controls" / "fashiongen_controls.json",
    RESULTS / "e26_controls" / "fashioniq_dqu_gc_mcot_controls.json",
    RESULTS / "e27_fashioniq_matrix3" / "fashioniq_matrix3.json",
)
PAPER_ASSETS = ROOT / "paper_assets"
PAPER_REQUIRED = (
    PAPER_ASSETS / "data" / "fashiongen_retrieval_cases.json",
    PAPER_ASSETS / "data" / "fashioniq_retrieval_cases.json",
    PAPER_ASSETS / "data" / "fashioniq_compatibility_matrix.csv",
    PAPER_ASSETS / "data" / "positive_module_ablation.csv",
    PAPER_ASSETS / "data" / "main_results_fashiongen.csv",
    PAPER_ASSETS / "data" / "main_results_fashioniq.csv",
    PAPER_ASSETS / "figures" / "fig_fashiongen_retrieval_cases.pdf",
    PAPER_ASSETS / "figures" / "fig_fashiongen_retrieval_cases.png",
    PAPER_ASSETS / "figures" / "fig_fashioniq_retrieval_cases.pdf",
    PAPER_ASSETS / "figures" / "fig_fashioniq_retrieval_cases.png",
    PAPER_ASSETS / "figures" / "fig_fashioniq_compatibility_matrix.pdf",
    PAPER_ASSETS / "figures" / "fig_fashioniq_compatibility_matrix.png",
    PAPER_ASSETS / "figures" / "fig_positive_module_ablation.pdf",
    PAPER_ASSETS / "figures" / "fig_positive_module_ablation.png",
    PAPER_ASSETS / "figures" / "fig_crosspath_framework.svg",
    PAPER_ASSETS / "figures" / "fig_crosspath_framework.pdf",
    PAPER_ASSETS / "figures" / "fig_crosspath_framework.png",
    PAPER_ASSETS / "figures" / "fig_crosspath_framework.vsdx",
    PAPER_ASSETS / "figures" / "fig_crosspath_framework.pptx",
    PAPER_ASSETS / "figures" / "fig_crosspath_framework.tex",
)


def verify_control_artifact(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    preserved_paths = ("q0_g0", "q1_g1", "diagonal_mean")
    for category, report in payload["categories"].items():
        if not report["preserved_path_rankings_identical"]:
            raise RuntimeError(f"failed scrambling invariant: {path.name}/{category}")
        metrics = report["metrics"]
        for ranking in preserved_paths:
            if metrics["real"][ranking] != metrics["scrambled"][ranking]:
                raise RuntimeError(
                    f"changed preserved metrics: {path.name}/{category}/{ranking}"
                )


def verify_matrix3_artifact(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    endpoints = tuple(payload["endpoints"])
    raw_paths = {f"{query}__{gallery}" for query in endpoints for gallery in endpoints}
    pairs = (
        (endpoints[0], endpoints[1]),
        (endpoints[0], endpoints[2]),
        (endpoints[1], endpoints[2]),
    )
    fused_paths = {
        f"{left}+{right}__{fusion}"
        for left, right in pairs
        for fusion in ("diagonal_mean", "cross_mean", "all_mean")
    }
    expected = raw_paths | fused_paths
    if set(payload["average"]) != expected:
        raise RuntimeError(f"incomplete 3x3 matrix: {path.relative_to(ROOT)}")
    for category, metrics in payload["categories"].items():
        if set(metrics) != expected:
            raise RuntimeError(f"incomplete 3x3 matrix category: {category}")


def verify_case_manifest(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if len(cases) != 3 or len({case["query_index"] for case in cases}) != 3:
        raise RuntimeError(f"retrieval cases must contain three unique queries: {path}")
    for case in cases:
        if not case["method_rank"] < case["base_rank"]:
            raise RuntimeError(f"non-improving retrieval case: {path}/{case['query_index']}")
        if case["method_rank"] > case["selection_cutoff"]:
            raise RuntimeError(f"case misses its selection cutoff: {path}")
        if case["target_id"] not in case["method_top_ids"]:
            raise RuntimeError(f"target absent from displayed CrossPath results: {path}")
        displayed_rank = case["method_top_ids"].index(case["target_id"]) + 1
        if displayed_rank != case["method_rank"]:
            raise RuntimeError(
                f"displayed target rank differs from exact rank: "
                f"{path}/{case['query_index']} ({displayed_rank} != "
                f"{case['method_rank']})"
            )
        required_ids = (
            [case["source_id"], case["target_id"]]
            + case["base_top_ids"]
            + case["method_top_ids"]
        )
        if any(not payload["source_images"].get(image_id) for image_id in required_ids):
            raise RuntimeError(f"case image missing from portable manifest: {path}")


def verify_figure(path):
    data = path.read_bytes()
    if path.suffix == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise RuntimeError(f"invalid PDF figure: {path.relative_to(ROOT)}")
        return
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) < 24:
        raise RuntimeError(f"invalid PNG figure: {path.relative_to(ROOT)}")
    width, height = struct.unpack(">II", data[16:24])
    if min(width, height) < 1500:
        raise RuntimeError(f"paper PNG is below high-resolution target: {width}x{height}")


def verify_editable_figure(path):
    data = path.read_bytes()
    if path.suffix == ".svg" and not data.lstrip().startswith(b"<svg"):
        raise RuntimeError(f"invalid SVG figure: {path.relative_to(ROOT)}")
    if path.suffix in (".pptx", ".vsdx") and not data.startswith(b"PK"):
        raise RuntimeError(f"invalid OOXML figure: {path.relative_to(ROOT)}")


def main():
    log = (ROOT / "experiment.md").read_text(encoding="utf-8")
    observed = {int(value) for value in re.findall(r"^## E(\d+)\b", log, re.M)}
    expected = set(range(29))
    if observed != expected:
        raise RuntimeError(
            f"experiment log mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    missing = [
        str(path.relative_to(ROOT))
        for path in REQUIRED + PAPER_REQUIRED
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing headline artifacts: {missing}")

    json_files = sorted(RESULTS.rglob("*.json"))
    npz_files = sorted(RESULTS.rglob("*.npz"))
    if not json_files or not npz_files:
        raise RuntimeError("result snapshot must contain JSON and NPZ artifacts")

    for path in json_files:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload in ({}, []):
            raise RuntimeError(f"empty JSON artifact: {path.relative_to(ROOT)}")

    for path in npz_files:
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise RuntimeError(f"empty NPZ artifact: {path.relative_to(ROOT)}")

    verify_control_artifact(RESULTS / "e26_controls" / "fashiongen_controls.json")
    verify_control_artifact(
        RESULTS / "e26_controls" / "fashioniq_dqu_gc_mcot_controls.json"
    )
    verify_matrix3_artifact(
        RESULTS / "e27_fashioniq_matrix3" / "fashioniq_matrix3.json"
    )
    verify_case_manifest(PAPER_ASSETS / "data" / "fashiongen_retrieval_cases.json")
    verify_case_manifest(PAPER_ASSETS / "data" / "fashioniq_retrieval_cases.json")
    for path in PAPER_REQUIRED:
        if path.suffix in (".pdf", ".png"):
            verify_figure(path)
        elif path.suffix in (".svg", ".pptx", ".vsdx"):
            verify_editable_figure(path)

    print(
        f"OK: E0-E28 present; {len(json_files)} JSON and "
        f"{len(npz_files)} NPZ artifacts readable; paper assets verified."
    )


if __name__ == "__main__":
    main()
