#!/usr/bin/env python3
"""Verify the public CrossPath experiment snapshot."""

import json
import re
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
)


def main():
    log = (ROOT / "experiment.md").read_text(encoding="utf-8")
    observed = {int(value) for value in re.findall(r"^## E(\d+)\b", log, re.M)}
    expected = set(range(23))
    if observed != expected:
        raise RuntimeError(
            f"experiment log mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
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

    print(
        f"OK: E0-E22 present; {len(json_files)} JSON and "
        f"{len(npz_files)} NPZ artifacts readable."
    )


if __name__ == "__main__":
    main()
