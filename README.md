# CrossPath

CrossPath is research code for combining two frozen composed-image-retrieval (CIR) endpoints through cross-compatible query/gallery paths and conservative, cutoff-aware routing.

This repository is a public research snapshot. It contains the method implementation, tests, experiment commands, machine-readable result artifacts, and the append-only E0–E28 experiment log. Large endpoint checkpoints and third-party datasets are not redistributed because of size and licensing constraints.

## Current results

| Benchmark and protocol | Method | Metrics (%) |
|---|---|---|
| FashionGen-val / FashionMV official gallery | Base endpoint | R@1 42.73 / R@5 79.26 / R@10 87.75 |
| FashionGen-val / FashionMV official gallery | CrossPath joint routing | **R@1 44.09 / R@5 80.36 / R@10 88.28** |
| FashionIQ / val-split | MCoT-MVS same-pipeline endpoint | R@10 63.56 / R@50 82.33 |
| FashionIQ / val-split | DQU Base × MCoT cross mean | **R@10 64.61** / R@50 82.57 |
| FashionIQ / val-split | DQU GradCache × MCoT cross mean | R@10 64.58 / **R@50 82.82** |

Under a uniform source-exclusion evaluator, heterogeneous CrossPath improves the same-pipeline MCoT-MVS endpoint by +1.02 R@10 and +0.48 R@50. An author-code-aligned evaluation that retains the source image gives 64.16/82.68, versus 62.95/82.14 for MCoT-MVS in the identical embedding evaluator. Full comparisons and protocol notes are in [03_MAIN_TABLE.md](03_MAIN_TABLE.md).

A coordinate-scrambling control preserves both endpoint rankings to numerical precision while reducing the FashionIQ cross path from 64.58/82.82 to 0.85/3.08 R@10/R@50. This isolates the result to genuine cross-coordinate compatibility rather than endpoint ensembling alone.

## Repository contents

- `weave_crosspath.py`: scale-invariant rank paths, boundary traces, and utility contracts.
- `weave_crosspath_gate.py`: candidate responsibility gate and listwise loss.
- `weave_build_crosspath_cache.py`: cache construction for matched, cross, and joint paths.
- `weave_train_crosspath_gate.py` / `weave_eval_crosspath_gate.py`: training, calibration, and frozen evaluation.
- `weave_train_crosspath_adapter.py`: the E18 frozen-embedding residual-adapter pilot.
- `weave_train_relational_crosspath.py`: the E19 single-endpoint relational-composer pilot.
- `weave_extract_dqu_branches.py` / `weave_train_composition_crosspath.py`: the E20–E21 full-gallery composition experiments.
- `weave_extract_crosspath_*.py`: endpoint embedding export for FashionMV/ProCIR and FashionIQ/DQU-CIR.
- `scripts/`: exact experiment orchestration, zero-parameter evaluators, and regression tests.
- `results/`: table-level JSON manifests and NPZ evaluation artifacts.
- `paper_assets/`: main-table CSVs, deterministic case manifests, camera-ready
  figures, and an editable CrossPath framework diagram.
- `experiment.md`: append-only E0–E28 experiment record, including rejected variants.
- `test_weave_*.py`: method and protocol regression tests.

## Quick verification

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/verify_release.py
```

The unit tests exercise the model-agnostic CrossPath implementation. Full benchmark reproduction additionally requires the upstream datasets, endpoint repositories, and endpoint checkpoints described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Upstream projects

- [FashionMV / ProCIR](https://github.com/yuandaxia2001/FashionMV)
- [DQU-CIR](https://github.com/iLearn-Lab/SIGIR24-DQU-CIR)
- [FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq)

CrossPath does not vendor code, model weights, or data from these projects. Follow their licenses and dataset terms.

## Status

Work in progress for a future academic submission. Results are reported from saved artifacts rather than reconstructed from prose; rejected experiments remain in the log to prevent accidental cherry-picking or duplicated runs.

## License

The original CrossPath code in this repository is released under the MIT License. Third-party datasets, checkpoints, and upstream code retain their own licenses.
