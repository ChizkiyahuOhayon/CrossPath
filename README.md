# CrossPath

CrossPath is research code for combining two frozen composed-image-retrieval (CIR) endpoints through cross-compatible query/gallery paths and conservative, cutoff-aware routing.

This repository is a public research snapshot. It contains the method implementation, tests, experiment commands, machine-readable result artifacts, and the append-only E0–E17 experiment log. Model checkpoints and third-party datasets are not redistributed because of size and licensing constraints.

## Current results

| Benchmark and protocol | Method | Metrics (%) |
|---|---|---|
| FashionGen-val / FashionMV official gallery | Base endpoint | R@1 42.73 / R@5 79.26 / R@10 87.75 |
| FashionGen-val / FashionMV official gallery | CrossPath joint routing | **R@1 44.09 / R@5 80.36 / R@10 88.28** |
| FashionIQ / DQU-CIR val-split | Reproduced DQU-CIR base | R@10 61.98 / R@50 81.57 |
| FashionIQ / DQU-CIR val-split | Cross-compatible mean | **R@10 62.85 / R@50 82.03** |

The FashionIQ result is near the MCoT-MVS result of 63.24/82.01 under the same val-split convention: it is higher at R@50 by 0.02 and lower at R@10 by 0.39. We therefore do not claim overall FashionIQ SOTA. Full comparisons and protocol notes are in [03_MAIN_TABLE.md](03_MAIN_TABLE.md).

## Repository contents

- `weave_crosspath.py`: scale-invariant rank paths, boundary traces, and utility contracts.
- `weave_crosspath_gate.py`: candidate responsibility gate and listwise loss.
- `weave_build_crosspath_cache.py`: cache construction for matched, cross, and joint paths.
- `weave_train_crosspath_gate.py` / `weave_eval_crosspath_gate.py`: training, calibration, and frozen evaluation.
- `weave_extract_crosspath_*.py`: endpoint embedding export for FashionMV/ProCIR and FashionIQ/DQU-CIR.
- `scripts/`: exact experiment orchestration and zero-parameter compatibility evaluators.
- `results/`: table-level JSON manifests and NPZ evaluation artifacts.
- `experiment.md`: append-only E0–E17 experiment record, including rejected variants.
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
