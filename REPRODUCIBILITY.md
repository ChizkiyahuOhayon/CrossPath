# Reproducibility guide

## Reproducibility scope

This snapshot supports three levels of verification:

1. **Table audit without a GPU:** parse every saved JSON/NPZ artifact and trace each reported value to `results/` and `experiment.md`.
2. **Method verification:** run all unit tests and run cache/gate/evaluation code from precomputed aligned endpoint embeddings.
3. **End-to-end benchmark reproduction:** obtain the third-party datasets and endpoint implementations, train or download the endpoint checkpoints, export aligned embeddings, then execute the scripts in `scripts/`.

The repository contains levels 1 and 2 completely. Level 3 cannot redistribute FashionIQ/FashionMV images or multi-gigabyte third-party-derived checkpoints; paths and commands are provided instead.

## Environment used for the reported runs

- GPU: NVIDIA GeForce RTX 4090 (24 GB)
- Python: 3.12.3
- PyTorch: 2.8.0+cu128
- NumPy: 2.3.2
- `PYTHONHASHSEED=0`

The core CrossPath code only requires NumPy and PyTorch. Encoder extraction inherits the dependencies of FashionMV/ProCIR or DQU-CIR. Use the upstream environment for the selected endpoint family.

## Data and endpoint preparation

### FashionGen / FashionMV

1. Clone FashionMV at commit `1c2f05cf6b4e166160e8f8e210f536e8cc8a08a7`.
2. Obtain FashionMV annotations and FashionGen images under the upstream terms.
3. Provide two aligned ProCIR-compatible checkpoints. The reported experiment used two independently trained FashionGen-specialized A1 checkpoints with seeds `20260718` and `20260722`.
4. Set the paths at the top of `scripts/run_A1seedpair_pipeline.sh`, or export equivalent path variables after adapting the script to the local machine.

### FashionIQ / DQU-CIR

1. Clone the official [DQU-CIR repository](https://github.com/iLearn-Lab/SIGIR24-DQU-CIR).
2. Prepare FashionIQ exactly as required by DQU-CIR, including resized images and its released caption/keyword files.
3. The reported endpoints are the reproduced seed-42 checkpoint and a GradCache effective-batch-128 checkpoint for each of `dress`, `shirt`, and `toptee`.
4. Configure the paths at the top of `scripts/run_fashioniq_dqu_crosspath.sh`.

Two FashionIQ gallery conventions occur in prior work. The paper-comparable main result here uses the DQU-CIR **val-split** gallery (the union of validation source and target images, with the source excluded per query). The stricter **original-split** full gallery is recorded separately under `results/fashioniq_original/` and must not be mixed with val-split numbers.

### FashionIQ / MCoT-MVS

1. Clone the official MCoT-MVS repository at commit `cb57a336843f94827a319a0b94b071ca29f9c187`.
2. Prepare its FashionIQ LLM annotations and `seg_features_vit-h_patch` assets using the upstream layout.
3. Download the author-released dress, shirt, and toptee checkpoints. Their SHA-256 values are recorded in `results/e24_heterogeneous/mcot/*/manifest.json`.
4. Configure the paths at the top of `scripts/run_e24_heterogeneous_crosspath.sh`. The script exports MCoT embeddings once, links them with both DQU endpoints, and evaluates the fixed diagonal/cross/all compatibility reducers.

MCoT-MVS's released FashionIQ evaluator retains the source image, while the DQU/CrossPath evaluator removes it. E24 therefore saves both protocols: `*_include_source_summary.json` reproduces the MCoT author-code convention, and `*_summary.json` applies uniform source exclusion to every local endpoint and fusion.

## Reproducing the reported stages

The principal sequence is:

```bash
# 1. Export aligned query/gallery embeddings from two frozen endpoints.
# See run_A1seedpair_pipeline.sh or run_fashioniq_dqu_crosspath.sh.

# 2. Build path boundary caches.
python weave_build_crosspath_cache.py \
  --embedding-dir /path/to/embeddings \
  --output-dir /path/to/cache \
  --cutoffs 1 10 50 \
  --exclude-source \
  --correction-path joint \
  --device cuda

# 3. Train and calibrate the gate only on the internal split.
python weave_train_crosspath_gate.py \
  --embedding-dir /path/to/internal/embeddings \
  --cache-dir /path/to/internal/cache \
  --output-dir /path/to/gate \
  --widths 128 256 --epochs 3 \
  --batch-records 64 --learning-rate 0.001 \
  --weight-decay 0.0001 --grad-clip 1.0 \
  --regression-cost 2.0 --seed 20260820 --device cuda

# 4. Freeze the gate and evaluate once on the benchmark split.
python weave_eval_crosspath_gate.py \
  --embedding-dir /path/to/official/embeddings \
  --cache-dir /path/to/official/cache \
  --gate-dir /path/to/gate \
  --output-dir /path/to/eval \
  --bootstrap-samples 100 --bootstrap-seed 20260820 --device cuda
```

Use cutoffs `1 5 10` and do not exclude the source for the FashionGen protocol. Use cutoffs `1 10 50` and `--exclude-source` for FashionIQ. The exact historical commands for each variant are retained in `scripts/` and the settings/outcomes are indexed in `experiment.md`.

The heterogeneous FashionIQ main result is reproduced with:

```bash
bash scripts/run_e24_heterogeneous_crosspath.sh
```

After E24 embeddings are available, the coordinate-scrambling control, rescue/harm decomposition, and FashionIQ 3×3 endpoint matrix are reproduced with:

```bash
bash scripts/run_e26_e27_controls.sh
```

The control applies one fixed signed permutation to both query and gallery coordinates of endpoint 1. This orthogonal map preserves all within-endpoint dot products while breaking off-diagonal coordinate correspondence. The saved report includes the maximum endpoint score error, per-path recall, and the complete rescue/harm partition at each cutoff.

Paper tables and quantitative figures are regenerated directly from the saved
artifacts with:

```bash
python figures/gen_fig_paper_results.py
```

Qualitative case export additionally requires the licensed benchmark images and
the aligned E24 embeddings. `scripts/export_retrieval_cases.py` writes a portable
manifest and copies the selected original files; after that, run
`python figures/gen_fig_retrieval_cases.py`. Exact dataset-specific arguments are
listed by `python scripts/export_retrieval_cases.py --help`.

## Artifact integrity

Run:

```bash
python scripts/verify_release.py
```

It checks that E0–E28 are all present, all JSON artifacts parse, all NPZ archives
are readable, required headline and paper artifacts exist, high-resolution figure
exports have valid signatures, and displayed qualitative ranks match the exact
stable rankings.

Absolute `/root/...` paths in saved manifests are provenance from the original experiment machine, not required installation paths. Configure local paths in the orchestration scripts when reproducing.
