# CrossPath paper assets

This directory contains paper-facing data exports and camera-ready figures.

- `data/`: deterministic case manifests, main-table CSVs, and exact plot values.
- `figures/`: vector PDF/SVG, editable PPTX/VSDX, and 600 DPI PNG exports.
- `source_images/`: byte-identical benchmark images used to build the qualitative
  figures. This directory is intentionally git-ignored because the benchmark
  licenses do not permit redistributing the raw images.

Retrieval cases are selected without manual browsing: for each requested cutoff
or FashionIQ category, `scripts/export_retrieval_cases.py` chooses the largest
base-miss/CrossPath-hit rank improvement, breaking ties by method rank and then
query index. The original dataset files remain unchanged; the plotting script only
reads them.

Regenerate quantitative figures from repository artifacts:

```bash
python3 figures/gen_fig_paper_results.py
```

After the local source-image bundle has been prepared, regenerate case figures:

```bash
python3 figures/gen_fig_retrieval_cases.py
```

The framework figure is specified in `figures/crosspath_framework.yaml`. Its
PPTX export embeds the two benchmark crops and is the preferred editable source.
The VSDX export uses image placeholders, a limitation of the deterministic Visio
backend. `fig_crosspath_framework.tex` provides the CVPR `figure*` integration.
