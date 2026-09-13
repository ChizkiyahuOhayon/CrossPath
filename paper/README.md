# CrossPath CVPR manuscript

This directory contains the anonymized CVPR review manuscript. The official
CVPR 2026 author kit is used provisionally until the CVPR 2027 kit is released.

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Paper figures are copied from `../paper_assets/figures/`. Quantitative figures
are vector PDFs. Retrieval panels are generated from original licensed dataset
files and should be regenerated locally rather than redistributed with raw
benchmark images.

`refcheck_report.json` is the machine-readable output from `academic-refchecker`.
The current scan resolves all 40 references, with zero title/author errors and
zero unverified entries. Its seven warnings compare formal publication years
against earlier arXiv release years; the BibTeX entries retain the official
proceedings or journal years.
