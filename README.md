# MADiff-T

**MADiff-T** is a multi-agent evolutionary tabular diffusion framework for joint
utility-privacy trade-off optimization on tabular data.

This repository accompanies the paper submission and is structured as a single
self-contained deliverable under [`madifft_final/`](madifft_final/).

## Repository contents

```text
madifft_final/
├── README.md                        # snapshot manifest & provenance
├── cas-sc-template-v2.tex           # paper LaTeX source (latest revision)
├── code/
│   ├── src/                         # MADiff-T training implementation
│   │   ├── fitness.py               # composite utility / privacy / fidelity fitness
│   │   ├── selection.py             # privacy-gated selection
│   │   └── mas_generators/tabddpm/  # tabular diffusion backbone
│   ├── build_all_figures_and_tables.py   # rebuild all figures + LaTeX tables
│   ├── build_journal_figures.py          # journal-style PDF figures
│   ├── extract_mia_roc_data.py           # per-sample MIA scores → ROC curves
│   ├── step4_before_after.py             # snapshot-vs-old comparison
│   └── verify_v2.py                      # regenerate-and-verify check
├── data/                            # frozen authoritative CSVs (TSTR, privacy, ablation, ...)
├── figures/                         # final PDF + PNG figures used in the paper
└── tables/                          # final LaTeX tables (table_tstr, table_privacy, ...)
```

The full file listing, source provenance for every CSV, known caveats, and
paper-text update notes are documented in [`madifft_final/README.md`](madifft_final/README.md).

## Reproducing the paper figures and tables

All figures and LaTeX tables can be regenerated from the frozen CSVs in
[`madifft_final/data/`](madifft_final/data/):

```bash
pip install -r requirements.txt
python madifft_final/code/build_all_figures_and_tables.py
```

This reads only `madifft_final/data/`, writes
`madifft_final/figures/*.csv` + PDFs and `madifft_final/tables/*.tex`, and
prints a head of every regenerated file.

## Citation

See [`CITATION.cff`](CITATION.cff).
