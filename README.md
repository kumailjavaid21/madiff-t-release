# MADiff-T

MADiff-T is a multi-agent evolutionary tabular diffusion framework for utility-privacy tradeoff optimization on tabular data. This public repository is trimmed for a journal-ready code release: core code, final public configs, final paper figures, and compact result tables are retained; bulky caches, logs, synthetic outputs, and old experimental artifacts are excluded.

## What is in this public repo

- Current execution pipeline in `paper_revC/`
- Current TabDDPM/MADiff backbone in `src/mas_generators/tabddpm/`
- Current fitness and selection logic in `src/fitness.py` and `src/selection.py`
- Final public configs in `configs/`
- Final journal-style figures in `outputs/final_bundle/journal_figures/`
- Compact retained result tables in `paper_revC/results_revC/` and `results/`

## Main entry point

The main experiment entry point is:

```bash
python -m paper_revC.run_all --config configs/main_eval.yaml --run-tag MYRUN --overwrite
```

Install the paper pipeline dependencies with:

```bash
pip install -r paper_revC/requirements_revC.txt
```

## Key retained configs

- `configs/main_eval.yaml` — default 5-dataset evaluation config
- `configs/pop6_seeded_20260321.yaml` — seeded M=6 run used for the final population study
- `configs/domias_pop6_seeded_20260321.yaml` — DOMIAS rerun on the seeded M=6 outputs
- `configs/abl_orig_m6_seeded_20260322.yaml` — base evolutionary calibration only
- `configs/abl_faloss_m6_seeded_20260322.yaml` — FA Loss only
- `configs/abl_final_m6_seeded_20260322.yaml` — FA Loss + adaptive PFS

## Retained compact results

The repo keeps only compact paper-facing outputs, not full generated caches.

- `paper_revC/results_revC/POP6_SEEDED_20260321/`
  - `run_manifest.json`
  - `tables/`
- `paper_revC/results_revC/DOMIAS_POP6_SEEDED_20260321/tables/privacy_strong_audits_raw.csv`
- `paper_revC/results_revC/ABL_ORIG_M6_SEEDED_20260322/tables/tstr_results_raw.csv`
- `paper_revC/results_revC/ABL_FALOSS_M6_SEEDED_20260322/tables/tstr_results_raw.csv`
- `paper_revC/results_revC/ABL_FINAL_M6_SEEDED_20260322/tables/tstr_results_raw.csv`
- `results/real_tstr_upper_bound_summary.csv`
- `results/tstr_results_with_tabsyn_stasy.csv`
- `results/tstr_summary_for_paper.csv`

## Final figures

Final paper figures are retained in:

```text
outputs/final_bundle/journal_figures/
```

Retained assets are:

- final Elsevier-style PDFs: `*_IS.pdf`
- figure source summaries: `*.csv`

LaTeX table artifacts are retained in:

```text
outputs/final_bundle/latex_tables/
```

## Supporting utility scripts

- `scripts/compute_real_tstr_upper_bound.py` — real-data TSTR upper bound with the same evaluation protocol
- `scripts/build_journal_style_figures.py` — regenerate the retained journal-style figures
- `scripts/eval_tstr_tabsyn_stasy.py` — evaluate external TabSyn/STaSy outputs with the paper protocol
- `scripts/merge_tstr_results.py` — merge TSTR outputs into compact paper tables
- `scripts/make_latex_tstr_table.py` — emit LaTeX-ready TSTR summary tables

## External baselines

Thin wrappers are retained for TabSyn and STaSy:

- `baselines/tabsyn/run_tabsyn.py`
- `baselines/stasy/run_stasy.py`

These wrappers are integration helpers. Full third-party baseline repositories and large pretrained artifacts are not bundled in this public cleanup.

## Repository layout

```text
paper_revC/                                  Current experiment harness
src/mas_generators/tabddpm/                  Current diffusion backbone
src/fitness.py                               Composite fitness
src/selection.py                             Privacy-gate selection
configs/                                     Retained public configs
scripts/                                     Retained analysis / figure scripts
baselines/                                   Thin baseline wrappers
outputs/final_bundle/         Final figures + LaTeX tables
paper_revC/results_revC/                     Compact retained run outputs
results/                                     Compact retained summary CSVs
```

## Notes

- Deterministic seeding is enabled in the retained MADiff-T code path.
- The public repo keeps code and compact final artifacts; it does not keep raw synthetic caches, large logs, local environments, or temporary experiment folders.
- Some retained configs reference prior cache roots from local experimentation. For a fresh rerun, update those cache paths or regenerate outputs from scratch.
