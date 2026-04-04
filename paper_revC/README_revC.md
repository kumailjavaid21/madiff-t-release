# Paper Revision C evaluation overview

This folder contains the experiment harness that generates **new result tables** for Situation C (missing baselines and stronger privacy audits) without touching the original artifacts.

## Setup

1. Activate your existing `venv` (it already contains `torch`, `sdv`, `opacus`, etc.).
2. Install the additional Python requirements from here (use the active venv):
   ```bash
   pip install -r paper_revC/requirements_revC.txt
   ```

## Running the revision pipeline

All outputs are written under `paper_revC/results_revC/<run_tag>/tables/`. Launch the orchestration with:

```bash
python -m paper_revC.run_all --config paper_revC/configs/revC.yaml
```

You can override the auto-generated run tag and force reruns with:

```bash
python -m paper_revC.run_all --config paper_revC/configs/revC.yaml --run-tag REVISIONC --overwrite
```

## Generated tables

| File | Description |
| --- | --- |
| `dataset_summary.csv` | Statistics per dataset (feature counts, class balance, missing rate, splits). |
| `tstr_results_raw.csv` | Train-synthetic/Test-real metrics per dataset/method/seed/classifier. |
| `privacy_audits_raw.csv` | Mixed-type DCR/MIA metrics per dataset/method/seed. |
| `privacy_strong_audits.csv` | DOMIAS/TAPAS rows when those tools are available; optional per config. |
| `privacy_strong_audits_summary.csv` | Aggregated mean/std across seeds for strong audits. |
| `madifft_selected_config.csv` | Winner agents and fitness breakdown per dataset after MADiff-T. |

The results directory also contains `figures/` and `logs/` subfolders (logging centered in `logs/run.log`).

## Optional tooling

- **TabSyn & STaSy**: Both baselines live outside the repo. Place the official code under:
  - `paper_revC/external/tabsyn/` (TabSyn repository from [amazon-science/tabsyn](https://github.com/amazon-science/tabsyn))
  - `paper_revC/external/stasy/` (STaSy repository such as [JayoungKim408/STaSy](https://github.com/JayoungKim408/STaSy))
  The wrappers now read **pre-generated CSV outputs** from:
  ```
  paper_revC/external/<method>/outputs/<dataset>/seed<seed>/synth.csv
  ```
  Supported filenames are configured in `paper_revC/configs/revC.yaml` under `external_outputs.file_names`.
  Each CSV must match the real dataset columns **exactly** (same names and order), including the label column.
  If a file is missing or mismatched, the baseline is skipped and the reason is logged.
  You can disable these baselines entirely by setting `INCLUDE_EXTERNAL_BASELINES: false` in `paper_revC/configs/revC.yaml`.

- **DOMIAS** (optional strong audit): install via `pip install domias`. Once available the harness will run `evaluate_performance(...)` and append DOMIAS-specific AUC/advantage metrics to `privacy_strong_audits.csv`.

- **TAPAS** (optional strong audit): install the TAPAS toolbox:
  ```bash
  pip install git+https://github.com/alan-turing-institute/privacy-sdg-toolbox
  ```
  If present, the pipeline will attempt a TAPAS membership-inference run and log results to `privacy_strong_audits.csv`.

## Notes

- The harness is resumable: rerunning with the same run tag stops early for already-recorded dataset/method/seed tuples (use `--overwrite` to force regeneration).
- Method-specific hyperparameters (TabDDPM, MADiff-T, etc.) are kept in `paper_revC/configs/revC.yaml` under `method_params`.
- Keep the original `paper_revC/results_revC` tree untouched; every invocation creates a fresh `<run_tag>` folder.
