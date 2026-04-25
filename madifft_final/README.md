# MADiff-T final data snapshot

Authoritative, frozen copy of the CSVs used to build the paper's figures and
LaTeX tables. This folder is a parallel, read-only snapshot — it never touches
`outputs/final_bundle/` or `paper_revC/results_revC/`.

Everything in `figures/` and `tables/` is regenerated from `data/` by a single
script: `code/build_all_figures_and_tables.py`.

## Folder layout

```
madifft_final/
├── data/                       # raw authoritative CSVs (never edit by hand)
│   ├── dataset_summary.csv
│   ├── run_manifest.json
│   ├── tstr/
│   │   └── tstr_all_methods_5ds_5seeds.csv
│   ├── privacy/
│   │   └── privacy_all_methods_5ds_5seeds.csv
│   ├── domias/
│   │   └── domias_all_methods_5ds_5seeds.csv
│   ├── evolution/
│   │   ├── madifft_generation_metrics.csv
│   │   └── madifft_selected_config.csv
│   ├── ablation/
│   │   ├── abl_orig_tstr.csv
│   │   ├── abl_faloss_tstr.csv
│   │   └── abl_final_tstr.csv
│   ├── external_baselines/
│   │   ├── tabsyn_stasy_tstr.csv
│   │   └── tabsyn_stasy_privacy.csv
│   └── real_upper_bound/
│       └── real_tstr_upper_bound.csv
├── figures/                    # figure-feeding CSVs (regenerated)
├── tables/                     # LaTeX tables (regenerated)
└── code/
    ├── build_all_figures_and_tables.py
    └── step4_before_after.py
```

## Source provenance

| File | Source run | Source path |
|---|---|---|
| `data/tstr/tstr_all_methods_5ds_5seeds.csv` | UNIFIED_FINAL_20260410 | `paper_revC/results_revC/UNIFIED_FINAL_20260410/tables/tstr_results_raw.csv` |
| `data/privacy/privacy_all_methods_5ds_5seeds.csv` | UNIFIED_FINAL_20260410 | `.../tables/privacy_audits_raw.csv` |
| `data/domias/domias_all_methods_5ds_5seeds.csv` | UNIFIED_FINAL_20260410 | `.../tables/privacy_strong_audits_raw.csv` |
| `data/evolution/madifft_generation_metrics.csv` | UNIFIED_FINAL_20260410 | `.../tables/madifft_generation_metrics.csv` |
| `data/evolution/madifft_selected_config.csv` | UNIFIED_FINAL_20260410 | `.../tables/madifft_selected_config.csv` |
| `data/dataset_summary.csv` | UNIFIED_FINAL_20260410 | `.../tables/dataset_summary.csv` |
| `data/run_manifest.json` | UNIFIED_FINAL_20260410 | `.../run_manifest.json` |
| `data/ablation/abl_orig_tstr.csv` | ABL_ORIG_M6_SEEDED_20260322 | `paper_revC/results_revC/ABL_ORIG_M6_SEEDED_20260322/tables/tstr_results_raw.csv` |
| `data/ablation/abl_faloss_tstr.csv` | ABL_FALOSS_M6_SEEDED_20260322 | `.../ABL_FALOSS_M6_SEEDED_20260322/tables/tstr_results_raw.csv` |
| `data/ablation/abl_final_tstr.csv` | ABL_FINAL_M6_SEEDED_20260322 | `.../ABL_FINAL_M6_SEEDED_20260322/tables/tstr_results_raw.csv` |
| `data/external_baselines/tabsyn_stasy_tstr.csv` | pre-generated | `results/tstr_results_with_tabsyn_stasy.csv` |
| `data/external_baselines/tabsyn_stasy_privacy.csv` | pre-generated | `outputs/final_bundle/journal_figures/privacy_tabsyn_stasy_raw.csv` |
| `data/real_upper_bound/real_tstr_upper_bound.csv` | real-only baseline | `results/real_tstr_upper_bound_summary.csv` |

All UNIFIED and ABL copies were verified byte-identical to their sources via
md5. The run tag `UNIFIED_FINAL_20260410` is the single primary run.

## Known caveats

1. **Ablation privacy not available.** The three ABL_* seeded runs
   (`ABL_ORIG`, `ABL_FALOSS`, `ABL_FINAL`) only emitted `tstr_results_raw.csv`.
   No `privacy_audits_raw.csv` was ever produced, so the ablation privacy
   table cannot be rebuilt without rerunning those pipelines with privacy
   auditing enabled. The build script therefore produces only
   `tables/table_ablation.tex` (TSTR-only), not an ablation-privacy table.

2. **TabSyn and STaSy are pre-generated external baselines.** They are not
   part of UNIFIED_FINAL_20260410. Their TSTR and privacy values come from
   `scripts/eval_tstr_tabsyn_stasy.py` and an earlier per-seed privacy run;
   rerunning them requires the TabSyn/STaSy baseline pipelines under
   `baselines/tabsyn/` and `baselines/stasy/`.

3. **DOMIAS is secondary.** DOMIAS AUC is reported only for MADiff-T in
   `tables/table_domias.tex` (per the paper's framing as a secondary audit).
   The raw CSV in `data/domias/` contains all methods in case a reviewer
   wants the full sweep.

4. **MIA ROC curve data is not available.** `figures/mia_roc_curve_data.csv`
   is intentionally a single-row stub documenting the gap. The privacy audit
   function only emits scalar MIA AUCs, not the underlying score vectors;
   the existing `mia_roc_curves_IS.pdf` in the paper comes from an older
   codepath. Regenerating the ROC figure from this snapshot requires
   extending `privacy_audit()` to return per-sample scores.

5. **Adult Income / Heart row counts differ from the paper.** The unified
   run uses `n=45,222` (Adult) and `n=270` (Heart) after cleaning, vs. the
   paper's older `48,842` / `303`. `tables/table_dataset_summary.tex`
   reflects the unified numbers; the paper text needs a small update.

## How to regenerate

```bash
cd D:/MAS_genAI
python madifft_final/code/build_all_figures_and_tables.py
```

This reads only `madifft_final/data/`, writes
`madifft_final/figures/*.csv` and `madifft_final/tables/*.tex`, and prints
the head of every generated file.

To print the old-vs-new comparison against the stale
`outputs/final_bundle/journal_figures/` snapshot:

```bash
python madifft_final/code/step4_before_after.py
```

## Regenerated outputs

**Figure-feeding CSVs (`figures/`):**

- `comparative_utility_summary.csv` — mean/std TSTR accuracy per
  dataset × method (7 methods including TabSyn/STaSy).
- `privacy_summary.csv` — mean/std DCR and MIA AUC per dataset × method.
- `domias_summary.csv` — mean/std DOMIAS AUC per dataset × method.
- `tradeoff_scatter_data.csv` — joint utility / DCR / MIA per point.
- `ablation_summary.csv` — Original / +FA Loss / +FA Loss+PFS accuracy.
- `evolution_curve_summary.csv` — mean/std/max/median fitness per
  dataset × generation plus selected-agent fitness.
- `mia_roc_curve_data.csv` — empty stub (see caveat 4).

**LaTeX tables (`tables/`):**

- `table_dataset_summary.tex` — Table 3.
- `table_tstr.tex` — Table 4 (7 methods × 5 datasets + real upper bound rows).
- `table_privacy.tex` — Table 5 (TabDDPM vs MADiff-T, DCR and MIA).
- `table_domias.tex` — Table 6 (MADiff-T DOMIAS AUC).
- `table_ablation.tex` — Table 7.

All table formatting matches `outputs/final_bundle/latex_tables/cas-sc-template.tex`
(`\scriptsize`, `\setlength{\tabcolsep}{...}`, `\rowcolors{2}{gray!10}{white}`,
`\rowcolor{gray!20}` header, etc.).

## Paper-text updates implied by this snapshot

Compared to the text currently in `cas-sc-template.tex`, the unified run
shifts several TabDDPM baseline numbers, so the inline TSTR deltas in
Section 4.1 should be updated:

| Dataset | Old claim | New Δpp |
|---|---|---:|
| Breast Cancer | +1.8 pp | **+1.93** |
| PIMA | +0.7 pp | **+1.30** |
| Heart | +5.9 pp | **+12.22** |
| Credit | −8.3 pp | **−6.00** |
| Adult Income | +0.6 pp | **+2.08** |

Privacy deltas (MADiff-T minus TabDDPM, negative MIA = safer):

| Dataset | DCR Δ | MIA AUC Δ |
|---|---:|---:|
| Breast Cancer | +1.97 | −0.023 |
| PIMA | +0.04 | −0.013 |
| Heart | +0.48 | −0.109 |
| Credit | +1.20 | −0.087 |
| Adult Income | +1.29 | +0.013 |

Full dataset × method comparison is printed by `step4_before_after.py`.
