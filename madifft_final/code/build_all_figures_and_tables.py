"""Rebuild all journal-figure CSVs and LaTeX tables from the single
authoritative `madifft_final/data/` snapshot.

Reads only from `madifft_final/data/`.
Writes only to  `madifft_final/figures/` and `madifft_final/tables/`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
TAB = ROOT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

DATASETS = ["breast_cancer", "pima", "heart", "credit", "adult_income"]
DATASET_LABELS = {
    "breast_cancer": "Breast Cancer",
    "pima": "PIMA",
    "heart": "Heart",
    "credit": "Credit",
    "adult_income": "Adult Income",
}
METHOD_DISPLAY = {
    "gaussian_copula": "GC",
    "tvae": "TVAE",
    "ctgan": "CTGAN",
    "tabddpm_single": "TabDDPM",
    "madifft": "MADiff-T",
    "tabsyn": "TabSyn",
    "stasy": "STaSy",
}
METHOD_ORDER = ["gaussian_copula", "tvae", "ctgan", "tabddpm_single",
                "tabsyn", "stasy", "madifft"]


def _fmt(mean: float, std: float, nd: int = 2) -> str:
    if pd.isna(mean):
        return "---"
    if pd.isna(std):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} $\\pm$ {std:.{nd}f}"


# ---------- Load inputs ----------------------------------------------------
tstr_unified = pd.read_csv(DATA / "tstr/tstr_all_methods_5ds_5seeds.csv")
priv_unified = pd.read_csv(DATA / "privacy/privacy_all_methods_5ds_5seeds.csv")
domias = pd.read_csv(DATA / "domias/domias_all_methods_5ds_5seeds.csv")
ext_tstr = pd.read_csv(DATA / "external_baselines/tabsyn_stasy_tstr.csv")
ext_priv = pd.read_csv(
    DATA / "external_baselines/tabsyn_stasy_privacy.csv"
)
real_ub = pd.read_csv(DATA / "real_upper_bound/real_tstr_upper_bound.csv")
ds_summary = pd.read_csv(DATA / "dataset_summary.csv")
gen_metrics = pd.read_csv(DATA / "evolution/madifft_generation_metrics.csv")

abl_orig = pd.read_csv(DATA / "ablation/abl_orig_tstr.csv")
abl_faloss = pd.read_csv(DATA / "ablation/abl_faloss_tstr.csv")
abl_final = pd.read_csv(DATA / "ablation/abl_final_tstr.csv")


# ---------- comparative_utility_summary.csv --------------------------------
acc_unified = tstr_unified[
    (tstr_unified["metric_name"] == "accuracy")
    & (tstr_unified["classifier"] == "logistic_regression")
].copy()
rows = (
    acc_unified.groupby(["dataset", "method"])["metric_value"]
    .agg(["mean", "std"])
    .reset_index()
)
rows["mean_acc_pct"] = rows["mean"] * 100
rows["std_acc_pct"] = rows["std"] * 100
rows = rows[["dataset", "method", "mean_acc_pct", "std_acc_pct"]]

ext_acc = ext_tstr[
    (ext_tstr["metric_name"] == "accuracy")
    & (ext_tstr["classifier"] == "logistic_regression")
].copy()
ext_rows = (
    ext_acc.groupby(["dataset", "method"])["metric_value"]
    .agg(["mean", "std"])
    .reset_index()
)
ext_rows["mean_acc_pct"] = ext_rows["mean"] * 100
ext_rows["std_acc_pct"] = ext_rows["std"] * 100
ext_rows = ext_rows[["dataset", "method", "mean_acc_pct", "std_acc_pct"]]

comparative_utility = pd.concat([rows, ext_rows], ignore_index=True)
comparative_utility["method_display"] = comparative_utility["method"].map(
    METHOD_DISPLAY
)
comparative_utility["dataset_display"] = comparative_utility["dataset"].map(
    DATASET_LABELS
)
comparative_utility = comparative_utility.sort_values(
    ["dataset", "method"]
).reset_index(drop=True)
comparative_utility.to_csv(FIG / "comparative_utility_summary.csv", index=False)


# ---------- privacy_summary.csv --------------------------------------------
priv_rows = (
    priv_unified.groupby(["dataset", "method"])[["dcr_q", "mia_auc"]]
    .agg(["mean", "std"])
    .reset_index()
)
priv_rows.columns = [
    "dataset", "method", "dcr_mean", "dcr_std", "mia_mean", "mia_std"
]

ext_priv_norm = ext_priv.copy()
ext_priv_norm["method"] = ext_priv_norm["method"].str.lower()
ext_priv_tabsyn_stasy = ext_priv_norm[
    ext_priv_norm["method"].isin(["tabsyn", "stasy"])
]
ext_priv_rows = (
    ext_priv_tabsyn_stasy.groupby(["dataset", "method"])[["dcr_q", "mia_auc"]]
    .agg(["mean", "std"])
    .reset_index()
)
ext_priv_rows.columns = [
    "dataset", "method", "dcr_mean", "dcr_std", "mia_mean", "mia_std"
]

privacy_summary = pd.concat([priv_rows, ext_priv_rows], ignore_index=True)
privacy_summary["method_display"] = privacy_summary["method"].map(METHOD_DISPLAY)
privacy_summary["dataset_display"] = privacy_summary["dataset"].map(
    DATASET_LABELS
)
privacy_summary = privacy_summary.sort_values(
    ["dataset", "method"]
).reset_index(drop=True)
privacy_summary.to_csv(FIG / "privacy_summary.csv", index=False)


# ---------- tradeoff_scatter_data.csv --------------------------------------
tradeoff = comparative_utility.merge(
    privacy_summary[["dataset", "method", "dcr_mean", "mia_mean"]],
    on=["dataset", "method"],
    how="left",
)
tradeoff = tradeoff[
    ["dataset", "dataset_display", "method", "method_display",
     "mean_acc_pct", "std_acc_pct", "dcr_mean", "mia_mean"]
]
tradeoff.to_csv(FIG / "tradeoff_scatter_data.csv", index=False)


# ---------- domias_summary.csv ---------------------------------------------
domias_rows = (
    domias[domias["status"] == "success"]
    .groupby(["dataset", "method"])["domias_auc"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
domias_rows.columns = ["dataset", "method", "domias_mean", "domias_std", "n_seeds"]
domias_rows["method_display"] = domias_rows["method"].map(METHOD_DISPLAY)
domias_rows["dataset_display"] = domias_rows["dataset"].map(DATASET_LABELS)
domias_rows = domias_rows.sort_values(["dataset", "method"]).reset_index(drop=True)
domias_rows.to_csv(FIG / "domias_summary.csv", index=False)


# ---------- ablation_summary.csv -------------------------------------------
def _abl_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    sub = df[
        (df["metric_name"] == "accuracy")
        & (df["classifier"] == "logistic_regression")
        & (df["method"] == "madifft")
    ].copy()
    g = sub.groupby("dataset")["metric_value"].agg(["mean", "std"]).reset_index()
    g["label"] = label
    g["utility_mean"] = g["mean"] * 100
    g["utility_std"] = g["std"] * 100
    return g[["dataset", "label", "utility_mean", "utility_std"]]


ablation_summary = pd.concat(
    [
        _abl_summary(abl_orig, "Original"),
        _abl_summary(abl_faloss, "+FA Loss"),
        _abl_summary(abl_final, "+FA Loss+PFS (Final)"),
    ],
    ignore_index=True,
)
ablation_summary["dataset_display"] = ablation_summary["dataset"].map(
    DATASET_LABELS
)
ablation_summary = ablation_summary.sort_values(
    ["dataset", "label"]
).reset_index(drop=True)
ablation_summary.to_csv(FIG / "ablation_summary.csv", index=False)


# ---------- evolution_curve_summary.csv ------------------------------------
ev = gen_metrics[
    gen_metrics["guard_failed"].isna() | (gen_metrics["guard_failed"] == False)
].copy()
ev = gen_metrics.copy()
ev_agg = (
    ev.groupby(["dataset", "generation"])["fitness"]
    .agg(["mean", "std", "max", "median"])
    .reset_index()
)
selected = (
    ev[ev["selected"] == True]
    .groupby(["dataset", "generation"])["fitness"]
    .mean()
    .reset_index()
    .rename(columns={"fitness": "selected_mean"})
)
evolution_curve = ev_agg.merge(selected, on=["dataset", "generation"], how="left")
evolution_curve["dataset_display"] = evolution_curve["dataset"].map(DATASET_LABELS)
evolution_curve = evolution_curve.sort_values(
    ["dataset", "generation"]
).reset_index(drop=True)
evolution_curve.to_csv(FIG / "evolution_curve_summary.csv", index=False)


# ---------- mia_roc_curve_data.csv (documented gap) ------------------------
mia_roc_note = pd.DataFrame(
    {
        "dataset": [np.nan],
        "method": [np.nan],
        "fpr": [np.nan],
        "tpr_mean": [np.nan],
        "tpr_std": [np.nan],
        "note": [
            "Per-seed ROC points not emitted by privacy_audit(); only "
            "scalar MIA AUCs are available. This file is intentionally "
            "empty. To regenerate, privacy_audit() must be extended to "
            "return the underlying score vectors."
        ],
    }
)
mia_roc_note.to_csv(FIG / "mia_roc_curve_data.csv", index=False)


# ---------- LaTeX tables ---------------------------------------------------

# Real upper bound lookup (dataset keys differ; map to unified keys)
REAL_UB = {
    "breast_cancer": (
        real_ub.loc[real_ub["Dataset"] == "BreastCancer", "Accuracy Mean"].iloc[0] * 100,
        real_ub.loc[real_ub["Dataset"] == "BreastCancer", "Accuracy Std"].iloc[0] * 100,
    ),
    "pima": (
        real_ub.loc[real_ub["Dataset"] == "PIMA", "Accuracy Mean"].iloc[0] * 100,
        real_ub.loc[real_ub["Dataset"] == "PIMA", "Accuracy Std"].iloc[0] * 100,
    ),
    "heart": (
        real_ub.loc[real_ub["Dataset"] == "Heart", "Accuracy Mean"].iloc[0] * 100,
        real_ub.loc[real_ub["Dataset"] == "Heart", "Accuracy Std"].iloc[0] * 100,
    ),
    "credit": (
        real_ub.loc[real_ub["Dataset"] == "GermanCredit", "Accuracy Mean"].iloc[0] * 100,
        real_ub.loc[real_ub["Dataset"] == "GermanCredit", "Accuracy Std"].iloc[0] * 100,
    ),
    "adult_income": (
        real_ub.loc[real_ub["Dataset"] == "AdultIncome", "Accuracy Mean"].iloc[0] * 100,
        real_ub.loc[real_ub["Dataset"] == "AdultIncome", "Accuracy Std"].iloc[0] * 100,
    ),
}


def _cell(df: pd.DataFrame, dataset: str, method: str,
          mean_col: str, std_col: str, nd: int = 2) -> tuple[float, float]:
    sub = df[(df["dataset"] == dataset) & (df["method"] == method)]
    if sub.empty:
        return (float("nan"), float("nan"))
    return (float(sub[mean_col].iloc[0]), float(sub[std_col].iloc[0]))


# ---- table_tstr.tex (Table 4) ---------------------------------------------
header = r"""\begin{table}[!t]
\centering
\caption{TSTR utility (Accuracy \%, mean $\pm$ std over 5 seeds)
across all datasets. Higher is better. \textbf{Bold} indicates
MADiff-T outperforms the TabDDPM single-agent baseline. Real data
upper bound rows (bottom, shaded) show classifier performance
trained directly on real data, providing context for the
synthetic-to-real utility gap. All non-external-baseline columns
come from a single unified run (\texttt{UNIFIED\_FINAL\_20260410}).}
\label{tab:results_utility_all}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\renewcommand{\arraystretch}{1.12}

\resizebox{\linewidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
\rowcolor{gray!20}
\textbf{Dataset} & \textbf{GC} & \textbf{TVAE} & \textbf{CTGAN} & \textbf{TabDDPM} & \textbf{TabSyn} & \textbf{STaSy} & \textbf{MADiff-T} \\
\midrule
"""

body_rows = []
for ds in DATASETS:
    cells = []
    for m in ["gaussian_copula", "tvae", "ctgan", "tabddpm_single",
              "tabsyn", "stasy"]:
        mean, std = _cell(comparative_utility, ds, m,
                          "mean_acc_pct", "std_acc_pct")
        cells.append(_fmt(mean, std))
    mad_mean, mad_std = _cell(comparative_utility, ds, "madifft",
                              "mean_acc_pct", "std_acc_pct")
    tab_mean, tab_std = _cell(comparative_utility, ds, "tabddpm_single",
                              "mean_acc_pct", "std_acc_pct")
    mad_str = _fmt(mad_mean, mad_std)
    if not pd.isna(mad_mean) and not pd.isna(tab_mean) and mad_mean > tab_mean:
        mad_str = f"\\textbf{{{mad_str}}}"
    cells.append(mad_str)
    row = f"{DATASET_LABELS[ds]} & " + " & ".join(cells) + r" \\"
    body_rows.append(row)

body = "\n".join(body_rows)

ub_section = r"""\midrule
\rowcolor{gray!5}
\multicolumn{8}{l}{\scriptsize\textit{Real data upper bound (classifier trained on real $D_{\text{train}}$, evaluated on real $D_{\text{test}}$; no synthesis)}} \\
"""
ub_lines = []
for ds in DATASETS:
    m, s = REAL_UB[ds]
    ub_lines.append(
        r"\rowcolor{gray!5}" "\n"
        f"{DATASET_LABELS[ds]} & "
        r"\multicolumn{6}{c}{---} & "
        f"{_fmt(m, s)} " r"\\"
    )
ub_body = "\n".join(ub_lines)

footer = r"""\bottomrule
\end{tabular}%
}
\end{table}
"""

(TAB / "table_tstr.tex").write_text(
    header + body + "\n" + ub_section + ub_body + "\n" + footer,
    encoding="utf-8",
)


# ---- table_privacy.tex (Table 5) ------------------------------------------
priv_header = r"""\begin{table}[!t]
\centering
\caption{Privacy audits across all datasets. Higher DCR is safer;
lower MIA AUC is safer (0.5 indicates chance under the chosen
attacker). \textbf{Bold} indicates MADiff-T safer than TabDDPM.
All rows come from a single unified run
(\texttt{UNIFIED\_FINAL\_20260410}).}
\label{tab:results_privacy_all}
\scriptsize
\setlength{\tabcolsep}{5.5pt}
\renewcommand{\arraystretch}{1.15}
\rowcolors{2}{gray!10}{white}
\begin{tabular}{l|c c|c c}
\rowcolor{gray!20}
\textbf{Dataset} &
\multicolumn{2}{c|}{\textbf{TabDDPM}} &
\multicolumn{2}{c}{\textbf{MADiff-T (ours)}} \\
\rowcolor{gray!20}
& DCR $\uparrow$ & MIA AUC $\downarrow$ & DCR $\uparrow$ & MIA AUC $\downarrow$ \\
\hline
"""

priv_rows_tex = []
for ds in DATASETS:
    tab_dcr, _ = _cell(privacy_summary, ds, "tabddpm_single",
                       "dcr_mean", "dcr_std")
    tab_mia, _ = _cell(privacy_summary, ds, "tabddpm_single",
                       "mia_mean", "mia_std")
    mad_dcr, _ = _cell(privacy_summary, ds, "madifft", "dcr_mean", "dcr_std")
    mad_mia, _ = _cell(privacy_summary, ds, "madifft", "mia_mean", "mia_std")

    def _bold(v, better):
        return f"\\textbf{{{v}}}" if better else v

    tab_dcr_s = f"{tab_dcr:.2f}"
    tab_mia_s = f"{tab_mia:.3f}"
    mad_dcr_s = _bold(f"{mad_dcr:.2f}", mad_dcr > tab_dcr)
    mad_mia_s = _bold(f"{mad_mia:.3f}", mad_mia < tab_mia)
    priv_rows_tex.append(
        f"{DATASET_LABELS[ds]} & {tab_dcr_s} & {tab_mia_s} & "
        f"{mad_dcr_s} & {mad_mia_s} " r"\\"
    )
priv_footer = r"""\end{tabular}
\end{table}
"""
(TAB / "table_privacy.tex").write_text(
    priv_header + "\n".join(priv_rows_tex) + "\n" + priv_footer,
    encoding="utf-8",
)


# ---- table_domias.tex (Table 6) -------------------------------------------
dom_header = r"""\begin{table}[!t]
\centering
\caption{MADiff-T DOMIAS membership-inference audit
($M=6$, seeded; mean $\pm$ std over 5 seeds; members $D_{\text{train}}$,
non-members $D_{\text{test}}$). Lower is safer; 0.5 indicates
chance-level distinguishability. From a single unified run
(\texttt{UNIFIED\_FINAL\_20260410}).}
\label{tab:results_domias}
\scriptsize
\setlength{\tabcolsep}{5.5pt}
\renewcommand{\arraystretch}{1.15}
\rowcolors{2}{gray!10}{white}
\begin{tabular}{l|c}
\rowcolor{gray!20}
\textbf{Dataset} & \textbf{MADiff-T DOMIAS AUC $\downarrow$} \\
\hline
"""
dom_rows = []
ds_order_dom = ["adult_income", "breast_cancer", "credit", "heart", "pima"]
ds_label_dom = {
    "adult_income": "Adult Income",
    "breast_cancer": "Breast Cancer",
    "credit": "German Credit",
    "heart": "Heart Disease",
    "pima": "PIMA Diabetes",
}
for ds in ds_order_dom:
    m, s = _cell(domias_rows, ds, "madifft", "domias_mean", "domias_std")
    dom_rows.append(f"{ds_label_dom[ds]} & {m:.3f} $\\pm$ {s:.3f} " r"\\")
dom_footer = r"""\end{tabular}
\end{table}
"""
(TAB / "table_domias.tex").write_text(
    dom_header + "\n".join(dom_rows) + "\n" + dom_footer, encoding="utf-8"
)


# ---- table_ablation.tex (Table 7) -----------------------------------------
abl_header = r"""\begin{table}[!t]
\centering
\caption{Ablation: TSTR accuracy (\%, mean $\pm$ std over 5 seeds,
$M=6$ seeded) as enhancements are added. Original = base evolutionary
calibration only (no FA Loss, no PFS); +FA Loss = Feature-Aggregated
Loss added; +FA Loss+PFS = full MADiff-T. Since Final equals +FA Loss
on all datasets where PFS is disabled, those rows are marked
$\dagger$. PFS is active only for PIMA ($n_{\text{train}} \geq 520$,
categorical ratio $< 0.5$).}
\label{tab:ablation_steps}
\scriptsize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\rowcolors{2}{gray!10}{white}
\begin{tabular}{l|c|c|c}
\rowcolor{gray!20}
\textbf{Dataset} & \textbf{Original} & \textbf{+FA Loss} & \textbf{+FA Loss+PFS (Final)} \\
\hline
"""

def _abl_cell(label: str, ds: str) -> str:
    sub = ablation_summary[
        (ablation_summary["dataset"] == ds)
        & (ablation_summary["label"] == label)
    ]
    if sub.empty:
        return "---"
    return f"{sub['utility_mean'].iloc[0]:.2f} $\\pm$ {sub['utility_std'].iloc[0]:.2f}"


abl_rows = []
# PIMA is the only dataset where PFS is active; all others get dagger
dagger_map = {
    "breast_cancer": True, "adult_income": True,
    "pima": False, "heart": True, "credit": True,
}
for ds in ["breast_cancer", "adult_income", "pima", "heart", "credit"]:
    label = (DATASET_LABELS[ds] + r"$^\dagger$") if dagger_map[ds] else DATASET_LABELS[ds]
    orig = _abl_cell("Original", ds)
    fal = _abl_cell("+FA Loss", ds)
    fin = _abl_cell("+FA Loss+PFS (Final)", ds)
    abl_rows.append(f"{label} & {orig} & {fal} & {fin} " r"\\")
abl_footer = r"""\end{tabular}
\end{table}
"""
(TAB / "table_ablation.tex").write_text(
    abl_header + "\n".join(abl_rows) + "\n" + abl_footer, encoding="utf-8"
)


# ---- table_dataset_summary.tex (Table 3) ----------------------------------
ds_tex_order = ["breast_cancer", "adult_income", "pima", "heart", "credit"]
ds_header = r"""\begin{table}[t]
\centering
\caption{Dataset summary. $d_c$ = continuous features, $d_d$ = categorical features, \#Feat = effective dimensionality after one-hot encoding, $n_{\text{train}}$ = training split size (70\%).}
\label{tab:datasets}
\scriptsize
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.1}
\begin{tabular}{l r r r r r r}
\toprule
\rowcolor{tablehead}
\textbf{Dataset} & \textbf{$N$} & \textbf{$d_c$} & \textbf{$d_d$} & \textbf{\#Feat} & \textbf{$n_{\text{train}}$} & \textbf{Classes} \\
\midrule
"""

def _fmtint(n: int) -> str:
    s = f"{int(n):,}"
    return s.replace(",", r"{,}")


ds_lines = []
for key in ds_tex_order:
    r = ds_summary[ds_summary["dataset"] == key].iloc[0]
    ds_lines.append(
        f"{DATASET_LABELS[key]} & {_fmtint(r['n_rows_total'])} & "
        f"{int(r['n_cont'])} & {int(r['n_cat'])} & "
        f"{int(r['n_features_encoded'])} & "
        f"{_fmtint(r['n_train'])} & {int(r['n_classes'])} " r"\\"
    )
ds_footer = r"""\bottomrule
\end{tabular}
\end{table}
"""
(TAB / "table_dataset_summary.tex").write_text(
    ds_header + "\n".join(ds_lines) + "\n" + ds_footer, encoding="utf-8"
)


# ---------- Print generated file heads -------------------------------------
print("=" * 70)
print("Generated figure CSVs:")
print("=" * 70)
for f in sorted(FIG.glob("*.csv")):
    print(f"\n--- {f.name} ---")
    df = pd.read_csv(f)
    print(f"shape: {df.shape}")
    print(df.head(20).to_string(index=False))

print("\n" + "=" * 70)
print("Generated LaTeX tables:")
print("=" * 70)
for f in sorted(TAB.glob("*.tex")):
    print(f"\n--- {f.name} ---")
    print(f.read_text(encoding="utf-8"))
