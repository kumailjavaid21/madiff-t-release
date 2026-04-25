"""Step 4: print old-vs-new comparison across TSTR utility and privacy."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OLD_FIG = Path("D:/MAS_genAI/outputs/final_bundle/journal_figures")
NEW_FIG = ROOT / "figures"

DATASETS = ["breast_cancer", "pima", "heart", "credit", "adult_income"]
DATASET_LABELS = {
    "breast_cancer": "Breast Cancer",
    "pima": "PIMA",
    "heart": "Heart",
    "credit": "Credit",
    "adult_income": "Adult Income",
}

# method code in old CSVs -> method code in new CSVs
METHOD_OLD_TO_NEW = {
    "GC": "gaussian_copula",
    "TVAE": "tvae",
    "CTGAN": "ctgan",
    "TabDDPM": "tabddpm_single",
    "MADiffT": "madifft",
    "MADiff-T": "madifft",
    "TabSyn": "tabsyn",
    "STaSy": "stasy",
}
DISPLAY = {
    "gaussian_copula": "GC",
    "tvae": "TVAE",
    "ctgan": "CTGAN",
    "tabddpm_single": "TabDDPM",
    "madifft": "MADiff-T",
    "tabsyn": "TabSyn",
    "stasy": "STaSy",
}

old_u = pd.read_csv(OLD_FIG / "comparative_utility_summary.csv")
new_u = pd.read_csv(NEW_FIG / "comparative_utility_summary.csv")
old_p = pd.read_csv(OLD_FIG / "privacy_summary_for_figures.csv")
new_p = pd.read_csv(NEW_FIG / "privacy_summary.csv")

old_u["method_norm"] = old_u["method"].map(METHOD_OLD_TO_NEW).fillna(old_u["method"])
old_p["method_norm"] = old_p["method"].map(METHOD_OLD_TO_NEW).fillna(old_p["method"])


def _lookup(df: pd.DataFrame, ds: str, mcol_val: str, col: str, mkey: str = "method_norm") -> float:
    if mkey not in df.columns:
        mkey = "method"
    sub = df[(df["dataset"] == ds) & (df[mkey] == mcol_val)]
    if sub.empty:
        return float("nan")
    return float(sub[col].iloc[0])


# ---- (a) + (b) TSTR utility: old figure CSV vs new, per dataset x method ----
print("=" * 88)
print("TSTR UTILITY: old (journal_figures) vs new (madifft_final) — accuracy %")
print("=" * 88)
header = f"{'Dataset':<15} {'Method':<10} {'OLD mean':>10} {'NEW mean':>10} {'Δ (pp)':>9}"
print(header)
print("-" * len(header))
methods = ["gaussian_copula", "tvae", "ctgan", "tabddpm_single",
           "tabsyn", "stasy", "madifft"]
rows_utility = []
for ds in DATASETS:
    for m in methods:
        old_v = _lookup(old_u, ds, m, "mean_acc_pct")
        new_v = _lookup(new_u, ds, m, "mean_acc_pct", mkey="method")
        delta = new_v - old_v if pd.notna(old_v) and pd.notna(new_v) else float("nan")
        marker = "  <--" if pd.notna(delta) and abs(delta) >= 0.05 else ""
        print(
            f"{DATASET_LABELS[ds]:<15} {DISPLAY[m]:<10} "
            f"{old_v:>10.2f} {new_v:>10.2f} {delta:>+9.2f}{marker}"
        )
        rows_utility.append(
            {"dataset": ds, "method": m, "old": old_v, "new": new_v, "delta": delta}
        )
    print()

# ---- (c) Privacy: DCR and MIA, per dataset x method -------------------------
print("=" * 88)
print("PRIVACY DCR: old vs new")
print("=" * 88)
header = f"{'Dataset':<15} {'Method':<10} {'OLD DCR':>10} {'NEW DCR':>10} {'Δ':>10}"
print(header)
print("-" * len(header))
rows_dcr = []
for ds in DATASETS:
    for m in methods:
        old_v = _lookup(old_p, ds, m, "dcr_mean")
        new_v = _lookup(new_p, ds, m, "dcr_mean", mkey="method")
        delta = new_v - old_v if pd.notna(old_v) and pd.notna(new_v) else float("nan")
        marker = "  <--" if pd.notna(delta) and abs(delta) >= 0.005 else ""
        print(
            f"{DATASET_LABELS[ds]:<15} {DISPLAY[m]:<10} "
            f"{old_v:>10.3f} {new_v:>10.3f} {delta:>+10.3f}{marker}"
        )
        rows_dcr.append(
            {"dataset": ds, "method": m, "old": old_v, "new": new_v, "delta": delta}
        )
    print()

print("=" * 88)
print("PRIVACY MIA AUC: old vs new")
print("=" * 88)
print(header.replace("DCR", "MIA"))
print("-" * len(header))
for ds in DATASETS:
    for m in methods:
        old_v = _lookup(old_p, ds, m, "mia_mean")
        new_v = _lookup(new_p, ds, m, "mia_mean", mkey="method")
        delta = new_v - old_v if pd.notna(old_v) and pd.notna(new_v) else float("nan")
        marker = "  <--" if pd.notna(delta) and abs(delta) >= 0.002 else ""
        print(
            f"{DATASET_LABELS[ds]:<15} {DISPLAY[m]:<10} "
            f"{old_v:>10.3f} {new_v:>10.3f} {delta:>+10.3f}{marker}"
        )
    print()

# ---- (d) Updated inline claims: MADiff-T minus TabDDPM pp deltas ------------
print("=" * 88)
print("UPDATED INLINE CLAIMS — MADiff-T minus TabDDPM (TSTR pp deltas)")
print("(These replace the +1.8/+0.6/+0.7/+5.9/... values in the paper text)")
print("=" * 88)
print(f"{'Dataset':<15} {'TabDDPM':>10} {'MADiff-T':>10} {'Δpp':>9} {'Direction':>14}")
print("-" * 64)
for ds in DATASETS:
    tab = _lookup(new_u, ds, "tabddpm_single", "mean_acc_pct")
    mad = _lookup(new_u, ds, "madifft", "mean_acc_pct")
    d = mad - tab
    direction = "improves" if d > 0 else ("trades" if d < 0 else "equal")
    print(
        f"{DATASET_LABELS[ds]:<15} {tab:>10.2f} {mad:>10.2f} "
        f"{d:>+9.2f} {direction:>14}"
    )

print("\nUpdated DCR deltas (MADiff-T minus TabDDPM):")
print(f"{'Dataset':<15} {'TabDDPM':>10} {'MADiff-T':>10} {'Δ':>10}")
print("-" * 50)
for ds in DATASETS:
    tab = _lookup(new_p, ds, "tabddpm_single", "dcr_mean")
    mad = _lookup(new_p, ds, "madifft", "dcr_mean")
    d = mad - tab
    print(f"{DATASET_LABELS[ds]:<15} {tab:>10.3f} {mad:>10.3f} {d:>+10.3f}")

print("\nUpdated MIA AUC deltas (MADiff-T minus TabDDPM; negative = safer):")
print(f"{'Dataset':<15} {'TabDDPM':>10} {'MADiff-T':>10} {'Δ':>10}")
print("-" * 50)
for ds in DATASETS:
    tab = _lookup(new_p, ds, "tabddpm_single", "mia_mean")
    mad = _lookup(new_p, ds, "madifft", "mia_mean")
    d = mad - tab
    print(f"{DATASET_LABELS[ds]:<15} {tab:>10.3f} {mad:>10.3f} {d:>+10.3f}")
