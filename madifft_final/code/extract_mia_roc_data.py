"""
Extract per-sample MIA scores and compute ROC curves from cached synthetic data.

Replicates the EXACT distance computation from paper_revC/privacy_basic.py:
  - Z-score normalize continuous features (std clamped to 1.0)
  - Integer-encode categorical features
  - Mixed distance: sqrt(euclidean^2 + hamming^2)
  - Attacker scores: -min_distance (negated so higher = more likely member)
  - sklearn.metrics.roc_curve(y_true, scores)

Reads:
  - Real splits: D:/MAS_genAI/data_exports/{export_name}/seed{s}/real_{train,val}.csv
  - Synthetic:   D:/MAS_genAI/outputs/20260205_070650/{ds}/{method}/seed_{s}/synthetic.csv

Writes:
  - D:/MAS_genAI/madifft_final/data/privacy/mia_roc_raw.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, roc_curve

# ── paths ──────────────────────────────────────────────────────────────────
DATA_EXPORTS = Path("D:/MAS_genAI/data_exports")
SYNTH_CACHE  = Path("D:/MAS_genAI/outputs/20260205_070650")
OUT_CSV      = Path("D:/MAS_genAI/madifft_final/data/privacy/mia_roc_raw.csv")
PRIV_CSV     = Path("D:/MAS_genAI/madifft_final/data/privacy/privacy_all_methods_5ds_5seeds.csv")

DATASETS = ["breast_cancer", "pima", "heart", "credit", "adult_income"]
METHODS  = ["gaussian_copula", "tvae", "ctgan", "tabddpm_single", "madifft"]
SEEDS    = [0, 1, 2, 3, 4]

# data_exports uses different directory names for some datasets
EXPORT_DIR = {
    "breast_cancer": "breast_cancer",
    "pima":          "pima",
    "heart":         "heart",
    "credit":        "german",
    "adult_income":  "adult",
}

LABEL_COL = {
    "breast_cancer": "target",
    "pima":          "Outcome",
    "heart":         "class",
    "credit":        "class",
    "adult_income":  "class",
}


# ── replicated from privacy_basic.py ───────────────────────────────────────
def _identify_column_groups(features: pd.DataFrame):
    numeric = [
        col for col in features.columns
        if pd.api.types.is_numeric_dtype(features[col]) and features[col].dtype != bool
    ]
    categorical = [col for col in features.columns if col not in numeric]
    return numeric, categorical


def _build_category_mappings(features: pd.DataFrame, cat_cols: list[str]):
    mappings = {}
    for col in cat_cols:
        categories = pd.Categorical(features[col]).categories
        mappings[col] = {cat: idx for idx, cat in enumerate(categories)}
    return mappings


def _encode_categories(df: pd.DataFrame, mappings: dict) -> np.ndarray:
    if not mappings:
        return np.zeros((len(df), 0))
    encoded = []
    for col, mapping in mappings.items():
        mapped = df[col].map(mapping)
        arr = pd.to_numeric(mapped, errors="coerce").to_numpy(dtype=float)
        arr = np.where(np.isnan(arr), -1.0, arr)
        encoded.append(arr)
    return np.vstack(encoded).T


def _normalize_numeric(df: pd.DataFrame, cont_cols, means, stds) -> np.ndarray:
    if not cont_cols:
        return np.zeros((len(df), 0))
    normalized = (df[cont_cols] - means) / stds.replace(0, 1)
    return normalized.to_numpy(dtype=float)


def _mixed_distances(cont_a, cont_b, cat_a, cat_b) -> np.ndarray:
    if cont_a.size and cont_b.size:
        dist_cont = cdist(cont_a, cont_b, metric="euclidean")
    else:
        dist_cont = np.zeros((cont_a.shape[0], cat_b.shape[0]))
    if cat_a.size and cat_b.size:
        dist_cat = cdist(cat_a, cat_b, metric="hamming")
    else:
        dist_cat = np.zeros_like(dist_cont)
    return np.sqrt(dist_cont ** 2 + dist_cat ** 2)


def _sample_frame(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    size = min(len(df), limit)
    if size == 0:
        return df.iloc[0:0]
    return df.sample(size, random_state=seed).reset_index(drop=True)


# ── main extraction ────────────────────────────────────────────────────────
def extract_roc(dataset: str, method: str, seed: int):
    """Return (fpr, tpr, auc) for one (dataset, method, seed)."""
    label_col = LABEL_COL[dataset]
    export_dir = EXPORT_DIR[dataset]

    real_train = pd.read_csv(DATA_EXPORTS / export_dir / f"seed{seed}" / "real_train.csv")
    real_val   = pd.read_csv(DATA_EXPORTS / export_dir / f"seed{seed}" / "real_val.csv")
    synth_path = SYNTH_CACHE / dataset / method / f"seed_{seed}" / "synthetic.csv"
    if not synth_path.exists():
        return None, None, float("nan")
    synth_df = pd.read_csv(synth_path)

    # Preprocessing — exactly as privacy_basic.py
    features_train = real_train.drop(columns=[label_col])
    cont_cols, cat_cols = _identify_column_groups(features_train)
    cat_mappings = _build_category_mappings(features_train, cat_cols)
    means = features_train[cont_cols].mean() if cont_cols else pd.Series(dtype=float)
    stds  = features_train[cont_cols].std().replace(0, 1) if cont_cols else pd.Series(dtype=float)

    # Sample sizes matching privacy_basic.py exactly
    syn_sample   = _sample_frame(synth_df.drop(columns=[label_col], errors="ignore"),
                                 limit=800, seed=seed + 5)
    train_sample = _sample_frame(real_train, limit=300, seed=seed + 10)
    val_sample   = _sample_frame(real_val,   limit=300, seed=seed + 11)

    if syn_sample.empty or train_sample.empty or val_sample.empty:
        return None, None, float("nan")

    train_feat = train_sample.drop(columns=[label_col])
    val_feat   = val_sample.drop(columns=[label_col])

    train_cont = _normalize_numeric(train_feat, cont_cols, means, stds)
    val_cont   = _normalize_numeric(val_feat,   cont_cols, means, stds)
    syn_cont   = _normalize_numeric(syn_sample, cont_cols, means, stds)
    train_cat  = _encode_categories(train_feat, cat_mappings)
    val_cat    = _encode_categories(val_feat,   cat_mappings)
    syn_cat    = _encode_categories(syn_sample, cat_mappings)

    train_dist = _mixed_distances(train_cont, syn_cont, train_cat, syn_cat).min(axis=1)
    val_dist   = _mixed_distances(val_cont,   syn_cont, val_cat,   syn_cat).min(axis=1)

    labels = np.concatenate([np.ones(len(train_dist)), np.zeros(len(val_dist))])
    scores = -np.concatenate([train_dist, val_dist])

    if len(np.unique(labels)) < 2:
        return None, None, float("nan")

    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    return fpr, tpr, auc


def main():
    rows = []
    sanity = []

    total = len(DATASETS) * len(METHODS) * len(SEEDS)
    done = 0

    for ds in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                done += 1
                tag = f"[{done}/{total}] {ds}/{method}/seed{seed}"
                fpr, tpr, auc = extract_roc(ds, method, seed)
                if fpr is None:
                    print(f"  {tag}: SKIPPED (missing data)")
                    continue
                print(f"  {tag}: AUC={auc:.4f}  ({len(fpr)} ROC points)")
                sanity.append({"dataset": ds, "method": method, "seed": seed, "auc": auc})
                for f, t in zip(fpr, tpr):
                    rows.append({
                        "dataset": ds,
                        "method": method,
                        "seed": seed,
                        "fpr": round(float(f), 6),
                        "tpr": round(float(t), 6),
                    })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out)} rows to {OUT_CSV}")

    # ── sanity check: compare per-(dataset, method) mean AUC against reported ──
    if PRIV_CSV.exists():
        ref = pd.read_csv(PRIV_CSV)
        san = pd.DataFrame(sanity)
        mean_auc = san.groupby(["dataset", "method"])["auc"].mean().reset_index()
        # compute reported mean from per-seed rows
        ref_mean = ref.groupby(["dataset", "method"])["mia_auc"].mean().reset_index()
        ref_mean.rename(columns={"mia_auc": "reported"}, inplace=True)
        print("\n--- Sanity check: recomputed vs reported MIA AUC ---")
        print(f"{'Dataset':<16} {'Method':<18} {'Recomp':>8} {'Reported':>8} {'Diff':>8}")
        print("-" * 62)
        for _, r in mean_auc.iterrows():
            ref_row = ref_mean[(ref_mean["dataset"] == r["dataset"]) & (ref_mean["method"] == r["method"])]
            if ref_row.empty:
                reported = float("nan")
            else:
                reported = float(ref_row["reported"].iloc[0])
            diff = r["auc"] - reported
            flag = " <-- MISMATCH" if abs(diff) > 0.002 else ""
            print(f"{r['dataset']:<16} {r['method']:<18} {r['auc']:>8.4f} {reported:>8.4f} {diff:>+8.4f}{flag}")
    else:
        print(f"(Skipping sanity check: {PRIV_CSV} not found)")


if __name__ == "__main__":
    main()
