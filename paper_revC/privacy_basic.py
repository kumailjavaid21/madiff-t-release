from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score


def _select_columns(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    return df.drop(columns=[label_col])


def _identify_column_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [
        col
        for col in features.columns
        if pd.api.types.is_numeric_dtype(features[col]) and features[col].dtype != bool
    ]
    categorical = [col for col in features.columns if col not in numeric]
    return numeric, categorical


def _build_category_mappings(features: pd.DataFrame, cat_cols: list[str]) -> dict[str, dict]:
    mappings: dict[str, dict] = {}
    for col in cat_cols:
        categories = pd.Categorical(features[col]).categories
        mappings[col] = {cat: idx for idx, cat in enumerate(categories)}
    return mappings


def _encode_categories(df: pd.DataFrame, mappings: dict[str, dict]) -> np.ndarray:
    if not mappings:
        return np.zeros((len(df), 0))
    encoded = []
    for col, mapping in mappings.items():
        mapped = df[col].map(mapping)
        mapped_array = pd.to_numeric(mapped, errors="coerce").to_numpy(dtype=float)
        mapped_array = np.where(np.isnan(mapped_array), -1.0, mapped_array)
        encoded.append(mapped_array)
    return np.vstack(encoded).T


def _normalize_numeric(
    df: pd.DataFrame, cont_cols: list[str], means: pd.Series, stds: pd.Series
) -> np.ndarray:
    if not cont_cols:
        return np.zeros((len(df), 0))
    normalized = (df[cont_cols] - means) / stds.replace(0, 1)
    return normalized.to_numpy(dtype=float)


def _sample_frame(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    size = min(len(df), limit)
    if size == 0:
        return df.iloc[0:0]
    return df.sample(size, random_state=seed).reset_index(drop=True)


def _mixed_distances(
    cont_a: np.ndarray,
    cont_b: np.ndarray,
    cat_a: np.ndarray,
    cat_b: np.ndarray,
) -> np.ndarray:
    if cont_a.size and cont_b.size:
        dist_cont = cdist(cont_a, cont_b, metric="euclidean")
    else:
        dist_cont = np.zeros((cont_a.shape[0], cat_b.shape[0]))
    if cat_a.size and cat_b.size:
        dist_cat = cdist(cat_a, cat_b, metric="hamming")
    else:
        dist_cat = np.zeros_like(dist_cont)
    return np.sqrt(dist_cont ** 2 + dist_cat ** 2)


def privacy_audit(
    real_train: pd.DataFrame,
    real_val: pd.DataFrame,
    synth_df: pd.DataFrame,
    label_col: str,
    q: float = 0.01,
    alpha: float = 0.5,
    seed: int = 0,
) -> dict[str, object]:
    features_train = _select_columns(real_train, label_col)
    cont_cols, cat_cols = _identify_column_groups(features_train)
    cat_mappings = _build_category_mappings(features_train, cat_cols)
    means = features_train[cont_cols].mean() if cont_cols else pd.Series(dtype=float)
    stds = features_train[cont_cols].std().replace(0, 1) if cont_cols else pd.Series(dtype=float)

    real_sample = _sample_frame(features_train, limit=500, seed=seed)
    syn_sample = _sample_frame(_select_columns(synth_df, label_col), limit=800, seed=seed + 5)
    if syn_sample.empty or real_sample.empty:
        logging.warning("Synthetic or real sample is empty for privacy audit.")
        distances = np.array([np.nan])
    else:
        real_cont = _normalize_numeric(real_sample, cont_cols, means, stds)
        syn_cont = _normalize_numeric(syn_sample, cont_cols, means, stds)
        real_cat = _encode_categories(real_sample, cat_mappings)
        syn_cat = _encode_categories(syn_sample, cat_mappings)
        distances = _mixed_distances(syn_cont, real_cont, syn_cat, real_cat)

    if distances.size:
        min_distances = distances.min(axis=1)
        dcr_value = float(np.quantile(min_distances, q))
    else:
        dcr_value = float("nan")

    loo_sample = _sample_frame(features_train, limit=400, seed=seed + 2)
    norm_dcr = float("nan")
    if len(loo_sample) >= 2:
        loo_cont = _normalize_numeric(loo_sample, cont_cols, means, stds)
        loo_cat = _encode_categories(loo_sample, cat_mappings)
        pairwise = _mixed_distances(loo_cont, loo_cont, loo_cat, loo_cat)
        np.fill_diagonal(pairwise, np.inf)
        min_real = pairwise.min(axis=1)
        real_quantile = float(np.quantile(min_real, q)) if min_real.size else float("nan")
        if real_quantile and not np.isnan(real_quantile):
            norm_dcr = dcr_value / real_quantile if real_quantile else float("nan")

    mia_auc = float("nan")
    train_sample = _sample_frame(real_train, limit=300, seed=seed + 10)
    val_source = real_val if len(real_val) else real_train
    val_sample = _sample_frame(val_source, limit=300, seed=seed + 11)
    if len(train_sample) and len(val_sample) and not syn_sample.empty:
        train_features = _select_columns(train_sample, label_col)
        val_features = _select_columns(val_sample, label_col)
        train_cont = _normalize_numeric(train_features, cont_cols, means, stds)
        val_cont = _normalize_numeric(val_features, cont_cols, means, stds)
        train_cat = _encode_categories(train_features, cat_mappings)
        val_cat = _encode_categories(val_features, cat_mappings)
        syn_cont = _normalize_numeric(syn_sample, cont_cols, means, stds)
        syn_cat = _encode_categories(syn_sample, cat_mappings)
        train_dist = _mixed_distances(train_cont, syn_cont, train_cat, syn_cat).min(axis=1)
        val_dist = _mixed_distances(val_cont, syn_cont, val_cat, syn_cat).min(axis=1)
        labels = np.concatenate([np.ones(len(train_dist)), np.zeros(len(val_dist))])
        scores = -np.concatenate([train_dist, val_dist])
        if len(np.unique(labels)) > 1:
            mia_auc = roc_auc_score(labels, scores)

    return {
        "dcr_q": dcr_value,
        "norm_dcr": norm_dcr,
        "mia_auc": mia_auc,
        "alpha": alpha,
        "q": q,
        "notes": f"Samples: real({len(real_sample)}), syn({len(syn_sample)})",
    }
