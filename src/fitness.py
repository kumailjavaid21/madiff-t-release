from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score


def mixed_type_distance(
    x: np.ndarray,
    xprime: np.ndarray,
    alpha: float,
    cont_idx: Iterable[int],
    cat_idx: Iterable[int],
) -> float:
    cont_idx = list(cont_idx)
    cat_idx = list(cat_idx)
    if cont_idx:
        cont_dist = np.linalg.norm(x[cont_idx] - xprime[cont_idx])
    else:
        cont_dist = 0.0
    if cat_idx:
        cat_dist = np.mean(x[cat_idx] != xprime[cat_idx])
    else:
        cat_dist = 0.0
    return float(np.sqrt((alpha * cont_dist) ** 2 + ((1 - alpha) * cat_dist) ** 2))


def _identify_column_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [
        col
        for col in features.columns
        if pd.api.types.is_numeric_dtype(features[col]) and features[col].dtype != bool
    ]
    categorical = [col for col in features.columns if col not in numeric]
    return numeric, categorical


def _encode_categories(df: pd.DataFrame, cat_cols: list[str]) -> np.ndarray:
    if not cat_cols:
        return np.zeros((len(df), 0))
    encoded = []
    for col in cat_cols:
        categories = pd.Categorical(df[col]).categories
        mapping = {cat: idx for idx, cat in enumerate(categories)}
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


def _mixed_distance_matrix(
    cont_a: np.ndarray,
    cont_b: np.ndarray,
    cat_a: np.ndarray,
    cat_b: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if cont_a.size and cont_b.size:
        dist_cont = cdist(cont_a, cont_b, metric="euclidean")
    else:
        dist_cont = np.zeros((cont_a.shape[0], cont_b.shape[0]))
    if cat_a.size and cat_b.size:
        dist_cat = cdist(cat_a, cat_b, metric="hamming")
    else:
        dist_cat = np.zeros_like(dist_cont)
    return np.sqrt((alpha * dist_cont) ** 2 + ((1 - alpha) * dist_cat) ** 2)


def compute_normdcr_q(
    synth: pd.DataFrame,
    real_train: pd.DataFrame,
    q: float,
    alpha: float,
    seed: int = 0,
    max_real: int = 500,
    max_synth: int = 800,
) -> float:
    rng = np.random.default_rng(seed)
    real_sample = real_train.sample(
        min(len(real_train), max_real), random_state=seed
    ).reset_index(drop=True)
    synth_sample = synth.sample(
        min(len(synth), max_synth), random_state=seed + 17
    ).reset_index(drop=True)
    if real_sample.empty or synth_sample.empty:
        return float("nan")

    cont_cols, cat_cols = _identify_column_groups(real_train)
    means = real_train[cont_cols].mean() if cont_cols else pd.Series(dtype=float)
    stds = real_train[cont_cols].std().replace(0, 1) if cont_cols else pd.Series(dtype=float)
    real_cont = _normalize_numeric(real_sample, cont_cols, means, stds)
    synth_cont = _normalize_numeric(synth_sample, cont_cols, means, stds)
    real_cat = _encode_categories(real_sample, cat_cols)
    synth_cat = _encode_categories(synth_sample, cat_cols)
    distances = _mixed_distance_matrix(synth_cont, real_cont, synth_cat, real_cat, alpha)
    if distances.size == 0:
        return float("nan")
    min_distances = distances.min(axis=1)
    dcr_value = float(np.quantile(min_distances, q))

    # LOO baseline on real data
    loo_sample = real_train.sample(
        min(len(real_train), max_real), random_state=seed + 31
    ).reset_index(drop=True)
    if len(loo_sample) < 2:
        return float("nan")
    loo_cont = _normalize_numeric(loo_sample, cont_cols, means, stds)
    loo_cat = _encode_categories(loo_sample, cat_cols)
    pairwise = _mixed_distance_matrix(loo_cont, loo_cont, loo_cat, loo_cat, alpha)
    np.fill_diagonal(pairwise, np.inf)
    min_real = pairwise.min(axis=1)
    real_quantile = float(np.quantile(min_real, q)) if min_real.size else float("nan")
    if real_quantile and not np.isnan(real_quantile):
        return float(dcr_value / real_quantile)
    return float("nan")


def compute_mia_auc_distance(
    real_members: pd.DataFrame,
    real_nonmembers: pd.DataFrame,
    synth: pd.DataFrame,
    alpha: float,
    seed: int = 0,
    max_members: int = 300,
    max_nonmembers: int = 300,
    max_synth: int = 800,
) -> float:
    if real_members.empty or real_nonmembers.empty or synth.empty:
        return float("nan")
    members = real_members.sample(
        min(len(real_members), max_members), random_state=seed
    ).reset_index(drop=True)
    nonmembers = real_nonmembers.sample(
        min(len(real_nonmembers), max_nonmembers), random_state=seed + 7
    ).reset_index(drop=True)
    synth_sample = synth.sample(
        min(len(synth), max_synth), random_state=seed + 11
    ).reset_index(drop=True)

    cont_cols, cat_cols = _identify_column_groups(members)
    combined = pd.concat([members, nonmembers], axis=0)
    means = combined[cont_cols].mean() if cont_cols else pd.Series(dtype=float)
    stds = combined[cont_cols].std().replace(0, 1) if cont_cols else pd.Series(dtype=float)

    mem_cont = _normalize_numeric(members, cont_cols, means, stds)
    non_cont = _normalize_numeric(nonmembers, cont_cols, means, stds)
    syn_cont = _normalize_numeric(synth_sample, cont_cols, means, stds)
    mem_cat = _encode_categories(members, cat_cols)
    non_cat = _encode_categories(nonmembers, cat_cols)
    syn_cat = _encode_categories(synth_sample, cat_cols)

    mem_dist = _mixed_distance_matrix(mem_cont, syn_cont, mem_cat, syn_cat, alpha).min(axis=1)
    non_dist = _mixed_distance_matrix(non_cont, syn_cont, non_cat, syn_cat, alpha).min(axis=1)
    labels = np.concatenate([np.ones(len(mem_dist)), np.zeros(len(non_dist))])
    scores = -np.concatenate([mem_dist, non_dist])
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_mia_advantage(auc: float) -> float:
    if auc is None or (isinstance(auc, float) and np.isnan(auc)):
        return float("nan")
    return float(2 * abs(auc - 0.5))


def minmax_normalize(values: Iterable[float], eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    denom = vmax - vmin
    if denom < eps or np.isnan(denom):
        return np.zeros_like(arr)
    return (arr - vmin) / (denom + eps)


def compute_fitness_per_generation(
    metrics_df: pd.DataFrame, weights: dict[str, float]
) -> pd.Series:
    required = ["utility", "fidelity", "normdcr", "mia_adv"]
    missing = [col for col in required if col not in metrics_df.columns]
    if missing:
        raise ValueError(f"Missing fitness columns: {missing}")
    u_norm = minmax_normalize(metrics_df["utility"].to_numpy())
    f_norm = minmax_normalize(metrics_df["fidelity"].to_numpy())
    # Cap runaway DCR values before privacy scoring.
    NORMDCR_CAP = 3.0
    normdcr_raw = metrics_df["normdcr"].to_numpy(dtype=float)
    normdcr_capped = np.minimum(normdcr_raw, NORMDCR_CAP)
    ndcr_norm = minmax_normalize(normdcr_capped)
    mia_norm = minmax_normalize(metrics_df["mia_adv"].to_numpy())
    privacy_score = 0.5 * ndcr_norm + 0.5 * (1.0 - mia_norm)

    wu = float(weights.get("utility", 1.0))
    wf = float(weights.get("fidelity", 1.0))
    wp = float(weights.get("privacy", 1.0))
    return pd.Series(wu * u_norm + wf * f_norm + wp * privacy_score, index=metrics_df.index)
