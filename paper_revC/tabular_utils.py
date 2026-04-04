from __future__ import annotations

from typing import Iterable

import logging

import numpy as np
import pandas as pd


def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list]]:
    encoded = df.copy()
    cat_cols = [
        col
        for col in df.columns
        if (df[col].dtype == object)
        or (str(df[col].dtype) == "category")
        or (df[col].dtype == bool)
        or pd.api.types.is_string_dtype(df[col])
    ]
    mappings: dict[str, list] = {}
    for col in cat_cols:
        categories = pd.Categorical(df[col]).categories
        mappings[col] = list(categories)
        encoded[col] = pd.Categorical(df[col], categories=categories).codes
    return encoded, mappings


def decode_categorical(df: pd.DataFrame, mappings: dict[str, list]) -> pd.DataFrame:
    decoded = df.copy()
    for col, categories in mappings.items():
        if col not in decoded.columns:
            continue
        if not categories:
            continue
        idx = np.round(decoded[col].to_numpy()).astype(int)
        idx = np.clip(idx, 0, len(categories) - 1)
        decoded[col] = pd.Categorical.from_codes(idx, categories=categories)
    return decoded


def sanitize_synthetic(
    df: pd.DataFrame, reference: pd.DataFrame, label_col: str
) -> pd.DataFrame:
    sanitized = df.copy()
    for col in sanitized.columns:
        if pd.api.types.is_numeric_dtype(reference[col]):
            ref_series = pd.to_numeric(reference[col], errors="coerce")
            col_series = pd.to_numeric(sanitized[col], errors="coerce")
            fill_val = float(ref_series.mean()) if ref_series.notna().any() else 0.0
            col_series = col_series.fillna(fill_val)
            col_series = col_series.clip(ref_series.min(), ref_series.max())
            sanitized[col] = col_series
        else:
            ref_series = reference[col].astype(str)
            mode = ref_series.mode(dropna=True)
            fill_val = mode.iloc[0] if not mode.empty else ref_series.dropna().iloc[0]
            sanitized[col] = sanitized[col].astype(str)
            sanitized[col] = sanitized[col].where(
                sanitized[col].isin(ref_series.unique()), fill_val
            )
            sanitized[col] = sanitized[col].fillna(fill_val)
    if label_col in sanitized.columns and label_col in reference.columns:
        ref_labels = reference[label_col].astype(str)
        fill_val = ref_labels.mode(dropna=True).iloc[0]
        sanitized[label_col] = sanitized[label_col].astype(str)
        sanitized[label_col] = sanitized[label_col].where(
            sanitized[label_col].isin(ref_labels.unique()), fill_val
        )
    return sanitized


def discretize_continuous(
    df: pd.DataFrame, cont_cols: Iterable[str], n_bins: int
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    discretized = df.copy()
    bin_info: dict[str, np.ndarray] = {}
    for col in cont_cols:
        series = pd.to_numeric(discretized[col], errors="coerce")
        values = series.dropna().to_numpy()
        if len(values) == 0:
            logging.info("Discretize: skipping %s (empty values).", col)
            discretized[col] = series.fillna(0.0)
            continue
        variance = float(np.nanvar(values))
        value_range = float(np.nanmax(values) - np.nanmin(values))
        unique_count = int(np.unique(values).size)
        if variance <= 1e-12 or value_range <= 1e-8:
            logging.info(
                "Discretize: skipping %s (near-constant: var=%s, range=%s).",
                col,
                variance,
                value_range,
            )
            fill_val = float(np.nanmean(values)) if len(values) else 0.0
            discretized[col] = series.fillna(fill_val)
            continue
        if unique_count < 3:
            logging.info("Discretize: skipping %s (unique<3).", col)
            fill_val = float(np.nanmean(values)) if len(values) else 0.0
            discretized[col] = series.fillna(fill_val)
            continue
        col_bins = max(2, min(n_bins, unique_count - 1))
        quantiles = np.linspace(0.0, 1.0, col_bins + 1)
        edges = np.unique(np.quantile(values, quantiles))
        if len(edges) < 2:
            min_val = float(np.nanmin(values))
            max_val = float(np.nanmax(values))
            edges = np.array([min_val, max_val])
        if len(edges) < 2 or float(edges[0]) == float(edges[-1]):
            base = float(edges[0]) if len(edges) else 0.0
            edges = np.array([base, base + 1.0])
        bin_info[col] = edges
        discretized[col] = np.digitize(series.fillna(edges[0]), edges[1:-1], right=False)
    return discretized, bin_info


def inverse_discretize(
    df: pd.DataFrame, bin_info: dict[str, np.ndarray]
) -> pd.DataFrame:
    restored = df.copy()
    for col, edges in bin_info.items():
        if col not in restored.columns:
            continue
        idx = pd.to_numeric(restored[col], errors="coerce").fillna(0).astype(int)
        idx = idx.clip(lower=0, upper=len(edges) - 2)
        mids = (edges[:-1] + edges[1:]) / 2.0
        restored[col] = mids[idx.to_numpy()]
    return restored


def parse_privbayes_method_name(method: str) -> float | None:
    if method.startswith("privbayes_eps"):
        suffix = method.replace("privbayes_eps", "")
        try:
            return float(suffix)
        except ValueError:
            return None
    return None
