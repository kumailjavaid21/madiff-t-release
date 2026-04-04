from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .datasets import DatasetSplit


def _feature_columns(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    return df.drop(columns=[label_col])


def _encode_features(features: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(features, drop_first=False)


def _class_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def _compute_imbalance(counts: dict[str, int]) -> float:
    if not counts:
        return 0.0
    values = [v for v in counts.values() if v > 0]
    if not values:
        return 0.0
    return max(values) / min(values)


def summarize_dataset(split: DatasetSplit) -> dict[str, object]:
    features = _feature_columns(split.df, split.label_col)
    encoded = _encode_features(features)
    cont_mask = features.select_dtypes(include=["number"]).columns
    cat_mask = [c for c in features.columns if c not in cont_mask]

    counts = _class_counts(split.train[split.label_col])
    missing = features.isna().sum().sum()
    total_entries = len(split.df) * max(features.shape[1], 1)
    missing_rate = float(missing / total_entries) if total_entries else 0.0

    return {
        "dataset": split.name,
        "domain": split.domain,
        "n_rows_total": len(split.df),
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_test": len(split.test),
        "n_features_raw": features.shape[1],
        "n_features_encoded": encoded.shape[1],
        "n_cont": len(cont_mask),
        "n_cat": len(cat_mask),
        "label_col": split.label_col,
        "n_classes": len(counts),
        "class_counts_train": json.dumps(counts),
        "missing_rate": missing_rate,
        "imbalance_ratio": _compute_imbalance(counts),
    }


def export_dataset_summary(
    dataset_splits: Iterable[DatasetSplit],
    tables_dir: Path,
) -> Path:
    rows = [summarize_dataset(split) for split in dataset_splits]
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / "dataset_summary.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path
