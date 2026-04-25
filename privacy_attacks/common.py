from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder


MISSING_TOKEN = "__MISSING__"


@dataclass
class EncodedData:
    X_train: np.ndarray
    X_test: np.ndarray
    X_synth: np.ndarray
    feature_names: List[str]


def _build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def align_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    aligned = df.copy()
    missing = [c for c in columns if c not in aligned.columns]
    for col in missing:
        aligned[col] = np.nan
    aligned = aligned.loc[:, list(columns)]
    return aligned


def encode_mixed(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> EncodedData:
    columns = list(train_df.columns)
    test_df = align_columns(test_df, columns)
    synth_df = align_columns(synth_df, columns)

    combined = pd.concat([train_df, test_df, synth_df], axis=0, ignore_index=True)
    num_cols = [c for c in columns if pd.api.types.is_numeric_dtype(combined[c])]
    cat_cols = [c for c in columns if c not in num_cols]

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in num_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            fill_value = pd.to_numeric(train_df[col], errors="coerce").median()
            if pd.isna(fill_value):
                fill_value = 0.0
            out[col] = out[col].fillna(fill_value)
        for col in cat_cols:
            out[col] = out[col].astype("string").fillna(MISSING_TOKEN)
        return out

    train_p = _prep(train_df)
    test_p = _prep(test_df)
    synth_p = _prep(synth_df)
    union_df = pd.concat([train_p, test_p, synth_p], axis=0, ignore_index=True)

    transformer = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", _build_one_hot_encoder(), cat_cols),
        ],
        remainder="drop",
    )
    transformer.fit(union_df)

    X_train = transformer.transform(train_p).astype(np.float32, copy=False)
    X_test = transformer.transform(test_p).astype(np.float32, copy=False)
    X_synth = transformer.transform(synth_p).astype(np.float32, copy=False)

    feature_names: List[str] = []
    if num_cols:
        feature_names.extend(num_cols)
    if cat_cols:
        encoder = transformer.named_transformers_["cat"]
        feature_names.extend(encoder.get_feature_names_out(cat_cols).tolist())

    return EncodedData(
        X_train=X_train,
        X_test=X_test,
        X_synth=X_synth,
        feature_names=feature_names,
    )


def distance_features(
    X_points: np.ndarray,
    X_reference: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    if X_reference.shape[0] == 0:
        raise ValueError("Reference set is empty; cannot compute distance features.")
    n_neighbors = max(1, min(k, X_reference.shape[0]))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nbrs.fit(X_reference)
    dists, _ = nbrs.kneighbors(X_points)
    feats = np.column_stack(
        [
            dists.min(axis=1),
            dists.mean(axis=1),
            dists.std(axis=1),
        ]
    )
    return feats.astype(np.float32, copy=False)
