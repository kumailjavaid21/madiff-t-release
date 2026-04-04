from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _stringify_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [str(col) for col in renamed.columns]
    return renamed


def build_feature_preprocessor(
    train_df: pd.DataFrame, label_col: str
) -> tuple[ColumnTransformer, list[str], list[str], list[str]]:
    features = _stringify_columns(train_df.drop(columns=[label_col]))
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    bool_cols = features.select_dtypes(include=["bool"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in bool_cols]
    categorical_cols = [col for col in features.columns if col not in numeric_cols]
    transformers: list[tuple[str, object, Iterable[str]]] = []
    if numeric_cols:
        transformers.append(("num", "passthrough", numeric_cols))
    if categorical_cols:
        transformers.append(("cat", _make_ohe(), categorical_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    preprocessor.fit(features)
    feature_names = list(preprocessor.get_feature_names_out())
    return preprocessor, feature_names, numeric_cols, categorical_cols


def encode_features(
    preprocessor: ColumnTransformer, df: pd.DataFrame, label_col: str
) -> np.ndarray:
    features = _stringify_columns(df.drop(columns=[label_col]))
    encoded = preprocessor.transform(features)
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    return np.asarray(encoded, dtype=np.float32)
