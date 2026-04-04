from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _stringify_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [str(col) for col in renamed.columns]
    return renamed


def _split_features(
    real_df: pd.DataFrame, synth_df: pd.DataFrame, label_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    real_X = real_df.drop(columns=[label_col])
    synth_X = synth_df.drop(columns=[label_col])
    y_synth = synth_df[label_col].astype(int).reset_index(drop=True)
    y_real = real_df[label_col].astype(int).reset_index(drop=True)
    return synth_X, real_X, y_synth, y_real


def _infer_column_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    cont_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    bool_cols = features.select_dtypes(include=["bool"]).columns.tolist()
    cont_cols = [col for col in cont_cols if col not in bool_cols]
    cat_cols = [col for col in features.columns if col not in cont_cols]
    if not cat_cols:
        cat_cols = []
    if not cont_cols:
        cont_cols = []
    return cont_cols, cat_cols


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    cont_cols, cat_cols = _infer_column_groups(features)
    transformers = []
    if cont_cols:
        transformers.append(("num", StandardScaler(), list(cont_cols)))
    if cat_cols:
        transformers.append(("cat", _make_ohe(), list(cat_cols)))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_classifier(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(
            solver="saga", max_iter=5000, n_jobs=-1, random_state=seed
        )
    if name == "xgb":
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=seed,
                verbosity=0,
            )
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed") from exc
    raise ValueError(f"Unknown classifier: {name}")


def tstr_eval(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    label_col: str,
    classifier_name: str,
    seed: int,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "classifier": classifier_name,
        "n_synth": len(synth_df),
        "notes": "",
        "converged": None,
        "n_iter": None,
        "status": "success",
    }
    try:
        if label_col not in real_df.columns or label_col not in synth_df.columns:
            raise KeyError(f"Label column {label_col} not found in input data.")
        X_syn_df, X_real_df, y_syn, y_real = _split_features(real_df, synth_df, label_col)
        X_syn_df = _stringify_columns(X_syn_df.reset_index(drop=True))
        X_real_df = _stringify_columns(X_real_df.reset_index(drop=True))
        X_syn_df = X_syn_df.reindex(columns=X_real_df.columns, fill_value=0)
        y_syn = y_syn.to_numpy()
        y_real = y_real.to_numpy()
        if len(np.unique(y_syn)) < 2:
            raise ValueError("Synthetic labels collapsed to a single class.")
        if len(np.unique(y_real)) < 2:
            raise ValueError("Real labels have <2 classes; cannot evaluate TSTR.")
        preprocessor = _build_preprocessor(pd.concat([X_syn_df, X_real_df], axis=0))
        clf = _build_classifier(classifier_name, seed)
        model = Pipeline([("preprocess", preprocessor), ("classifier", clf)])
        model.fit(X_syn_df, y_syn)
        y_pred = model.predict(X_real_df)
        result["accuracy"] = accuracy_score(y_real, y_pred)
        result["f1_macro"] = f1_score(y_real, y_pred, average="macro")
        if len(np.unique(y_real)) == 2:
            try:
                y_score = model.predict_proba(X_real_df)[:, 1]
            except AttributeError:
                y_score = model.decision_function(X_real_df)
            result["roc_auc"] = roc_auc_score(y_real, y_score)
        else:
            result["roc_auc"] = None
        for metric in ("accuracy", "f1_macro", "roc_auc"):
            value = result.get(metric)
            if isinstance(value, float) and np.isnan(value):
                raise ValueError(f"TSTR metric {metric} is NaN")
        if classifier_name == "logistic_regression":
            estimator = model.named_steps.get("classifier")
            if estimator is not None and hasattr(estimator, "n_iter_"):
                n_iter = estimator.n_iter_
                max_iter = getattr(estimator, "max_iter", None)
                if isinstance(n_iter, np.ndarray):
                    n_iter_value = int(n_iter.max())
                else:
                    n_iter_value = int(n_iter)
                result["n_iter"] = n_iter_value
                if max_iter is not None:
                    result["converged"] = n_iter_value < int(max_iter)
                else:
                    result["converged"] = True
            else:
                result["converged"] = False
        else:
            result["converged"] = True
    except Exception as exc:
        result["notes"] = str(exc)
        result["accuracy"] = None
        result["f1_macro"] = None
        result["roc_auc"] = None
        result["converged"] = False
        result["status"] = "FAIL"
    return result
