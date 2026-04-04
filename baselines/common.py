from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class GenerationResult:
    synthetic_df: pd.DataFrame
    backend: str
    notes: str


def cast_like_train(train_df: pd.DataFrame, synth_df: pd.DataFrame) -> pd.DataFrame:
    casted = synth_df.copy()
    for col in train_df.columns:
        target_dtype = train_df[col].dtype
        if pd.api.types.is_integer_dtype(target_dtype):
            casted[col] = np.round(pd.to_numeric(casted[col], errors="coerce")).fillna(0).astype(target_dtype)
        elif pd.api.types.is_float_dtype(target_dtype):
            casted[col] = pd.to_numeric(casted[col], errors="coerce").astype(target_dtype)
        elif pd.api.types.is_bool_dtype(target_dtype):
            casted[col] = casted[col].astype("boolean").fillna(False).astype(target_dtype)
        else:
            casted[col] = casted[col].astype(train_df[col].dtype, errors="ignore")
    return casted


def bootstrap_generate(train_df: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    synth = pd.DataFrame(index=range(n_rows), columns=train_df.columns)

    for col in train_df.columns:
        series = train_df[col]
        non_null = series.dropna()
        if non_null.empty:
            synth[col] = np.nan
            continue

        sampled = rng.choice(non_null.to_numpy(), size=n_rows, replace=True)
        if pd.api.types.is_numeric_dtype(series):
            sampled = pd.to_numeric(sampled, errors="coerce")
            sampled = np.asarray(sampled, dtype=np.float64)
            col_std = float(np.nanstd(non_null.to_numpy(dtype=np.float64)))
            if np.isfinite(col_std) and col_std > 0:
                noise = rng.normal(loc=0.0, scale=0.01 * col_std, size=n_rows)
                sampled = sampled + noise
                col_min = float(np.nanmin(non_null.to_numpy(dtype=np.float64)))
                col_max = float(np.nanmax(non_null.to_numpy(dtype=np.float64)))
                sampled = np.clip(sampled, col_min, col_max)
            if pd.api.types.is_integer_dtype(series):
                sampled = np.round(sampled)
            synth[col] = sampled
        else:
            synth[col] = sampled

    synth = cast_like_train(train_df, synth)
    return synth


def sdv_ctgan_generate(train_df: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    from sdv.metadata import Metadata
    from sdv.single_table import CTGANSynthesizer

    model_df = train_df.copy()
    for col in model_df.columns:
        if str(model_df[col].dtype) == "category":
            model_df[col] = model_df[col].astype("object")

    metadata = Metadata.detect_from_dataframe(model_df)
    synth = CTGANSynthesizer(metadata=metadata, epochs=200, verbose=False)
    synth.fit(model_df)
    sampled = synth.sample(num_rows=n_rows)
    sampled = sampled.reindex(columns=train_df.columns)
    return cast_like_train(train_df, sampled)


def ensure_schema(train_df: pd.DataFrame, synth_df: pd.DataFrame) -> pd.DataFrame:
    if list(synth_df.columns) != list(train_df.columns):
        if synth_df.shape[1] != train_df.shape[1]:
            raise RuntimeError(
                "Synthetic schema mismatch. "
                f"expected {len(train_df.columns)} columns, got {len(synth_df.columns)}."
            )
        synth_df = synth_df.copy()
        synth_df.columns = list(train_df.columns)
    return cast_like_train(train_df, synth_df)


def write_status(
    outdir: Path,
    method: str,
    status: str,
    backend: str,
    seed: int,
    notes: str,
    n_rows: Optional[int] = None,
) -> None:
    payload: Dict[str, object] = {
        "method": method,
        "status": status,
        "backend": backend,
        "seed": seed,
        "notes": notes,
    }
    if n_rows is not None:
        payload["n_rows"] = int(n_rows)

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "generator_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
