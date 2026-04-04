from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from urllib.request import urlretrieve
from zipfile import ZipFile
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplit:
    name: str
    domain: str
    label_col: str
    df: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    description: str


def _drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    if df.isna().any().any():
        return df.dropna().reset_index(drop=True)
    return df


def _load_breast_cancer() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df.columns = list(data.feature_names) + ["target"]
    df["target"] = df["target"].astype(int)
    return df


def _cache_file(cache_dir: Path, url: str, filename: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if not target.exists():
        urlretrieve(url, target)
    return target


def _load_diabetes130_hospitals() -> pd.DataFrame:
    cache_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "diabetes130_hospitals"
    archive_path = _cache_file(
        cache_dir,
        "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
        "diabetes130_us_hospitals.zip",
    )
    csv_path = cache_dir / "diabetic_data.csv"
    if not csv_path.exists():
        with ZipFile(archive_path) as zf:
            zf.extract("diabetic_data.csv", path=cache_dir)

    df = pd.read_csv(csv_path).copy()
    df = df.drop(
        columns=[
            "encounter_id",
            "patient_nbr",
            "weight",
            "payer_code",
            "medical_specialty",
        ]
    )
    df["readmitted"] = (df["readmitted"].astype(str).str.strip() == "<30").astype(int)

    categorical_cols = [
        "race",
        "gender",
        "age",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "diag_1",
        "diag_2",
        "diag_3",
        "max_glu_serum",
        "A1Cresult",
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "examide",
        "citoglipton",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
        "change",
        "diabetesMed",
    ]
    continuous_cols = [
        col
        for col in df.columns
        if col not in categorical_cols and col != "readmitted"
    ]

    for col in categorical_cols:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("?", pd.NA)
        mode = df[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "missing"
        df[col] = df[col].fillna(fill_value)
        counts = df[col].value_counts(dropna=False)
        if len(counts) > 20:
            keep = set(counts.head(20).index.tolist())
            df[col] = df[col].where(df[col].isin(keep), "other")
        df[col] = df[col].astype("string")

    for col in continuous_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df.reset_index(drop=True)


def _load_bank_marketing() -> pd.DataFrame:
    cache_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "bank_marketing"
    archive_path = _cache_file(
        cache_dir,
        "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
        "bank_marketing.zip",
    )
    csv_path = cache_dir / "bank-additional" / "bank-additional-full.csv"
    if not csv_path.exists():
        with ZipFile(archive_path) as zf:
            zf.extractall(path=cache_dir)
        nested_archive = cache_dir / "bank-additional.zip"
        if nested_archive.exists() and not csv_path.exists():
            with ZipFile(nested_archive) as zf:
                zf.extractall(path=cache_dir)

    df = pd.read_csv(csv_path, sep=";").copy()
    df["y"] = (df["y"].astype(str).str.strip().str.lower() == "yes").astype(int)

    categorical_cols = [
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "poutcome",
    ]
    continuous_cols = [col for col in df.columns if col not in categorical_cols and col != "y"]

    for col in categorical_cols:
        df[col] = df[col].astype("string").str.strip()

    for col in continuous_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def _load_openml_dataset(
    name: str,
    version: int | None = 1,
    label_col: str | None = None,
    label_map: Dict[str, int] | Dict[int, int] | None = None,
) -> tuple[pd.DataFrame, str]:
    frame = fetch_openml(name, version=version, as_frame=True).frame.copy()
    frame = _drop_missing(frame)
    if label_col is None:
        label_col = frame.columns[-1]
    if label_col not in frame.columns:
        raise KeyError(f"Label column {label_col} not found in dataset {name}")
    if label_map:
        original = frame[label_col]
        mapped = original.map(label_map)
        frame[label_col] = mapped.where(mapped.notna(), original)
    try:
        frame[label_col] = frame[label_col].astype(int)
    except Exception:
        try:
            frame[label_col] = frame[label_col].map(lambda x: int(float(x)))
        except Exception:
            frame[label_col] = frame[label_col].astype("category").cat.codes
    return frame, label_col


DATASET_REGISTRY: dict[str, dict] = {
    "breast_cancer": {
        "loader": _load_breast_cancer,
        "domain": "health",
        "label_col": "target",
        "description": "Breast cancer diagnostic features (sklearn)",
    },
    "adult_income": {
        "loader": lambda: _load_openml_dataset(
            "adult", version=2, label_col="class", label_map={"<=50K": 0, ">50K": 1}
        )[0],
        "domain": "census",
        "label_col": "class",
        "description": "UCI Adult census income",
    },
    "pima": {
        "loader": lambda: _load_openml_dataset(
            "pima-indians-diabetes", version=1, label_col="Outcome"
        )[0],
        "domain": "health",
        "label_col": "Outcome",
        "description": "Pima Indians Diabetes dataset",
    },
    "heart": {
        "loader": lambda: _load_openml_dataset(
            "heart-statlog",
            version=1,
            label_col="class",
            label_map={"present": 1, "absent": 0},
        )[0],
        "domain": "health",
        "label_col": "class",
        "description": "Statlog Heart disease (openml)",
    },
    "credit": {
        "loader": lambda: _load_openml_dataset(
            "credit-g",
            version=1,
            label_col="class",
            label_map={"good": 1, "bad": 0},
        )[0],
        "domain": "finance",
        "label_col": "class",
        "description": "German Credit dataset (openml)",
    },
    "diabetes130_hospitals": {
        "loader": _load_diabetes130_hospitals,
        "domain": "health",
        "label_col": "readmitted",
        "description": "Diabetes 130-US Hospitals 1999-2008 (UCI)",
    },
    "bank_marketing": {
        "loader": _load_bank_marketing,
        "domain": "finance",
        "label_col": "y",
        "description": "UCI Bank Marketing (bank-additional-full)",
    },
}


def load_and_split_dataset(
    name: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> DatasetSplit:
    key = name.lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {name} is not registered.")
    registry = DATASET_REGISTRY[key]
    df = registry["loader"]()
    domain = registry["domain"]
    label_col = registry["label_col"]
    description = registry.get("description", "")
    df = _drop_missing(df)
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col} missing after loading dataset {name}.")
    df = df.reset_index(drop=True)
    stratify = df[label_col] if df[label_col].nunique() <= 20 else None
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    if val_size <= 0 or len(train_val) == 0:
        train = train_val
        val = train_val.iloc[0:0]
    else:
        val_fraction = min(max(val_size / max(1.0 - test_size, 1e-6), 0.0), 0.5)
        stratify_tv = train_val[label_col] if train_val[label_col].nunique() <= 20 else None
        train, val = train_test_split(
            train_val,
            test_size=val_fraction,
            random_state=random_state,
            stratify=stratify_tv,
        )
    return DatasetSplit(
        name=name,
        domain=domain,
        label_col=label_col,
        df=df,
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        description=description,
    )
