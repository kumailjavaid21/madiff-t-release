from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_revC.datasets import load_and_split_dataset
from paper_revC.tstr_eval import tstr_eval


DATASET_DISPLAY = {
    "breast_cancer": "BreastCancer",
    "adult_income": "AdultIncome",
    "pima": "PIMA",
    "heart": "Heart",
    "credit": "GermanCredit",
    "diabetes130_hospitals": "Diabetes130Hospitals",
    "bank_marketing": "BankMarketing",
}


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _metric_row(dataset: str, seed: int, classifier: str, metrics: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric_name in ("accuracy", "f1_macro", "roc_auc"):
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "classifier": classifier,
                "metric_name": metric_name,
                "metric_value": metrics.get(metric_name),
                "status": metrics.get("status", "unknown"),
                "notes": metrics.get("notes", ""),
                "source_file": "paper_revC/tstr_eval.py",
                "source_function": "tstr_eval",
                "split_source_file": "paper_revC/datasets.py",
                "split_source_function": "load_and_split_dataset",
                "split_test_size": 0.2,
                "split_val_size": 0.1,
            }
        )
    return rows


def _build_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    ok = raw_df[raw_df["status"].astype(str).str.lower().isin(["success", "ok"])].copy()
    ok["metric_value"] = pd.to_numeric(ok["metric_value"], errors="coerce")
    ok = ok.dropna(subset=["metric_value"])
    summary = (
        ok.groupby(["dataset", "metric_name"])["metric_value"]
        .agg(["mean", "std"])
        .reset_index()
        .pivot(index="dataset", columns="metric_name", values=["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for stat, metric in summary.columns]
    summary = summary.reset_index()
    for col in (
        "accuracy_mean",
        "accuracy_std",
        "f1_macro_mean",
        "f1_macro_std",
        "roc_auc_mean",
        "roc_auc_std",
    ):
        if col not in summary.columns:
            summary[col] = float("nan")
    summary["Dataset"] = summary["dataset"].map(DATASET_DISPLAY).fillna(summary["dataset"])
    summary = summary[
        [
            "Dataset",
            "accuracy_mean",
            "accuracy_std",
            "f1_macro_mean",
            "f1_macro_std",
            "roc_auc_mean",
            "roc_auc_std",
        ]
    ].rename(
        columns={
            "accuracy_mean": "Accuracy Mean",
            "accuracy_std": "Accuracy Std",
            "f1_macro_mean": "F1 Macro Mean",
            "f1_macro_std": "F1 Macro Std",
            "roc_auc_mean": "ROC-AUC Mean",
            "roc_auc_std": "ROC-AUC Std",
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute real-data TSTR upper bound with the existing evaluation pipeline.")
    parser.add_argument("--config", default="configs/neurocomputing.yaml")
    parser.add_argument("--output-raw", default="results/real_tstr_upper_bound_raw.csv")
    parser.add_argument("--output-summary", default="results/real_tstr_upper_bound_summary.csv")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_config(config_path)
    datasets = cfg.get("datasets", ["breast_cancer", "adult_income", "pima", "heart", "credit"])
    seeds = [int(seed) for seed in cfg.get("seeds", [0, 1, 2, 3, 4])]
    classifiers = cfg.get("tstr", {}).get("classifiers", ["logistic_regression"])
    raw_test_size = float(cfg.get("split", {}).get("test_size", 0.2))
    raw_val_size = float(cfg.get("split", {}).get("val_size", 0.1))

    raw_rows: list[dict[str, object]] = []
    for dataset_entry in datasets:
        if isinstance(dataset_entry, dict):
            dataset = dataset_entry.get("name")
            test_size = float(dataset_entry.get("test_size", raw_test_size))
            val_size = float(dataset_entry.get("val_size", raw_val_size))
            split_random_state = int(dataset_entry.get("random_state", 42))
        else:
            dataset = dataset_entry
            test_size = raw_test_size
            val_size = raw_val_size
            split_random_state = 42
        for seed in seeds:
            split = load_and_split_dataset(
                dataset,
                test_size=test_size,
                val_size=val_size,
                random_state=split_random_state,
            )
            for classifier in classifiers:
                # Reuse the exact TSTR evaluation path: the "synthetic" dataframe is replaced
                # with the real training split, while the evaluation dataframe remains the real test split.
                metrics = tstr_eval(
                    real_df=split.test,
                    synth_df=split.train,
                    label_col=split.label_col,
                    classifier_name=classifier,
                    seed=seed,
                )
                raw_rows.extend(_metric_row(dataset, seed, classifier, metrics))

    raw_df = pd.DataFrame(raw_rows)
    raw_path = Path(args.output_raw).resolve()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_path, index=False)

    summary_df = _build_summary(raw_df)
    summary_path = Path(args.output_summary).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    print(f"Config: {config_path}")
    print("Split protocol source: paper_revC/datasets.py::load_and_split_dataset(test_size=0.2, val_size=0.1)")
    print("Metric protocol source: paper_revC/tstr_eval.py::tstr_eval")
    print(f"Raw rows written to: {raw_path}")
    print(f"Summary written to: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
