from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_revC.tstr_eval import tstr_eval


DEFAULT_DATASETS = ["adult", "breast_cancer", "pima", "heart", "german"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_METHODS = ["tabsyn", "stasy"]
METRICS = ["accuracy", "f1_macro", "roc_auc"]
DATASET_CANONICAL = {
    "adult": "adult_income",
    "german": "credit",
}


def _label_col(train_df: pd.DataFrame) -> str:
    if train_df.empty:
        raise ValueError("real_train.csv is empty")
    return str(train_df.columns[-1])


def _encode_labels_for_tstr(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match label format expected by paper_revC.tstr_eval (integer labels)."""
    train_vals = pd.Series(train_df[label_col]).dropna()
    if train_vals.empty:
        raise ValueError("Training labels are empty.")

    # Numeric labels already compatible.
    if pd.api.types.is_numeric_dtype(train_vals):
        test_out = test_df.copy()
        synth_out = synth_df.copy()
        test_out[label_col] = pd.to_numeric(test_out[label_col], errors="raise").astype(int)
        synth_out[label_col] = pd.to_numeric(synth_out[label_col], errors="raise").astype(int)
        return test_out, synth_out

    # String/categorical labels: deterministic train-based mapping.
    ordered = pd.Series(train_vals.astype(str).unique()).tolist()
    mapping = {val: idx for idx, val in enumerate(ordered)}

    def _map_col(df: pd.DataFrame) -> pd.Series:
        s = df[label_col].astype(str)
        unknown = sorted(set(s.unique()) - set(mapping.keys()))
        if not unknown:
            return s.map(mapping).astype(int)

        # Some generators emit already-encoded class ids (0..K-1).
        numeric = pd.to_numeric(df[label_col], errors="coerce")
        if numeric.notna().all():
            ints = numeric.astype(int)
            if set(ints.unique()).issubset(set(range(len(mapping)))):
                return ints

        raise ValueError(f"Unknown labels in {label_col}: {unknown[:5]}")

    test_out = test_df.copy()
    synth_out = synth_df.copy()
    test_out[label_col] = _map_col(test_out)
    synth_out[label_col] = _map_col(synth_out)
    return test_out, synth_out


def _missing_rows(
    run_tag: str,
    dataset: str,
    source_dataset: str,
    method: str,
    seed: int,
    classifier: str,
    reason: str,
) -> list[dict]:
    rows: list[dict] = []
    for metric in METRICS:
        rows.append(
            {
                "run_tag": run_tag,
                "dataset": dataset,
                "source_dataset": source_dataset,
                "method": method,
                "epsilon": np.nan,
                "seed": seed,
                "classifier": classifier,
                "metric_name": metric,
                "metric": metric,
                "metric_value": np.nan,
                "value": np.nan,
                "converged": False,
                "n_iter": np.nan,
                "status": "FAIL",
                "fallback_used": np.nan,
                "n_synth": 0,
                "notes": reason,
            }
        )
    return rows


def _evaluate_one(
    run_tag: str,
    dataset: str,
    method: str,
    seed: int,
    classifier: str,
    data_root: Path,
    synth_root: Path,
) -> list[dict]:
    split_dir = data_root / dataset / f"seed{seed}"
    real_train = split_dir / "real_train.csv"
    real_test = split_dir / "real_test.csv"
    synthetic = synth_root / dataset / method / str(seed) / "synthetic.csv"

    required = [real_train, real_test, synthetic]
    dataset_out = DATASET_CANONICAL.get(dataset, dataset)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return _missing_rows(
            run_tag=run_tag,
            dataset=dataset_out,
            source_dataset=dataset,
            method=method,
            seed=seed,
            classifier=classifier,
            reason=f"Missing required file(s): {missing}",
        )

    try:
        train_df = pd.read_csv(real_train)
        test_df = pd.read_csv(real_test)
        synth_df = pd.read_csv(synthetic)
        label = _label_col(train_df)
        if label not in test_df.columns or label not in synth_df.columns:
            return _missing_rows(
                run_tag=run_tag,
                dataset=dataset_out,
                source_dataset=dataset,
                method=method,
                seed=seed,
                classifier=classifier,
                reason=f"Label column '{label}' missing in test/synthetic data.",
            )

        test_eval, synth_eval = _encode_labels_for_tstr(
            train_df=train_df,
            test_df=test_df,
            synth_df=synth_df,
            label_col=label,
        )

        out = tstr_eval(
            real_df=test_eval,
            synth_df=synth_eval,
            label_col=label,
            classifier_name=classifier,
            seed=seed,
        )

        rows: list[dict] = []
        for metric in METRICS:
            val = out.get(metric)
            value_percent = float(val) * 100.0 if val is not None and pd.notna(val) else np.nan
            rows.append(
                {
                    "run_tag": run_tag,
                    "dataset": dataset_out,
                    "source_dataset": dataset,
                    "method": method,
                    "epsilon": np.nan,
                    "seed": seed,
                    "classifier": classifier,
                    "metric_name": metric,
                    "metric": metric,
                    "metric_value": val,
                    "value": value_percent,
                    "converged": out.get("converged"),
                    "n_iter": out.get("n_iter"),
                    "status": out.get("status", "FAIL"),
                    "fallback_used": np.nan,
                    "n_synth": out.get("n_synth", len(synth_eval)),
                    "notes": out.get("notes", ""),
                }
            )
        return rows
    except Exception as exc:
        return _missing_rows(
            run_tag=run_tag,
            dataset=dataset_out,
            source_dataset=dataset,
            method=method,
            seed=seed,
            classifier=classifier,
            reason=f"TSTR evaluation error: {exc}",
        )


def run_eval(
    root: Path,
    data_root: Path,
    synth_root: Path,
    datasets: Iterable[str],
    seeds: Iterable[int],
    methods: Iterable[str],
    classifier: str,
    output_csv: Path,
) -> Path:
    run_tag = f"TSTR_TABSYN_STASY_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rows: list[dict] = []

    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                rows.extend(
                    _evaluate_one(
                        run_tag=run_tag,
                        dataset=dataset,
                        method=method,
                        seed=int(seed),
                        classifier=classifier,
                        data_root=data_root,
                        synth_root=synth_root,
                    )
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(
            subset=["dataset", "method", "seed", "classifier", "metric_name"],
            keep="last",
        ).sort_values(["dataset", "method", "seed", "metric_name"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} rows -> {output_csv}")
    print(df.groupby(["method", "status"]).size().to_string())
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TSTR for TabSyn/STaSy using existing pipeline.")
    parser.add_argument("--root", default="D:\\MAS_genAI")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--synthetic-root", default=None)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--methods", default="tabsyn,stasy")
    parser.add_argument("--classifier", default="logistic_regression")
    parser.add_argument("--output", default="results/tstr_results_with_tabsyn_stasy.csv")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else (root / "data_exports")
    synth_root = (
        Path(args.synthetic_root).resolve()
        if args.synthetic_root
        else (root / "outputs" / "synthetic")
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    methods = [x.strip().lower() for x in args.methods.split(",") if x.strip()]

    run_eval(
        root=root,
        data_root=data_root,
        synth_root=synth_root,
        datasets=datasets,
        seeds=seeds,
        methods=methods,
        classifier=args.classifier,
        output_csv=output,
    )


if __name__ == "__main__":
    main()
