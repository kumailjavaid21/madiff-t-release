from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _format_value(mean: float, std: float) -> str:
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def _compute_summary(
    df: pd.DataFrame,
    metric_name: str,
    classifier: str | None,
    required_seeds: list[int],
) -> dict[tuple[str, str], str]:
    results: dict[tuple[str, str], str] = {}
    if classifier is not None:
        df = df[df["classifier"] == classifier]
    df = df[df["metric_name"] == metric_name]

    for (dataset, method), group in df.groupby(["dataset", "method"], dropna=False):
        seeds = set(group["seed"].tolist())
        has_all_seeds = all(seed in seeds for seed in required_seeds)
        status_ok = group["status"].astype(str).str.lower().eq("success").all()
        values = group["metric_value"]
        if not has_all_seeds or not status_ok or values.isna().any():
            results[(dataset, method)] = r"\na"
            continue
        mean = float(values.mean())
        std = float(values.std())
        results[(dataset, method)] = _format_value(mean, std)
    return results


def _write_table(
    path: Path,
    datasets: list[str],
    methods: list[str],
    values: dict[tuple[str, str], str],
    caption: str,
    label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    col_spec = "l" + "c" * len(methods)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    header = "Dataset & " + " & ".join(methods) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    for dataset in datasets:
        row_vals = [values.get((dataset, method), r"\na") for method in methods]
        lines.append(dataset + " & " + " & ".join(row_vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tables-dir",
        type=str,
        default="paper_revC/results_revC/REVISIONC_FINAL/tables",
        help="Directory containing raw CSV tables.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="generated_tables",
        help="Output directory for LaTeX tables.",
    )
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    out_dir = Path(args.out_dir)

    dataset_summary = pd.read_csv(tables_dir / "dataset_summary.csv")
    tstr = pd.read_csv(tables_dir / "tstr_results_raw.csv")
    privacy = pd.read_csv(tables_dir / "privacy_audits_raw.csv")

    datasets = dataset_summary["dataset"].tolist()
    methods = sorted(tstr["method"].unique().tolist())
    required_seeds = sorted(tstr["seed"].unique().tolist())

    utility_vals = _compute_summary(
        tstr,
        metric_name="roc_auc",
        classifier="logistic_regression",
        required_seeds=required_seeds,
    )
    privacy_long = privacy.copy()
    privacy_long["metric_name"] = "mia_auc"
    privacy_long = privacy_long.rename(columns={"mia_auc": "metric_value"})
    privacy_vals = _compute_summary(
        privacy_long,
        metric_name="mia_auc",
        classifier=None,
        required_seeds=required_seeds,
    )

    _write_table(
        out_dir / "utility_table.tex",
        datasets,
        methods,
        utility_vals,
        caption="TSTR ROC-AUC (mean $\\pm$ std) over 5 seeds.",
        label="tab:utility",
    )

    _write_table(
        out_dir / "privacy_table.tex",
        datasets,
        methods,
        privacy_vals,
        caption="MIA AUC (mean $\\pm$ std) over 5 seeds.",
        label="tab:privacy",
    )


if __name__ == "__main__":
    main()
