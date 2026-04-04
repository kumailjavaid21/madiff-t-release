from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_COLS = ["GC", "TVAE", "CTGAN", "TabDDPM", "TabSyn", "STaSy", "MADiffT"]
DATASET_DISPLAY = {
    "adult": "Adult Income",
    "adult_income": "Adult Income",
    "breast_cancer": "Breast Cancer",
    "pima": "PIMA",
    "heart": "Heart",
    "credit": "Credit",
    "german": "German",
}
DATASET_ORDER = ["breast_cancer", "adult_income", "adult", "pima", "heart", "credit", "german"]


def fmt_cell(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        std = 0.0
    return f"{mean * 100:.2f} $\\pm$ {std * 100:.2f}"


def build_table(summary: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{TSTR Accuracy (\\%) over 5 seeds (mean $\\pm$ std).}")
    lines.append("\\label{tab:tstr_with_tabsyn_stasy}")
    col_spec = "l" + "c" * len(METHOD_COLS)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")
    header = "Dataset & " + " & ".join(["GC", "TVAE", "CTGAN", "TabDDPM", "TabSyn", "STaSy", "MADiff-T"]) + " \\\\"
    lines.append(header)
    lines.append("\\hline")

    def ds_sort_key(ds: str):
        return (DATASET_ORDER.index(ds) if ds in DATASET_ORDER else 999, ds)

    for _, row in summary.sort_values("dataset", key=lambda s: s.map(ds_sort_key)).iterrows():
        ds = str(row["dataset"])
        ds_name = DATASET_DISPLAY.get(ds, ds)
        vals = []
        for m in METHOD_COLS:
            vals.append(fmt_cell(row.get(f"{m}_mean", np.nan), row.get(f"{m}_std", np.nan)))
        lines.append(f"{ds_name} & " + " & ".join(vals) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print LaTeX TSTR table from summary CSV.")
    parser.add_argument("--root", default="D:\\MAS_genAI")
    parser.add_argument("--summary", default="results/tstr_summary_for_paper.csv")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = root / summary_path

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")

    summary = pd.read_csv(summary_path)
    latex = build_table(summary)
    print(latex)


if __name__ == "__main__":
    main()
