from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ALIAS = {
    "gaussian_copula": "GC",
    "tvae": "TVAE",
    "ctgan": "CTGAN",
    "tabddpm_single": "TabDDPM",
    "tabsyn": "TabSyn",
    "stasy": "STaSy",
    "madifft": "MADiffT",
}

DATASET_ORDER = ["breast_cancer", "adult_income", "adult", "pima", "heart", "credit", "german"]


def find_existing_tstr_csv(root: Path) -> Path:
    candidates = []
    for p in root.rglob("tstr_results_raw.csv"):
        lower = str(p).lower()
        if "venv" in lower or "site-packages" in lower:
            continue
        score = 0
        if "\\outputs\\" in lower or "/outputs/" in lower:
            score += 10
        if "release_bundle" in lower:
            score -= 20
        try:
            size = p.stat().st_size
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if size < 1024:
            score -= 5
        candidates.append((score, mtime, p))

    if not candidates:
        raise FileNotFoundError("No existing tstr_results_raw.csv found.")

    # Highest score, then newest mtime.
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Common expected columns across this repo.
    required = [
        "run_tag",
        "dataset",
        "method",
        "epsilon",
        "seed",
        "classifier",
        "metric_name",
        "metric_value",
        "converged",
        "n_iter",
        "status",
        "fallback_used",
        "n_synth",
        "notes",
    ]
    for col in required:
        if col not in out.columns:
            out[col] = np.nan
    return out


def build_summary_for_paper(full_df: pd.DataFrame, out_path: Path) -> Path:
    df = full_df.copy()
    if "metric_name" not in df.columns or "metric_value" not in df.columns:
        raise ValueError("Full TSTR dataframe missing metric_name/metric_value columns.")

    df = df[df["metric_name"].astype(str).str.lower() == "accuracy"].copy()
    df["status"] = df["status"].astype(str).str.lower()
    df = df[df["status"].isin(["success", "ok"])]
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
    df = df[df["metric_value"].notna()]

    if "classifier" in df.columns:
        df = df[df["classifier"].astype(str).str.lower() == "logistic_regression"]

    df["method_alias"] = df["method"].map(METHOD_ALIAS)
    df = df[df["method_alias"].notna()]

    grouped = (
        df.groupby(["dataset", "method_alias"])["metric_value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)

    datasets = sorted(grouped["dataset"].unique(), key=lambda x: (DATASET_ORDER.index(x) if x in DATASET_ORDER else 999, x))
    method_order = ["GC", "TVAE", "CTGAN", "TabDDPM", "TabSyn", "STaSy", "MADiffT"]

    rows = []
    for ds in datasets:
        row = {"dataset": ds}
        sub = grouped[grouped["dataset"] == ds].set_index("method_alias")
        for m in method_order:
            if m in sub.index:
                row[f"{m}_mean"] = float(sub.loc[m, "mean"])
                row[f"{m}_std"] = float(sub.loc[m, "std"])
            else:
                row[f"{m}_mean"] = np.nan
                row[f"{m}_std"] = np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge existing TSTR CSV with TabSyn/STaSy TSTR rows.")
    parser.add_argument("--root", default="D:\\MAS_genAI")
    parser.add_argument("--existing", default=None, help="Optional explicit existing tstr_results_raw.csv path.")
    parser.add_argument("--new", default="results/tstr_results_with_tabsyn_stasy.csv")
    parser.add_argument("--out-full", default="results/tstr_results_full_with_tabsyn_stasy.csv")
    parser.add_argument("--out-summary", default="results/tstr_summary_for_paper.csv")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    existing_path = Path(args.existing).resolve() if args.existing else find_existing_tstr_csv(root)
    new_path = Path(args.new)
    if not new_path.is_absolute():
        new_path = root / new_path
    out_full = Path(args.out_full)
    if not out_full.is_absolute():
        out_full = root / out_full
    out_summary = Path(args.out_summary)
    if not out_summary.is_absolute():
        out_summary = root / out_summary

    if not existing_path.exists():
        raise FileNotFoundError(f"Existing TSTR CSV not found: {existing_path}")
    if not new_path.exists():
        raise FileNotFoundError(f"New TabSyn/STaSy TSTR CSV not found: {new_path}")

    existing = _standardize(pd.read_csv(existing_path))
    new = _standardize(pd.read_csv(new_path))

    full = pd.concat([existing, new], ignore_index=True)
    full = full.drop_duplicates(
        subset=["dataset", "method", "seed", "classifier", "metric_name"],
        keep="last",
    ).sort_values(["dataset", "method", "seed", "classifier", "metric_name"])

    out_full.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_full, index=False)
    summary_path = build_summary_for_paper(full, out_summary)

    print(f"Existing TSTR CSV used: {existing_path}")
    print(f"New TabSyn/STaSy CSV:   {new_path}")
    print(f"Wrote merged CSV:       {out_full} ({len(full)} rows)")
    print(f"Wrote paper summary:    {summary_path}")


if __name__ == "__main__":
    main()
