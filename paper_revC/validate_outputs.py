from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from src.selection import select_with_privacy_gate


def _red_flag(message: str) -> str:
    return f"\x1b[31mRED-FLAG\x1b[0m {message}"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def validate_outputs(run_dir: Path) -> int:
    tables_dir = run_dir / "tables"
    dataset_summary = _load_csv(tables_dir / "dataset_summary.csv")
    tstr = _load_csv(tables_dir / "tstr_results_raw.csv")
    privacy = _load_csv(tables_dir / "privacy_audits_raw.csv")
    strong_path = tables_dir / "privacy_strong_audits_raw.csv"
    if not strong_path.exists():
        strong_path = tables_dir / "privacy_strong_audits.csv"
    strong = _load_csv(strong_path)
    madifft = _load_csv(tables_dir / "madifft_selected_config.csv")
    madifft_metrics_path = tables_dir / "madifft_generation_metrics.csv"
    madifft_metrics = _load_csv(madifft_metrics_path) if madifft_metrics_path.exists() else pd.DataFrame()

    errors: list[str] = []

    # dataset_summary: exactly 1 row per dataset
    ds_counts = dataset_summary["dataset"].value_counts()
    if (ds_counts != 1).any():
        bad = ds_counts[ds_counts != 1]
        errors.append(f"dataset_summary rows not 1 per dataset: {bad.to_dict()}")

    # tstr_results_raw checks
    if tstr["metric_value"].isna().any():
        errors.append("tstr_results_raw has NaNs in metric_value.")
    expected_seeds = {0, 1, 2, 3, 4}
    for (dataset, method), group in tstr.groupby(["dataset", "method"]):
        seeds = set(group["seed"].unique())
        if seeds != expected_seeds:
            errors.append(
                f"tstr_results_raw missing seeds for {dataset}/{method}: {sorted(expected_seeds - seeds)}"
            )
    dup_cols = ["dataset", "method", "seed", "classifier", "metric_name"]
    if tstr.duplicated(dup_cols).any():
        errors.append("tstr_results_raw has duplicate rows by dataset/method/seed/classifier/metric_name.")

    # privacy_audits_raw checks
    privacy_dedup = privacy.drop_duplicates(subset=["dataset", "method", "seed", "alpha", "q"])
    failed_privacy = privacy_dedup[privacy_dedup.get("status", "") != "success"]
    if not failed_privacy.empty:
        errors.append(
            f"privacy_audits_raw has failed rows: {failed_privacy[['dataset','method','seed']].to_dict(orient='records')}"
        )

    # privacy_strong_audits checks
    if strong.empty:
        errors.append("privacy_strong_audits.csv is empty.")
    else:
        expected = privacy[["dataset", "method", "seed"]].drop_duplicates()
        strong_keys = strong[["dataset", "method", "seed"]].drop_duplicates()
        merged = expected.merge(
            strong_keys, on=["dataset", "method", "seed"], how="left", indicator=True
        )
        missing_strong = merged[merged["_merge"] == "left_only"]
        if not missing_strong.empty:
            errors.append(
                f"privacy_strong_audits missing rows for: "
                f"{missing_strong[['dataset','method','seed']].to_dict(orient='records')}"
            )
    failures = strong[strong.get("status", "") != "success"]
    if not failures.empty:
        errors.append(
            f"privacy_strong_audits has failures: {failures[['dataset','method','seed','status']].to_dict(orient='records')}"
        )

    # madifft_selected_config should not be empty
    if madifft.empty:
        errors.append("madifft_selected_config.csv is empty.")
    elif madifft_metrics.empty:
        errors.append("madifft_generation_metrics.csv is missing or empty.")
    else:
        for dataset in madifft["dataset"].unique():
            selected = madifft[madifft["dataset"] == dataset].iloc[0]
            gen = selected.get("generation")
            agent_id = selected.get("agent_id")
            if agent_id is None or agent_id == "":
                errors.append(f"MADiff-T selected agent missing for dataset {dataset}.")
                continue
            metrics_subset = madifft_metrics[madifft_metrics["dataset"] == dataset]
            if gen != "" and not pd.isna(gen):
                metrics_subset = metrics_subset[metrics_subset["generation"] == int(gen)]
            if metrics_subset.empty:
                errors.append(f"MADiff-T metrics missing for dataset {dataset}.")
                continue
            gated_df, selection = select_with_privacy_gate(metrics_subset)
            best_row = selection.selected_row
            if str(best_row["agent_id"]) != str(agent_id):
                errors.append(
                    f"MADiff-T selection mismatch for {dataset}: "
                    f"selected={agent_id}, gate_best={best_row['agent_id']}"
                )

    if errors:
        print(_red_flag("Validation failed with issues:"))
        for err in errors:
            print(_red_flag(f"- {err}"))
        return 1

    print("Validation passed: all outputs are paper-ready.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Run directory to validate.")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    return validate_outputs(run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
