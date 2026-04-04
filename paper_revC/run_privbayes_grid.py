from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .baselines_wrappers import BaselineUnavailable, generate_synth
from .datasets import load_and_split_dataset
from .privacy_basic import privacy_audit
from .tstr_eval import tstr_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir)
    tables_dir = run_dir / "tables"
    tstr_path = tables_dir / "tstr_results_raw.csv"
    privacy_path = tables_dir / "privacy_audits_raw.csv"

    datasets = config.get("datasets", [])
    seeds = config.get("seeds", [0, 1, 2, 3, 4])
    eps_grid = config.get("privbayes_eps_grid", [0.1, 0.5, 1, 2, 5, 10])
    tstr_requested = config.get("tstr", {}).get("classifiers", ["logistic_regression"])
    classifiers = [c.lower() for c in tstr_requested]

    run_tag = run_dir.name

    dataset_splits = {}
    for entry in datasets:
        name = entry if isinstance(entry, str) else entry.get("name")
        dataset_splits[name] = load_and_split_dataset(name=name)

    # Load existing CSVs and drop privbayes rows
    tstr_df = pd.read_csv(tstr_path) if tstr_path.exists() else pd.DataFrame()
    privacy_df = pd.read_csv(privacy_path) if privacy_path.exists() else pd.DataFrame()

    if not tstr_df.empty:
        tstr_df = tstr_df[~tstr_df["method"].str.startswith("privbayes")]
    if not privacy_df.empty:
        privacy_df = privacy_df[~privacy_df["method"].str.startswith("privbayes")]

    new_tstr_rows = []
    new_priv_rows = []

    for dataset_name, split in dataset_splits.items():
        for seed in seeds:
            for eps in eps_grid:
                method = f"privbayes_eps{eps}"
                status = "success"
                error_reason = ""
                try:
                    synth_df, _ = generate_synth(split, method, seed, config)
                except BaselineUnavailable as exc:
                    synth_df = None
                    status = "FAILED"
                    error_reason = str(exc)

                for classifier in classifiers:
                    if synth_df is None:
                        metrics = {"accuracy": None, "f1_macro": None, "roc_auc": None, "status": status}
                    else:
                        metrics = tstr_eval(split.test, synth_df, split.label_col, classifier, seed)
                    for metric in ["accuracy", "f1_macro", "roc_auc"]:
                        value = metrics.get(metric)
                        if value is None or (isinstance(value, float) and pd.isna(value)):
                            value = float("nan")
                            status_value = "FAILED"
                        else:
                            status_value = metrics.get("status", status)
                        new_tstr_rows.append(
                            {
                                "run_tag": run_tag,
                                "dataset": dataset_name,
                                "method": method,
                                "seed": seed,
                                "classifier": classifier,
                                "metric_name": metric,
                                "metric_value": value,
                                "converged": metrics.get("converged"),
                                "n_iter": metrics.get("n_iter"),
                                "status": status_value,
                                "fallback_used": "",
                                "n_synth": len(synth_df) if synth_df is not None else 0,
                                "notes": metrics.get("notes", error_reason),
                            }
                        )

                if synth_df is None:
                    priv = {
                        "dcr_q": None,
                        "norm_dcr": None,
                        "mia_auc": None,
                        "alpha": config.get("privacy", {}).get("alpha_mixdist", 0.5),
                        "q": config.get("privacy", {}).get("dcr_q", 0.01),
                        "notes": error_reason,
                        "status": status,
                    }
                else:
                    priv = privacy_audit(
                        split.train,
                        split.val,
                        synth_df,
                        split.label_col,
                        q=config.get("privacy", {}).get("dcr_q", 0.01),
                        alpha=config.get("privacy", {}).get("alpha_mixdist", 0.5),
                        seed=seed,
                    )
                    priv["status"] = "success"
                for key in ["dcr_q", "norm_dcr", "mia_auc"]:
                    value = priv.get(key)
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        priv[key] = float("nan")
                        priv["status"] = "FAILED"

                new_priv_rows.append(
                    {
                        "run_tag": run_tag,
                        "dataset": dataset_name,
                        "method": method,
                        "seed": seed,
                        "dcr_q": priv["dcr_q"],
                        "norm_dcr": priv["norm_dcr"],
                        "mia_auc": priv["mia_auc"],
                        "alpha": priv["alpha"],
                        "q": priv["q"],
                        "status": priv.get("status", status),
                        "fallback_used": "",
                        "notes": priv["notes"],
                    }
                )

    tstr_df = pd.concat([tstr_df, pd.DataFrame(new_tstr_rows)], ignore_index=True)
    privacy_df = pd.concat([privacy_df, pd.DataFrame(new_priv_rows)], ignore_index=True)

    tstr_df.to_csv(tstr_path, index=False)
    privacy_df.to_csv(privacy_path, index=False)


if __name__ == "__main__":
    main()
