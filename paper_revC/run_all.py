from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import json
import sys
import yaml
from importlib import metadata as importlib_metadata

from .baselines_wrappers import BaselineUnavailable, generate_synth
from .datasets import DatasetSplit, load_and_split_dataset
from .export_dataset_summary import export_dataset_summary
from .io_utils import make_run_dir
from .privacy_basic import privacy_audit
from .preprocess_utils import build_feature_preprocessor
from .privacy_strong import privacy_strong_audits
from .tstr_eval import tstr_eval
from src.selection import compute_utility_floor, select_with_privacy_gate


def _resolve_classifiers(requested: Iterable[str]) -> list[str]:
    classifiers = []
    for name in requested:
        if name.lower() == "xgb":
            try:
                import xgboost  # noqa: F401

                classifiers.append("xgb")
            except ImportError:
                logging.info("xgboost not installed; skipping xgb classifier.")
        else:
            classifiers.append(name.lower())
    return classifiers


def _write_madifft_summary(path: Path, records: Iterable[dict]) -> None:
    df = pd.DataFrame(records)
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _extract_epsilon(method: str) -> float | None:
    if method.startswith("privbayes_eps"):
        suffix = method.replace("privbayes_eps", "", 1)
        try:
            return float(suffix)
        except ValueError:
            return None
    return None


def _cache_dir(root: str | None, dataset: str, method: str, seed: int) -> Path | None:
    if not root:
        return None
    return Path(root) / dataset / method / f"seed_{seed}"


def _load_cached_synth(cache_dir: Path) -> pd.DataFrame | None:
    csv_path = cache_dir / "synthetic.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    parquet_path = cache_dir / "synthetic.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return None


def _save_cached_synth(cache_dir: Path, synth_df: pd.DataFrame, fmt: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        synth_df.to_parquet(cache_dir / "synthetic.parquet", index=False)
    else:
        synth_df.to_csv(cache_dir / "synthetic.csv", index=False)


def _load_cached_metrics(cache_dir: Path) -> dict | None:
    metrics_path = cache_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cached_metrics(
    cache_dir: Path,
    tstr_rows: list[dict],
    privacy_row: dict | None,
    strong_rows: list[dict],
) -> None:
    payload = {
        "tstr_rows": tstr_rows,
        "privacy_row": privacy_row,
        "strong_rows": strong_rows,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    if strong_rows:
        pd.DataFrame(strong_rows).to_csv(cache_dir / "audits.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", required=True, help="Path to the revision C config file."
    )
    parser.add_argument(
        "--run-tag",
        "-t",
        default=None,
        help="Optional run tag (defaults to current timestamp).",
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Allow rewriting existing rows for dataset/method/seed combinations.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} does not exist.")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_specs = config.get("datasets", [])
    methods = [m.lower() for m in config.get("methods", [])]
    include_external = bool(config.get("INCLUDE_EXTERNAL_BASELINES", False))
    cache_root = config.get("cache_root") or config.get("CACHE_ROOT")
    cache_mode = str(config.get("cache_mode", config.get("CACHE_MODE", "read_write"))).lower()
    cache_format = str(
        config.get("cache_format", config.get("CACHE_FORMAT", "csv"))
    ).lower()
    skip_generate = bool(config.get("skip_generate") or config.get("SKIP_GENERATE"))
    skip_tstr = bool(config.get("skip_tstr") or config.get("SKIP_TSTR"))
    skip_privacy = bool(config.get("skip_privacy") or config.get("SKIP_PRIVACY"))
    skip_strong = bool(
        config.get("skip_strong_audits") or config.get("SKIP_STRONG_AUDITS")
    )
    if not include_external:
        methods = [m for m in methods if m not in {"tabsyn", "stasy"}]
    privbayes_eps = config.get("privbayes_eps_grid", None)
    if privbayes_eps:
        methods = [m for m in methods if m != "privbayes"]
        for eps in privbayes_eps:
            methods.append(f"privbayes_eps{eps}")
    disabled_methods = {m.lower() for m in config.get("DISABLED_METHODS", [])}
    disable_privbayes = bool(config.get("DISABLE_PRIVBAYES", False))
    if disable_privbayes:
        disabled_methods.add("privbayes")
    seeds = config.get("seeds", [0, 1, 2, 3, 4])
    tstr_requested = config.get("tstr", {}).get("classifiers", ["logistic_regression"])
    classifiers = _resolve_classifiers(tstr_requested)
    privacy_cfg = config.get("privacy", {})
    strong_cfg = config.get("strong_audits", {})
    split_cfg = config.get("split", {})
    raw_test_size = split_cfg.get("test_size", 0.2)
    val_size = split_cfg.get("val_size", 0.1)

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir, tables_dir, figures_dir, logs_dir = make_run_dir(run_tag)
    config["_run_dir"] = str(run_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / "run.log", encoding="utf-8"),
        ],
    )
    try:
        import torch

        if torch.cuda.is_available():
            logging.info("CUDA is available; GPU-enabled baselines will use it.")
        else:
            logging.warning("CUDA is not available; running on CPU.")
    except Exception:
        logging.warning("Unable to detect CUDA availability.")

    dataset_splits: dict[str, DatasetSplit] = {}
    for entry in dataset_specs:
        name = entry if isinstance(entry, str) else entry.get("name")
        logging.info("Preparing dataset %s", name)
        dataset_splits[name] = load_and_split_dataset(
            name=name,
            test_size=entry.get("test_size", raw_test_size)
            if isinstance(entry, dict)
            else raw_test_size,
            val_size=entry.get("val_size", val_size)
            if isinstance(entry, dict)
            else val_size,
            random_state=entry.get("random_state", 42)
            if isinstance(entry, dict)
            else 42,
        )

    export_dataset_summary(dataset_splits.values(), tables_dir)

    feature_schema: dict[str, dict] = {}
    for dataset_name, split in dataset_splits.items():
        preprocessor, feature_names, _, _ = build_feature_preprocessor(
            split.train, split.label_col
        )
        feature_schema[dataset_name] = {
            "label_col": split.label_col,
            "raw_feature_cols": [
                col for col in split.train.columns if col != split.label_col
            ],
            "encoded_feature_names": feature_names,
            "n_features_encoded": len(feature_names),
        }
    (run_dir / "feature_schema.json").write_text(
        json.dumps(feature_schema, indent=2), encoding="utf-8"
    )

    def _safe_version(pkg: str) -> str | None:
        try:
            return importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            return None

    manifest = {
        "run_tag": run_tag,
        "datasets": list(dataset_splits.keys()),
        "methods": methods,
        "seeds": seeds,
        "include_external_baselines": include_external,
        "library_versions": {
            "python": sys.version.split()[0],
            "numpy": _safe_version("numpy"),
            "pandas": _safe_version("pandas"),
            "scikit-learn": _safe_version("scikit-learn"),
            "torch": _safe_version("torch"),
            "sdv": _safe_version("sdv"),
            "domias": _safe_version("domias"),
            "privacy_sdg_toolbox": _safe_version("privacy-sdg-toolbox"),
            "dpart": _safe_version("dpart"),
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    tstr_path = tables_dir / "tstr_results_raw.csv"
    privacy_path = tables_dir / "privacy_audits_raw.csv"
    strong_path = tables_dir / "privacy_strong_audits_raw.csv"
    strong_summary_path = tables_dir / "privacy_strong_audits_summary.csv"
    madiff_path = tables_dir / "madifft_selected_config.csv"

    skipped: list[tuple[str, str, int, str]] = []
    failed: list[tuple[str, str, int, str]] = []

    madifft_meta: dict[tuple[str, int], dict] = {}
    tstr_rows: list[dict] = []
    privacy_rows: list[dict] = []
    strong_rows: list[dict] = []
    tstr_keys: set[tuple] = set()
    privacy_keys: set[tuple] = set()
    strong_keys: set[tuple] = set()

    for dataset_name, split in dataset_splits.items():
        for seed in seeds:
            for method in methods:
                status = "success"
                fallback_used = ""
                error_reason = ""
                metadata: dict = {}
                synth_df = None
                epsilon = _extract_epsilon(method)
                privacy_row: dict | None = None
                cache_dir = _cache_dir(cache_root, dataset_name, method, seed)
                cached_payload = None
                if cache_dir and cache_mode in {"read", "read_write"}:
                    cached_payload = _load_cached_metrics(cache_dir)

                is_privbayes = method.startswith("privbayes")
                if method in disabled_methods or (disable_privbayes and is_privbayes):
                    status = "skipped"
                    error_reason = f"Method disabled via config: {method}"
                    skipped.append((dataset_name, method, seed, error_reason))
                else:
                    loaded_from_cache = False
                    if cache_dir and cache_mode in {"read", "read_write"}:
                        cached_synth = _load_cached_synth(cache_dir)
                        if cached_synth is not None:
                            synth_df = cached_synth
                            loaded_from_cache = True
                            metadata["notes"] = "Loaded synthetic data from cache."
                    if synth_df is None and skip_generate:
                        status = "skipped"
                        error_reason = "Generation skipped and no cached synthetic data."
                        skipped.append((dataset_name, method, seed, error_reason))
                        synth_df = None
                    if synth_df is None and not skip_generate:
                        try:
                            synth_df, metadata = generate_synth(split, method, seed, config)
                        except BaselineUnavailable as exc:
                            failed.append((dataset_name, method, seed, str(exc)))
                            status = "FAILED" if is_privbayes else "failed"
                            error_reason = str(exc)
                            synth_df = None
                    if (
                        synth_df is not None
                        and cache_dir is not None
                        and cache_mode in {"write", "read_write"}
                        and not loaded_from_cache
                    ):
                        _save_cached_synth(cache_dir, synth_df, cache_format)

                n_synth = len(synth_df) if synth_df is not None else 0
                cached_tstr_rows = cached_payload.get("tstr_rows") if cached_payload else None
                if skip_tstr:
                    if cached_tstr_rows:
                        for row in cached_tstr_rows:
                            metric = row.get("metric_name")
                            classifier = row.get("classifier")
                            key = (dataset_name, method, seed, classifier, metric)
                            if key in tstr_keys:
                                continue
                            tstr_keys.add(key)
                            row.update(
                                {
                                    "run_tag": run_tag,
                                    "dataset": dataset_name,
                                    "method": method,
                                    "epsilon": epsilon,
                                    "seed": seed,
                                }
                            )
                            tstr_rows.append(row)
                    else:
                        for classifier in classifiers:
                            for metric in ["accuracy", "f1_macro", "roc_auc"]:
                                key = (dataset_name, method, seed, classifier, metric)
                                if key in tstr_keys:
                                    continue
                                tstr_keys.add(key)
                                tstr_rows.append(
                                    {
                                        "run_tag": run_tag,
                                        "dataset": dataset_name,
                                        "method": method,
                                        "epsilon": epsilon,
                                        "seed": seed,
                                        "classifier": classifier,
                                        "metric_name": metric,
                                        "metric_value": float("nan"),
                                        "converged": False,
                                        "n_iter": None,
                                        "status": "skipped",
                                        "fallback_used": fallback_used,
                                        "n_synth": n_synth,
                                        "notes": "TSTR skipped; no cached metrics.",
                                    }
                                )
                else:
                    for classifier in classifiers:
                        if synth_df is None:
                            tstr_metrics = {
                                "accuracy": None,
                                "f1_macro": None,
                                "roc_auc": None,
                                "notes": f"No synthetic data available. {error_reason}",
                                "converged": False,
                                "n_iter": None,
                                "status": status,
                            }
                        else:
                            tstr_metrics = tstr_eval(
                                split.test, synth_df, split.label_col, classifier, seed
                            )
                        for metric in ["accuracy", "f1_macro", "roc_auc"]:
                            metric_value = tstr_metrics.get(metric)
                            if metric_value is None or (
                                isinstance(metric_value, float) and pd.isna(metric_value)
                            ):
                                if is_privbayes:
                                    metric_value = float("nan")
                                    tstr_metrics["status"] = "FAILED"
                                else:
                                    metric_value = 0.0
                                    if tstr_metrics.get("status") != "skipped":
                                        tstr_metrics["status"] = "failed"
                            key = (dataset_name, method, seed, classifier, metric)
                            if key in tstr_keys:
                                raise ValueError(f"Duplicate TSTR key detected: {key}")
                            tstr_keys.add(key)
                            tstr_rows.append(
                                {
                                    "run_tag": run_tag,
                                    "dataset": dataset_name,
                                    "method": method,
                                    "epsilon": epsilon,
                                    "seed": seed,
                                    "classifier": classifier,
                                    "metric_name": metric,
                                    "metric_value": metric_value,
                                    "converged": tstr_metrics.get("converged"),
                                    "n_iter": tstr_metrics.get("n_iter"),
                                    "status": tstr_metrics.get("status", status),
                                    "fallback_used": fallback_used,
                                    "n_synth": n_synth,
                                    "notes": tstr_metrics.get("notes", ""),
                                }
                            )

                cached_privacy_row = cached_payload.get("privacy_row") if cached_payload else None
                if skip_privacy:
                    p_key = (dataset_name, method, seed)
                    if cached_privacy_row:
                        if p_key not in privacy_keys:
                            cached_privacy_row.update(
                                {
                                    "run_tag": run_tag,
                                    "dataset": dataset_name,
                                    "method": method,
                                    "epsilon": epsilon,
                                    "seed": seed,
                                }
                            )
                            privacy_keys.add(p_key)
                            privacy_rows.append(cached_privacy_row)
                            privacy_row = cached_privacy_row
                    else:
                        if p_key not in privacy_keys:
                            privacy_keys.add(p_key)
                            privacy_row = {
                                "run_tag": run_tag,
                                "dataset": dataset_name,
                                "method": method,
                                "epsilon": epsilon,
                                "seed": seed,
                                "dcr_q": float("nan"),
                                "norm_dcr": float("nan"),
                                "mia_auc": float("nan"),
                                "alpha": privacy_cfg.get("alpha_mixdist", 0.5),
                                "q": privacy_cfg.get("dcr_q", 0.01),
                                "status": "skipped",
                                "fallback_used": fallback_used,
                                "notes": "Privacy audit skipped; no cached metrics.",
                            }
                            privacy_rows.append(
                                {
                                    **privacy_row,
                                }
                            )
                else:
                    if synth_df is None:
                        privacy_metrics = {
                            "dcr_q": None,
                            "norm_dcr": None,
                            "mia_auc": None,
                            "alpha": privacy_cfg.get("alpha_mixdist", 0.5),
                            "q": privacy_cfg.get("dcr_q", 0.01),
                            "notes": f"No synthetic data available. {error_reason}",
                            "status": status,
                        }
                    else:
                        privacy_metrics = privacy_audit(
                            split.train,
                            split.val,
                            synth_df,
                            split.label_col,
                            q=privacy_cfg.get("dcr_q", 0.01),
                            alpha=privacy_cfg.get("alpha_mixdist", 0.5),
                            seed=seed,
                        )
                        privacy_metrics["status"] = "success"
                    for key in ["dcr_q", "norm_dcr", "mia_auc"]:
                        value = privacy_metrics.get(key)
                        if value is None or (isinstance(value, float) and pd.isna(value)):
                            if is_privbayes:
                                privacy_metrics[key] = float("nan")
                                privacy_metrics["status"] = "FAILED"
                            else:
                                privacy_metrics[key] = 0.0
                                if privacy_metrics.get("status") != "skipped":
                                    privacy_metrics["status"] = "failed"
                    p_key = (dataset_name, method, seed)
                    if p_key in privacy_keys:
                        raise ValueError(f"Duplicate privacy key detected: {p_key}")
                    privacy_keys.add(p_key)
                    privacy_row = {
                        "run_tag": run_tag,
                        "dataset": dataset_name,
                        "method": method,
                        "epsilon": epsilon,
                        "seed": seed,
                        "dcr_q": privacy_metrics["dcr_q"],
                        "norm_dcr": privacy_metrics["norm_dcr"],
                        "mia_auc": privacy_metrics["mia_auc"],
                        "alpha": privacy_metrics["alpha"],
                        "q": privacy_metrics["q"],
                        "status": privacy_metrics.get("status", status),
                        "fallback_used": fallback_used,
                        "notes": privacy_metrics["notes"],
                    }
                    privacy_rows.append(privacy_row)

                strong_methods = strong_cfg.get("methods")
                if strong_methods is not None:
                    strong_methods = [m.lower() for m in strong_methods]
                if (
                    (strong_cfg.get("enable_domias", False) or strong_cfg.get("enable_tapas", False))
                    and (strong_methods is None or method in strong_methods)
                ):
                    cached_strong_rows = cached_payload.get("strong_rows") if cached_payload else None
                    s_key = (dataset_name, method, seed)
                    if skip_strong and cached_strong_rows:
                        for row in cached_strong_rows:
                            if s_key in strong_keys:
                                continue
                            strong_keys.add(s_key)
                            row.update(
                                {
                                    "run_tag": run_tag,
                                    "dataset": dataset_name,
                                    "method": method,
                                    "epsilon": epsilon,
                                    "seed": seed,
                                }
                            )
                            strong_rows.append(row)
                    else:
                        domias_auc = float("nan")
                        error_message = ""
                        status_value = "skipped" if skip_strong else "failed"
                        if not skip_strong and synth_df is not None:
                            audit_rows = privacy_strong_audits(
                                split, synth_df, split.label_col, seed, strong_cfg
                            )
                            domias_scores = []
                            for row in audit_rows:
                                if str(row.get("audit_name", "")).lower() != "domias":
                                    continue
                                score_value = row.get("score_value")
                                if (
                                    score_value is not None
                                    and not (isinstance(score_value, float) and pd.isna(score_value))
                                    and "auc" in str(row.get("score_name", ""))
                                    and row.get("status") == "success"
                                ):
                                    domias_scores.append(float(score_value))
                                elif row.get("status") in {"failed", "skipped"}:
                                    error_message = row.get("notes", "")
                            if domias_scores:
                                domias_auc = float(max(domias_scores))
                                status_value = "success"
                            else:
                                status_value = "failed"
                                if not error_message:
                                    error_message = "DOMIAS produced no usable AUC."
                        elif synth_df is None:
                            error_message = f"No synthetic data available. {error_reason}"
                            status_value = "failed"
                        elif skip_strong:
                            error_message = "Strong audits skipped."
                            status_value = "skipped"

                        if s_key not in strong_keys:
                            strong_keys.add(s_key)
                            strong_rows.append(
                                {
                                    "run_tag": run_tag,
                                    "dataset": dataset_name,
                                    "method": method,
                                    "epsilon": epsilon,
                                    "seed": seed,
                                    "domias_auc": domias_auc,
                                    "dist_mia_auc": (privacy_row or {}).get("mia_auc"),
                                    "dcr_q": (privacy_row or {}).get("dcr_q"),
                                    "normdcr": (privacy_row or {}).get("norm_dcr"),
                                    "status": status_value,
                                    "error": error_message,
                                }
                            )

                if method == "madifft":
                    madifft_meta[(dataset_name, seed)] = metadata.copy()

                if cache_dir is not None and cache_mode in {"write", "read_write"}:
                    tstr_cache_rows = [
                        row
                        for row in tstr_rows
                        if row["dataset"] == dataset_name
                        and row["method"] == method
                        and row["seed"] == seed
                    ]
                    privacy_cache_row = next(
                        (
                            row
                            for row in privacy_rows
                            if row["dataset"] == dataset_name
                            and row["method"] == method
                            and row["seed"] == seed
                        ),
                        None,
                    )
                    strong_cache_rows = [
                        row
                        for row in strong_rows
                        if row["dataset"] == dataset_name
                        and row["method"] == method
                        and row["seed"] == seed
                    ]
                    _save_cached_metrics(cache_dir, tstr_cache_rows, privacy_cache_row, strong_cache_rows)

    tstr_df = pd.DataFrame(tstr_rows)
    privacy_df = pd.DataFrame(privacy_rows)
    strong_df = pd.DataFrame(strong_rows)

    if not tstr_df.empty:
        dupes = tstr_df.duplicated(
            subset=["dataset", "method", "seed", "classifier", "metric_name"], keep=False
        )
        if dupes.any():
            raise ValueError("Duplicate TSTR rows detected.")
    else:
        tstr_df = pd.DataFrame(
            columns=[
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
        )
    tstr_df.to_csv(tstr_path, index=False)

    if not privacy_df.empty:
        dupes = privacy_df.duplicated(subset=["dataset", "method", "seed"], keep=False)
        if dupes.any():
            raise ValueError("Duplicate privacy rows detected.")
    else:
        privacy_df = pd.DataFrame(
            columns=[
                "run_tag",
                "dataset",
                "method",
                "epsilon",
                "seed",
                "dcr_q",
                "norm_dcr",
                "mia_auc",
                "alpha",
                "q",
                "status",
                "fallback_used",
                "notes",
            ]
        )
    privacy_df.to_csv(privacy_path, index=False)

    if not strong_df.empty:
        dupes = strong_df.duplicated(
            subset=["dataset", "method", "seed"], keep=False
        )
        if dupes.any():
            raise ValueError("Duplicate strong audit rows detected.")
    else:
        strong_df = pd.DataFrame(
            columns=[
                "run_tag",
                "dataset",
                "method",
                "epsilon",
                "seed",
                "domias_auc",
                "dist_mia_auc",
                "dcr_q",
                "normdcr",
                "status",
                "error",
            ]
        )
    strong_df.to_csv(strong_path, index=False)

    madifft_rows: list[dict] = []
    for dataset_name in dataset_splits.keys():
        candidate_rows = []
        for seed in seeds:
            meta = madifft_meta.get((dataset_name, seed))
            if meta:
                candidate_rows.append(
                    {
                        "seed": seed,
                        "agent_id": meta.get("best_agent_id", ""),
                        "generation": meta.get("generation", ""),
                        "utility": meta.get("utility", float("nan")),
                        "fidelity": meta.get("fidelity", float("nan")),
                        "normdcr": meta.get("normdcr", float("nan")),
                        "mia_auc": meta.get("mia_auc", float("nan")),
                        "mia_adv": meta.get("mia_adv", float("nan")),
                        "fitness": meta.get("fitness", meta.get("fitness_total", float("-inf"))),
                        "best_params": meta.get("best_params", {}),
                        "selected_by": meta.get("selected_by", ""),
                        "notes": meta.get("notes", ""),
                    }
                )
        if not candidate_rows:
            madifft_rows.append(
                {
                    "dataset": dataset_name,
                    "generation": "",
                    "agent_id": "",
                    "hyperparams": "",
                    "utility": float("nan"),
                    "fidelity": float("nan"),
                    "normdcr": float("nan"),
                    "mia_auc": float("nan"),
                    "mia_adv": float("nan"),
                    "fitness": float("nan"),
                    "selected_by": "",
                    "notes": "No MADiff-T metadata available.",
                }
            )
            continue
        candidates_df = pd.DataFrame(candidate_rows)
        train_labels = dataset_splits[dataset_name].train[
            dataset_splits[dataset_name].label_col
        ].to_numpy()
        utility_floor = compute_utility_floor(
            y_train=train_labels,
            base_floor=0.60,
            margin=0.04,
            max_floor=0.68,
        )
        candidates_df, selection = select_with_privacy_gate(
            candidates_df,
            utility_floor=utility_floor,
        )
        best_row = selection.selected_row
        best_seed = int(best_row["seed"])
        madifft_rows.append(
                {
                    "dataset": dataset_name,
                    "generation": best_row.get("generation", ""),
                    "agent_id": best_row.get("agent_id", f"madifft_seed_{dataset_name}_{best_seed}"),
                    "hyperparams": json.dumps(
                        best_row.get("best_params", {}),
                        default=lambda obj: obj.tolist() if hasattr(obj, "tolist") else str(obj),
                    ),
                    "utility": best_row.get("utility", float("nan")),
                    "fidelity": best_row.get("fidelity", float("nan")),
                    "normdcr": best_row.get("normdcr", float("nan")),
                    "mia_auc": best_row.get("mia_auc", float("nan")),
                "mia_adv": best_row.get("mia_adv", float("nan")),
                "fitness": best_row.get("fitness", float("nan")),
                "selected_by": selection.selected_by,
                "notes": f"selected_seed={best_seed}; gate={selection.selected_by}",
            }
        )

    _write_madifft_summary(madiff_path, madifft_rows)

    if not strong_df.empty and "status" in strong_df.columns:
        numeric_cols = ["domias_auc", "dist_mia_auc", "dcr_q", "normdcr"]
        for col in numeric_cols:
            if col in strong_df.columns:
                strong_df[col] = pd.to_numeric(strong_df[col], errors="coerce")
        summary_df = (
            strong_df[strong_df["status"] == "success"]
            .groupby(["dataset", "method", "epsilon"], dropna=False)[numeric_cols]
            .agg(["mean", "std", "count"])
        )
        if not summary_df.empty:
            summary_df.columns = [
                f"{col}_{stat}" for col, stat in summary_df.columns.to_flat_index()
            ]
            summary_df = summary_df.reset_index()
            summary_df.to_csv(strong_summary_path, index=False)

    expected = {(d, m, s) for d in dataset_splits.keys() for m in methods for s in seeds}
    actual = {(r["dataset"], r["method"], r["seed"]) for r in privacy_rows}
    missing = expected - actual

    print(f"Run directory: {run_dir}")
    print(
        f"Rows: dataset_summary={len(dataset_splits)}, "
        f"tstr={len(tstr_rows)}, privacy={len(privacy_rows)}, strong={len(strong_rows)}"
    )
    print("Skipped baselines:", len(skipped))
    if skipped:
        for dataset, method, seed, reason in skipped:
            print(f"  {dataset}/{method} seed={seed}: {reason}")
    if failed:
        print("Failed baselines:", len(failed))
        for dataset, method, seed, reason in failed:
            print(f"  {dataset}/{method} seed={seed}: {reason}")
    print(f"Missing dataset-method-seed combinations: {len(missing)}")
    if missing:
        for dataset, method, seed in sorted(missing):
            print(f"  {dataset}/{method} seed={seed}")


if __name__ == "__main__":
    main()
