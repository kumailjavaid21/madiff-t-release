from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common import (  # noqa: E402
    GenerationResult,
    bootstrap_generate,
    cast_like_train,
    ensure_schema,
    sdv_ctgan_generate,
    write_status,
)


def _resolve_repo_dir(explicit_repo_dir: str | None) -> Path:
    candidates: List[Path] = []
    if explicit_repo_dir:
        candidates.append(Path(explicit_repo_dir))

    import os

    env_repo = os.environ.get("TABSYN_REPO")
    if env_repo:
        candidates.append(Path(env_repo))

    candidates.append(PROJECT_ROOT / "baselines" / "tabsyn" / "repo")
    candidates.append(PROJECT_ROOT / "paper_revC" / "external" / "tabsyn")

    for candidate in candidates:
        if (candidate / "main.py").exists():
            return candidate
    raise FileNotFoundError("TabSyn repository not found. Set --repo_dir or TABSYN_REPO.")


def _check_tabsyn_dependencies(repo_dir: Path) -> None:
    required_modules = ["torch", "icecream", "category_encoders", "sklearn", "numpy", "pandas", "tomli"]
    missing: List[str] = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)

    if missing:
        req_path = repo_dir / "requirements.txt"
        raise RuntimeError(
            "Missing TabSyn dependencies: "
            + ", ".join(missing)
            + f". Install them in tabsyn_env (e.g. pip install -r {req_path})."
        )


def _infer_label_col(train_df: pd.DataFrame, label_col: str | None) -> str:
    if label_col and label_col in train_df.columns:
        return label_col
    if "target" in train_df.columns:
        return "target"
    if "class" in train_df.columns:
        return "class"
    return train_df.columns[-1]


def _is_classification(y: pd.Series) -> bool:
    unique = y.dropna().nunique()
    if pd.api.types.is_numeric_dtype(y):
        return unique <= 20
    return True


def _compute_index_mappings(columns: List[str], num_cols: List[str], cat_cols: List[str], label_col: str) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, str]]:
    idx_by_name = {name: i for i, name in enumerate(columns)}
    ordered = num_cols + cat_cols + [label_col]
    idx_mapping: Dict[int, int] = {}
    for new_idx, name in enumerate(ordered):
        idx_mapping[idx_by_name[name]] = new_idx
    inverse = {v: k for k, v in idx_mapping.items()}
    idx_name = {i: name for i, name in enumerate(columns)}
    return idx_mapping, inverse, idx_name


def _prepare_tabsyn_dataset(
    repo_dir: Path,
    dataset_key: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
) -> None:
    data_dir = repo_dir / "data" / dataset_key
    info_dir = repo_dir / "data" / "Info"
    data_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    columns = list(train_df.columns)
    feature_cols = [c for c in columns if c != label_col]

    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    y_train = train_df[[label_col]].to_numpy()
    y_test = test_df[[label_col]].to_numpy()

    if num_cols:
        x_num_train = train_df[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        x_num_test = test_df[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        np.save(data_dir / "X_num_train.npy", x_num_train)
        np.save(data_dir / "X_num_test.npy", x_num_test)

    if cat_cols:
        train_cat = train_df[cat_cols].astype(str).fillna("__MISSING__").copy()
        test_cat = test_df[cat_cols].astype(str).fillna("__MISSING__").copy()

        # Guard against unseen categorical levels in test split.
        for col in cat_cols:
            known = set(train_cat[col].tolist())
            if not known:
                known = {"__MISSING__"}
            fallback_value = next(iter(known))
            test_cat[col] = test_cat[col].apply(lambda v: v if v in known else fallback_value)

        x_cat_train = train_cat.to_numpy(dtype=object)
        x_cat_test = test_cat.to_numpy(dtype=object)
    else:
        # TabSyn expects categorical arrays to exist in classification mode.
        x_cat_train = np.empty((len(train_df), 0), dtype=object)
        x_cat_test = np.empty((len(test_df), 0), dtype=object)
    np.save(data_dir / "X_cat_train.npy", x_cat_train)
    np.save(data_dir / "X_cat_test.npy", x_cat_test)

    np.save(data_dir / "y_train.npy", y_train)
    np.save(data_dir / "y_test.npy", y_test)

    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    idx_mapping, inverse, idx_name = _compute_index_mappings(columns, num_cols, cat_cols, label_col)
    label_idx = columns.index(label_col)

    y_series = train_df[label_col]
    if _is_classification(y_series):
        n_classes = int(pd.Series(y_series).dropna().nunique())
        task_type = "binclass" if n_classes <= 2 else "multiclass"
    else:
        n_classes = None
        task_type = "regression"

    info_payload: Dict[str, object] = {
        "name": dataset_key,
        "task_type": task_type,
        "n_classes": n_classes,
        "column_names": columns,
        "num_col_idx": [columns.index(c) for c in num_cols],
        "cat_col_idx": [columns.index(c) for c in cat_cols],
        "target_col_idx": [label_idx],
        "train_size": int(len(train_df)),
        "val_size": 0,
        "test_size": int(len(test_df)),
        "n_num_features": int(len(num_cols)),
        "n_cat_features": int(len(cat_cols)),
        "idx_mapping": idx_mapping,
        "inverse_idx_mapping": inverse,
        "idx_name_mapping": idx_name,
    }

    (data_dir / "info.json").write_text(json.dumps(info_payload, indent=2), encoding="utf-8")

    info_meta = {
        "name": dataset_key,
        "task_type": task_type,
        "header": "infer",
        "column_names": columns,
        "num_col_idx": [columns.index(c) for c in num_cols],
        "cat_col_idx": [columns.index(c) for c in cat_cols],
        "target_col_idx": [label_idx],
        "file_type": "csv",
        "data_path": str((data_dir / "train.csv").as_posix()),
        "test_path": str((data_dir / "test.csv").as_posix()),
    }
    (info_dir / f"{dataset_key}.json").write_text(json.dumps(info_meta, indent=2), encoding="utf-8")


def _ensure_tabsyn_package_layout(repo_dir: Path) -> None:
    package_inits = [
        repo_dir / "tabsyn" / "__init__.py",
        repo_dir / "tabsyn" / "vae" / "__init__.py",
        repo_dir / "baselines" / "__init__.py",
    ]
    for init_file in package_inits:
        init_file.parent.mkdir(parents=True, exist_ok=True)
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")


def _run_command(cmd: List[str], cwd: Path, timeout_seconds: int) -> None:
    print(f"[TabSyn] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True, timeout=timeout_seconds)


def _run_external_tabsyn(
    repo_dir: Path,
    dataset_key: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    outdir: Path,
    python_exec: str,
    timeout_train_seconds: int,
    timeout_sample_seconds: int,
    gpu: int,
    vae_epochs: int,
    tabsyn_epochs: int,
    batch_size: int,
    num_workers: int,
    sample_steps: int,
) -> pd.DataFrame:
    _ensure_tabsyn_package_layout(repo_dir)

    merged_train = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    _prepare_tabsyn_dataset(
        repo_dir=repo_dir,
        dataset_key=dataset_key,
        train_df=merged_train,
        test_df=test_df,
        label_col=label_col,
    )

    tmp_csv = outdir.resolve() / "_tabsyn_sample.csv"
    tmp_csv.parent.mkdir(parents=True, exist_ok=True)

    _run_command(
        [
            python_exec,
            "main.py",
            "--dataname",
            dataset_key,
            "--method",
            "vae",
            "--mode",
            "train",
            "--gpu",
            str(gpu),
            "--vae_epochs",
            str(vae_epochs),
            "--tabsyn_batch_size",
            str(batch_size),
            "--tabsyn_num_workers",
            str(num_workers),
        ],
        cwd=repo_dir,
        timeout_seconds=timeout_train_seconds,
    )

    _run_command(
        [
            python_exec,
            "main.py",
            "--dataname",
            dataset_key,
            "--method",
            "tabsyn",
            "--mode",
            "train",
            "--gpu",
            str(gpu),
            "--tabsyn_epochs",
            str(tabsyn_epochs),
            "--tabsyn_batch_size",
            str(batch_size),
            "--tabsyn_num_workers",
            str(num_workers),
        ],
        cwd=repo_dir,
        timeout_seconds=timeout_train_seconds,
    )

    _run_command(
        [
            python_exec,
            "main.py",
            "--dataname",
            dataset_key,
            "--method",
            "tabsyn",
            "--mode",
            "sample",
            "--save_path",
            str(tmp_csv),
            "--gpu",
            str(gpu),
            "--steps",
            str(sample_steps),
        ],
        cwd=repo_dir,
        timeout_seconds=timeout_sample_seconds,
    )

    if not tmp_csv.exists():
        raise RuntimeError("TabSyn did not generate sample CSV.")

    sampled = pd.read_csv(tmp_csv)
    if len(sampled) != len(train_df):
        sampled = sampled.sample(n=len(train_df), replace=len(sampled) < len(train_df), random_state=0)

    sampled = ensure_schema(train_df, sampled)
    return sampled.reset_index(drop=True)


def generate(args: argparse.Namespace) -> GenerationResult:
    repo_dir = _resolve_repo_dir(args.repo_dir)
    _check_tabsyn_dependencies(repo_dir)
    outdir = Path(args.outdir)

    train_df = pd.read_csv(args.real_train_csv)
    val_df = pd.read_csv(args.real_val_csv)
    test_df = pd.read_csv(args.real_test_csv)
    label_col = _infer_label_col(train_df, args.label_col)

    try:
        sampled = _run_external_tabsyn(
            repo_dir=repo_dir,
            dataset_key=args.dataset_name,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            label_col=label_col,
            outdir=outdir,
            python_exec=args.python_exec,
            timeout_train_seconds=args.timeout_train,
            timeout_sample_seconds=args.timeout_sample,
            gpu=args.gpu,
            vae_epochs=args.vae_epochs,
            tabsyn_epochs=args.tabsyn_epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_steps=args.sample_steps,
        )
        return GenerationResult(synthetic_df=sampled, backend="tabsyn_external", notes=f"repo={repo_dir}")
    except Exception as exc:
        if not args.allow_fallbacks:
            raise RuntimeError(f"TabSyn failed without fallbacks enabled: {exc}\n{traceback.format_exc()}") from exc

        if args.fallback_method == "sdv_ctgan":
            sampled = sdv_ctgan_generate(train_df=train_df, n_rows=len(train_df), seed=args.seed)
            sampled = ensure_schema(train_df, sampled)
            return GenerationResult(
                synthetic_df=sampled,
                backend="sdv_ctgan_fallback",
                notes=f"fallback after TabSyn failure: {exc}",
            )

        sampled = bootstrap_generate(train_df=train_df, n_rows=len(train_df), seed=args.seed)
        sampled = cast_like_train(train_df, sampled)
        return GenerationResult(
            synthetic_df=sampled,
            backend="bootstrap_fallback",
            notes=f"fallback after TabSyn failure: {exc}",
        )



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TabSyn wrapper and emit synthetic.csv")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--real_train_csv", required=True)
    parser.add_argument("--real_val_csv", required=True)
    parser.add_argument("--real_test_csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--repo_dir", default=None)
    parser.add_argument("--python_exec", default=sys.executable)
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout_train", type=int, default=7200)
    parser.add_argument("--timeout_sample", type=int, default=1800)
    parser.add_argument("--vae_epochs", type=int, default=100)
    parser.add_argument("--tabsyn_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--allow_fallbacks", action="store_true")
    parser.add_argument("--fallback_method", choices=["sdv_ctgan", "bootstrap"], default="sdv_ctgan")
    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        result = generate(args)
        result.synthetic_df.to_csv(outdir / "synthetic.csv", index=False)
        write_status(
            outdir=outdir,
            method="tabsyn",
            status="ok",
            backend=result.backend,
            seed=args.seed,
            notes=result.notes,
            n_rows=len(result.synthetic_df),
        )
    except Exception as exc:
        write_status(
            outdir=outdir,
            method="tabsyn",
            status="fail",
            backend="none",
            seed=args.seed,
            notes=str(exc),
            n_rows=0,
        )
        raise


if __name__ == "__main__":
    main()
