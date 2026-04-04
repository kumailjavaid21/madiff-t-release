from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common import (  # noqa: E402
    GenerationResult,
    bootstrap_generate,
    ensure_schema,
    sdv_ctgan_generate,
    write_status,
)


DATASET_ALIAS = {
    "adult_income": "adult",
    "adult": "adult",
    "breast_cancer": "breast_cancer",
    "pima": "pima",
    "heart": "heart",
    "credit": "german",
    "german": "german",
}


def _resolve_repo_dir(explicit_repo_dir: str | None) -> Path:
    candidates: List[Path] = []
    if explicit_repo_dir:
        candidates.append(Path(explicit_repo_dir))

    import os

    env_repo = os.environ.get("STASY_REPO")
    if env_repo:
        candidates.append(Path(env_repo))

    candidates.append(PROJECT_ROOT / "baselines" / "stasy" / "repo")
    candidates.append(PROJECT_ROOT / "paper_revC" / "external" / "stasy")

    for candidate in candidates:
        if (candidate / "main.py").exists():
            return candidate
    raise FileNotFoundError("STaSy repository not found. Set --repo_dir or STASY_REPO.")


def _check_stasy_dependencies() -> None:
    required_modules = ["tensorflow", "ml_collections", "torch", "torchdiffeq", "numpy", "pandas", "sklearn", "tensorflow_datasets"]
    missing: List[str] = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    if missing:
        raise RuntimeError(
            "Missing STaSy dependencies: "
            + ", ".join(missing)
            + ". Install them in stasy_env per repository requirement.yaml."
        )


def _canonical_dataset_name(name: str) -> str:
    key = name.lower().strip()
    return DATASET_ALIAS.get(key, key)


def _patch_stasy_datasets_py(repo_dir: Path) -> None:
    datasets_py = repo_dir / "datasets.py"
    text = datasets_py.read_text(encoding="utf-8")
    needle = "  if batch_size % torch.cuda.device_count() != 0:"
    replacement = (
        "  n_devices = max(torch.cuda.device_count(), 1)\n"
        "  if batch_size % n_devices != 0:"
    )
    if needle in text and "n_devices = max(torch.cuda.device_count(), 1)" not in text:
        text = text.replace(needle, replacement)
        text = text.replace(
            "f'the number of devices ({torch.cuda.device_count()})')",
            "f'the number of devices ({n_devices})')",
        )
        datasets_py.write_text(text, encoding="utf-8")


def _build_stasy_arrays(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    train = train_df.copy().reset_index(drop=True)
    test = test_df.copy().reset_index(drop=True)

    train = train.loc[:, train_df.columns]
    test = test.loc[:, train_df.columns]

    metadata_columns: List[Dict[str, object]] = []

    for col in train.columns:
        if pd.api.types.is_numeric_dtype(train[col]):
            train[col] = pd.to_numeric(train[col], errors="coerce")
            test[col] = pd.to_numeric(test[col], errors="coerce")
            fill = float(train[col].median()) if np.isfinite(train[col].median()) else 0.0
            train[col] = train[col].fillna(fill)
            test[col] = test[col].fillna(fill)
            metadata_columns.append(
                {
                    "name": col,
                    "type": "continuous",
                    "min": float(train[col].min()),
                    "max": float(train[col].max()),
                }
            )
        else:
            train[col] = train[col].astype("string").fillna("__MISSING__")
            test[col] = test[col].astype("string").fillna("__MISSING__")
            cats = pd.Index(pd.concat([train[col], test[col]], axis=0).dropna().unique()).tolist()
            cat_to_idx = {cat: idx for idx, cat in enumerate(cats)}
            train[col] = train[col].map(cat_to_idx).astype(np.float32)
            test[col] = test[col].map(cat_to_idx).astype(np.float32)
            metadata_columns.append(
                {
                    "name": col,
                    "type": "categorical",
                    "size": int(len(cats)),
                    "i2s": cats,
                }
            )

    train_arr = train.to_numpy(dtype=np.float32)
    test_arr = test.to_numpy(dtype=np.float32)

    metadata = {
        "columns": metadata_columns,
        "problem_type": "binary_classification",
    }
    return train_arr, test_arr, metadata


def _output_dim_from_metadata(metadata: Dict[str, object]) -> int:
    dim = 0
    for col in metadata.get("columns", []):
        if col.get("type") == "categorical":
            dim += int(col.get("size", 0))
        else:
            dim += 1
    return int(dim)


def _write_stasy_dataset_and_config(
    repo_dir: Path,
    dataset_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    train_epochs: int,
) -> Path:
    canonical = _canonical_dataset_name(dataset_name)

    train_arr, test_arr, metadata = _build_stasy_arrays(train_df, test_df)

    dataset_dir = repo_dir / "tabular_datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.savez(dataset_dir / f"{canonical}.npz", train=train_arr, test=test_arr)
    (dataset_dir / f"{canonical}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    n_features = _output_dim_from_metadata(metadata)
    batch_size = int(min(max(64, len(train_df)), 1024))
    batch_size = max(32, batch_size)

    config_text = f'''from configs.default_tabular_configs import get_default_configs\n\n\ndef get_config():\n    config = get_default_configs()\n    config.seed = {seed}\n    config.data.dataset = "{canonical}"\n    config.data.image_size = {n_features}\n\n    config.training.batch_size = {batch_size}\n    config.eval.batch_size = {batch_size}\n    config.training.epoch = {int(train_epochs)}\n\n    config.training.sde = "vesde"\n    config.training.continuous = True\n    config.training.reduce_mean = True\n\n    config.sampling.method = "ode"\n    config.sampling.predictor = "euler_maruyama"\n    config.sampling.corrector = "none"\n\n    config.model.layer_type = "concatsquash"\n    config.model.name = "ncsnpp_tabular"\n    config.model.scale_by_sigma = False\n    config.model.ema_rate = 0.999\n    config.model.activation = "elu"\n    config.model.nf = 64\n    config.model.hidden_dims = (256, 512, 512, 256)\n    config.model.conditional = True\n    config.model.embedding_type = "fourier"\n    config.model.fourier_scale = 16\n    config.model.conv_size = 3\n\n    config.model.sigma_min = 0.01\n    config.model.sigma_max = 10.0\n\n    config.optim.lr = 2e-3\n    config.test.n_iter = 1\n    return config\n'''

    config_path = repo_dir / "configs" / f"{canonical}.py"
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def _run_command(cmd: List[str], cwd: Path, timeout_seconds: int) -> None:
    print(f"[STaSy] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True, timeout=timeout_seconds)


def _run_external_stasy(
    repo_dir: Path,
    dataset_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    outdir: Path,
    python_exec: str,
    timeout_seconds: int,
    seed: int,
    train_epochs: int,
) -> pd.DataFrame:
    _patch_stasy_datasets_py(repo_dir)

    merged_train = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    config_path = _write_stasy_dataset_and_config(
        repo_dir=repo_dir,
        dataset_name=dataset_name,
        train_df=merged_train,
        test_df=test_df,
        seed=seed,
        train_epochs=train_epochs,
    )

    workdir = (outdir / "_stasy_work").resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    config_arg = config_path.relative_to(repo_dir).as_posix()

    _run_command(
        [python_exec, "main.py", f"--config={config_arg}", "--mode", "train", "--workdir", str(workdir)],
        cwd=repo_dir,
        timeout_seconds=timeout_seconds,
    )
    sample_candidates = sorted((workdir / "samples").glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sample_candidates:
        raise RuntimeError("STaSy finished but no samples/*.csv found.")

    sampled = pd.read_csv(sample_candidates[0], index_col=0)
    sampled = ensure_schema(train_df, sampled)
    if len(sampled) != len(train_df):
        sampled = sampled.sample(n=len(train_df), replace=len(sampled) < len(train_df), random_state=seed)
    return sampled.reset_index(drop=True)


def generate(args: argparse.Namespace) -> GenerationResult:
    repo_dir = _resolve_repo_dir(args.repo_dir)
    _check_stasy_dependencies()
    outdir = Path(args.outdir)

    train_df = pd.read_csv(args.real_train_csv)
    val_df = pd.read_csv(args.real_val_csv)
    test_df = pd.read_csv(args.real_test_csv)

    try:
        sampled = _run_external_stasy(
            repo_dir=repo_dir,
            dataset_name=args.dataset_name,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            outdir=outdir,
            python_exec=args.python_exec,
            timeout_seconds=args.external_timeout,
            seed=args.seed,
            train_epochs=args.train_epochs,
        )
        return GenerationResult(synthetic_df=sampled, backend="stasy_external", notes=f"repo={repo_dir}")
    except Exception as exc:
        if not args.allow_fallbacks:
            raise RuntimeError(f"STaSy failed without fallbacks enabled: {exc}\n{traceback.format_exc()}") from exc

        if args.fallback_method == "sdv_ctgan":
            sampled = sdv_ctgan_generate(train_df=train_df, n_rows=len(train_df), seed=args.seed)
            sampled = ensure_schema(train_df, sampled)
            return GenerationResult(
                synthetic_df=sampled,
                backend="sdv_ctgan_fallback",
                notes=f"fallback after STaSy failure: {exc}",
            )

        sampled = bootstrap_generate(train_df=train_df, n_rows=len(train_df), seed=args.seed)
        sampled = ensure_schema(train_df, sampled)
        return GenerationResult(
            synthetic_df=sampled,
            backend="bootstrap_fallback",
            notes=f"fallback after STaSy failure: {exc}",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run STaSy wrapper and emit synthetic.csv")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--real_train_csv", required=True)
    parser.add_argument("--real_val_csv", required=True)
    parser.add_argument("--real_test_csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--repo_dir", default=None)
    parser.add_argument("--python_exec", default=sys.executable)
    parser.add_argument("--external_timeout", type=int, default=7200)
    parser.add_argument("--train_epochs", type=int, default=100)
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
            method="stasy",
            status="ok",
            backend=result.backend,
            seed=args.seed,
            notes=result.notes,
            n_rows=len(result.synthetic_df),
        )
    except Exception as exc:
        write_status(
            outdir=outdir,
            method="stasy",
            status="fail",
            backend="none",
            seed=args.seed,
            notes=str(exc),
            n_rows=0,
        )
        raise


if __name__ == "__main__":
    main()
