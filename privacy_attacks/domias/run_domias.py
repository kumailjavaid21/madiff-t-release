from __future__ import annotations

import argparse
import importlib
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force CPU execution for DOMIAS internals to avoid mixed-device CTGAN failures.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from privacy_attacks.common import encode_mixed
from privacy_attacks.shadow_mia.run_shadow_mia import run_shadow_mia


def _patch_domias_runtime() -> None:
    try:
        from ctgan.data_sampler import DataSampler
    except Exception:
        return

    if getattr(DataSampler, "_madifft_patched", False):
        return

    original_init = DataSampler.__init__
    original_sample_data = DataSampler.sample_data

    def compat_init(self, data, output_info, log_frequency):
        original_init(self, data, output_info, log_frequency)
        self._compat_data_ref = data

    def compat_sample_data(self, *args):
        if len(args) == 3:
            n, col, opt = args
            data = getattr(self, "_compat_data_ref", None)
            if data is None:
                raise TypeError("Missing cached data for compatibility sample_data call")
            return original_sample_data(self, data, n, col, opt)
        return original_sample_data(self, *args)

    DataSampler.__init__ = compat_init
    DataSampler.sample_data = compat_sample_data
    DataSampler._madifft_patched = True



def _run_official_domias(
    x_train: np.ndarray,
    x_test: np.ndarray,
    x_synth: np.ndarray,
    seed: int,
) -> float:
    import torch
    from scipy.stats import multivariate_normal
    import domias.evaluator as domias_evaluator
    import domias.baselines as domias_baselines
    from domias.models.generator import GeneratorInterface

    # DOMIAS may default to CUDA when available; pin to CPU for stability on Windows.
    try:
        torch.set_default_device("cpu")
    except Exception:
        pass
    torch.cuda.is_available = lambda: False

    class ReplayGenerator(GeneratorInterface):
        def __init__(self, source: np.ndarray, seed_value: int) -> None:
            self._source = pd.DataFrame(source)
            self._rng = np.random.default_rng(seed_value)

        def fit(self, data: pd.DataFrame):
            return self

        def generate(self, count: int) -> pd.DataFrame:
            if len(self._source) == 0:
                raise RuntimeError("Synthetic source is empty.")
            idx = self._rng.integers(0, len(self._source), size=count)
            return self._source.iloc[idx].reset_index(drop=True)

    # Cap evaluation size for numerical stability and to avoid CTGAN/torch
    # crashes on very large tabular splits in Windows environments.
    member_size = min(len(x_train), len(x_test), 500)
    if member_size < 10:
        raise RuntimeError("DOMIAS requires at least 10 train and 10 test rows.")

    reference_size = max(10, min(len(x_test), member_size))
    dataset = np.vstack([x_train[:member_size], x_test[:member_size], x_test[:reference_size]])

    original_pdf = multivariate_normal.pdf
    original_compute_wd = domias_evaluator.compute_wd
    original_compute_metrics_baseline = domias_baselines.compute_metrics_baseline
    original_evaluator_compute_metrics = getattr(domias_evaluator, "compute_metrics_baseline", None)

    def _safe_pdf(*args, **kwargs):
        kwargs.setdefault("allow_singular", True)
        return original_pdf(*args, **kwargs)

    def _safe_compute_wd(x_real, x_syn):
        xr = np.asarray(x_real, dtype=np.float32)
        xs = np.asarray(x_syn, dtype=np.float32)
        n = int(min(len(xr), len(xs)))
        if n == 0:
            return 0.0
        # Lightweight stable proxy to avoid geomloss/pykeops runtime issues on Windows.
        return float(np.linalg.norm(xr[:n] - xs[:n], axis=1).mean())

    def _safe_compute_metrics_baseline(*args, **kwargs):
        if "y_scores" in kwargs:
            ys = kwargs["y_scores"]
            ys = np.asarray(ys, dtype=np.float64)
            kwargs["y_scores"] = np.nan_to_num(ys, nan=0.5, posinf=1.0, neginf=0.0)
        elif len(args) >= 1:
            arg_list = list(args)
            ys = np.asarray(arg_list[0], dtype=np.float64)
            arg_list[0] = np.nan_to_num(ys, nan=0.5, posinf=1.0, neginf=0.0)
            args = tuple(arg_list)
        return original_compute_metrics_baseline(*args, **kwargs)

    multivariate_normal.pdf = _safe_pdf
    domias_evaluator.compute_wd = _safe_compute_wd
    domias_baselines.compute_metrics_baseline = _safe_compute_metrics_baseline
    if original_evaluator_compute_metrics is not None:
        domias_evaluator.compute_metrics_baseline = _safe_compute_metrics_baseline
    try:
        perf = domias_evaluator.evaluate_performance(
            generator=ReplayGenerator(x_synth, seed),
            dataset=dataset,
            mem_set_size=member_size,
            reference_set_size=reference_size,
            training_epochs=5,
            synthetic_sizes=[max(50, min(len(x_synth), 500))],
            density_estimator="kde",
            seed=seed,
            device="cpu",
        )
    finally:
        multivariate_normal.pdf = original_pdf
        domias_evaluator.compute_wd = original_compute_wd
        domias_baselines.compute_metrics_baseline = original_compute_metrics_baseline
        if original_evaluator_compute_metrics is not None:
            domias_evaluator.compute_metrics_baseline = original_evaluator_compute_metrics
    size_key = next(iter(perf.keys()))
    return float(perf[size_key]["MIA_performance"]["domias"]["aucroc"])


def _stabilize_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
    x_synth: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Replace non-finite values with train-column medians so DOMIAS metrics
    # never receive NaNs/Infs from mixed-type preprocessing artifacts.
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    x_synth = np.asarray(x_synth, dtype=np.float32)

    train_finite = np.where(np.isfinite(x_train), x_train, np.nan)
    medians = np.nanmedian(train_finite, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)

    def _fill_nonfinite(x: np.ndarray) -> np.ndarray:
        bad = ~np.isfinite(x)
        if bad.any():
            x = x.copy()
            rows, cols = np.where(bad)
            x[rows, cols] = medians[cols]
        return x

    x_train = _fill_nonfinite(x_train)
    x_test = _fill_nonfinite(x_test)
    x_synth = _fill_nonfinite(x_synth)

    var = np.var(x_train, axis=0)
    keep = np.isfinite(var) & (var > eps)
    if keep.sum() == 0:
        keep = np.ones(x_train.shape[1], dtype=bool)

    x_train = x_train[:, keep]
    x_test = x_test[:, keep]
    x_synth = x_synth[:, keep]

    # Add tiny jitter to avoid singular covariance in Gaussian density estimators.
    rng = np.random.default_rng(0)
    noise_scale = 1e-6
    x_train = x_train + rng.normal(0.0, noise_scale, size=x_train.shape).astype(np.float32)
    x_test = x_test + rng.normal(0.0, noise_scale, size=x_test.shape).astype(np.float32)
    x_synth = x_synth + rng.normal(0.0, noise_scale, size=x_synth.shape).astype(np.float32)
    return x_train, x_test, x_synth


def run_domias(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    seed: int,
) -> float:
    encoded = encode_mixed(train_df=train_df, test_df=test_df, synth_df=synth_df)
    x_train = np.asarray(encoded.X_train, dtype=np.float32)
    x_test = np.asarray(encoded.X_test, dtype=np.float32)
    x_synth = np.asarray(encoded.X_synth, dtype=np.float32)
    x_train, x_test, x_synth = _stabilize_features(x_train=x_train, x_test=x_test, x_synth=x_synth)

    _patch_domias_runtime()
    return _run_official_domias(x_train=x_train, x_test=x_test, x_synth=x_synth, seed=seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DOMIAS attack with strict/fallback behavior.")
    parser.add_argument("--real_train_csv", required=True)
    parser.add_argument("--real_test_csv", required=True)
    parser.add_argument("--synthetic_csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_fallbacks", action="store_true")
    parser.add_argument("--k", type=int, default=5, help="k for shadow-mia fallback")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.real_train_csv)
    test_df = pd.read_csv(args.real_test_csv)
    synth_df = pd.read_csv(args.synthetic_csv)

    try:
        importlib.import_module("domias")
    except Exception as exc:
        if not args.allow_fallbacks:
            failure = {
                "status": "fail",
                "error": f"domias import failed: {exc}",
                "traceback": traceback.format_exc(),
            }
            (outdir / "domias_failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
            raise RuntimeError(f"domias dependency missing and fallbacks disabled: {exc}") from exc

        # Fallback allowed: run shadow MIA and persist only shadow artifact.
        shadow_auc = run_shadow_mia(
            train_df=train_df,
            test_df=test_df,
            synth_df=synth_df,
            seed=args.seed,
            k=args.k,
        )
        payload = {
            "status": "ok",
            "shadow_auc": float(shadow_auc),
            "method_used": "shadow_mia_fallback",
            "reason": f"domias import failed: {exc}",
        }
        (outdir / "shadow_auc.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    try:
        auc = run_domias(train_df=train_df, test_df=test_df, synth_df=synth_df, seed=args.seed)
        payload: Dict[str, Any] = {
            "status": "ok",
            "domias_auc": float(auc),
            "method_used": "domias_official",
            "error": None,
        }
        (outdir / "domias_auc.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        if args.allow_fallbacks:
            shadow_auc = run_shadow_mia(
                train_df=train_df,
                test_df=test_df,
                synth_df=synth_df,
                seed=args.seed,
                k=args.k,
            )
            payload = {
                "status": "ok",
                "shadow_auc": float(shadow_auc),
                "method_used": "shadow_mia_fallback",
                "reason": f"domias runtime failed: {exc}",
            }
            (outdir / "shadow_auc.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return

        failure = {
            "status": "fail",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (outdir / "domias_failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
