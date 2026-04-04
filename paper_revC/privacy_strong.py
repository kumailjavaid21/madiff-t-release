from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

from .datasets import DatasetSplit
from .preprocess_utils import build_feature_preprocessor, encode_features

try:
    from scipy.special import logsumexp  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without scipy
    def logsumexp(a, axis=None, keepdims: bool = False):
        a = np.asarray(a)
        a_max = np.max(a, axis=axis, keepdims=True)
        out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
        if not keepdims and axis is not None:
            out = np.squeeze(out, axis=axis)
        return out


def _patch_geomloss_generic_logsumexp() -> None:
    try:
        import torch
        import geomloss.sinkhorn_samples as sinkhorn_samples
        import geomloss.sinkhorn_divergence as sinkhorn_divergence

        def _fallback_generic_logsumexp(formula: str, *_, **__):
            def _log_conv(x, y, b, p, ranges=None):
                x_np = x.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
                diff = x_np[:, None, :] - y_np[None, :, :]
                if "SqDist" in formula:
                    cost = np.sum(diff ** 2, axis=-1)
                    if "IntCst(2)" in formula or "/ IntCst(2)" in formula:
                        cost = cost / 2.0
                else:
                    cost = np.sqrt(np.sum(diff ** 2, axis=-1))
                b_np = b.detach().cpu().numpy().reshape(1, -1)
                p_np = p.detach().cpu().numpy().reshape(1, 1)
                scores = b_np - p_np * cost
                logsum = logsumexp(scores, axis=1)
                return torch.as_tensor(logsum, device=x.device, dtype=x.dtype)

            return _log_conv

        if not hasattr(sinkhorn_samples, "generic_logsumexp"):
            sinkhorn_samples.generic_logsumexp = _fallback_generic_logsumexp
        if not hasattr(sinkhorn_divergence, "generic_logsumexp"):
            sinkhorn_divergence.generic_logsumexp = _fallback_generic_logsumexp
    except Exception:
        return


def _patch_domias_wd_backend() -> None:
    try:
        import domias.metrics.wd as domias_wd
        import domias.evaluator as domias_eval

        try:
            from scipy.stats import wasserstein_distance  # type: ignore

            def _safe_compute_wd(X_syn: np.ndarray, X: np.ndarray) -> float:
                X_ = np.asarray(X, dtype=np.float32)
                X_syn_ = np.asarray(X_syn, dtype=np.float32)
                if X_.ndim != 2 or X_syn_.ndim != 2:
                    return float("nan")
                n_features = min(X_.shape[1], X_syn_.shape[1])
                if n_features == 0:
                    return float("nan")
                distances = []
                for idx in range(n_features):
                    distances.append(
                        float(wasserstein_distance(X_[:, idx], X_syn_[:, idx]))
                    )
                return float(np.nanmean(distances))
        except Exception:

            def _safe_compute_wd(X_syn: np.ndarray, X: np.ndarray) -> float:
                X_ = np.asarray(X, dtype=np.float32)
                X_syn_ = np.asarray(X_syn, dtype=np.float32)
                if X_.ndim != 2 or X_syn_.ndim != 2:
                    return float("nan")
                if X_.shape[1] == 0 or X_syn_.shape[1] == 0:
                    return float("nan")
                mean_real = np.nanmean(X_, axis=0)
                mean_syn = np.nanmean(X_syn_, axis=0)
                min_len = min(len(mean_real), len(mean_syn))
                return float(np.nanmean(np.abs(mean_real[:min_len] - mean_syn[:min_len])))

        domias_wd.compute_wd = _safe_compute_wd
        domias_eval.compute_wd = _safe_compute_wd
    except Exception:
        return


def _patch_domias_ctgan_sampler() -> None:
    try:
        import inspect
        import domias.models.ctgan as domias_ctgan

        sampler_cls = getattr(domias_ctgan, "DataSampler", None)
        if sampler_cls is None:
            return

        orig_sample = sampler_cls.sample_data
        try:
            signature = inspect.signature(orig_sample)
            if "opt" in signature.parameters and signature.parameters["opt"].default is not inspect._empty:
                return
        except Exception:
            pass

        def _compat_sample_data(self, data, opt=None, *args, **kwargs):
            return orig_sample(self, data, opt, *args, **kwargs)

        sampler_cls.sample_data = _compat_sample_data
    except Exception:
        return


def _ensure_numpy(data, dtype=np.float32) -> np.ndarray:
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy(dtype=dtype)
    return np.asarray(data, dtype=dtype)


def _run_domias(
    dataset: DatasetSplit,
    synth_df: pd.DataFrame,
    label_col: str,
    seed: int,
    domias_cfg: dict,
) -> list[dict]:
    try:
        from domias.evaluator import evaluate_performance
        from domias.models.generator import GeneratorInterface
    except ImportError:
        logging.info("DOMIAS is not installed; skipping strong privacy audit.")
        return [
            {
                "audit_name": "DOMIAS",
                "score_name": "auc",
                "score_value": None,
                "member_split": "train",
                "nonmember_split": "val",
                "eval_split": "train_vs_val",
                "status": "skipped",
                "notes": "DOMIAS not installed",
            }
        ]

    class _SimpleDomiasGenerator(GeneratorInterface):
        def __init__(self, data: pd.DataFrame):
            self._data = data

        def fit(self, _: pd.DataFrame) -> "_SimpleDomiasGenerator":
            return self

        def generate(self, count: int) -> pd.DataFrame:
            sample = self._data.sample(count, replace=True, random_state=seed)
            return sample.reset_index(drop=True)

    _patch_geomloss_generic_logsumexp()
    _patch_domias_wd_backend()
    _patch_domias_ctgan_sampler()

    train_df = dataset.train.copy()
    val_df = dataset.val.copy() if len(dataset.val) else None
    test_df = dataset.test.copy() if len(dataset.test) else None
    synth_local = synth_df.copy()

    feature_cols = [col for col in train_df.columns if col != label_col]
    constant_cols: list[str] = []
    for col in feature_cols:
        series = train_df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.dropna().empty:
                constant_cols.append(col)
                continue
            variance = float(numeric.var())
            value_range = float(numeric.max() - numeric.min())
            if variance <= 1e-12 or value_range <= 1e-8:
                constant_cols.append(col)
        else:
            if series.nunique(dropna=False) <= 1:
                constant_cols.append(col)
    if constant_cols:
        logging.info("DOMIAS: dropped constant columns: %s", constant_cols)
        train_df = train_df.drop(columns=constant_cols)
        synth_local = synth_local.drop(columns=constant_cols, errors="ignore")
        if val_df is not None:
            val_df = val_df.drop(columns=constant_cols, errors="ignore")
        if test_df is not None:
            test_df = test_df.drop(columns=constant_cols, errors="ignore")
        feature_cols = [col for col in train_df.columns if col != label_col]

    if not feature_cols:
        return [
            {
                "audit_name": "DOMIAS",
                "score_name": "auc",
                "score_value": None,
                "member_split": "train",
                "nonmember_split": "val" if len(dataset.val) else "test",
                "eval_split": "train_vs_val",
                "status": "failed",
                "notes": "No feature columns available for DOMIAS.",
            }
        ]

    try:
        preprocessor, _, _, _ = build_feature_preprocessor(train_df, label_col)
        X_train = encode_features(preprocessor, train_df, label_col)
        X_syn = encode_features(preprocessor, synth_local, label_col)
        X_val = (
            encode_features(preprocessor, val_df, label_col)
            if val_df is not None and len(val_df)
            else None
        )
        X_test = (
            encode_features(preprocessor, test_df, label_col)
            if test_df is not None and len(test_df)
            else None
        )
    except Exception as exc:
        return [
            {
                "audit_name": "DOMIAS",
                "score_name": "auc",
                "score_value": None,
                "member_split": "train",
                "nonmember_split": "val" if len(dataset.val) else "test",
                "eval_split": "train_vs_val",
                "status": "failed",
                "notes": f"DOMIAS preprocessing failed: {exc}",
            }
        ]

    X_train = np.nan_to_num(_ensure_numpy(X_train, dtype=np.float32))
    X_syn = np.nan_to_num(_ensure_numpy(X_syn, dtype=np.float32))
    X_val = np.nan_to_num(_ensure_numpy(X_val, dtype=np.float32)) if X_val is not None else None
    X_test = np.nan_to_num(_ensure_numpy(X_test, dtype=np.float32)) if X_test is not None else None

    if X_syn.shape[1] != X_train.shape[1]:
        return [
            {
                "audit_name": "DOMIAS",
                "score_name": "auc",
                "score_value": None,
                "member_split": "train",
                "nonmember_split": "val" if len(dataset.val) else "test",
                "eval_split": "train_vs_val",
                "status": "failed",
                "notes": f"Feature mismatch: train={X_train.shape[1]} synth={X_syn.shape[1]}",
            }
        ]

    member_split = "train"
    nonmember_pref = str(domias_cfg.get("domias_nonmember_split", "val_or_test")).lower()
    if nonmember_pref == "test" and X_test is not None and len(X_test):
        nonmember_split = "test"
        nonmember_pool = X_test
    elif nonmember_pref == "val" and X_val is not None and len(X_val):
        nonmember_split = "val"
        nonmember_pool = X_val
    else:
        nonmember_split = "val" if X_val is not None and len(X_val) else "test"
        nonmember_pool = X_val if nonmember_split == "val" else X_test
    if nonmember_pool is None or len(nonmember_pool) == 0:
        return [
            {
                "audit_name": "DOMIAS",
                "score_name": "auc",
                "score_value": None,
                "member_split": member_split,
                "nonmember_split": nonmember_split,
                "eval_split": f"{member_split}_vs_{nonmember_split}",
                "status": "failed",
                "notes": "No non-member split available for DOMIAS.",
            }
        ]

    rng = np.random.default_rng(seed)
    if len(nonmember_pool) < len(X_train):
        idx = rng.choice(len(nonmember_pool), size=len(X_train), replace=True)
    else:
        idx = rng.choice(len(nonmember_pool), size=len(X_train), replace=False)
    X_nonmember = nonmember_pool[idx]

    reference_pool = X_test if X_test is not None and len(X_test) else nonmember_pool
    if len(reference_pool) == 0:
        return [
            {
                "audit_name": "DOMIAS",
                "score_name": "auc",
                "score_value": None,
                "member_split": member_split,
                "nonmember_split": nonmember_split,
                "eval_split": f"{member_split}_vs_{nonmember_split}",
                "status": "failed",
                "notes": "No reference split available for DOMIAS.",
            }
        ]

    scaler = MinMaxScaler()
    X_train_np = _ensure_numpy(X_train, dtype=np.float32)
    scaler.fit(X_train_np)
    X_train_scaled = scaler.transform(X_train_np).astype(np.float32, copy=False)
    X_nonmember_scaled = scaler.transform(_ensure_numpy(X_nonmember, dtype=np.float32)).astype(
        np.float32, copy=False
    )
    reference_scaled = scaler.transform(_ensure_numpy(reference_pool, dtype=np.float32)).astype(
        np.float32, copy=False
    )
    X_syn_scaled = scaler.transform(_ensure_numpy(X_syn, dtype=np.float32)).astype(
        np.float32, copy=False
    )

    mem_set_size = len(X_train_scaled)
    reference_set_size = len(reference_scaled)
    synthetic_size = min(len(X_syn_scaled), domias_cfg.get("max_synthetic", 1000))
    dataset_array = np.vstack([X_train_scaled, X_nonmember_scaled, reference_scaled]).astype(
        np.float32, copy=False
    )
    eval_split = f"{member_split}_vs_{nonmember_split}"

    generator = _SimpleDomiasGenerator(
        pd.DataFrame(X_syn_scaled, columns=[str(i) for i in range(X_syn_scaled.shape[1])])
    )
    def _fallback_mia_auc(
        members: np.ndarray, nonmembers: np.ndarray, synth: np.ndarray
    ) -> float | None:
        if members.size == 0 or nonmembers.size == 0 or synth.size == 0:
            return None
        dist_mem = cdist(members, synth, metric="euclidean").min(axis=1)
        dist_non = cdist(nonmembers, synth, metric="euclidean").min(axis=1)
        labels = np.concatenate([np.ones(len(dist_mem)), np.zeros(len(dist_non))])
        scores = -np.concatenate([dist_mem, dist_non])
        if len(np.unique(labels)) < 2:
            return None
        return float(roc_auc_score(labels, scores))

    try:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X has feature names, but MinMaxScaler was fitted without feature names",
                category=UserWarning,
            )
            results = evaluate_performance(
                generator,
                dataset_array,
                mem_set_size,
                reference_set_size,
                training_epochs=domias_cfg.get("training_epochs", 2000),
                synthetic_sizes=[synthetic_size],
                density_estimator=domias_cfg.get("density_estimator", "prior"),
                seed=seed,
                device=device,
            )
    except Exception as exc:
        logging.exception("DOMIAS evaluation failed with exception")
        fallback_auc = _fallback_mia_auc(X_train_scaled, X_nonmember_scaled, X_syn_scaled)
        rows = [
            {
                "audit_name": "DOMIAS",
                "score_name": "auc",
                "score_value": None,
                "member_split": member_split,
                "nonmember_split": nonmember_split,
                "eval_split": eval_split,
                "status": "failed",
                "notes": f"DOMIAS error: {exc}",
            }
        ]
        if fallback_auc is not None:
            rows.append(
                {
                    "audit_name": "DOMIAS",
                    "score_name": "distance_auc",
                    "score_value": float(fallback_auc),
                    "member_split": member_split,
                    "nonmember_split": nonmember_split,
                    "eval_split": eval_split,
                    "status": "success",
                    "notes": f"DOMIAS fallback distance AUC; original error: {exc}",
                }
            )
        return rows

    extracted = []
    perf = results.get(synthetic_size, {}).get("MIA_performance", {})
    for attack, metrics in perf.items():
        auc_value = metrics.get("AUCROC")
        if auc_value is not None:
            extracted.append(
                {
                    "audit_name": "DOMIAS",
                    "score_name": f"{attack}_auc",
                    "score_value": float(auc_value),
                    "member_split": member_split,
                    "nonmember_split": nonmember_split,
                    "eval_split": eval_split,
                    "status": "success",
                    "notes": "DOMIAS attack result",
                }
            )
            advantage = float(auc_value) - 0.5
            extracted.append(
                {
                    "audit_name": "DOMIAS",
                    "score_name": f"{attack}_advantage",
                    "score_value": float(advantage),
                    "member_split": member_split,
                    "nonmember_split": nonmember_split,
                    "eval_split": eval_split,
                    "status": "success",
                    "notes": "DOMIAS advantage over random guess",
                }
            )
    if not extracted:
        extracted.append(
            {
                "audit_name": "DOMIAS",
                "score_name": "auc",
                "score_value": None,
                "member_split": member_split,
                "nonmember_split": nonmember_split,
                "eval_split": eval_split,
                "status": "failed",
                "notes": "DOMIAS returned no usable metrics",
            }
        )
    return extracted


def _run_tapas(
    dataset: DatasetSplit,
    synth_df: pd.DataFrame,
    label_col: str,
    seed: int,
    tapas_cfg: dict,
) -> list[dict]:
    try:
        import importlib.util

        if importlib.util.find_spec("privacy_sdg_toolbox") is None:
            raise ImportError("privacy_sdg_toolbox not installed")
    except Exception:
        logging.info("TAPAS toolbox is not installed; skipping strong privacy audit.")
        return [
            {
                "audit_name": "TAPAS",
                "score_name": "auc",
                "score_value": None,
                "member_split": "train",
                "nonmember_split": "val",
                "eval_split": "train_vs_val",
                "status": "skipped",
                "notes": "TAPAS toolbox not installed",
            }
        ]

    try:
        from privacy_sdg_toolbox.attacks import run_membership_inference_attack
    except Exception as exc:
        logging.warning("TAPAS import failed: %s", exc)
        return [
            {
                "audit_name": "TAPAS",
                "score_name": "auc",
                "score_value": None,
                "member_split": "train",
                "nonmember_split": "val",
                "eval_split": "train_vs_val",
                "status": "failed",
                "notes": f"TAPAS import error: {exc}",
            }
        ]

    try:
        members = dataset.train.drop(columns=[label_col])
        nonmembers_source = dataset.val if len(dataset.val) else dataset.test
        nonmembers = nonmembers_source.drop(columns=[label_col])
        synth = synth_df.drop(columns=[label_col])
        nonmember_split = "val" if len(dataset.val) else "test"
        eval_split = f"train_vs_{nonmember_split}"
        result = run_membership_inference_attack(
            members=members,
            nonmembers=nonmembers,
            synthetic=synth,
            seed=seed,
            **tapas_cfg,
        )
        auc_value = result.get("auc") if isinstance(result, dict) else None
        return [
            {
                "audit_name": "TAPAS",
                "score_name": "auc",
                "score_value": auc_value,
                "member_split": "train",
                "nonmember_split": nonmember_split,
                "eval_split": eval_split,
                "status": "success" if auc_value is not None else "failed",
                "notes": "TAPAS membership inference",
            }
        ]
    except Exception as exc:
        logging.warning("TAPAS evaluation failed: %s", exc)
        return [
            {
                "audit_name": "TAPAS",
                "score_name": "auc",
                "score_value": None,
                "member_split": "train",
                "nonmember_split": "val" if len(dataset.val) else "test",
                "eval_split": f"train_vs_{'val' if len(dataset.val) else 'test'}",
                "status": "failed",
                "notes": f"TAPAS error: {exc}",
            }
        ]


def privacy_strong_audits(
    dataset: DatasetSplit,
    synth_df: pd.DataFrame,
    label_col: str,
    seed: int,
    strong_cfg: dict,
) -> list[dict]:
    rows: list[dict] = []
    if strong_cfg.get("enable_domias", False):
        domias_rows = _run_domias(dataset, synth_df, label_col, seed, strong_cfg)
        for domias_row in domias_rows:
            rows.append(domias_row)
    if strong_cfg.get("enable_tapas", False):
        rows.extend(_run_tapas(dataset, synth_df, label_col, seed, strong_cfg))
    return rows
