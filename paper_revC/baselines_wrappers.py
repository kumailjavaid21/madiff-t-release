from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer

from src.mas_generators.tabddpm.generator import TabDDPMGenerator
from .datasets import DatasetSplit
from .madifft_engine import run_madifft_evolution
from .tstr_eval import _build_preprocessor
from .tabular_utils import (
    encode_categorical,
    decode_categorical,
    sanitize_synthetic,
    discretize_continuous,
    inverse_discretize,
    parse_privbayes_method_name,
)


class BaselineUnavailable(Exception):
    """Indicate a baseline cannot run with the current dependencies."""


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _assemble_synthetic_df(
    reference: pd.DataFrame, label_col: str, X_syn: np.ndarray, y_syn: np.ndarray
) -> pd.DataFrame:
    feature_cols = [col for col in reference.columns if col != label_col]
    df = pd.DataFrame(X_syn, columns=feature_cols)
    df[label_col] = y_syn
    return df[reference.columns]


def _stringify_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [str(col) for col in renamed.columns]
    return renamed


def _class_guard_summary(
    dataset: DatasetSplit,
    synth_df: pd.DataFrame,
    threshold: float = 0.15,
) -> tuple[bool, float, float]:
    label_col = dataset.label_col
    train_X = _stringify_feature_frame(
        dataset.train.drop(columns=[label_col]).reset_index(drop=True)
    )
    synth_X = _stringify_feature_frame(
        synth_df.drop(columns=[label_col]).reset_index(drop=True)
    )
    synth_X = synth_X.reindex(columns=train_X.columns, fill_value=0)
    combined_X = pd.concat([train_X, synth_X], axis=0, ignore_index=True)
    preprocessor = _build_preprocessor(combined_X)
    X_train = preprocessor.fit_transform(train_X)
    X_synth = preprocessor.transform(synth_X)
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_synth, "toarray"):
        X_synth = X_synth.toarray()
    y_train = dataset.train[label_col].astype(int).to_numpy()
    classes, counts = np.unique(y_train, return_counts=True)
    majority_label = classes[np.argmax(counts)]
    real_majority_rate = float(counts.max() / counts.sum())
    knn = KNeighborsClassifier(n_neighbors=max(1, min(5, len(y_train))))
    knn.fit(np.asarray(X_train), y_train)
    y_synth_check = knn.predict(np.asarray(X_synth))
    synth_majority_rate = float(np.mean(y_synth_check == majority_label))
    class_ok = abs(synth_majority_rate - real_majority_rate) <= float(threshold)
    return class_ok, synth_majority_rate, real_majority_rate


def _synthesize_with_class_guard(
    dataset: DatasetSplit,
    base_seed: int,
    generator_fn,
    max_attempts: int = 5,
    threshold: float = 0.15,
) -> tuple[pd.DataFrame, dict]:
    last_df: pd.DataFrame | None = None
    last_meta: dict[str, Any] = {}
    for attempt in range(max_attempts):
        attempt_seed = int(base_seed + attempt * 1000)
        _set_seed(attempt_seed)
        synth_df, metadata = generator_fn(attempt_seed)
        class_ok, synth_rate, real_rate = _class_guard_summary(
            dataset,
            synth_df,
            threshold=threshold,
        )
        metadata = dict(metadata)
        metadata.update(
            {
                "class_guard_attempt": attempt + 1,
                "class_guard_seed": attempt_seed,
                "class_guard_synth_rate": synth_rate,
                "class_guard_real_rate": real_rate,
                "class_guard_ok": class_ok,
            }
        )
        if class_ok:
            print(
                f"[synthesis] seed={attempt_seed} attempt={attempt+1} "
                f"class_ok: synth_rate={synth_rate:.3f} real_rate={real_rate:.3f}"
            )
            return synth_df, metadata
        print(
            f"[synthesis] seed={attempt_seed} attempt={attempt+1} "
            f"INVERTED: synth_rate={synth_rate:.3f} real_rate={real_rate:.3f}. "
            "Resampling..."
        )
        last_df = synth_df
        last_meta = metadata
    print(
        f"[synthesis] WARNING: Could not fix inversion after {max_attempts} attempts"
    )
    if last_df is None:
        raise BaselineUnavailable("Class-guard synthesis produced no samples.")
    return last_df, last_meta


def _sdv_synthesize(
    model: Any, real_df: pd.DataFrame, seed: int, n_samples: int | None = None
) -> pd.DataFrame:
    np.random.seed(seed)
    if hasattr(model, "set_random_state"):
        model.set_random_state(seed)
    model.fit(real_df)
    count = n_samples or len(real_df)
    return model.sample(count)


def _prepare_sdv_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for col in prepared.columns:
        if str(prepared[col].dtype) == "category":
            prepared[col] = prepared[col].astype(object)
    return prepared


def drop_constant_columns(
    df: pd.DataFrame, exclude: Iterable[str] | None = None
) -> tuple[pd.DataFrame, dict[str, object]]:
    constants: dict[str, object] = {}
    drop_cols: list[str] = []
    exclude_set = set(exclude or [])
    for col in df.columns:
        if col in exclude_set:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.dropna().empty:
                drop_cols.append(col)
                constants[col] = 0.0
                continue
            variance = float(numeric.var())
            value_range = float(numeric.max() - numeric.min())
            if variance <= 1e-12 or value_range <= 1e-8:
                value = numeric.dropna().iloc[0] if numeric.notna().any() else 0.0
                constants[col] = float(value)
                drop_cols.append(col)
        else:
            if series.nunique(dropna=False) <= 1:
                value = series.iloc[0] if len(series) else ""
                constants[col] = value if pd.notnull(value) else ""
                drop_cols.append(col)
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logging.info("Dropped constant/near-constant columns: %s", drop_cols)
    return df, constants


def _build_sdv_metadata(df: pd.DataFrame):
    try:
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        return metadata
    except Exception:
        return None


def _generate_tabddpm(
    dataset: DatasetSplit,
    params: dict,
    seed: int,
    n_samples: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    sample_count = n_samples or len(dataset.train)
    label_col = dataset.label_col
    full_df = dataset.train.copy()
    labels = full_df[label_col].to_numpy()
    features = full_df.drop(columns=[label_col])
    encoded_features, mappings = encode_categorical(features)

    label_sampling = str(params.get("label_sampling", "train")).lower()
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        raise BaselineUnavailable("Training data has <2 label classes; cannot run class-conditional DDPM.")
    if label_sampling == "balanced":
        probs = np.full(len(unique_labels), 1.0 / len(unique_labels))
    else:
        probs = label_counts / label_counts.sum()

    rng = np.random.default_rng(seed)
    y_syn = rng.choice(unique_labels, size=sample_count, p=probs)
    attempts = 0
    while len(np.unique(y_syn)) < 2 and attempts < 5:
        y_syn = rng.choice(unique_labels, size=sample_count, p=probs)
        attempts += 1
    if len(np.unique(y_syn)) < 2:
        raise BaselineUnavailable("Synthetic label sampling collapsed to a single class.")

    X_syn_parts: list[np.ndarray] = []
    y_syn_parts: list[np.ndarray] = []
    for label in unique_labels:
        n_cls = int(np.sum(y_syn == label))
        if n_cls == 0:
            continue
        mask = labels == label
        X_cls = encoded_features.loc[mask].to_numpy(dtype=float)
        if X_cls.shape[0] < 2:
            raise BaselineUnavailable(
                f"Class {label} has too few samples ({X_cls.shape[0]}) for conditional training."
            )
        class_params = dict(params)
        class_params["include_label"] = False
        class_params["agent_id"] = f"tabddpm_single_seed{seed}_class{label}"
        generator = TabDDPMGenerator(class_params)
        generator.fit(X_cls, np.full(X_cls.shape[0], label))
        X_cls_syn, _ = generator.generate(n_cls)
        X_syn_parts.append(X_cls_syn)
        y_syn_parts.append(np.full(n_cls, label))

    if not X_syn_parts:
        raise BaselineUnavailable("Class-conditional generation produced zero samples.")
    X_syn = np.vstack(X_syn_parts)
    y_syn = np.concatenate(y_syn_parts)
    perm = rng.permutation(len(y_syn))
    X_syn = X_syn[perm]
    y_syn = y_syn[perm]

    if len(np.unique(y_syn)) < 2:
        raise BaselineUnavailable("Synthetic labels collapsed to a single class.")

    synth_df = pd.DataFrame(X_syn, columns=encoded_features.columns)
    synth_df = decode_categorical(synth_df, mappings)
    synth_df[label_col] = y_syn
    synth_df = synth_df[dataset.train.columns]
    cont_scale = params.get("cont_scale", 1.0)
    if cont_scale is not None and cont_scale != 1.0:
        num_cols = dataset.train.drop(columns=[label_col]).select_dtypes(include=[np.number]).columns
        if len(num_cols):
            means = dataset.train[num_cols].mean()
            synth_df[num_cols] = means + cont_scale * (synth_df[num_cols] - means)
    synth_df = sanitize_synthetic(synth_df, dataset.train, label_col)
    return synth_df, {
        "timestep": params.get("timesteps"),
        "n_samples": sample_count,
        "seed": seed,
        "include_label": False,
        "label_sampling": label_sampling,
    }


def _madifft_metadata(dataset: DatasetSplit, synth_df: pd.DataFrame, base_meta: dict) -> dict:
    label_col = dataset.label_col
    combined = pd.concat([synth_df, dataset.test], axis=0, ignore_index=True)
    encoded = pd.get_dummies(combined.drop(columns=[label_col]), drop_first=False).fillna(0)
    syn_encoded = encoded.iloc[: len(synth_df)]
    test_encoded = encoded.iloc[len(synth_df) :]
    clf = LogisticRegression(max_iter=500)
    clf.fit(syn_encoded, synth_df[label_col])
    acc = clf.score(test_encoded, dataset.test[label_col])
    real_numeric = dataset.train.drop(columns=[label_col]).select_dtypes(include=[np.number])
    synth_numeric = synth_df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    if real_numeric.shape[1] == 0 or synth_numeric.shape[1] == 0:
        fidelity_distance = 0.0
        privacy_score = 0.0
    else:
        real_means = real_numeric.mean()
        synth_means = synth_numeric.mean()
        fidelity_distance = float(np.abs(real_means - synth_means).mean())
        privacy_score = float(np.linalg.norm(real_means - synth_means))
    fitness_utility = float(acc)
    fitness_privacy = 1.0 / (1.0 + privacy_score)
    fitness_fidelity = 1.0 / (1.0 + fidelity_distance)
    return {
        "best_agent_id": f"madifft_agent_seed_{dataset.name}_{base_meta.get('seed', '0')}",
        "generation": "MADiff-T",
        "T": base_meta.get("timestep"),
        "schedule": base_meta.get("schedule", "cosine"),
        "beta_scale": base_meta.get("beta_scale", 1.0),
        "dropout": base_meta.get("dropout", 0.0),
        "cont_scale": base_meta.get("cont_scale", 1.0),
        "fitness_total": fitness_utility + fitness_privacy + fitness_fidelity,
        "fitness_utility": fitness_utility,
        "fitness_privacy": fitness_privacy,
        "fitness_fidelity": fitness_fidelity,
        "notes": "MADiff-T surrogate derived from TabDDPM",
    }


def _sdv_method(
    model_cls: Any,
    dataset: DatasetSplit,
    seed: int,
    method_cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    n_samples = method_cfg.get("n_samples")
    kwargs = method_cfg.get("model_kwargs", {})
    prepared = _prepare_sdv_frame(dataset.train)
    metadata = _build_sdv_metadata(prepared)
    try:
        if metadata is not None:
            model = model_cls(metadata=metadata, **kwargs)
        else:
            model = model_cls(**kwargs)
    except TypeError:
        model = model_cls(**kwargs)
    df = _sdv_synthesize(model, prepared, seed, n_samples=n_samples)
    return df, {"notes": f"SDV {model_cls.__name__} (seed={seed})"}


def _load_privbayes_cls():
    try:
        from dpart.engines import PrivBayes as DpartPrivBayes

        return DpartPrivBayes
    except ImportError:
        try:
            from dpart.engines.privbayes import PrivBayes as DpartPrivBayes

            return DpartPrivBayes
        except ImportError:
            return None


def _privbayes_synthesize(
    dataset: DatasetSplit,
    seed: int,
    method_cfg: dict,
    epsilon_override: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    model_cls = _load_privbayes_cls()
    if model_cls is None:
        raise BaselineUnavailable("dpart PrivBayes is not installed.")
    try:
        import dpart.dpart as dpart_module
        import dpart.methods.utils.bin_encoder as bin_encoder
        import sklearn.preprocessing as sk_pre
        import sklearn.preprocessing._data as sk_data
        from sklearn.preprocessing import KBinsDiscretizer as SKKBinsDiscretizer
        from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler

        if not hasattr(SKMinMaxScaler, "_compat_wrapped"):
            orig_init = SKMinMaxScaler.__init__

            def _compat_init(self, feature_range=(0, 1), *, copy=True, clip=False):
                if isinstance(feature_range, list):
                    feature_range = tuple(feature_range)
                return orig_init(self, feature_range=feature_range, copy=copy, clip=clip)

            SKMinMaxScaler.__init__ = _compat_init  # type: ignore[assignment]
            SKMinMaxScaler._compat_wrapped = True  # type: ignore[attr-defined]

        class _CompatKBinsDiscretizer(SKKBinsDiscretizer):
            def __init__(
                self,
                n_bins=5,
                *,
                encode="onehot",
                strategy="quantile",
                quantile_method="linear",
                dtype=None,
                subsample=200000,
                random_state=None,
            ):
                super().__init__(
                    n_bins=n_bins,
                    encode=encode,
                    strategy=strategy,
                    quantile_method=quantile_method,
                    dtype=dtype,
                    subsample=subsample,
                    random_state=random_state,
                )

        class _CompatBinEncoder(bin_encoder.BinEncoder):
            def __init__(self, n_bins=5):
                super().__init__(n_bins=n_bins)
                self.n_bins = n_bins
                self._uses_bins = False

            def fit(self, data: pd.Series):
                self.dkind = data.dtype.kind
                if self.dkind in "OSb":
                    self.encoder = bin_encoder.OrdinalEncoder()
                    self._uses_bins = False
                else:
                    numeric = pd.to_numeric(data, errors="coerce")
                    if numeric.dropna().empty:
                        self.encoder = bin_encoder.OrdinalEncoder()
                        self._uses_bins = False
                        self.encoder.fit(data.to_frame())
                        return
                    variance = float(numeric.var())
                    value_range = float(numeric.max() - numeric.min())
                    if variance <= 1e-12 or value_range <= 1e-8:
                        self.encoder = bin_encoder.OrdinalEncoder()
                        self._uses_bins = False
                    else:
                        unique_count = int(numeric.nunique(dropna=True))
                        if unique_count < 4:
                            self.encoder = bin_encoder.OrdinalEncoder()
                            self._uses_bins = False
                            self.encoder.fit(data.to_frame())
                            return
                        n_bins = max(2, min(self.n_bins, unique_count - 1))
                        self.encoder = _CompatKBinsDiscretizer(
                            n_bins=n_bins,
                            encode="ordinal",
                            strategy="quantile",
                            quantile_method="linear",
                            subsample=None,
                        )
                        self._uses_bins = True
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Bins whose width are too small.*",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Feature .* is constant and will be replaced with 0.*",
                        category=UserWarning,
                    )
                    self.encoder.fit(data.to_frame())

            def inverse_transform(self, data: pd.Series) -> pd.Series:
                if not self._uses_bins or not hasattr(self.encoder, "bin_edges_"):
                    samples = self.encoder.inverse_transform(data.to_frame()).squeeze()
                else:
                    bin_edges = self.encoder.bin_edges_[0]
                    bin_idx = pd.to_numeric(data, errors="coerce").fillna(0).to_numpy().astype(int)
                    bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)
                    low = bin_edges[bin_idx]
                    high = bin_edges[bin_idx + 1]
                    low = np.nan_to_num(low, nan=0.0, posinf=0.0, neginf=0.0)
                    high = np.nan_to_num(high, nan=0.0, posinf=0.0, neginf=0.0)
                    invalid = high <= low
                    if np.any(invalid):
                        high = high.copy()
                        high[invalid] = low[invalid] + 1.0
                    samples = np.random.uniform(low, high)

                return pd.Series(samples, index=data.index, name=data.name)

        dpart_module.MinMaxScaler = SKMinMaxScaler
        sk_pre.MinMaxScaler = SKMinMaxScaler
        sk_data.MinMaxScaler = SKMinMaxScaler
        bin_encoder.KBinsDiscretizer = _CompatKBinsDiscretizer
        bin_encoder.BinEncoder = _CompatBinEncoder
        try:
            import dpart.methods.probability_tensor as probability_tensor

            probability_tensor.BinEncoder = _CompatBinEncoder
        except Exception:
            pass
    except Exception:
        pass
    kwargs = method_cfg.get("model_kwargs", {})
    prepared = _prepare_sdv_frame(dataset.train)
    prepared, constant_cols = drop_constant_columns(prepared, exclude=[dataset.label_col])
    if constant_cols:
        logging.info("PrivBayes: dropped constant columns before discretization: %s", list(constant_cols.keys()))
    label_col = dataset.label_col

    cont_cols = prepared.select_dtypes(include=[np.number]).columns.tolist()
    cont_cols = [col for col in cont_cols if col != label_col]
    requested_bins = method_cfg.get("n_bins", kwargs.pop("n_bins", 5))
    if not isinstance(requested_bins, int) or requested_bins <= 1:
        requested_bins = 5

    discretized, bin_info = discretize_continuous(prepared, cont_cols, requested_bins)
    if label_col in discretized.columns:
        discretized[label_col] = discretized[label_col].astype("category")

    bounds: dict = {}
    categorical_domains: dict[str, list] = {}
    for col in discretized.columns:
        series = discretized[col]
        if pd.api.types.is_numeric_dtype(series) and col in cont_cols:
            min_val = float(series.min())
            max_val = float(series.max())
            if min_val >= max_val:
                max_val = min_val + 1.0
            bounds[col] = {"min": min_val, "max": max_val}
        else:
            categories = sorted(series.dropna().unique(), key=str)
            bounds[col] = {"categories": categories}
            categorical_domains[col] = categories

    epsilon = epsilon_override if epsilon_override is not None else method_cfg.get("epsilon")
    model = model_cls(bounds=bounds, n_bins=requested_bins, epsilon=epsilon, **kwargs)
    if hasattr(model, "bounds"):
        model.bounds = bounds
    if hasattr(model, "__dict__"):
        model.categorical_domains = categorical_domains
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"target .* has no specified method will use default ProbabilityTensor",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Bins whose width are too small.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Feature .* is constant and will be replaced with 0.*",
                category=UserWarning,
            )
            model.fit(discretized)
    except TypeError as exc:
        raise BaselineUnavailable(f"PrivBayes fit failed: {exc}") from exc
    n_samples = method_cfg.get("n_samples") or len(dataset.train)
    if hasattr(model, "sample"):
        synth = model.sample(n_samples)
    elif hasattr(model, "generate"):
        synth = model.generate(n_samples)
    else:
        raise BaselineUnavailable("PrivBayes model does not expose sample/generate.")
    if isinstance(synth, pd.DataFrame):
        df = synth.copy()
    else:
        df = pd.DataFrame(synth, columns=dataset.train.columns)
    if constant_cols:
        for col, value in constant_cols.items():
            df[col] = value
    df = df.reindex(columns=dataset.train.columns)
    df = inverse_discretize(df, bin_info)
    df = sanitize_synthetic(df, dataset.train, dataset.label_col)
    return df, {
        "notes": f"PrivBayes (dpart, seed={seed}, eps={epsilon}, bins={requested_bins})",
        "epsilon": epsilon,
        "n_bins": requested_bins,
    }


def _load_external_synth(
    method: str,
    dataset: DatasetSplit,
    seed: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict]:
    external = config.get("external_outputs", {})
    root_key = f"{method}_root"
    root = external.get(root_key)
    if not root:
        raise BaselineUnavailable(f"Missing external_outputs.{root_key} in config.")
    file_names = external.get("file_names", ["synth.csv", "samples.csv", "synthetic.csv"])
    dataset_name = dataset.name
    candidates = []
    for fname in file_names:
        candidates.append(Path(root) / dataset_name / method / str(seed) / fname)
        candidates.append(Path(root) / dataset_name / method / f"seed{seed}" / fname)
        candidates.append(Path(root) / dataset_name / f"seed{seed}" / fname)
        candidates.append(Path(root) / dataset_name / fname)
    existing = next((p for p in candidates if p.exists()), None)
    if existing is None:
        raise BaselineUnavailable(f"No external CSV found under {root} for {dataset_name}/seed{seed}.")
    df = pd.read_csv(existing)
    expected_cols = list(dataset.train.columns)
    if list(df.columns) != expected_cols:
        if set(df.columns) == set(expected_cols):
            df = df.reindex(columns=expected_cols)
        else:
            raise BaselineUnavailable(
                f"External CSV columns mismatch. Expected {expected_cols}, got {list(df.columns)}."
            )
    return df, {"notes": f"Loaded external CSV: {existing}"}


def generate_synth(
    dataset: DatasetSplit,
    method: str,
    seed: int,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict]:
    _set_seed(seed)
    cfg = config or {}
    lower = method.lower()
    if lower.startswith("privbayes"):
        method_cfg = cfg.get("method_params", {}).get("privbayes", {})
    else:
        method_cfg = cfg.get("method_params", {}).get(lower, {})
    metadata: dict[str, Any] = {"method": method, "seed": seed}
    try:
        if lower in {"gaussian_copula", "ctgan", "tvae"}:
            models = {
                "gaussian_copula": GaussianCopulaSynthesizer,
                "ctgan": CTGANSynthesizer,
                "tvae": TVAESynthesizer,
            }
            df, md = _sdv_method(models[lower], dataset, seed, method_cfg)
            metadata.update(md)
            return df, metadata

        if lower.startswith("privbayes"):
            eps = parse_privbayes_method_name(lower)
            df, md = _privbayes_synthesize(dataset, seed, method_cfg, epsilon_override=eps)
            metadata.update(md)
            return df, metadata

        if lower in {"tabddpm_single", "tabddpm"}:
            params = method_cfg.get("params", {})
            df, meta = _synthesize_with_class_guard(
                dataset,
                seed,
                lambda attempt_seed: _generate_tabddpm(
                    dataset,
                    params,
                    attempt_seed,
                    n_samples=method_cfg.get("n_samples"),
                ),
            )
            metadata.update(meta)
            metadata["notes"] = "TabDDPM single agent"
            return df, metadata

        if lower == "madifft":
            run_dir = cfg.get("_run_dir")
            synth_df, evo_meta = _synthesize_with_class_guard(
                dataset,
                seed,
                lambda attempt_seed: run_madifft_evolution(
                    dataset,
                    attempt_seed,
                    method_cfg,
                    run_dir=run_dir,
                ),
            )
            metadata.update(evo_meta)
            return synth_df, metadata

        if lower == "tabsyn":
            df, md = _load_external_synth("tabsyn", dataset, seed, cfg)
            metadata.update(md)
            return df, metadata

        if lower == "stasy":
            df, md = _load_external_synth("stasy", dataset, seed, cfg)
            metadata.update(md)
            return df, metadata

    except Exception as exc:
        raise BaselineUnavailable(str(exc)) from exc
    raise BaselineUnavailable(f"Unknown method: {method}")
