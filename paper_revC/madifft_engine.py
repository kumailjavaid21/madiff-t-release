from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.mas_generators.tabddpm.generator import TabDDPMGenerator
from src.mas_generators.tabddpm.losses import (
    compute_adaptive_epochs,
    compute_auto_lambda_cat,
    compute_feature_weights,
)
from src.fitness import (
    compute_fitness_per_generation,
    compute_mia_advantage,
    compute_mia_auc_distance,
    compute_normdcr_q,
)
from src.selection import compute_utility_floor, select_with_privacy_gate
from .datasets import DatasetSplit
from .tstr_eval import tstr_eval
from .tabular_utils import encode_categorical, decode_categorical, sanitize_synthetic


@dataclass
class _Agent:
    agent_id: str
    params: dict[str, Any]
    generators: dict[object, TabDDPMGenerator]
    fitness: float = 0.0
    utility: float = 0.0
    privacy: float = 0.0
    fidelity: float = 0.0
    class_kl: float = 0.0


def ensure_per_feature_params(h: dict[str, Any], d_c: int) -> dict[str, Any]:
    if "feature_alpha" not in h:
        h["feature_alpha"] = np.zeros(d_c, dtype=np.float32)
    if "feature_gamma" not in h:
        h["feature_gamma"] = np.zeros(d_c, dtype=np.float32)
    h["feature_alpha"] = np.asarray(h["feature_alpha"], dtype=np.float32).copy()
    h["feature_gamma"] = np.asarray(h["feature_gamma"], dtype=np.float32).copy()
    return h


def _mutate_params(
    base_params: dict[str, Any],
    rng: np.random.Generator,
    mutation_rate: float,
    d_c: int,
    mutate_feature_schedule: bool = True,
    pfs_active: bool = True,
) -> dict[str, Any]:
    params = ensure_per_feature_params(copy.deepcopy(base_params), d_c)
    if rng.random() < mutation_rate:
        params["hidden_dim"] = max(64, int(params.get("hidden_dim", 128) * rng.uniform(0.7, 1.3)))
    if rng.random() < mutation_rate:
        params["num_layers"] = int(np.clip(params.get("num_layers", 3) + rng.integers(-1, 2), 2, 6))
    if rng.random() < mutation_rate:
        params["time_embed_dim"] = max(
            32, int(params.get("time_embed_dim", 64) * rng.uniform(0.7, 1.3))
        )
    if rng.random() < mutation_rate:
        params["timesteps"] = int(np.clip(params.get("timesteps", 300) + rng.integers(-50, 51), 50, 1000))
    if rng.random() < mutation_rate:
        params["lr"] = float(params.get("lr", 1e-3) * (10 ** rng.uniform(-0.3, 0.3)))
    if rng.random() < mutation_rate:
        params["batch_size"] = int(
            np.clip(params.get("batch_size", 128) + rng.integers(-32, 33), 32, 512)
        )
    if pfs_active and mutate_feature_schedule and d_c > 0 and rng.random() < mutation_rate:
        mask = (rng.random(d_c) < 0.3).astype(np.float32)
        noise = rng.normal(0.0, 0.02, d_c).astype(np.float32)
        params["feature_alpha"] = np.clip(
            np.asarray(params["feature_alpha"], dtype=np.float32) + mask * noise,
            -0.15,
            0.15,
        ).astype(np.float32)
    if pfs_active and mutate_feature_schedule and d_c > 0 and rng.random() < mutation_rate:
        mask = (rng.random(d_c) < 0.3).astype(np.float32)
        noise = rng.normal(0.0, 0.002, d_c).astype(np.float32)
        params["feature_gamma"] = np.clip(
            np.asarray(params["feature_gamma"], dtype=np.float32) + mask * noise,
            -0.01,
            0.01,
        ).astype(np.float32)
    return params


def run_madifft_evolution(
    dataset: DatasetSplit,
    seed: int,
    method_cfg: dict,
    run_dir: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    evo_cfg = method_cfg.get("evolution", {})
    ablation_cfg = method_cfg.get("ablation", {})
    use_fa_loss = bool(ablation_cfg.get("use_fa_loss", True))
    enable_pfs = bool(ablation_cfg.get("enable_pfs", True))

    population = int(evo_cfg.get("population", 4))
    generations = int(evo_cfg.get("generations", 3))
    steps_per_gen = int(evo_cfg.get("steps_per_gen", 5))
    elite_k = int(evo_cfg.get("elite_k", max(1, population // 2)))
    mutation_rate = float(evo_cfg.get("mutation_rate", 0.4))

    w_util = float(evo_cfg.get("w_utility", 1.0))
    w_priv = float(evo_cfg.get("w_privacy", 1.0))
    w_fid = float(evo_cfg.get("w_fidelity", 1.0))
    w_kl = float(evo_cfg.get("w_class_kl", 0.0))

    full_df = dataset.train.copy()
    label_col = dataset.label_col
    labels = full_df[label_col].to_numpy()
    features = full_df.drop(columns=[label_col])
    encoded_features, mappings = encode_categorical(features)
    X_train = encoded_features.to_numpy(dtype=float)
    feature_cols = list(encoded_features.columns)
    cat_cols = [col for col in feature_cols if col in mappings]
    cont_cols = [col for col in feature_cols if col not in mappings]
    continuous_idx = [feature_cols.index(col) for col in cont_cols]
    categorical_idx = [feature_cols.index(col) for col in cat_cols]
    categorical_cardinalities = [len(mappings[col]) for col in cat_cols]
    n_cont = len(continuous_idx)
    n_cat = len(categorical_idx)
    d_c = n_cont
    d_cat = n_cat
    n_train = int(len(labels))
    cat_ratio = float(d_cat) / float(d_c + d_cat) if (d_c + d_cat) > 0 else 0.0
    pfs_active = enable_pfs and (cat_ratio < 0.5) and (n_train >= 520)
    auto_lambda_cat = compute_auto_lambda_cat(
        n_cont=n_cont,
        n_cat=n_cat,
    )
    _, label_counts = np.unique(labels, return_counts=True)
    majority_rate = float(label_counts.max() / label_counts.sum())
    utility_floor = compute_utility_floor(
        y_train=labels,
        base_floor=0.60,
        margin=0.04,
        max_floor=0.68,
    )
    adaptive_epochs = compute_adaptive_epochs(
        n_cont=n_cont,
        n_cat=n_cat,
        base_epochs=50,
    )
    print(
        f"[MADiff-T] dataset={dataset.name} seed={seed} "
        f"auto_lambda_cat={auto_lambda_cat:.3f} "
        f"(n_cont={n_cont}, n_cat={n_cat})"
    )
    print(
        f"[MADiff-T] dataset={dataset.name} seed={seed} "
        f"auto_epochs={adaptive_epochs} "
        f"(n_cont={n_cont}, n_cat={n_cat})"
    )
    print(
        f"[MADiff-T] dataset={dataset.name} seed={seed} "
        f"utility_floor={utility_floor:.4f} "
        f"(majority_rate={majority_rate:.4f})"
    )
    print(
        f"[MADiff-T] dataset={dataset.name} seed={seed} "
        f"w_class_kl={w_kl:.3f}"
    )
    print(
        f"[PFS] dataset={dataset.name} cat_ratio={cat_ratio:.2f} "
        f"n_train={n_train} pfs_active={pfs_active}"
    )

    X_train_cont = (
        encoded_features[cont_cols].to_numpy(dtype=float)
        if cont_cols
        else np.zeros((len(encoded_features), 0), dtype=float)
    )
    X_train_cat = (
        encoded_features[cat_cols].to_numpy(dtype=int)
        if cat_cols
        else np.zeros((len(encoded_features), 0), dtype=int)
    )
    # Deterministic on D_train; shared for all agents on this dataset.
    weights_cont = None
    weights_cat = None
    if use_fa_loss:
        weights_cont, weights_cat = compute_feature_weights(X_train_cont, X_train_cat)
    population_feature_state = {
        "feature_weights_cont": (
            weights_cont.detach().cpu().tolist() if weights_cont is not None else None
        ),
        "feature_weights_cat": (
            weights_cat.detach().cpu().tolist() if weights_cat is not None else None
        ),
        "continuous_idx": continuous_idx,
        "categorical_idx": categorical_idx,
        "categorical_cardinalities": categorical_cardinalities,
    }

    base_params = dict(method_cfg.get("params", {}))
    base_params["seed"] = seed
    print(f"[MADiff-T] include_label={base_params.get('include_label')}")
    base_params["lambda_cat"] = auto_lambda_cat if use_fa_loss else float(
        base_params.get("lambda_cat", 0.5)
    )
    base_params["epochs"] = adaptive_epochs
    base_params.update(population_feature_state)
    base_params = ensure_per_feature_params(base_params, d_c)
    label_sampling = str(method_cfg.get("label_sampling", "train")).lower()
    print(
        f"[FA] dataset={dataset.name} seed={seed} "
        f"use_fa_loss={use_fa_loss} lambda_cat={base_params['lambda_cat']:.3f}"
    )

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        raise ValueError("MADiff-T requires at least two label classes in training data.")
    if label_sampling == "balanced":
        label_probs = np.full(len(unique_labels), 1.0 / len(unique_labels))
    else:
        label_probs = label_counts / label_counts.sum()
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    X_by_label: dict[object, np.ndarray] = {
        label: X_train[labels == label] for label in unique_labels
    }

    def _apply_per_feature_params(agent: _Agent, gen_idx: int) -> None:
        agent.params = ensure_per_feature_params(agent.params, d_c)
        alpha = np.asarray(agent.params["feature_alpha"], dtype=np.float32)
        gamma = np.asarray(agent.params["feature_gamma"], dtype=np.float32)
        alpha_range = float(alpha.max() - alpha.min()) if alpha.size else 0.0
        gamma_range = float(gamma.max() - gamma.min()) if gamma.size else 0.0
        print(
            f"[PFS] gen={gen_idx} agent={agent.agent_id} "
            f"alpha_range={alpha_range:.4f} gamma_range={gamma_range:.6f} "
            f"d_c={d_c}"
        )
        for generator in agent.generators.values():
            generator.set_per_feature_schedule(alpha, gamma, agent_id=agent.agent_id)

    def _decode(X_syn: np.ndarray, y_syn: np.ndarray) -> pd.DataFrame:
        feature_cols = [col for col in full_df.columns if col != label_col]
        encoded_df = pd.DataFrame(X_syn, columns=feature_cols)
        encoded_df[label_col] = y_syn
        encoded_df = encoded_df[full_df.columns]
        decoded = decode_categorical(encoded_df, mappings)
        decoded = sanitize_synthetic(decoded, full_df, label_col)
        return decoded

    def _evaluate(synth_df: pd.DataFrame) -> dict[str, float]:
        y_syn = synth_df[label_col].to_numpy()
        if len(np.unique(y_syn)) < 2:
            return {
                "utility": 0.0,
                "fidelity": 0.0,
                "normdcr": float("nan"),
                "mia_auc": float("nan"),
                "mia_adv": float("nan"),
                "class_kl": float("inf"),
            }
        tstr = tstr_eval(dataset.val, synth_df, label_col, "logistic_regression", seed)
        util = tstr.get("roc_auc")
        if util is None or (isinstance(util, float) and np.isnan(util)):
            util = tstr.get("accuracy") or 0.0
        normdcr = compute_normdcr_q(
            synth_df.drop(columns=[label_col]),
            dataset.train.drop(columns=[label_col]),
            q=float(evo_cfg.get("privacy_q", 0.01)),
            alpha=float(evo_cfg.get("privacy_alpha", 0.5)),
            seed=seed,
        )
        mia_auc = compute_mia_auc_distance(
            dataset.train.drop(columns=[label_col]),
            dataset.val.drop(columns=[label_col]),
            synth_df.drop(columns=[label_col]),
            alpha=float(evo_cfg.get("privacy_alpha", 0.5)),
            seed=seed,
        )
        mia_adv = compute_mia_advantage(mia_auc)

        num_cols = full_df.drop(columns=[label_col]).select_dtypes(include=[np.number]).columns
        if len(num_cols):
            fid_dist = float(np.abs(full_df[num_cols].mean() - synth_df[num_cols].mean()).mean())
            fidelity_score = 1.0 / (1.0 + fid_dist)
        else:
            fidelity_score = 0.0

        syn_counts = np.array(
            [np.sum(y_syn == label) for label in unique_labels], dtype=float
        )
        syn_probs = syn_counts / max(syn_counts.sum(), 1.0)
        real_probs = label_counts / label_counts.sum()
        eps = 1e-8
        class_kl = float(np.sum(syn_probs * np.log((syn_probs + eps) / (real_probs + eps))))

        return {
            "utility": float(util),
            "fidelity": float(fidelity_score),
            "normdcr": float(normdcr) if normdcr is not None else float("nan"),
            "mia_auc": float(mia_auc) if mia_auc is not None else float("nan"),
            "mia_adv": float(mia_adv) if mia_adv is not None else float("nan"),
            "class_kl": float(class_kl),
        }

    def _sample_counts(rng: np.random.Generator, total: int) -> dict[object, int]:
        sampled = rng.choice(unique_labels, size=total, p=label_probs)
        attempts = 0
        while len(np.unique(sampled)) < 2 and attempts < 5:
            sampled = rng.choice(unique_labels, size=total, p=label_probs)
            attempts += 1
        counts = {label: int(np.sum(sampled == label)) for label in unique_labels}
        if len([c for c in counts.values() if c > 0]) < 2:
            # force at least one sample per class
            counts = {label: max(1, count) for label, count in counts.items()}
            overflow = sum(counts.values()) - total
            if overflow > 0:
                # subtract overflow from the majority class
                max_label = max(counts, key=counts.get)
                counts[max_label] = max(1, counts[max_label] - overflow)
        return counts

    def check_class_distribution(y_synth_raw, y_train, threshold=0.15):
        """
        Guard against class-inverted synthetic data.
        y_synth_raw may be continuous floats from diffusion output.
        We round to nearest integer before comparing distributions.
        """
        if y_synth_raw is None or len(y_synth_raw) == 0:
            return True
        y_synth_int = np.round(np.array(y_synth_raw)).astype(int)
        y_train_arr = np.array(y_train).astype(int)

        classes, real_counts = np.unique(y_train_arr, return_counts=True)
        real_rates = real_counts / real_counts.sum()
        synth_rates = {int(cls): float(np.mean(y_synth_int == cls)) for cls in classes}
        real_rates_map = {int(cls): float(rate) for cls, rate in zip(classes, real_rates)}

        ok = True
        for cls, real_rate in zip(classes, real_rates):
            synth_rate = np.mean(y_synth_int == cls)
            if abs(synth_rate - real_rate) > threshold:
                ok = False
                break
        if not ok:
            print(
                f"[MADiff-T] Class guard: synth_rates={synth_rates} "
                f"real_rates={real_rates_map}"
            )
        return ok

    def _synthesize(agent: _Agent, total: int) -> tuple[np.ndarray, np.ndarray]:
        counts = _sample_counts(rng, total)
        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        include_label_mode = bool(agent.params.get("include_label", False))
        for label in unique_labels:
            n_cls = counts.get(label, 0)
            if n_cls <= 0:
                continue
            gen = agent.generators[label]
            X_cls_syn, y_cls_syn = gen.generate(n_cls)
            # Some backends emit joint samples [X, y] when include_label is enabled.
            if include_label_mode and X_cls_syn.shape[1] == (X_train.shape[1] + 1):
                y_cls_syn = X_cls_syn[:, -1]
                X_cls_syn = X_cls_syn[:, :-1]
            if X_cls_syn.shape[1] != X_train.shape[1]:
                raise ValueError(
                    "Synthetic feature shape mismatch: "
                    f"expected {X_train.shape[1]} features, got {X_cls_syn.shape[1]} "
                    f"(include_label={include_label_mode})"
                )
            X_parts.append(X_cls_syn)
            if include_label_mode:
                y_raw = np.asarray(y_cls_syn).reshape(-1)
                y_idx = np.rint(y_raw).astype(int)
                y_idx = np.clip(y_idx, 0, len(unique_labels) - 1)
                y_parts.append(np.asarray(unique_labels)[y_idx])
            else:
                y_parts.append(np.full(n_cls, label))
        if not X_parts:
            return np.empty((0, X_train.shape[1])), np.empty((0,))
        X_syn = np.vstack(X_parts)
        y_syn = np.concatenate(y_parts)
        perm = rng.permutation(len(y_syn))
        return X_syn[perm], y_syn[perm]

    agents: list[_Agent] = []
    for idx in range(population):
        params = _mutate_params(
            base_params,
            rng,
            mutation_rate,
            d_c,
            mutate_feature_schedule=False,
            pfs_active=pfs_active,
        )
        agent_id = f"g0_a{idx}"
        params["agent_id"] = agent_id
        generators = {label: TabDDPMGenerator(params) for label in unique_labels}
        agent = _Agent(agent_id=agent_id, params=params, generators=generators)
        agents.append(agent)

    logs: list[dict[str, Any]] = []
    gen_metrics_rows: list[dict[str, Any]] = []
    for gen_idx in range(generations):
        gen_metrics: list[dict[str, Any]] = []
        for agent in agents:
            _apply_per_feature_params(agent, gen_idx)
            guard_failed = False
            for label in unique_labels:
                X_cls = X_by_label[label]
                if X_cls.shape[0] < 2:
                    agent.fitness = float("-inf")
                    agent.utility = 0.0
                    agent.privacy = 0.0
                    agent.fidelity = 0.0
                    agent.class_kl = float("inf")
                    continue
                gen = agent.generators[label]
                gen.epochs = steps_per_gen
                if bool(agent.params.get("include_label", False)):
                    label_idx = float(label_to_index[label])
                    joint_train = np.column_stack([X_cls, np.full(X_cls.shape[0], label_idx)])
                    gen.fit(joint_train, np.full(X_cls.shape[0], label_idx))
                else:
                    gen.fit(X_cls, np.full(X_cls.shape[0], label))
            X_syn, y_syn = _synthesize(agent, len(full_df))
            if X_syn.shape[0] == 0:
                metrics = {
                    "utility": 0.0,
                    "fidelity": 0.0,
                    "normdcr": float("nan"),
                    "mia_auc": float("nan"),
                    "mia_adv": float("nan"),
                    "class_kl": float("inf"),
                }
                synth_df = full_df.copy()
            else:
                class_ok = check_class_distribution(y_syn, labels, threshold=0.15)
                if not class_ok:
                    print(
                        "[MADiff-T] Class guard triggered. Assigning fitness=0."
                    )
                    guard_failed = True
                    metrics = {
                        "utility": 0.0,
                        "fidelity": 0.0,
                        "normdcr": 0.0,
                        "mia_auc": 0.5,
                        "mia_adv": 0.0,
                        "class_kl": float("inf"),
                    }
                    gen_metrics.append(
                        {
                            "dataset": dataset.name,
                            "seed": seed,
                            "generation": gen_idx,
                            "agent_id": agent.agent_id,
                            "utility": metrics["utility"],
                            "fidelity": metrics["fidelity"],
                            "normdcr": metrics["normdcr"],
                            "mia_auc": metrics["mia_auc"],
                            "mia_adv": metrics["mia_adv"],
                            "class_kl": metrics["class_kl"],
                            "params": dict(agent.params),
                            "guard_failed": guard_failed,
                        }
                    )
                    continue
                synth_df = _decode(X_syn, y_syn)
                metrics = _evaluate(synth_df)
            gen_metrics.append(
                {
                    "dataset": dataset.name,
                    "seed": seed,
                    "generation": gen_idx,
                    "agent_id": agent.agent_id,
                    "utility": metrics["utility"],
                    "fidelity": metrics["fidelity"],
                    "normdcr": metrics["normdcr"],
                    "mia_auc": metrics["mia_auc"],
                    "mia_adv": metrics["mia_adv"],
                    "class_kl": metrics["class_kl"],
                    "params": dict(agent.params),
                    "guard_failed": guard_failed,
                }
            )

        gen_df = pd.DataFrame(gen_metrics)
        fitness_scores = compute_fitness_per_generation(
            gen_df,
            weights={"utility": w_util, "privacy": w_priv, "fidelity": w_fid},
        )
        if w_kl > 0.0 and "class_kl" in gen_df.columns:
            class_kl_penalty = pd.to_numeric(
                gen_df["class_kl"], errors="coerce"
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            class_kl_penalty = class_kl_penalty.clip(lower=0.0)
            fitness_scores = fitness_scores - (w_kl * class_kl_penalty)
        gen_df["fitness"] = fitness_scores
        if "guard_failed" in gen_df.columns:
            gen_df.loc[gen_df["guard_failed"].fillna(False), "fitness"] = 0.0
        class_kl_debug = ", ".join(
            f"{row.agent_id}:{float(row.class_kl):.4f}"
            for row in gen_df.itertuples(index=False)
        )
        print(
            f"[MADiff-T] dataset={dataset.name} seed={seed} "
            f"generation={gen_idx} class_kl={{ {class_kl_debug} }}"
        )
        for idx, agent in enumerate(agents):
            agent.fitness = float(gen_df.iloc[idx]["fitness"])
            agent.utility = float(gen_df.iloc[idx]["utility"])
            agent.privacy = float(gen_df.iloc[idx]["mia_adv"]) if not np.isnan(gen_df.iloc[idx]["mia_adv"]) else 0.0
            agent.fidelity = float(gen_df.iloc[idx]["fidelity"])
            agent.class_kl = float(gen_df.iloc[idx]["class_kl"])
        gen_metrics_rows.extend(gen_df.to_dict(orient="records"))
        logs.extend(gen_df.to_dict(orient="records"))

        agents.sort(key=lambda a: a.fitness, reverse=True)
        if gen_idx < generations - 1:
            elites = agents[:elite_k]
            new_agents: list[_Agent] = [*elites]

            while len(new_agents) < population:
                parent = rng.choice(elites)
                params = _mutate_params(
                    parent.params,
                    rng,
                    mutation_rate,
                    d_c,
                    pfs_active=pfs_active,
                )
                generators: dict[object, TabDDPMGenerator] = {}
                for label in unique_labels:
                    generator = TabDDPMGenerator(params)
                    # weight inheritance per class
                    try:
                        input_dim = X_train.shape[1] + (
                            1 if bool(params.get("include_label", False)) else 0
                        )
                        generator._build_model(input_dim)
                        generator.model.load_state_dict(
                            parent.generators[label].model.state_dict()
                        )
                    except Exception:
                        pass
                    generators[label] = generator
                new_agents.append(
                    _Agent(
                        agent_id=f"g{gen_idx+1}_a{len(new_agents)}",
                        params=params,
                        generators=generators,
                    )
                )
                new_agents[-1].params["agent_id"] = new_agents[-1].agent_id
            agents = new_agents

    best_agent = max(agents, key=lambda a: a.fitness)
    best_row = None
    selection_meta: dict[str, Any] = {}
    if gen_metrics_rows:
        metrics_df = pd.DataFrame(gen_metrics_rows)
        last_gen = metrics_df[metrics_df["generation"] == (generations - 1)]
        candidate_df = last_gen if not last_gen.empty else metrics_df
        candidate_df, selection = select_with_privacy_gate(
            candidate_df,
            utility_floor=utility_floor,
        )
        selection_meta = {
            "selected_by": selection.selected_by,
            "privacy_gate_count": selection.gate_count,
            "relaxed_gate_count": selection.relaxed_count,
            "total_candidates": selection.total_candidates,
            **selection.thresholds,
        }
        best_row = selection.selected_row
        best_agent_id = str(best_row.get("agent_id", best_agent.agent_id))
        selection_meta["selected_agent_id"] = best_agent_id
        match = next((agent for agent in agents if agent.agent_id == best_agent_id), None)
        if match is not None:
            best_agent = match
        else:
            best_row = metrics_df.sort_values("fitness", ascending=False).iloc[0]
    final_params = ensure_per_feature_params(copy.deepcopy(best_agent.params), d_c)
    final_params["epochs"] = int(method_cfg.get("full_epochs", final_params.get("epochs", 50)))
    final_generators = {label: TabDDPMGenerator(final_params) for label in unique_labels}
    for label in unique_labels:
        X_cls = X_by_label[label]
        if X_cls.shape[0] < 2:
            raise ValueError(f"Class {label} has too few samples for final MADiff-T training.")
        if bool(final_params.get("include_label", False)):
            label_idx = float(label_to_index[label])
            joint_train = np.column_stack([X_cls, np.full(X_cls.shape[0], label_idx)])
            final_generators[label].fit(joint_train, np.full(X_cls.shape[0], label_idx))
        else:
            final_generators[label].fit(X_cls, np.full(X_cls.shape[0], label))
    final_params["agent_id"] = "final"
    final_agent = _Agent("final", final_params, final_generators)
    _apply_per_feature_params(final_agent, generations)
    X_syn, y_syn = _synthesize(final_agent, len(full_df))
    if len(np.unique(y_syn)) < 2:
        raise ValueError("Final MADiff-T synthesis collapsed to a single class.")
    synth_df = _decode(X_syn, y_syn)

    if run_dir:
        log_path = Path(run_dir) / "logs" / f"madifft_{dataset.name}_seed{seed}.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(logs).to_csv(log_path, index=False)
        metrics_path = Path(run_dir) / "tables" / "madifft_generation_metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df = pd.DataFrame(gen_metrics_rows)
        if selection_meta:
            for key, value in selection_meta.items():
                metrics_df[key] = value
            metrics_df["selected"] = metrics_df["agent_id"] == selection_meta.get(
                "selected_agent_id", best_agent.agent_id
            )
        if metrics_path.exists():
            existing = pd.read_csv(metrics_path)
            metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
        metrics_df.to_csv(metrics_path, index=False)

    notes_msg = "MADiff-T evolutionary selection"
    if selection_meta.get("selected_by") in {"fallback_sum", "fitness", "fitness_escape"}:
        notes_msg = "MADiff-T selection: privacy gate failed; fallback used"
    metadata = {
        "best_agent_id": best_agent.agent_id,
        "best_params": dict(best_agent.params),
        "fitness_total": best_agent.fitness,
        "fitness_utility": best_agent.utility,
        "fitness_privacy": best_agent.privacy,
        "fitness_fidelity": best_agent.fidelity,
        "class_kl": best_agent.class_kl,
        "notes": notes_msg,
    }
    if selection_meta:
        metadata.update(selection_meta)
        if best_row is not None:
            metadata["selected_by"] = selection_meta.get("selected_by")
    if best_row is not None:
        metadata.update(
            {
                "generation": int(best_row.get("generation", generations - 1)),
                "utility": float(best_row.get("utility", best_agent.utility)),
                "fidelity": float(best_row.get("fidelity", best_agent.fidelity)),
                "normdcr": float(best_row.get("normdcr", float("nan"))),
                "mia_auc": float(best_row.get("mia_auc", float("nan"))),
                "mia_adv": float(best_row.get("mia_adv", float("nan"))),
                "fitness": float(best_row.get("fitness", best_agent.fitness)),
            }
        )
    return synth_df, metadata
