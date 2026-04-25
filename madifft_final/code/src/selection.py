from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SelectionResult:
    selected_row: pd.Series
    selected_by: str
    gate_count: int
    relaxed_count: int
    total_candidates: int
    thresholds: dict[str, float]


def _is_valid_series(series: pd.Series) -> pd.Series:
    return ~series.isna()


def compute_utility_floor(
    y_train,
    base_floor: float = 0.60,
    margin: float = 0.04,
    max_floor: float = 0.68,
) -> float:
    """
    Compute a dataset-aware utility floor for the privacy gate.
    """
    y = np.array(y_train)
    if len(y) == 0:
        return float(base_floor)
    _, counts = np.unique(y, return_counts=True)
    majority_rate = counts.max() / counts.sum()
    adaptive = majority_rate + margin
    effective = max(base_floor, min(max_floor, adaptive))
    return float(effective)


def select_with_privacy_gate(
    df: pd.DataFrame,
    *,
    mia_thr: float = 0.55,
    normdcr_thr: float = 1.0,
    relaxed_mia_thr: float = 0.58,
    relaxed_normdcr_thr: float = 0.95,
    utility_floor: float = 0.60,
    utility_col: str = "utility",
    fitness_col: str = "fitness",
) -> tuple[pd.DataFrame, SelectionResult]:
    if df.empty:
        raise ValueError("Selection input is empty.")

    work = df.copy()
    mia = work["mia_auc"] if "mia_auc" in work.columns else pd.Series(np.nan, index=work.index)
    normdcr = work["normdcr"] if "normdcr" in work.columns else pd.Series(np.nan, index=work.index)
    utility = (
        work[utility_col]
        if utility_col in work.columns
        else pd.Series(np.nan, index=work.index)
    )
    utility_scores = pd.to_numeric(utility, errors="coerce")
    n_candidates = int(len(utility_scores))
    n_passing = int((utility_scores >= utility_floor).fillna(False).sum())
    pass_rate = (n_passing / n_candidates) if n_candidates > 0 else 0.0
    print(
        f"[selection] utility_floor={utility_floor:.4f} "
        f"pass_rate={pass_rate:.2f} ({n_passing}/{n_candidates})"
    )

    if n_passing < 2:
        print(
            f"[selection] Gate escape: only {n_passing}/{n_candidates} agents "
            f"passed floor={utility_floor:.4f}. Using fitness ranking."
        )
        selected_by = "fitness_escape"
        if fitness_col in work.columns and work[fitness_col].notna().any():
            best_idx = work[fitness_col].idxmax()
        else:
            best_idx = utility_scores.idxmax()
        work["gate_pass"] = False
        work["relaxed_pass"] = False
        work["selected"] = False
        work.loc[best_idx, "selected"] = True
        work["selected_by"] = selected_by
        thresholds = {
            "mia_thr": mia_thr,
            "normdcr_thr": normdcr_thr,
            "relaxed_mia_thr": relaxed_mia_thr,
            "relaxed_normdcr_thr": relaxed_normdcr_thr,
            "utility_floor": utility_floor,
            "pass_rate": pass_rate,
            "n_passing": float(n_passing),
            "n_candidates": float(n_candidates),
        }
        return (
            work,
            SelectionResult(
                selected_row=work.loc[best_idx],
                selected_by=selected_by,
                gate_count=0,
                relaxed_count=0,
                total_candidates=n_candidates,
                thresholds=thresholds,
            ),
        )

    valid_mask = _is_valid_series(mia) & _is_valid_series(normdcr) & _is_valid_series(utility)
    utility_ok = utility >= utility_floor

    gate_pass = valid_mask & utility_ok & (mia <= mia_thr) & (normdcr >= normdcr_thr)
    relaxed_pass = (
        valid_mask & utility_ok & (mia <= relaxed_mia_thr) & (normdcr >= relaxed_normdcr_thr)
    )

    work["gate_pass"] = gate_pass
    work["relaxed_pass"] = relaxed_pass

    gate_count = int(gate_pass.sum())
    relaxed_count = int(relaxed_pass.sum())
    total_candidates = int(len(work))

    if gate_count > 0:
        selected_by = "privacy_gate"
        candidates = work[gate_pass]
        selected_idx = candidates[utility_col].idxmax()
    elif relaxed_count > 0:
        selected_by = "relaxed_gate"
        candidates = work[relaxed_pass]
        selected_idx = candidates[utility_col].idxmax()
    else:
        selected_by = "fitness"
        if fitness_col in work.columns and work[fitness_col].notna().any():
            selected_idx = work[fitness_col].idxmax()
        else:
            selected_idx = work[utility_col].idxmax()

    work["selected"] = False
    work.loc[selected_idx, "selected"] = True
    work["selected_by"] = selected_by
    thresholds = {
        "mia_thr": mia_thr,
        "normdcr_thr": normdcr_thr,
        "relaxed_mia_thr": relaxed_mia_thr,
        "relaxed_normdcr_thr": relaxed_normdcr_thr,
        "utility_floor": utility_floor,
        "pass_rate": pass_rate,
        "n_passing": float(n_passing),
        "n_candidates": float(n_candidates),
    }

    return (
        work,
        SelectionResult(
            selected_row=work.loc[selected_idx],
            selected_by=selected_by,
            gate_count=gate_count,
            relaxed_count=relaxed_count,
            total_candidates=total_candidates,
            thresholds=thresholds,
        ),
    )
