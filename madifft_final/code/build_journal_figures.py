"""Build the seven journal figures for the MADiff-T paper from the
authoritative `madifft_final/data/` snapshot.

Reads only from `madifft_final/data/` and `madifft_final/figures/*.csv`
(the figure-feeding CSVs produced by `build_all_figures_and_tables.py`).
Writes PDF + PNG into `madifft_final/figures/{pdf,png}/`.
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---------- paths ----------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
PDF_DIR = FIG / "pdf"
PNG_DIR = FIG / "png"
for d in (PDF_DIR, PNG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------- global style ---------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 12,
    "figure.titleweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

SINGLE_W = 3.46  # inches (88 mm)
DOUBLE_W = 7.08  # inches (180 mm)

# ---------- palette --------------------------------------------------------
DATASETS = ["breast_cancer", "pima", "heart", "credit", "adult_income"]
DATASET_LABELS = {
    "breast_cancer": "Breast Cancer",
    "pima": "PIMA",
    "heart": "Heart",
    "credit": "Credit",
    "adult_income": "Adult Income",
}
METHOD_ORDER = ["gaussian_copula", "tvae", "ctgan", "tabddpm_single",
                "tabsyn", "stasy", "madifft"]
METHOD_LABEL = {
    "gaussian_copula": "GC",
    "tvae": "TVAE",
    "ctgan": "CTGAN",
    "tabddpm_single": "TabDDPM",
    "tabsyn": "TabSyn",
    "stasy": "STaSy",
    "madifft": "MADiff-T",
}
METHOD_COLOR = {
    "gaussian_copula": "#1D9E75",
    "tvae": "#378ADD",
    "ctgan": "#D85A30",
    "tabddpm_single": "#73726c",
    "tabsyn": "#AFA9EC",
    "stasy": "#B4B2A9",
    "madifft": "#534AB7",
}
METHOD_MARKER = {
    "gaussian_copula": "o",
    "tvae": "s",
    "ctgan": "^",
    "tabddpm_single": "D",
    "tabsyn": "P",
    "stasy": "X",
    "madifft": "*",
}

REAL_UB_KEY = {
    "breast_cancer": "BreastCancer",
    "pima": "PIMA",
    "heart": "Heart",
    "credit": "GermanCredit",
    "adult_income": "AdultIncome",
}

# red-yellow-green colormap with lighter endpoints so annotated text
# remains readable on the darkest cells.
_CMAP_STOPS = [
    (0.00, "#c6363a"),  # light-ish red (not too dark)
    (0.25, "#f2a154"),  # orange
    (0.50, "#fff5a8"),  # pale yellow (center)
    (0.75, "#8ec87a"),  # medium green
    (1.00, "#2d8a4e"),  # capped dark green (still readable with white text)
]
CMAP_GOOD = LinearSegmentedColormap.from_list(
    "madifft_ryg", _CMAP_STOPS, N=256)
_CMAP_STOPS_R = [(1.0 - p, c) for p, c in _CMAP_STOPS][::-1]
CMAP_GOOD_R = LinearSegmentedColormap.from_list(
    "madifft_ryg_r", _CMAP_STOPS_R, N=256)


def _luminance(rgb) -> float:
    r, g, b = rgb[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_text(cmap, value: float, vmin: float, vmax: float) -> str:
    """Return black or white hex depending on background luminance."""
    if np.isnan(value):
        return "#1a1a1a"
    t = np.clip((value - vmin) / (vmax - vmin + 1e-12), 0, 1)
    rgba = cmap(t)
    return "#ffffff" if _luminance(rgba) < 0.55 else "#1a1a1a"


# ---------- loaders --------------------------------------------------------
def load_tstr() -> pd.DataFrame:
    """Return per-(dataset,method) mean/std accuracy %, 7 methods."""
    u = pd.read_csv(DATA / "tstr/tstr_all_methods_5ds_5seeds.csv")
    e = pd.read_csv(DATA / "external_baselines/tabsyn_stasy_tstr.csv")
    u = u[(u["metric_name"] == "accuracy")
          & (u["classifier"] == "logistic_regression")]
    e = e[(e["metric_name"] == "accuracy")
          & (e["classifier"] == "logistic_regression")]
    all_df = pd.concat([u[["dataset", "method", "seed", "metric_value"]],
                        e[["dataset", "method", "seed", "metric_value"]]],
                       ignore_index=True)
    g = all_df.groupby(["dataset", "method"])["metric_value"].agg(
        ["mean", "std"]).reset_index()
    g["mean_pct"] = g["mean"] * 100
    g["std_pct"] = g["std"] * 100
    return g


def load_privacy() -> pd.DataFrame:
    """Return per-(dataset,method) mean/std DCR and MIA AUC, 7 methods."""
    u = pd.read_csv(DATA / "privacy/privacy_all_methods_5ds_5seeds.csv")
    e = pd.read_csv(DATA / "external_baselines/tabsyn_stasy_privacy.csv")
    e = e.copy()
    e["method"] = e["method"].str.lower()
    all_df = pd.concat(
        [u[["dataset", "method", "seed", "dcr_q", "mia_auc"]],
         e[["dataset", "method", "seed", "dcr_q", "mia_auc"]]],
        ignore_index=True,
    )
    g = all_df.groupby(["dataset", "method"])[["dcr_q", "mia_auc"]].agg(
        ["mean", "std"]).reset_index()
    g.columns = ["dataset", "method",
                 "dcr_mean", "dcr_std", "mia_mean", "mia_std"]
    return g


def load_real_ub() -> dict:
    r = pd.read_csv(DATA / "real_upper_bound/real_tstr_upper_bound.csv")
    out = {}
    for ds in DATASETS:
        row = r[r["Dataset"] == REAL_UB_KEY[ds]].iloc[0]
        out[ds] = (float(row["Accuracy Mean"]) * 100,
                   float(row["Accuracy Std"]) * 100)
    return out


def _lookup(df: pd.DataFrame, ds: str, m: str, col: str) -> float:
    sub = df[(df["dataset"] == ds) & (df["method"] == m)]
    return float(sub[col].iloc[0]) if not sub.empty else float("nan")


def _matrix(df: pd.DataFrame, col: str) -> np.ndarray:
    M = np.full((len(DATASETS), len(METHOD_ORDER)), np.nan)
    for i, ds in enumerate(DATASETS):
        for j, m in enumerate(METHOD_ORDER):
            M[i, j] = _lookup(df, ds, m, col)
    return M


def _save(fig: plt.Figure, name: str) -> tuple[int, int]:
    pdf = PDF_DIR / f"{name}.pdf"
    png = PNG_DIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    # also place a copy at figures/ root for backward compatibility
    shutil.copyfile(pdf, FIG / f"{name}.pdf")
    shutil.copyfile(png, FIG / f"{name}.png")
    plt.close(fig)
    return pdf.stat().st_size, png.stat().st_size


# ---------- FIGURE 1: comparative utility ----------------------------------
def fig1_comparative_utility(tstr: pd.DataFrame, real_ub: dict):
    mean_M = _matrix(tstr, "mean_pct")
    std_M = _matrix(tstr, "std_pct")

    # append real-upper-bound row (last row, only MADiff-T column filled)
    ub_row_mean = np.full(len(METHOD_ORDER), np.nan)
    ub_row_std = np.full(len(METHOD_ORDER), np.nan)
    row_labels = [DATASET_LABELS[d] for d in DATASETS] + ["Real upper bound"]
    heat_mean = np.vstack([mean_M, ub_row_mean[None, :]])
    # Row of real upper bounds as a separate overlay; we'll render as a
    # shaded row below the heatmap.

    # deltas for panel B
    tab_idx = METHOD_ORDER.index("tabddpm_single")
    mad_idx = METHOD_ORDER.index("madifft")
    deltas = mean_M[:, mad_idx] - mean_M[:, tab_idx]

    fig = plt.figure(figsize=(DOUBLE_W, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.0])
    axH = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # --- heatmap (no cell annotations — numbers live in Table 4)
    vmin, vmax = 30, 100
    im = axH.imshow(mean_M, aspect="auto", cmap=CMAP_GOOD,
                    vmin=vmin, vmax=vmax)
    axH.set_xticks(range(len(METHOD_ORDER)))
    axH.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER],
                        rotation=30, ha="right", fontweight="bold",
                        fontsize=10)
    axH.set_yticks(range(len(DATASETS)))
    axH.set_yticklabels([DATASET_LABELS[d] for d in DATASETS],
                        fontweight="bold", fontsize=10)
    axH.set_title("TSTR accuracy (\\%)", fontsize=12, fontweight="bold")

    # thin cell grid
    axH.set_xticks(np.arange(-.5, len(METHOD_ORDER), 1), minor=True)
    axH.set_yticks(np.arange(-.5, len(DATASETS), 1), minor=True)
    axH.grid(which="minor", color="white", linewidth=1.0)
    axH.tick_params(which="minor", length=0)

    # MADiff-T column emphasis
    axH.add_patch(Rectangle(
        (mad_idx - 0.5, -0.5), 1, len(DATASETS),
        fill=False, edgecolor="black", linewidth=1.8, zorder=10))

    # --- real upper bound shaded strip below
    # draw a thin annotation under the x-axis showing real upper bound values
    real_vals = [real_ub[d][0] for d in DATASETS]
    real_stds = [real_ub[d][1] for d in DATASETS]
    # Place as text below the heatmap on the right edge
    ub_txt = "Real upper bound: " + ", ".join(
        f"{DATASET_LABELS[d][:8]} {m:.1f}"
        for d, m in zip(DATASETS, real_vals)
    )
    axH.set_xlabel(ub_txt, fontsize=8, fontweight="normal", labelpad=8)

    cbar = fig.colorbar(im, ax=axH, fraction=0.04, pad=0.02)
    cbar.set_label("Accuracy (\\%)", fontsize=10, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    # --- panel B: delta bars (labels outside)
    y = np.arange(len(DATASETS))
    colors = ["#2ca05a" if d >= 0 else "#d04a3a" for d in deltas]
    axB.barh(y, deltas, color=colors, edgecolor="black", linewidth=0.4)
    axB.axvline(0, color="black", linewidth=0.6)
    axB.set_yticks(y)
    axB.set_yticklabels([DATASET_LABELS[d] for d in DATASETS],
                        fontsize=9, fontweight="bold")
    axB.invert_yaxis()
    axB.set_title("MADiff-T $-$ TabDDPM", fontsize=11, fontweight="bold")
    axB.set_xlabel("$\\Delta$ accuracy (pp)", fontsize=9, fontweight="bold")
    for spine in ("top", "right"):
        axB.spines[spine].set_visible(False)
    xmax = max(abs(deltas.min()), abs(deltas.max())) * 1.9
    axB.set_xlim(-xmax, xmax)
    pad = xmax * 0.04
    for yi, d in enumerate(deltas):
        if d >= 0:
            axB.text(d + pad, yi, f"{d:+.1f}",
                     va="center", ha="left", fontsize=10, fontweight="bold")
        else:
            axB.text(d - pad, yi, f"{d:+.1f}",
                     va="center", ha="right", fontsize=10, fontweight="bold")
    axB.tick_params(axis="y", labelsize=9)
    axB.tick_params(axis="x", labelsize=9)

    return _save(fig, "comparative_utility_IS")


# ---------- FIGURE 2: privacy audits ---------------------------------------
def fig2_privacy(priv: pd.DataFrame):
    dcr_M = _matrix(priv, "dcr_mean")
    dcr_S = _matrix(priv, "dcr_std")
    mia_M = _matrix(priv, "mia_mean")
    mia_S = _matrix(priv, "mia_std")

    tab_idx = METHOD_ORDER.index("tabddpm_single")
    mad_idx = METHOD_ORDER.index("madifft")
    dcr_delta = dcr_M[:, mad_idx] - dcr_M[:, tab_idx]
    mia_delta = mia_M[:, mad_idx] - mia_M[:, tab_idx]

    fig = plt.figure(figsize=(DOUBLE_W + 0.9, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[2.75, 2.75, 1.0],
                          height_ratios=[1, 1], wspace=0.18)
    axD = fig.add_subplot(gs[:, 0])
    axM = fig.add_subplot(gs[:, 1])
    axB1 = fig.add_subplot(gs[0, 2])
    axB2 = fig.add_subplot(gs[1, 2])

    def _heat(ax, M, S, title, cmap, vmin, vmax, label):
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER],
                           rotation=30, ha="right", fontweight="bold",
                           fontsize=10)
        ax.set_yticks(range(len(DATASETS)))
        ax.set_yticklabels([DATASET_LABELS[d] for d in DATASETS],
                           fontweight="bold", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(-.5, len(METHOD_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(DATASETS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", length=0)
        ax.add_patch(Rectangle(
            (mad_idx - 0.5, -0.5), 1, len(DATASETS),
            fill=False, edgecolor="black", linewidth=1.8, zorder=10))
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
        cbar.set_label(label, fontsize=10, fontweight="bold")
        cbar.ax.tick_params(labelsize=9)

    # DCR: higher = safer → green. Log-like compression via sqrt so that the
    # spread across datasets (0-17) doesn't crush small values.
    dcr_cap = float(np.nanpercentile(dcr_M, 98))
    _heat(axD, dcr_M, dcr_S,
          "DCR (higher = safer)", CMAP_GOOD, 0, dcr_cap, "DCR")
    # MIA: lower = safer → reverse colormap. Typical range 0.45-0.60.
    _heat(axM, mia_M, mia_S,
          "MIA AUC (closer to 0.5 = safer)", CMAP_GOOD_R, 0.44, 0.60, "MIA AUC")

    # delta panels
    def _delta_bar(ax, deltas, title, xlab, negative_is_good=False):
        y = np.arange(len(DATASETS))
        if negative_is_good:
            colors = ["#2ca05a" if d <= 0 else "#d04a3a" for d in deltas]
        else:
            colors = ["#2ca05a" if d >= 0 else "#d04a3a" for d in deltas]
        ax.barh(y, deltas, color=colors, edgecolor="black", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([DATASET_LABELS[d][:8] for d in DATASETS],
                           fontsize=9, fontweight="bold")
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlab, fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        xmax = max(abs(deltas.min()), abs(deltas.max())) * 2.4
        ax.set_xlim(-xmax, xmax)
        pad = xmax * 0.04
        for yi, d in enumerate(deltas):
            if d >= 0:
                ax.text(d + pad, yi, f"{d:+.2f}",
                        ha="left", va="center", fontsize=9,
                        fontweight="bold")
            else:
                ax.text(d - pad, yi, f"{d:+.2f}",
                        ha="right", va="center", fontsize=9,
                        fontweight="bold")

    _delta_bar(axB1, dcr_delta, "MADiff-T $-$ TabDDPM (DCR)",
               "$\\Delta$ DCR", negative_is_good=False)
    _delta_bar(axB2, mia_delta, "MADiff-T $-$ TabDDPM (MIA)",
               "$\\Delta$ MIA AUC", negative_is_good=True)

    return _save(fig, "privacy_audits_IS")


# ---------- FIGURE 3: fidelity analysis ------------------------------------
def fig3_fidelity(gen: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(SINGLE_W * 2, 3.6),
                             constrained_layout=True)
    axes_list = axes.flatten()

    # Scale fidelity to 10^-3 units so the axis never shows "1e-5" offset
    SCALE = 1e3
    UNIT = r"fidelity proxy ($\times10^{-3}$)"

    for idx, ds in enumerate(DATASETS):
        ax = axes_list[idx]
        sub = gen[gen["dataset"] == ds]
        g = sub.groupby("generation")["fidelity"].agg(
            ["mean", "std"]).reset_index()
        sel = sub[sub["selected"]].groupby("generation")["fidelity"].mean()
        x = g["generation"].values
        mean = g["mean"].values * SCALE
        std = g["std"].fillna(0).values * SCALE
        ax.plot(x, mean, color="#534AB7", linewidth=1.6,
                marker="o", markersize=4, label="population mean")
        ax.fill_between(x, mean - std, mean + std,
                        color="#534AB7", alpha=0.18, linewidth=0)
        sel_x = sel.index.values
        sel_y = sel.values * SCALE
        ax.plot(sel_x, sel_y, linestyle="none", marker="*",
                markersize=9, color="#D85A30",
                markeredgecolor="black", markeredgewidth=0.4,
                label="selected")
        ax.set_title(DATASET_LABELS[ds], fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xlabel("generation", fontsize=10, fontweight="bold")
        if idx % 3 == 0:
            ax.set_ylabel(UNIT, fontsize=9, fontweight="bold")
        ax.tick_params(axis="both", labelsize=9)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(True, color="#E0E0E0", linewidth=0.3)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # legend panel
    axL = axes_list[5]
    axL.axis("off")
    handles = [
        plt.Line2D([], [], color="#534AB7", marker="o",
                   linewidth=1.8, markersize=5, label="population mean"),
        plt.Rectangle((0, 0), 1, 1, color="#534AB7", alpha=0.18,
                      label="$\\pm$1 std"),
        plt.Line2D([], [], color="#D85A30", linestyle="none", marker="*",
                   markersize=11, markeredgecolor="black",
                   markeredgewidth=0.4, label="selected agent"),
    ]
    axL.legend(handles=handles, loc="center", frameon=False, fontsize=9)

    fig.suptitle("Fidelity proxy across generations",
                 fontsize=12, fontweight="bold")
    return _save(fig, "fidelity_analysis_IS")


# ---------- FIGURE 4: MIA ROC curves (stub check) --------------------------
def fig4_mia_roc() -> tuple[int | None, int | None, str]:
    roc_csv = DATA / "privacy" / "mia_roc_raw.csv"
    if not roc_csv.exists():
        return None, None, "WARNING: mia_roc_raw.csv not found. SKIPPING."
    df = pd.read_csv(roc_csv)
    if len(df) <= 1 or df["fpr"].isna().all():
        return None, None, ("WARNING: mia_roc_raw.csv is a stub. SKIPPING.")

    # methods present in the ROC data (subset of METHOD_ORDER)
    roc_methods = [m for m in METHOD_ORDER if m in df["method"].unique()]

    fig, axes = plt.subplots(2, 3, figsize=(SINGLE_W * 2.05, SINGLE_W * 1.45),
                             constrained_layout=True)
    axes_flat = axes.flatten()

    fpr_grid = np.linspace(0, 1, 201)

    for idx, ds in enumerate(DATASETS):
        ax = axes_flat[idx]
        # diagonal baseline
        ax.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=0.7,
                linestyle="--", zorder=1)
        ds_df = df[df["dataset"] == ds]

        # plot non-MADiff-T first, then MADiff-T on top
        for m in roc_methods:
            if m == "madifft":
                continue
            _plot_mean_roc(ax, ds_df, m, fpr_grid, lw=1.0, alpha=0.85)
        # MADiff-T last (on top)
        if "madifft" in roc_methods:
            _plot_mean_roc(ax, ds_df, "madifft", fpr_grid, lw=1.8, alpha=1.0)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_title(DATASET_LABELS[ds], fontsize=10, fontweight="bold")

        # axis labels only on edge panels
        if idx >= 3:  # bottom row
            ax.set_xlabel("FPR", fontsize=9, fontweight="bold")
        if idx % 3 == 0:  # left column
            ax.set_ylabel("TPR", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    # 6th panel = legend
    axL = axes_flat[5]
    axL.axis("off")
    handles = []
    for m in roc_methods:
        lw = 1.8 if m == "madifft" else 1.0
        handles.append(plt.Line2D([], [], color=METHOD_COLOR[m],
                                  linewidth=lw, label=METHOD_LABEL[m]))
    handles.append(plt.Line2D([], [], color="#bbbbbb", linewidth=0.7,
                              linestyle="--", label="Random"))
    axL.legend(handles=handles, loc="center", frameon=False, fontsize=9,
               handlelength=2.0)

    return _save(fig, "mia_roc_curves_IS")


def _plot_mean_roc(ax, ds_df, method, fpr_grid, lw=1.0, alpha=1.0):
    """Plot seed-averaged ROC curve for one method on one dataset."""
    m_df = ds_df[ds_df["method"] == method]
    if m_df.empty:
        return
    seeds = sorted(m_df["seed"].unique())
    tpr_interps = []
    for s in seeds:
        seed_df = m_df[m_df["seed"] == s].sort_values("fpr")
        fpr_s = seed_df["fpr"].values
        tpr_s = seed_df["tpr"].values
        tpr_interp = np.interp(fpr_grid, fpr_s, tpr_s)
        tpr_interps.append(tpr_interp)
    mean_tpr = np.mean(tpr_interps, axis=0)
    mean_tpr[0] = 0.0  # ensure starts at origin
    ax.plot(fpr_grid, mean_tpr, color=METHOD_COLOR[method],
            linewidth=lw, alpha=alpha, zorder=5 if method == "madifft" else 3)


# ---------- FIGURE 5: tradeoff scatter -------------------------------------
def fig5_tradeoff(tstr: pd.DataFrame, priv: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_W, 4.4),
                             constrained_layout=True)
    axes_list = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes_list[idx]
        for m in METHOD_ORDER:
            acc = _lookup(tstr, ds, m, "mean_pct")
            mia = _lookup(priv, ds, m, "mia_mean")
            if np.isnan(acc) or np.isnan(mia):
                continue
            size = 170 if m == "madifft" else 55
            edge = "black" if m == "madifft" else "none"
            edge_w = 1.4 if m == "madifft" else 0
            ax.scatter(mia, acc, s=size,
                       marker=METHOD_MARKER[m],
                       color=METHOD_COLOR[m],
                       edgecolor=edge, linewidth=edge_w,
                       label=METHOD_LABEL[m],
                       zorder=5 if m == "madifft" else 3)
        ax.set_title(DATASET_LABELS[ds], fontsize=10, fontweight="bold")
        ax.set_xlabel("MIA AUC", fontsize=10, fontweight="bold")
        if idx % 3 == 0:
            ax.set_ylabel("TSTR accuracy (\\%)",
                          fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, color="#E0E0E0", linewidth=0.3)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        # axis-edge "better" hints (italic gray)
        ax.annotate("$\\leftarrow$ safer",
                    xy=(0.02, -0.22), xycoords="axes fraction",
                    fontsize=7.5, color="#666666",
                    style="italic", ha="left", va="top")
        ax.annotate("higher utility $\\uparrow$",
                    xy=(-0.22, 0.98), xycoords="axes fraction",
                    fontsize=7.5, color="#666666",
                    style="italic", ha="left", va="top",
                    rotation=90)

    # legend panel
    axL = axes_list[5]
    axL.axis("off")
    handles = []
    for m in METHOD_ORDER:
        handles.append(
            plt.Line2D([], [], linestyle="none",
                       marker=METHOD_MARKER[m],
                       markersize=9 if m == "madifft" else 7,
                       color=METHOD_COLOR[m],
                       markeredgecolor="black" if m == "madifft" else "none",
                       markeredgewidth=0.8 if m == "madifft" else 0,
                       label=METHOD_LABEL[m])
        )
    axL.legend(handles=handles, loc="center", frameon=False,
               fontsize=9, ncol=1, handlelength=1.4)

    fig.suptitle("Utility–privacy trade-off",
                 fontsize=12, fontweight="bold")
    return _save(fig, "tradeoff_scatter_IS")


# ---------- FIGURE 6: ablation bar chart -----------------------------------
def fig6_ablation():
    frames = {
        "Original": pd.read_csv(DATA / "ablation/abl_orig_tstr.csv"),
        "+FA Loss": pd.read_csv(DATA / "ablation/abl_faloss_tstr.csv"),
        "+FA+PFS (Final)": pd.read_csv(DATA / "ablation/abl_final_tstr.csv"),
    }
    means = {}
    stds = {}
    for label, df in frames.items():
        sub = df[(df["metric_name"] == "accuracy")
                 & (df["classifier"] == "logistic_regression")
                 & (df["method"] == "madifft")]
        g = sub.groupby("dataset")["metric_value"].agg(["mean", "std"])
        means[label] = [g.loc[ds, "mean"] * 100 for ds in DATASETS]
        stds[label] = [g.loc[ds, "std"] * 100 for ds in DATASETS]

    labels = list(frames.keys())
    colors = ["#CFCFCF", "#8A9DD0", "#534AB7"]

    fig, ax = plt.subplots(figsize=(SINGLE_W * 1.9, 3.2),
                           constrained_layout=True)
    x = np.arange(len(DATASETS))
    w = 0.26
    for i, label in enumerate(labels):
        xs = x + (i - 1) * w
        ax.bar(xs, means[label], w, yerr=stds[label],
               color=colors[i], edgecolor="black", linewidth=0.4,
               label=label,
               error_kw=dict(ecolor="#333333", lw=0.5, capsize=1.8,
                             capthick=0.5))
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS],
                       rotation=25, ha="right", fontsize=9,
                       fontweight="bold")
    ax.set_ylabel("TSTR accuracy (%)", fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylim(30, 90)
    ax.grid(True, axis="y", color="#E0E0E0", linewidth=0.3)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, 1.14), ncol=3, handlelength=1.6,
              columnspacing=1.6)
    ax.set_title("Ablation: stepwise enhancement contribution",
                 fontsize=12, fontweight="bold", pad=28)

    return _save(fig, "ablation_study_IS")


# ---------- FIGURE 7: evolution curves -------------------------------------
def fig7_evolution(gen: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(SINGLE_W * 2, 3.6),
                             constrained_layout=True)
    axes_list = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes_list[idx]
        sub = gen[gen["dataset"] == ds]
        g = sub.groupby("generation")["fitness"].agg(
            ["mean", "std", "max"]).reset_index()
        sel = sub[sub["selected"]].groupby("generation")["fitness"].mean()
        x = g["generation"].values
        mean = g["mean"].values
        std = g["std"].fillna(0).values
        mx = g["max"].values
        ax.plot(x, mean, color="#534AB7", linewidth=1.6,
                marker="o", markersize=4, label="mean")
        ax.fill_between(x, mean - std, mean + std,
                        color="#534AB7", alpha=0.18, linewidth=0)
        ax.plot(x, mx, color="#D85A30", linewidth=1.2,
                linestyle="--", marker="s", markersize=3.5, label="max")
        ax.plot(sel.index.values, sel.values, linestyle="none",
                marker="*", markersize=9, color="#2ca05a",
                markeredgecolor="black", markeredgewidth=0.4,
                label="selected")
        ax.set_title(DATASET_LABELS[ds], fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xlabel("generation", fontsize=10, fontweight="bold")
        if idx % 3 == 0:
            ax.set_ylabel("fitness", fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, color="#E0E0E0", linewidth=0.3)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axL = axes_list[5]
    axL.axis("off")
    handles = [
        plt.Line2D([], [], color="#534AB7", marker="o",
                   linewidth=1.8, markersize=6, label="population mean"),
        plt.Rectangle((0, 0), 1, 1, color="#534AB7", alpha=0.18,
                      label="$\\pm$1 std"),
        plt.Line2D([], [], color="#D85A30", marker="s", linestyle="--",
                   linewidth=1.4, markersize=6, label="population max"),
        plt.Line2D([], [], color="#2ca05a", linestyle="none", marker="*",
                   markersize=12, markeredgecolor="black",
                   markeredgewidth=0.4, label="selected agent"),
    ]
    axL.legend(handles=handles, loc="center", frameon=False, fontsize=9)

    fig.suptitle("Evolutionary selection fitness",
                 fontsize=12, fontweight="bold")
    return _save(fig, "evolution_curves_IS")


# ---------- driver ---------------------------------------------------------
def main():
    print("=" * 72)
    print("Building journal figures from madifft_final/data/")
    print("=" * 72)

    tstr = load_tstr()
    priv = load_privacy()
    real_ub = load_real_ub()
    gen = pd.read_csv(DATA / "evolution/madifft_generation_metrics.csv")

    generated = []
    warnings_list = []

    print("\n[1/7] comparative_utility_IS ...")
    generated.append(("comparative_utility_IS",
                      *fig1_comparative_utility(tstr, real_ub)))

    print("[2/7] privacy_audits_IS ...")
    generated.append(("privacy_audits_IS", *fig2_privacy(priv)))

    print("[3/7] fidelity_analysis_IS ...")
    generated.append(("fidelity_analysis_IS", *fig3_fidelity(gen)))

    print("[4/7] mia_roc_curves_IS ...")
    result = fig4_mia_roc()
    if len(result) == 3 and result[0] is None:
        # (None, None, warning_msg)
        warnings_list.append(result[2])
        print("   " + result[2])
    elif len(result) == 2:
        generated.append(("mia_roc_curves_IS", result[0], result[1]))
    else:
        generated.append(("mia_roc_curves_IS", result[0], result[1]))

    print("[5/7] tradeoff_scatter_IS ...")
    generated.append(("tradeoff_scatter_IS", *fig5_tradeoff(tstr, priv)))

    print("[6/7] ablation_study_IS ...")
    generated.append(("ablation_study_IS", *fig6_ablation()))

    print("[7/7] evolution_curves_IS ...")
    generated.append(("evolution_curves_IS", *fig7_evolution(gen)))

    # summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'figure':<28} {'PDF size':>12} {'PNG size':>12}")
    print("-" * 54)
    for name, pdf_sz, png_sz in generated:
        print(f"{name:<28} {pdf_sz/1024:>9.1f} KB {png_sz/1024:>9.1f} KB")

    if warnings_list:
        print("\nWarnings / skipped:")
        for w in warnings_list:
            print("  - " + w)
    else:
        print("\nNo warnings.")

    print("\nFiles in figures/:")
    for f in sorted(FIG.glob("*.pdf")):
        print(f"  {f.name}")
    for f in sorted(FIG.glob("*.png")):
        print(f"  {f.name}")
    print("\nFiles in figures/pdf/:")
    for f in sorted(PDF_DIR.glob("*.pdf")):
        print(f"  {f.name}")
    print("\nFiles in figures/png/:")
    for f in sorted(PNG_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
